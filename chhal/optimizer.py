"""The constrained evasion optimizer — the novel core, with the plausibility guardrail.

Given a seed AttackBatch and the CURRENT detector, nudge what an attacker actually
controls to lower the detector's fraud score — but only within a realistic, executable
envelope. Without that guardrail you get "attacks" that fool the model yet that no real
fraudster could execute, which would destroy the real-world-feasibility score.

Why the search moves a TIMELINE and not a feature vector
--------------------------------------------------------
Nine features are attacker-influenced, but only four of them are *settable*: the amount,
the payee flag, the rail, and the destination. The other five —

    hour, velocity_1h, velocity_24h, time_since_last_txn_min, amount_to_avg_ratio

— are not choices at all. They are what a timeline LOOKS LIKE once it has happened. You
cannot decide to have transacted four times in the past hour; you can only transact four
times, and then that is what the counter reads.

An earlier version of this file perturbed all nine as independent scalars. That silently
undid the entire campaign architecture: 59-93% of the rows it emitted were physically
impossible, including 13-42% claiming more transactions in the last hour than in the last
twenty-four — an arithmetic impossibility, since the 1h window is contained in the 24h
one. The seed batches were clean; the optimizer was the thing breaking them, and because
its output IS the benchmark, the fidelity population and the training additions, nothing
downstream was ever measured on a coherent row.

So the search now moves the two things a fraudster genuinely decides — WHEN to transact
and FOR HOW MUCH — plus the three flags, and re-derives the rest through
`chhal.behaviour.derive`, the same function applied to the 590,540 real transactions.
Consistency is not restored afterwards; it cannot be violated, for the same reason it
cannot be violated in the red team's renderer.

This costs evasion power: the search space is smaller and every move has downstream
consequences the attacker must live with. That is the correct trade. An attack that
cannot be executed is not an attack.

Method: gradient-free (evolutionary hill-climb) over the timeline parameters. No
gradients through LightGBM needed, which keeps it simple and model-agnostic.

Constraints on every candidate:
  (a) business rules   — takeover happens after the card is compromised, gaps are
                         positive, amounts non-negative, valid channel codes;
  (b) realistic manifold — each DIRECTLY SET feature stays within [q0.5%, q99.5%] of the
                         base population, and inter-arrival gaps stay within the observed
                         envelope;
  (c) attacker control — only settable features move; the issuer's view is untouchable
                         and the derived block is recomputed, never assigned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .behaviour import assemble_frame, consistency_violations
from .contract import (ATTACKER_DIRECT, DERIVED_FEATURES, ENTITY_LEVEL_FEATURES,
                       FEATURE_COLUMNS, INHERITED_FEATURES, INTEGER_FEATURES,
                       AttackBatch)
from .detector import Detector
from .redteam.campaign import MAX_TAKEOVER_WAIT_DAYS, TAKEOVER_GAP_S

MIN_GAP_S = 1.0          # two transactions cannot share an instant


@dataclass
class OptimizerConfig:
    generations: int = 12
    population: int = 24
    step_scale: float = 0.15      # perturbation size as fraction of feature range
    gap_step: float = 0.45        # timing perturbation, in natural logs of the gap
    gap_widen: float = 4.0        # how far a campaign may be re-paced, either way
    amount_widen: float = 3.0     # how far amounts may leave the vector's own band
    velocity_1h_cap: int = 25     # business rule: hard velocity ceiling
    velocity_24h_cap: int = 150


class EvasionOptimizer:
    """Hill-climb over an attack's timeline and flags, re-deriving behaviour each step."""

    def __init__(self, feature_stats: pd.DataFrame, cfg: OptimizerConfig | None = None):
        self.cfg = cfg or OptimizerConfig()
        # realistic manifold bounds from the base population
        self.lo = feature_stats.loc[0.005].copy()
        self.hi = feature_stats.loc[0.995].copy()

        # Business rules and the statistical manifold are intersected ONCE, here, rather
        # than applied in sequence. Applying them in sequence lets the second one push a
        # value back out of the first: clipping amount to [q0.5, q99.5] and then to
        # >= 0.5 walks any row whose manifold ceiling is below 0.5 straight off the
        # manifold, and the same ordering bug applies to both velocity caps.
        self._bounds = {}
        business = {
            "amount": (0.5, np.inf),
            "channel_code": (0, 2),
            "is_new_beneficiary": (0, 1),
            "is_cross_border": (0, 1),
        }
        for col in ATTACKER_DIRECT:
            b_lo, b_hi = business.get(col, (-np.inf, np.inf))
            lo_c, hi_c = max(self.lo[col], b_lo), min(self.hi[col], b_hi)
            if col in INTEGER_FEATURES:
                # Integer features are rounded after clipping, and rounding can walk a
                # value straight back out: a ceiling of 3.63 becomes 4. Pull the bounds
                # INWARD to whole numbers first, so anything inside them stays inside
                # after rounding. Skipped where no whole number fits.
                lo_i, hi_i = np.ceil(lo_c), np.floor(hi_c)
                if lo_i <= hi_i:
                    lo_c, hi_c = lo_i, hi_i
            if lo_c > hi_c:                      # degenerate: keep it non-empty
                lo_c = hi_c = float(np.clip(0.0, lo_c, hi_c)) if np.isfinite(lo_c) else hi_c
            self._bounds[col] = (float(lo_c), float(hi_c))

        # Global ceiling on any single inter-arrival gap. Used only as a backstop; the
        # binding constraint is per-campaign and set in `_gap_envelope`.
        self._gap_ceiling = max(float(self.hi["time_since_last_txn_min"]) * 60.0, 60.0)
        # WHEN to first use a card you have just compromised is a different decision from
        # how fast to move once you are using it, so the two get different envelopes.
        # This one matches `_takeover_time`'s own range exactly.
        self._takeover_bounds = (float(TAKEOVER_GAP_S),
                                 float(TAKEOVER_GAP_S + MAX_TAKEOVER_WAIT_DAYS * 86_400))

    # -- constraints ---------------------------------------------------------
    def _bind_direct_envelope(self, seed: pd.DataFrame) -> None:
        """Narrow the settable features to what THIS vector declared, and freeze identity.

        Two rules, both read off the seed batch so the optimizer never needs to know
        which vector it is holding.

        1. A column the seed holds CONSTANT is not a preference, it is the attack's
           definition. "Card testing" means probing a card rail; an optimizer that moves
           40% of those probes onto UPI has not evaded anything, it has quietly changed
           the experiment. Constant columns are frozen.
        2. A column the seed spreads over a band may move, but only `amount_widen` times
           outside that band. Otherwise the search walks every vector down to the same
           q0.5% amount floor — 46% of rows landed on exactly one value that way — which
           destroys both the vector's meaning and any diversity claim made about it.

        Both envelopes are intersected with the global plausibility manifold, which still
        binds; these only ever narrow it.
        """
        self._frozen = set()
        self._env = {}
        for col in ATTACKER_DIRECT:
            v = seed[col].to_numpy(np.float64)
            lo_m, hi_m = self._bounds[col]
            if len(v) == 0 or np.allclose(v, v[0]):
                self._frozen.add(col)
                self._env[col] = (float(v[0]), float(v[0])) if len(v) else (lo_m, hi_m)
                continue
            p05, p95 = np.percentile(v, 5), np.percentile(v, 95)
            w = self.cfg.amount_widen
            lo_c = max(lo_m, p05 / w if p05 > 0 else p05 - abs(p05) * w - 1.0)
            hi_c = min(hi_m, p95 * w if p95 > 0 else p95 + abs(p95) * w + 1.0)
            if col in INTEGER_FEATURES:
                lo_c, hi_c = np.ceil(lo_c), np.floor(hi_c)
            self._env[col] = (float(lo_c), float(max(hi_c, lo_c)))

    def _clip_direct(self, values: dict) -> dict:
        """Pull the four directly-set features back inside the plausibility envelope."""
        out = {}
        for col, v in values.items():
            lo_c, hi_c = self._env[col]
            v = np.clip(v, lo_c, hi_c)
            if col in INTEGER_FEATURES:
                v = np.round(v)
            out[col] = v
        return out

    def _off_manifold_frac(self, values: dict) -> float:
        """Fraction of directly-set cells outside the manifold BEFORE clipping.

        Non-tautological companion to fidelity.on_manifold_rate: that metric is measured
        on this optimizer's OWN clipped output against the SAME bounds, so it is ~1.0 by
        construction and only proves the clip is wired correctly. This measures how often
        the search actually WANTED to leave the manifold — whether the guardrail is doing
        real work, not merely present.

        Measured over the four settable features only. The derived block is no longer
        clipped at all: it is recomputed from a timeline, so its values are whatever
        really happens when transactions occur at those times, which is a stronger
        feasibility claim than "inside a quantile box". `derived_on_manifold_rate` in the
        returned provenance reports where those land, without constraining them.
        """
        total = sum(np.size(v) for v in values.values())
        if total == 0:
            return 0.0
        viol = 0
        for col, v in values.items():
            lo_c, hi_c = self._bounds[col]      # the MANIFOLD, not the narrowed envelope
            viol += int(np.sum((v < lo_c) | (v > hi_c)))
        return viol / total

    # -- timeline surgery ----------------------------------------------------
    @staticmethod
    def _unpack(timeline: pd.DataFrame):
        """Split a campaign into the parts the search moves and the parts it cannot."""
        ent = timeline["entity"].to_numpy()
        ts = timeline["timestamp_s"].to_numpy(np.float64)
        atk = timeline["is_attack"].to_numpy(bool)
        idx = np.flatnonzero(atk)
        if len(idx) == 0:
            raise ValueError("timeline contains no attack rows")
        # Entities are contiguous blocks and within a block the host's real history comes
        # first, so an attack row's predecessor is always the row immediately before it.
        if idx[0] == 0:
            raise ValueError("first timeline row is an attack — host history is missing")
        gaps = ts[idx] - ts[idx - 1]
        ent_a = ent[idx]
        first_of_entity = np.r_[True, ent_a[1:] != ent_a[:-1]]
        # anchor each entity's attack block to the host's LAST real transaction
        anchor = (pd.Series(np.where(atk, np.nan, ts)).groupby(ent)
                  .transform("max").to_numpy()[idx])
        return ent, ts, atk, idx, ent_a, gaps, first_of_entity, anchor

    def _gap_envelope(self, seed_gaps: np.ndarray, first_of_entity: np.ndarray,
                      ent_code: np.ndarray):
        """How far the attacker may re-time hops WITHIN a campaign, from the seed itself.

        Not from the global manifold. `time_since_last_txn_min`'s q99.5 is about three
        weeks, because it is dominated by dormant cards and by the 30-day placeholder a
        first-ever transaction gets — so using it as a per-hop ceiling let a bust-out
        spread itself over a year and a card-testing run put its probes 39 days apart.
        Those are not slower versions of the attack, they are different attacks, and a
        vector that can be re-timed into another vector is not a vector.

        Bounded PER CAMPAIGN, not per batch, and the difference is not cosmetic. Once
        the hero vector started reading its cadence off each victim, one batch held
        campaigns pacing minutes apart and campaigns pacing weeks apart; a single
        batch-wide 95th percentile handed the fast ones the slow ones' ceiling, and the
        optimizer walked a victim who transacts daily out to a median gap of 28 DAYS.
        The point of per-victim mimicry is that each campaign moves at ITS victim's
        pace, so that is the envelope it gets to move inside.

        The seed's own gaps carry the vector's declared shape without the optimizer
        needing to know which vector it is holding. The attacker may go `gap_widen`
        times faster or slower than that campaign's own pace, and no further.
        """
        n = len(seed_gaps)
        lo = np.full(n, MIN_GAP_S, dtype=np.float64)
        hi = np.full(n, self._gap_ceiling, dtype=np.float64)
        inner = ~first_of_entity
        if not inner.any():
            return lo, hi
        g = pd.DataFrame({"e": ent_code[inner], "g": seed_gaps[inner]}).groupby("e")["g"]
        per_lo = (g.quantile(0.05) / self.cfg.gap_widen).clip(lower=MIN_GAP_S)
        per_hi = (g.quantile(0.95) * self.cfg.gap_widen).clip(upper=self._gap_ceiling)
        per_hi = np.maximum(per_hi, per_lo)
        # campaigns with no inner gap keep the global fallback already in lo/hi
        lo[inner] = per_lo.reindex(ent_code[inner]).to_numpy()
        hi[inner] = per_hi.reindex(ent_code[inner]).to_numpy()
        return lo, hi

    def _clip_gaps(self, gaps: np.ndarray, first_of_entity: np.ndarray) -> np.ndarray:
        out = np.empty_like(gaps)
        t_lo, t_hi = self._takeover_bounds
        g_lo, g_hi = self._gap_bounds
        out[first_of_entity] = np.clip(gaps[first_of_entity], t_lo, t_hi)
        out[~first_of_entity] = np.clip(gaps[~first_of_entity],
                                        g_lo[~first_of_entity], g_hi[~first_of_entity])
        return out

    @staticmethod
    def _rebuild_timestamps(gaps, ent_a, anchor) -> np.ndarray:
        """Lay the attack transactions back out in time from their inter-arrival gaps."""
        cum = pd.Series(gaps).groupby(ent_a).cumsum().to_numpy()
        return anchor + cum

    # -- main ----------------------------------------------------------------
    def optimize(
        self, batch: AttackBatch, detector: Detector, rng: np.random.Generator
    ) -> AttackBatch:
        """Return an adapted AttackBatch that evades `detector` while staying executable."""
        if batch.timeline is None:
            raise ValueError(
                f"AttackBatch[{batch.vector_id}] has no timeline. The optimizer searches "
                f"over WHEN and FOR HOW MUCH an attacker transacts and re-derives the "
                f"behavioural block from that, so it needs the campaign the rows came "
                f"from. Build the batch with AttackVector.batch(), which attaches it."
            )
        tl = batch.timeline
        ent, ts0, atk, idx, ent_a, gaps0, first_of_entity, anchor = self._unpack(tl)
        amount0 = tl["amount"].to_numpy(np.float64)
        self._amount_all = amount0.copy()

        # entity index per attack row, compacted, so entity-level flags broadcast cheaply
        _, ent_code = np.unique(ent_a, return_inverse=True)

        seed_rows = batch.transactions
        # bind the per-campaign envelopes before any clipping happens
        self._gap_bounds = self._gap_envelope(gaps0, first_of_entity, ent_code)
        self._bind_direct_envelope(seed_rows := batch.transactions)

        params = {
            "gaps": self._clip_gaps(gaps0, first_of_entity),
            "amount": np.clip(amount0[idx], *self._env["amount"]),
            "is_new_beneficiary": seed_rows["is_new_beneficiary"].to_numpy(np.float64),
            # one choice per compromised account, not per transaction
            "channel_code": (pd.Series(seed_rows["channel_code"].to_numpy())
                             .groupby(ent_code).transform("first")
                             .to_numpy()[np.r_[True, ent_code[1:] != ent_code[:-1]]]
                             .astype(np.float64)),
            "is_cross_border": (pd.Series(seed_rows["is_cross_border"].to_numpy())
                                .groupby(ent_code).transform("first")
                                .to_numpy()[np.r_[True, ent_code[1:] != ent_code[:-1]]]
                                .astype(np.float64)),
        }

        self._clip_hits: List[float] = []
        inherited = tl[INHERITED_FEATURES].to_numpy(np.float64)
        best_rows = self._materialise(params, ent, ts0, atk, idx, ent_a, anchor,
                                      ent_code, inherited)
        best_score = detector.score(best_rows)

        for _ in range(self.cfg.generations):
            for _ in range(self.cfg.population):
                cand = self._perturb(params, ent_code, first_of_entity, rng)
                rows = self._materialise(cand, ent, ts0, atk, idx, ent_a, anchor,
                                         ent_code, inherited)
                score = detector.score(rows)
                improved = score < best_score          # lower = more evasive
                if improved.any():
                    # Accept in PARAMETER space, then re-derive. Accepting finished
                    # feature rows piecemeal would splice two different timelines
                    # together and reintroduce exactly the incoherence this file exists
                    # to prevent.
                    for k in ("gaps", "amount", "is_new_beneficiary"):
                        params[k] = np.where(improved, cand[k], params[k])
                    ent_improved = (pd.Series(improved).groupby(ent_code).transform("any")
                                    .to_numpy()[np.r_[True, ent_code[1:] != ent_code[:-1]]])
                    for k in ENTITY_LEVEL_FEATURES:
                        params[k] = np.where(ent_improved, cand[k], params[k])
            # one honest re-score of the merged parameter set: moving row i's gap shifts
            # every later transaction in its campaign, so per-row scores taken against
            # different candidates are not directly comparable.
            best_rows = self._materialise(params, ent, ts0, atk, idx, ent_a, anchor,
                                          ent_code, inherited)
            best_score = detector.score(best_rows)

        viol = consistency_violations(best_rows)
        derived_ok = self._derived_on_manifold(best_rows)

        adapted = AttackBatch(
            vector_id=batch.vector_id,
            iteration=batch.iteration,
            transactions=best_rows[FEATURE_COLUMNS].reset_index(drop=True),
            provenance={
                **batch.provenance,
                "optimized": True,
                "mean_evasion_score": float(best_score.mean()),
                "generations": self.cfg.generations,
                # non-tautological guardrail evidence (see fidelity.py docstring):
                # how often a proposed move landed outside the manifold pre-clip
                "frac_off_manifold_pre_clip": (
                    float(np.mean(self._clip_hits)) if self._clip_hits else 0.0
                ),
                # the derived block is never clipped — this reports where it landed
                "derived_on_manifold_rate": derived_ok,
                "consistency_violations": {k: float(v) for k, v in viol.items()},
            },
            timeline=self._new_timeline(tl, idx, params, ent_a, anchor),
        ).validate()
        return adapted

    # -- internals -----------------------------------------------------------
    def _perturb(self, params: dict, ent_code, first_of_entity,
                 rng: np.random.Generator) -> dict:
        cand = dict(params)
        # Timing moves multiplicatively: the gap between two card-testing probes and the
        # gap between two bust-out purchases differ by four orders of magnitude, so an
        # additive step that is meaningful for one is meaningless for the other.
        cand["gaps"] = self._clip_gaps(
            params["gaps"] * np.exp(rng.normal(0, self.cfg.gap_step, len(params["gaps"]))),
            first_of_entity,
        )
        raw = {}
        for col in ATTACKER_DIRECT:
            if col in self._frozen:             # the attack's definition, not a knob
                raw[col] = params[col]
                continue
            n = len(params[col])
            lo_c, hi_c = self._env[col]
            span = max(hi_c - lo_c, 1e-6)
            if not np.isfinite(span):
                span = max(abs(float(np.mean(params[col]))), 1.0)
            raw[col] = params[col] + rng.normal(0, self.cfg.step_scale * span, n)
        self._clip_hits.append(self._off_manifold_frac(raw))
        cand.update(self._clip_direct(raw))
        return cand

    def _materialise(self, params, ent, ts0, atk, idx, ent_a, anchor,
                     ent_code, inherited) -> pd.DataFrame:
        """Turn a parameter set back into feature rows, re-deriving the behavioural block.

        This is the whole point of the file: `assemble_frame` is the SAME function the
        red team renders with and the same one applied to the 590,540 real transactions,
        so a moved timeline produces a coherent row by construction rather than by a
        check afterwards.
        """
        ts = ts0.copy()
        ts[idx] = self._rebuild_timestamps(params["gaps"], ent_a, anchor)
        amount = self._amount_all.copy()
        amount[idx] = params["amount"]

        df = assemble_frame(ent, ts, amount, atk, inherited, INHERITED_FEATURES)
        rows = df[atk].reset_index(drop=True)
        # the four settable columns are assigned, not derived
        rows["amount"] = params["amount"]
        rows["is_new_beneficiary"] = params["is_new_beneficiary"]
        for col in ENTITY_LEVEL_FEATURES:
            rows[col] = params[col][ent_code]
        for col in INTEGER_FEATURES:
            rows[col] = rows[col].round().astype(int)
        return rows

    def _derived_on_manifold(self, rows: pd.DataFrame) -> float:
        cols = [c for c in DERIVED_FEATURES if c in self.lo.index]
        if not cols or len(rows) == 0:
            return 1.0
        inside = np.ones(len(rows), bool)
        for c in cols:
            inside &= (rows[c] >= self.lo[c]) & (rows[c] <= self.hi[c])
        return round(float(inside.mean()), 4)

    @staticmethod
    def _new_timeline(tl, idx, params, ent_a, anchor) -> pd.DataFrame:
        out = tl.copy()
        ts = out["timestamp_s"].to_numpy(np.float64).copy()
        amt = out["amount"].to_numpy(np.float64).copy()
        cum = pd.Series(params["gaps"]).groupby(ent_a).cumsum().to_numpy()
        ts[idx] = anchor + cum
        amt[idx] = params["amount"]
        out["timestamp_s"] = ts.astype(np.int64)
        out["amount"] = amt
        return out

