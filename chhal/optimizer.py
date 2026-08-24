"""The constrained evasion optimizer — the novel core, with the plausibility guardrail.

Given a seed AttackBatch and the CURRENT detector, nudge attacker-controllable features
to lower the detector's fraud score (maximise evasion) — but only within a realistic,
executable envelope. Without that guardrail you get "attacks" that fool the model yet
that no real fraudster could execute (impossible timing, off-manifold values), which
would destroy the real-world-feasibility score. The guardrail is the point.

Method: gradient-free (evolutionary hill-climb). No gradients through LightGBM needed,
which keeps it simple and model-agnostic.

Constraints on every candidate:
  (a) business rules  — velocity caps, non-negative amounts, valid channel codes;
  (b) realistic manifold — each feature stays within [q0.5%, q99.5%] of the base data;
  (c) attacker control — only ATTACKER_CONTROLLED features may move.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .contract import (ATTACKER_CONTROLLED, FEATURE_COLUMNS,
                       INTEGER_FEATURES, AttackBatch)
from .detector import Detector


@dataclass
class OptimizerConfig:
    generations: int = 12
    population: int = 24
    step_scale: float = 0.15      # perturbation size as fraction of feature range
    velocity_1h_cap: int = 25     # business rule: hard velocity ceiling
    velocity_24h_cap: int = 150


class EvasionOptimizer:
    def __init__(self, feature_stats: pd.DataFrame, cfg: OptimizerConfig | None = None):
        self.cfg = cfg or OptimizerConfig()
        # realistic manifold bounds from the base population
        self.lo = feature_stats.loc[0.005].copy()
        self.hi = feature_stats.loc[0.995].copy()
        self._int_features = set(INTEGER_FEATURES)

        # Integer features are rounded at the end of _clip_to_manifold, and rounding can
        # walk a value straight back out of the manifold: a q99.5 ceiling of 3.63 on
        # velocity_1h becomes 4 after rounding, which is off-manifold by construction.
        # So pull the bounds INWARD to the nearest whole numbers first — then any value
        # inside them is still inside them after rounding. (Skipped in the degenerate
        # case where no whole number fits between the bounds.)
        for col in self._int_features:
            lo_i, hi_i = np.ceil(self.lo[col]), np.floor(self.hi[col])
            if lo_i <= hi_i:
                self.lo[col], self.hi[col] = lo_i, hi_i

    # -- constraints ---------------------------------------------------------
    def _clip_to_manifold(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in FEATURE_COLUMNS:
            df[col] = df[col].clip(self.lo[col], self.hi[col])
        # business rules (hard, independent of the statistical manifold)
        df["amount"] = df["amount"].clip(lower=0.5)
        df["velocity_1h"] = df["velocity_1h"].clip(0, self.cfg.velocity_1h_cap)
        df["velocity_24h"] = df["velocity_24h"].clip(0, self.cfg.velocity_24h_cap)
        df["channel_code"] = df["channel_code"].round().clip(0, 2)
        df["hour"] = df["hour"].round().clip(0, 23)
        df["day_of_week"] = df["day_of_week"].round().clip(0, 6)
        for c in ("is_new_beneficiary", "is_cross_border"):
            df[c] = df[c].round().clip(0, 1)
        for c in self._int_features:
            df[c] = df[c].round()
        return df

    def _off_manifold_frac(self, df: pd.DataFrame) -> float:
        """Fraction of (row, attacker-feature) cells outside the manifold bounds
        BEFORE clipping. Non-tautological companion to fidelity.on_manifold_rate:
        that metric is measured on the optimizer's OWN clipped output against the
        SAME bounds, so it is ~1.0 by construction and only proves the clip is wired
        correctly. This measures how often the search actually wanted to leave the
        manifold — i.e. whether the guardrail is doing real work, not just present.
        """
        total = len(df) * len(ATTACKER_CONTROLLED)
        if total == 0:
            return 0.0
        violations = sum(
            int(((df[col] < self.lo[col]) | (df[col] > self.hi[col])).sum())
            for col in ATTACKER_CONTROLLED
        )
        return violations / total

    def _perturb(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        for col in ATTACKER_CONTROLLED:
            rng_span = max(self.hi[col] - self.lo[col], 1e-6)
            noise = rng.normal(0, self.cfg.step_scale * rng_span, len(df))
            out[col] = out[col] + noise
        self._clip_hits.append(self._off_manifold_frac(out))
        return self._clip_to_manifold(out)

    # -- main ----------------------------------------------------------------
    def optimize(
        self, batch: AttackBatch, detector: Detector, rng: np.random.Generator
    ) -> AttackBatch:
        """Return an adapted AttackBatch that evades `detector` while staying plausible."""
        self._clip_hits: List[float] = []   # reset per call; filled by _perturb
        best = self._clip_to_manifold(batch.transactions.copy())
        best_score = detector.score(best)                     # per-row fraud proba

        for _ in range(self.cfg.generations):
            # generate a population of perturbations, keep the best per row
            for _ in range(self.cfg.population):
                cand = self._perturb(best, rng)
                cand_score = detector.score(cand)
                improved = cand_score < best_score            # lower = more evasive
                if improved.any():
                    best.loc[improved] = cand.loc[improved].values
                    best_score = np.where(improved, cand_score, best_score)

        adapted = AttackBatch(
            vector_id=batch.vector_id,
            iteration=batch.iteration,
            transactions=best[FEATURE_COLUMNS].reset_index(drop=True),
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
            },
        ).validate()
        return adapted
