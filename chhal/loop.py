"""The closed loop — orchestration that produces the arms-race curve.

We measure the two things that tell the true story, and keep them separate:

All recalls on the curve are measured at a FIXED false-positive budget (0.1% of real
legitimate traffic), not at a 0.5 threshold — see evaluation.py for why.

  * BENCHMARK (blue generalisation, the money chart): a FIXED set of hard adaptive
    attacks, built once against the baseline detector and NEVER trained on. As the loop
    feeds the detector diverse adaptive attacks, its recall on this fixed held-out
    benchmark should RISE — evidence the defence learns the *shape* of adaptive fraud,
    not specific attacks.

  * PRESSURE (red's ongoing probing): each iteration the red team optimises a FRESH
    batch against the current detector and holds it out. Scored by the retrained
    detector, this line stays volatile/low — the red team keeps finding new evasions.

Per iteration t:
  1. Red team renders seed attacks, the evasion optimizer adapts them against
     detector_{t-1}. A held-out slice becomes this iteration's PRESSURE probe.
  2. The train slice is added to the pool; the detector retrains -> detector_t.
  3. Record BENCHMARK (detector_t on the fixed benchmark) and PRESSURE (detector_t on
     this iteration's held-out fresh attacks).

Nothing in the benchmark ever enters training; the base train/test split is frozen
before any attack is injected. That is the no-leakage guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .behaviour import consistency_violations
from .contract import FEATURE_COLUMNS, LABEL_COLUMN, AttackBatch
from .data import BaseData, load_base_data
from .detector import Detector
from .evaluation import evaluate, split_attacks
from .fidelity import fidelity_report
from .optimizer import EvasionOptimizer
from .redteam import ALL_VECTORS
from .redteam.base import BaseProfile
from .redteam.hosts import HostPool


@dataclass
class LoopConfig:
    iterations: int = 8
    attacks_per_vector: int = 400       # fresh adaptive attacks per iteration
    benchmark_per_vector: int = 500     # fixed held-out benchmark (never trained on)
    heldout_frac: float = 0.4
    seed: int = 7


@dataclass
class LoopResult:
    curve: pd.DataFrame                 # one row per (iteration, phase)
    per_vector_recall: pd.DataFrame     # BENCHMARK recall per vector over iterations
    baseline: Dict
    sample_attacks: pd.DataFrame        # a few adapted rows for the UI stream
    fidelity: Dict = field(default_factory=dict)         # headline fidelity numbers
    fidelity_per_vector: pd.DataFrame = field(default_factory=pd.DataFrame)  # KS vs legit
    fidelity_ks_table: pd.DataFrame = field(default_factory=pd.DataFrame)    # mimic feature KS
    fidelity_legit: pd.DataFrame = field(default_factory=pd.DataFrame)       # legit sample (plot)
    fidelity_mimic: pd.DataFrame = field(default_factory=pd.DataFrame)       # mimic attacks (plot)
    # Evidence for the no-leakage claim, computed rather than asserted. Every count in
    # here must be zero; see `_leakage_audit`.
    leakage_audit: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)


def _row_keys(df: pd.DataFrame) -> set:
    """Exact-match identity for feature rows, for the leakage audit only.

    Rounded at nine places so a float round-trip through concat/parquet cannot make a
    leaked row look novel. Two genuinely distinct rows colliding on all 26 features is
    not a failure mode worth engineering around; a leaked row slipping through is.
    """
    return set(map(tuple, np.round(df[FEATURE_COLUMNS].to_numpy(float), 9)))


def _host_accounts(batches: List[AttackBatch]) -> set:
    """The real accounts a set of batches was mounted on."""
    out: set = set()
    for b in batches:
        if b.timeline is not None and "host_account" in b.timeline.columns:
            out |= set(pd.unique(b.timeline["host_account"]))
    return out


def _leakage_audit(train_pool, bench_attacks, bench_batches, pressure, base) -> Dict:
    """Count the ways the headline could be cheating. Every number must be zero.

    The docstring at the top of this file makes three promises. Mutation testing showed
    all three were unguarded: deleting the benchmark/train separation, collapsing the
    held-out slice onto the train slice, and dropping the train/test account exclusion
    each left the whole suite green — the first two make recall RISE, so the tests
    passed harder for the leak. These counts are what a reader can check instead.
    """
    pool_keys = _row_keys(train_pool)
    bench_accounts = _host_accounts(bench_batches)
    return {
        "benchmark_rows": int(len(bench_attacks)),
        "benchmark_rows_in_training_pool": int(len(_row_keys(bench_attacks) & pool_keys)),
        "pressure_rows_in_training_pool":
            int(len(_row_keys(pressure) & pool_keys)) if len(pressure) else 0,
        "benchmark_host_accounts": int(len(bench_accounts)),
        "benchmark_host_accounts_seen_in_train":
            int(len(bench_accounts & set(base.train['_account'].unique()))),
    }


def run_loop(cfg: LoopConfig | None = None, base: BaseData | None = None) -> LoopResult:
    cfg = cfg or LoopConfig()
    rng = np.random.default_rng(cfg.seed)
    base = base or load_base_data(seed=cfg.seed)
    print(f"[data] {base.describe()}")

    detector = Detector(seed=cfg.seed).fit(base.train, LABEL_COLUMN)
    optimizer = EvasionOptimizer(base.feature_stats)
    # Bind every vector to THIS population before it renders anything. A vector has
    # no absolute scale of its own — it describes where in legitimate traffic it sits,
    # so it must be told what legitimate traffic looks like here.
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)

    # Two host pools, and which one a campaign is mounted on is a leakage decision.
    # The FIXED BENCHMARK is the headline number, scored against test-side legitimate
    # traffic, so its campaigns compromise TEST accounts — otherwise an evaluation attack
    # would carry issuer-side context the detector trained on. The per-iteration attacks
    # mostly exist to be retrained on, so they compromise TRAIN accounts.
    train_hosts = HostPool(base.train)
    test_hosts = HostPool(base.test, exclude_accounts=base.train['_account'])
    print(f"[hosts] train: {train_hosts.describe()}")
    print(f"[hosts] test:  {test_hosts.describe()}")
    vectors = [V().calibrate(profile, train_hosts) for V in ALL_VECTORS]
    bench_vectors = [V().calibrate(profile, test_hosts) for V in ALL_VECTORS]
    legit_eval = base.test[base.test[LABEL_COLUMN] == 0]

    # -- build the FIXED adversarial benchmark once, against the baseline detector ----
    # These are hard (they evade the baseline). They are held out forever — the detector
    # never trains on them — so improving on them proves generalisation, not memory.
    bench_batches = [
        optimizer.optimize(v.batch(cfg.benchmark_per_vector, 0, rng), detector, rng)
        for v in bench_vectors
    ]
    bench_attacks = pd.concat([b.transactions for b in bench_batches], ignore_index=True)
    bench_vec = np.concatenate([[b.vector_id] * len(b) for b in bench_batches])

    def bench_report(det: Detector, it: int):
        return evaluate(det, legit_eval, bench_attacks, bench_vec, it, "heldout_novel",
                        real_traffic=base.test)

    curve_rows: List[Dict] = []
    per_vec_rows: List[Dict] = []

    # iteration 0 — baseline detector vs the benchmark (expected: low, they evade it)
    b0 = bench_report(detector, 0)
    curve_rows.append({**b0.as_row(), "phase": "benchmark"})
    baseline_report = b0
    for vid, rec in b0.per_vector_recall_at_fpr.items():
        per_vec_rows.append({"iteration": 0, "vector": vid, "recall": rec,
                             "recall_at_threshold_0.5": b0.per_vector_recall.get(vid, 0.0)})

    train_pool = base.train.copy()
    last_adapted: List[AttackBatch] = []
    last_pressure = pd.DataFrame(columns=FEATURE_COLUMNS)

    for t in range(1, cfg.iterations + 1):
        # 1. red team adapts a FRESH batch against the CURRENT detector
        adapted: List[AttackBatch] = []
        for v in vectors:
            seed = v.batch(cfg.attacks_per_vector, t, rng)
            adapted.append(optimizer.optimize(seed, detector, rng))
        last_adapted = adapted

        # 2. split: train slice feeds retraining; held-out slice is this iter's PRESSURE
        tr, ho, ho_vec = split_attacks(adapted, cfg.heldout_frac, rng)

        # 3. retrain, adding ONLY the train slice (benchmark + pressure never leak in)
        tr_labeled = tr.copy()
        tr_labeled[LABEL_COLUMN] = 1
        train_pool = pd.concat([train_pool, tr_labeled], ignore_index=True)
        detector = Detector(seed=cfg.seed).fit(train_pool, LABEL_COLUMN)

        # 4a. BENCHMARK — blue generalisation on the fixed held-out set (should rise)
        bench = bench_report(detector, t)
        curve_rows.append({**bench.as_row(), "phase": "benchmark"})
        for vid, rec in bench.per_vector_recall_at_fpr.items():
            per_vec_rows.append({"iteration": t, "vector": vid, "recall": rec,
                                 "recall_at_threshold_0.5": bench.per_vector_recall.get(vid, 0.0)})

        # 4b. PRESSURE — retrained detector on this iteration's fresh evasions
        pressure = evaluate(detector, legit_eval, ho, ho_vec, t, "heldout_novel",
                            real_traffic=base.test)
        curve_rows.append({**pressure.as_row(), "phase": "pressure"})
        last_pressure = ho

    # a small sample of the final adapted attacks for the live-stream panel
    sample = pd.concat(
        [b.transactions.head(15).assign(vector=b.vector_id) for b in last_adapted],
        ignore_index=True,
    )

    # -- fidelity: measured on the (hard, fully-optimized) benchmark attacks --------------
    # These are the most heavily optimized, so they are the strongest test of both the
    # plausibility guardrail (on-manifold rate) and the mimicry claim (KS vs legit).
    legit_pop = legit_eval[FEATURE_COLUMNS]
    fid = fidelity_report(legit_pop, bench_attacks, bench_vec, base.feature_stats)
    # non-tautological companion to on_manifold_rate (see fidelity.py docstring): how
    # often the search proposed a move outside the manifold and the guardrail pulled it
    # back, averaged over the fully-optimized benchmark batches.
    fid["frac_off_manifold_pre_clip"] = round(float(np.mean(
        [b.provenance["frac_off_manifold_pre_clip"] for b in bench_batches])), 4)
    # Physical consistency, measured on the OPTIMIZED benchmark rather than the seed.
    # Measuring it on the seed is what let a version of the optimizer ship that broke
    # the invariant on 59-93% of the rows anything downstream actually saw.
    fid["consistency_violations"] = {
        k: round(float(v), 6) for k, v in consistency_violations(bench_attacks).items()
    }
    per_vector_fid = fid.pop("per_vector")
    mimic_ks = fid.pop("mimicry_ks_table")
    mimic_attacks = bench_attacks[bench_vec == fid["mimicry_vector"]]

    return LoopResult(
        curve=pd.DataFrame(curve_rows),
        per_vector_recall=pd.DataFrame(per_vec_rows),
        baseline=baseline_report.as_row(),
        sample_attacks=sample,
        fidelity=fid,
        fidelity_per_vector=per_vector_fid,
        fidelity_ks_table=mimic_ks,
        fidelity_legit=legit_pop.sample(
            min(6000, len(legit_pop)), random_state=cfg.seed).reset_index(drop=True),
        fidelity_mimic=mimic_attacks.reset_index(drop=True),
        leakage_audit=_leakage_audit(train_pool, bench_attacks, bench_batches,
                                     last_pressure, base),
        config={**asdict(cfg), "data_source": base.source,
                "train_rows": len(base.train), "test_rows": len(base.test)},
    )
