"""The closed loop — orchestration that produces the arms-race curve.

We measure the two things that tell the true story, and keep them separate:

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

from .contract import FEATURE_COLUMNS, LABEL_COLUMN, AttackBatch
from .data import BaseData, load_base_data
from .detector import Detector
from .evaluation import evaluate, split_attacks
from .fidelity import fidelity_report
from .optimizer import EvasionOptimizer
from .redteam import ALL_VECTORS
from .redteam.base import BaseProfile


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
    config: Dict = field(default_factory=dict)


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
    vectors = [V().calibrate(profile) for V in ALL_VECTORS]
    legit_eval = base.test[base.test[LABEL_COLUMN] == 0]

    # -- build the FIXED adversarial benchmark once, against the baseline detector ----
    # These are hard (they evade the baseline). They are held out forever — the detector
    # never trains on them — so improving on them proves generalisation, not memory.
    bench_batches = [
        optimizer.optimize(v.batch(cfg.benchmark_per_vector, 0, rng), detector, rng)
        for v in vectors
    ]
    bench_attacks = pd.concat([b.transactions for b in bench_batches], ignore_index=True)
    bench_vec = np.concatenate([[b.vector_id] * len(b) for b in bench_batches])

    def bench_report(det: Detector, it: int):
        return evaluate(det, legit_eval, bench_attacks, bench_vec, it, "heldout_novel")

    curve_rows: List[Dict] = []
    per_vec_rows: List[Dict] = []

    # iteration 0 — baseline detector vs the benchmark (expected: low, they evade it)
    b0 = bench_report(detector, 0)
    curve_rows.append({**b0.as_row(), "phase": "benchmark"})
    baseline_report = b0
    for vid, rec in b0.per_vector_recall.items():
        per_vec_rows.append({"iteration": 0, "vector": vid, "recall": rec})

    train_pool = base.train.copy()
    last_adapted: List[AttackBatch] = []

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
        for vid, rec in bench.per_vector_recall.items():
            per_vec_rows.append({"iteration": t, "vector": vid, "recall": rec})

        # 4b. PRESSURE — retrained detector on this iteration's fresh evasions
        pressure = evaluate(detector, legit_eval, ho, ho_vec, t, "heldout_novel")
        curve_rows.append({**pressure.as_row(), "phase": "pressure"})

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
        config={**asdict(cfg), "data_source": base.source,
                "train_rows": len(base.train), "test_rows": len(base.test)},
    )
