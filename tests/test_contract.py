"""Smoke tests for the loop interface contract and a mini end-to-end pass.

    python -m pytest tests/ -q      (or just: python tests/test_contract.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, AttackBatch
from chhal.data import load_base_data
from chhal.detector import Detector
from chhal.loop import LoopConfig, run_loop
from chhal.optimizer import EvasionOptimizer
from chhal.redteam import ALL_VECTORS


def test_attackbatch_rejects_wrong_feature_space():
    bad = pd.DataFrame({"amount": [1.0], "surprise_col": [2.0]})
    with pytest.raises(ValueError):
        AttackBatch("bad", 0, bad).validate()


def test_all_vectors_emit_frozen_feature_space():
    rng = np.random.default_rng(0)
    for V in ALL_VECTORS:
        batch = V().batch(20, 1, rng)              # .validate() runs inside .batch()
        assert list(batch.transactions.columns) == FEATURE_COLUMNS
        assert len(batch) == 20


def test_optimizer_lowers_detector_score_and_stays_plausible():
    base = load_base_data(n_legit=4000, n_baseline_fraud=100, seed=1)
    detector = Detector(seed=1).fit(base.train)
    opt = EvasionOptimizer(base.feature_stats)
    rng = np.random.default_rng(1)

    seed_batch = ALL_VECTORS[1]().batch(200, 1, rng)   # bustout: easy to detect
    before = detector.score(seed_batch.transactions).mean()
    adapted = opt.optimize(seed_batch, detector, rng)
    after = detector.score(adapted.transactions).mean()

    assert after < before, "evasion optimizer should lower the detector's fraud score"
    # plausibility guardrail: every feature stays within the manifold bounds
    lo, hi = base.feature_stats.loc[0.005], base.feature_stats.loc[0.995]
    for col in FEATURE_COLUMNS:
        assert adapted.transactions[col].min() >= lo[col] - 1e-6
        assert adapted.transactions[col].max() <= hi[col] + 1e-6


def test_loop_runs_and_produces_a_curve():
    result = run_loop(
        LoopConfig(iterations=2, attacks_per_vector=150, benchmark_per_vector=150, seed=3)
    )
    assert {"benchmark", "pressure"}.issubset(set(result.curve["phase"]))
    assert result.curve["f1"].between(0, 1).all()
    assert result.curve["fp_rate_on_legit"].between(0, 1).all()

    # the closed loop must lift benchmark recall well above the static baseline:
    # generalisation to attacks the detector never trained on.
    bench = result.curve[result.curve["phase"] == "benchmark"].sort_values("iteration")
    assert bench["recall"].iloc[-1] > bench["recall"].iloc[0] + 0.3

    # fidelity is populated and the guardrail keeps attacks on the manifold
    assert result.fidelity["on_manifold_rate"] > 0.98
    assert 0.0 <= result.fidelity["mimicry_mean_ks_vs_legit"] <= 1.0
    assert set(result.fidelity_per_vector.columns) == {
        "vector", "mean_ks_vs_legit", "features_like_legit"}
    # the mimicry vector must be the closest to legit of all vectors
    pv = result.fidelity_per_vector.sort_values("mean_ks_vs_legit")
    assert pv.iloc[0]["vector"] == "threshold_hugging"


def test_fidelity_flags_off_manifold_attacks():
    """Pushing an in-distribution sample off the manifold must lower the on-manifold rate."""
    from chhal.contract import FEATURE_COLUMNS
    from chhal.data import load_base_data
    from chhal.fidelity import on_manifold_rate

    base = load_base_data(n_legit=4000, n_baseline_fraud=100, seed=2)
    # legit rows are in-distribution by construction -> rate should be near 1.0
    legit = base.test[base.test["is_fraud"] == 0][FEATURE_COLUMNS].copy()
    good = on_manifold_rate(legit, base.feature_stats)
    assert good > 0.95

    legit["amount"] = legit["amount"] * 1000  # blatantly above the manifold ceiling
    bad = on_manifold_rate(legit, base.feature_stats)
    assert bad < good


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
