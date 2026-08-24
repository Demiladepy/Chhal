"""Smoke tests for the loop interface contract and a mini end-to-end pass.

    python -m pytest tests/ -q      (or just: python tests/test_contract.py)

These run on the SYNTHETIC source on purpose: they must pass on a fresh clone with no
683MB download, and they must be fast. The real-data path is covered by
`test_real_ieee_source_if_prepared`, which skips when the parquet has not been built.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, INTEGER_FEATURES, AttackBatch
from chhal.data import DEFAULT_IEEE_PARQUET, load_base_data
from chhal.detector import Detector
from chhal.loop import LoopConfig, run_loop
from chhal.optimizer import EvasionOptimizer
from chhal.redteam import ALL_VECTORS
from chhal.redteam.base import BaseProfile

SMALL = dict(source="synthetic", n_legit=4000, n_baseline_fraud=100)


def profile_for(base) -> BaseProfile:
    return BaseProfile(base.legit_quantiles, base.legit_categoricals)


def test_attackbatch_rejects_wrong_feature_space():
    bad = pd.DataFrame({"amount": [1.0], "surprise_col": [2.0]})
    with pytest.raises(ValueError):
        AttackBatch("bad", 0, bad).validate()


def test_vector_requires_calibration():
    """A vector has no absolute scale of its own — rendering uncalibrated must fail loudly
    rather than silently emitting values from some other dataset's range."""
    with pytest.raises(RuntimeError, match="not calibrated"):
        ALL_VECTORS[0]().batch(5, 0, np.random.default_rng(0))


def test_all_vectors_emit_frozen_feature_space():
    base = load_base_data(seed=1, **SMALL)
    prof = profile_for(base)
    rng = np.random.default_rng(0)
    for V in ALL_VECTORS:
        batch = V().calibrate(prof).batch(20, 1, rng)   # .validate() runs inside .batch()
        assert list(batch.transactions.columns) == FEATURE_COLUMNS
        assert len(batch) == 20
        for col in INTEGER_FEATURES:
            assert (batch.transactions[col] % 1 == 0).all(), f"{col} must be whole numbers"


def test_calibrated_vectors_stay_inside_the_real_value_range():
    """Seed attacks are drawn through the legit inverse CDF, so before the optimizer
    touches anything they must already sit inside the observed range of the population."""
    base = load_base_data(seed=5, **SMALL)
    prof = profile_for(base)
    rng = np.random.default_rng(5)
    legit = base.train[base.train["is_fraud"] == 0]
    for V in ALL_VECTORS:
        rows = V().calibrate(prof).batch(300, 0, rng).transactions
        for col in ("amount", "account_age_days", "merchant_risk", "time_since_last_txn_min"):
            assert rows[col].min() >= legit[col].min() - 1e-6, f"{V.vector_id}.{col} below observed"
            assert rows[col].max() <= legit[col].max() + 1e-6, f"{V.vector_id}.{col} above observed"


def test_optimizer_lowers_detector_score_and_stays_plausible():
    base = load_base_data(seed=1, **SMALL)
    detector = Detector(seed=1).fit(base.train)
    opt = EvasionOptimizer(base.feature_stats)
    rng = np.random.default_rng(1)

    seed_batch = ALL_VECTORS[1]().calibrate(profile_for(base)).batch(200, 1, rng)  # bustout
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
    base = load_base_data(seed=3, **SMALL)
    result = run_loop(
        LoopConfig(iterations=2, attacks_per_vector=150, benchmark_per_vector=150, seed=3),
        base=base,
    )
    assert result.config["data_source"] == "synthetic"   # provenance must be recorded
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
    from chhal.fidelity import on_manifold_rate

    base = load_base_data(seed=2, **SMALL)
    # legit rows are in-distribution by construction -> rate should be near 1.0
    legit = base.test[base.test["is_fraud"] == 0][FEATURE_COLUMNS].copy()
    good = on_manifold_rate(legit, base.feature_stats)
    assert good > 0.95

    legit["amount"] = legit["amount"] * 1000  # blatantly above the manifold ceiling
    bad = on_manifold_rate(legit, base.feature_stats)
    assert bad < good


@pytest.mark.skipif(not os.path.exists(DEFAULT_IEEE_PARQUET),
                    reason="run scripts/prepare_ieee.py to test the real-data path")
def test_real_ieee_source_if_prepared():
    """The real path must load, split temporally, and carry the genuine fraud rate."""
    base = load_base_data(source="ieee")
    assert base.source == "ieee"
    assert len(base.train) + len(base.test) == 590_540
    total_fraud = base.train["is_fraud"].sum() + base.test["is_fraud"].sum()
    assert total_fraud == 20_663, "not the genuine IEEE-CIS label set"
    assert list(base.legit_quantiles.columns) == FEATURE_COLUMNS
    # manifold must come from train only — never from rows the detector will be tested on
    assert base.feature_stats["amount"].loc[0.995] <= base.train["amount"].max()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
