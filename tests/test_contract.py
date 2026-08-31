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

from chhal.behaviour import consistency_violations
from chhal.contract import (FEATURE_COLUMNS, INTEGER_FEATURES,
                            OPERATING_POINTS, PRIMARY_FPR, AttackBatch)
from chhal.data import DEFAULT_IEEE_PARQUET, load_base_data
from chhal.detector import Detector
from chhal.loop import LoopConfig, run_loop
from chhal.optimizer import EvasionOptimizer
from chhal.redteam import ALL_VECTORS
from chhal.redteam.base import BaseProfile
from chhal.redteam.hosts import HostPool

SMALL = dict(source="synthetic", n_legit=4000, n_baseline_fraud=100)


def profile_for(base) -> BaseProfile:
    return BaseProfile(base.legit_quantiles, base.legit_categoricals)


def armed(V, base):
    """A vector bound to the population AND to the accounts it may compromise."""
    return V().calibrate(profile_for(base), HostPool(base.train))


def test_attackbatch_rejects_wrong_feature_space():
    bad = pd.DataFrame({"amount": [1.0], "surprise_col": [2.0]})
    with pytest.raises(ValueError):
        AttackBatch("bad", 0, bad).validate()


def test_vector_requires_calibration():
    """A vector has no absolute scale of its own, rendering uncalibrated must fail loudly
    rather than silently emitting values from some other dataset's range."""
    with pytest.raises(RuntimeError, match="not calibrated"):
        ALL_VECTORS[0]().batch(5, 0, np.random.default_rng(0))


def test_all_vectors_emit_frozen_feature_space():
    base = load_base_data(seed=1, **SMALL)
    rng = np.random.default_rng(0)
    for V in ALL_VECTORS:
        batch = armed(V, base).batch(20, 1, rng)        # .validate() runs inside .batch()
        assert list(batch.transactions.columns) == FEATURE_COLUMNS
        assert len(batch) == 20
        for col in INTEGER_FEATURES:
            # dtype, not `% 1 == 0`, a float column satisfies the latter, so dropping
            # the integer coercion entirely left this assertion green.
            assert pd.api.types.is_integer_dtype(batch.transactions[col]), (
                f"{col} is {batch.transactions[col].dtype}, not an integer type")


def test_calibrated_vectors_stay_inside_the_real_value_range():
    """Columns drawn through the legit inverse CDF must sit inside the observed range
    before the optimizer touches anything.

    Only the non-derived columns are checked. `velocity_*`, `time_since_last_txn_min`
    and `amount_to_avg_ratio` now come from the campaign timeline rather than the
    quantile grid (see chhal/behaviour.py), so they can legitimately land outside the
    range a small sample happens to contain, a 7-day gap is ordinary in real traffic
    but may simply be absent from 3,000 synthetic rows. Their correctness is a
    consistency property, tested in test_behaviour.py.
    """
    base = load_base_data(seed=5, **SMALL)
    rng = np.random.default_rng(5)
    legit = base.train[base.train["is_fraud"] == 0]
    for V in ALL_VECTORS:
        rows = armed(V, base).batch(300, 0, rng).transactions
        for col in ("amount",):
            assert rows[col].min() >= legit[col].min() - 1e-6, f"{V.vector_id}.{col} below observed"
            assert rows[col].max() <= legit[col].max() + 1e-6, f"{V.vector_id}.{col} above observed"


def test_optimizer_lowers_detector_score_and_stays_plausible():
    base = load_base_data(seed=1, **SMALL)
    detector = Detector(seed=1).fit(base.train)
    opt = EvasionOptimizer(base.feature_stats)
    rng = np.random.default_rng(1)

    seed_batch = armed(ALL_VECTORS[1], base).batch(200, 1, rng)   # bustout
    before = detector.score(seed_batch.transactions).mean()
    adapted = opt.optimize(seed_batch, detector, rng)
    after = detector.score(adapted.transactions).mean()

    assert after < before, "evasion optimizer should lower the detector's fraud score"
    # Plausibility guardrail: every feature the attacker SETS stays within the manifold.
    from chhal.contract import ATTACKER_DIRECT
    lo, hi = base.feature_stats.loc[0.005], base.feature_stats.loc[0.995]
    for col in ATTACKER_DIRECT:
        assert adapted.transactions[col].min() >= lo[col] - 1e-6
        assert adapted.transactions[col].max() <= hi[col] + 1e-6


def test_the_optimizer_cannot_emit_an_impossible_transaction():
    """The guarantee the campaign architecture buys must survive the search.

    This is the test whose absence let the defect ship. The old optimizer perturbed
    velocity_1h, velocity_24h and time_since_last_txn_min as three independent scalars,
    so 59-93% of the rows it emitted were physically impossible, including rows claiming
    more transactions in the last hour than in the last twenty-four. The seed batches
    were clean and the only consistency test ran on THOSE, before the optimizer touched
    anything, so the suite stayed green while the benchmark, the fidelity population and
    the training additions were all incoherent.
    """
    base = load_base_data(seed=1, **SMALL)
    detector = Detector(seed=1).fit(base.train)
    opt = EvasionOptimizer(base.feature_stats)
    rng = np.random.default_rng(1)

    for vector in ALL_VECTORS:
        seed_batch = armed(vector, base).batch(150, 1, rng)
        adapted = opt.optimize(seed_batch, detector, rng)
        for name, frame in (("seed", seed_batch.transactions),
                            ("optimized", adapted.transactions)):
            viol = consistency_violations(frame)
            assert viol["velocity_1h_exceeds_24h"] == 0.0, (
                f"{vector.__name__} {name}: a 1-hour count cannot exceed the 24-hour "
                f"window that contains it, {viol}")
            assert viol["violates_1h_rule"] == 0.0, f"{vector.__name__} {name}: {viol}"
            assert viol["violates_24h_rule"] == 0.0, f"{vector.__name__} {name}: {viol}"


def test_loop_runs_and_produces_a_curve():
    base = load_base_data(seed=3, **SMALL)
    result = run_loop(
        LoopConfig(iterations=2, attacks_per_vector=150, benchmark_per_vector=150, seed=3),
        base=base,
    )
    assert result.config["data_source"] == "synthetic"   # provenance must be recorded
    assert {"benchmark", "pressure"}.issubset(set(result.curve["phase"]))

    # Every operating point must have honoured the budget it is named after. The old
    # assertions here were `f1.between(0, 1)` and `fp_rate.between(0, 1)`, which are
    # true of any number a probability can be. They passed with inverted predictions.
    for fpr in OPERATING_POINTS:
        realised = result.curve[f"realised_fpr_{fpr}"]
        assert (realised <= fpr + 1e-12).all(), (
            f"a threshold quoted at {fpr} realised {realised.max()}")

    # the closed loop must lift benchmark recall well above the static baseline:
    # generalisation to attacks the detector never trained on. Both a RISE and a FLOOR
    #. The rise alone would pass while recall collapsed from 0.92 to 0.30.
    bench = result.curve[result.curve["phase"] == "benchmark"].sort_values("iteration")
    primary = f"recall_at_fpr_{PRIMARY_FPR}"
    assert bench["recall"].iloc[-1] > bench["recall"].iloc[0] + 0.3
    assert bench[primary].iloc[-1] > bench[primary].iloc[0] + 0.2
    assert bench[primary].iloc[-1] > 0.30, (
        f"the loop ends at {bench[primary].iloc[-1]:.3f} recall inside a "
        f"{PRIMARY_FPR:.1%} budget. It rose, but from and to nowhere useful")

    # fidelity is populated and the guardrail keeps what the attacker SETS on the
    # manifold. The derived block is reported, not constrained, see fidelity.py.
    assert result.fidelity["on_manifold_rate"] > 0.98
    # ...and because that number is ~1.0 BY CONSTRUCTION (the optimizer clips to exactly
    # these bounds), assert the non-tautological companion too: the guardrail has to
    # have actually bound on a meaningful share of proposals, or it is decorative.
    assert result.fidelity["frac_off_manifold_pre_clip"] > 0.01, (
        "the plausibility guardrail never bound on anything")
    assert 0.0 <= result.fidelity["derived_on_manifold_rate"] <= 1.0

    # The mimicry claim, as a comparison rather than a range check. `0 <= ks <= 1` is
    # true by the definition of KS and passed with attacks identical to legit AND with
    # attacks at 1e12.
    ks = result.fidelity_per_vector.set_index("vector")["mean_ks_vs_legit"]
    assert ks["threshold_hugging"] < ks.max() * 0.75, (
        f"the mimicry vector is not measurably closer to legit than the loudest one: "
        f"{ks.to_dict()}")
    assert set(result.fidelity_per_vector.columns) == {
        "vector", "mean_ks_vs_legit", "mean_ks_controlled",
        # the same two distances in multiples of the legit-vs-legit noise floor, which
        # is the only form that is comparable across features, see fidelity.KS_NULL_FLOOR
        "mean_degradation_ratio", "mean_degradation_ratio_controlled",
        "features_like_legit", "controlled_like_legit", "n_controlled"}
    # the restricted distance must be the LARGER one, if it is not, the inherited
    # columns are not matching by construction and something upstream is wrong
    fpv = result.fidelity_per_vector
    assert (fpv["mean_ks_controlled"] >= fpv["mean_ks_vs_legit"] - 1e-9).all(), fpv
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
    import pandas as pd

    base = load_base_data(source="ieee")
    assert base.source == "ieee"
    assert list(base.legit_quantiles.columns) == FEATURE_COLUMNS
    # manifold must come from train only, never from rows the detector will be tested on
    assert base.feature_stats["amount"].loc[0.995] <= base.train["amount"].max()

    # train + test no longer sum to the corpus: a 7-day delay period and the straddling
    # accounts are deliberately held out of both. Assert against the parquet instead, or
    # this test silently passes on a split that has quietly stopped purging anything.
    raw = pd.read_parquet(DEFAULT_IEEE_PARQUET)
    assert len(raw) == 590_540
    assert raw["is_fraud"].sum() == 20_663, "not the genuine IEEE-CIS label set"
    assert set(raw["split"]) == {"train", "test", "embargo", "straddle"}
    assert len(base.train) + len(base.test) < len(raw), "nothing is being held out"

    # the two holdouts, asserted rather than trusted
    train_end = raw.loc[raw.split == "train", "_ts"].max()
    test_start = raw.loc[raw.split == "test", "_ts"].min()
    assert test_start - train_end >= 7 * 86_400, "delay period shorter than 7 days"

    train_accounts = set(raw.loc[raw.split == "train", "_account"])
    test_accounts = set(raw.loc[raw.split == "test", "_account"])
    assert not (train_accounts & test_accounts), "an account appears in both train and test"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
