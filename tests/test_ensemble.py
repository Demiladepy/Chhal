"""Tests for the second detector arm — including that its failure mode stays visible."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, OPERATING_POINTS, LABEL_COLUMN          # noqa: E402
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.ensemble import AnomalyArm, Ensemble, StackedDetector  # noqa: E402
from chhal.evaluation import evaluate                             # noqa: E402

SMALL = dict(source="synthetic", n_legit=4000, n_baseline_fraud=100, seed=1)


@pytest.fixture(scope="module")
def base():
    return load_base_data(**SMALL)


def test_arms_refuse_to_score_before_being_fitted(base):
    with pytest.raises(RuntimeError, match="before fit"):
        AnomalyArm().score(base.test)
    with pytest.raises(RuntimeError, match="before fit"):
        StackedDetector().score(base.test)


def test_ensemble_refuses_before_a_reference_is_set(base):
    det = Detector(seed=1).fit(base.train)
    ens = Ensemble(det, AnomalyArm().fit(base.train))
    with pytest.raises(RuntimeError, match="fit_reference"):
        ens.score(base.test)


def test_anomaly_arm_never_sees_a_fraud_label(base):
    """It is trained on legitimate rows only — that is what keeps it meaningful for an
    attack family invented after training."""
    arm = AnomalyArm()
    seen = {}
    real_fit = arm.model.fit

    def spy(X, *a, **k):
        seen["rows"] = len(X)
        return real_fit(X, *a, **k)

    arm.model.fit = spy
    arm.fit(base.train, LABEL_COLUMN)
    assert seen["rows"] == int((base.train[LABEL_COLUMN] == 0).sum())


def test_ensemble_percentiles_are_calibrated_on_legit(base):
    det = Detector(seed=1).fit(base.train)
    legit = base.test[base.test[LABEL_COLUMN] == 0]
    ens = Ensemble(det, AnomalyArm().fit(base.train)).fit_reference(legit)
    sup, anom = ens.arm_percentiles(legit)
    for p in (sup, anom):
        assert ((p >= 0) & (p <= 1)).all()
        assert 0.4 < p.mean() < 0.6, "legit traffic should sit mid-distribution by construction"
        # a percentile transform is a rank map, so the reference population has to come
        # out roughly uniform — a constant 0.5 would satisfy the mean check above
        assert 0.2 < np.median(p) < 0.8
        assert p.std() > 0.2, "percentiles are collapsed, not spread over the reference"
    assert (ens.score(legit) >= np.maximum(sup, anom) - 1e-9).all()


def test_stacked_detector_uses_the_anomaly_score_and_keeps_the_frozen_contract(base):
    """The anomaly score is an issuer-side signal added at scoring time, so attacks still
    live in FEATURE_COLUMNS and the contract is untouched."""
    st = StackedDetector(seed=1).fit(base.train)
    assert st.columns == FEATURE_COLUMNS + [StackedDetector.ANOMALY_COLUMN]
    assert StackedDetector.ANOMALY_COLUMN not in FEATURE_COLUMNS
    # scoring a frame that only has FEATURE_COLUMNS must work — nothing upstream changes
    out = st.score(base.test[FEATURE_COLUMNS])
    assert out.shape == (len(base.test),) and ((out >= 0) & (out <= 1)).all()


def test_stacked_detector_drops_into_evaluate_unchanged(base):
    """Same surface as Detector, so the loop and the mitigation policy take it as-is."""
    st = StackedDetector(seed=1).fit(base.train)
    legit = base.test[base.test[LABEL_COLUMN] == 0]
    fraud = base.test[base.test[LABEL_COLUMN] == 1]
    rep = evaluate(st, legit, fraud, np.array(["real"] * len(fraud)), 1, "heldout_novel")
    # `0 <= pr_auc <= 1` and `0 <= recall <= 1` are true of any probability, so they
    # passed with an inverted detector. Assert the report is INTERNALLY CONSISTENT
    # instead, which is what a drop-in replacement actually has to be.
    assert rep.pr_auc > len(fraud) / (len(fraud) + len(legit)), (
        "PR AUC is at or below the prevalence floor — this is not a working detector")
    assert set(rep.recall_at_fpr) == set(OPERATING_POINTS)
    for fpr, realised in rep.realised_fpr_at_fpr.items():
        assert realised <= fpr + 1e-12
    # relaxing the budget can never catch less
    ordered = [rep.recall_at_fpr[f] for f in sorted(OPERATING_POINTS)]
    assert ordered == sorted(ordered)


def test_anomaly_arm_is_blind_to_on_manifold_attacks(base):
    """The documented negative result, locked in. Rows drawn from inside legitimate
    traffic are, to an outlier detector, indistinguishable from legitimate traffic —
    which is exactly why the plausibility guardrail defeats this arm."""
    arm = AnomalyArm().fit(base.train, LABEL_COLUMN)
    legit = base.test[base.test[LABEL_COLUMN] == 0][FEATURE_COLUMNS]
    on_manifold = legit.sample(400, random_state=3)          # stand-in for a clipped attack
    thr = np.quantile(arm.score(legit), 0.999)
    assert (arm.score(on_manifold) >= thr).mean() < 0.05
