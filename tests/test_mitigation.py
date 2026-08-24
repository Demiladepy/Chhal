"""Tests for the mitigation layer — the part that turns a score into a decision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.mitigation import (Action, ActionPolicy, Calibrator, CostModel,  # noqa: E402
                              PolicyConfig, allow_all_baseline,
                              calibration_error, threshold_baseline)


def test_calibrator_refuses_to_predict_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        Calibrator()(np.array([0.3, 0.7]))


def test_calibration_improves_expected_calibration_error():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.05).astype(int)
    # a badly scaled score: right ordering, wrong magnitudes
    raw = np.clip(rng.beta(2, 5, 20_000) + 0.35 * y, 0, 1)
    cal = Calibrator().fit(raw, y)
    assert calibration_error(cal(raw), y) < calibration_error(raw, y)


def test_expected_costs_have_one_column_per_action():
    c = CostModel()
    cost = c.expected_costs(np.array([0.01, 0.5, 0.99]), np.array([10.0, 100.0, 1000.0]))
    assert cost.shape == (3, len(Action))
    assert np.isfinite(cost).all()


def test_certain_fraud_is_never_allowed_and_certain_legit_is_never_blocked():
    pol = ActionPolicy(CostModel(), PolicyConfig(max_review_rate=1.0))
    amt = np.array([50.0, 500.0, 5000.0])
    assert (pol.decide(np.full(3, 0.999), amt) != Action.ALLOW).all()
    assert (pol.decide(np.full(3, 0.0001), amt) != Action.BLOCK).all()


def test_the_decision_is_amount_aware():
    """The same fraud probability must resolve differently on a small and a large amount —
    this is the thing a single global threshold structurally cannot do.

    At p=0.30 the defaults price a $5 transaction into a cheap OTP challenge (expected
    cost 3.56, against 9.00 to allow it) and a $5,000 one into analyst review (116.60,
    against 188.37 to challenge and 1507.50 to allow)."""
    pol = ActionPolicy(CostModel(), PolicyConfig(max_review_rate=1.0))
    actions = pol.decide(np.full(2, 0.30), np.array([5.0, 5000.0]))
    assert actions[0] == Action.STEP_UP
    assert actions[1] == Action.REVIEW


def test_review_queue_respects_its_capacity_cap():
    rng = np.random.default_rng(1)
    n = 20_000
    p, amt = rng.random(n), rng.lognormal(4, 1, n)
    pol = ActionPolicy(CostModel(), PolicyConfig(max_review_rate=0.005))
    actions = pol.decide(p, amt)
    assert (actions == Action.REVIEW).mean() <= 0.005 + 1e-9
    assert set(np.unique(actions)) <= {int(a) for a in Action}


def test_realised_cost_matches_the_cost_model_by_hand():
    c = CostModel()
    pol = ActionPolicy(c)
    amt = np.array([100.0, 100.0, 100.0, 100.0])
    y = np.array([1, 0, 1, 0])
    actions = np.array([Action.ALLOW, Action.BLOCK, Action.BLOCK, Action.ALLOW])
    got = pol.realised_cost(actions, y, amt)
    assert got[0] == pytest.approx(100.0 + c.chargeback_fee)   # allowed a fraud
    assert got[1] == pytest.approx(0.20 * 100.0 + c.false_decline_penalty)  # declined a customer
    assert got[2] == 0.0                                       # blocked a fraud
    assert got[3] == 0.0                                       # allowed a customer


def _calibrated_population(seed: int = 3, n: int = 40_000):
    """A population where p really is P(fraud): draw the probability, then draw the
    label from it. Anything else is not a probability and the economics do not hold."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1.0, 12.0, n)
    y = (rng.random(n) < p).astype(int)
    amt = rng.lognormal(4.2, 1.1, n)
    return p, y, amt


def test_expected_cost_policy_beats_a_fixed_threshold_and_doing_nothing():
    p, y, amt = _calibrated_population()
    c = CostModel()

    nothing = allow_all_baseline(c, y, amt)["total_cost"]
    naive = threshold_baseline(c, p, y, amt, 0.5)["total_cost"]
    pol = ActionPolicy(c, PolicyConfig(max_review_rate=0.005))
    smart = pol.report(pol.decide(p, amt), y, amt)["total_cost"]

    assert smart < naive < nothing


def test_miscalibrated_scores_degrade_the_policy():
    """Why Calibrator is not optional. Feed the same population's scores through a
    monotone squash — ranking untouched, so every recall/AUC number is identical — and
    the expected-cost policy gets measurably worse, because it is multiplying a number
    that is no longer a probability by a dollar amount."""
    p, y, amt = _calibrated_population()
    c = CostModel()
    pol = ActionPolicy(c, PolicyConfig(max_review_rate=0.005))

    honest = pol.report(pol.decide(p, amt), y, amt)["total_cost"]
    squashed = p * 0.5                       # same ordering, halved magnitudes
    misled = pol.report(pol.decide(squashed, amt), y, amt)["total_cost"]

    assert misled > honest, "under-confident scores should make the policy under-act"
