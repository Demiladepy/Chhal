"""Tests for the mitigation layer. The part that turns a score into a decision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.mitigation import (Action, ActionPolicy, Calibrator, CostModel,  # noqa: E402
                              PolicyConfig, allow_all_baseline,
                              calibration_error, fraud_loss_avoided,
                              segment_costs, threshold_baseline,
                              tune_two_thresholds, two_threshold_baseline)


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
    """The same fraud probability must resolve differently on a small and a large amount.
    This is the thing a single global threshold structurally cannot do.

    At p=0.30 the defaults price a $5 transaction into a cheap OTP challenge (expected
    cost 4.16, against 9.00 to allow it) and a $5,000 one into analyst review (116.60,
    against 188.98 to challenge and 1507.50 to allow). Both caps are lifted here. This
    test is about the ordering the cost model produces, not about capacity."""
    pol = ActionPolicy(CostModel(),
                       PolicyConfig(max_review_rate=1.0, max_stepup_rate=1.0))
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
    monotone squash, ranking untouched, so every recall/AUC number is identical, and
    the expected-cost policy gets measurably worse, because it is multiplying a number
    that is no longer a probability by a dollar amount."""
    p, y, amt = _calibrated_population()
    c = CostModel()
    pol = ActionPolicy(c, PolicyConfig(max_review_rate=0.005))

    honest = pol.report(pol.decide(p, amt), y, amt)["total_cost"]
    squashed = p * 0.5                       # same ordering, halved magnitudes
    misled = pol.report(pol.decide(squashed, amt), y, amt)["total_cost"]

    assert misled > honest, "under-confident scores should make the policy under-act"


def test_the_tuned_ladder_is_the_best_threshold_pair_that_exists():
    """The comparator has to be the strongest amount-blind policy, not a convenient one.

    tune_two_thresholds claims a GLOBAL optimum via prefix sums rather than a grid search.
    If that claim is wrong, some random pair of thresholds will beat it, so try many.
    """
    p, y, amt = _calibrated_population()
    c = CostModel()
    t_s, t_b = tune_two_thresholds(c, p, y, amt)
    best = two_threshold_baseline(c, p, y, amt, t_s, t_b)["total_cost"]

    rng = np.random.default_rng(11)
    for _ in range(200):
        a, b = np.sort(rng.choice(p, 2, replace=False))
        rival = two_threshold_baseline(c, p, y, amt, float(a), float(b))["total_cost"]
        assert rival >= best - 1e-6, f"({a}, {b}) beat the supposed optimum"


def test_the_capped_ladder_is_the_best_pair_that_fits_the_friction_budget():
    """Same optimality claim under a constraint, which is where it is easiest to fake.

    The constrained search must (a) actually respect the cap and (b) still be optimal
    among the pairs that do. A rival that spends more friction than allowed does not
    count as beating it.
    """
    p, y, amt = _calibrated_population()
    c, cap, n = CostModel(), 0.05, len(p)
    t_s, t_b = tune_two_thresholds(c, p, y, amt, max_stepup_frac=cap)
    rep = two_threshold_baseline(c, p, y, amt, t_s, t_b)
    assert rep["stepup_rate"] <= cap + 1e-9, rep["stepup_rate"]

    rng = np.random.default_rng(12)
    for _ in range(200):
        a, b = np.sort(rng.choice(p, 2, replace=False))
        rival = two_threshold_baseline(c, p, y, amt, float(a), float(b))
        if rival["stepup_rate"] > cap + 1e-9:
            continue                     # spends friction it is not allowed to spend
        assert rival["total_cost"] >= rep["total_cost"] - 1e-6, (
            f"({a}, {b}) fits the cap and beat the supposed constrained optimum")
    assert n > 0


def test_the_real_edge_is_measured_against_the_tuned_ladder_not_the_naive_threshold():
    """The claim we are allowed to make.

    Beating `score >= 0.5` is not evidence for expected-cost decisions; it is evidence
    that 0.5 is a bad threshold. The policy must beat the best AMOUNT-BLIND ladder, and
    the gap between the two is the honest size of the contribution.
    """
    p, y, amt = _calibrated_population()
    c = CostModel()

    cfg = PolicyConfig(max_review_rate=0.005, max_stepup_rate=0.05)
    naive = threshold_baseline(c, p, y, amt, 0.5)["total_cost"]
    # The comparator gets the SAME friction budget the policy has to live inside.
    # Capping our own step-up while letting the thing we measure against challenge
    # everyone would flatter us, in exactly the way the naive 0.5 cutoff used to.
    t_s, t_b = tune_two_thresholds(c, p, y, amt, max_stepup_frac=cfg.max_stepup_rate)
    tuned = two_threshold_baseline(c, p, y, amt, t_s, t_b)["total_cost"]
    pol = ActionPolicy(c, cfg)
    smart = pol.report(pol.decide(p, amt), y, amt)["total_cost"]

    # A third measurement, so the saving can be attributed rather than just claimed:
    # the same policy with the analyst queue closed. It has the ladder's exact action
    # set (allow / step-up / block) and the ladder's exact friction budget, so the only
    # thing left between them is that it prices each decision against the amount.
    no_queue = ActionPolicy(c, PolicyConfig(max_review_rate=0.0, max_stepup_rate=0.05))
    without_queue = no_queue.report(no_queue.decide(p, amt), y, amt)["total_cost"]

    assert tuned < naive, "an untuned 0.5 cutoff is a straw man, and this proves it"
    assert smart < tuned, "amount-awareness has to earn its place against a tuned ladder"
    assert without_queue < tuned, (
        "with the queue closed the policy differs from the ladder ONLY by being "
        "amount-aware, if it cannot win here, amount-awareness is not what is winning")
    # The review queue is the least deployable piece of the policy (it is capped at
    # 0.5% of traffic because analysts are a real, finite resource). If most of the
    # saving turned out to come from it, the honest headline would be about staffing
    # an ops team, not about expected-cost decisions.
    assert (without_queue - smart) < (tuned - without_queue), (
        "the analyst queue is doing more work than the economics are")


def test_fraud_loss_avoided_is_a_different_number_from_cost_reduction():
    """Two true numbers, one label, was the bug. Blocking everything avoids essentially
    all fraud loss while costing far more than doing nothing, so if the two measures
    ever agree, one of them is not measuring what it says."""
    p, y, amt = _calibrated_population()
    c = CostModel()
    block_all = np.full(len(y), int(Action.BLOCK))

    avoided = fraud_loss_avoided(c, block_all, y, amt)
    pol = ActionPolicy(c, PolicyConfig(max_review_rate=0.0))
    cost = pol.realised_cost(block_all, y, amt).sum()
    nothing = allow_all_baseline(c, y, amt)["total_cost"]

    assert avoided > 0.99, "blocking everything does avoid the fraud loss"
    assert cost > nothing, "...and is still worse than doing nothing, which is the point"


def test_segment_costs_do_not_credit_real_fraud_with_attacks_we_wrote():
    p, y, amt = _calibrated_population()
    c = CostModel()
    rng = np.random.default_rng(5)
    is_adaptive = rng.random(len(y)) < 0.25
    pol = ActionPolicy(c, PolicyConfig(max_review_rate=0.005))
    actions = pol.decide(p, amt)

    segs = segment_costs(c, actions, y, amt, is_adaptive)
    fraud = y == 1
    total_exposure = c.fraud_loss(amt[fraud]).sum()

    assert (segs["real_fraud_and_legit"]["n"] + segs["adaptive_attacks_only"]["n"]
            == len(y)), "the two priced segments must partition the population"
    assert np.isclose(segs["real_fraud_only_excl_legit_friction"]["do_nothing_cost"]
                      + segs["adaptive_attacks_only"]["do_nothing_cost"],
                      total_exposure, rtol=1e-4), "exposure must split, not double-count"
