"""Tests for the operating-point metrics. The numbers we actually quote."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import OPERATING_POINTS, PRIMARY_FPR, ScoreReport  # noqa: E402
from chhal.evaluation import recall_at_fpr, threshold_for_fpr          # noqa: E402


def test_threshold_for_fpr_flags_the_requested_share_of_legit():
    """On continuous scores the threshold lands within a hair of the budget.

    The tolerance is one part in ten thousand of the budget itself, not the flat
    +/-5e-4 this used to allow. That was a +/-50% band around a 0.1% budget and would
    have passed a threshold flagging 0.0006 or 0.0015 without complaint.
    """
    rng = np.random.default_rng(0)
    legit = rng.random(100_000)
    for fpr in OPERATING_POINTS:
        thr = threshold_for_fpr(legit, fpr)
        assert (legit >= thr).mean() == pytest.approx(fpr, rel=0.02)


def test_the_budget_is_never_exceeded_even_when_scores_are_one_big_tie():
    """The bug this guards: a quantile can land inside a block of identical scores, and
    `>=` then sweeps the entire block in. Measured on the real detector, that reported
    recall "at a 0.1% budget" while flagging 43.9% of legitimate traffic. A tree
    ensemble produces exactly this shape, so the test uses it: 40% of legit sitting on
    one value, right where a 0.1% quantile would like to sit."""
    legit = np.concatenate([np.full(40_000, 0.5), np.linspace(0.0, 0.49, 59_000),
                            np.linspace(0.51, 1.0, 1_000)])
    for fpr in OPERATING_POINTS:
        thr = threshold_for_fpr(legit, fpr)
        realised = (legit >= thr).mean()
        assert realised <= fpr + 1e-12, (
            f"budget {fpr} quoted, {realised:.4f} realised")


def test_a_budget_too_tight_for_any_threshold_returns_no_recall_not_a_bargain():
    """If even the single highest block of scores busts the budget, the honest answer
    is that nothing can be flagged, not a threshold that flags the block anyway."""
    legit = np.full(1_000, 0.7)
    attacks = np.full(100, 0.7)
    rec, thr, realised = recall_at_fpr(legit, attacks, 0.001)
    assert rec == 0.0 and realised == 0.0 and thr > 0.7


def test_recall_is_monotone_in_the_false_positive_budget():
    """Relaxing the budget can never catch less fraud."""
    rng = np.random.default_rng(1)
    legit = rng.beta(1.5, 8, 50_000)
    attacks = rng.beta(5, 3, 3_000)
    recalls = [recall_at_fpr(legit, attacks, f)[0] for f in OPERATING_POINTS]
    assert recalls == sorted(recalls)
    assert 0.0 <= recalls[0] <= recalls[-1] <= 1.0


def test_a_useless_detector_scores_near_the_budget_it_is_given():
    """Random scores catch fraud at exactly the rate they flag legit traffic."""
    rng = np.random.default_rng(2)
    legit, attacks = rng.random(200_000), rng.random(20_000)
    got, _, realised = recall_at_fpr(legit, attacks, 0.01)
    assert realised <= 0.01 + 1e-12
    assert got == pytest.approx(0.01, abs=0.004)


def test_pr_auc_is_the_honest_summary_under_imbalance():
    """The reason PR AUC leads and ROC AUC is demoted: at 3.5% prevalence a mediocre
    detector still posts a spectacular ROC AUC, because the true-negative pile it
    divides by is enormous. PR AUC does not flatter it."""
    rng = np.random.default_rng(3)
    n = 200_000
    y = (rng.random(n) < 0.035).astype(int)
    score = np.clip(rng.normal(0.2, 0.15, n) + 0.35 * y, 0, 1)
    roc, pr = roc_auc_score(y, score), average_precision_score(y, score)
    assert roc > 0.90, "ROC AUC looks strong"
    assert pr < roc - 0.3, "PR AUC tells the less flattering, more useful truth"


def test_score_report_row_leads_with_operating_points():
    rep = ScoreReport(iteration=1, split="heldout_novel", precision=0.5, recall=0.5,
                      f1=0.5, auc=0.99, fp_rate_on_legit=0.02, pr_auc=0.61,
                      recall_at_fpr={f: 0.5 for f in OPERATING_POINTS},
                      realised_fpr_at_fpr={f: f for f in OPERATING_POINTS},
                      alert_rate=0.013)
    row = rep.as_row()
    for fpr in OPERATING_POINTS:
        assert f"recall_at_fpr_{fpr}" in row
        assert row[f"realised_fpr_{fpr}"] <= fpr + 1e-12
    assert row["pr_auc"] == 0.61 and row["alert_rate"] == 0.013
    assert PRIMARY_FPR in OPERATING_POINTS


def test_empty_attack_set_does_not_crash():
    legit = np.random.default_rng(4).random(1000)
    rec, thr, realised = recall_at_fpr(legit, np.array([]), 0.001)
    assert rec == 0.0 and np.isfinite(thr) and realised <= 0.001 + 1e-12
