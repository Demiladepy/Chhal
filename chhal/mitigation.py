"""Mitigation — turning a score into a decision, and a decision into money.

The brief says the defence must "detect, flag, and **mitigate**". Everything upstream
of this file only detects: `Detector.score` returns a fraud probability and stops. A
probability is not a mitigation, and neither is `score >= 0.5` — no payments system has
ever been tuned that way, because the four things you can do to a transaction have
wildly different costs and the right one depends on the amount.

    ALLOW    free if legitimate; you eat the full amount plus a chargeback fee if not
    STEP_UP  3-D Secure / OTP. Cheap, stops most fraudsters, annoys some real customers
             into abandoning the purchase
    REVIEW   an analyst looks. Accurate, but costs minutes of a person's time and delays
             the payment, and there are only so many analysts
    BLOCK    zero fraud loss; if you are wrong you have declined a real customer, which
             costs the margin on the sale AND lasting goodwill

So the policy picks, per transaction, the action with the lowest EXPECTED cost:

    E[cost | action] = P(fraud) * (what that action costs when it IS fraud)
                     + P(legit) * (what that action costs when it is NOT)

Two consequences fall straight out, and both are the point:

  * The decision becomes amount-aware for free. A $12 transaction at 40% fraud
    probability is cheaper to allow than to decline; a $3,000 one at the same
    probability is not. A single global threshold cannot express that.
  * It only works on CALIBRATED probabilities. A raw gradient-boosting score is not
    P(fraud) — and ours is trained on a pool with attacks injected, so its implied base
    rate is not the deployment base rate either. Multiplying an uncalibrated score by a
    dollar amount produces confident nonsense, so calibration is mandatory here rather
    than a refinement.

Every constant below is a named, documented parameter with an industry-typical default.
They are meant to be argued with and re-run, not believed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression



class Action(IntEnum):
    ALLOW = 0
    STEP_UP = 1
    REVIEW = 2
    BLOCK = 3


ACTION_NAMES = {a: a.name.lower() for a in Action}


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
class Calibrator:
    """Isotonic map from raw detector score to an actual probability of fraud.

    Fit on a validation slice the detector never trained on, whose class balance matches
    what the system will really see. Isotonic rather than Platt because the raw score
    distribution here is nowhere near a sigmoid and we have plenty of validation rows.
    """

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        self.iso.fit(raw_scores, y_true)
        self._fitted = True
        return self

    def __call__(self, raw_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator used before fit(); expected-cost decisions "
                               "on uncalibrated scores are meaningless.")
        return np.clip(self.iso.predict(raw_scores), 1e-6, 1 - 1e-6)


def calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Expected calibration error: mean |predicted - observed| over equal-mass bins."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    """What each outcome costs, in the currency of the base data (USD for IEEE-CIS).

    Defaults are industry-typical order-of-magnitude figures, not measured constants —
    they are here to be replaced with an issuer's real numbers, at which point every
    result below re-derives itself.
    """

    chargeback_fee: float = 25.0        # fixed admin cost of a fraud chargeback
    margin_rate: float = 0.20           # margin lost when a real sale is declined
    false_decline_penalty: float = 35.0 # goodwill/churn cost of declining a real customer
    stepup_fixed_cost: float = 2.0      # cost of issuing a 3DS/OTP challenge
    stepup_abandon_rate: float = 0.05   # real customers who give up at the challenge
    stepup_catch_rate: float = 0.90     # fraudsters who cannot complete the challenge
    review_cost: float = 4.00           # analyst time per case
    review_delay_cost: float = 1.00     # cost of holding the payment
    review_accuracy: float = 0.95       # analyst gets it right this often

    def fraud_loss(self, amount: np.ndarray) -> np.ndarray:
        return amount + self.chargeback_fee

    def decline_loss(self, amount: np.ndarray) -> np.ndarray:
        return self.margin_rate * amount + self.false_decline_penalty

    def expected_costs(self, p_fraud: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """(n, 4) expected cost of each Action for every transaction."""
        p, q = p_fraud, 1.0 - p_fraud
        loss, decline = self.fraud_loss(amount), self.decline_loss(amount)

        allow = p * loss
        block = q * decline
        step_up = (p * (1 - self.stepup_catch_rate) * loss
                   + q * (self.stepup_fixed_cost
                          + self.stepup_abandon_rate * decline))
        review = (self.review_cost + self.review_delay_cost
                  + p * (1 - self.review_accuracy) * loss
                  + q * (1 - self.review_accuracy) * decline)
        return np.column_stack([allow, step_up, review, block])


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    max_review_rate: float = 0.005   # analysts can only look at 0.5% of traffic
    max_block_rate: float | None = None   # optional hard ceiling on outright declines


class ActionPolicy:
    """Expected-cost-minimising action per transaction, under an operational capacity cap.

    The review queue is a real, finite resource: a policy that routes 8% of traffic to
    analysts is not deployable no matter how good its economics look. So reviews are
    rationed by how much reviewing actually BUYS on each transaction (the saving over the
    next-best action), and everything that does not make the cut falls back to that
    next-best action.
    """

    def __init__(self, costs: CostModel | None = None, cfg: PolicyConfig | None = None):
        self.costs = costs or CostModel()
        self.cfg = cfg or PolicyConfig()

    def decide(self, p_fraud: np.ndarray, amount: np.ndarray) -> np.ndarray:
        cost = self.costs.expected_costs(p_fraud, amount)
        actions = cost.argmin(axis=1)
        actions = self._ration(actions, cost, Action.REVIEW, self.cfg.max_review_rate)
        if self.cfg.max_block_rate is not None:
            actions = self._ration(actions, cost, Action.BLOCK, self.cfg.max_block_rate)
        return actions

    @staticmethod
    def _ration(actions: np.ndarray, cost: np.ndarray, action: int,
                max_rate: float) -> np.ndarray:
        """Keep only the `max_rate` share of `action` that buys the most, demote the rest."""
        chosen = np.flatnonzero(actions == action)
        budget = int(round(max_rate * len(actions)))
        if len(chosen) <= budget:
            return actions
        alt = cost[chosen].copy()
        alt[:, action] = np.inf                      # cost of the next-best action
        benefit = alt.min(axis=1) - cost[chosen, action]
        keep = chosen[np.argsort(benefit)[::-1][:budget]]
        demoted = np.setdiff1d(chosen, keep, assume_unique=False)
        actions = actions.copy()
        actions[demoted] = alt[np.searchsorted(chosen, demoted)].argmin(axis=1)
        return actions

    # -- reporting -----------------------------------------------------------
    def realised_cost(self, actions: np.ndarray, y_true: np.ndarray,
                      amount: np.ndarray) -> np.ndarray:
        """Cost actually incurred per transaction, given the TRUE label.

        This is the honest scorecard: expected cost drove the decision, this is what the
        decision then cost in a world where we know who was really a fraudster.
        """
        c = self.costs
        loss, decline = c.fraud_loss(amount), c.decline_loss(amount)
        fraud = y_true == 1
        out = np.zeros(len(actions), float)

        a = actions == Action.ALLOW
        out[a & fraud] = loss[a & fraud]

        b = actions == Action.BLOCK
        out[b & ~fraud] = decline[b & ~fraud]

        s = actions == Action.STEP_UP
        out[s & fraud] = (1 - c.stepup_catch_rate) * loss[s & fraud]
        out[s & ~fraud] = c.stepup_fixed_cost + c.stepup_abandon_rate * decline[s & ~fraud]

        r = actions == Action.REVIEW
        base = c.review_cost + c.review_delay_cost
        out[r & fraud] = base + (1 - c.review_accuracy) * loss[r & fraud]
        out[r & ~fraud] = base + (1 - c.review_accuracy) * decline[r & ~fraud]
        return out

    def report(self, actions: np.ndarray, y_true: np.ndarray,
               amount: np.ndarray) -> Dict:
        realised = self.realised_cost(actions, y_true, amount)
        fraud = y_true == 1
        stopped = np.isin(actions, [Action.BLOCK, Action.REVIEW]) | (actions == Action.STEP_UP)
        return {
            "n": int(len(actions)),
            "action_mix": {ACTION_NAMES[a]: round(float((actions == a).mean()), 5)
                           for a in Action},
            "review_rate": round(float((actions == Action.REVIEW).mean()), 5),
            "block_rate_on_legit": round(float((actions[~fraud] == Action.BLOCK).mean()), 5),
            "fraud_touched_rate": round(float(stopped[fraud].mean()), 5),
            "fraud_blocked_rate": round(float((actions[fraud] == Action.BLOCK).mean()), 5),
            "total_cost": round(float(realised.sum()), 2),
            "cost_per_1k_txns": round(float(realised.sum() / len(actions) * 1000), 2),
        }


def allow_all_baseline(costs: CostModel, y_true: np.ndarray, amount: np.ndarray) -> Dict:
    """Do nothing at all — the number every other policy has to beat."""
    realised = np.where(y_true == 1, costs.fraud_loss(amount), 0.0)
    return {"n": int(len(y_true)), "total_cost": round(float(realised.sum()), 2),
            "cost_per_1k_txns": round(float(realised.sum() / len(y_true) * 1000), 2)}


def threshold_baseline(costs: CostModel, p_fraud: np.ndarray, y_true: np.ndarray,
                       amount: np.ndarray, threshold: float = 0.5) -> Dict:
    """Block above a fixed threshold, allow below — the naive policy, priced honestly."""
    actions = np.where(p_fraud >= threshold, int(Action.BLOCK), int(Action.ALLOW))
    pol = ActionPolicy(costs, PolicyConfig(max_review_rate=0.0))
    rep = pol.report(actions, y_true, amount)
    rep["threshold"] = threshold
    return rep


def _cost_if_all(costs: CostModel, action: int, y_true: np.ndarray,
                 amount: np.ndarray) -> np.ndarray:
    """Realised cost of taking `action` on every row, given the true labels."""
    pol = ActionPolicy(costs, PolicyConfig(max_review_rate=0.0))
    return pol.realised_cost(np.full(len(y_true), int(action)), y_true, amount)


def tune_two_thresholds(costs: CostModel, p_fraud: np.ndarray, y_true: np.ndarray,
                        amount: np.ndarray) -> tuple[float, float]:
    """The best amount-BLIND policy that exists, found exactly rather than guessed.

    `block at score >= 0.5` is a straw man: nobody deploys an untuned threshold, so
    beating it proves nothing about the expected-cost machinery. The honest comparator is
    the strongest thing a fraud team builds without any of our economics — allow below one
    threshold, challenge between, block above — with both thresholds tuned on the same
    cost model we use ourselves.

    Sorting by score turns the search into prefix sums, so we return the GLOBAL optimum
    over all threshold pairs rather than the best of a sampled grid. Tune on a slice, then
    price on another: a comparator tuned on its own evaluation set would be unbeatable and
    meaningless.
    """
    order = np.argsort(p_fraud, kind="stable")
    ps, ys, amts = p_fraud[order], y_true[order], amount[order]
    n = len(ps)

    zero = np.zeros(1)
    A = np.concatenate([zero, np.cumsum(_cost_if_all(costs, Action.ALLOW, ys, amts))])
    S = np.concatenate([zero, np.cumsum(_cost_if_all(costs, Action.STEP_UP, ys, amts))])
    B = np.concatenate([zero, np.cumsum(_cost_if_all(costs, Action.BLOCK, ys, amts))])

    # rows [0,i) allow, [i,j) step_up, [j,n) block
    #   cost(i, j) = A[i] + (S[j] - S[i]) + (B[n] - B[j])
    #              = (A - S)[i] + (S - B)[j] + B[n]
    d = A - S
    j = int(np.argmin(np.minimum.accumulate(d) + (S - B)))
    i = int(np.argmin(d[: j + 1]))

    hi = float(ps[-1]) + 1.0
    return (hi if i >= n else float(ps[i])), (hi if j >= n else float(ps[j]))


def two_threshold_baseline(costs: CostModel, p_fraud: np.ndarray, y_true: np.ndarray,
                           amount: np.ndarray, t_stepup: float,
                           t_block: float) -> Dict:
    """Price a tuned allow / step-up / block ladder — amount-blind, no review queue."""
    actions = np.where(p_fraud >= t_block, int(Action.BLOCK),
                       np.where(p_fraud >= t_stepup, int(Action.STEP_UP),
                                int(Action.ALLOW)))
    rep = ActionPolicy(costs, PolicyConfig(max_review_rate=0.0)).report(actions, y_true, amount)
    rep["t_stepup"], rep["t_block"] = round(t_stepup, 6), round(t_block, 6)
    return rep


def fraud_loss_avoided(costs: CostModel, actions: np.ndarray, y_true: np.ndarray,
                       amount: np.ndarray) -> float:
    """Share of actual FRAUD LOSS avoided — which is not the same as cost saved.

    Total cost reduction nets the friction we impose on legitimate customers against the
    fraud we stop, so it is the number a CFO wants. Fraud loss avoided ignores that
    friction and answers only "how much of the money that would have walked out did we
    keep?". Both are true; labelling one as the other is not.
    """
    fraud = y_true == 1
    if not fraud.any():
        return 0.0
    exposed = costs.fraud_loss(amount[fraud]).sum()
    pol = ActionPolicy(costs, PolicyConfig(max_review_rate=0.0))
    incurred = pol.realised_cost(actions, y_true, amount)[fraud].sum()
    return float(1.0 - incurred / exposed)


def segment_costs(costs: CostModel, actions: np.ndarray, y_true: np.ndarray,
                  amount: np.ndarray, is_adaptive: np.ndarray) -> Dict:
    """Price the policy separately on real fraud and on our own attacks.

    Detection is already reported per segment; economics was not, and that hid something:
    a quarter of the cost denominator is fraud WE generated, and the policy is far better
    at it than at the real thing. One blended percentage flatters us on real fraud by
    borrowing credit from attacks we wrote ourselves.
    """
    pol = ActionPolicy(costs, PolicyConfig(max_review_rate=0.0))
    realised = pol.realised_cost(actions, y_true, amount)
    exposure = np.where(y_true == 1, costs.fraud_loss(amount), 0.0)
    fraud = y_true == 1

    out: Dict = {}
    for name, keep in (("real_fraud_and_legit", ~(fraud & is_adaptive)),
                       ("adaptive_attacks_only", fraud & is_adaptive),
                       ("real_fraud_only_excl_legit_friction", fraud & ~is_adaptive)):
        do_nothing = float(exposure[keep].sum())
        spent = float(realised[keep].sum())
        out[name] = {
            "n": int(keep.sum()),
            "do_nothing_cost": round(do_nothing, 2),
            "policy_cost": round(spent, 2),
            "net_cost_reduction_pct": round(100 * (do_nothing - spent) / do_nothing, 2)
            if do_nothing else 0.0,
        }
    return out
