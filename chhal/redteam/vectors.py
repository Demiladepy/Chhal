"""The four live-loop attack vectors — each emits transaction features.

These flow through the tabular detector and constitute the closed loop. Text/agent
"showcase" vectors (voice clone, prompt injection) live in the write-up, not here,
because they do not emit tabular features — see the strategy doc.

Every number below is a QUANTILE of real legitimate traffic, never a raw value. A band
of (0.35, 0.75) on `amount` means "the middle of what real cardholders actually spend";
(0.99, 0.9995) on `velocity_1h` means "as fast as the busiest real accounts, and no
faster". Values are drawn back through the real inverse CDF (see base.BaseProfile), so
a seed attack is assembled out of values that genuinely occur in the population before
the evasion optimizer adapts it at all. Change the dataset and the vectors re-scale.

The bands are what separate the vectors from each other: `threshold_hugging` lives
inside the legitimate body, `card_testing` lives in the extreme tails, and the other two
sit in between. That separation is what the per-vector KS table then measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttackVector


class ThresholdHugging(AttackVector):
    """HERO VECTOR. LLM-tuned sequences that sit just under velocity/amount rules and
    imitate the victim's own normal behaviour. Hardest to catch — the heart of the
    arms race. If everything else is cut, this stays.

    Every band here is deliberately INSIDE the legitimate body, never in a tail. That
    is the whole claim: this vector should be the one with the lowest KS distance from
    real traffic, and the per-vector fidelity table is where that gets checked rather
    than asserted.
    """

    vector_id = "threshold_hugging"
    storyline = (
        "An LLM profiles the victim's normal spend and emits transactions just below "
        "every velocity and amount threshold, mimicking legitimate behaviour so the "
        "detector sees nothing anomalous."
    )

    def render(self, n, rng):
        p = self.p
        return pd.DataFrame({
            "amount": p.band("amount", 0.35, 0.75, n, rng),                 # ordinary size
            "hour": p.band("hour", 0.25, 0.85, n, rng),                     # active hours
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": p.band("velocity_1h", 0.80, 0.95, n, rng),       # under the cap
            "velocity_24h": p.band("velocity_24h", 0.80, 0.95, n, rng),     # looks normal
            "amount_to_avg_ratio": p.band("amount_to_avg_ratio", 0.40, 0.70, n, rng),
            "account_age_days": p.band("account_age_days", 0.55, 0.95, n, rng),
            "time_since_last_txn_min": p.band("time_since_last_txn_min", 0.35, 0.75, n, rng),
            "is_new_beneficiary": np.zeros(n, int),                         # known payee
            "is_cross_border": np.zeros(n, int),
            "channel_code": p.categorical("channel_code", n, rng),          # real channel mix
            "merchant_risk": p.band("merchant_risk", 0.10, 0.50, n, rng),   # low, issuer-side
        })


class SyntheticBustout(AttackVector):
    """Age a synthetic-identity account to look clean, then max it out in a burst."""

    vector_id = "bustout"
    storyline = (
        "A GenAI synthetic identity (face + docs + backstory) passes onboarding, ages "
        "quietly for months, then busts out: a sudden burst of high-value transfers to "
        "fresh beneficiaries."
    )

    def render(self, n, rng):
        p = self.p
        return pd.DataFrame({
            "amount": p.band("amount", 0.93, 0.995, n, rng),                # large, still real
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": p.band("velocity_1h", 0.95, 0.995, n, rng),      # burst
            "velocity_24h": p.band("velocity_24h", 0.96, 0.999, n, rng),
            "amount_to_avg_ratio": p.band("amount_to_avg_ratio", 0.95, 0.999, n, rng),
            # "aged quietly for months" has to mean months in THIS population: the
            # 0.75-0.93 band of real account age is 118-428 days. The naive middle of
            # the distribution would have been 0-3 days, because real card populations
            # are dominated by recently-first-seen cards.
            "account_age_days": p.band("account_age_days", 0.75, 0.93, n, rng),
            "time_since_last_txn_min": p.band("time_since_last_txn_min", 0.02, 0.20, n, rng),
            "is_new_beneficiary": np.ones(n, int),
            # elevated vs the 0.7% legit / 2.2% fraud base rate, because cashing out
            # abroad is this vector's point — but not so high it leaves the manifold.
            "is_cross_border": p.bernoulli(0.10, n, rng),
            "channel_code": p.categorical("channel_code", n, rng),
            "merchant_risk": p.band("merchant_risk", 0.70, 0.97, n, rng),
        })


class CardTesting(AttackVector):
    """Intelligent BIN/card-testing that adapts probe size to velocity limits."""

    vector_id = "card_testing"
    storyline = (
        "An agent probes stolen card ranges with many micro-authorizations, spacing "
        "and sizing them to stay under velocity limits until a live card is found."
    )

    def render(self, n, rng):
        p = self.p
        return pd.DataFrame({
            "amount": p.band("amount", 0.005, 0.06, n, rng),                # tiny probes
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": p.band("velocity_1h", 0.98, 0.9995, n, rng),     # rapid
            "velocity_24h": p.band("velocity_24h", 0.99, 0.9995, n, rng),
            "amount_to_avg_ratio": p.band("amount_to_avg_ratio", 0.005, 0.10, n, rng),
            "account_age_days": p.band("account_age_days", 0.0, 0.25, n, rng),
            "time_since_last_txn_min": p.band("time_since_last_txn_min", 0.0, 0.05, n, rng),
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": p.bernoulli(0.08, n, rng),
            "channel_code": np.zeros(n, int),                               # card rail
            "merchant_risk": p.band("merchant_risk", 0.50, 0.90, n, rng),
        })


class UpiCollectScam(AttackVector):
    """India rail: a fraudulent UPI collect-request followed by rapid drain."""

    vector_id = "upi_collect"
    storyline = (
        "A GenAI social-engineering script tricks a victim into approving a UPI "
        "collect-request; the funds are then drained through a chain of fresh VPAs "
        "within minutes."
    )

    def render(self, n, rng):
        p = self.p
        return pd.DataFrame({
            "amount": p.band("amount", 0.75, 0.96, n, rng),
            "hour": p.band("hour", 0.30, 0.90, n, rng),                     # victim awake
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": p.band("velocity_1h", 0.85, 0.97, n, rng),
            "velocity_24h": p.band("velocity_24h", 0.85, 0.97, n, rng),
            "amount_to_avg_ratio": p.band("amount_to_avg_ratio", 0.85, 0.98, n, rng),
            "account_age_days": p.band("account_age_days", 0.20, 0.80, n, rng),
            "time_since_last_txn_min": p.band("time_since_last_txn_min", 0.01, 0.15, n, rng),
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": np.zeros(n, int),
            "channel_code": np.ones(n, int),                                # upi rail
            "merchant_risk": p.band("merchant_risk", 0.40, 0.80, n, rng),
        })


ALL_VECTORS = [ThresholdHugging, SyntheticBustout, CardTesting, UpiCollectScam]
