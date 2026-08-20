"""The four live-loop attack vectors — each emits transaction features.

These flow through the tabular detector and constitute the closed loop. Text/agent
"showcase" vectors (voice clone, prompt injection) live in the write-up, not here,
because they do not emit tabular features — see the strategy doc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttackVector


class ThresholdHugging(AttackVector):
    """HERO VECTOR. LLM-tuned sequences that sit just under velocity/amount rules and
    imitate the victim's own normal behaviour. Hardest to catch — the heart of the
    arms race. If everything else is cut, this stays."""

    vector_id = "threshold_hugging"
    storyline = (
        "An LLM profiles the victim's normal spend and emits transactions just below "
        "every velocity and amount threshold, mimicking legitimate behaviour so the "
        "detector sees nothing anomalous."
    )

    def render(self, n, rng):
        return pd.DataFrame({
            "amount": np.round(rng.lognormal(6.6, 0.5, n), 2),         # ordinary size
            "hour": rng.integers(8, 22, n),                            # normal hours
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": rng.integers(0, 2, n),                      # under the cap
            "velocity_24h": rng.integers(3, 8, n),                     # looks normal
            "amount_to_avg_ratio": np.clip(rng.normal(1.05, 0.15, n), 0.5, 1.8),
            "account_age_days": rng.gamma(3.0, 260.0, n).clip(200, 4000),
            "time_since_last_txn_min": rng.exponential(300.0, n).clip(20, 5000),
            "is_new_beneficiary": np.zeros(n, int),                    # known payee
            "is_cross_border": np.zeros(n, int),
            "channel_code": rng.choice([0, 1], n, p=[0.5, 0.5]),
            "merchant_risk": np.clip(rng.beta(2.0, 8.0, n), 0, 1),     # low, issuer-side
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
        return pd.DataFrame({
            "amount": np.round(rng.lognormal(8.6, 0.6, n), 2),        # large
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": rng.integers(2, 6, n),                     # burst
            "velocity_24h": rng.integers(15, 40, n),
            "amount_to_avg_ratio": np.clip(rng.normal(6.0, 2.0, n), 2.0, 30.0),
            "account_age_days": rng.uniform(120, 400, n),             # aged, then pops
            "time_since_last_txn_min": rng.exponential(15.0, n).clip(0.5, 200),
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": (rng.random(n) < 0.3).astype(int),
            "channel_code": rng.choice([0, 2], n, p=[0.6, 0.4]),
            "merchant_risk": np.clip(rng.beta(3.0, 5.0, n), 0, 1),
        })


class CardTesting(AttackVector):
    """Intelligent BIN/card-testing that adapts probe size to velocity limits."""

    vector_id = "card_testing"
    storyline = (
        "An agent probes stolen card ranges with many micro-authorizations, spacing "
        "and sizing them to stay under velocity limits until a live card is found."
    )

    def render(self, n, rng):
        return pd.DataFrame({
            "amount": np.round(rng.uniform(1.0, 40.0, n), 2),         # tiny probes
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": rng.integers(5, 20, n),                    # rapid
            "velocity_24h": rng.integers(40, 120, n),
            "amount_to_avg_ratio": np.clip(rng.normal(0.2, 0.1, n), 0.02, 1.0),
            "account_age_days": rng.uniform(1, 90, n),
            "time_since_last_txn_min": rng.exponential(2.0, n).clip(0.1, 60),
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": (rng.random(n) < 0.5).astype(int),
            "channel_code": np.zeros(n, int),                         # card
            "merchant_risk": np.clip(rng.beta(3.0, 4.0, n), 0, 1),
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
        return pd.DataFrame({
            "amount": np.round(rng.lognormal(7.4, 0.7, n), 2),
            "hour": rng.integers(9, 23, n),
            "day_of_week": rng.integers(0, 7, n),
            "velocity_1h": rng.integers(1, 4, n),
            "velocity_24h": rng.integers(2, 10, n),
            "amount_to_avg_ratio": np.clip(rng.normal(2.5, 1.0, n), 0.5, 12.0),
            "account_age_days": rng.uniform(30, 900, n),
            "time_since_last_txn_min": rng.exponential(8.0, n).clip(0.5, 120),
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": np.zeros(n, int),
            "channel_code": np.ones(n, int),                          # upi
            "merchant_risk": np.clip(rng.beta(2.5, 6.0, n), 0, 1),
        })


ALL_VECTORS = [ThresholdHugging, SyntheticBustout, CardTesting, UpiCollectScam]
