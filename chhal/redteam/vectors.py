"""The four live-loop attack vectors — each one a campaign shape, not a row shape.

A vector declares two things and nothing else:

  `temporal`          how the attack unfolds once the account is compromised — how many
                      transactions, how far apart, how the amount moves. What came before
                      is not declared here: it is the host account's real history. The base class
                      lays out the timeline and DERIVES amount, hour, day_of_week, both
                      velocities, the inter-transaction gap and the amount-to-average
                      ratio from it, using the same function applied to the 590,540 real
                      transactions. Those seven columns are therefore internally
                      consistent by construction rather than by inspection.

  `static_features`   the few things a fraudster actually chooses — the payee, the rail,
                      the destination. Account age, merchant risk and the entity-linkage
                      counts are NOT here: they are inherited from the real account the
                      campaign is mounted on, because an attacker cannot set them.

Every number is a QUANTILE of real legitimate traffic, never a raw value: (0.35, 0.75)
on `amount` means "the middle of what real cardholders actually spend". Change the
dataset and the vectors re-scale themselves.

The bands and the campaign shapes together are what separate the vectors:
`threshold_hugging` lives inside the legitimate body and moves at a legitimate pace,
`card_testing` lives in the extreme tails and moves in seconds. That separation is what
the per-vector KS table then measures.

Text/agent "showcase" vectors (voice clone, prompt injection) live in the write-up, not
here, because they do not emit tabular features — see the strategy doc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttackVector
from .campaign import TemporalProfile


class ThresholdHugging(AttackVector):
    """HERO VECTOR. LLM-tuned sequences that sit just under velocity/amount rules and
    imitate the victim's own normal behaviour. Hardest to catch — the heart of the arms
    race. If everything else is cut, this stays.

    Everything here is deliberately INSIDE the legitimate body: bands in the middle of
    the distribution, and a cadence (an hour to two days apart) indistinguishable from
    ordinary card use. Its history is drawn from the same band as its attack, so
    `amount_to_avg_ratio` lands near 1 — the account is spending what it always spends.
    It should be the vector with the lowest KS distance from real traffic, and the
    per-vector fidelity table is where that is checked rather than asserted.
    """

    vector_id = "threshold_hugging"
    storyline = (
        "An LLM profiles the victim's normal spend and emits transactions just below "
        "every velocity and amount threshold, mimicking legitimate behaviour so the "
        "detector sees nothing anomalous."
    )
    temporal = TemporalProfile(
        txns_per_entity=(3, 9),
        inter_arrival_s=(3_600.0, 172_800.0),      # 1 hour to 2 days — a normal cadence
        amount_band=(0.35, 0.75),
        start_hour_band=(0.25, 0.85),
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": np.zeros(n, int),                 # known payee
            "is_cross_border": np.zeros(n, int),
            "channel_code": p.categorical("channel_code", n, rng),  # real channel mix
        }


class SyntheticBustout(AttackVector):
    """Age a synthetic-identity account to look clean, then max it out in a burst."""

    vector_id = "bustout"
    storyline = (
        "A GenAI synthetic identity (face + docs + backstory) passes onboarding, ages "
        "quietly for months, then busts out: a sudden burst of high-value transfers to "
        "fresh beneficiaries."
    )
    temporal = TemporalProfile(
        txns_per_entity=(8, 25),
        inter_arrival_s=(120.0, 3_600.0),          # the burst: minutes apart, over hours
        amount_band=(0.90, 0.995),
        amount_trend=1.6,                          # escalating as it empties the account
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": np.ones(n, int),
            # elevated vs the 0.7% legit / 2.2% fraud base rate, because cashing out
            # abroad is this vector's point — but not so high it leaves the manifold.
            "is_cross_border": p.bernoulli(0.10, n, rng),
            "channel_code": p.categorical("channel_code", n, rng),
        }


class CardTesting(AttackVector):
    """Intelligent BIN/card-testing that adapts probe size to velocity limits."""

    vector_id = "card_testing"
    storyline = (
        "An agent probes stolen card ranges with many micro-authorizations, spacing "
        "and sizing them to stay under velocity limits until a live card is found."
    )
    temporal = TemporalProfile(
        txns_per_entity=(20, 60),                  # many probes on one stolen range
        inter_arrival_s=(2.0, 120.0),              # seconds to two minutes apart
        amount_band=(0.005, 0.06),                 # tiny probes
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": p.bernoulli(0.08, n, rng),
            "channel_code": np.zeros(n, int),                       # card rail
        }


class UpiCollectScam(AttackVector):
    """India rail: a fraudulent UPI collect-request followed by rapid drain."""

    vector_id = "upi_collect"
    storyline = (
        "A GenAI social-engineering script tricks a victim into approving a UPI "
        "collect-request; the funds are then drained through a chain of fresh VPAs "
        "within minutes."
    )
    temporal = TemporalProfile(
        txns_per_entity=(3, 7),                    # a short chain of fresh VPAs
        inter_arrival_s=(30.0, 600.0),             # "within minutes"
        amount_band=(0.75, 0.96),
        start_hour_band=(0.30, 0.90),              # the victim has to be awake to approve
        amount_trend=0.7,                          # each hop takes less as funds run out
    )

    def static_features(self, n, rng):
        p = self.p
        return {
            "is_new_beneficiary": np.ones(n, int),
            "is_cross_border": np.zeros(n, int),
            "channel_code": np.ones(n, int),                        # upi rail
        }


ALL_VECTORS = [ThresholdHugging, SyntheticBustout, CardTesting, UpiCollectScam]
