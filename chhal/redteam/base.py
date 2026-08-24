"""Red-team vector interface, and the sampler that keeps attacks on real traffic.

Each vector knows how to render a *storyline* (the GenAI narrative a judge reads) and a
*transaction pattern* (rows in the frozen feature space). The evasion optimizer then
adapts the pattern against the current detector — that adaptation is the loop.

Why a vector describes a campaign rather than a pile of rows
------------------------------------------------------------
`velocity_1h`, `velocity_24h`, `time_since_last_txn_min` and `amount_to_avg_ratio` are
four views of one timeline, so sampling them independently produces transactions that
cannot exist. It did: 100% of the hero vector's rows claimed activity in the last 24
hours while also claiming the previous transaction was days ago, against 0% of real
traffic. A judge who checks that once stops believing the rest.

So a vector now declares a `TemporalProfile` — accounts, transactions each, gaps, amount
trajectory — and the base class lays out a real timeline, then DERIVES the behavioural
features from it with `chhal.behaviour.derive`, the same function used on the 590,540
real transactions. `hour` and `day_of_week` come from the timestamps too. Consistency is
not checked afterwards; it cannot be violated.

Each campaign is mounted on a REAL, never-fraudulent account (see hosts.py), so the
history `amount_to_avg_ratio` is measured against is that card's actual spend, and the
issuer-side context the attacker cannot control — account age, merchant risk, and the
dataset's anonymised entity-linkage counts — is inherited rather than invented. That is
what let the feature space grow to include the linkage block at all: those counts carry
most of the real-fraud signal and cannot be reconstructed, so a generated attacker could
only have faked them.

Why vectors no longer contain hand-picked numbers
-------------------------------------------------
A vector used to say `amount ~ lognormal(6.6, 0.5)`. Against the real IEEE-CIS
population that is a median of ~735 where real traffic sits at 68.50 — every row would
land outside the plausibility manifold and be clipped flat onto the boundary, which
destroys both the vector's meaning and any fidelity claim made about it.

So a vector now describes itself in the only terms that survive a change of dataset:
*where in the legitimate population it sits*. `profile.band("amount", 0.35, 0.75)`
means "amounts typical of the middle of real traffic". Values come back through the
inverse CDF of real legitimate transactions, so every attack row is built out of values
that genuinely occur — before the optimizer has adapted anything. Point the loader at a
different dataset and the vectors re-scale themselves.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..behaviour import day_of_week_of, derive, hour_of
from ..contract import (FEATURE_COLUMNS, INHERITED_FEATURES, INTEGER_FEATURES,
                        AttackBatch)
from .campaign import TemporalProfile, generate


class BaseProfile:
    """Inverse-CDF sampler over the legitimate half of the loaded base population."""

    def __init__(self, legit_quantiles: pd.DataFrame,
                 legit_categoricals: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None):
        self._levels = legit_quantiles.index.to_numpy(dtype=float)
        self._curves = {c: legit_quantiles[c].to_numpy(dtype=float)
                        for c in legit_quantiles.columns}
        self._cats = legit_categoricals or {}

    def band(self, feature: str, lo: float, hi: float, n: int,
             rng: np.random.Generator) -> np.ndarray:
        """n values drawn from real legit traffic, restricted to the [lo, hi] quantile band.

        lo/hi are quantile levels in [0, 1], not raw values — that is what makes a
        vector portable across datasets.
        """
        u = rng.uniform(lo, hi, n)
        return np.interp(u, self._levels, self._curves[feature])

    def categorical(self, feature: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a discrete feature from its real frequency in legitimate traffic."""
        values, probs = self._cats[feature]
        return rng.choice(values, size=n, p=probs)

    @staticmethod
    def bernoulli(p: float, n: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.random(n) < p).astype(int)


class AttackVector(ABC):
    vector_id: str = "abstract"
    storyline: str = ""

    def __init__(self) -> None:
        self.profile: BaseProfile | None = None
        self.hosts = None

    def calibrate(self, profile: BaseProfile, hosts=None) -> "AttackVector":
        """Bind this vector to the loaded population and the accounts it may compromise."""
        self.profile = profile
        self.hosts = hosts
        return self

    @property
    def h(self):
        if getattr(self, "hosts", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} has no host pool. Call "
                f"vector.calibrate(profile, HostPool(frame)) — a campaign is mounted on a "
                f"real account so that issuer-side features are inherited, not invented."
            )
        return self.hosts

    @property
    def p(self) -> BaseProfile:
        if self.profile is None:
            raise RuntimeError(
                f"{type(self).__name__} was not calibrated. Call "
                f"vector.calibrate(BaseProfile(base.legit_quantiles, base.legit_categoricals)) "
                f"before rendering — a vector has no absolute scale of its own."
            )
        return self.profile

    # Choices that belong to the compromised ACCOUNT, not to a single transaction. An
    # attacker picks a rail and a destination once; they do not change partway through a
    # campaign. Sampled once per entity and broadcast.
    ENTITY_LEVEL = ("is_cross_border", "channel_code")

    temporal: TemporalProfile = None  # each vector must declare its own

    @abstractmethod
    def static_features(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """The few columns the attacker actually chooses, per row.

        Not the timeline (amount, hour, day_of_week, both velocities, the gap, the amount
        ratio) — the campaign produces those. Not the issuer's view either (account age,
        merchant risk, linkage counts) — the host account supplies those. What is left is
        what a fraudster genuinely decides: the payee, the rail, the destination.
        """

    def render(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """n fraud rows: a campaign mounted on a real account, derived from its timeline."""
        if self.temporal is None:
            raise RuntimeError(f"{type(self).__name__} declares no TemporalProfile")
        camp = generate(self.temporal, n, self.p, rng, self.h)
        beh = derive(camp.entity, camp.timestamp_s, camp.amount)

        df = pd.DataFrame({
            "amount": camp.amount,
            "hour": hour_of(camp.timestamp_s),
            "day_of_week": day_of_week_of(camp.timestamp_s),
            **{c: beh[c].to_numpy() for c in beh.columns},
            **self.static_features(len(camp.amount), rng),
        })
        # issuer-side context, inherited from the host account rather than invented
        for j, col in enumerate(INHERITED_FEATURES):
            df[col] = camp.inherited[:, j]

        for col in self.ENTITY_LEVEL:                    # one choice per account
            df[col] = df.groupby(camp.entity)[col].transform("first")

        # The card keeps ageing while the attacker holds it: the host's age was measured
        # at its last real transaction, and the takeover happens some days after that.
        first_ts = pd.Series(camp.timestamp_s).groupby(camp.entity).transform("min")
        df["account_age_days"] = df["account_age_days"] + (camp.timestamp_s - first_ts) / 86_400.0

        # the host's real transactions exist only to give the account a baseline
        return df[camp.is_attack].head(n).reset_index(drop=True)

    def batch(self, n: int, iteration: int, rng: np.random.Generator) -> AttackBatch:
        df = self.render(n, rng)[FEATURE_COLUMNS].reset_index(drop=True)
        for col in INTEGER_FEATURES:          # one place, so no vector can forget
            df[col] = df[col].round().astype(int)
        return AttackBatch(
            vector_id=self.vector_id,
            iteration=iteration,
            transactions=df,
            provenance={"storyline": self.storyline},
        ).validate()
