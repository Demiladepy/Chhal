"""Red-team vector interface, and the sampler that keeps attacks on real traffic.

Each vector knows how to render a *storyline* (the GenAI narrative a judge reads) and a
*transaction pattern* (rows in the frozen feature space). The evasion optimizer then
adapts the pattern against the current detector — that adaptation is the loop.

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

from ..contract import FEATURE_COLUMNS, INTEGER_FEATURES, AttackBatch


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

    def calibrate(self, profile: BaseProfile) -> "AttackVector":
        """Bind this vector to the loaded population. Required before rendering."""
        self.profile = profile
        return self

    @property
    def p(self) -> BaseProfile:
        if self.profile is None:
            raise RuntimeError(
                f"{type(self).__name__} was not calibrated. Call "
                f"vector.calibrate(BaseProfile(base.legit_quantiles, base.legit_categoricals)) "
                f"before rendering — a vector has no absolute scale of its own."
            )
        return self.profile

    @abstractmethod
    def render(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Return n fraud rows in FEATURE_COLUMNS order (un-optimized seed attacks)."""

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
