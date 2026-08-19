"""Red-team vector interface.

Each vector knows how to render a *storyline* (the GenAI narrative that a judge reads)
and a *transaction pattern* (rows in the frozen feature space). The evasion optimizer
then adapts the pattern against the current detector — that adaptation is the loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..contract import FEATURE_COLUMNS, AttackBatch


class AttackVector(ABC):
    vector_id: str = "abstract"
    storyline: str = ""

    @abstractmethod
    def render(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Return n fraud rows in FEATURE_COLUMNS order (un-optimized seed attacks)."""

    def batch(self, n: int, iteration: int, rng: np.random.Generator) -> AttackBatch:
        df = self.render(n, rng)[FEATURE_COLUMNS].reset_index(drop=True)
        return AttackBatch(
            vector_id=self.vector_id,
            iteration=iteration,
            transactions=df,
            provenance={"storyline": self.storyline},
        ).validate()
