"""Chhal — a closed-loop adversarial engine for GenAI payment fraud."""
from .contract import (
    ATTACKER_CONTROLLED,
    CHANNELS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    AttackBatch,
    ScoreReport,
)

__version__ = "0.1.0"
__all__ = [
    "AttackBatch",
    "ScoreReport",
    "FEATURE_COLUMNS",
    "LABEL_COLUMN",
    "ATTACKER_CONTROLLED",
    "CHANNELS",
]
