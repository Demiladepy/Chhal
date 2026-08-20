"""The loop interface contract — the single frozen artifact both sides build against.

Everything else in Chhal can change. These two structs and FEATURE_COLUMNS may
not, without a deliberate, agreed schema bump. This is what lets the red side and the
blue side develop in parallel without integration hell.

The one rule that keeps the loop honest:
    An AttackBatch may ONLY contain rows in FEATURE_COLUMNS — the same feature space
    the detector sees for legitimate traffic. No attack may invent a column the
    detector cannot observe, otherwise "detection" is trivial and the result is fake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# The frozen feature space. Attacks live here or they do not exist.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "amount",                 # transaction amount
    "hour",                   # 0-23, local hour of transaction
    "day_of_week",            # 0-6
    "velocity_1h",            # count of this account's txns in the last hour
    "velocity_24h",           # count in the last 24h
    "amount_to_avg_ratio",    # amount / account's rolling average amount
    "account_age_days",       # age of the originating account
    "time_since_last_txn_min",
    "is_new_beneficiary",     # 0/1 first-time payee
    "is_cross_border",        # 0/1
    "channel_code",           # 0 = card, 1 = upi, 2 = imps/rtp
    "merchant_risk",          # 0..1 issuer-side merchant risk score
]

LABEL_COLUMN = "is_fraud"

# Features an attacker actually controls (used by the evasion optimizer). Issuer-side
# signals such as merchant_risk are deliberately excluded — a fraudster cannot set them.
ATTACKER_CONTROLLED: List[str] = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "amount_to_avg_ratio",
    "time_since_last_txn_min",
    "is_new_beneficiary",
    "channel_code",
    "is_cross_border",
]

CHANNELS = {"card": 0, "upi": 1, "imps": 2}


@dataclass
class AttackBatch:
    """What the red team emits. All rows are fraud (label == 1)."""

    vector_id: str                       # e.g. "threshold_hugging"
    iteration: int                       # which loop pass produced it
    transactions: pd.DataFrame           # columns == FEATURE_COLUMNS, exactly
    provenance: Dict = field(default_factory=dict)  # seed, optimizer params, storyline

    def validate(self) -> "AttackBatch":
        cols = list(self.transactions.columns)
        if cols != FEATURE_COLUMNS:
            missing = set(FEATURE_COLUMNS) - set(cols)
            extra = set(cols) - set(FEATURE_COLUMNS)
            raise ValueError(
                f"AttackBatch[{self.vector_id}] violates the feature-space rule. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        if self.transactions.isna().any().any():
            raise ValueError(f"AttackBatch[{self.vector_id}] contains NaNs")
        return self

    def __len__(self) -> int:
        return len(self.transactions)


@dataclass
class ScoreReport:
    """What the detector returns after scoring a set of transactions."""

    iteration: int
    split: str                           # "train" | "heldout_known" | "heldout_novel"
    precision: float
    recall: float
    f1: float
    auc: float
    fp_rate_on_legit: float
    per_vector_recall: Dict[str, float] = field(default_factory=dict)
    shap_top_features: List[str] = field(default_factory=list)

    def as_row(self) -> Dict:
        return {
            "iteration": self.iteration,
            "split": self.split,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "fp_rate_on_legit": self.fp_rate_on_legit,
        }
