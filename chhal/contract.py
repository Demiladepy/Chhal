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
] + [f"linkage_c{i}" for i in range(1, 15)]

# IEEE-CIS's C1-C14: anonymised entity-linkage counts — how many addresses, devices,
# emails and cards are associated with this card, over undisclosed windows. We do not
# name them beyond what we can defend, because nobody outside Vesta knows exactly what
# each one counts.
#
# They matter enormously and cannot be faked. On real fraud they take recall at a 0.1%
# false-positive budget from 3.1% to 19.7% — a 6.4x lift that nothing else comes close
# to, including all 339 V-features (which add under 2 points on top). We also tried to
# rebuild the same signal from what we DO understand — distinct counterparties, addresses
# and emails per account over time — and got +0.16 points, essentially nothing. Whatever
# they aggregate over (devices, phones, IPs, cross-card linkage) is not in the columns
# this dataset exposes. See scripts/feature_ablation.py.
#
# So the red team does not invent them. It inherits them, by mounting each attack
# campaign on a REAL account whose linkage history is whatever it actually was.
LINKAGE_FEATURES: List[str] = [f"linkage_c{i}" for i in range(1, 15)]

# Features that belong to the issuer, not the transaction. An attacker who compromises a
# card cannot set how many devices that card is associated with, nor its age, nor the
# issuer's opinion of the merchant. The red team inherits every one of these from the
# host account rather than sampling them.
INHERITED_FEATURES: List[str] = [
    "account_age_days",
    "merchant_risk",
] + LINKAGE_FEATURES

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

# Of the nine above, only these FOUR are set directly, one value at a time, by the
# person committing the fraud. The other five are not settable at all: they are what a
# timeline LOOKS like once it has happened. You cannot choose to have transacted four
# times in the last hour; you can only transact four times, and then that is what the
# counter says. The evasion optimizer therefore searches over these four plus the
# timeline itself, and re-derives the rest — see optimizer.py.
ATTACKER_DIRECT: List[str] = [
    "amount",
    "is_new_beneficiary",
    "channel_code",
    "is_cross_border",
]

# Chosen once per compromised account, not per transaction: an attacker picks a rail and
# a destination when they take the card over, and does not switch partway through.
ENTITY_LEVEL_FEATURES: List[str] = [
    "channel_code",
    "is_cross_border",
]

# Produced by chhal.behaviour.derive (and hour_of / day_of_week_of) from a timeline.
# NOTHING may perturb these directly. Doing so re-creates the exact defect the campaign
# architecture exists to prevent: four independent views of one timeline, mutually
# contradicting. If a search wants to move them, it must move the timeline.
DERIVED_FEATURES: List[str] = [
    "hour",
    "day_of_week",
    "velocity_1h",
    "velocity_24h",
    "time_since_last_txn_min",
    "amount_to_avg_ratio",
]

# Columns of AttackBatch.timeline — the campaign a batch was rendered from, carried
# alongside the feature rows so the optimizer can re-derive instead of perturbing.
TIMELINE_COLUMNS: List[str] = ["entity", "timestamp_s", "amount", "is_attack"]

# Features that must hold whole numbers wherever they are produced or perturbed.
# Shared by the red team's sampler and the evasion optimizer so the two cannot drift.
INTEGER_FEATURES: List[str] = [
    "hour",
    "day_of_week",
    "velocity_1h",
    "velocity_24h",
    "is_new_beneficiary",
    "is_cross_border",
    "channel_code",
]

CHANNELS = {"card": 0, "upi": 1, "imps": 2}


@dataclass
class AttackBatch:
    """What the red team emits. All rows are fraud (label == 1)."""

    vector_id: str                       # e.g. "threshold_hugging"
    iteration: int                       # which loop pass produced it
    transactions: pd.DataFrame           # columns == FEATURE_COLUMNS, exactly
    provenance: Dict = field(default_factory=dict)  # seed, optimizer params, storyline
    # The campaign these rows were rendered from: one row per transaction INCLUDING the
    # host account's real history, with is_attack marking which of them are ours. The
    # optimizer needs the history because velocity and amount_to_avg_ratio are measured
    # against it. None only for hand-built batches in tests.
    timeline: "pd.DataFrame | None" = None

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
        if self.timeline is not None:
            missing_t = set(TIMELINE_COLUMNS) - set(self.timeline.columns)
            if missing_t:
                raise ValueError(
                    f"AttackBatch[{self.vector_id}].timeline is missing "
                    f"{sorted(missing_t)}"
                )
            n_attack = int(self.timeline["is_attack"].sum())
            if n_attack != len(self.transactions):
                raise ValueError(
                    f"AttackBatch[{self.vector_id}] timeline has {n_attack} attack rows "
                    f"but {len(self.transactions)} feature rows — they must correspond "
                    f"one-to-one and in order."
                )
        return self

    def __len__(self) -> int:
        return len(self.transactions)


# The false-positive rates a payments team actually tunes to. Fraud systems are not
# operated at "score >= 0.5"; they are operated at "flag no more than X% of good
# traffic, and catch as much as possible inside that budget". 0.1% is the tight,
# realistic setting; the looser two show how much is bought by relaxing it.
OPERATING_POINTS = (0.001, 0.005, 0.01)
PRIMARY_FPR = 0.001


@dataclass
class ScoreReport:
    """What the detector returns after scoring a set of transactions.

    Two families of numbers live here and they are not equally trustworthy.

    Operating-point metrics (`recall_at_fpr`, `pr_auc`, `alert_rate`) are the ones to
    quote. Recall at a fixed false-positive rate is the question an issuer actually
    asks, and `pr_auc` (average precision) is the honest summary under this much class
    imbalance — ROC `auc` looks spectacular at 3.5% prevalence no matter what, because
    the true-negative pile it divides by is enormous.

    Threshold metrics (`precision`, `recall`, `f1`, `fp_rate_on_legit`) are kept for
    continuity and comparison ONLY. They are computed at a fixed 0.5 cutoff, which no
    deployed system uses.
    """

    iteration: int
    split: str                           # "train" | "heldout_known" | "heldout_novel"
    precision: float
    recall: float
    f1: float
    auc: float                           # ROC AUC — flattering under imbalance, see above
    fp_rate_on_legit: float
    pr_auc: float = 0.0                  # average precision — the honest summary
    recall_at_fpr: Dict[float, float] = field(default_factory=dict)
    threshold_at_fpr: Dict[float, float] = field(default_factory=dict)
    # The false-positive rate each threshold ACTUALLY realises. At most the budget it
    # was asked for; carried so a reader can verify that from the results, not take it
    # on trust. See evaluation.threshold_for_fpr for why the two can come apart.
    realised_fpr_at_fpr: Dict[float, float] = field(default_factory=dict)
    alert_rate: float = 0.0              # share of the SCORED MIXTURE flagged at PRIMARY_FPR
    # Share of REAL traffic flagged at the same threshold. The number above is not an
    # operational rate: the mixture it divides by is legitimate traffic plus however
    # many attacks the run happened to generate, so it moves with a config knob
    # (100 / 500 / 2000 attacks per vector took it 0.0016 -> 0.0080 -> 0.0319 with the
    # detector unchanged). This one is measured on the real test set alone.
    alert_rate_on_real_traffic: float = 0.0
    per_vector_recall: Dict[str, float] = field(default_factory=dict)
    per_vector_recall_at_fpr: Dict[str, float] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)  # LightGBM gain ranking, not SHAP

    def as_row(self) -> Dict:
        row = {
            "iteration": self.iteration,
            "split": self.split,
            # lead with the operating-point numbers
            "pr_auc": self.pr_auc,
            "alert_rate": self.alert_rate,
            "alert_rate_on_real_traffic": self.alert_rate_on_real_traffic,
            # threshold-0.5 numbers, retained for comparison only
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "fp_rate_on_legit": self.fp_rate_on_legit,
        }
        for fpr in OPERATING_POINTS:
            row[f"recall_at_fpr_{fpr}"] = self.recall_at_fpr.get(fpr, float("nan"))
            row[f"realised_fpr_{fpr}"] = self.realised_fpr_at_fpr.get(fpr, float("nan"))
        return row
