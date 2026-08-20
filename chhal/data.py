"""Base transaction distribution.

We generate a PaySim/UPI-flavoured base distribution programmatically so the whole
repo is self-contained and reproducible with one command. It is deliberately written
as a single swappable function: point `load_base_data` at real PaySim / IEEE-CIS
features instead and nothing downstream changes, because everything downstream only
knows FEATURE_COLUMNS.

The train/test split is FROZEN here, before any attack is ever injected. That freeze
is what lets us make a no-leakage claim in the write-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contract import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass
class BaseData:
    train: pd.DataFrame          # legit + a little baseline (non-GenAI) fraud
    test: pd.DataFrame           # frozen hold-out (legit + baseline fraud); filter to
                                  # is_fraud==0 for legit-only FP measurement
    feature_stats: pd.DataFrame  # per-feature quantiles of the realistic manifold


def _sample_legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A believable spread of legitimate payment behaviour."""
    channel = rng.choice([0, 1, 2], size=n, p=[0.45, 0.45, 0.10])
    # UPI skews small, card mid, imps larger
    base_amt = np.where(channel == 1, rng.lognormal(6.0, 0.9, n),
                np.where(channel == 0, rng.lognormal(7.2, 0.8, n),
                                       rng.lognormal(8.4, 0.7, n)))
    df = pd.DataFrame({
        "amount": np.round(base_amt, 2),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "velocity_1h": rng.poisson(0.7, n),
        "velocity_24h": rng.poisson(6.0, n),
        "amount_to_avg_ratio": np.clip(rng.normal(1.0, 0.35, n), 0.05, 6.0),
        "account_age_days": rng.gamma(shape=3.0, scale=260.0, size=n).clip(1, 4000),
        "time_since_last_txn_min": rng.exponential(240.0, n).clip(0.2, 20000),
        "is_new_beneficiary": (rng.random(n) < 0.15).astype(int),
        "is_cross_border": (rng.random(n) < 0.05).astype(int),
        "channel_code": channel,
        "merchant_risk": np.clip(rng.beta(2.0, 8.0, n), 0, 1),
        LABEL_COLUMN: 0,
    })
    return df


def _sample_baseline_fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Ordinary, non-GenAI fraud so the baseline detector is not starting from zero."""
    df = _sample_legit(n, rng)
    # crude classic fraud: bigger amounts, odd hours, new beneficiaries, more velocity
    df["amount"] = np.round(df["amount"] * rng.uniform(2.5, 6.0, n), 2)
    df["hour"] = rng.choice(list(range(0, 5)) + list(range(22, 24)), n)
    df["is_new_beneficiary"] = 1
    df["velocity_24h"] = df["velocity_24h"] + rng.poisson(10, n)
    df["amount_to_avg_ratio"] = np.clip(df["amount_to_avg_ratio"] * rng.uniform(2, 5, n), 0.05, 30)
    df["merchant_risk"] = np.clip(df["merchant_risk"] + rng.uniform(0.2, 0.5, n), 0, 1)
    df[LABEL_COLUMN] = 1
    return df


def load_base_data(
    n_legit: int = 40_000,
    n_baseline_fraud: int = 800,
    test_frac: float = 0.25,
    seed: int = 7,
) -> BaseData:
    rng = np.random.default_rng(seed)
    legit = _sample_legit(n_legit, rng)
    fraud = _sample_baseline_fraud(n_baseline_fraud, rng)
    full = pd.concat([legit, fraud], ignore_index=True)
    full = full.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cut = int(len(full) * (1 - test_frac))
    train, test = full.iloc[:cut].copy(), full.iloc[cut:].copy()

    # realistic manifold = quantiles of the whole base population, used by the
    # evasion optimizer's plausibility guardrail.
    qs = [0.005, 0.05, 0.5, 0.95, 0.995]
    feature_stats = full[FEATURE_COLUMNS].quantile(qs)

    return BaseData(train=train, test=test, feature_stats=feature_stats)
