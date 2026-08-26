"""Base transaction distribution — real by default, synthetic as a fallback.

Two sources, one interface
--------------------------
`"ieee"`      590,540 REAL card transactions (Vesta / IEEE-CIS, 3.499% fraud, 182
              days), derived into FEATURE_COLUMNS by `scripts/prepare_ieee.py`. This
              is what every headline number should be quoted from: "fidelity of
              simulation" is judged against real payment data, and a distance measured
              against a distribution we invented ourselves proves nothing.
`"synthetic"` The original programmatic distribution. Kept so the repo still runs end
              to end with no download, and so tests stay fast — never for headline
              numbers. It is measurably wrong: against real traffic its median amount
              is 13x too high, its median account does 6 transactions a day where the
              real median does 0, and its median inter-transaction gap is 138x too
              short. Worse, its synthetic fraud multiplies amount by 2.5-6x, inventing
              a separation that does not exist in real data (real fraud mean 149.2 vs
              legit 134.5) and making detection look far easier than it is.

Two things are computed here that the rest of the loop depends on:

`feature_stats`     coarse quantiles used by the evasion optimizer as the plausibility
                    manifold. Computed on TRAIN ONLY — deriving them from train+test
                    would let the optimizer's guardrail see the future.
`legit_quantiles`   a fine quantile grid over LEGITIMATE TRAIN traffic only. The red
                    team samples every continuous feature through this grid's inverse
                    CDF, so attack values are drawn from the shape of real traffic
                    rather than from hand-picked constants, and the vectors port to any
                    dataset without rescaling.

The split is TEMPORAL on both sources (train on the past, test on the future). A random
split leaks future fraud patterns backwards and inflates every metric.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .contract import FEATURE_COLUMNS, LABEL_COLUMN, LINKAGE_FEATURES

DEFAULT_IEEE_PARQUET = os.environ.get(
    "CHHAL_IEEE_PARQUET", os.path.expanduser("~/chhal-data/ieee_base.parquet")
)

# Helper columns carried alongside the features, never part of them: which account a
# transaction belongs to and when it happened. The red team needs both to mount a
# campaign on a real account (see redteam/hosts.py). Every model path selects
# FEATURE_COLUMNS explicitly, so their presence is inert.
HOST_COLUMNS = ["_account", "_ts"]

MANIFOLD_QUANTILES = [0.005, 0.05, 0.5, 0.95, 0.995]
GRID = np.round(np.linspace(0.0, 1.0, 1001), 5)   # inverse-CDF grid for the red team
CATEGORICAL_FEATURES = ["channel_code"]


@dataclass
class BaseData:
    train: pd.DataFrame           # legit + fraud, the past
    test: pd.DataFrame            # frozen hold-out, the future; filter is_fraud==0 for FP
    feature_stats: pd.DataFrame   # coarse manifold quantiles, TRAIN only
    legit_quantiles: pd.DataFrame # fine quantile grid over legit TRAIN traffic
    legit_categoricals: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    source: str = "synthetic"

    def describe(self) -> str:
        return (f"source={self.source} train={len(self.train):,} "
                f"({self.train[LABEL_COLUMN].mean()*100:.3f}% fraud) "
                f"test={len(self.test):,} "
                f"({self.test[LABEL_COLUMN].mean()*100:.3f}% fraud)")


# ---------------------------------------------------------------------------
# shared: derive the manifold + the red team's sampling grid from a TRAIN frame
# ---------------------------------------------------------------------------
def _profile_from_train(train: pd.DataFrame) -> tuple:
    legit = train[train[LABEL_COLUMN] == 0]
    feature_stats = train[FEATURE_COLUMNS].quantile(MANIFOLD_QUANTILES)
    legit_quantiles = legit[FEATURE_COLUMNS].quantile(GRID)
    cats = {}
    for col in CATEGORICAL_FEATURES:
        vc = legit[col].value_counts(normalize=True).sort_index()
        cats[col] = (vc.index.to_numpy(float), vc.to_numpy(float))
    return feature_stats, legit_quantiles, cats


# ---------------------------------------------------------------------------
# source: real IEEE-CIS
# ---------------------------------------------------------------------------
# What the prepared parquet has to look like before we are willing to label results
# `source=ieee`. The floor is deliberately loose — it is there to catch a truncated or
# fabricated file, not to pin an exact row count that a future prep change may move.
IEEE_TOTAL_ROWS = 590_540
MIN_IEEE_ROWS = 500_000

def _load_ieee(path: str, seed: int, max_rows: int | None) -> BaseData:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it once with:\n"
            f"    python scripts/prepare_ieee.py\n"
            f"(downloads the real IEEE-CIS transactions, ~683MB, then derives "
            f"FEATURE_COLUMNS in a few seconds)."
        )
    df = pd.read_parquet(path)
    train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    if not set(HOST_COLUMNS) <= set(train.columns):
        raise SystemExit(f"{path} predates the host pool; rerun scripts/prepare_ieee.py")
    # The file's NAME is the only thing asserting it holds real IEEE-CIS, and a name is
    # not evidence. A truncated or hand-made parquet used to load silently and every
    # result downstream would then be reported as `source=ieee`. Check the shape the
    # real thing actually has before agreeing to call it that.
    n = len(train) + len(test)
    if n < MIN_IEEE_ROWS:
        raise SystemExit(
            f"{path} holds {n:,} rows; real IEEE-CIS has {IEEE_TOTAL_ROWS:,}. "
            f"This is a truncated or hand-made file and must not be reported as "
            f"source=ieee. Rerun scripts/prepare_ieee.py.")
    if not len(train) or not len(test):
        raise SystemExit(f"{path} has an empty split (train={len(train)}, test={len(test)})")
    if max_rows:   # stratified subsample for fast iteration; never for headline numbers
        rng = np.random.default_rng(seed)
        def sub(d, n):
            if len(d) <= n:
                return d
            keep = rng.choice(len(d), n, replace=False)
            return d.iloc[np.sort(keep)].reset_index(drop=True)
        train, test = sub(train, max_rows), sub(test, max_rows // 3)
    fs, lq, cats = _profile_from_train(train)
    return BaseData(train, test, fs, lq, cats, source="ieee")


# ---------------------------------------------------------------------------
# source: synthetic fallback (unchanged behaviour, kept for offline runs and tests)
# ---------------------------------------------------------------------------
def _synthetic_accounts(n: int, rng: np.random.Generator) -> tuple:
    """Give the fallback accounts and a clock too, so the red team can mount campaigns on
    it exactly as it does on real data. Roughly four transactions per account, spread over
    a 180-day window."""
    acct = rng.integers(0, max(n // 4, 1), n)
    ts = np.sort(rng.integers(0, 180 * 86_400, n))
    return acct, ts


def _sample_legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    channel = rng.choice([0, 1, 2], size=n, p=[0.45, 0.45, 0.10])
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
    acct, ts = _synthetic_accounts(n, rng)
    df["_account"], df["_ts"] = acct, ts
    # The fallback cannot represent entity linkage — those counts aggregate over devices,
    # phones and cross-card relationships that no generator here has. They are zero-filled
    # and carry no signal, which is one more reason this source must never be quoted.
    for col in LINKAGE_FEATURES:
        df[col] = 0.0
    return df


def _sample_baseline_fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = _sample_legit(n, rng)
    df["_account"] = df["_account"] + 1_000_000        # a disjoint account space
    df["amount"] = np.round(df["amount"] * rng.uniform(2.5, 6.0, n), 2)
    df["hour"] = rng.choice(list(range(0, 5)) + list(range(22, 24)), n)
    df["is_new_beneficiary"] = 1
    df["velocity_24h"] = df["velocity_24h"] + rng.poisson(10, n)
    df["amount_to_avg_ratio"] = np.clip(df["amount_to_avg_ratio"] * rng.uniform(2, 5, n), 0.05, 30)
    df["merchant_risk"] = np.clip(df["merchant_risk"] + rng.uniform(0.2, 0.5, n), 0, 1)
    df[LABEL_COLUMN] = 1
    return df


def _load_synthetic(n_legit: int, n_baseline_fraud: int, test_frac: float,
                    seed: int) -> BaseData:
    rng = np.random.default_rng(seed)
    full = pd.concat([_sample_legit(n_legit, rng),
                      _sample_baseline_fraud(n_baseline_fraud, rng)], ignore_index=True)
    # Temporal here too, for the same reason it is temporal on the real source: train on
    # the past, test on the future. This used to shuffle and cut, which is a random
    # split — and since the whole test suite runs on this source, the one place the
    # no-leakage discipline gets exercised was the one place it did not hold. It also
    # made an account appear on both sides only by chance, so the train/test account
    # exclusion looked unnecessary. Sorting by time makes both real.
    full = full.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    cut = int(len(full) * (1 - test_frac))
    train, test = full.iloc[:cut].copy(), full.iloc[cut:].copy()
    fs, lq, cats = _profile_from_train(train)
    return BaseData(train, test, fs, lq, cats, source="synthetic")


# ---------------------------------------------------------------------------
def load_base_data(
    source: str = "auto",
    n_legit: int = 40_000,
    n_baseline_fraud: int = 800,
    test_frac: float = 0.25,
    seed: int = 7,
    ieee_path: str | None = None,
    max_rows: int | None = None,
) -> BaseData:
    """Load the base population.

    source="auto" uses real IEEE-CIS when the prepared parquet exists and falls back to
    synthetic otherwise. Callers that must be unambiguous (anything producing a number
    for the write-up) should pass source explicitly. `n_legit`, `n_baseline_fraud` and
    `test_frac` apply to the synthetic source only — the real split is temporal and
    baked into the parquet by scripts/prepare_ieee.py.
    """
    path = ieee_path or DEFAULT_IEEE_PARQUET
    if source == "auto":
        source = "ieee" if os.path.exists(path) else "synthetic"
    if source == "ieee":
        return _load_ieee(path, seed, max_rows)
    if source == "synthetic":
        return _load_synthetic(n_legit, n_baseline_fraud, test_frac, seed)
    raise ValueError(f"unknown source {source!r}; expected 'ieee', 'synthetic' or 'auto'")
