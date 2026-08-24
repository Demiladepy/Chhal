"""Behavioural features derived from an actual timeline — one implementation, two callers.

`velocity_1h`, `velocity_24h`, `time_since_last_txn_min` and `amount_to_avg_ratio` are
not independent numbers. They are four views of the same underlying thing: a sequence of
transactions by one account, in time order. Sampling them separately produces rows that
cannot exist — a row claiming four transactions in the last hour while also claiming the
previous one was five hours ago. Before this module existed, 100% of the hero vector's
rows violated that constraint, against 0% of real traffic, because real traffic is
derived from timelines and the attacks were not.

So both sides now go through the same function. `scripts/prepare_ieee.py` calls it on
590,540 real transactions; the red team calls it on generated campaigns. Whatever
relationships hold between these four features in real data hold in the attacks too,
because it is literally the same arithmetic.

Every value uses only transactions STRICTLY BEFORE the row it describes, so no row can
see its own future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIOURAL_COLUMNS = [
    "velocity_1h", "velocity_24h", "time_since_last_txn_min", "amount_to_avg_ratio",
]
FIRST_TXN_GAP_MIN = 43_200.0   # 30 days, used when an account has no prior transaction


def _block_starts(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index of the contiguous block each row belongs to, and the block starts."""
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return np.repeat(starts, np.diff(np.r_[starts, len(codes)])), starts


def derive(uid: np.ndarray, timestamp_s: np.ndarray, amount: np.ndarray) -> pd.DataFrame:
    """Behavioural features for each transaction, given account, time and amount.

    Rows may arrive in any order; the result is returned in the order given.
    """
    uid = np.asarray(uid)
    dt = np.asarray(timestamp_s, dtype=np.int64)
    amt = np.asarray(amount, dtype=np.float64)
    if not (len(uid) == len(dt) == len(amt)):
        raise ValueError("uid, timestamp_s and amount must be the same length")
    if len(uid) == 0:
        return pd.DataFrame({c: np.array([]) for c in BEHAVIOURAL_COLUMNS})

    codes = pd.factorize(uid)[0]
    order = np.lexsort((dt, codes))            # by account, then time
    c_s, dt_s, amt_s = codes[order], dt[order], amt[order]
    block_start, starts = _block_starts(c_s)
    pos = np.arange(len(c_s)) - block_start    # how many prior txns this account has

    def velocity(window_s: int) -> np.ndarray:
        out = np.empty(len(c_s), np.int64)
        for s, e in zip(starts, np.r_[starts[1:], len(c_s)]):
            seg = dt_s[s:e]
            lo = np.searchsorted(seg, seg - window_s, side="left")
            out[s:e] = np.arange(e - s) - lo
        return out

    gap_min = (dt_s - np.r_[0, dt_s[:-1]]) / 60.0
    gap_min[pos == 0] = FIRST_TXN_GAP_MIN

    csum = np.r_[0.0, np.cumsum(amt_s)[:-1]]           # exclusive cumulative sum
    prior_sum = csum - csum[block_start]
    with np.errstate(divide="ignore", invalid="ignore"):
        prior_mean = np.where(pos > 0, prior_sum / np.maximum(pos, 1), np.nan)
        ratio = np.where(pos > 0, amt_s / prior_mean, 1.0)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0)

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return pd.DataFrame({
        "velocity_1h": velocity(3_600)[inv],
        "velocity_24h": velocity(86_400)[inv],
        "time_since_last_txn_min": gap_min[inv],
        "amount_to_avg_ratio": ratio[inv],
    })


def hour_of(timestamp_s: np.ndarray, offset_hours: int = 0) -> np.ndarray:
    return ((np.asarray(timestamp_s, dtype=np.int64) // 3_600) + offset_hours) % 24


def day_of_week_of(timestamp_s: np.ndarray) -> np.ndarray:
    return (np.asarray(timestamp_s, dtype=np.int64) // 86_400) % 7


def consistency_violations(df: pd.DataFrame) -> dict:
    """How many rows are physically impossible. Zero, for anything derive() produced.

    If k transactions happened within the last hour then the previous one was at most an
    hour ago, and likewise for the 24h window. Used by the tests and worth running over
    any frame claiming to be transaction-like.
    """
    return {
        "violates_1h_rule": float(((df["velocity_1h"] >= 1)
                                   & (df["time_since_last_txn_min"] > 60)).mean()),
        "violates_24h_rule": float(((df["velocity_24h"] >= 1)
                                    & (df["time_since_last_txn_min"] > 1440)).mean()),
        "velocity_1h_exceeds_24h": float((df["velocity_1h"] > df["velocity_24h"]).mean()),
    }
