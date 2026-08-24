"""Attacks as campaigns on an account, not as independent rows.

A fraud vector is not a cloud of transactions, it is something that HAPPENS to an
account over time: a card gets probed forty times in ten minutes, an aged synthetic
identity empties itself in an afternoon, a victim's account drains through three
transfers in four minutes. Sampling `velocity_24h` and `time_since_last_txn_min`
separately cannot represent any of that, and produces rows that contradict themselves.

So a vector now declares a `TemporalProfile` — how many accounts, how many transactions
each, how far apart, how the amount moves across the campaign — and the generator lays
out an actual timeline. The behavioural features are then DERIVED from that timeline by
`chhal.behaviour`, the same function that derives them from the 590,540 real
transactions. Consistency is not enforced afterwards; it is impossible to violate.

Each campaign also gets a short HISTORY of ordinary transactions before the attack
begins. That is what gives `amount_to_avg_ratio` a real meaning: the ratio is now
against what this account actually spent before, rather than a number drawn from a
distribution. A bust-out reads as large *for that account*, which is the whole signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW_DAYS = 180        # attacks are spread over a window comparable to the real data


@dataclass
class TemporalProfile:
    """How one vector lays itself out in time. All bands are LEGIT quantile levels."""

    txns_per_entity: tuple[int, int]         # attack transactions per compromised account
    inter_arrival_s: tuple[float, float]     # seconds between them (log-uniform)
    amount_band: tuple[float, float]         # quantile band of legit amount
    history_txns: tuple[int, int] = (3, 12)  # ordinary transactions before the attack
    history_gap_s: tuple[float, float] = (7_200.0, 604_800.0)   # 2h - 7d apart
    history_amount_band: tuple[float, float] = (0.30, 0.70)     # the account's normal spend
    start_hour_band: tuple[float, float] = (0.0, 1.0)           # when the campaign begins
    amount_trend: float = 1.0                # multiplier from first to last attack txn


@dataclass
class Campaigns:
    entity: np.ndarray      # entity index per row
    timestamp_s: np.ndarray
    amount: np.ndarray
    is_attack: np.ndarray   # False for the history rows that establish the baseline


def _log_uniform(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.uniform(np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6)), n))


def generate(profile: TemporalProfile, n_attack_rows: int, base_profile,
             rng: np.random.Generator) -> Campaigns:
    """Lay out enough campaigns to yield at least `n_attack_rows` attack transactions."""
    ent, ts, amt, atk = [], [], [], []
    produced, idx = 0, 0

    while produced < n_attack_rows:
        n_a = int(rng.integers(profile.txns_per_entity[0], profile.txns_per_entity[1] + 1))
        n_h = int(rng.integers(profile.history_txns[0], profile.history_txns[1] + 1))

        # when this campaign starts: a real hour-of-day, on a random day of the window
        start_hour = float(base_profile.band("hour", *profile.start_hour_band, 1, rng)[0])
        start = (int(rng.integers(0, WINDOW_DAYS)) * 86_400
                 + int(start_hour) * 3_600 + int(rng.integers(0, 3_600)))

        # history: ordinary spend, walking backwards from the campaign start
        h_gaps = _log_uniform(*profile.history_gap_s, n_h, rng)
        h_ts = start - np.cumsum(h_gaps)[::-1]
        h_amt = base_profile.band("amount", *profile.history_amount_band, n_h, rng)

        # the attack itself, walking forwards
        a_gaps = _log_uniform(*profile.inter_arrival_s, n_a, rng)
        a_ts = start + np.cumsum(a_gaps)
        a_amt = base_profile.band("amount", *profile.amount_band, n_a, rng)
        if profile.amount_trend != 1.0 and n_a > 1:
            a_amt = a_amt * np.linspace(1.0, profile.amount_trend, n_a)

        ent.append(np.full(n_h + n_a, idx))
        ts.append(np.r_[h_ts, a_ts])
        amt.append(np.r_[h_amt, a_amt])
        atk.append(np.r_[np.zeros(n_h, bool), np.ones(n_a, bool)])
        produced += n_a
        idx += 1

    return Campaigns(
        entity=np.concatenate(ent),
        timestamp_s=np.concatenate(ts).astype(np.int64),
        amount=np.concatenate(amt),
        is_attack=np.concatenate(atk),
    )
