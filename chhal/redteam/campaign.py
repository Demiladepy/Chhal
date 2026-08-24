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

The history a campaign is measured against is not generated either — it is the REAL
transaction history of a real, never-fraudulent account (see hosts.py). So
`amount_to_avg_ratio` is the ratio against what that card actually spent, and the
issuer-side context the attacker cannot control — account age, merchant risk, and the
dataset's entity-linkage counts — is inherited rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..behaviour import hour_of

TAKEOVER_GAP_S = 3_600           # a campaign starts at least an hour after the last real txn
MAX_TAKEOVER_WAIT_DAYS = 30      # ...and within a month of it


@dataclass
class TemporalProfile:
    """How one vector lays itself out in time. All bands are LEGIT quantile levels."""

    txns_per_entity: tuple[int, int]         # attack transactions per compromised account
    inter_arrival_s: tuple[float, float]     # seconds between them (log-uniform)
    amount_band: tuple[float, float]         # quantile band of legit amount
    start_hour_band: tuple[float, float] = (0.0, 1.0)           # when the campaign begins
    amount_trend: float = 1.0                # multiplier from first to last attack txn


@dataclass
class Campaigns:
    entity: np.ndarray      # entity index per row
    timestamp_s: np.ndarray
    amount: np.ndarray
    is_attack: np.ndarray   # False for the host's real transactions
    inherited: np.ndarray   # (n_rows, len(INHERITED_FEATURES)) — the host's issuer-side state


def _log_uniform(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.exp(rng.uniform(np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6)), n))


def _takeover_time(last_real_ts: int, hour_band: tuple[float, float],
                   base_profile, rng: np.random.Generator) -> int:
    """When the compromised card is first used, strictly after its last real transaction.

    Snapped forward to an hour of day this vector would plausibly start at, so the
    campaign's clock still lines up with the vector's story.
    """
    target = int(base_profile.band("hour", *hour_band, 1, rng)[0]) % 24
    earliest = (last_real_ts + TAKEOVER_GAP_S
                + int(rng.integers(0, MAX_TAKEOVER_WAIT_DAYS * 86_400)))
    delta = (target - int(hour_of(np.array([earliest]))[0])) % 24
    return earliest + delta * 3_600 + int(rng.integers(0, 3_600))


def generate(profile: TemporalProfile, n_attack_rows: int, base_profile,
             rng: np.random.Generator, hosts) -> Campaigns:
    """Mount campaigns on real accounts until at least `n_attack_rows` attacks exist."""
    ent, ts, amt, atk, inh = [], [], [], [], []
    produced, idx = 0, 0

    while produced < n_attack_rows:
        host = hosts.sample(rng)
        n_a = int(rng.integers(profile.txns_per_entity[0], profile.txns_per_entity[1] + 1))

        start = _takeover_time(host.last_ts, profile.start_hour_band, base_profile, rng)
        a_gaps = _log_uniform(*profile.inter_arrival_s, n_a, rng)
        a_ts = start + np.cumsum(a_gaps)
        a_amt = base_profile.band("amount", *profile.amount_band, n_a, rng)
        if profile.amount_trend != 1.0 and n_a > 1:
            a_amt = a_amt * np.linspace(1.0, profile.amount_trend, n_a)

        n_h = len(host.history_ts)
        ent.append(np.full(n_h + n_a, idx))
        ts.append(np.r_[host.history_ts, a_ts])
        amt.append(np.r_[host.history_amount, a_amt])
        atk.append(np.r_[np.zeros(n_h, bool), np.ones(n_a, bool)])
        inh.append(np.tile(host.inherited, (n_h + n_a, 1)))
        produced += n_a
        idx += 1

    return Campaigns(
        entity=np.concatenate(ent),
        timestamp_s=np.concatenate(ts).astype(np.int64),
        amount=np.concatenate(amt),
        is_attack=np.concatenate(atk),
        inherited=np.concatenate(inh),
    )
