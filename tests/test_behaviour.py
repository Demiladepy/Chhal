"""Tests for timeline-derived features. The defect Fix 2 existed to remove.

Before campaigns, `velocity_*`, `time_since_last_txn_min` and `amount_to_avg_ratio` were
sampled independently, so 100% of `threshold_hugging`'s rows were physically impossible:
activity claimed inside the last 24 hours while the previous transaction was days ago.
Real traffic violated it 0% of the time, because real traffic is derived from timelines.
These tests keep it that way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.behaviour import (BEHAVIOURAL_COLUMNS, consistency_violations,  # noqa: E402
                             day_of_week_of, derive, hour_of)
from chhal.data import load_base_data                                      # noqa: E402
from chhal.redteam import ALL_VECTORS                                      # noqa: E402
from chhal.redteam.base import BaseProfile                                 # noqa: E402
from chhal.redteam.hosts import HostPool                                   # noqa: E402

SMALL = dict(source="synthetic", n_legit=4000, n_baseline_fraud=100, seed=1)


def test_derive_matches_hand_computation():
    """One account: transactions at t=0, 30min, 2h with amounts 100, 50, 150."""
    d = derive(np.array(["a", "a", "a"]), np.array([0, 1800, 7200]),
               np.array([100.0, 50.0, 150.0]))
    assert list(d["velocity_1h"]) == [0, 1, 0]          # the 2h txn has none in its last hour
    assert list(d["velocity_24h"]) == [0, 1, 2]
    assert d["time_since_last_txn_min"].tolist()[1:] == [30.0, 90.0]
    assert d["amount_to_avg_ratio"].tolist() == [1.0, 0.5, 2.0]   # 150 / mean(100, 50)


def test_derive_never_looks_at_the_future_and_ignores_input_order():
    ts = np.array([7200, 0, 1800])
    d = derive(np.array(["a"] * 3), ts, np.array([150.0, 100.0, 50.0]))
    assert d["amount_to_avg_ratio"].iloc[1] == 1.0      # earliest row has no prior
    assert d["velocity_24h"].iloc[0] == 2               # latest row sees both priors


def test_accounts_do_not_leak_into_each_other():
    d = derive(np.array(["a", "b", "a"]), np.array([0, 10, 20]),
               np.array([100.0, 999.0, 100.0]))
    assert d["velocity_1h"].iloc[1] == 0                # b's only transaction
    assert d["amount_to_avg_ratio"].iloc[2] == 1.0      # a's second: 100 / 100


def test_derive_output_is_always_self_consistent():
    rng = np.random.default_rng(0)
    n = 20_000
    d = derive(rng.integers(0, 500, n), np.sort(rng.integers(0, 10**7, n)),
               rng.lognormal(4, 1, n))
    assert all(v == 0.0 for v in consistency_violations(d).values())
    assert list(d.columns) == BEHAVIOURAL_COLUMNS


@pytest.mark.parametrize("V", ALL_VECTORS, ids=lambda V: V.vector_id)
def test_every_vector_emits_physically_possible_transactions(V):
    """The regression that matters. threshold_hugging used to fail this on 100% of rows."""
    base = load_base_data(**SMALL)
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    rows = V().calibrate(prof, HostPool(base.train)).batch(
        1500, 0, np.random.default_rng(3)).transactions
    for rule, share in consistency_violations(rows).items():
        assert share == 0.0, f"{V.vector_id} violates {rule} on {share:.1%} of rows"


def test_campaigns_return_exactly_the_rows_asked_for_and_no_history():
    """History transactions exist to give the account a baseline; they are not attacks."""
    base = load_base_data(**SMALL)
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    pool = HostPool(base.train)
    for V in ALL_VECTORS:
        for n in (17, 250):
            assert len(V().calibrate(prof, pool).batch(n, 0, np.random.default_rng(1))) == n


def test_hour_and_day_of_week_come_from_the_timestamps():
    """hour_of defaults to the shared HOUR_OFFSET, so real data and generated campaigns
    read the same clock. They briefly did not, and every generated hour was five hours
    out without anything failing."""
    from chhal.behaviour import HOUR_OFFSET
    assert HOUR_OFFSET == -5
    ts = np.array([0, 3_600, 90_000])
    assert list(hour_of(ts)) == [19, 20, 20]                 # offset applied
    assert list(hour_of(ts, offset_hours=0)) == [0, 1, 1]    # raw, for comparison
    assert list(day_of_week_of(ts)) == [0, 0, 1]


def test_consistency_checker_actually_catches_a_bad_frame():
    """A checker that never fails is not a check."""
    bad = pd.DataFrame({"velocity_1h": [3], "velocity_24h": [1],
                        "time_since_last_txn_min": [5000.0]})
    v = consistency_violations(bad)
    assert v["violates_1h_rule"] == 1.0
    assert v["violates_24h_rule"] == 1.0
    assert v["velocity_1h_exceeds_24h"] == 1.0
