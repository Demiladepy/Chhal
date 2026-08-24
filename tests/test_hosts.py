"""Tests for mounting campaigns on real accounts — and for the leakage rules that allows.

The feature space contains the dataset's anonymised entity-linkage counts, which carry
most of the real-fraud signal and cannot be reconstructed from anything we understand.
The only honest way to have them is to inherit them from a real account, which puts the
whole weight of the design on the rules in hosts.py. These tests are those rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import (FEATURE_COLUMNS, INHERITED_FEATURES,  # noqa: E402
                            LABEL_COLUMN, LINKAGE_FEATURES)
from chhal.data import load_base_data                             # noqa: E402
from chhal.redteam import ALL_VECTORS                             # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import ACCOUNT_COLUMN, HostPool          # noqa: E402

SMALL = dict(source="synthetic", n_legit=6000, n_baseline_fraud=200, seed=1)


@pytest.fixture(scope="module")
def base():
    return load_base_data(**SMALL)


@pytest.fixture(scope="module")
def pool(base):
    return HostPool(base.train)


def test_only_never_fraudulent_accounts_may_host_a_campaign(base, pool):
    """A fraudulent account's rows carry label information. Mounting an attack on one
    would smuggle it into the attack row."""
    fraud_accounts = set(base.train.loc[base.train[LABEL_COLUMN] == 1, ACCOUNT_COLUMN])
    rng = np.random.default_rng(0)
    inherited_seen = {tuple(pool.sample(rng).inherited) for _ in range(300)}
    clean = base.train[~base.train[ACCOUNT_COLUMN].isin(fraud_accounts)]
    allowed = {tuple(r) for r in clean[INHERITED_FEATURES].to_numpy()}
    assert inherited_seen <= allowed


def test_exclude_accounts_removes_them_entirely(base):
    seen_in_train = set(base.train[ACCOUNT_COLUMN])
    strict = HostPool(base.test, exclude_accounts=seen_in_train)
    rng = np.random.default_rng(1)
    # every sampled host must come from an account absent from training
    test_only = base.test[~base.test[ACCOUNT_COLUMN].isin(seen_in_train)]
    allowed_ts = set(test_only["_ts"].tolist())
    for _ in range(200):
        assert set(strict.sample(rng).history_ts.tolist()) <= allowed_ts


def test_a_pool_with_no_eligible_accounts_fails_loudly(base):
    with pytest.raises(ValueError, match="no eligible host"):
        HostPool(base.train, exclude_accounts=base.train[ACCOUNT_COLUMN])


def test_pool_requires_the_helper_columns(base):
    with pytest.raises(ValueError, match="host pool needs"):
        HostPool(base.train.drop(columns=["_account"]))


def test_a_vector_without_a_host_pool_fails_loudly(base):
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    v = ALL_VECTORS[0]().calibrate(prof)          # no pool
    with pytest.raises(RuntimeError, match="no host pool"):
        v.batch(10, 0, np.random.default_rng(0))


@pytest.mark.parametrize("V", ALL_VECTORS, ids=lambda V: V.vector_id)
def test_attacks_happen_strictly_after_the_host_last_real_transaction(V, base, pool):
    """A campaign continues an account. It cannot reach into its past."""
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    rng = np.random.default_rng(2)
    from chhal.redteam.campaign import generate
    camp = generate(V.temporal, 400, prof, rng, pool)
    for e in np.unique(camp.entity):
        m = camp.entity == e
        real, attack = camp.timestamp_s[m & ~camp.is_attack], camp.timestamp_s[m & camp.is_attack]
        if len(real) and len(attack):
            assert attack.min() > real.max(), "an attack reached into the host's past"


@pytest.mark.parametrize("V", ALL_VECTORS, ids=lambda V: V.vector_id)
def test_issuer_side_features_are_inherited_not_invented(V, base, pool):
    """Every inherited value on an attack row must be a value some real account actually
    had. If a vector were sampling them, values would appear that no account ever held."""
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    rows = V().calibrate(prof, pool).batch(400, 0, np.random.default_rng(4)).transactions
    real = {tuple(r) for r in base.train[LINKAGE_FEATURES].to_numpy()}
    emitted = {tuple(r) for r in rows[LINKAGE_FEATURES].to_numpy()}
    assert emitted <= real, "a vector emitted a linkage vector no real account had"


def test_the_attacker_cannot_move_inherited_features(base, pool):
    """The evasion optimizer must not be able to optimise over the issuer's own view."""
    from chhal.contract import ATTACKER_CONTROLLED
    assert not (set(INHERITED_FEATURES) & set(ATTACKER_CONTROLLED))
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    from chhal.detector import Detector
    from chhal.optimizer import EvasionOptimizer
    det = Detector(seed=1).fit(base.train)
    batch = ALL_VECTORS[1]().calibrate(prof, pool).batch(200, 0, np.random.default_rng(5))
    before = batch.transactions[INHERITED_FEATURES].copy()
    after = EvasionOptimizer(base.feature_stats).optimize(
        batch, det, np.random.default_rng(5)).transactions[INHERITED_FEATURES]
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_account_age_advances_while_the_attacker_holds_the_card(base, pool):
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    rows = ALL_VECTORS[2]().calibrate(prof, pool).batch(500, 0, np.random.default_rng(6)).transactions
    assert (rows["account_age_days"] >= 0).all()
    assert rows["account_age_days"].std() > 0, "age should vary across hosts and time"


def test_feature_space_partitions_cleanly():
    from chhal.contract import ATTACKER_CONTROLLED
    covered = set(ATTACKER_CONTROLLED) | set(INHERITED_FEATURES) | {"day_of_week"}
    assert covered == set(FEATURE_COLUMNS)
    assert len(LINKAGE_FEATURES) == 14
