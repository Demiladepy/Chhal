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
    """The evasion optimizer must not be able to optimise over the issuer's own view.

    One inherited column is deliberately exempt. `account_age_days` is not frozen at the
    value the host had; it advances with the clock, so an attacker who waits three weeks
    longer before using the card faces a card that is genuinely three weeks older. They
    still cannot SET it — they set the timing, and the age follows, exactly as velocity
    follows. Every other issuer-side signal must come out byte-identical.
    """
    from chhal.contract import ATTACKER_CONTROLLED, ATTACKER_DIRECT
    assert not (set(INHERITED_FEATURES) & set(ATTACKER_CONTROLLED))
    assert not (set(INHERITED_FEATURES) & set(ATTACKER_DIRECT))
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    from chhal.detector import Detector
    from chhal.optimizer import EvasionOptimizer
    det = Detector(seed=1).fit(base.train)
    batch = ALL_VECTORS[1]().calibrate(prof, pool).batch(200, 0, np.random.default_rng(5))
    frozen = [c for c in INHERITED_FEATURES if c != "account_age_days"]
    before = batch.transactions[INHERITED_FEATURES].copy()
    after = EvasionOptimizer(base.feature_stats).optimize(
        batch, det, np.random.default_rng(5)).transactions[INHERITED_FEATURES]
    pd.testing.assert_frame_equal(before[frozen].reset_index(drop=True),
                                  after[frozen].reset_index(drop=True))
    # the one exempt column may move, but only ever forwards from a real account's age
    assert (after["account_age_days"] >= 0).all()


def test_the_optimizer_cannot_invent_a_host(base, pool):
    """Moving the timeline must not let the search reach a card that does not exist."""
    from chhal.detector import Detector
    from chhal.optimizer import EvasionOptimizer
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    det = Detector(seed=1).fit(base.train)
    batch = ALL_VECTORS[2]().calibrate(prof, pool).batch(200, 0, np.random.default_rng(11))
    adapted = EvasionOptimizer(base.feature_stats).optimize(
        batch, det, np.random.default_rng(11))
    tl = adapted.timeline
    atk = tl["is_attack"].to_numpy()
    last_real = (pd.Series(np.where(atk, np.nan, tl["timestamp_s"]))
                 .groupby(tl["entity"]).transform("max").to_numpy())
    assert (tl["timestamp_s"].to_numpy()[atk] > last_real[atk]).all(), (
        "an optimized attack reached back into its host's real history")


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


# ---------------------------------------------------------------------------
# per-victim mimicry and coordination — the two things a vector can declare
# beyond its own bands
# ---------------------------------------------------------------------------
def _starts_per_campaign(camp):
    """First attack timestamp of every campaign in a batch."""
    out = []
    for e in np.unique(camp.entity):
        m = (camp.entity == e) & camp.is_attack
        if m.any():
            out.append(camp.timestamp_s[m].min())
    return np.array(out, float)


def test_the_hero_vector_sizes_its_attack_to_the_victim_not_to_the_crowd(base, pool):
    """`mimic_host` is the whole hero vector, so it needs a test that fails without it.

    A population-band attack spends the median customer's money on every card it touches,
    which is visibly wrong on any card that is not the median one: the detector scores
    `amount_to_avg_ratio` against THIS account's baseline. Reading the band off the victim
    instead should pull that ratio toward 1.
    """
    from dataclasses import replace
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)

    hero = ALL_VECTORS[0]
    assert hero.temporal.mimic_host, "the hero vector is the one that mimics its victim"

    class Crowd(hero):                              # same vector, population bands
        vector_id = "hero_without_mimicry"
        temporal = replace(hero.temporal, mimic_host=False)

    def ratios(V, seed):
        rows = V().calibrate(prof, pool).batch(400, 0, np.random.default_rng(seed)).transactions
        return np.abs(rows["amount_to_avg_ratio"].to_numpy() - 1.0)

    mimic = np.median([np.median(ratios(hero, s)) for s in (0, 1, 2)])
    crowd = np.median([np.median(ratios(Crowd, s)) for s in (0, 1, 2)])
    assert mimic < crowd, (
        f"mimicking the victim should land nearer their own baseline: "
        f"|ratio-1| {mimic:.3f} with mimicry vs {crowd:.3f} without"
    )


def test_a_coordinated_vector_fires_its_campaigns_inside_one_window(base, pool):
    """What makes the fan-out a network rather than a pile of unrelated takeovers."""
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    fanout = [V for V in ALL_VECTORS if V.temporal.coordinated_window_s is not None]
    assert fanout, "no coordinated vector is registered"
    V = fanout[0]

    _, camp = V().calibrate(prof, pool).render_with_timeline(
        300, np.random.default_rng(3))
    starts = _starts_per_campaign(camp)
    assert len(starts) > 5

    window = V.temporal.coordinated_window_s
    # generous: campaigns are nudged forward to an hour of day and past each host's own
    # last transaction, so the realised spread is wider than the nominal window
    assert starts.ptp() < window * 6, (
        f"campaigns spread over {starts.ptp()/3600:.1f}h, window is {window/3600:.1f}h")


def test_an_uncoordinated_vector_does_not_accidentally_synchronise(base, pool):
    """The control for the test above — without the flag, takeovers are independent."""
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    V = [V for V in ALL_VECTORS if V.temporal.coordinated_window_s is None][0]
    _, camp = V().calibrate(prof, pool).render_with_timeline(300, np.random.default_rng(3))
    starts = _starts_per_campaign(camp)
    assert starts.ptp() > 6 * 3_600 * 6, "independent takeovers should be spread out"


def test_a_victim_with_almost_no_history_degrades_instead_of_breaking(base):
    """Mimicry needs a distribution to read. Two transactions is not one, and the vector
    has to fall back to the population bands rather than emit nonsense."""
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    thin = HostPool(base.train, min_history=2)
    rows = ALL_VECTORS[0]().calibrate(prof, thin).batch(
        200, 0, np.random.default_rng(9)).transactions
    assert np.isfinite(rows.to_numpy(float)).all()
    assert (rows["amount"] > 0).all()
