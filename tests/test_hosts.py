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
from chhal.redteam.hosts import (ACCOUNT_COLUMN, HostPool,        # noqa: E402
                                 TIME_COLUMN)

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


def test_mimicry_sizes_its_attack_to_the_victim_not_to_the_crowd(base, pool):
    """`mimic_host` is the whole mimicry vector, so it needs a test that fails without it.

    A population-band attack spends the median customer's money on every card it touches,
    which is visibly wrong on any card that is not the median one: the detector scores
    `amount_to_avg_ratio` against THIS account's baseline. Reading the band off the victim
    instead should pull that ratio toward 1.
    """
    from dataclasses import replace
    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)

    mimicry = ALL_VECTORS[0]
    assert mimicry.temporal.mimic_host, "ALL_VECTORS[0] must be the vector that mimics its victim"

    class Crowd(mimicry):                           # same vector, population bands
        vector_id = "mimicry_ablated"
        temporal = replace(mimicry.temporal, mimic_host=False)

    def ratios(V, seed):
        rows = V().calibrate(prof, pool).batch(400, 0, np.random.default_rng(seed)).transactions
        return np.abs(rows["amount_to_avg_ratio"].to_numpy() - 1.0)

    mimic = np.median([np.median(ratios(mimicry, s)) for s in (0, 1, 2)])
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


# ---------------------------------------------------------------------------------
# trajectory replay: copying the victim's joint structure instead of its marginals
# ---------------------------------------------------------------------------------

LONG_HISTORY = 20


def _replay_pool(base):
    """Hosts with enough history that EVERY campaign can cut a block.

    `TrajectoryReplay` runs up to 9 attack transactions and `_host_trajectory` needs
    n + 2 real ones, so 11 is the worst case. Making every host at least that long makes
    "did it fall back?" unambiguous: with this pool a fallback is a bug, and the tests
    below can assert on every campaign rather than on whichever ones happened to be long
    enough.

    The small synthetic fixture has no account that long, so the legitimate rows are
    re-blocked into accounts of {n} — real column values, real timestamps, longer
    histories. Nothing here is a claim about the data; it is a bench with enough runway
    for the mechanism to be observable at all.
    """.format(n=LONG_HISTORY)
    df = base.train[base.train[LABEL_COLUMN] == 0].sort_values(TIME_COLUMN).copy()
    df = df.iloc[: (len(df) // LONG_HISTORY) * LONG_HISTORY]
    df[ACCOUNT_COLUMN] = np.repeat(np.arange(len(df) // LONG_HISTORY), LONG_HISTORY)
    return HostPool(df, min_history=LONG_HISTORY)


def _campaigns(camp):
    """Split a Campaigns bundle into (history_ts, history_amt, attack_ts, attack_amt)."""
    for e in np.unique(camp.entity):
        m = camp.entity == e
        h, a = m & ~camp.is_attack, m & camp.is_attack
        yield (camp.timestamp_s[h].astype(float), camp.amount[h],
               camp.timestamp_s[a].astype(float), camp.amount[a])


def test_replay_copies_a_contiguous_block_of_the_victims_own_history(base):
    """The claim `replay_host` makes, stated as an assertion.

    `mimic_host` matches the victim's marginals and destroys everything else: six
    independent draws from a card's own quantile band can easily be its 90th-percentile
    spend six times running — each value unremarkable, the sequence something that card
    has never once done. Replay is supposed to copy an ACTUAL slice, so the evidence for
    it is that one offset j into the victim's real history explains the whole campaign:
    the gaps are that slice's gaps up to per-gap jitter, and the amounts are that slice's
    amounts times ONE scale factor.

    Fails if replay silently degrades to mimicry, to the population bands, or to
    independent draws from the victim's own values — none of which admit a single j.
    """
    from chhal.redteam.campaign import REPLAY_JITTER, REPLAY_SCALE
    from chhal.redteam.vectors import TrajectoryReplay

    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    _, camp = TrajectoryReplay().calibrate(prof, _replay_pool(base)).render_with_timeline(
        400, np.random.default_rng(5))

    campaigns = list(_campaigns(camp))[:-1]        # the last one is truncated mid-way
    assert len(campaigns) > 10
    for h_ts, h_amt, a_ts, a_amt in campaigns:
        if len(a_ts) < 3:
            continue
        real_gaps, atk_gaps = np.diff(h_ts), np.diff(a_ts)
        k = len(atk_gaps)
        # The jitter band, plus two seconds of absolute slack for the int64 timestamp
        # cast. The slack has to be absolute rather than a widened ratio: campaigns can
        # contain seven-second gaps, where losing under a second to truncation twice over
        # is a 14% relative error and no honest ratio band would survive it.
        lo = np.maximum(real_gaps * REPLAY_JITTER[0], 1.0) - 2.0
        hi = real_gaps * REPLAY_JITTER[1] + 2.0
        offsets = [j for j in range(len(real_gaps) - k + 1)
                   if np.all(atk_gaps >= lo[j:j + k]) and np.all(atk_gaps <= hi[j:j + k])]
        assert offsets, (
            "no offset into this victim's real history explains the campaign's gaps, so "
            "nothing was replayed"
        )
        # the same j must also explain the amounts, and with a single shared scale
        assert any(
            np.all(h_amt[j:j + len(a_amt)] > 0)
            and np.allclose(r := a_amt / h_amt[j:j + len(a_amt)], r[0], rtol=1e-6)
            and REPLAY_SCALE[0] - 1e-9 <= r[0] <= REPLAY_SCALE[1] + 1e-9
            for j in offsets
        ), "the amounts are not one real slice under one scale factor"


def test_replay_leaves_the_victims_last_real_transaction_alone():
    """A block is cut from the card's settled past, not from whatever it did immediately
    before the takeover — copying that would make the campaign a literal continuation of
    the moment it interrupted.

    The history here is built so the offset is recoverable: amounts are 1..20, so a slice
    scaled by one factor still has consecutive-integer ratios and the block's position
    can be read straight back out of it.
    """
    from chhal.redteam.campaign import _host_trajectory

    rng = np.random.default_rng(0)
    n, ts = 4, np.arange(20, dtype=np.int64) * 86_400 + 1_000
    amt = np.arange(1.0, 21.0)
    for _ in range(200):
        out = _host_trajectory(ts, amt, n, rng)
        assert out is not None, "a twenty-transaction history is long enough to replay"
        _, amounts, _, _ = out
        scale = float(amounts[1] - amounts[0])           # consecutive values, one scale
        j = round(amounts[0] / scale) - 1
        assert j >= 0
        # the block reads n+1 timestamps, ts[j] .. ts[j + n], to get its n gaps
        assert j + n <= len(ts) - 2, (
            f"block ts[{j}:{j + n + 1}] reached the final real transaction at index "
            f"{len(ts) - 1}"
        )


def test_replay_refuses_a_victim_it_cannot_cut_a_block_from(base):
    """Too short to replay must mean *fall back*, not *emit nonsense*. The chain is
    replay -> mimicry -> population bands, and a thin-history pool has to come out the
    far end with valid rows."""
    from chhal.redteam.campaign import _host_trajectory
    from chhal.redteam.vectors import TrajectoryReplay

    rng = np.random.default_rng(0)
    short = np.arange(4, dtype=np.int64) * 86_400
    assert _host_trajectory(short, np.arange(4.0), 4, rng) is None

    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    rows = TrajectoryReplay().calibrate(prof, HostPool(base.train, min_history=2)).batch(
        300, 0, np.random.default_rng(4)).transactions
    assert np.isfinite(rows.to_numpy(float)).all()
    assert (rows["amount"] > 0).all()


def test_replay_preserves_a_cadence_that_mimicry_flattens(base):
    """Why copying beats resampling, on a statistic the marginals do not constrain.

    `_host_gaps` draws every gap from the victim's own 25th-75th percentile band, so a
    mimicry campaign has an unnaturally even cadence: real cards go quiet for a week and
    then spend three times in an evening, and a middle-quantile draw does neither. The
    spread of the gaps within one campaign, relative to the spread within the victim's
    own real history, is where that shows.

    This is the test that distinguishes the two vectors at all — they match on amount and
    gap marginals by construction, so a marginal comparison cannot tell them apart.
    """
    from chhal.redteam.vectors import ThresholdHugging, TrajectoryReplay

    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    pool = _replay_pool(base)

    def cv_ratio(V, seed):
        """Median over campaigns of (campaign gap CV) / (that victim's real gap CV)."""
        _, camp = V().calibrate(prof, pool).render_with_timeline(
            400, np.random.default_rng(seed))
        out = []
        for h_ts, _, a_ts, _ in list(_campaigns(camp))[:-1]:
            if len(a_ts) < 3 or len(h_ts) < 3:
                continue
            rg, ag = np.diff(h_ts), np.diff(a_ts)
            if rg.mean() <= 0 or ag.mean() <= 0:
                continue
            out.append((ag.std() / ag.mean()) / (rg.std() / rg.mean() + 1e-12))
        return float(np.median(out))

    # The reference is NOT 1.00. A campaign is 3-9 transactions, and a slice that short
    # does not carry the whole history's dispersion, so even a perfect uncopied block
    # scores well below 1. Measure that ceiling on this same fixture and score both
    # vectors against it, or the test rewards being closer to something unreachable.
    def ceiling(seed):
        rng = np.random.default_rng(seed)
        lo, hi = TrajectoryReplay.temporal.txns_per_entity
        out = []
        for i in rng.integers(0, len(pool), 2_000):
            ts = pool._host(int(i)).history_ts.astype(float)
            n = int(rng.integers(lo, hi + 1))
            if n < 3 or len(ts) < n + 2:
                continue
            rg = np.diff(ts)
            j = int(rng.integers(0, len(ts) - n - 1))
            bg = np.diff(ts[j:j + n])
            if rg.mean() <= 0 or rg.std() <= 0 or bg.mean() <= 0:
                continue
            out.append((bg.std() / bg.mean()) / (rg.std() / rg.mean()))
        return float(np.median(out))

    ceil = np.median([ceiling(s) for s in (0, 1, 2)])
    replay = np.median([cv_ratio(TrajectoryReplay, s) for s in (0, 1, 2)])
    mimic = np.median([cv_ratio(ThresholdHugging, s) for s in (0, 1, 2)])
    assert abs(replay - ceil) < abs(mimic - ceil), (
        f"replay should carry the victim's own burstiness: gap-CV ratio {replay:.2f} "
        f"(replay) vs {mimic:.2f} (mimicry), against a real block's {ceil:.2f}"
    )


def test_phase_alignment_lands_on_the_blocks_own_hour_and_weekday(base):
    """A replayed block carries a weekly rhythm in its gaps. Started on the wrong weekday
    the whole sequence lands on the wrong days, and `day_of_week` is a column the
    detector reads — so the start is aligned on both, and never lands before the takeover
    is allowed to happen."""
    from chhal.behaviour import day_of_week_of, hour_of
    from chhal.redteam.campaign import _phase_align

    from chhal.redteam.campaign import MAX_TAKEOVER_WAIT_DAYS

    rng = np.random.default_rng(1)
    window = MAX_TAKEOVER_WAIT_DAYS * 86_400
    for _ in range(300):
        earliest = int(rng.integers(0, 400 * 86_400))
        h, d = int(rng.integers(0, 24)), int(rng.integers(0, 7))
        t = _phase_align(earliest, earliest + window, h, d, rng)
        assert t >= earliest
        assert int(hour_of(np.array([t]))[0]) == h
        assert int(day_of_week_of(np.array([t]))[0]) == d
        # inside the window it was handed, not merely at the first matching instant
        assert t - earliest < window + 3_600


def test_replay_still_starts_after_the_last_real_transaction(base):
    """Phase alignment moves the start around; it must never move it into the victim's
    past. The rule that a campaign continues an account rather than reaching into it is
    the one every leakage argument rests on."""
    from chhal.redteam.vectors import TrajectoryReplay

    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    _, camp = TrajectoryReplay().calibrate(prof, _replay_pool(base)).render_with_timeline(
        400, np.random.default_rng(2))
    for h_ts, _, a_ts, _ in _campaigns(camp):
        if len(a_ts):
            assert a_ts.min() > h_ts.max()


def test_replay_waits_about_as_long_as_mimicry_does(base):
    """The probe compares the two vectors on the columns the RED TEAM controls, and
    `time_since_last_txn_min` is one of them — so the takeover wait has to be a property
    of the pair, not a difference between them.

    `_phase_align` can push a start most of a week past the moment it was handed, which
    is the price of landing on the block's own weekday. Drawn from the full
    `MAX_TAKEOVER_WAIT_DAYS` window that pushed replayed campaigns a systematic four days
    further from their victim's last real transaction than mimicked ones, and put 12% of
    them outside the documented month. `PHASE_ALIGN_SLACK_S` shortens the draw by exactly
    what the alignment can add back.
    """
    from chhal.redteam.campaign import MAX_TAKEOVER_WAIT_DAYS
    from chhal.redteam.vectors import ThresholdHugging, TrajectoryReplay

    prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    pool = _replay_pool(base)

    def waits(V):
        out = []
        for seed in (0, 1, 2):
            _, camp = V().calibrate(prof, pool).render_with_timeline(
                400, np.random.default_rng(seed))
            for h_ts, _, a_ts, _ in _campaigns(camp):
                if len(a_ts):
                    out.append((a_ts[0] - h_ts[-1]) / 86_400.0)
        return np.array(out)

    replay, mimic = waits(TrajectoryReplay), waits(ThresholdHugging)
    assert replay.min() > 0, "a campaign may not start before its victim's last real txn"
    # a day of slack over the nominal window: both paths snap forward onto an hour
    assert replay.max() < MAX_TAKEOVER_WAIT_DAYS + 1, (
        f"replay waited {replay.max():.1f} days, past the {MAX_TAKEOVER_WAIT_DAYS}-day "
        "window the vector documents"
    )
    # The MEAN, not the median. Aligned instants recur weekly, so a replayed wait is one
    # of about four values per victim; the median of something that discrete jumps between
    # atoms and says nothing. What has to match is the level, because that is what a
    # systematic offset in `time_since_last_txn_min` would look like.
    assert abs(replay.mean() - mimic.mean()) < 2.0, (
        f"replay waits a mean {replay.mean():.1f} days and mimicry {mimic.mean():.1f} — "
        "that is a controlled-column difference between two vectors that are supposed to "
        "differ only in where the timeline comes from"
    )


def test_the_probe_cannot_reach_the_shipped_suite():
    """`replay_host` was wired into `generate()` after every headline number was already
    measured, and the one thing that could not be allowed to happen is that wiring moving
    those numbers. It does not, because the shipped vectors never take the replay branch —
    but that is a property of their profiles, not of the code, so it is asserted here
    rather than left to whoever edits `vectors.py` next.

    A shipped vector that set `replay_host=True` would change its own results AND, by
    consuming the generator differently, every vector rendered after it in the same batch.
    """
    from chhal.redteam.vectors import TrajectoryReplay

    assert TrajectoryReplay not in ALL_VECTORS, (
        "TrajectoryReplay is a probe run against threshold_hugging as its control; in the "
        "suite it would be a second near-identical vector reweighting every aggregate"
    )
    offenders = [V.vector_id for V in ALL_VECTORS if V.temporal.replay_host]
    assert not offenders, (
        f"{offenders} take the replay path, so the headline numbers no longer come from "
        "the code that produced them"
    )


def test_the_pool_reports_how_often_mimicry_actually_engages(base):
    """`mimic_host` is the flagship vector's entire claim, and on short histories it does
    not happen: `_host_gaps` returns None below MIN_HISTORY_TO_MIMIC and the campaign
    quietly uses the population bands — the very thing the vector argues against.

    The fallback was always documented; the RATE was not, which is what made it a problem.
    A disguise that engages on fewer than one campaign in three is a different claim from
    one that engages always, so the rate is computed here and printed by `describe()` on
    every run rather than left for a reader to derive.
    """
    from chhal.redteam.campaign import MIN_HISTORY_TO_MIMIC

    pool = HostPool(base.train)
    rate = pool.mimicry_engagement()
    sizes = pool._ends - pool._starts
    assert rate == pytest.approx(float(np.mean(sizes >= MIN_HISTORY_TO_MIMIC)))
    assert 0.0 <= rate <= 1.0
    assert "mimicry engages on" in pool.describe(), (
        "the engagement rate has to be printed with the pool, not buried in a method "
        "nobody calls"
    )
