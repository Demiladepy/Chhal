"""The tests that were missing when it mattered.

A mutation run over this project killed 8 of 17 mutants. Every survivor was a
LEAKAGE-DISCIPLINE mutation, benchmark rows injected into training, the held-out
slice collapsed onto the train slice, the train/test account exclusion deleted, the
temporal split swapped for a random one. The suite was strong on leaf primitives and
blind on the wiring, which is precisely where the headline claims live.

The reason those mutants survived is worth stating plainly, because it generalises:
**leaking makes the numbers go UP**, so a test that asserts "recall improved" passes
harder the more the run cheats. Nothing here asserts that a number is good. Everything
here asserts that a number was earned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN                  # noqa: E402
from chhal.data import load_base_data                                     # noqa: E402
from chhal.detector import Detector                                       # noqa: E402
from chhal.evaluation import campaign_ids, host_ids, split_attacks       # noqa: E402
from chhal.loop import LoopConfig, run_loop                               # noqa: E402
from chhal.redteam import ALL_VECTORS                                     # noqa: E402
from chhal.redteam.base import BaseProfile                                # noqa: E402
from chhal.redteam.hosts import HostPool                                  # noqa: E402

SMALL = dict(source="synthetic", n_legit=4000, n_baseline_fraud=100)


@pytest.fixture(scope="module")
def loop_result():
    return run_loop(
        LoopConfig(iterations=2, attacks_per_vector=150, benchmark_per_vector=150, seed=3),
        base=load_base_data(seed=3, **SMALL),
    )


def test_not_one_benchmark_row_reaches_the_training_pool(loop_result):
    """The headline is benchmark recall. If a benchmark row can be trained on, the
    headline measures memory. Mutating the loop to inject them left every other test
    green, recall simply rose."""
    audit = loop_result.leakage_audit
    assert audit["benchmark_rows"] > 0, "nothing was benchmarked; the audit is vacuous"
    assert audit["benchmark_rows_in_training_pool"] == 0, audit


def test_the_pressure_slice_is_not_also_the_training_slice(loop_result):
    """`split_attacks` returns (train, heldout). Mutating it to return the same frame
    twice was invisible to the suite: nothing compared the two."""
    audit = loop_result.leakage_audit
    assert audit["pressure_rows_in_training_pool"] == 0, audit


def test_benchmark_attacks_compromise_accounts_the_detector_never_trained_on(loop_result):
    """Sixteen of twenty-six features are inherited from the compromised account. An
    evaluation attack mounted on a TRAIN account carries issuer-side context the
    detector has already seen, which is leakage wearing a red team's clothes."""
    audit = loop_result.leakage_audit
    assert audit["benchmark_host_accounts"] > 0
    assert audit["benchmark_host_accounts_seen_in_train"] == 0, audit


def test_the_exclusion_is_what_keeps_the_pools_apart_not_luck():
    """Directly: the same construction WITHOUT `exclude_accounts` does overlap. If this
    ever stops being true the test above has become a tautology and should be deleted."""
    base = load_base_data(seed=3, **SMALL)
    train_accounts = set(base.train["_account"].unique())
    guarded = HostPool(base.test, exclude_accounts=base.train["_account"])
    unguarded = HostPool(base.test)
    assert not (set(guarded.accounts) & train_accounts)
    assert set(unguarded.accounts) & train_accounts, (
        "the splits do not share accounts at all, so the exclusion proves nothing here")


def test_a_campaign_never_straddles_the_train_heldout_boundary():
    """Every row of a campaign shares one host's age, merchant history and fourteen
    linkage counts. Splitting rows at random put 98.1% of the "never trained on" rows
    in an account the detector had just learned."""
    base = load_base_data(seed=4, **SMALL)
    rng = np.random.default_rng(4)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    hosts = HostPool(base.train)
    batches = [V().calibrate(profile, hosts).batch(200, 1, rng) for V in ALL_VECTORS]

    tr, ho, ho_vec = split_attacks(batches, 0.4, rng)
    assert len(tr) and len(ho) and len(ho) == len(ho_vec)

    # rebuild the campaign label for each side by matching rows back to their batch
    tr_keys = set(map(tuple, np.round(tr[FEATURE_COLUMNS].to_numpy(float), 9)))
    ho_keys = set(map(tuple, np.round(ho[FEATURE_COLUMNS].to_numpy(float), 9)))
    assert not (tr_keys & ho_keys), "a row is on both sides of the split"

    for b in batches:
        ent = campaign_ids(b)
        assert ent is not None, f"{b.vector_id} lost its timeline; the split cannot be honest"
        rows = np.round(b.transactions[FEATURE_COLUMNS].to_numpy(float), 9)
        side = {}
        for e, row in zip(ent, map(tuple, rows)):
            if row in tr_keys:
                side.setdefault(e, set()).add("train")
            if row in ho_keys:
                side.setdefault(e, set()).add("heldout")
        straddling = [e for e, s in side.items() if len(s) > 1]
        assert not straddling, (
            f"{b.vector_id}: {len(straddling)} campaigns have rows on both sides")


def test_a_host_account_never_straddles_the_train_heldout_boundary():
    """The campaign test above passes even when the split keys on the campaign, because
    host-disjointness implies campaign-disjointness but not the reverse. Hosts are drawn
    WITH replacement, so two campaigns can land on one account; keying the split on the
    campaign then puts that account on both sides and the sixteen inherited features go
    with it. Mutating `split_attacks` back to `campaign_ids` must fail here.
    """
    base = load_base_data(seed=11, **SMALL)
    rng = np.random.default_rng(11)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    hosts = HostPool(base.train)
    batches = [V().calibrate(profile, hosts).batch(200, 1, rng) for V in ALL_VECTORS]

    tr, ho, _ = split_attacks(batches, 0.4, rng)
    tr_keys = set(map(tuple, np.round(tr[FEATURE_COLUMNS].to_numpy(float), 9)))
    ho_keys = set(map(tuple, np.round(ho[FEATURE_COLUMNS].to_numpy(float), 9)))

    shared_seen = 0
    for b in batches:
        acc = host_ids(b)
        assert acc is not None, f"{b.vector_id} lost its timeline; the split cannot be honest"
        camp = campaign_ids(b)
        rows = np.round(b.transactions[FEATURE_COLUMNS].to_numpy(float), 9)
        side, campaigns_per_host = {}, {}
        for a, c, row in zip(acc, camp, map(tuple, rows)):
            campaigns_per_host.setdefault(a, set()).add(c)
            if row in tr_keys:
                side.setdefault(a, set()).add("train")
            if row in ho_keys:
                side.setdefault(a, set()).add("heldout")
        straddling = [a for a, s in side.items() if len(s) > 1]
        assert not straddling, (
            f"{b.vector_id}: {len(straddling)} host accounts have rows on both sides")
        shared_seen += sum(1 for v in campaigns_per_host.values() if len(v) > 1)

    assert shared_seen > 0, (
        "no host carried two campaigns in this run, so the assertion above cannot tell a "
        "host-keyed split from a campaign-keyed one. The test proves nothing; raise the "
        "batch size or reseed"
    )


def test_a_single_campaign_cannot_be_split_and_does_not_pretend_to_be():
    """The degenerate case, made explicit: one campaign goes to train whole. Faking a
    holdout out of its own rows is exactly the thing this split exists to stop."""
    base = load_base_data(seed=6, **SMALL)
    rng = np.random.default_rng(6)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    batch = ALL_VECTORS[0]().calibrate(profile, HostPool(base.train)).batch(3, 1, rng)
    ent = campaign_ids(batch)
    if len(set(ent)) != 1:
        pytest.skip("this batch spans more than one campaign")
    tr, ho, _ = split_attacks([batch], 0.4, rng)
    assert len(ho) == 0 and len(tr) == len(batch)


def test_the_detector_is_deterministic_at_a_fixed_seed():
    """`random_state` could be deleted and nothing noticed. Every paired comparison in
    this project. The ablation, the ensemble check, the arms-race curve, reads a
    difference between two runs and would silently be reading noise instead."""
    base = load_base_data(seed=2, **SMALL)
    X = base.test[FEATURE_COLUMNS]
    a = Detector(seed=11).fit(base.train, LABEL_COLUMN).score(X)
    b = Detector(seed=11).fit(base.train, LABEL_COLUMN).score(X)
    assert np.array_equal(a, b), "same seed, different scores"
    c = Detector(seed=12).fit(base.train, LABEL_COLUMN).score(X)
    assert not np.array_equal(a, c), (
        "different seeds give identical scores. The seed is not wired to anything")


def test_the_base_split_is_temporal_not_random():
    """`prepare_ieee.py` has no test of its own, and swapping its temporal split for a
    random one survived mutation. A random split lets the detector learn from the
    future, which is the single most flattering mistake in fraud modelling."""
    base = load_base_data(seed=1, **SMALL)
    assert base.train["_ts"].max() <= base.test["_ts"].min(), (
        "train and test overlap in time")


def test_integer_features_are_integers_by_dtype_not_by_luck():
    """The old assertion was `x % 1 == 0`, which a float column satisfies. Dropping the
    integer coercion left it green."""
    from chhal.contract import INTEGER_FEATURES
    base = load_base_data(seed=8, **SMALL)
    rng = np.random.default_rng(8)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    hosts = HostPool(base.train)
    for V in ALL_VECTORS:
        rows = V().calibrate(profile, hosts).batch(20, 1, rng).transactions
        for col in INTEGER_FEATURES:
            assert pd.api.types.is_integer_dtype(rows[col]), (
                f"{V.vector_id}.{col} is {rows[col].dtype}, not an integer type")


def _causal_encode():
    """`scripts/` is not a package; import the encoder the way the scripts do."""
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prepare_ieee", root / "scripts" / "prepare_ieee.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_ieee"] = mod
    spec.loader.exec_module(mod)
    return mod.causal_target_encode


def test_merchant_risk_encoding_cannot_see_the_future():
    """A random K-fold target encoding stops a row encoding *itself*, which is the leak
    everyone checks for, and leaves a second one open: on temporally ordered data the
    other folds span the whole train window, so a January row is scored with outcomes
    from June. The encoding must depend on strictly earlier rows only.

    Tested by mutation rather than by inspection: flip a label at the end of time and
    every earlier encoding must come back bit-identical."""
    encode = _causal_encode()
    n = 400
    key = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    ts = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.3).astype(np.float64)
    is_train = np.ones(n, bool)

    # the shrinkage target is a global scalar and is documented as non-causal, so pin it
    # and test the thing that actually carries per-row information: the bucket history.
    before = encode(key, y, is_train, ts, prior=0.3)

    y2 = y.copy()
    last_a = np.flatnonzero(key == "A")[-1]
    y2[last_a] = 1.0 - y2[last_a]          # change the future
    after = encode(key, y2, is_train, ts, prior=0.3)

    earlier = ts < ts[last_a]
    assert np.array_equal(before[earlier], after[earlier]), (
        "changing a later outcome moved an earlier row's encoding. The feature is "
        "reading the future")


def test_merchant_risk_encoding_excludes_the_whole_simultaneous_block():
    """Excluding only the row itself is not enough when several transactions in the same
    bucket share a timestamp: they would encode each other, which is the same leak at a
    smaller radius."""
    encode = _causal_encode()
    # one bucket, four rows, all at t=0, all fraudulent; nothing precedes them
    key = np.array(["A"] * 4)
    ts = np.zeros(4)
    y = np.ones(4)
    is_train = np.ones(4, bool)
    out = encode(key, y, is_train, ts, smoothing=50.0, prior=0.1)
    assert np.allclose(out, 0.1), (
        f"simultaneous rows encoded each other: {out} != prior 0.1")


def test_merchant_risk_encoding_does_accumulate_history():
    """The causality guard must not have flattened the feature into a constant."""
    encode = _causal_encode()
    n = 200
    key = np.array(["A"] * n)
    ts = np.arange(n, dtype=np.float64)
    y = np.ones(n)                          # a bucket that is always fraudulent
    out = encode(key, y, np.ones(n, bool), ts, smoothing=5.0, prior=0.05)
    assert out[0] < out[-1], "encoding never moves. No history is accumulating"
    assert np.all(np.diff(out) >= -1e-12), "encoding is not monotone in a constant bucket"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
