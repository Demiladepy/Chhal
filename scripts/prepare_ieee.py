"""Derive Chhal's frozen feature space from the REAL IEEE-CIS fraud dataset.

Why this file exists
--------------------
"Fidelity of attacks in simulation" is a judged criterion, and it is judged against
*real payment data*. A KS distance measured against a distribution we invented
ourselves proves nothing. This script replaces the invented base distribution with
590,540 real card transactions (Vesta / IEEE-CIS, 3.499% fraud, 182 days), derived
into the exact same FEATURE_COLUMNS the detector and the red team already share, so
nothing downstream changes.

Source
------
IEEE-CIS Fraud Detection (Vesta Corporation), the standard public benchmark for card
fraud. Mirrored ungated at huggingface.co/datasets/aliceczr/ieee-fraud-detection.
Verified on load: 590,540 rows, 20,663 frauds (3.4990%).

The account identity problem
----------------------------
IEEE-CIS has no account column. The community-standard reconstruction is

    uid = card1 + addr1 + (transaction_day - D1)

where D1 is "days since this card first transacted", so (day - D1) is the card's
first-seen day and the triple is stable for one card. It yields 217,850 entities over
590,540 transactions (mean 2.71, median 1, p99 20). Every behavioural feature below
-- velocity, time-since-last, amount-to-average -- is computed *within* a uid over the
real time ordering, using only rows strictly before the row being scored. That is what
makes them behavioural rather than sampled.

Honest approximations, all documented in the write-up
-----------------------------------------------------
* hour        - TransactionDT is seconds from an undisclosed epoch. The empirical
                diurnal minimum sits at raw hour ~8, so we shift by HOUR_OFFSET to put
                the trough near 04:00 local. The *shape* is real; only the clock label
                is aligned.
* channel_code- IEEE-CIS is US e-commerce, not Indian rails. ProductCD is the real
                product/channel split (W 74.5% / C 11.6% / R,H,S 13.9%) and carries
                genuine signal (fraud rate 2.0% on W vs 11.7% on C). We map it to the
                three channel slots rather than pretend UPI data exists here.
* is_new_beneficiary - no merchant column exists. We use first-ever occurrence of the
                (uid, ProductCD, R_emaildomain) counterparty pair in time order.
* merchant_risk - no merchant column exists. Smoothed historical fraud rate of the
                (ProductCD, R_emaildomain) bucket, fit on the TRAIN split only and
                out-of-fold within train. This mirrors what an issuer actually has: a
                merchant risk score built from past outcomes, never from the present
                transaction.

Split
-----
TEMPORAL, not random: the first 75% of the 182-day window trains, the last 25% tests.
Random splits leak future fraud patterns into the past and inflate every number. The
manifold quantiles used by the evasion optimizer are computed on TRAIN ONLY.

Two further guards sit on top of the temporal cut, and both were added because the cut
alone measurably was not enough. A 7-day DELAY PERIOD separates the splits -- the gap
used to be sixty seconds, so the detector was being credited with catching attacks it
had effectively already been told about. And test rows on accounts that also appear in
train are PURGED: 42.2% of the test split sat on entities the model had memorised, and
every behavioural feature here is computed within an entity. Both sets of rows keep a
`split` label of their own (`embargo`, `straddle`) rather than being deleted, so the
count is auditable from the parquet and the leakage delta stays reportable.

Usage
-----
    python scripts/prepare_ieee.py

Downloads the raw transactions (~683MB) on first run if they are not already present,
derives the features in a few seconds, and writes an 18.6MB parquet. Re-running is a
no-op once that parquet exists; pass --force to rebuild.

Keep the raw CSV if you intend to run `scripts/feature_ablation.py`, which reads the
339 V-columns the parquet does not carry and will otherwise re-download the whole file.
Everything else in the project needs only the parquet.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chhal.behaviour import derive, hour_of  # noqa: E402
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN, LINKAGE_FEATURES  # noqa: E402

# Expected shape of the genuine dataset — we refuse to proceed on anything else.
EXPECTED_ROWS = 590_540
EXPECTED_FRAUDS = 20_663

DOMESTIC_ADDR2 = 87.0     # 88.1% of rows; everything else is cross-border
TE_SMOOTHING = 50.0       # Bayesian prior weight for merchant-risk target encoding
TE_FOLDS = 5
TEST_FRAC = 0.25
EMBARGO_DAYS = 7          # delay period between the splits — see the split block below

RAW_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "addr1", "addr2", "R_emaildomain", "D1",
] + [f"C{i}" for i in range(1, 15)]

# Helper columns kept beside the feature space, never part of it. They are what lets the
# red team mount a campaign on a real account: which account a transaction belongs to,
# and when it happened.
ACCOUNT_COLUMN = "_account"
TIME_COLUMN = "_ts"
CHANNEL_MAP = {"W": 0, "C": 1, "R": 2, "H": 2, "S": 2}


SOURCE_URL = ("https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/"
              "resolve/main/train_transaction.csv")


def _download(path: str) -> None:
    """Fetch the raw transactions. Written to a temp file and moved into place, so an
    interrupted download can never be mistaken for a complete one on the next run."""
    import shutil
    import urllib.request

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    print(f"[fetch ] {SOURCE_URL}\n         -> {path} (~683MB, one time)")
    try:
        r = urllib.request.urlopen(SOURCE_URL)
    except Exception as e:
        raise SystemExit(
            f"Could not download the raw IEEE-CIS transactions ({e}).\n"
            f"This step needs network access once. If you already have "
            f"train_transaction.csv, put it at {path} and rerun.") from None
    with r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r         {100*done/total:5.1f}%  {done/1e6:7.1f} / {total/1e6:.0f} MB",
                      end="", flush=True)
    print()
    shutil.move(tmp, path)


def _load_raw(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "train_transaction.csv")
    if not os.path.exists(path):
        _download(path)
    df = pd.read_csv(path, usecols=RAW_COLUMNS)
    n, f = len(df), int(df.isFraud.sum())
    if n != EXPECTED_ROWS or f != EXPECTED_FRAUDS:
        raise SystemExit(
            f"Refusing to proceed: {path} has {n} rows / {f} frauds, expected "
            f"{EXPECTED_ROWS} / {EXPECTED_FRAUDS}. This is not the genuine IEEE-CIS "
            f"train_transaction.csv."
        )
    print(f"[verify] {n:,} rows, {f:,} frauds ({100*f/n:.4f}%) — genuine IEEE-CIS")
    return df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)


def _behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity, recency and amount-ratio computed within each uid over real time.

    The arithmetic lives in `chhal.behaviour`, which the red team also calls on its
    generated campaigns. One implementation, both sides — so whatever relationships hold
    between these four features in real data hold in the attacks too, and neither can
    drift away from the other.
    """
    day = (df.TransactionDT // 86_400).astype(np.int64)
    first_seen_day = day - df.D1.fillna(0).astype(np.int64)
    uid = (df.card1.astype(str) + "_" + df.addr1.astype(str) + "_"
           + first_seen_day.astype(str))
    uid_code = pd.factorize(uid)[0]
    out = derive(uid_code, df.TransactionDT.to_numpy(np.int64),
                 df.TransactionAmt.to_numpy(np.float64))
    out["_uid"] = uid_code
    return out


def _merchant_risk(df: pd.DataFrame, is_train: np.ndarray, seed: int) -> np.ndarray:
    """Smoothed historical fraud rate per (ProductCD, R_emaildomain) bucket.

    Fit on TRAIN ONLY. Within train it is computed out-of-fold, so no row contributes
    to the encoding it is scored with — otherwise the detector would be reading the
    label through the feature.
    """
    key = (df.ProductCD.astype(str) + "|" + df.R_emaildomain.fillna("NA").astype(str)).to_numpy()
    y = df[LABEL_COLUMN].to_numpy(np.float64)
    prior = y[is_train].mean()
    out = np.full(len(df), prior, np.float64)

    def encode(fit_mask: np.ndarray, apply_mask: np.ndarray) -> None:
        agg = pd.DataFrame({"k": key[fit_mask], "y": y[fit_mask]}).groupby("k").y.agg(["sum", "count"])
        sm = (agg["sum"] + prior * TE_SMOOTHING) / (agg["count"] + TE_SMOOTHING)
        out[apply_mask] = pd.Series(key[apply_mask]).map(sm).fillna(prior).to_numpy()

    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(is_train)
    folds = rng.integers(0, TE_FOLDS, len(train_idx))
    for f in range(TE_FOLDS):                       # out-of-fold within train
        val = train_idx[folds == f]
        fit = np.zeros(len(df), bool); fit[train_idx[folds != f]] = True
        apply = np.zeros(len(df), bool); apply[val] = True
        encode(fit, apply)
    encode(is_train, ~is_train)                     # test scored with the full train fit
    return out


def build(raw_dir: str, out_path: str, seed: int = 7, force: bool = False) -> pd.DataFrame:
    """Derive the prepared parquet from the raw IEEE-CIS transactions.

    Idempotent: a usable parquet already at `out_path` is left alone. Re-running this
    used to re-download 683MB and redo the whole derivation even when nothing had
    changed, which is a slow way to find out you already had the file. `--force`
    rebuilds anyway.
    """
    if not force and os.path.exists(out_path):
        try:
            existing = pd.read_parquet(out_path)
        except Exception as e:                        # corrupt/partial file -> rebuild
            print(f"[reuse ] {out_path} unreadable ({e}); rebuilding")
        else:
            # The `embargo` / `straddle` labels are the marker of a post-fix build.
            # Without this clause a parquet derived before the delay period and the
            # entity purge existed would load silently, and every number downstream
            # would carry the leak while the code looked correct.
            ok = (len(existing) == EXPECTED_ROWS
                  and set(FEATURE_COLUMNS) <= set(existing.columns)
                  and "split" in existing.columns
                  and {"embargo", "straddle"} <= set(existing["split"].unique()))
            if ok:
                print(f"[reuse ] {out_path} already holds {len(existing):,} prepared rows. "
                      f"Nothing to do (pass --force to rebuild).")
                return existing
            print(f"[reuse ] {out_path} exists but does not match the expected shape; "
                  f"rebuilding")
    raw = _load_raw(raw_dir)
    beh = _behavioural_features(raw)

    hour = hour_of(raw.TransactionDT.to_numpy(np.int64))
    out = pd.DataFrame({
        "amount": raw.TransactionAmt.astype(np.float64),
        "hour": hour.astype(np.int64),
        "day_of_week": ((raw.TransactionDT // 86_400) % 7).astype(np.int64),
        "velocity_1h": beh.velocity_1h,
        "velocity_24h": beh.velocity_24h,
        "amount_to_avg_ratio": beh.amount_to_avg_ratio,
        "account_age_days": raw.D1.fillna(0.0).astype(np.float64),
        "time_since_last_txn_min": beh.time_since_last_txn_min,
        "is_cross_border": (raw.addr2.notna() & (raw.addr2 != DOMESTIC_ADDR2)).astype(np.int64),
        "channel_code": raw.ProductCD.map(CHANNEL_MAP).fillna(2).astype(np.int64),
    })

    # first-ever (uid, counterparty) pair, in true time order
    pair = (beh["_uid"].astype(str) + "|" + raw.ProductCD.astype(str) + "|"
            + raw.R_emaildomain.fillna("NA").astype(str))
    out["is_new_beneficiary"] = (~pair.duplicated()).astype(np.int64)

    out[LABEL_COLUMN] = raw.isFraud.astype(np.int64)

    # TEMPORAL split — train on the past, test on the future, with a gap between them.
    #
    # There used to be no gap at all. Measured on the previous build: train `_ts` max
    # 11,246,605, test min 11,246,665 — SIXTY SECONDS apart. Every headline number was
    # therefore the recall of a detector that learns each attack one minute after it
    # happens, which is not a thing any issuer can do: a card fraud label arrives when
    # the cardholder disputes the charge, days later. The Fraud Detection Handbook calls
    # the gap the delay period; rows inside it are dropped from BOTH sides rather than
    # handed to either, because they are exactly the rows whose labels would not be
    # known yet at the moment the test period begins.
    cut_dt = raw.TransactionDT.quantile(1 - TEST_FRAC)
    embargo_end = cut_dt + EMBARGO_DAYS * 86_400
    is_train = (raw.TransactionDT <= cut_dt).to_numpy()
    is_test = (raw.TransactionDT > embargo_end).to_numpy()
    split = np.where(is_train, "train", np.where(is_test, "test", "embargo"))

    # ENTITY LEAKAGE — accounts that sit on both sides of the cut.
    #
    # Measured on the previous build: 23,688 accounts appeared in train AND test, which
    # is 62,245 of 147,635 test rows — 42.2% of the test split sat on accounts the
    # detector had already trained on. A temporal split alone does not prevent this,
    # because an account that transacts across the cut lands in both halves.
    #
    # It matters more here than it would elsewhere. Every behavioural feature is computed
    # WITHIN a uid over real time, `account_age_days` and `merchant_risk` are properties
    # of the entity, and the fourteen linkage counts are the dataset's own per-entity
    # aggregates. A test row on a straddling account is not an unseen customer; it is a
    # customer the model has memorised, scored a little later. Those rows are moved out
    # of test.
    #
    # Report both the purged and unpurged numbers. The delta between them IS the leakage
    # measurement, so this is a result to publish, not only a fix to make quietly.
    train_accounts = pd.unique(beh["_uid"].to_numpy()[is_train])
    straddles = is_test & beh["_uid"].isin(train_accounts).to_numpy()
    n_straddle_accounts = int(pd.unique(beh["_uid"].to_numpy()[straddles]).size)
    n_test_pre_purge = int(is_test.sum())
    split = np.where(straddles, "straddle", split)
    out["split"] = split

    out["merchant_risk"] = _merchant_risk(
        pd.concat([raw[["ProductCD", "R_emaildomain"]], out[[LABEL_COLUMN]]], axis=1),
        is_train, seed,
    )

    # The dataset's own anonymised entity-linkage counts, carried through unchanged.
    # These are the columns the red team inherits rather than invents — see
    # contract.LINKAGE_FEATURES for why, and for what happened when we tried to rebuild
    # the signal from features we understand.
    for i, col in enumerate(LINKAGE_FEATURES, start=1):
        out[col] = raw[f"C{i}"].fillna(0.0).astype(np.float64)

    # helper columns: which account, and when. Not features; the host pool needs them.
    out[ACCOUNT_COLUMN] = beh["_uid"].astype(np.int64)
    out[TIME_COLUMN] = raw.TransactionDT.astype(np.int64)

    out = out[FEATURE_COLUMNS + [LABEL_COLUMN, "split", ACCOUNT_COLUMN, TIME_COLUMN]]
    if out.isna().any().any():
        raise SystemExit(f"NaNs produced in: {out.columns[out.isna().any()].tolist()}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_parquet(out_path, index=False)

    tr, te = out[out["split"] == "train"], out[out["split"] == "test"]
    n_embargo = int((out["split"] == "embargo").sum())
    n_straddle = int((out["split"] == "straddle").sum())
    print(f"[split ] temporal cut at TransactionDT={int(cut_dt):,}  "
          f"train={len(tr):,} ({tr[LABEL_COLUMN].mean()*100:.3f}% fraud)  "
          f"test={len(te):,} ({te[LABEL_COLUMN].mean()*100:.3f}% fraud)")
    print(f"[delay ] {EMBARGO_DAYS}-day delay period holds out {n_embargo:,} rows "
          f"between the splits (the gap used to be 60 seconds)")
    print(f"[leak  ] {n_straddle:,} of {n_test_pre_purge:,} post-embargo test rows "
          f"({100*n_straddle/max(n_test_pre_purge, 1):.1f}%) sat on "
          f"{n_straddle_accounts:,} accounts the detector also trains on — purged")
    n_acct = out[ACCOUNT_COLUMN].nunique()
    clean = out.groupby(ACCOUNT_COLUMN)[LABEL_COLUMN].max()
    print(f"[hosts ] {n_acct:,} accounts, {int((clean == 0).sum()):,} of them never "
          f"fraudulent — those are the ones a campaign may be mounted on")
    print(f"[write ] {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB, "
          f"{len(FEATURE_COLUMNS)} features)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/chhal-data/raw"))
    ap.add_argument("--out", default=os.path.expanduser("~/chhal-data/ieee_base.parquet"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the prepared parquet already exists")
    a = ap.parse_args()
    build(a.raw, a.out, a.seed, a.force)
