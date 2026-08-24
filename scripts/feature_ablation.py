"""Which features actually carry the real-fraud signal, and can we rebuild the ones we
cannot explain?

This is the evidence behind two decisions in the design, and it is here so both can be
re-checked rather than taken on trust.

**Why the feature space includes the linkage block.** Twelve hand-derived features catch
3.1% of real IEEE-CIS fraud at a 0.1% false-positive budget. Adding the dataset's
anonymised entity-linkage counts (C1-C14) takes that to 19.7% — a 6.4x lift that nothing
else approaches. All 339 V-features add two more points on top of it; D1-D15 add nothing.

**Why we could not simply build our own.** Those counts aggregate over devices, phones,
IPs and cross-card relationships the dataset does not expose. We tried to rebuild the
signal from what we do understand — distinct counterparties, addresses, emails and card
attributes per account over time, plus longer velocity windows — and got +0.45 points.
Adding them ON TOP of C1-C14 makes things slightly worse, so they are not merely weaker,
they are subsumed.

That is what forced the honest design: the red team does not invent linkage counts, it
inherits them by mounting each campaign on a real account (see chhal/redteam/hosts.py).

Real data only, no red team, no attacks. Temporal split, same cut as prepare_ieee.py.

    python scripts/feature_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chhal.behaviour import HOUR_OFFSET, derive, hour_of   # noqa: E402
from prepare_ieee import CHANNEL_MAP, DOMESTIC_ADDR2, _download  # noqa: E402

RAW = os.path.expanduser("~/chhal-data/raw/train_transaction.csv")
FPRS = (0.001, 0.005, 0.01)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    if not os.path.exists(RAW):
        _download(RAW)
    df = pd.read_csv(RAW, low_memory=False)
    if len(df) != 590_540 or df.isFraud.sum() != 20_663:
        raise SystemExit("not the genuine IEEE-CIS train_transaction.csv")
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    y = df.isFraud.to_numpy()

    day = (df.TransactionDT // 86_400).astype(np.int64)
    uid = pd.factorize(df.card1.astype(str) + "_" + df.addr1.astype(str) + "_"
                       + (day - df.D1.fillna(0).astype(np.int64)).astype(str))[0]
    ts = df.TransactionDT.to_numpy(np.int64)
    beh = derive(uid, ts, df.TransactionAmt.to_numpy(np.float64))
    is_tr = (df.TransactionDT <= df.TransactionDT.quantile(0.75)).to_numpy()
    print(f"[split] train={is_tr.sum():,}  test={(~is_tr).sum():,}")

    def target_encode(key, k=50.0, folds=5, seed=7):
        prior = y[is_tr].mean(); out = np.full(len(df), prior)
        def enc(fit, app):
            a = pd.DataFrame({"k": key[fit], "y": y[fit]}).groupby("k").y.agg(["sum", "count"])
            sm = (a["sum"] + prior * k) / (a["count"] + k)
            out[app] = pd.Series(key[app]).map(sm).fillna(prior).to_numpy()
        rng = np.random.default_rng(seed); ti = np.flatnonzero(is_tr)
        f = rng.integers(0, folds, len(ti))
        for i in range(folds):
            fit = np.zeros(len(df), bool); fit[ti[f != i]] = True
            app = np.zeros(len(df), bool); app[ti[f == i]] = True
            enc(fit, app)
        enc(is_tr, ~is_tr)
        return out

    counterparty = (df.ProductCD.astype(str) + "|" + df.R_emaildomain.fillna("NA")).to_numpy()
    BASE = pd.DataFrame({
        "amount": df.TransactionAmt, "hour": hour_of(ts, HOUR_OFFSET),
        "day_of_week": (ts // 86_400) % 7,
        "velocity_1h": beh.velocity_1h, "velocity_24h": beh.velocity_24h,
        "amount_to_avg_ratio": beh.amount_to_avg_ratio,
        "account_age_days": df.D1.fillna(0),
        "time_since_last_txn_min": beh.time_since_last_txn_min,
        "is_new_beneficiary": (~pd.Series(pd.Series(uid).astype(str) + "|" + counterparty)
                               .duplicated()).astype(int),
        "is_cross_border": (df.addr2.notna() & (df.addr2 != DOMESTIC_ADDR2)).astype(int),
        "channel_code": df.ProductCD.map(CHANNEL_MAP).fillna(2).astype(int),
        "merchant_risk": target_encode(counterparty),
    })

    def expanding_nunique(g, k):
        first = ~pd.DataFrame({"g": g, "k": k}).duplicated(["g", "k"])
        return (first.groupby(pd.Series(g)).cumsum() - first.astype(int)).to_numpy()

    def window_count(g, t, w):
        order = np.lexsort((t, g)); gs, tss = g[order], t[order]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]); out = np.empty(len(gs), np.int64)
        for s, e in zip(starts, np.r_[starts[1:], len(gs)]):
            seg = tss[s:e]
            out[s:e] = np.arange(e - s) - np.searchsorted(seg, seg - w, "left")
        inv = np.empty_like(order); inv[order] = np.arange(len(order)); return out[inv]

    gs, amt = pd.Series(uid), df.TransactionAmt
    OURS = pd.DataFrame({
        "n_distinct_counterparties": expanding_nunique(uid, counterparty),
        "n_distinct_addr": expanding_nunique(uid, df.addr1.fillna(-1).to_numpy()),
        "n_distinct_email": expanding_nunique(uid, df.P_emaildomain.fillna("NA").to_numpy()),
        "n_distinct_card_attr": expanding_nunique(uid, df.card2.fillna(-1).to_numpy()),
        "txns_so_far": gs.groupby(gs).cumcount().to_numpy(),
        "velocity_7d": window_count(uid, ts, 604_800),
        "velocity_30d": window_count(uid, ts, 2_592_000),
        "amount_max_ratio_so_far": (amt / amt.groupby(gs).cummax().shift().fillna(amt)).to_numpy(),
    })
    C = df[[f"C{i}" for i in range(1, 15)]].fillna(-1)
    D = df[[f"D{i}" for i in range(1, 16)]].fillna(-1)
    V = df[[c for c in df.columns if c.startswith("V")]].fillna(-1).astype(np.float32)

    TIERS = {
        "the 12 we hand-derived": BASE,
        "+ linkage counts we built ourselves": pd.concat([BASE, OURS], axis=1),
        "+ the dataset's C1-C14": pd.concat([BASE, C], axis=1),
        "+ both": pd.concat([BASE, OURS, C], axis=1),
        "+ C1-C14 and D1-D15": pd.concat([BASE, C, D], axis=1),
        "+ everything incl. 339 V (ceiling)": pd.concat([BASE, OURS, C, D, V], axis=1),
    }

    rows = []
    for name, X in TIERS.items():
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=48,
                           subsample=0.8, colsample_bytree=0.8, random_state=7,
                           n_jobs=-1, verbose=-1)
        m.fit(X[is_tr].to_numpy(np.float32), y[is_tr])
        p = m.predict_proba(X[~is_tr].to_numpy(np.float32))[:, 1]
        yt = y[~is_tr]; legit = p[yt == 0]
        rows.append({"features": name, "n": X.shape[1],
                     **{f"recall@{f}": round(float((p[yt == 1] >= np.quantile(legit, 1 - f)).mean()), 4)
                        for f in FPRS},
                     "pr_auc": round(float(average_precision_score(yt, p)), 4)})
        print("  ", rows[-1])

    out = pd.DataFrame(rows)
    print("\n=== recall on REAL IEEE-CIS fraud, at a fixed share of real legit traffic flagged ===")
    print(out.to_string(index=False))
    print("\nThe linkage block is the whole story. The counts we can build ourselves recover")
    print("almost none of it, and stacking ours on top of theirs is a wash — subsumed, not")
    print("merely weaker. That is why the red team inherits linkage from a real account")
    print("instead of inventing it: the signal is not reconstructible from what we can see.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "feature_ablation.json").write_text(json.dumps(rows, indent=2))
    out.to_csv(RESULTS / "feature_ablation.csv", index=False)
    print(f"-> {RESULTS/'feature_ablation.json'}")


if __name__ == "__main__":
    main()
