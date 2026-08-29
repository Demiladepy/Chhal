"""Why does every vector score 0.00%, and why is the loudest one worst?

Entry point: `card_testing`.

`card_testing` is the most separable thing Chhal generates. It sits KS 0.483 from
legitimate traffic on the columns the red team controls — 43x the sampling-noise floor —
and a classifier tells it apart from any other vector at AUC 1.000. It is also caught
**0.00%** of the time at a 0.1% false-positive budget, before the optimizer runs and after
it, and a detector trained on the other five vectors reaches it only **12.2%** of the time.

Loud, distinctive, and invisible. That combination is the whole paper in one vector, and
until it is explained it is an anecdote. This script is the explanation.

The hypothesis this script was written to test: these rows are not *evading* the detector,
they are landing where it has no training data. A gradient-boosted tree does not
extrapolate; push a row past the last split point and it receives whatever the leaf on that
side happens to hold.

**That hypothesis is wrong, and experiment D kills it.** Replace all ten columns the red
team controls with values drawn from real fraud and recall stays at 0.00%. The off-support
excursion is real — 72% of `card_testing` rows are past the training p99.9 on
`velocity_1h` — and it is not what is holding the score down.

Experiment E finds what is. Every campaign is mounted on a real account selected for never
having been fraudulent, and inherits that account's issuer-side context whole:
`account_age_days`, `merchant_risk` and the fourteen linkage counts. That block is where
almost all the real-fraud signal lives (2.25% -> 15.46% recall, `feature_ablation.py`).
Transplant it from real fraud and detectability comes back immediately.

The host-selection rule exists for anti-leakage reasons and is described in the README as
belt and braces — hosts are all-legitimate, so recognising one pushes an attack toward
*legit* and makes detection harder rather than easier. It is not belt and braces. It is the
entire explanation for the 0.00%, and therefore for the left-hand end of the arms-race
curve.

    A  where the scores actually land, against legit and against real fraud
    B  how far outside the training support each controlled feature is
    C  what the trees hold in the leaves these rows reach
    D  single-feature repair — replace ONE controlled column with a legitimate draw
    E  block transplant — swap the controlled block, then the inherited block, for real
       fraud's own values, across all six vectors
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chhal.contract import (FEATURE_COLUMNS, INHERITED_FEATURES,   # noqa: E402
                            LABEL_COLUMN)
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.evaluation import threshold_for_fpr                    # noqa: E402
from chhal.fidelity import CONTROLLED_FEATURES                    # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam import ALL_VECTORS                             # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402

N = 1000
SEED = 7
FPR = 0.001
TARGET = "card_testing"
FOIL = "threshold_hugging"    # the quiet vector, for contrast


def pct(x, qs=(1, 25, 50, 75, 99)):
    return "  ".join(f"p{q}={np.percentile(x, q):.4f}" for q in qs)


def main() -> None:
    base = load_base_data(source="ieee")
    rng = np.random.default_rng(SEED)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    hosts = HostPool(base.test, exclude_accounts=base.train["_account"])
    det = Detector(seed=SEED).fit(base.train, LABEL_COLUMN)
    opt = EvasionOptimizer(base.feature_stats)

    train = base.train
    legit = base.test[base.test[LABEL_COLUMN] == 0]
    fraud = base.test[base.test[LABEL_COLUMN] == 1]
    thr = threshold_for_fpr(det.score(legit[FEATURE_COLUMNS]), FPR)

    batches = {}
    for V in ALL_VECTORS:
        v = V().calibrate(profile, hosts)
        batches[V.vector_id] = opt.optimize(v.batch(N, 0, rng), det, rng).transactions
    tgt, foil = batches[TARGET], batches[FOIL]

    # ------------------------------------------------------------------ A
    print("=== A. where the scores land ===")
    print(f"threshold at {FPR:.1%} FPR = {thr:.6f}\n")
    for name, X in (("real legit", legit[FEATURE_COLUMNS]),
                    ("real fraud", fraud[FEATURE_COLUMNS]),
                    (FOIL, foil), (TARGET, tgt)):
        s = det.score(X)
        print(f"  {name:20s} {pct(s)}   >= thr: {(s >= thr).mean():6.2%}")
    ct = det.score(tgt)
    lg = det.score(legit[FEATURE_COLUMNS])
    print(f"\n  {TARGET} median score sits at the {100 * (lg < np.median(ct)).mean():.1f}th "
          f"percentile of LEGITIMATE traffic.")
    print("  It is not near the threshold and failing to clear it. It is scoring like "
          "a\n  quieter-than-average legitimate transaction.")

    # ------------------------------------------------------------------ B
    print("\n=== B. how far outside the training support ===")
    print("  fraction of rows beyond the min/max ever seen in training, per column\n")
    rows = []
    for f in CONTROLLED_FEATURES:
        lo, hi = train[f].min(), train[f].max()
        q_lo, q_hi = train[f].quantile(0.001), train[f].quantile(0.999)
        for name, X in ((TARGET, tgt), (FOIL, foil), ("real fraud", fraud)):
            v = X[f].to_numpy()
            rows.append(dict(feature=f, population=name,
                             beyond_minmax=float(((v < lo) | (v > hi)).mean()),
                             beyond_q999=float(((v < q_lo) | (v > q_hi)).mean()),
                             median=float(np.median(v)),
                             train_p999=float(q_hi)))
    b = pd.DataFrame(rows)
    piv = b.pivot(index="feature", columns="population", values="beyond_q999")
    print(piv[[TARGET, FOIL, "real fraud"]].sort_values(TARGET, ascending=False)
             .map(lambda x: f"{x:6.1%}").to_string())
    worst = piv[TARGET].sort_values(ascending=False)
    print(f"\n  worst column for {TARGET}: {worst.index[0]} "
          f"({worst.iloc[0]:.1%} of rows past the training p99.9)")

    # ------------------------------------------------------------------ C
    print("\n=== C. what the trees hold where these rows land ===")
    booster = det.model.booster_
    tr_leaves = booster.predict(train[FEATURE_COLUMNS].to_numpy(), pred_leaf=True)
    y_tr = train[LABEL_COLUMN].to_numpy()

    def leaf_stats(X):
        lv = booster.predict(X.to_numpy(), pred_leaf=True)
        occ, frac_fraud = [], []
        for t in range(0, lv.shape[1], max(1, lv.shape[1] // 30)):   # sample 30 trees
            counts = np.bincount(tr_leaves[:, t])
            fraud_counts = np.bincount(tr_leaves[:, t], weights=y_tr)
            idx = lv[:, t]
            occ.append(counts[idx])
            with np.errstate(invalid="ignore", divide="ignore"):
                frac_fraud.append(np.where(counts[idx] > 0, fraud_counts[idx] / np.maximum(counts[idx], 1), np.nan))
        return np.concatenate(occ), np.concatenate(frac_fraud)

    for name, X in ((TARGET, tgt), (FOIL, foil), ("real fraud", fraud[FEATURE_COLUMNS])):
        occ, ff = leaf_stats(X)
        print(f"  {name:20s} median training rows in the reached leaf: {np.median(occ):8.0f}"
              f"   fraud rate there: {np.nanmedian(ff):.4%}")
    print(f"\n  train base fraud rate: {y_tr.mean():.4%}")
    print("  A leaf reached by these rows is a leaf fitted on LEGITIMATE traffic. The score\n"
          "  is not a judgement about fraud; it is whatever the legitimate rows that "
          "reached\n  that leaf left behind.")

    # ------------------------------------------------------------------ D
    print("\n=== D. single-feature repair — replace ONE column with a legitimate draw ===")
    print("  (higher recall after repair = that column was carrying the excursion)\n")
    base_score = det.score(tgt)
    legit_pool = legit[FEATURE_COLUMNS].to_numpy()
    reps = []
    for f in CONTROLLED_FEATURES:
        rep = tgt.copy()
        draw = rng.choice(legit[f].to_numpy(), size=len(rep), replace=True)
        rep[f] = draw
        s = det.score(rep)
        reps.append(dict(feature=f, recall=float((s >= thr).mean()),
                         d_median_score=float(np.median(s) - np.median(base_score))))
    d = pd.DataFrame(reps).sort_values("recall", ascending=False)
    for _, r in d.iterrows():
        print(f"  repair {r.feature:26s} recall {r.recall:7.2%}   "
              f"median score {r.d_median_score:+.5f}")

    allrep = tgt.copy()
    for f in CONTROLLED_FEATURES:
        allrep[f] = rng.choice(legit[f].to_numpy(), size=len(allrep), replace=True)
    s_all = det.score(allrep)
    print(f"\n  repair ALL ten at once           recall {float((s_all >= thr).mean()):7.2%}")
    print(f"  no repair (as generated)         recall {float((base_score >= thr).mean()):7.2%}")

    # ------------------------------------------------------------------ E
    # D refutes the off-support hypothesis outright: repairing all ten controlled columns
    # leaves recall at 0.00%. So whatever is holding the score down is not in the columns
    # the red team sets. That leaves the sixteen it INHERITS.
    print("\n=== E. the sixteen inherited columns ===")
    print("  Every campaign is mounted on a real account that was selected for never having"
          "\n  committed fraud, and it inherits that account's issuer-side context whole:"
          "\n  account_age_days, merchant_risk and the fourteen linkage counts.\n")

    fraud_pool = fraud[FEATURE_COLUMNS].reset_index(drop=True)
    legit_pool_df = legit[FEATURE_COLUMNS].reset_index(drop=True)

    def transplant(donor: pd.DataFrame, cols) -> float:
        out = tgt.reset_index(drop=True).copy()
        idx = rng.integers(0, len(donor), len(out))
        for c in cols:
            out[c] = donor[c].to_numpy()[idx]
        s = det.score(out)
        return float((s >= thr).mean()), float(np.median(s))

    controlled_only = [c for c in CONTROLLED_FEATURES]
    for label, donor, cols in (
        ("inherited <- REAL FRAUD", fraud_pool, INHERITED_FEATURES),
        ("inherited <- legit",      legit_pool_df, INHERITED_FEATURES),
        ("controlled <- REAL FRAUD", fraud_pool, controlled_only),
        ("everything <- REAL FRAUD", fraud_pool, FEATURE_COLUMNS),
    ):
        r, m = transplant(donor, cols)
        print(f"  {label:26s} recall {r:7.2%}   median score {m:.5f}")
    print(f"  {'as generated':26s} recall {float((base_score >= thr).mean()):7.2%}   "
          f"median score {np.median(base_score):.5f}")
    print(f"  {'real fraud itself':26s} recall {float((det.score(fraud[FEATURE_COLUMNS]) >= thr).mean()):7.2%}")

    print("\n  Every vector, not just this one:\n")
    print(f"  {'vector':20s} {'as generated':>13s} {'ctrl<-fraud':>13s} "
          f"{'inherit<-fraud':>15s} {'all<-fraud':>12s}")
    gen_rows = []
    for vid, rows_ in batches.items():
        def tp(donor, cols, src=rows_):
            o = src.reset_index(drop=True).copy()
            idx = rng.integers(0, len(donor), len(o))
            for c in cols:
                o[c] = donor[c].to_numpy()[idx]
            return float((det.score(o) >= thr).mean())
        r0 = float((det.score(rows_) >= thr).mean())
        rc = tp(fraud_pool, controlled_only)
        ri = tp(fraud_pool, INHERITED_FEATURES)
        ra = tp(fraud_pool, FEATURE_COLUMNS)
        gen_rows.append(dict(vector=vid, as_generated=r0, controlled_from_fraud=rc,
                             inherited_from_fraud=ri, all_from_fraud=ra))
        print(f"  {vid:20s} {r0:12.2%} {rc:13.2%} {ri:15.2%} {ra:12.2%}")
    real_r = float((det.score(fraud[FEATURE_COLUMNS]) >= thr).mean())
    print(f"  {'(real fraud itself)':20s} {'—':>12s} {'—':>13s} {'—':>15s} {real_r:12.2%}")
    pd.DataFrame(gen_rows).to_csv(
        Path(__file__).resolve().parents[2] / "results" / "inherited_block_transplant.csv",
        index=False)

    print("\n  This is the answer, and it is not evasion and not off-support. The attack rows"
          "\n  carry the issuer-side fingerprint of a KNOWN-GOOD customer, because the "
          "host\n  selection rule requires it. The linkage block is where nearly all the "
          "real-fraud\n  signal lives (2.25% -> 15.46% recall in feature_ablation.py), and "
          "the red team\n  inherits it from an account that has never been fraudulent. The "
          "detector is not\n  failing to see card testing. It is correctly reporting that "
          "this is a clean account,\n  because that is what we built the row out of.")

    out = Path(__file__).resolve().parents[2] / "results" / "card_testing_offsupport.csv"
    d.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
