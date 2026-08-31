"""CP@k. The metric a real alert queue is actually scored on, computed once.

An investigation team works a finite queue: k cards a day, whatever the model's ROC looks
like. Dal Pozzolo et al. (ULB / Worldline) formalised that as Card Precision at k, rank
CARDS by their highest score of the day, take the top k, and ask what fraction of them
really were compromised. It is the closest thing in the fraud literature to an operational
metric, which is exactly why it needs handling carefully here.

CP@k flatters this system, and it flatters it in the one direction the rest of the repo is
arguing against. Recall at a 0.1% FPR budget is ~14% on real fraud; CP@100 is several times
that, because ranking a hundred cards a day is a far easier problem than separating fraud
from 73,000 legitimate transactions at a fixed false-positive rate. Both numbers are true.
Leading with the friendlier one, in a paper whose finding is that this system has been
measuring itself too kindly, would hand a reviewer their paragraph.

So it is computed here, reported once, and never quoted as a headline.

Reported per day and averaged over days, on the post-purge test split, never pooled
across the whole window, which would let one busy day dominate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN          # noqa: E402
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.evaluation import threshold_for_fpr                    # noqa: E402

KS = (20, 50, 100, 200)
SEED = 7


def main() -> None:
    base = load_base_data(source="ieee")
    det = Detector(seed=SEED).fit(base.train, LABEL_COLUMN)

    te = base.test.copy()
    te["_score"] = det.score(te[FEATURE_COLUMNS])
    te["_day"] = (te["_ts"] // 86_400).astype(np.int64)

    # one row per (card, day): its highest score, and whether it was really compromised
    card_day = (te.groupby(["_day", "_account"])
                  .agg(score=("_score", "max"), fraud=(LABEL_COLUMN, "max"))
                  .reset_index())

    print(f"test split: {len(te):,} transactions, "
          f"{card_day._day.nunique()} days, "
          f"{card_day._account.nunique():,} distinct cards")
    print(f"cards per day: median {int(card_day.groupby('_day').size().median()):,}, "
          f"min {int(card_day.groupby('_day').size().min()):,}")

    rows = []
    for k in KS:
        daily = []
        skipped = 0
        for day, g in card_day.groupby("_day"):
            if len(g) < k:                      # a day too small to fill the queue
                skipped += 1
                continue
            top = g.nlargest(k, "score")
            daily.append(float(top.fraud.mean()))
        rows.append(dict(k=k, cp_at_k=float(np.mean(daily)),
                         sd=float(np.std(daily)), days=len(daily), days_skipped=skipped))

    df = pd.DataFrame(rows)
    print("\n=== CP@k, averaged over days ===")
    for _, r in df.iterrows():
        print(f"  CP@{int(r.k):<4d} {r.cp_at_k:6.2%}  (sd {r.sd:.2%} over {int(r.days)} days"
              + (f", {int(r.days_skipped)} days too small to fill the queue)" if r.days_skipped else ")"))

    # the number the rest of the repo leads with, for contrast on the same split
    legit = te[te[LABEL_COLUMN] == 0]
    fraud = te[te[LABEL_COLUMN] == 1]
    thr = threshold_for_fpr(det.score(legit[FEATURE_COLUMNS]), 0.001)
    recall = float((det.score(fraud[FEATURE_COLUMNS]) >= thr).mean())

    # the base rate a queue filled at random would hit
    base_rate = float(card_day.fraud.mean())
    cp100 = df[df.k == 100].cp_at_k.iloc[0]
    print(f"\n  for contrast, recall @ 0.1% FPR on the same split : {recall:.2%}")
    print(f"  fraudulent share of all card-days (random queue)  : {base_rate:.2%}")
    print(f"  CP@100 lift over a random queue                   : {cp100 / base_rate:.1f}x")
    print("\n  Both are true. CP@k asks whether the top of the ranking is worth an "
          "analyst's day;\n  recall at a fixed FPR asks how much fraud the system misses. "
          "This repo's finding is\n  about the second, so the second is the headline and "
          "this file is the footnote.")

    out = Path(__file__).resolve().parents[2] / "results" / "card_precision_at_k.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
