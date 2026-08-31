"""Are six vectors six attacks, or fewer attacks wearing six names?

`upi_collect` is described in the README as a distinct attack on a distinct rail. On the
frozen 26 columns it may be nothing of the kind: it shares `bustout`'s high amount band and
`mimic_host=False`, and differs mainly in the SIGN of its amount trend (0.7 draining vs 1.6
escalating) and in being shorter. If a classifier cannot tell the two apart, then it is not
a sixth attack. It is free evidence for generalisation, which is a better thing to have
and an honest thing to say.

Measured, not asserted, two ways:

1. **Pairwise separability.** Train a classifier to tell vector A's rows from vector B's,
   on the ten columns the red team controls. AUC 0.5 means indistinguishable; AUC 1.0
   means they are different populations. Reported as a matrix.
2. **Transfer.** The leave-one-vector-out number from `scripts/generalisation_check.py`
   already says which vectors a detector reaches without ever seeing them. A vector that
   is both inseparable from a sibling AND well-reached when held out is a sibling, not a
   sixth family.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chhal.contract import LABEL_COLUMN                           # noqa: E402
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.fidelity import CONTROLLED_FEATURES                    # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam import ALL_VECTORS                             # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402

N = 800
SEED = 7


def main() -> None:
    base = load_base_data(source="ieee")
    rng = np.random.default_rng(SEED)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    hosts = HostPool(base.test, exclude_accounts=base.train["_account"])
    det = Detector(seed=SEED).fit(base.train, LABEL_COLUMN)
    opt = EvasionOptimizer(base.feature_stats)

    rows = {}
    for V in ALL_VECTORS:
        v = V().calibrate(profile, hosts)
        rows[v.vector_id] = opt.optimize(v.batch(N, 0, rng), det, rng).transactions
    ids = list(rows)

    print(f"pairwise separability on the {len(CONTROLLED_FEATURES)} controlled columns, "
          f"n={N} each, 3-fold out-of-fold AUC\n")
    out = []
    for a, b in combinations(ids, 2):
        X = pd.concat([rows[a][CONTROLLED_FEATURES], rows[b][CONTROLLED_FEATURES]],
                      ignore_index=True).to_numpy(np.float64)
        y = np.r_[np.zeros(len(rows[a])), np.ones(len(rows[b]))]
        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                             random_state=SEED, n_jobs=-1, verbose=-1)
        p = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
        out.append(dict(a=a, b=b, auc=float(roc_auc_score(y, p))))

    m = pd.DataFrame(np.nan, index=ids, columns=ids)
    for r in out:
        m.loc[r["a"], r["b"]] = m.loc[r["b"], r["a"]] = round(r["auc"], 3)
    print(m.fillna("n/a").to_string())

    df = pd.DataFrame(out).sort_values("auc")
    print("\nclosest pairs:")
    for _, r in df.head(3).iterrows():
        print(f"  {r.a:20s} vs {r.b:20s}  AUC {r.auc:.3f}")
    print("furthest pairs:")
    for _, r in df.tail(3).iterrows():
        print(f"  {r.a:20s} vs {r.b:20s}  AUC {r.auc:.3f}")

    lo = df.iloc[0]
    print(f"\nVERDICT: the closest pair is {lo.a} / {lo.b} at AUC {lo.auc:.3f}.")
    if lo.auc < 0.75:
        print("  That is close enough to call them one family with two labels.")
    elif lo.auc < 0.90:
        print("  Separable, but not cleanly, a detector that learns one gets part of the "
              "other\n  for free, which is what the leave-one-out transfer number should "
              "then show.")
    else:
        print("  Every pair is cleanly separable on the controlled columns, so all six are "
              "distinct\n  populations in the feature space the detector actually sees.")

    p = Path(__file__).resolve().parents[2] / "results" / "vector_separability.csv"
    df.to_csv(p, index=False)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
