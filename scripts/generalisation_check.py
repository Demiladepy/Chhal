"""Leave-one-vector-out: does the defence generalise, or has it memorised the bands?

The loop's headline is "recall on a held-out adversarial benchmark rises to ~99%".
Held-out protects against memorising exact ROWS. It does not protect against memorising
the REGION a vector lives in: every `card_testing` row is drawn from the same quantile
bands, so once the detector has seen a thousand of them, held-out rows from the same
bands are trivial. The competition asks for defence against *emerging, novel* attacks,
which is the harder question.

So: train on every vector but one, score the one held out, which the detector has never
seen in any form. Recall is reported at a FIXED low false-positive operating point (0.1% of real
legitimate traffic), not at an arbitrary 0.5 threshold — that is how a payments system
is actually tuned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN            # noqa: E402
from chhal.data import load_base_data                               # noqa: E402
from chhal.detector import Detector                                 # noqa: E402
from chhal.evaluation import threshold_for_fpr                      # noqa: E402
from chhal.optimizer import EvasionOptimizer                        # noqa: E402
from chhal.redteam import ALL_VECTORS                               # noqa: E402
from chhal.redteam.base import BaseProfile                          # noqa: E402
from chhal.redteam.hosts import HostPool                            # noqa: E402

N_PER_VECTOR = 500
TARGET_FPR = 0.001          # 0.1% of legitimate traffic flagged


def recall_at_fpr(det, legit, attacks, fpr=TARGET_FPR):
    """Threshold set so at most `fpr` of real legit traffic is flagged; recall there.

    Uses the shared `threshold_for_fpr` rather than a local `np.quantile`, which is not
    a stylistic preference: a bare quantile can land inside a block of tied scores and
    the `>=` rule then flags the whole block, overshooting the budget it claims. This
    script quotes its numbers "at 0.1% FPR", so it has to honour that the same way the
    loop does.
    """
    thr = threshold_for_fpr(det.score(legit), fpr)
    return float((det.score(attacks) >= thr).mean()), float(thr)


def main() -> None:
    base = load_base_data(source="ieee")
    print(f"base: {base.describe()}\n")
    rng = np.random.default_rng(7)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    legit_eval = base.test[base.test[LABEL_COLUMN] == 0][FEATURE_COLUMNS]

    baseline = Detector(seed=7).fit(base.train, LABEL_COLUMN)
    opt = EvasionOptimizer(base.feature_stats)

    # every vector, adapted once against the baseline detector
    adapted = {}
    # These attacks are scored against test-side legitimate traffic, so they compromise
    # TEST accounts — an evaluation attack must not carry context the detector trained on.
    hosts = HostPool(base.test, exclude_accounts=base.train['_account'])
    print(f"[hosts] {hosts.describe()}")
    for V in ALL_VECTORS:
        v = V().calibrate(profile, hosts)
        adapted[v.vector_id] = opt.optimize(v.batch(N_PER_VECTOR, 0, rng), baseline, rng).transactions
    ids = list(adapted)

    rows = []
    for held in ids:
        seen = [adapted[i] for i in ids if i != held]
        pool = pd.concat([base.train] + [d.assign(**{LABEL_COLUMN: 1}) for d in seen],
                         ignore_index=True)
        det = Detector(seed=7).fit(pool, LABEL_COLUMN)

        r_unseen, _ = recall_at_fpr(det, legit_eval, adapted[held])
        r_seen = float(np.mean([recall_at_fpr(det, legit_eval, adapted[i])[0]
                                for i in ids if i != held]))
        r_base, _ = recall_at_fpr(baseline, legit_eval, adapted[held])
        rows.append({"held_out_vector": held,
                     "recall_baseline": round(r_base, 4),
                     "recall_if_seen(avg of the rest)": round(r_seen, 4),
                     "recall_UNSEEN": round(r_unseen, 4)})
        print(f"  {held:<18} baseline={r_base:.3f}  seen={r_seen:.3f}  UNSEEN={r_unseen:.3f}")

    df = pd.DataFrame(rows)
    print(f"\n=== leave-one-vector-out, recall @ {TARGET_FPR:.1%} FPR on real legit traffic ===")
    print(df.to_string(index=False))
    gap = df["recall_if_seen(avg of the rest)"].mean() - df["recall_UNSEEN"].mean()
    print(f"\nmean recall when the vector WAS trained on : {df['recall_if_seen(avg of the rest)'].mean():.3f}")
    print(f"mean recall when the vector was NEVER seen : {df['recall_UNSEEN'].mean():.3f}")
    print(f"generalisation gap                         : {gap:.3f}")
    out = Path(__file__).resolve().parents[1] / "results" / "generalisation_leave_one_out.csv"
    out.parent.mkdir(parents=True, exist_ok=True)   # 28s of work should not die on a mkdir
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
