"""Does the anomaly arm actually buy anything? Measured, not assumed.

The argument for it is structural: a supervised detector cannot recognise an attack
family absent from its training data, and the brief is about emerging attacks. That is
an argument, not evidence. This script is the evidence, and it is built to be able to
come back negative.

Two questions, both at a FIXED 0.1% false-positive budget on real legitimate traffic so
the comparison is like for like:

  1. UNSEEN FAMILY — train on three attack vectors, score the fourth, which neither arm
     has seen in any form. Does supervised + anomaly beat supervised alone?
  2. REAL FRAUD — IEEE-CIS's own labelled fraud, where the supervised arm scores only
     ~3% at this budget through twelve derived features. Does the anomaly arm help there?

Also reports how often the anomaly arm is the one carrying a catch, which is the number
that says whether it is adding information or just agreeing with the supervised arm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN            # noqa: E402
from chhal.data import load_base_data                               # noqa: E402
from chhal.detector import Detector                                 # noqa: E402
from chhal.ensemble import AnomalyArm, Ensemble, StackedDetector    # noqa: E402
from chhal.optimizer import EvasionOptimizer                        # noqa: E402
from chhal.redteam import ALL_VECTORS                               # noqa: E402
from chhal.redteam.base import BaseProfile                          # noqa: E402
from chhal.redteam.hosts import HostPool                            # noqa: E402

SEED = 7
N_PER_VECTOR = 500
FPR = 0.001
REF_ROWS = 60_000          # legit rows used to place both arms on a common axis
RESULTS = Path(__file__).resolve().parents[1] / "results"
VARIANTS = ("supervised", "max_fusion", "stacked")


def recall_at(scores_legit, scores_pos, fpr=FPR):
    thr = np.quantile(scores_legit, 1 - fpr)
    return float((scores_pos >= thr).mean()), thr


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")

    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    test_legit = base.test[base.test[LABEL_COLUMN] == 0]
    test_fraud = base.test[base.test[LABEL_COLUMN] == 1][FEATURE_COLUMNS]
    ref_legit = test_legit.sample(min(REF_ROWS, len(test_legit)), random_state=SEED)
    eval_legit = test_legit.drop(ref_legit.index)[FEATURE_COLUMNS]
    print(f"[eval ] {len(eval_legit):,} legit for the budget, {len(ref_legit):,} as the "
          f"shared axis, {len(test_fraud):,} real frauds")

    baseline = Detector(seed=SEED).fit(base.train, LABEL_COLUMN)
    opt = EvasionOptimizer(base.feature_stats)
    adapted = {}
    hosts = HostPool(base.test, exclude_accounts=base.train['_account'])
    print(f"[hosts] {hosts.describe()}")
    for V in ALL_VECTORS:
        v = V().calibrate(profile, hosts)
        adapted[v.vector_id] = opt.optimize(v.batch(N_PER_VECTOR, 0, rng), baseline, rng).transactions
    ids = list(adapted)

    # the anomaly arm never sees a fraud label, so it is fit once and reused
    anom = AnomalyArm().fit(base.train, LABEL_COLUMN)

    rows, carried = [], []
    for held in ids:
        pool = pd.concat([base.train] + [adapted[i].assign(**{LABEL_COLUMN: 1})
                                         for i in ids if i != held], ignore_index=True)
        det = Detector(seed=SEED).fit(pool, LABEL_COLUMN)
        ens = Ensemble(det, anom).fit_reference(ref_legit)

        stack = StackedDetector(anomaly=anom, seed=SEED).fit(pool, LABEL_COLUMN)
        unseen = adapted[held]

        r = {}
        for name, sc in (("supervised", det.score), ("max_fusion", ens.score),
                         ("stacked", stack.score)):
            r[f"unseen_{name}"] = round(recall_at(sc(eval_legit), sc(unseen))[0], 4)
            r[f"real_{name}"] = round(recall_at(sc(eval_legit), sc(test_fraud))[0], 4)

        # of the unseen attacks max-fusion catches, how many did the anomaly arm carry?
        sup_p, anom_p = ens.arm_percentiles(unseen)
        thr = np.quantile(ens.score(eval_legit), 1 - FPR)
        caught = ens.score(unseen) >= thr
        carried.append(float((anom_p[caught] > sup_p[caught]).mean()) if caught.any() else 0.0)

        rows.append({"held_out_vector": held, **r})
        print(f"  {held:<18} unseen sup={r['unseen_supervised']:.3f} "
              f"max={r['unseen_max_fusion']:.3f} stack={r['unseen_stacked']:.3f} | "
              f"real sup={r['real_supervised']:.4f} stack={r['real_stacked']:.4f}")

    df = pd.DataFrame(rows)
    print(f"\n=== leave-one-vector-out, recall @ {FPR:.1%} FPR on real legit traffic ===")
    print(df.to_string(index=False))

    summary = pd.DataFrame(
        {"unseen attack family": [df[f"unseen_{k}"].mean() for k in VARIANTS],
         "real IEEE-CIS fraud": [df[f"real_{k}"].mean() for k in VARIANTS]},
        index=list(VARIANTS)).round(4)
    print("\n=== mean over the four held-out families ===")
    print(summary.to_string())

    # diagnostic: the arm on its own, which is the number that explains the rest
    alone_u = float(np.mean([recall_at(anom.score(eval_legit), anom.score(adapted[i]))[0]
                             for i in ids]))
    alone_r = recall_at(anom.score(eval_legit), anom.score(test_fraud))[0]
    print(f"\nanomaly arm ALONE    : unseen {alone_u:.4f}   real fraud {alone_r:.4f}")
    print(f"catches it carries under max fusion: {np.mean(carried):.1%}")

    d_max = df["unseen_max_fusion"].mean() - df["unseen_supervised"].mean()
    d_stk = df["unseen_stacked"].mean() - df["unseen_supervised"].mean()
    # The verdict is DERIVED from the measurement, never asserted alongside it. An
    # earlier version hard-coded "stacking is the variant to ship" into this string and
    # into ensemble.py's docstring. That was true of the 12-feature space it was written
    # for; the linkage block changed it, and the sentence stayed, so the script printed a
    # recommendation its own two numbers contradicted.
    best = max(("supervised", 0.0), ("max fusion", d_max), ("stacking", d_stk),
               key=lambda kv: kv[1])[0]
    verdict = (
        f"max fusion {d_max:+.4f} and stacking {d_stk:+.4f} against the supervised "
        f"detector on an unseen family. The anomaly arm carries {np.mean(carried):.1%} of "
        f"catches and scores {alone_u:.4f} alone, because attacks are on-manifold by "
        f"construction — the better the fidelity guarantee, the less an outlier detector "
        f"can contribute. Ship: {best}."
        + ("" if best != "supervised" else
           " Neither fusion earns its 32MB; the single supervised model wins.")
    )
    print(f"\nverdict: {verdict}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ensemble_check.json").write_text(json.dumps(
        {"fpr": FPR, "per_vector": rows,
         "means": {k: {"unseen": round(df[f"unseen_{k}"].mean(), 4),
                       "real_fraud": round(df[f"real_{k}"].mean(), 4)} for k in VARIANTS},
         "anomaly_alone": {"unseen": round(alone_u, 4), "real_fraud": round(alone_r, 4)},
         "anomaly_carried_share": round(float(np.mean(carried)), 4),
         "verdict": verdict}, indent=2))
    df.to_csv(RESULTS / "ensemble_check.csv", index=False)
    print(f"-> {RESULTS/'ensemble_check.json'}")


if __name__ == "__main__":
    main()
