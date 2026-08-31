"""Does `mule_fanout` rest on one binary column, and can its coordination be jittered away?

Two questions, one adapted detector.

**A. Single-column dependence.** `mule_fanout` exists to measure whether coordination is
visible in a feature space with no counterparty edge. That measurement is worthless if the
vector is really being caught on `is_new_beneficiary`, which it sets at 0.75 against a
legitimate base rate of 0.552. Hold every other knob fixed, move that one column to the
legitimate rate, and read the difference. An earlier draft of this vector had exactly this
problem with `is_cross_border` (0.35 against a 0.7% base rate, 93.4% recall), so the
failure mode is not hypothetical.

**B. Evasion sweep.** Sweep an evasion budget alpha and watch recall fall.

PROVENANCE, STATED PLAINLY. The jitter/chaff/permutation recipe and the alpha in [0, 0.5]
parameterisation come from arXiv 2607.27370, which is **Ethereum Sybil-cluster discovery
via Gzip normalised compression distance**, NOT a money-mule paper and not a
defender-validated mule evasion recipe. Using it here is a cross-domain translation and is
labelled as one wherever the number appears.

The translation is also incomplete, and the gap is itself a result. Of the paper's three
operations only two have an analogue in this feature space:

    jitter      -> widen the window the network fires in.        implemented
    chaff       -> pad each mule account with extra transfers.   implemented
    permutation -> shuffle the edges of the transaction graph.   NO ANALOGUE

There is no counterparty column in the frozen 26, so there is no graph and nothing to
permute. That is the same absence `mule_fanout` was built to measure, arriving from the
other direction.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN          # noqa: E402
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.evaluation import threshold_for_fpr                    # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402
from chhal.redteam.vectors import MuleFanout                      # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FPR = 0.001
SEEDS = (7, 11, 13)
N_TRAIN_ATTACK = 600      # mule campaigns the detector is retrained on
N_EVAL = 500              # fresh mule campaigns scored per condition
LEGIT_NEW_PAYEE = 0.552   # measured on the post-purge test split

BASE_WINDOW_S = 6 * 3_600.0
JITTER_MAX_EXTRA_S = 48 * 3_600.0     # alpha=0.5 -> a 30-hour window
CHAFF_MAX_EXTRA_TXNS = 20             # alpha=0.5 -> up to 15 transfers per mule


def variant(new_payee_rate=None, window_s=None, max_txns=None):
    """A MuleFanout with one knob moved. Everything else is inherited unchanged."""
    class V(MuleFanout):
        pass
    if new_payee_rate is not None:
        V.new_payee_rate = new_payee_rate
    tp = MuleFanout.temporal
    if window_s is not None:
        tp = replace(tp, coordinated_window_s=window_s)
    if max_txns is not None:
        tp = replace(tp, txns_per_entity=(tp.txns_per_entity[0], max_txns))
    V.temporal = tp
    return V


def run_seed(seed: int, base):
    """Fit a detector that has ALREADY adapted to mule_fanout, then measure against it.

    Measuring on the un-retrained detector would be meaningless: every vector scores 0.00%
    there, so every condition in this sweep would read 0.00% and the sweep would look like
    a flat line of perfect evasion. The interesting question is whether these knobs help
    against a detector that has seen this vector.
    """
    rng = np.random.default_rng(seed)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    train_hosts = HostPool(base.train)
    test_hosts = HostPool(base.test, exclude_accounts=base.train["_account"])
    optimizer = EvasionOptimizer(base.feature_stats)

    baseline_det = Detector(seed=seed).fit(base.train, LABEL_COLUMN)

    # retrain on mule campaigns mounted on TRAIN hosts, adapted against the baseline
    tr_v = MuleFanout().calibrate(profile, train_hosts)
    tr_batch = optimizer.optimize(tr_v.batch(N_TRAIN_ATTACK, 0, rng), baseline_det, rng)
    pool = tr_batch.transactions.copy()
    pool[LABEL_COLUMN] = 1
    det = Detector(seed=seed).fit(pd.concat([base.train, pool], ignore_index=True),
                                  LABEL_COLUMN)

    legit = base.test[base.test[LABEL_COLUMN] == 0]
    thr = threshold_for_fpr(det.score(legit[FEATURE_COLUMNS]), FPR)

    def recall(VectorCls) -> float:
        v = VectorCls().calibrate(profile, test_hosts)
        b = optimizer.optimize(v.batch(N_EVAL, 1, rng), det, rng)   # attacker adapts too
        return float((det.score(b.transactions[FEATURE_COLUMNS]) >= thr).mean())

    rows = []
    rows.append(dict(experiment="column", knob="new_payee_rate", alpha=np.nan,
                     value=MuleFanout.new_payee_rate, recall=recall(MuleFanout)))
    rows.append(dict(experiment="column", knob="new_payee_rate", alpha=np.nan,
                     value=LEGIT_NEW_PAYEE,
                     recall=recall(variant(new_payee_rate=LEGIT_NEW_PAYEE))))

    for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        w = BASE_WINDOW_S + a * JITTER_MAX_EXTRA_S
        rows.append(dict(experiment="jitter", knob="coordinated_window_h", alpha=a,
                         value=w / 3600.0, recall=recall(variant(window_s=w))))
        m = int(round(MuleFanout.temporal.txns_per_entity[1] + a * CHAFF_MAX_EXTRA_TXNS))
        rows.append(dict(experiment="chaff", knob="max_txns_per_mule", alpha=a,
                         value=float(m), recall=recall(variant(max_txns=m))))

    df = pd.DataFrame(rows)
    df["seed"] = seed
    return df


def main() -> None:
    t0 = time.time()
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")
    df = pd.concat([run_seed(s, base) for s in SEEDS], ignore_index=True)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "mule_alpha_sweep_raw.csv", index=False)

    agg = (df.groupby(["experiment", "knob", "alpha", "value"], dropna=False)
             .recall.agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg.to_csv(RESULTS / "mule_alpha_sweep.csv", index=False)

    # --- A. the single-column question -------------------------------------------
    col = df[df.experiment == "column"].pivot_table(index="seed", columns="value",
                                                    values="recall")
    native, legit_rate = MuleFanout.new_payee_rate, LEGIT_NEW_PAYEE
    delta = col[native] - col[legit_rate]                 # paired, per seed
    print("\n=== A. does mule_fanout rest on is_new_beneficiary? ===")
    print(f"  new_payee_rate = {native}  (native) : {col[native].mean():6.2%}")
    print(f"  new_payee_rate = {legit_rate}  (legit)  : {col[legit_rate].mean():6.2%}")
    print(f"  paired delta over {len(delta)} seeds        : "
          f"{delta.mean() * 100:+.1f} +- {delta.sem() * 100:.1f} pts")

    # --- B. the sweep -------------------------------------------------------------
    print("\n=== B. evasion sweep (cross-domain translation of arXiv 2607.27370) ===")
    for exp in ("jitter", "chaff"):
        e = agg[agg.experiment == exp]
        print(f"\n  {exp}  ({e.knob.iloc[0]})")
        for _, r in e.iterrows():
            print(f"    alpha={r.alpha:.1f}  {r.knob}={r.value:7.2f}  "
                  f"recall={r['mean']:6.2%} +- {r['sem']:.2%}")
        lo, hi = e[e.alpha == 0.0]["mean"].iloc[0], e[e.alpha == 0.5]["mean"].iloc[0]
        print(f"    alpha 0 -> 0.5 : {lo:.2%} -> {hi:.2%}  ({(hi - lo) * 100:+.1f} pts)")

    print("\n  permutation: no analogue. The frozen feature space has no counterparty "
          "column,\n  so there is no transaction graph to permute, which is the absence "
          "mule_fanout\n  was built to measure.")
    print(f"\n-> {RESULTS / 'mule_alpha_sweep.csv'}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
