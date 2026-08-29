"""What does the detector do to legitimate subscription retries?

`card_testing` is the easiest vector in the suite: PCI-mandated to catch, 97% caught after
the loop, and it anchors the top of the scale. That number means nothing on its own,
because the cheapest way to reach it is to flag every repeated same-amount attempt on a
card — which is also what a dunning run looks like. Stripe's own documentation says
subscription retries "can look like card testing".

So this script reports card-testing recall and the false-positive rate on dunning at the
SAME threshold. A detector that buys the first with the second has not solved anything: a
merchant running a subscription book pays for that recall in declined renewals.

`Dunning` is a legitimate population (`is_fraud = 0`), is not in `ALL_VECTORS`, is never
optimized against the detector, and the loop never trains on it.

Two variants:
  strict   is_new_beneficiary = 0.0  — a retry goes to an already-paid merchant
  hard     is_new_beneficiary = 0.90 — the card-testing rate; happens after a card update
                                       creates a fresh payment record

The second exists because if the two populations separate only on that one binary column,
the "detector distinguishes them behaviourally" claim is false and this script should say
so rather than let the strict number imply otherwise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN          # noqa: E402
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.evaluation import threshold_for_fpr                    # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam import ALL_VECTORS                             # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402
from chhal.redteam.vectors import CardTesting, Dunning            # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FPR = 0.001
SEEDS = (7, 11, 13)
N_TRAIN_ATTACK = 400
N_EVAL = 500
ADAPT_ROUNDS = 3   # the loop runs 8; three is enough to reach a comparable regime


def hard_dunning():
    class HardDunning(Dunning):
        new_payee_rate = CardTesting.new_payee_rate
    return HardDunning


def run_seed(seed: int, base) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    train_hosts = HostPool(base.train)
    test_hosts = HostPool(base.test, exclude_accounts=base.train["_account"])
    optimizer = EvasionOptimizer(base.feature_stats)

    baseline = Detector(seed=seed).fit(base.train, LABEL_COLUMN)

    # A detector that has adapted to the whole suite, as in the loop. Card testing is
    # 0.00% on the un-retrained detector, so the confusion question only has content once
    # the detector has actually learned to catch it. Three rounds rather than the loop's
    # eight: absolute recall lands lower than the headline, and what is being compared
    # here is two populations at ONE threshold, not a level against the loop's.
    pool = [base.train]
    det = baseline
    for rnd in range(ADAPT_ROUNDS):
        for V in ALL_VECTORS:
            v = V().calibrate(profile, train_hosts)
            b = optimizer.optimize(v.batch(N_TRAIN_ATTACK, rnd, rng), det, rng)
            rows = b.transactions.copy()
            rows[LABEL_COLUMN] = 1
            pool.append(rows)
        det = Detector(seed=seed).fit(pd.concat(pool, ignore_index=True), LABEL_COLUMN)

    legit = base.test[base.test[LABEL_COLUMN] == 0]
    thr = threshold_for_fpr(det.score(legit[FEATURE_COLUMNS]), FPR)

    def flagged(VectorCls, optimize: bool) -> float:
        v = VectorCls().calibrate(profile, test_hosts)
        b = v.batch(N_EVAL, 1, rng)
        if optimize:
            b = optimizer.optimize(b, det, rng)     # only an ATTACKER evades
        return float((det.score(b.transactions[FEATURE_COLUMNS]) >= thr).mean())

    return pd.DataFrame([
        dict(seed=seed, population="card_testing (attack)", is_fraud=1,
             rate=flagged(CardTesting, optimize=True)),
        dict(seed=seed, population="dunning strict (legit)", is_fraud=0,
             rate=flagged(Dunning, optimize=False)),
        dict(seed=seed, population="dunning hard (legit)", is_fraud=0,
             rate=flagged(hard_dunning(), optimize=False)),
        dict(seed=seed, population="real legit traffic", is_fraud=0,
             rate=float((det.score(legit[FEATURE_COLUMNS]) >= thr).mean())),
    ])


def main() -> None:
    t0 = time.time()
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")
    df = pd.concat([run_seed(s, base) for s in SEEDS], ignore_index=True)

    RESULTS.mkdir(exist_ok=True)
    agg = df.groupby(["population", "is_fraud"]).rate.agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    agg.to_csv(RESULTS / "dunning_control.csv", index=False)

    print(f"\n=== flagged at a {FPR:.1%} FPR budget, {len(SEEDS)} seeds ===")
    order = ["card_testing (attack)", "dunning hard (legit)",
             "dunning strict (legit)", "real legit traffic"]
    for name in order:
        r = agg[agg.population == name].iloc[0]
        tag = "CAUGHT " if r.is_fraud else "FALSE +"
        print(f"  {tag} {name:26s} {r['mean']:7.2%} +- {r['sem']:.2%}")

    ct = agg[agg.population == "card_testing (attack)"]["mean"].iloc[0]
    strict = agg[agg.population == "dunning strict (legit)"]["mean"].iloc[0]
    hard = agg[agg.population == "dunning hard (legit)"]["mean"].iloc[0]
    legit_rate = agg[agg.population == "real legit traffic"]["mean"].iloc[0]

    print("\n=== reading ===")
    print(f"  card testing caught                     : {ct:.2%}")
    print(f"  cost to a merchant running dunning      : {strict:.2%} of retries declined")
    print(f"  ... if a card update resets the payee   : {hard:.2%}")
    print(f"  ... against {legit_rate:.3%} on ordinary traffic "
          f"({strict / legit_rate:.0f}x / {hard / legit_rate:.0f}x the base rate)")
    # Compare as a RATIO. Both variants sit near the floor, so a percentage-point
    # threshold would call 2.00% and 0.33% "in agreement" when one is six times the other.
    ratio = hard / strict if strict > 0 else float("inf")
    if ratio >= 2.0:
        print(f"\n  The hard variant is flagged {ratio:.1f}x as often as the strict one, so "
              "much of the\n  separation is `is_new_beneficiary` rather than the retry "
              "shape. The detector is\n  reading WHO is being paid, not HOW the sequence "
              "behaves — and a merchant whose\n  card updater resets the payee record "
              f"loses {hard:.2%} of its retries.")
    else:
        print("\n  Both dunning variants land together, so the separation is behavioural "
              "rather\n  than one binary column.")
    print(f"\n-> {RESULTS / 'dunning_control.csv'}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
