"""What is the detector actually catching? Two ablations, one controlled run.

Both of this project's vector-level design claims are the kind that sound obviously true
and are therefore worth making prove themselves. Neither is measurable from the loop,
because eight rounds of adaptation is not a controlled comparison: the variants have to
differ in exactly one thing.

`mule_fanout` is the one vector whose defining property is not a property of any single
transaction. One operator drives many accounts; each account does something modest and
unremarkable; the attack is the fact that they all did it inside the same window.

The frozen feature space has no counterparty. No beneficiary id, no destination account,
no edge between two rows, so there is a strong prior that the detector cannot see this at
all. A prior is not a result, and "our features cannot represent X" is exactly the kind of
claim a submission should be made to prove rather than assert. So this script ablates it.

Three variants, one detector, one operating point:

    threshold_hugging                as shipped (reads its bands off the victim)
    threshold_hugging_no_mimicry     mimic_host OFF, everything else byte-identical

    mule_fanout                      as shipped
    mule_fanout_uncoordinated        coordination OFF, everything else byte-identical
    mule_fanout_known_payee          is_new_beneficiary forced to 0, otherwise identical
    mule_fanout_always_new_payee     is_new_beneficiary forced to 1. The old behaviour

The first pair asks whether per-victim mimicry is doing anything, or whether the mimicry
vector would be just as hard to catch drawing from the population. The second pair asks
whether the detector can see coordination, and the third exists to answer the obvious
follow-up: if not coordination, then what IS it catching?

Single-pass detector throughout: one round of attacks, one retrain. Recall is lower than
the loop reports for the same vectors, and that is the price of a comparison that means
something.

Two things this script learned the hard way, and now guards against:

* One seed is not a result. The coordination delta came out at -0.8, -3.2 and -8.3 points
  on three consecutive runs of an earlier version. Same code, same data, different rng,
  which means any one of those numbers, quoted alone, would have been a story invented
  from noise. Every variant is therefore run across SEEDS, and what gets reported is the
  mean with its spread.
* One operating point is not enough. At a 0.1% budget the mimicry vector and its ablation
  both score exactly zero, so the comparison is floored and says nothing. Recall is
  reported at three budgets, and a comparison is only read where it is not against a wall.

Outputs results/coordination_check.json.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t   # `stats` is a local dict below
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN          # noqa: E402
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam import ALL_VECTORS                             # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402

SEEDS = (7, 11, 23, 42, 101)
N_TRAIN_PER_VECTOR = 1200
N_EVAL = 600
FPRS = (0.001, 0.01, 0.05)
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _variants():
    mimicry = ALL_VECTORS[0]
    mf = [V for V in ALL_VECTORS if V.vector_id == "mule_fanout"][0]

    class NoMimicry(mimicry):
        vector_id = "threshold_hugging_no_mimicry"
        temporal = replace(mimicry.temporal, mimic_host=False)

    class Uncoordinated(mf):
        vector_id = "mule_fanout_uncoordinated"
        temporal = replace(mf.temporal, coordinated_window_s=None)

    class KnownPayee(mf):
        vector_id = "mule_fanout_known_payee"

        def static_features(self, n, rng):
            d = mf.static_features(self, n, rng)
            d["is_new_beneficiary"] = np.zeros(n, int)
            return d

    class AlwaysNewPayee(mf):
        """The vector as it USED to ship: the flag hard-coded to 1 on every single row.

        Kept as a variant rather than deleted, because the difference between this and
        the shipped 0.75 is the only honest measure of what that fix was worth. The
        feature carries real signal for this vector either way, a mule fan-out does
        send money somewhere new, so the question is not whether the detector may use
        it, but how much of its recall came from a determinism no real campaign has.
        """

        vector_id = "mule_fanout_always_new_payee"

        def static_features(self, n, rng):
            d = mf.static_features(self, n, rng)
            d["is_new_beneficiary"] = np.ones(n, int)
            return d

    return [mimicry, NoMimicry, mf, Uncoordinated, KnownPayee, AlwaysNewPayee]


def _one_seed(base, seed: int):
    """Train a detector on this seed's attacks, then score every variant against it."""
    rng = np.random.default_rng(seed)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    train_hosts = HostPool(base.train)
    test_hosts = HostPool(base.test, exclude_accounts=base.train["_account"])
    opt = EvasionOptimizer(base.feature_stats)

    seed_det = Detector(seed=seed).fit(base.train, LABEL_COLUMN)
    train_rows = [
        opt.optimize(V().calibrate(profile, train_hosts).batch(N_TRAIN_PER_VECTOR, 0, rng),
                     seed_det, rng).transactions.assign(**{LABEL_COLUMN: 1})
        for V in ALL_VECTORS
    ]
    det = Detector(seed=seed).fit(
        pd.concat([base.train] + train_rows, ignore_index=True), LABEL_COLUMN)

    legit = det.score(base.test.loc[base.test[LABEL_COLUMN] == 0, FEATURE_COLUMNS])
    thresholds = {fpr: float(np.quantile(legit, 1 - fpr)) for fpr in FPRS}

    out = {}
    for V in _variants():
        rows = opt.optimize(V().calibrate(profile, test_hosts).batch(N_EVAL, 9, rng),
                            det, rng).transactions
        sc = det.score(rows[FEATURE_COLUMNS])
        out[V.vector_id] = {fpr: float((sc >= t).mean()) for fpr, t in thresholds.items()}
    return out


def main() -> None:
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")
    print(f"[plan] {len(SEEDS)} seeds x {len(_variants())} variants "
          f"at {len(FPRS)} operating points")

    runs = []
    for seed in SEEDS:
        runs.append(_one_seed(base, seed))
        print(f"  seed {seed} done")

    names = [V.vector_id for V in _variants()]
    stats = {n: {f: (float(np.mean([r[n][f] for r in runs])),
                     float(np.std([r[n][f] for r in runs])))
                 for f in FPRS} for n in names}

    print(f"\n=== recall, mean +/- sd over {len(SEEDS)} seeds ===")
    print(f"{'variant':<32}" + "".join(f"{f:>18.1%}" for f in FPRS))
    for n in names:
        row = "".join(f"{stats[n][f][0]*100:>11.1f} +/-{stats[n][f][1]*100:>4.1f}"
                      for f in FPRS)
        print(f"{n:<32}{row}")

    def delta(a, b, fpr):
        """Recall(a) - Recall(b), paired per seed.

        Returns the mean difference and the STANDARD ERROR of that mean, not the sample
        spread. The spread answers "how much does one run vary"; the standard error
        answers "how well do I know the average", and only the second one licenses a
        claim. Paired per seed because both variants share a detector within a seed, so
        the seed's own difficulty cancels instead of being counted as noise.
        """
        d = np.array([runs[i][a][fpr] - runs[i][b][fpr] for i in range(len(runs))])
        sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
        return float(d.mean()) * 100, float(sem) * 100

    comparisons = [
        ("per-victim mimicry", "threshold_hugging_no_mimicry", "threshold_hugging"),
        ("coordination", "mule_fanout_uncoordinated", "mule_fanout"),
        ("the new-beneficiary flag", "mule_fanout_known_payee", "mule_fanout"),
        ("...and hard-coding it to 1", "mule_fanout", "mule_fanout_always_new_payee"),
    ]
    print("\n=== what each design choice is worth to the ATTACKER ===")
    print("(points of recall the detector LOSES because the choice is switched on)")
    findings = {}
    for label, off, on in comparisons:
        # Read each comparison where it is most SENSITIVE, not where recall is highest:
        # at a loose budget the mule variants all sit near saturation and any real
        # difference is squeezed flat, while at a tight one the mimicry variants are both
        # on the floor. Picking by signal-to-noise avoids choosing either wall.
        scored = [(f,) + delta(off, on, f) for f in FPRS]
        best, m, sem = max(scored, key=lambda x: abs(x[1]) / (x[2] + 1e-9))
        # Two corrections, because the line above SELECTED this operating point by
        # maximising |t| and a naive test on a selected maximum is not a 95% test.
        #   1. Student's t on len(SEEDS)-1 degrees of freedom, not 2.0. With five seeds the
        #      normal quantile is 2.00 against a correct 2.776 -- a 39% understatement,
        #      in the direction that manufactures findings.
        #   2. Bonferroni over the len(FPRS) operating points the maximum was taken over,
        #      since the alternative is to pre-register one and this script's whole reason
        #      for scanning is that it does not know in advance which is sensitive.
        crit = float(student_t.ppf(1 - 0.05 / (2 * len(FPRS)), len(SEEDS) - 1))
        decisive = abs(m) > crit * sem and abs(m) >= 1.0
        findings[label] = {
            "read_at_fpr": best, "mean_pp": round(m, 2), "sem_pp": round(sem, 2),
            "crit": round(crit, 3), "selected_over_n_fprs": len(FPRS),
            "beats_its_own_noise": decisive,
            "all_operating_points": {str(f): [round(mm, 2), round(ss, 2)]
                                     for f, mm, ss in scored},
        }
        verdict = (f"clears {crit:.2f} SEM (t, Bonferroni over {len(FPRS)} FPRs)" if decisive
                   else "INSIDE THE NOISE - not a finding")
        print(f"  {label:<26} {m:+6.1f} +/- {sem:4.1f} pp  at {best:.1%}   {verdict}")

    print("\nOne confound, stated rather than buried: switching coordination on also "
          "changes\nWHICH accounts are used. A coordinated batch draws hosts that were "
          "live shortly\nbefore its window, so the two mule variants do not share a host "
          "population. Any\ndifference between them is timing AND host mix, and this "
          "design cannot separate them.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "coordination_check.json").write_text(json.dumps({
        "seeds": list(SEEDS), "fprs": list(FPRS), "n_eval_per_variant": N_EVAL,
        "protocol": "single-pass detector (one round of attacks, one retrain) per seed, "
                    "so the variants differ in exactly one thing; deltas are computed "
                    "per seed and then averaged, so the spread is of the difference",
        "recall_mean": {n: {str(f): round(stats[n][f][0], 4) for f in FPRS} for n in names},
        "recall_sd": {n: {str(f): round(stats[n][f][1], 4) for f in FPRS} for n in names},
        "comparisons": findings,
    }, indent=2))
    print(f"\n-> {RESULTS/'coordination_check.json'}")


if __name__ == "__main__":
    main()
