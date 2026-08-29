"""Does copying a victim's real trajectory beat resampling its marginals?

A PROBE, and the prediction is written down before the measurement.

`mimic_host` reads a victim's amount and gap distributions and then draws from them
INDEPENDENTLY. That matches the marginals and destroys everything else. A card's spending
is not i.i.d.: a large purchase is followed by a quiet week, an evening burst repeats
weekly, a subscription lands on the same day each month. Draw six amounts independently
from a card's own quantile band and you can produce its 90th-percentile spend six times
running — every value individually unremarkable, the sequence something that card has
never once done. Sajja et al. (arXiv 2604.13125) make the general version of this
argument: a generator that matches marginals cannot preserve joint structure, and the gap
is measurable.

`replay_host` is the constructive answer. Do not approximate the joint — copy it, from the
one account entitled to it. A contiguous block of the victim's real history: its own gap
sequence with per-gap jitter, its own amount sequence under one shared scale factor,
started on the weekday and hour the block originally ran at. The attacker needs statement
access, which is exactly what an account takeover provides; it is not a stronger
assumption than `mimic_host`, only a better use of the same one.

THE PREDICTION
--------------
Recall does not move off the false-positive budget.

`why_the_attacks_score_zero.py` established that the ten columns the red team controls
carry no usable signal in this setup at all: replace every one of them with values drawn
from REAL FRAUD and recall stays at 0.00%. A better sequence model improves exactly those
ten columns and nothing else. So the headline cannot move, and if it does, the correct
reading is that the earlier result is wrong — not that this vector is good.

THE FIRST RESULT IS THAT MOST VICTIMS CANNOT BE REPLAYED
--------------------------------------------------------
A block of k attack transactions has to be cut from k+2 real ones, and IEEE-CIS accounts
are short: the eligible test hosts have a median of TWO transactions each. So on the pool
the loop actually uses, only a small minority of campaigns can replay at all and the rest
fall back to mimicry. That is not an implementation limit, it is the same limit a real
attacker faces — you cannot replay a statement with three lines on it — and it caps what
any sequence-based attack can do on this population before any detector is involved.

It also means a comparison run on that pool is mostly mimicry against mimicry and cannot
answer anything. So both vectors are additionally run on a GATED pool of hosts long enough
that every campaign replays. Same victims for both vectors, so the comparison stays
controlled; the price is that those hosts are a small and unrepresentative slice of the
data, and the feasibility rate below is what says how small.

WHAT THE PROBE CAN STILL DECIDE
-------------------------------
  1. Is the sequence actually more realistic? Measured on statistics the marginals do not
     constrain, so a vector that matched only the marginals could not score well on them
     by accident.

  2. Does that realism buy anything once the ceiling is removed? Transplant the inherited
     block from real fraud, as in experiment E of `why_the_attacks_score_zero.py`. That
     makes the campaigns detectable at all, which means the controlled columns are finally
     visible — and it is the only place a difference between the two vectors could show
     up. If replay is genuinely stealthier, its transplanted recall is LOWER than
     mimicry's. If the two land together, then better sequence modelling buys nothing even
     with the ceiling removed, and that is the finding.

`TrajectoryReplay` is `ThresholdHugging` with one flag changed and nothing else: same
payee rate, same static features, same host pool, same campaign sizes. The comparison is
controlled, not two vectors that happen to differ.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chhal.contract import (FEATURE_COLUMNS, INHERITED_FEATURES,   # noqa: E402
                            LABEL_COLUMN)
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.evaluation import threshold_for_fpr                    # noqa: E402
from chhal.fidelity import CONTROLLED_FEATURES, ks_table          # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam.campaign import MIN_HISTORY_TO_REPLAY          # noqa: E402
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402
from chhal.redteam.vectors import ThresholdHugging, TrajectoryReplay   # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
FPR = 0.001
SEEDS = (7, 11, 13, 17, 19, 23)
N = 1000
PAIR = (("threshold_hugging (resamples marginals)", ThresholdHugging),
        ("trajectory_replay (copies the joint)", TrajectoryReplay))

# Enough history that EVERY campaign can cut a block: the vector runs up to 9 attack
# transactions and `_host_trajectory` needs n + 2 real ones.
MIN_REPLAYABLE = max(MIN_HISTORY_TO_REPLAY, TrajectoryReplay.temporal.txns_per_entity[1] + 2)

# A recall at or below the false-positive budget means the detector is flagging these rows
# at the same rate it flags ordinary legitimate traffic. Three times the budget is the
# line for "something is actually being detected"; below it, single flagged rows out of a
# thousand are noise and should not be read as the prediction breaking.
NOISE_MULTIPLE = 3.0

# How much lower the replay's transplanted recall has to be before we call it stealthier.
# A ratio, not a percentage-point gap: both sit in the low tens of percent, where five
# points is either noise or a third of the effect depending on where you are.
STEALTH_RATIO = 1.15


def replay_feasibility(pool: HostPool, temporal) -> float:
    """The share of campaigns that can replay at all on this pool.

    Exact rather than estimated: `HostPool.sample` is uniform over hosts and the campaign
    length is uniform over the vector's band, so this is the generator's own condition
    `len(history) >= max(MIN_HISTORY_TO_REPLAY, n + 2)` averaged over both.
    """
    lens = pool._ends - pool._starts
    lo, hi = temporal.txns_per_entity
    return float(np.mean([[l >= max(MIN_HISTORY_TO_REPLAY, n + 2) for n in range(lo, hi + 1)]
                          for l in lens]))


def sequence_stats(vector_cls, profile, hosts, rng) -> dict:
    """Realism on statistics the amount and gap MARGINALS do not constrain.

    Both vectors match the victim's marginals by construction, so a marginal comparison
    cannot separate them and is not evidence either way. All three below are functions of
    the ORDER of a campaign, which is precisely what independent draws throw away.

    Each is expressed relative to the victim's own real history, so the ideal value means
    "this campaign behaves like this card actually behaves" and both vectors are scored on
    one scale.
    """
    v = vector_cls().calibrate(profile, hosts)
    _, camp = v.render_with_timeline(N, rng)
    ratio = v.build_frame(camp, rng)["amount_to_avg_ratio"].to_numpy()

    cv, ac, over = [], [], []
    for e in np.unique(camp.entity):
        m = camp.entity == e
        h, a = m & ~camp.is_attack, m & camp.is_attack
        h_ts, a_ts = camp.timestamp_s[h].astype(float), camp.timestamp_s[a].astype(float)
        h_amt, a_amt = camp.amount[h], camp.amount[a]
        if len(a_ts) < 4 or len(h_ts) < 4:
            continue

        # 1. burstiness. `_host_gaps` draws every gap from the victim's 25th-75th
        #    percentile band, so a mimicry campaign has an unnaturally even cadence: real
        #    cards go quiet for a week and then spend three times in an evening. Ideal 1.
        rg, ag = np.diff(h_ts), np.diff(a_ts)
        if rg.mean() > 0 and ag.mean() > 0 and rg.std() > 0:
            cv.append((ag.std() / ag.mean()) / (rg.std() / rg.mean()))

        # 2. lag-1 autocorrelation of log amount, campaign minus victim. Independent draws
        #    have none by definition; a real card has whatever it has. Ideal 0. NOISY: a
        #    campaign is 3-9 amounts long, so this is a weak statistic and is reported for
        #    completeness rather than relied on.
        def lag1(x):
            z = np.log(np.maximum(x, 1e-9))
            z = z - z.mean()
            d = (z ** 2).sum()
            return float((z[:-1] * z[1:]).sum() / d) if d > 0 else np.nan
        r_ac, a_ac = lag1(h_amt), lag1(a_amt)
        if np.isfinite(r_ac) and np.isfinite(a_ac):
            ac.append(a_ac - r_ac)

        # 3. how often the campaign's peak `amount_to_avg_ratio` goes above anything this
        #    card has ever really done. NOT a joint-structure statistic and not scored as
        #    one: it is a marginal property, and mimicry wins it by construction because
        #    it draws inside the victim's 35th-75th percentile band and so can barely
        #    exceed a peak. What it measures for replay is REPLAY_SCALE — the block is the
        #    victim's own shape multiplied by up to 1.3, and a block containing the
        #    victim's largest transaction then clears their record. It is reported because
        #    that trade is worth naming: a replay at scale 1.0 is perfectly faithful and
        #    steals exactly what the victim would have spent anyway.
        h_r, a_r = ratio[h], ratio[a]
        if len(h_r) and np.isfinite(h_r).any():
            over.append(float(np.nanmax(a_r) > np.nanmax(h_r)))

    return {"gap_cv_vs_victim": float(np.median(cv)),
            "lag1_autocorr_gap": float(np.median(ac)),
            "takes_more_than_victim_ever_did": float(np.mean(over))}


def run_seed(seed: int, base, pool_name: str, min_history: int):
    rng = np.random.default_rng(seed)
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    hosts = HostPool(base.test, exclude_accounts=base.train["_account"],
                     min_history=min_history)
    det = Detector(seed=seed).fit(base.train, LABEL_COLUMN)
    opt = EvasionOptimizer(base.feature_stats)

    legit = base.test[base.test[LABEL_COLUMN] == 0]
    fraud = base.test[base.test[LABEL_COLUMN] == 1].reset_index(drop=True)
    thr = threshold_for_fpr(det.score(legit[FEATURE_COLUMNS]), FPR)

    # ONE donor draw, shared by both vectors. Drawn here rather than inside the loop so
    # the two transplants differ only in the campaigns they are applied to; a fresh draw
    # per vector would add unpaired noise to the one comparison this script exists for.
    donor_idx = rng.integers(0, len(fraud), N)

    detect, seq = [], []
    for label, V in PAIR:
        v = V().calibrate(profile, hosts)
        rows = opt.optimize(v.batch(N, 0, rng), det, rng).transactions

        def transplant(cols) -> float:
            """Swap `cols` for values drawn from real fraud, then rescore. Experiment E."""
            out = rows.reset_index(drop=True).copy()
            for c in cols:
                out[c] = fraud[c].to_numpy()[donor_idx[:len(out)]]
            return float((det.score(out) >= thr).mean())

        ctrl = ks_table(legit, rows)
        ctrl = ctrl[ctrl["feature"].isin(CONTROLLED_FEATURES)]
        detect.append(dict(
            pool=pool_name, seed=seed, vector=label,
            replay_feasible=replay_feasibility(hosts, V.temporal),
            as_generated=float((det.score(rows) >= thr).mean()),
            inherited_from_fraud=transplant(INHERITED_FEATURES),
            mean_ks_controlled=float(ctrl["ks_stat"].mean()),
            mean_degradation_controlled=float(ctrl["degradation_ratio"].mean()),
        ))
        seq.append(dict(pool=pool_name, seed=seed, vector=label,
                        **sequence_stats(V, profile, hosts, rng)))

    detect.append(dict(pool=pool_name, seed=seed, vector="(real fraud itself)",
                       replay_feasible=np.nan,
                       as_generated=float((det.score(fraud[FEATURE_COLUMNS]) >= thr).mean()),
                       inherited_from_fraud=np.nan, mean_ks_controlled=np.nan,
                       mean_degradation_controlled=np.nan))
    return pd.DataFrame(detect), pd.DataFrame(seq)


def report(pool_name: str, detect: pd.DataFrame, seq: pd.DataFrame) -> None:
    def by_vector(df):
        df = df[df["pool"] == pool_name].drop(columns=["pool", "seed"])
        return df.groupby("vector", sort=False).agg(["mean", "sem"])

    d, s = by_vector(detect), by_vector(seq)
    feas = d.loc[PAIR[1][0]][("replay_feasible", "mean")]

    print(f"\n{'=' * 78}\n=== POOL: {pool_name}   "
          f"(campaigns able to replay at all: {feas:.1%})\n{'=' * 78}")

    print("\n  1. is the sequence more realistic?  "
          "(statistics the marginals do not constrain)\n")
    print(f"  {'vector':42s} {'gap CV / victim':>16s} {'lag-1 ac gap':>14s} "
          f"{'over victim peak':>17s}")
    print(f"  {'ideal':42s} {'1.00':>16s} {'0.00':>14s} {'—':>17s}")
    for label, _ in PAIR:
        r = s.loc[label]
        print(f"  {label:42s} {r[('gap_cv_vs_victim', 'mean')]:16.2f} "
              f"{r[('lag1_autocorr_gap', 'mean')]:14.2f} "
              f"{r[('takes_more_than_victim_ever_did', 'mean')]:16.1%}")

    print("\n  2. does it change what the detector does?\n")
    print(f"  {'vector':42s} {'as generated':>13s} {'inherited<-fraud':>19s} "
          f"{'KS controlled':>15s}")
    for label, _ in PAIR:
        r = d.loc[label]
        print(f"  {label:42s} {r[('as_generated', 'mean')]:12.2%} "
              f"{r[('inherited_from_fraud', 'mean')]:14.2%} +-"
              f"{r[('inherited_from_fraud', 'sem')]:5.2%} "
              f"{r[('mean_ks_controlled', 'mean')]:15.4f}")
    print(f"  {'(real fraud itself)':42s} "
          f"{d.loc['(real fraud itself)'][('as_generated', 'mean')]:12.2%}")

    mim, rep = [d.loc[label] for label, _ in PAIR]
    cv_m, cv_r = [abs(s.loc[label][("gap_cv_vs_victim", "mean")] - 1.0) for label, _ in PAIR]
    gen = max(mim[("as_generated", "mean")], rep[("as_generated", "mean")])
    tr_m, tr_r = mim[("inherited_from_fraud", "mean")], rep[("inherited_from_fraud", "mean")]

    print("\n  reading")
    print(f"    realism   : replay's cadence sits {cv_r:.2f} from the victim's own, "
          f"mimicry's {cv_m:.2f}.")
    print(f"                {'The copied sequence is the more faithful one.' if cv_r < cv_m else 'Copying buys no cadence realism here.'}")
    if gen <= NOISE_MULTIPLE * FPR:
        print(f"    prediction: held. Neither vector is caught above {NOISE_MULTIPLE * FPR:.1%}, "
              f"which is the rate at\n                which this detector flags ordinary "
              "legitimate traffic. Nothing is being\n                detected, so there "
              "was nothing for a better sequence model to move.")
    else:
        print(f"    prediction: BROKEN — {gen:.2%} caught as generated, well above the "
              f"{FPR:.1%} budget.\n                That contradicts experiment E, and it "
              "is the earlier result that needs\n                re-checking rather than "
              "this one.")

    # PAIRED, not two independent levels. Both vectors run on the same seed, the same
    # detector, the same threshold and the same transplant donors, so the per-seed
    # difference cancels almost all of the variance that the two levels carry separately.
    # Comparing the levels would leave this comparison badly underpowered.
    wide = (detect[(detect["pool"] == pool_name) & (detect.vector != "(real fraud itself)")]
            .pivot(index="seed", columns="vector", values="inherited_from_fraud"))
    delta = wide[PAIR[0][0]] - wide[PAIR[1][0]]        # mimicry minus replay
    d_mean = float(delta.mean())
    d_sem = float(delta.sem())
    ci = 1.96 * d_sem
    ratio = tr_m / tr_r if tr_r > 0 else float("inf")

    print(f"    ceiling   : with the inherited block transplanted from real fraud, "
          f"mimicry is caught\n                {tr_m:.2%} and replay {tr_r:.2%} — paired "
          f"difference {d_mean:+.2%} +-{d_sem:.2%} "
          f"({len(delta)} seeds).")
    if abs(d_mean) > ci and ratio >= STEALTH_RATIO:
        print(f"                Mimicry is caught {ratio:.2f}x as often, so the copied "
              "trajectory IS\n                stealthier once the controlled columns can "
              "be seen at all — and that\n                would matter on any deployment "
              "where the attacker does not inherit a\n                known-good "
              "customer's context.")
    elif abs(d_mean) > ci and ratio <= 1 / STEALTH_RATIO:
        print(f"                Replay is caught {1 / ratio:.2f}x as often, the opposite "
              "of its claim: copying\n                a real block makes the campaign MORE "
              "visible once the controlled\n                columns can be seen.")
    else:
        print("                Zero is inside the interval, so the two land together. "
              "Even with the\n                ceiling removed, copying the victim's joint "
              "structure buys no evasion\n                over resampling its marginals — "
              "the sequence realism above is a fidelity\n                result, not a "
              "security one.")
        print(f"                Powered to rule out a paired gap beyond {ci:.2%}, so an "
              f"effect smaller\n                than that is not excluded and this is not "
              "a proof of no difference.")


def main() -> None:
    t0 = time.time()
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")
    print(f"[probe] {len(SEEDS)} seeds, {N} attack rows each, un-adapted detector — the "
          f"same setting\n        experiment E of why_the_attacks_score_zero.py ran in.")

    pools = (("as the loop uses it", 2), (f"gated to >={MIN_REPLAYABLE} real txns", MIN_REPLAYABLE))
    out = [run_seed(s, base, name, mh) for name, mh in pools for s in SEEDS]
    detect = pd.concat([d for d, _ in out], ignore_index=True)
    seq = pd.concat([s for _, s in out], ignore_index=True)

    RESULTS.mkdir(exist_ok=True)
    detect.to_csv(RESULTS / "trajectory_replay_detection.csv", index=False)
    seq.to_csv(RESULTS / "trajectory_replay_sequence.csv", index=False)

    for name, _ in pools:
        report(name, detect, seq)

    loose = detect[(detect["pool"] == pools[0][0])
                   & (detect.vector == PAIR[1][0])]["replay_feasible"].mean()
    print(f"\n{'=' * 78}\n  Standing result, independent of any detector: on the pool the "
          f"loop actually uses,\n  only {loose:.1%} of campaigns have a victim with enough "
          "history to replay. IEEE-CIS\n  accounts are short — the eligible test hosts "
          "have a median of two transactions —\n  so a sequence-level attack has almost no "
          "material to work with on this population,\n  and neither would a real attacker "
          "reading those statements.")
    print(f"\n-> {RESULTS / 'trajectory_replay_detection.csv'}")
    print(f"-> {RESULTS / 'trajectory_replay_sequence.csv'}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
