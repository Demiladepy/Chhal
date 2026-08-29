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
A block of k attack transactions has to be cut from k+2 real ones — k that are actually
replayed, one whose gap is read and discarded, and one held back so the victim's last real
transaction is never copied — and IEEE-CIS accounts are short: the eligible test hosts have
a median of TWO transactions each. So on the pool the loop actually uses, only a small
minority of campaigns can replay at all and the rest fall back to mimicry. Both the k+2
rate and the k+1 rate the arithmetic strictly needs are printed, so the price of that
last-transaction guarantee is visible instead of folded into the headline. That is not an implementation limit, it is the same limit a real
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
     by accident — and scored against a MEASURED ceiling rather than against the victim's
     full history. A three-to-nine transaction slice of a real series does not carry that
     series' dispersion or autocorrelation, so the obvious references (gap CV ratio 1.00,
     autocorrelation gap 0.00) are unreachable by any replay, and `ceiling_stats` cuts
     real uncopied blocks to find out what is actually reachable.

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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chhal.behaviour import hour_of                               # noqa: E402
from chhal.contract import (FEATURE_COLUMNS, INHERITED_FEATURES,   # noqa: E402
                            LABEL_COLUMN)
from chhal.data import load_base_data                             # noqa: E402
from chhal.detector import Detector                               # noqa: E402
from chhal.evaluation import threshold_for_fpr                    # noqa: E402
from chhal.fidelity import CONTROLLED_FEATURES, ks_table          # noqa: E402
from chhal.optimizer import EvasionOptimizer                      # noqa: E402
from chhal.redteam.campaign import (MIN_HISTORY_TO_REPLAY,        # noqa: E402
                                    REPLAY_JITTER, REPLAY_SCALE)
from chhal.redteam.base import BaseProfile                        # noqa: E402
from chhal.redteam.hosts import HostPool                          # noqa: E402
from chhal.redteam.vectors import ThresholdHugging, TrajectoryReplay   # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
FPR = 0.001
# Twelve, not six. The paired interval scales with 1/sqrt(seeds) and six left it at
# +-0.6% on a 12.7% level -- enough to see a large effect, not enough to call the
# absence of one a result. The whole probe is eighty seconds a pool; there is no
# reason to be underpowered about the one comparison it exists for.
SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
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

# The realism statistics run on their own generator so both vectors see the same victims;
# offset off the seed so they are still reproducible and still vary across seeds.
SEQ_SEED_OFFSET = 100_000

CEILING = "(a real block of the victim's own past)"


def replay_feasibility(pool: HostPool, temporal, margin: int = 2) -> float:
    """The share of campaigns that can replay at all on this pool.

    Exact rather than estimated: `HostPool.sample` is uniform over hosts and the campaign
    length is uniform over the vector's band, so this is the generator's own condition
    `len(history) >= max(MIN_HISTORY_TO_REPLAY, n + margin)` averaged over both.

    `margin=2` is what the generator asks for. `margin=1` is what the arithmetic strictly
    needs, and the gap between the two is the price of never copying the victim's last
    real transaction — reported so the cost of that guarantee is visible rather than
    folded silently into the headline feasibility rate.
    """
    lens = pool._ends - pool._starts
    lo, hi = temporal.txns_per_entity
    return float(np.mean([[l >= max(MIN_HISTORY_TO_REPLAY, n + margin)
                           for n in range(lo, hi + 1)] for l in lens]))


def _lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of log amount. Independent draws have none by definition."""
    z = np.log(np.maximum(x, 1e-9))
    z = z - z.mean()
    d = (z ** 2).sum()
    return float((z[:-1] * z[1:]).sum() / d) if d > 0 else np.nan


def ceiling_stats(hosts: HostPool, temporal, rng, n_blocks: int = 8_000) -> dict:
    """What a PERFECT replay could score. Measured, not assumed — and it is not 1.00.

    The obvious reference for "does this campaign behave like this card behaves" is the
    card's own full history: gap CV ratio 1.00, autocorrelation gap 0.00. That reference
    is wrong, and wrong in a direction that understates the result.

    A campaign is three to nine transactions. A contiguous slice that short does not carry
    the whole series' dispersion or its autocorrelation — a card that goes quiet for a
    week and then spends three times in an evening has a high CV over a year and a much
    lower one inside any one window of it. So even a replay that copied a real block with
    no jitter and no scaling at all would not score 1.00, and scoring it against 1.00
    marks a perfect copy as two-thirds of the way there.

    This measures the real ceiling directly: draw hosts the way the generator does, cut
    real blocks of the same lengths, apply the same `REPLAY_SCALE` and `REPLAY_JITTER`,
    and score them on the same statistics. Nothing is generated, so whatever comes out is
    what copying can achieve on this pool, and the two vectors are then read against it.

    One caveat, and it is why the gated pool is the one the README quotes: a block can
    only be cut from a host long enough to cut one from, so on an ungated pool this is the
    ceiling for the replayable MINORITY while the vectors are scored over every campaign,
    most of which fell back to mimicry. On the gated pool every campaign replays and the
    two are measured over the same hosts.
    """
    lo, hi = temporal.txns_per_entity
    cv, ac, hr = [], [], []
    for i in rng.integers(0, len(hosts), n_blocks):
        h = hosts._host(int(i))
        ts, amt = h.history_ts.astype(np.float64), h.history_amount.astype(np.float64)
        n = int(rng.integers(lo, hi + 1))
        # the same filters `sequence_stats` applies, so the numbers are comparable
        if n < 4 or len(ts) < 4 or len(ts) < n + 2:
            continue
        rg = np.diff(ts)
        if rg.mean() <= 0 or rg.std() <= 0:
            continue
        j = int(rng.integers(0, len(ts) - n - 1))
        bg = np.maximum(np.diff(ts[j:j + n]) * rng.uniform(*REPLAY_JITTER, n - 1), 1.0)
        if bg.mean() > 0:
            cv.append((bg.std() / bg.mean()) / (rg.std() / rg.mean()))
        r, b = _lag1(amt), _lag1(amt[j:j + n] * rng.uniform(*REPLAY_SCALE))
        if np.isfinite(r) and np.isfinite(b):
            ac.append(b - r)
        # The hour ceiling is measured under the SAME jitter rather than set to 100%.
        # An unjittered copy would trivially score 100% — its hours ARE the victim's —
        # but the generator jitters every gap, and +-10% of a week is +-17 hours. What
        # this asks is how much of the hour alignment survives that, which is the only
        # version of the number the two vectors can fairly be read against.
        vh = set(int(x) for x in hour_of(h.history_ts))
        blk = (ts[j] + np.r_[0.0, np.cumsum(bg)]).astype(np.int64)
        hr.extend(int(x) in vh for x in hour_of(blk)[1:])
    return {"gap_cv_vs_victim": float(np.median(cv)),
            "lag1_autocorr_gap": float(np.median(ac)),
            "later_txns_on_a_victim_hour": float(np.mean(hr)),
            "takes_more_than_victim_ever_did": np.nan}


def sequence_stats(vector_cls, profile, hosts, rng) -> dict:
    """Realism on statistics the amount and gap MARGINALS do not constrain.

    Both vectors match the victim's marginals by construction, so a marginal comparison
    cannot separate them and is not evidence either way. The first three below are
    functions of the ORDER of a campaign, which is precisely what independent draws throw
    away.

    Each is expressed relative to the victim's own real history, so both vectors are
    scored on one scale — and each is read against `ceiling_stats`, not against the
    victim's full history, because a three-to-nine transaction slice cannot reach the
    latter even when it is a genuine copy.
    """
    v = vector_cls().calibrate(profile, hosts)
    _, camp = v.render_with_timeline(N, rng)
    ratio = v.build_frame(camp, rng)["amount_to_avg_ratio"].to_numpy()

    cv, ac, over, hr = [], [], [], []
    for e in np.unique(camp.entity):
        m = camp.entity == e
        h, a = m & ~camp.is_attack, m & camp.is_attack
        h_ts, a_ts = camp.timestamp_s[h].astype(float), camp.timestamp_s[a].astype(float)
        h_amt, a_amt = camp.amount[h], camp.amount[a]
        if len(a_ts) < 4 or len(h_ts) < 4:
            continue

        # 1. burstiness. `_host_gaps` draws every gap from the victim's 25th-75th
        #    percentile band, so a mimicry campaign has an unnaturally even cadence: real
        #    cards go quiet for a week and then spend three times in an evening.
        rg, ag = np.diff(h_ts), np.diff(a_ts)
        if rg.mean() > 0 and ag.mean() > 0 and rg.std() > 0:
            cv.append((ag.std() / ag.mean()) / (rg.std() / rg.mean()))

        # 2. lag-1 autocorrelation of log amount, campaign minus victim. Independent
        #    draws have none by definition; a real card has whatever it has. NOISY: a
        #    campaign is 3-9 amounts long, so this is a weak statistic and is reported for
        #    completeness rather than relied on.
        r_ac, a_ac = _lag1(h_amt), _lag1(a_amt)
        if np.isfinite(r_ac) and np.isfinite(a_ac):
            ac.append(a_ac - r_ac)

        # 3. how often a campaign transaction AFTER the first lands on an hour of day the
        #    victim genuinely transacts at. The two vectors keep the clock differently:
        #    mimicry snaps each transaction onto one of the victim's hours, replay lets
        #    the block's own gaps carry it. Replay should win outright and does not
        #    entirely, because REPLAY_JITTER is multiplicative and +-10% of a week-long
        #    gap is +-17 hours. The first transaction is excluded because `_phase_align`
        #    places it on the block's own hour by construction and it would score ~100%
        #    for a reason that says nothing about the sequence.
        vh = set(int(x) for x in hour_of(camp.timestamp_s[h]))
        hr.extend(int(x) in vh for x in hour_of(camp.timestamp_s[a])[1:])

        # 4. how often the campaign's peak `amount_to_avg_ratio` goes above anything this
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
            "later_txns_on_a_victim_hour": float(np.mean(hr)),
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
            replay_feasible_min_margin=replay_feasibility(hosts, V.temporal, margin=1),
            as_generated=float((det.score(rows) >= thr).mean()),
            inherited_from_fraud=transplant(INHERITED_FEATURES),
            mean_ks_controlled=float(ctrl["ks_stat"].mean()),
            mean_degradation_controlled=float(ctrl["degradation_ratio"].mean()),
        ))
        # A FRESH generator, identical for both vectors, rather than the running `rng`.
        # The detection comparison above is paired on seed, detector, threshold and
        # transplant donors; leaving the realism statistics on the running stream would
        # have handed the two vectors different victims and left the one comparison that
        # actually separates them as the only unpaired thing in the script.
        seq.append(dict(pool=pool_name, seed=seed, vector=label,
                        **sequence_stats(V, profile, hosts,
                                         np.random.default_rng(SEQ_SEED_OFFSET + seed))))

    seq.append(dict(pool=pool_name, seed=seed, vector=CEILING,
                    **ceiling_stats(hosts, PAIR[1][1].temporal,
                                    np.random.default_rng(SEQ_SEED_OFFSET + seed))))
    detect.append(dict(pool=pool_name, seed=seed, vector="(real fraud itself)",
                       replay_feasible=np.nan, replay_feasible_min_margin=np.nan,
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
    feas1 = d.loc[PAIR[1][0]][("replay_feasible_min_margin", "mean")]

    print(f"\n{'=' * 86}\n=== POOL: {pool_name}   "
          f"(campaigns able to replay at all: {feas:.1%}; {feas1:.1%} without the "
          f"held-back last txn)\n{'=' * 86}")

    print("\n  1. is the sequence more realistic?  "
          "(statistics the marginals do not constrain)\n")
    print(f"  {'':42s} {'gap CV / victim':>16s} {'lag-1 ac gap':>13s} "
          f"{'later txns on a':>16s} {'over victim':>12s}")
    print(f"  {'':42s} {'':>16s} {'':>13s} {'victim hour':>16s} {'peak':>12s}")
    for label in (CEILING, *[l for l, _ in PAIR]):
        r = s.loc[label]
        over = r[("takes_more_than_victim_ever_did", "mean")]
        print(f"  {label:42s} {r[('gap_cv_vs_victim', 'mean')]:16.2f} "
              f"{r[('lag1_autocorr_gap', 'mean')]:13.2f} "
              f"{r[('later_txns_on_a_victim_hour', 'mean')]:15.1%} "
              f"{('—' if not np.isfinite(over) else f'{over:.1%}'):>12s}")
    print("\n  The top row is the CEILING, not an ideal: it is what real, uncopied blocks "
          "of these\n  victims' own histories score on the same statistics. A campaign is "
          "3-9 transactions,\n  and a slice that short does not carry the whole series' "
          "dispersion or autocorrelation,\n  so 1.00 and 0.00 are unreachable by any "
          "replay and scoring against them would mark a\n  perfect copy as two-thirds of "
          "the way there.")

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
    ceil_cv = s.loc[CEILING][("gap_cv_vs_victim", "mean")]
    cv_m, cv_r = [abs(s.loc[label][("gap_cv_vs_victim", "mean")] - ceil_cv)
                  for label, _ in PAIR]
    gen = max(mim[("as_generated", "mean")], rep[("as_generated", "mean")])
    tr_m, tr_r = mim[("inherited_from_fraud", "mean")], rep[("inherited_from_fraud", "mean")]

    print("\n  reading")
    print(f"    realism   : against the {ceil_cv:.2f} ceiling, replay's cadence is off by "
          f"{cv_r:.2f} and mimicry's\n                by {cv_m:.2f}.")
    print(f"                {'Copying reaches what copying can reach; resampling does not.' if cv_r < cv_m else 'Copying buys no cadence realism here.'}")
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
    n = len(delta)
    # Student's t, not 1.96. With six seeds the normal quantile understates the interval
    # by about a third, which is exactly the direction that would flatter a null result.
    tcrit = float(stats.t.ppf(0.975, n - 1))
    ci = tcrit * d_sem
    pval = float(stats.ttest_rel(wide[PAIR[0][0]], wide[PAIR[1][0]]).pvalue)
    ratio = tr_m / tr_r if tr_r > 0 else float("inf")

    print(f"    ceiling   : with the inherited block transplanted from real fraud, "
          f"mimicry is caught\n                {tr_m:.2%} and replay {tr_r:.2%} — paired "
          f"difference {d_mean:+.2%}, 95% CI +-{ci:.2%}\n                "
          f"(t_{{.975,{n - 1}}}={tcrit:.2f}, {n} seeds, p={pval:.2f}).")
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

    first = detect[(detect["pool"] == pools[0][0]) & (detect.vector == PAIR[1][0])]
    loose, loose1 = first["replay_feasible"].mean(), first["replay_feasible_min_margin"].mean()
    print(f"\n{'=' * 86}\n  Standing result, independent of any detector: on the pool the "
          f"loop actually uses,\n  only {loose:.1%} of campaigns have a victim with enough "
          "history to replay. IEEE-CIS\n  accounts are short — the eligible test hosts "
          "have a median of two transactions —\n  so a sequence-level attack has almost no "
          "material to work with on this population,\n  and neither would a real attacker "
          "reading those statements. Dropping the held-back\n  last transaction, which is "
          f"the loosest bound the arithmetic allows, moves it to {loose1:.1%};\n  the "
          "shortage is the population, not the margin.")
    print(f"\n-> {RESULTS / 'trajectory_replay_detection.csv'}")
    print(f"-> {RESULTS / 'trajectory_replay_sequence.csv'}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
