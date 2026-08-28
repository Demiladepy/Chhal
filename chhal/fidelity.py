"""Fidelity of simulation — a metric, not a claim.

"Fidelity of simulation" is a judged criterion. A team that *says* its attacks are
realistic loses to one that *measures* it. We measure two things that matter, and frame
them honestly:

  1. **Plausibility / on-manifold rate** — the fraction of (already-clipped) attack feature
     values that fall inside the realistic manifold bounds [q0.5%, q99.5%] of the base data.
     Because the optimizer hard-clips every candidate to these EXACT bounds
     (`EvasionOptimizer._clip_to_manifold`), this is ~1.0 by construction — it proves the
     clip is wired correctly, not that the guardrail did meaningful work. Target: ~100%.

  1b. **Guardrail binding rate** (`frac_off_manifold_pre_clip`, tracked in optimizer.py and
     surfaced in loop.py) — the fraction of proposed perturbations that landed OUTSIDE the
     manifold *before* clipping and had to be pulled back. This is the non-tautological
     evidence the guardrail actually does work: it measures how often the search pushed
     against the plausibility envelope, not whether the output technically satisfies it.

  2. **Per-vector distribution distance from legitimate traffic** (two-sample KS vs legit).
     Fraud is *supposed* to differ from normal — that difference is the fraud signal, not a
     simulation defect. The insight the number captures: the hero vector `threshold_hugging`
     sits **closest to legit** (it mimics normal behaviour — that is why it is hardest to
     catch), while overt vectors (`bustout`, `card_testing`) sit further out by design. Low
     KS on the mimicry vector is a quantified proof the mimicry is real.

Note: these numbers are now measured against REAL payment data by default — 590,540
IEEE-CIS card transactions (Vesta), 3.499% fraud, over 182 days, with a temporal split.
See data.py. The KS distances below are therefore distances from real legitimate
traffic, not from a distribution we invented. Runs against the synthetic fallback are
labelled `data_source: "synthetic"` in results/summary.json and must not be quoted as
fidelity evidence.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .contract import ATTACKER_DIRECT, DERIVED_FEATURES, FEATURE_COLUMNS

MIMICRY_VECTOR = "threshold_hugging"

# The columns the red team actually sets or produces. The mean KS over all 26 features
# is a flattering number and it is worth being explicit about why: 16 of the 26 are
# inherited whole from a real host account, so they match legitimate traffic by
# construction and contribute a KS of roughly zero each. Averaging them in is 62% free
# passes. Every mimicry claim is therefore reported twice — over all features, and over
# these, which is where mimicry either happened or did not.
CONTROLLED_FEATURES: List[str] = list(dict.fromkeys(ATTACKER_DIRECT + DERIVED_FEATURES))

# The KS a feature scores when BOTH samples are legitimate traffic — the noise floor.
#
# `matches_legit` used to be `stat < 0.1`, a constant, and the constant is wrong in a way
# that is easy to miss: two samples drawn from the SAME legit population do not score
# zero. Measured legit-vs-legit at n=m=500 (scripts/audit/ks_null_floor.py) the floor
# spans nearly twenty-fold across the feature space, from `is_cross_border` at 0.0022 to
# `amount_to_avg_ratio` at 0.0411, mean ~0.024. So `< 0.1` meant "within 2.4x of pure
# noise" on one row of the table and "within 45x" on another, in the same table, and the
# rows were being read as if they said the same thing.
#
# Report multiples of the floor instead. This is also the metric Sajja, arXiv 2604.13125
# (13 Apr 2026), "Synthetic Tabular Generators Fail to Preserve Behavioral Fraud
# Patterns", publishes on IEEE-CIS — degradation over the noise floor — four months
# before this repo. Matching it is what makes our fidelity numbers comparable to theirs
# rather than merely adjacent.
KS_NULL_FLOOR_N = 500              # the n at which the floors below were measured
KS_NULL_FLOOR = {
    "amount_to_avg_ratio": 0.0411,
    "amount": 0.0374,
    "time_since_last_txn_min": 0.0318,
    "hour": 0.0297,
    "day_of_week": 0.0277,
    "is_new_beneficiary": 0.0198,
    "channel_code": 0.0162,
    "velocity_1h": 0.0159,
    "velocity_24h": 0.0153,
    "is_cross_border": 0.0022,
}
# For the sixteen inherited columns (linkage counts, account age, merchant risk) the
# floor was not measured per column; the n=500 mean is the honest stand-in, and those
# columns match legit by construction anyway.
KS_NULL_FLOOR_DEFAULT = 0.024
# "Matches legit" now means "within this many multiples of that feature's own noise
# floor" rather than "under a fixed 0.1".
#
# Why 2.0 and not 1.0: the floors above are MEAN null KS values, so a single legit-vs-legit
# comparison lands above its own floor roughly half the time by construction. At 2x the
# rule reads "within twice the expected noise", and measured on two 500-row legit draws it
# passes 22 of 26 features at a mean ratio of 1.44 — against 13.51 for real IEEE-CIS fraud
# over the same features. The remaining four are single-draw fluctuation, not a defect, and
# the ratio column is there precisely so a reader can see that rather than trust the flag.
LEGIT_RATIO = 2.0


def ks_null_floor(feature: str, n: int, m: int) -> float:
    """The legit-vs-legit KS for `feature` at sample sizes n and m.

    The floor is not a constant across sample size either — it fell from ~0.024 at
    n=m=500 to ~0.008 at n=5,100 in the same measurement. So the measured value is
    rescaled by the two-sample KS null's own dependence on sample size,
    sqrt((n+m)/nm), instead of being applied flat. A 500-row attack batch and a
    5,100-row one are then judged against their own floors rather than each other's.
    """
    base = KS_NULL_FLOOR.get(feature, KS_NULL_FLOOR_DEFAULT)
    if n <= 0 or m <= 0:
        return base
    reference_scale = np.sqrt(2.0 / KS_NULL_FLOOR_N)          # n = m = KS_NULL_FLOOR_N
    return float(base * np.sqrt((n + m) / (n * m)) / reference_scale)


def ks_table(reference: pd.DataFrame, sample: pd.DataFrame,
             features: List[str] | None = None) -> pd.DataFrame:
    """Per-feature two-sample KS between a reference population and a sample.

    KS in [0, 1]; lower = closer distributions. p > 0.05 means we cannot reject
    "same distribution" for that feature.

    `ks_stat` is the raw distance and is what it always was. `degradation_ratio` is that
    distance in multiples of the feature's own legit-vs-legit noise floor, which is the
    only one of the two that is comparable ACROSS features — see KS_NULL_FLOOR. A ratio
    near 1.0 means the sample is as close to legit as legit is to itself.
    """
    features = features or FEATURE_COLUMNS
    rows = []
    for col in features:
        ref, smp = reference[col].to_numpy(), sample[col].to_numpy()
        stat, p = ks_2samp(ref, smp)
        floor = ks_null_floor(col, len(smp), len(ref))
        rows.append({"feature": col, "ks_stat": float(stat), "p_value": float(p),
                     "ks_null_floor": round(floor, 4),
                     "degradation_ratio": round(float(stat) / floor, 2) if floor > 0
                                          else float("inf"),
                     "matches_legit": bool(float(stat) <= LEGIT_RATIO * floor)})
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)


def _cell_rate(sample: pd.DataFrame, feature_stats: pd.DataFrame, columns) -> float:
    lo, hi = feature_stats.loc[0.005], feature_stats.loc[0.995]
    cols = [c for c in columns if c in sample.columns and c in lo.index]
    if not cols or len(sample) == 0:
        return 1.0
    inside = np.ones((len(sample), len(cols)), dtype=bool)
    for j, col in enumerate(cols):
        inside[:, j] = (sample[col] >= lo[col] - 1e-9) & (sample[col] <= hi[col] + 1e-9)
    return float(inside.mean())


def _rows_fully_inside(sample: pd.DataFrame, feature_stats: pd.DataFrame, columns) -> float:
    """Fraction of ROWS with every one of `columns` inside the manifold.

    A cell rate of 0.9964 sounds like nothing escapes; measured per row it means 6.1% of
    rows carry an off-manifold value somewhere. Both are true and only one of them is
    the question a reader is asking.
    """
    lo, hi = feature_stats.loc[0.005], feature_stats.loc[0.995]
    cols = [c for c in columns if c in sample.columns and c in lo.index]
    if not cols or len(sample) == 0:
        return 1.0
    inside = np.ones(len(sample), dtype=bool)
    for col in cols:
        inside &= ((sample[col] >= lo[col] - 1e-9) & (sample[col] <= hi[col] + 1e-9)).to_numpy()
    return float(inside.mean())


def on_manifold_rate(sample: pd.DataFrame, feature_stats: pd.DataFrame) -> float:
    """Fraction of (row, directly-set feature) cells inside the manifold bounds.

    Scope is the four features an attacker actually SETS — amount, the payee flag, the
    rail, the destination. Those are the only ones the guardrail governs, and the only
    ones it can govern: the behavioural block is recomputed from a timeline rather than
    assigned, so there is nothing there to clip. See `derived_on_manifold_rate` for where
    that block lands, and report the two together — quoting this number alone invites the
    obvious question of what the other columns are doing.

    Note: when `sample` is the optimizer's OUTPUT, this is ~1.0 BY CONSTRUCTION — the
    optimizer hard-clips every candidate to exactly these bounds, so this only confirms
    the clip is wired correctly, not that the guardrail did meaningful work. For that,
    see `frac_off_manifold_pre_clip` in each AttackBatch.provenance (optimizer.py).
    """
    return _cell_rate(sample, feature_stats, ATTACKER_DIRECT)


def derived_on_manifold_rate(sample: pd.DataFrame, feature_stats: pd.DataFrame) -> float:
    """Fraction of DERIVED cells that happen to fall inside the legit manifold.

    Deliberately not a guardrail and deliberately not ~1.0. These columns are what a
    timeline looks like once it has happened, so constraining them would mean forbidding
    an attack from having the shape it really has — a card-testing run that probes forty
    times in ten minutes HAS a velocity above anything legitimate traffic shows, and
    clipping it back into the legit envelope while still calling it card testing is the
    lie this measurement exists to prevent. A low number here is evidence the attack is
    aggressive, not evidence it is fake.
    """
    return _cell_rate(sample, feature_stats, DERIVED_FEATURES)


def ks_by_vector(legit: pd.DataFrame, attacks: pd.DataFrame,
                 attack_vectors: np.ndarray) -> pd.DataFrame:
    """Mean KS distance from legit traffic, per attack vector. Lower = more legit-like."""
    rows = []
    for v in sorted(np.unique(attack_vectors)):
        sub = attacks[attack_vectors == v]
        ks = ks_table(legit, sub)
        controlled = ks[ks["feature"].isin(CONTROLLED_FEATURES)]
        rows.append({
            "vector": v,
            "mean_ks_vs_legit": round(float(ks["ks_stat"].mean()), 4),
            # the same distance restricted to what the red team controls — see
            # CONTROLLED_FEATURES for why the two differ by so much
            "mean_ks_controlled": round(float(controlled["ks_stat"].mean()), 4),
            # the same two distances in multiples of the noise floor. Quote these when
            # comparing vectors to each other or to a published number; quote the raw
            # KS only when comparing a vector to itself across runs.
            "mean_degradation_ratio": round(float(ks["degradation_ratio"].mean()), 2),
            "mean_degradation_ratio_controlled": round(
                float(controlled["degradation_ratio"].mean()), 2),
            "features_like_legit": int(ks["matches_legit"].sum()),
            "controlled_like_legit": int(controlled["matches_legit"].sum()),
            "n_controlled": int(len(controlled)),
        })
    return pd.DataFrame(rows).sort_values("mean_ks_vs_legit").reset_index(drop=True)


def fidelity_report(legit: pd.DataFrame, attacks: pd.DataFrame,
                    attack_vectors: np.ndarray, feature_stats: pd.DataFrame) -> dict:
    """Headline fidelity numbers for the deck / dashboard."""
    per_vector = ks_by_vector(legit, attacks, attack_vectors)
    mimic_mask = attack_vectors == MIMICRY_VECTOR
    mimic_ks = ks_table(legit, attacks[mimic_mask]) if mimic_mask.any() else pd.DataFrame()
    return {
        "on_manifold_rate": round(on_manifold_rate(attacks, feature_stats), 4),
        # reported alongside, always: the guardrail governs what the attacker sets, and
        # this says where the block it does NOT govern actually landed.
        "derived_on_manifold_rate": round(
            derived_on_manifold_rate(attacks, feature_stats), 4),
        "mimicry_vector": MIMICRY_VECTOR,
        # on-manifold over EVERY feature, not just the four the guardrail governs: the
        # headline 1.0000 is scoped to those four, and a reader is entitled to the
        # number that includes the columns it does not touch.
        "on_manifold_rate_all_features": round(
            _cell_rate(attacks, feature_stats, FEATURE_COLUMNS), 4),
        "rows_fully_on_manifold": round(float(_rows_fully_inside(
            attacks, feature_stats, FEATURE_COLUMNS)), 4),
        "mimicry_vector_note": (
            "mean_ks_vs_legit averages 26 features, 16 of which are inherited from a "
            "real host and match by construction; mean_ks_controlled is the same "
            "distance over the columns the red team actually sets."),
        "mimicry_mean_ks_vs_legit": float(per_vector.loc[
            per_vector["vector"] == MIMICRY_VECTOR, "mean_ks_vs_legit"].iloc[0])
            if (per_vector["vector"] == MIMICRY_VECTOR).any() else None,
        "mimicry_mean_ks_controlled": float(per_vector.loc[
            per_vector["vector"] == MIMICRY_VECTOR, "mean_ks_controlled"].iloc[0])
            if (per_vector["vector"] == MIMICRY_VECTOR).any() else None,
        "per_vector": per_vector,          # DataFrame — popped out by the caller
        "mimicry_ks_table": mimic_ks,      # DataFrame — popped out by the caller
    }


def plot_mimicry(legit: pd.DataFrame, mimic_attacks: pd.DataFrame, out_path: str,
                 features: List[str] = ("amount", "velocity_24h",
                                        "amount_to_avg_ratio", "hour")) -> str:
    """Legit vs the mimicry vector — overlapping mass is the fidelity picture for the deck."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), features):
        ref = legit[col].to_numpy()
        atk = mimic_attacks[col].to_numpy()
        lo = min(ref.min(), atk.min())
        hi = np.quantile(np.concatenate([ref, atk]), 0.99)
        bins = np.linspace(lo, hi, 40)
        ax.hist(ref, bins=bins, density=True, alpha=0.55, label="legit traffic", color="#2980b9")
        ax.hist(atk, bins=bins, density=True, alpha=0.55, label="threshold-hugging", color="#c0392b")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Fidelity: the mimicry vector (threshold-hugging) vs legitimate traffic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
