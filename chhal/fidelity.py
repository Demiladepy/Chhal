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

from .contract import FEATURE_COLUMNS

MIMICRY_VECTOR = "threshold_hugging"


def ks_table(reference: pd.DataFrame, sample: pd.DataFrame,
             features: List[str] | None = None) -> pd.DataFrame:
    """Per-feature two-sample KS between a reference population and a sample.

    KS in [0, 1]; lower = closer distributions. p > 0.05 means we cannot reject
    "same distribution" for that feature.
    """
    features = features or FEATURE_COLUMNS
    rows = []
    for col in features:
        stat, p = ks_2samp(reference[col].to_numpy(), sample[col].to_numpy())
        rows.append({"feature": col, "ks_stat": float(stat), "p_value": float(p),
                     "matches_legit": stat < 0.1})
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)


def on_manifold_rate(sample: pd.DataFrame, feature_stats: pd.DataFrame) -> float:
    """Fraction of (row, feature) cells inside the realistic manifold bounds.

    Note: when `sample` is the optimizer's OUTPUT, this is ~1.0 BY CONSTRUCTION — the
    optimizer hard-clips every candidate to exactly these bounds, so this only confirms
    the clip is wired correctly, not that the guardrail did meaningful work. For that,
    see `frac_off_manifold_pre_clip` in each AttackBatch.provenance (optimizer.py).
    """
    lo, hi = feature_stats.loc[0.005], feature_stats.loc[0.995]
    inside = np.ones((len(sample), len(FEATURE_COLUMNS)), dtype=bool)
    for j, col in enumerate(FEATURE_COLUMNS):
        inside[:, j] = (sample[col] >= lo[col] - 1e-9) & (sample[col] <= hi[col] + 1e-9)
    return float(inside.mean())


def ks_by_vector(legit: pd.DataFrame, attacks: pd.DataFrame,
                 attack_vectors: np.ndarray) -> pd.DataFrame:
    """Mean KS distance from legit traffic, per attack vector. Lower = more legit-like."""
    rows = []
    for v in sorted(np.unique(attack_vectors)):
        sub = attacks[attack_vectors == v]
        ks = ks_table(legit, sub)
        rows.append({
            "vector": v,
            "mean_ks_vs_legit": round(float(ks["ks_stat"].mean()), 4),
            "features_like_legit": int(ks["matches_legit"].sum()),
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
        "mimicry_vector": MIMICRY_VECTOR,
        "mimicry_mean_ks_vs_legit": float(per_vector.loc[
            per_vector["vector"] == MIMICRY_VECTOR, "mean_ks_vs_legit"].iloc[0])
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
