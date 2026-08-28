"""The evaluation protocol — the part that turns a pretty chart into a defensible one.

Two questions get answered here, and they are separate.

**Is the improvement real, or circular?** The red team optimises against our detector,
we retrain on those attacks, then we score — of course it improves. The answer is the
held-out split: attacks are split into `train` (the detector may learn them) and
`heldout_novel` (the detector NEVER trains on them). Everything reported is measured on
`heldout_novel`. For the stronger version of the same question — generalisation to a
whole attack FAMILY never seen in any form — see `scripts/generalisation_check.py`.

**Is the number quotable?** Not at a 0.5 threshold, and not as ROC AUC. No payments
system decides at `score >= 0.5`; they are tuned to a false-positive budget, because
flagging good customers is the expensive failure. And at 3.5% fraud prevalence ROC AUC
is flattered by an enormous true-negative pile — 0.999 there is unremarkable. So the
headline metrics are:

  * **recall at a fixed false-positive rate** (0.1% / 0.5% / 1% of legitimate traffic)
    — "inside the budget we can actually afford, how much fraud do we catch?"
  * **PR AUC** (average precision) — the summary that stays honest under imbalance.
  * **alert rate** — what share of ALL traffic the operating point flags, which is the
    number that decides whether the queue behind it is staffable.

The 0.5-threshold precision/recall/F1 are still computed, so the two can be compared
directly, but they are reported as the naive baseline they are.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .contract import (FEATURE_COLUMNS, LABEL_COLUMN, OPERATING_POINTS, PRIMARY_FPR,
                       AttackBatch, ScoreReport)
from .detector import Detector

DECISION_THRESHOLD = 0.5   # the naive cutoff, kept only as a comparison baseline


def campaign_ids(b: AttackBatch) -> "np.ndarray | None":
    """Which compromised account each of `b`'s attack rows belongs to, or None.

    `AttackBatch.validate` guarantees the attack rows of the timeline correspond
    one-to-one and in order with the feature rows, so the campaign label is just the
    timeline's `entity` column restricted to those rows. Hand-built batches in tests
    carry no timeline and get None.
    """
    if b.timeline is None or "entity" not in b.timeline.columns:
        return None
    ent = b.timeline.loc[b.timeline["is_attack"].astype(bool), "entity"].to_numpy()
    return ent if len(ent) == len(b.transactions) else None


def split_attacks(
    batches: List[AttackBatch], heldout_frac: float, rng: np.random.Generator
):
    """Split each vector's rows into (train, heldout_novel), preserving vector labels.

    The split is by CAMPAIGN, not by row. A campaign is one compromised host account
    and its 3-60 transactions, and every one of those rows carries that account's
    inherited age, merchant history and fourteen linkage counts. Splitting rows at
    random left 98.1% of the "never trained on" rows sharing a host with a row the
    detector had just learned, so `heldout_novel` was scoring memorised accounts at
    least as much as novel attacks. Whole campaigns go to one side or the other.

    Batches with no timeline (hand-built, in tests) fall back to the row split, and a
    vector that produced a single campaign cannot be split at all — it goes to train,
    which is the honest outcome rather than a fabricated holdout.
    """
    train_frames, heldout_frames, heldout_vectors = [], [], []
    for b in batches:
        n = len(b)
        ent = campaign_ids(b)
        if ent is None:
            idx = rng.permutation(n)
            k = int(n * (1 - heldout_frac))
            tr_idx, ho_idx = idx[:k], idx[k:]
        else:
            uniq = pd.unique(ent)
            order = rng.permutation(len(uniq))
            k = int(round(len(uniq) * (1 - heldout_frac)))
            if len(uniq) > 1:
                k = min(max(k, 1), len(uniq) - 1)
            held = set(uniq[order[k:]])
            is_held = np.fromiter((e in held for e in ent), dtype=bool, count=n)
            tr_idx, ho_idx = np.flatnonzero(~is_held), np.flatnonzero(is_held)
        tr, ho = b.transactions.iloc[tr_idx], b.transactions.iloc[ho_idx]
        train_frames.append(tr)
        heldout_frames.append(ho)
        heldout_vectors.extend([b.vector_id] * len(ho))
    train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    heldout = pd.concat(heldout_frames, ignore_index=True) if heldout_frames else pd.DataFrame()
    return train, heldout, np.array(heldout_vectors)


def threshold_for_fpr(legit_proba: np.ndarray, fpr: float) -> float:
    """The lowest score whose `>=` rule flags AT MOST `fpr` of legitimate traffic.

    A quantile on its own is not enough, and the difference is not academic. A gradient
    boosted ensemble emits large blocks of identical scores; the quantile routinely
    lands *inside* one, and `>=` then sweeps the whole block in. Measured on this
    detector, that reported recall "at a 0.1% budget" while the threshold was really
    flagging 43.9% of legitimate traffic — a budget quoted, not honoured.

    So walk the distinct scores and take the lowest one whose realised rate actually
    fits. If even the single highest block busts the budget, return a threshold above
    every score: recall 0 is the honest answer, not recall bought on credit.
    """
    if len(legit_proba) == 0:
        return float("inf")
    s = np.sort(legit_proba)
    uniq = np.unique(s)
    n_at_or_above = len(s) - np.searchsorted(s, uniq, side="left")
    fits = np.flatnonzero(n_at_or_above <= fpr * len(s))
    if len(fits) == 0:
        return float(uniq[-1]) + 1.0
    return float(uniq[fits[0]])


def recall_at_fpr(legit_proba: np.ndarray, attack_proba: np.ndarray,
                  fpr: float) -> Tuple[float, float, float]:
    """Recall and threshold at `fpr` — plus the false-positive rate actually realised.

    The third value exists so the budget can be checked rather than trusted. It is at
    most `fpr` by construction of `threshold_for_fpr`, and reporting it is what makes
    that claim auditable from the results files alone.
    """
    thr = threshold_for_fpr(legit_proba, fpr)
    realised = float((legit_proba >= thr).mean()) if len(legit_proba) else 0.0
    if len(attack_proba) == 0:
        return 0.0, thr, realised
    return float((attack_proba >= thr).mean()), thr, realised


def evaluate(
    detector: Detector,
    legit: pd.DataFrame,
    attacks: pd.DataFrame,
    attack_vectors: np.ndarray,
    iteration: int,
    split: str,
    real_traffic: "pd.DataFrame | None" = None,
) -> ScoreReport:
    """Score the detector on legit (label 0) + attack (label 1) rows.

    `real_traffic`, when given, is the untouched real test set — legitimate rows AND
    real fraud, no generated attacks. Two numbers come off it, and neither can be got
    from the scored mixture:

      * the alert rate an ops team would actually see, because the alert rate over the
        mixture is a function of how many attacks the run generated and moves with a
        config knob rather than with the detector;
      * recall on the REAL fraud in that set, at the same threshold. The curve tracked
        the generated attacks and nothing else, so a detector that learned our generator
        and learned nothing about fraud produced exactly the same picture as one that
        learned both. This separates them.
    """
    X = pd.concat([legit[FEATURE_COLUMNS], attacks[FEATURE_COLUMNS]], ignore_index=True)
    y = np.concatenate([np.zeros(len(legit)), np.ones(len(attacks))])
    proba = detector.score(X)
    legit_proba, attack_proba = proba[: len(legit)], proba[len(legit):]

    # --- the numbers worth quoting -------------------------------------------------
    rec_at, thr_at, realised_at = {}, {}, {}
    for fpr in OPERATING_POINTS:
        rec_at[fpr], thr_at[fpr], realised_at[fpr] = recall_at_fpr(
            legit_proba, attack_proba, fpr)
    primary_thr = thr_at.get(PRIMARY_FPR, threshold_for_fpr(legit_proba, PRIMARY_FPR))
    alert_rate = float((proba >= primary_thr).mean())
    alert_real, real_fraud_recall = 0.0, float("nan")
    if real_traffic is not None and len(real_traffic):
        real_proba = detector.score(real_traffic[FEATURE_COLUMNS])
        alert_real = float((real_proba >= primary_thr).mean())
        if LABEL_COLUMN in real_traffic.columns:
            real_y = real_traffic[LABEL_COLUMN].to_numpy()
            if real_y.sum():
                real_fraud_recall = float((real_proba[real_y == 1] >= primary_thr).mean())
    pr_auc = float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else 0.0

    per_vector_at_fpr: Dict[str, float] = {}
    for v in np.unique(attack_vectors):
        mask = attack_vectors == v
        per_vector_at_fpr[v] = float((attack_proba[mask] >= primary_thr).mean()) if mask.any() else 0.0

    # --- the naive 0.5-threshold baseline, for comparison only ----------------------
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    fp_rate = float(pred[: len(legit)].mean()) if len(legit) else 0.0
    attack_pred = pred[len(legit):]
    per_vector: Dict[str, float] = {}
    for v in np.unique(attack_vectors):
        mask = attack_vectors == v
        per_vector[v] = float(attack_pred[mask].mean()) if mask.any() else 0.0

    return ScoreReport(
        iteration=iteration,
        split=split,
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        auc=float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.5,
        fp_rate_on_legit=fp_rate,
        pr_auc=pr_auc,
        recall_at_fpr=rec_at,
        threshold_at_fpr=thr_at,
        realised_fpr_at_fpr=realised_at,
        alert_rate=alert_rate,
        alert_rate_on_real_traffic=alert_real,
        real_fraud_recall_at_fpr=real_fraud_recall,
        per_vector_recall=per_vector,
        per_vector_recall_at_fpr=per_vector_at_fpr,
        top_features=detector.top_gain_features(),
    )
