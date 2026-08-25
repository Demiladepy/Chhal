"""A second detector that does not need to have seen the attack.

The supervised arm has a structural blind spot that no amount of retraining removes:
it can only recognise what it has been shown. The brief asks for defence against
*emerging, novel* attacks — by definition the ones absent from training. So the loop
retraining on last iteration's attacks is necessary and not sufficient.

The anomaly arm is trained on LEGITIMATE traffic only and never sees a fraud label. It
answers a different question — "how unlike normal traffic is this?" — which stays
meaningful for an attack family invented after training. It is weaker than the
supervised arm wherever the supervised arm has coverage, and that is fine: it is not
there to win, it is there to be uncorrelated.

What the measurement said
-------------------------
Run `scripts/ensemble_check.py`. The honest answer is mixed and worth reading before
quoting any of this:

  * The anomaly arm **on its own is useless here** — 0.000 recall on an unseen attack
    family and 0.006 on real fraud at a 0.1% budget. Essentially random.
  * Fusing by `max` on a shared percentile axis is **worse than not doing it**, because
    it spends part of the false-positive budget on an arm that carries 0.0% of the
    catches.
  * Feeding the anomaly score to the supervised model **as an extra feature** does not
    rescue it either. It helped once, on the 12-feature space this module was first
    written for; the linkage block changed that and this docstring did not keep up, so
    for a while the file recommended a variant its own script measured as worse. Numbers
    are deliberately not quoted here any more — `scripts/ensemble_check.py` prints the
    current ones and derives its recommendation from them, so the two cannot drift apart
    again.

The reason the arm fails is our own doing, and it is the interesting part. Attack rows
are drawn through the inverse CDF of real legitimate traffic, and what the attacker sets
is held inside the plausibility manifold, so they sit on-manifold *by construction* —
that is exactly what the fidelity guardrail was built to guarantee. A detector whose entire question is "is
this off-manifold?" cannot see them. The better our fidelity claim gets, the less an
outlier detector can contribute.

Both `Ensemble` and `StackedDetector` are kept, and neither is presented as the thing to
ship: run `scripts/ensemble_check.py` and read the verdict it computes. The negative
result is worth being able to reproduce, and worth reporting — a measurement that changed
our own minds is better evidence than one that confirmed them.

Fusing them
-----------
Raw scores from a gradient-boosting model and an isolation forest are not comparable,
so both are mapped onto the same axis first: the percentile of REFERENCE LEGITIMATE
TRAFFIC they sit above. 0.999 from either arm means "more extreme than 99.9% of real
customers". On that shared axis the fusion is `max` — flag if EITHER arm finds the
transaction extreme, which is exactly the semantics wanted when one arm may be blind.

That also keeps the false-positive budget controllable: thresholding the fused score
against the same reference legit distribution gives a directly interpretable FPR, so
everything downstream (evaluation, mitigation) works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .contract import FEATURE_COLUMNS, LABEL_COLUMN
from .detector import Detector


@dataclass
class AnomalyConfig:
    n_estimators: int = 200
    max_samples: int = 50_000     # isolation forests saturate quickly; this is plenty
    fit_rows: int = 150_000       # cap on legit rows used, for runtime
    seed: int = 7


class AnomalyArm:
    """Isolation forest over legitimate traffic. Never sees a fraud label."""

    def __init__(self, cfg: AnomalyConfig | None = None):
        self.cfg = cfg or AnomalyConfig()
        self.model = IsolationForest(
            n_estimators=self.cfg.n_estimators,
            max_samples=min(self.cfg.max_samples, self.cfg.fit_rows),
            contamination="auto",
            random_state=self.cfg.seed,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, train: pd.DataFrame, label_col: str = LABEL_COLUMN) -> "AnomalyArm":
        legit = train[train[label_col] == 0] if label_col in train.columns else train
        if len(legit) > self.cfg.fit_rows:
            legit = legit.sample(self.cfg.fit_rows, random_state=self.cfg.seed)
        self.model.set_params(max_samples=min(self.model.max_samples, len(legit)))
        self.model.fit(legit[FEATURE_COLUMNS].to_numpy())
        self._fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous (sklearn's score_samples is higher = more normal)."""
        if not self._fitted:
            raise RuntimeError("AnomalyArm used before fit()")
        if isinstance(X, pd.DataFrame):
            X = X[FEATURE_COLUMNS].to_numpy()
        return -self.model.score_samples(X)


class Ensemble:
    """Supervised detector + anomaly arm, fused on a shared legit-percentile axis."""

    def __init__(self, detector: Detector, anomaly: AnomalyArm):
        self.detector = detector
        self.anomaly = anomaly
        self._ref_sup: np.ndarray | None = None
        self._ref_anom: np.ndarray | None = None

    def fit_reference(self, legit: pd.DataFrame) -> "Ensemble":
        """Record where each arm scores real legitimate traffic. This is the only thing
        that makes the two arms comparable, so it must be legit rows the arms did not
        train on."""
        self._ref_sup = np.sort(self.detector.score(legit[FEATURE_COLUMNS]))
        self._ref_anom = np.sort(self.anomaly.score(legit[FEATURE_COLUMNS]))
        return self

    @staticmethod
    def _pct(ref: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.searchsorted(ref, s, side="left") / max(len(ref), 1)

    def arm_percentiles(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._ref_sup is None:
            raise RuntimeError("Ensemble used before fit_reference(); the two arms' raw "
                               "scores are not comparable without it.")
        return (self._pct(self._ref_sup, self.detector.score(X[FEATURE_COLUMNS])),
                self._pct(self._ref_anom, self.anomaly.score(X[FEATURE_COLUMNS])))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        sup, anom = self.arm_percentiles(X)
        return np.maximum(sup, anom)

    def top_gain_features(self, n: int = 5):
        return self.detector.top_gain_features(n)


class StackedDetector:
    """Supervised detector whose feature space is FEATURE_COLUMNS **plus** the anomaly score.

    This is the fusion that actually helps. The anomaly score is an issuer-side signal
    computed at scoring time, not something an attacker sets, so attacks still live in
    FEATURE_COLUMNS exactly as before and the frozen contract is untouched — the model
    simply gets one more column the red team cannot reach.

    Exposes the same `score` / `top_gain_features` surface as `Detector`, so evaluation,
    the loop and the mitigation policy all take it unchanged.
    """

    ANOMALY_COLUMN = "anomaly_score"

    def __init__(self, anomaly: AnomalyArm | None = None, seed: int = 7):
        from lightgbm import LGBMClassifier
        self.anomaly = anomaly or AnomalyArm(AnomalyConfig(seed=seed))
        self._anomaly_fitted = anomaly is not None and anomaly._fitted
        self.columns = FEATURE_COLUMNS + [self.ANOMALY_COLUMN]
        self.model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=48, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
        )
        self._fitted = False

    def _augment(self, df: pd.DataFrame) -> np.ndarray:
        out = df[FEATURE_COLUMNS].copy()
        out[self.ANOMALY_COLUMN] = self.anomaly.score(df)
        return out[self.columns].to_numpy()

    def fit(self, df: pd.DataFrame, label_col: str = LABEL_COLUMN) -> "StackedDetector":
        if not self._anomaly_fitted:
            self.anomaly.fit(df, label_col)      # legit rows only, never a fraud label
            self._anomaly_fitted = True
        self.model.fit(self._augment(df), df[label_col].to_numpy())
        self._fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackedDetector used before fit()")
        return self.model.predict_proba(self._augment(X))[:, 1]

    def top_gain_features(self, n: int = 5):
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return [self.columns[i] for i in np.argsort(imp)[::-1][:n]]
