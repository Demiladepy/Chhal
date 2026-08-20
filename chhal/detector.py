"""The blue-team detector: LightGBM over the frozen feature space, with SHAP.

LightGBM is pragmatic SOTA for tabular fraud — fast, strong, interpretable, and
deployable, which is exactly what "real-world feasibility" rewards. Swap for XGBoost
by changing this one class; nothing else depends on the model internals.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .contract import FEATURE_COLUMNS


class Detector:
    def __init__(self, seed: int = 7):
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame, label_col: str = "is_fraud") -> "Detector":
        X = df[FEATURE_COLUMNS].to_numpy()
        y = df[label_col].to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each row. Accepts a DataFrame or ndarray."""
        if isinstance(X, pd.DataFrame):
            X = X[FEATURE_COLUMNS].to_numpy()
        return self.model.predict_proba(X)[:, 1]

    def top_shap_features(self, df: pd.DataFrame, n: int = 5) -> List[str]:
        """Cheap global attribution via gain importance (SHAP-compatible ranking).

        We use LightGBM gain here to keep the loop fast; the dashboard computes true
        per-transaction SHAP on demand for the "why flagged" panel.
        """
        imp = self.model.booster_.feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1][:n]
        return [FEATURE_COLUMNS[i] for i in order]
