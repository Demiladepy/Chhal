"""The blue-team detector: LightGBM over the frozen feature space.

LightGBM is pragmatic SOTA for tabular fraud, fast, strong, interpretable, and
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
            # Without a non-zero frequency LightGBM never bags at all, so `subsample=0.8`
            # sat here doing nothing: predictions were bit-identical at subsample=0.1.
            subsample_freq=1,
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

    def top_gain_features(self, n: int = 5) -> List[str]:
        """Global feature ranking by LightGBM gain importance.

        This is a WHOLE-MODEL ranking, not a per-transaction attribution. It is the
        same for every call until the model is next retrained. It does not vary by
        batch/row, so there is no `df` argument to pass in.
        """
        imp = self.model.booster_.feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1][:n]
        return [FEATURE_COLUMNS[i] for i in order]
