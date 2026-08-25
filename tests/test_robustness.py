"""Fast unit tests for the multi-seed robustness aggregation (no loop runs)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness import HEADLINE_KEYS, aggregate


def _headline(seed, recall, pr):
    return {"seed": seed, "final_recall_at_fpr": recall, "final_pr_auc": pr,
            "final_alert_rate": 0.014, "binding_rate": 0.30, "mimicry_ks": 0.27}


def test_aggregate_shapes_and_means():
    recall = [[0.0, 0.5, 0.9], [0.0, 0.6, 0.8]]        # 2 seeds x 3 iters
    pr = [[0.01, 0.7, 0.95], [0.01, 0.6, 0.90]]
    headline = [_headline(1, 0.90, 0.95), _headline(2, 0.80, 0.90)]
    agg = aggregate(np.array(recall, float), np.array(pr, float), headline, "synthetic")

    assert agg["n_seeds"] == 2
    assert agg["seeds"] == [1, 2]
    assert agg["data_source"] == "synthetic"
    assert len(agg["per_iteration"]) == 3
    # final-iteration recall mean = (0.9 + 0.8)/2 = 0.85
    assert abs(agg["per_iteration"][-1]["recall_at_fpr_mean"] - 0.85) < 1e-9
    hr = agg["headline"]["final_recall_at_fpr"]
    assert abs(hr["mean"] - 0.85) < 1e-9
    assert hr["std"] >= 0.0
    for key in HEADLINE_KEYS:
        assert set(agg["headline"][key]) == {"mean", "std"}


def test_zero_variance_gives_zero_std():
    recall = [[0.0, 0.9]] * 3                          # identical across seeds
    pr = [[0.01, 0.95]] * 3
    headline = [_headline(s, 0.9, 0.95) for s in (1, 2, 3)]
    agg = aggregate(np.array(recall, float), np.array(pr, float), headline, "ieee")
    assert agg["per_iteration"][-1]["recall_at_fpr_std"] == 0.0
    assert agg["headline"]["final_recall_at_fpr"]["std"] == 0.0
    assert agg["data_source"] == "ieee"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
