"""Mitigation: what the defence actually DOES, and what it saves.

Detection is scored in recall and AUC. Mitigation has to be scored in money, because
the decision is an economic one — see chhal/mitigation.py.

Protocol
--------
1. Split the real training window TEMPORALLY into fit (first 85%) and calibration
   (last 15%). The calibration slice is never trained on.
2. Fit the detector on the fit slice plus adaptive attacks (the closed loop's output).
3. Calibrate on the calibration slice plus attacks held out from step 2. Raw
   gradient-boosting scores are not probabilities, and this pool has attacks injected,
   so its implied base rate is not the deployment base rate either. Expected-cost
   decisions on uncalibrated scores are meaningless, so this step is mandatory.
4. Score the frozen future: real test legit + real test fraud + adaptive attacks the
   detector has never seen in any form.
5. Price three policies against each other on that same population.

Outputs results/mitigation.json and results/mitigation_actions.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN                  # noqa: E402
from chhal.data import load_base_data                                     # noqa: E402
from chhal.detector import Detector                                       # noqa: E402
from chhal.mitigation import (ACTION_NAMES, Action, ActionPolicy,         # noqa: E402
                              Calibrator, CostModel, PolicyConfig,
                              allow_all_baseline, calibration_error,
                              threshold_baseline)
from chhal.optimizer import EvasionOptimizer                              # noqa: E402
from chhal.redteam import ALL_VECTORS                                     # noqa: E402
from chhal.redteam.base import BaseProfile                                # noqa: E402

SEED = 7
CALIB_FRAC = 0.15          # last 15% of the training window, never fitted on
N_PER_VECTOR = 1200        # split three ways: train / calibrate / evaluate
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    rng = np.random.default_rng(SEED)
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")

    # 1. temporal fit/calibration split inside the training window
    cut = int(len(base.train) * (1 - CALIB_FRAC))
    fit_df, calib_df = base.train.iloc[:cut], base.train.iloc[cut:]
    print(f"[split] fit={len(fit_df):,}  calibration={len(calib_df):,} (never fitted on)")

    # 2. adaptive attacks, generated against a detector that has seen no attacks
    profile = BaseProfile(base.legit_quantiles, base.legit_categoricals)
    seed_det = Detector(seed=SEED).fit(fit_df, LABEL_COLUMN)
    opt = EvasionOptimizer(base.feature_stats)
    parts = {"train": [], "calib": [], "eval": []}
    for V in ALL_VECTORS:
        v = V().calibrate(profile)
        rows = opt.optimize(v.batch(N_PER_VECTOR, 0, rng), seed_det, rng).transactions
        a, b = N_PER_VECTOR // 3, 2 * N_PER_VECTOR // 3
        for name, sl in zip(parts, (rows.iloc[:a], rows.iloc[a:b], rows.iloc[b:])):
            parts[name].append(sl.assign(**{LABEL_COLUMN: 1, "vector": v.vector_id}))
    atk = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    print(f"[red  ] {len(atk['train']):,} train / {len(atk['calib']):,} calibrate / "
          f"{len(atk['eval']):,} evaluate (evaluate slice never seen)")

    # 3. detector on fit + attack-train
    pool = pd.concat([fit_df, atk["train"].drop(columns=["vector"])], ignore_index=True)
    det = Detector(seed=SEED).fit(pool, LABEL_COLUMN)

    # 4. calibrate on the untouched calibration window + its attack slice
    cal_pool = pd.concat([calib_df, atk["calib"].drop(columns=["vector"])], ignore_index=True)
    raw = det.score(cal_pool[FEATURE_COLUMNS])
    y_cal = cal_pool[LABEL_COLUMN].to_numpy()
    calib = Calibrator().fit(raw, y_cal)
    print(f"[calib] ECE raw={calibration_error(raw, y_cal):.4f} -> "
          f"calibrated={calibration_error(calib(raw), y_cal):.4f}")

    # 5. the frozen future: real legit + real fraud + unseen adaptive attacks
    ev = pd.concat([base.test, atk["eval"].drop(columns=["vector"])], ignore_index=True)
    p = calib(det.score(ev[FEATURE_COLUMNS]))
    y = ev[LABEL_COLUMN].to_numpy()
    amt = ev["amount"].to_numpy()
    print(f"[eval ] {len(ev):,} txns, {y.mean()*100:.3f}% fraud "
          f"({int(base.test[LABEL_COLUMN].sum()):,} real + {len(atk['eval']):,} adaptive)")

    # Detection quality at an operating point a payments team would actually run.
    # Split by segment, because the two are NOT the same problem and one number over
    # both hides that: `adaptive` is our red team's GenAI attacks, which the loop is
    # built to beat; `real_fraud` is IEEE-CIS's own labelled fraud, ordinary card fraud
    # seen through only twelve derived features. Reporting a single blended recall would
    # flatter us on one and slander us on the other.
    is_adaptive = np.zeros(len(ev), bool)
    is_adaptive[len(base.test):] = True
    seg = {"real_fraud": (y == 1) & ~is_adaptive,
           "adaptive_attacks": (y == 1) & is_adaptive,
           "all_fraud": y == 1}
    legit_scores = p[y == 0]
    rows = []
    for fpr in (0.001, 0.005, 0.01):
        thr = np.quantile(legit_scores, 1 - fpr)
        row = {"fpr": fpr, "threshold": round(float(thr), 6)}
        for name, mask in seg.items():
            row[f"recall_{name}"] = round(float((p[mask] >= thr).mean()), 4)
        rows.append(row)
    detection = pd.DataFrame(rows)
    print("\n=== detection at fixed false-positive rates on real legitimate traffic ===")
    print(f"    ({int(seg['real_fraud'].sum()):,} real IEEE-CIS frauds, "
          f"{int(seg['adaptive_attacks'].sum()):,} unseen adaptive attacks)")
    print(detection.to_string(index=False))

    # 6. price the policies
    costs = CostModel()
    policy = ActionPolicy(costs, PolicyConfig(max_review_rate=0.005))
    actions = policy.decide(p, amt)

    do_nothing = allow_all_baseline(costs, y, amt)
    naive = threshold_baseline(costs, p, y, amt, threshold=0.5)
    smart = policy.report(actions, y, amt)

    def line(name, rep):
        saved = do_nothing["total_cost"] - rep["total_cost"]
        pct = 100 * saved / do_nothing["total_cost"] if do_nothing["total_cost"] else 0.0
        return {"policy": name, "cost_per_1k_txns": rep["cost_per_1k_txns"],
                "total_cost": rep["total_cost"], "vs_do_nothing": round(saved, 2),
                "loss_avoided_pct": round(pct, 2)}

    comparison = pd.DataFrame([line("do nothing (allow all)", do_nothing),
                               line("block at score >= 0.5", naive),
                               line("expected-cost policy", smart)])
    print("\n=== mitigation: cost of each policy on the same population ===")
    print(comparison.to_string(index=False))

    print("\n=== what the expected-cost policy does ===")
    mix = pd.DataFrame({
        "share_of_all": pd.Series(smart["action_mix"]),
        "share_of_legit": pd.Series({ACTION_NAMES[a]: round(float((actions[y == 0] == a).mean()), 5)
                                     for a in Action}),
        "share_of_fraud": pd.Series({ACTION_NAMES[a]: round(float((actions[y == 1] == a).mean()), 5)
                                     for a in Action}),
    })
    print(mix.to_string())
    print(f"\nreview queue          : {smart['review_rate']*100:.3f}% of traffic "
          f"(cap {policy.cfg.max_review_rate*100:.1f}%)")
    print(f"outright declines on legit: {smart['block_rate_on_legit']*100:.3f}%")
    print(f"fraud stopped or challenged: {smart['fraud_touched_rate']*100:.2f}% "
          f"(real {(np.isin(actions[seg['real_fraud']], [Action.BLOCK, Action.REVIEW, Action.STEP_UP]).mean())*100:.2f}%, "
          f"adaptive {(np.isin(actions[seg['adaptive_attacks']], [Action.BLOCK, Action.REVIEW, Action.STEP_UP]).mean())*100:.2f}%)")
    print(f"naive 0.5 threshold declines {naive['block_rate_on_legit']*100:.3f}% of real customers")

    RESULTS.mkdir(exist_ok=True)
    out = {"data_source": base.source, "eval_rows": int(len(ev)),
           "eval_fraud_rate": round(float(y.mean()), 5),
           "calibration_ece_raw": round(calibration_error(raw, y_cal), 4),
           "calibration_ece_calibrated": round(calibration_error(calib(raw), y_cal), 4),
           "detection_at_fixed_fpr": detection.to_dict("records"),
           "cost_model": costs.__dict__, "policy_config": policy.cfg.__dict__,
           "policies": comparison.to_dict("records"), "expected_cost_policy": smart,
           "naive_threshold_policy": naive}
    (RESULTS / "mitigation.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"p_fraud": p, "amount": amt, "is_fraud": y,
                  "action": [ACTION_NAMES[Action(a)] for a in actions]}
                 ).to_csv(RESULTS / "mitigation_actions.csv", index=False)
    print(f"\n-> {RESULTS/'mitigation.json'}")


if __name__ == "__main__":
    main()
