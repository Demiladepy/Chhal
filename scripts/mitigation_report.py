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
   The calibration pool is itself split in half: isotonic is FITTED on one half and its
   error is REPORTED on the other. An isotonic fit scored on its own rows returns ECE
   0.0000 by construction -- a tautology, not a calibration result.
4. Tune the amount-blind comparator on that held-back half. `block at score >= 0.5` is a
   straw man; the honest thing to beat is the best allow/step-up/block ladder that exists
   without our economics, tuned on the same cost model.
5. Score the frozen future: real test legit + real test fraud + adaptive attacks the
   detector has never seen in any form.
6. Price four policies against each other on that same population, and split the
   economics by segment -- a quarter of the cost denominator is fraud we generated
   ourselves, and the policy is much better at that than at the real thing.

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
                              fraud_loss_avoided, segment_costs,
                              threshold_baseline, tune_two_thresholds,
                              two_threshold_baseline)
from chhal.optimizer import EvasionOptimizer                              # noqa: E402
from chhal.redteam import ALL_VECTORS                                     # noqa: E402
from chhal.redteam.base import BaseProfile                                # noqa: E402
from chhal.redteam.hosts import HostPool                                  # noqa: E402

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
    # Campaigns for training and calibration compromise TRAIN accounts; the ones that get
    # scored compromise TEST accounts, so no evaluated attack carries context the detector
    # or the calibrator has already seen.
    train_hosts = HostPool(base.train)
    test_hosts = HostPool(base.test, exclude_accounts=base.train['_account'])
    print(f"[hosts] train {len(train_hosts):,} / test {len(test_hosts):,} eligible accounts")
    parts = {"train": [], "calib": [], "eval": []}
    for V in ALL_VECTORS:
        for name, pool, n in (("train", train_hosts, N_PER_VECTOR // 3),
                              ("calib", train_hosts, N_PER_VECTOR // 3),
                              ("eval", test_hosts, N_PER_VECTOR // 3)):
            v = V().calibrate(profile, pool)
            rows = opt.optimize(v.batch(n, 0, rng), seed_det, rng).transactions
            parts[name].append(rows.assign(**{LABEL_COLUMN: 1, "vector": v.vector_id}))
    atk = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    print(f"[red  ] {len(atk['train']):,} train / {len(atk['calib']):,} calibrate / "
          f"{len(atk['eval']):,} evaluate (evaluate slice never seen)")

    # 3. detector on fit + attack-train
    pool = pd.concat([fit_df, atk["train"].drop(columns=["vector"])], ignore_index=True)
    det = Detector(seed=SEED).fit(pool, LABEL_COLUMN)

    # 4. calibrate on the untouched calibration window + its attack slice.
    #    Split it in half: fit isotonic on one half, MEASURE the error on the other.
    #    Real rows split temporally, attack rows split at random, so each half looks like
    #    the whole pool rather than like one end of it.
    atk_cal = atk["calib"].drop(columns=["vector"])
    half_real = len(calib_df) // 2
    shuf = rng.permutation(len(atk_cal))
    half_atk = len(atk_cal) // 2
    fit_pool = pd.concat([calib_df.iloc[:half_real], atk_cal.iloc[shuf[:half_atk]]],
                         ignore_index=True)
    hold_pool = pd.concat([calib_df.iloc[half_real:], atk_cal.iloc[shuf[half_atk:]]],
                          ignore_index=True)

    raw_fit = det.score(fit_pool[FEATURE_COLUMNS])
    calib = Calibrator().fit(raw_fit, fit_pool[LABEL_COLUMN].to_numpy())

    raw_hold = det.score(hold_pool[FEATURE_COLUMNS])
    y_hold = hold_pool[LABEL_COLUMN].to_numpy()
    p_hold = calib(raw_hold)
    ece_raw = calibration_error(raw_hold, y_hold)
    ece_cal = calibration_error(p_hold, y_hold)
    ece_insample = calibration_error(calib(raw_fit), fit_pool[LABEL_COLUMN].to_numpy())
    print(f"[calib] fit={len(fit_pool):,} hold={len(hold_pool):,}  "
          f"ECE raw={ece_raw:.4f} -> calibrated={ece_cal:.4f} "
          f"(in-sample {ece_insample:.4f}, which is why we do not quote it)")

    # the comparator, tuned on the held-back half and never on what it is priced against
    t_stepup, t_block = tune_two_thresholds(
        CostModel(), p_hold, y_hold,
        hold_pool["amount"].to_numpy())
    print(f"[tuned] amount-blind ladder: step_up >= {t_stepup:.4f}, block >= {t_block:.4f}")

    # 5. the frozen future: real legit + real fraud + unseen adaptive attacks
    ev = pd.concat([base.test, atk["eval"].drop(columns=["vector"])], ignore_index=True)
    raw_ev = det.score(ev[FEATURE_COLUMNS])
    p = calib(raw_ev)
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
    # Threshold the RAW score, not the calibrated one. Calibration is a monotone map, so
    # it cannot change the ranking -- but isotonic collapses long runs of scores onto one
    # value, and at a 0.1% budget the threshold lands inside such a plateau, where a
    # tie-break decides whether hundreds of rows count as caught. The ranking is the thing
    # being measured; the calibrated probability is only needed for the economics below.
    legit_scores = raw_ev[y == 0]
    rows = []
    for fpr in (0.001, 0.005, 0.01):
        thr = np.quantile(legit_scores, 1 - fpr)
        row = {"fpr": fpr, "threshold": round(float(thr), 6)}
        for name, mask in seg.items():
            row[f"recall_{name}"] = round(float((raw_ev[mask] >= thr).mean()), 4)
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

    naive_actions = np.where(p >= 0.5, int(Action.BLOCK), int(Action.ALLOW))
    tuned_actions = np.where(p >= t_block, int(Action.BLOCK),
                             np.where(p >= t_stepup, int(Action.STEP_UP), int(Action.ALLOW)))

    do_nothing = allow_all_baseline(costs, y, amt)
    naive = threshold_baseline(costs, p, y, amt, threshold=0.5)
    tuned = two_threshold_baseline(costs, p, y, amt, t_stepup, t_block)
    smart = policy.report(actions, y, amt)

    def line(name, rep, acts=None):
        saved = do_nothing["total_cost"] - rep["total_cost"]
        pct = 100 * saved / do_nothing["total_cost"] if do_nothing["total_cost"] else 0.0
        # "loss avoided" was the wrong label: this is NET cost reduction, which nets the
        # friction we impose on legitimate customers against the fraud we stop. Fraud loss
        # avoided is a different, larger number, and it is reported beside it.
        row = {"policy": name, "cost_per_1k_txns": rep["cost_per_1k_txns"],
               "total_cost": rep["total_cost"], "vs_do_nothing": round(saved, 2),
               "net_cost_reduction_pct": round(pct, 2)}
        row["fraud_loss_avoided_pct"] = (
            0.0 if acts is None
            else round(100 * fraud_loss_avoided(costs, acts, y, amt), 2))
        return row

    comparison = pd.DataFrame([
        line("do nothing (allow all)", do_nothing),
        line("block at score >= 0.5 (untuned)", naive, naive_actions),
        line("tuned allow/step-up/block (amount-blind)", tuned, tuned_actions),
        line("expected-cost policy", smart, actions)])
    print("\n=== mitigation: cost of each policy on the same population ===")
    print(comparison.to_string(index=False))
    edge = (tuned["cost_per_1k_txns"] - smart["cost_per_1k_txns"]) / tuned["cost_per_1k_txns"]
    print("\nThe defensible claim is the LAST row against the THIRD, not against the second:")
    print(f"  amount-awareness + the capacity cap are worth {edge*100:.2f}% over the best "
          f"amount-blind ladder\n  ({comparison.iloc[3]['net_cost_reduction_pct']:.2f}% vs "
          f"{comparison.iloc[2]['net_cost_reduction_pct']:.2f}% net cost reduction).")

    # 7. the same economics, split by segment
    segs = segment_costs(costs, actions, y, amt, is_adaptive)
    print("\n=== whose fraud is being avoided ===")
    print(pd.DataFrame(segs).T.to_string())
    print("A quarter of the denominator is fraud we generated. The real-fraud-and-legit"
          "\nrow is the number that would survive contact with a production book.")

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
           "calibration": {
               "protocol": "isotonic fitted on one half of the calibration pool, "
                           "error measured on the other half",
               "ece_raw_holdout": round(ece_raw, 4),
               "ece_calibrated_holdout": round(ece_cal, 4),
               "ece_calibrated_in_sample": round(ece_insample, 4),
               "calibration_pool_fraud_rate": round(float(y_hold.mean()), 5),
               "deployment_fraud_rate": round(float(y.mean()), 5),
           },
           "detection_at_fixed_fpr": detection.to_dict("records"),
           "cost_model": costs.__dict__, "policy_config": policy.cfg.__dict__,
           "policies": comparison.to_dict("records"),
           "segment_economics": segs,
           "expected_cost_policy": smart,
           "tuned_amount_blind_policy": tuned,
           "naive_threshold_policy": naive}
    (RESULTS / "mitigation.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"p_fraud": p, "amount": amt, "is_fraud": y,
                  "action": [ACTION_NAMES[Action(a)] for a in actions]}
                 ).to_csv(RESULTS / "mitigation_actions.csv", index=False)
    print(f"\n-> {RESULTS/'mitigation.json'}")


if __name__ == "__main__":
    main()
