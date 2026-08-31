"""Run the full closed loop and persist results for the dashboard to replay.

    python scripts/run_loop.py            # default 8 iterations
    python scripts/run_loop.py --fast     # quick 4-iteration smoke run

Outputs to results/:
    curve.csv               arms-race curve (pre/post per iteration)
    per_vector_recall.csv   held-out recall per vector over iterations
    sample_attacks.csv      final adapted attacks for the live-stream panel
    summary.json            headline numbers for the deck
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.loop import LoopConfig, run_loop  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--fast", action="store_true", help="4 iters, fewer attacks")
    args = ap.parse_args()

    cfg = LoopConfig(iterations=4, attacks_per_vector=300) if args.fast \
        else LoopConfig(iterations=args.iterations)

    print(f"Running Chhal loop: {cfg}")
    t0 = time.time()
    result = run_loop(cfg)
    dt = time.time() - t0

    RESULTS.mkdir(exist_ok=True)
    result.curve.to_csv(RESULTS / "curve.csv", index=False)
    result.per_vector_recall.to_csv(RESULTS / "per_vector_recall.csv", index=False)
    result.sample_attacks.to_csv(RESULTS / "sample_attacks.csv", index=False)
    result.fidelity_per_vector.to_csv(RESULTS / "fidelity_per_vector.csv", index=False)
    result.fidelity_ks_table.to_csv(RESULTS / "fidelity_mimicry_ks.csv", index=False)

    # fidelity picture: the mimicry vector vs legitimate traffic (overlap == fidelity)
    from chhal.fidelity import plot_mimicry  # noqa: E402
    plot_mimicry(result.fidelity_legit, result.fidelity_mimic, str(RESULTS / "fidelity.png"))

    bench = result.curve[result.curve["phase"] == "benchmark"]
    op_cols = [c for c in bench.columns if c.startswith("recall_at_fpr_")]
    summary = {
        # Which population every number below was measured on. Quoted numbers are
        # meaningless without it: "synthetic" means we measured against a distribution
        # we invented, "ieee" means 590,540 real card transactions.
        "data_source": result.config["data_source"],
        "train_rows": result.config["train_rows"],
        "test_rows": result.config["test_rows"],
        # --- headline: measured at a false-positive budget a payments team can afford ---
        "operating_points": {
            c.replace("recall_at_fpr_", "recall_at_fpr="): {
                "baseline": round(float(bench[c].iloc[0]), 4),
                "final": round(float(bench[c].iloc[-1]), 4),
            } for c in op_cols
        },
        # The control, and the number to read the operating points against. If the
        # attack recall climbs while this stays flat, the loop taught the detector our
        # generator rather than fraud, which is a finding, not a bug, and it has to be
        # visible in the same file as the headline.
        "real_fraud_recall_at_fpr_0.001": {
            "baseline": round(float(bench["real_fraud_recall_at_fpr"].iloc[0]), 4),
            "final": round(float(bench["real_fraud_recall_at_fpr"].iloc[-1]), 4),
        },
        "baseline_pr_auc": round(float(bench["pr_auc"].iloc[0]), 4),
        "final_pr_auc": round(float(bench["pr_auc"].iloc[-1]), 4),
        "final_alert_rate": round(float(bench["alert_rate"].iloc[-1]), 5),
        # The rate above is over the scored mixture (real legit + however many attacks
        # this run generated), so it moves with a config knob. This one is measured on
        # the real test set alone and is the number an ops team would staff against.
        "final_alert_rate_on_real_traffic": round(
            float(bench["alert_rate_on_real_traffic"].iloc[-1]), 5),
        # Each threshold's REALISED false-positive rate. Must be at or under the budget
        # it is named for; printed so the budget can be checked rather than trusted.
        "realised_fpr": {
            c.replace("realised_fpr_", "budget="): round(float(bench[c].iloc[-1]), 6)
            for c in bench.columns if c.startswith("realised_fpr_")
        },
        # Every count here must be zero. See chhal/loop.py:_leakage_audit.
        "leakage_audit": result.leakage_audit,
        # --- the naive 0.5 cutoff, retained for comparison only, never as the headline ---
        "naive_threshold_0.5": {
            "baseline_f1": round(float(bench["f1"].iloc[0]), 4),
            "final_f1": round(float(bench["f1"].iloc[-1]), 4),
            "final_recall": round(float(bench["recall"].iloc[-1]), 4),
            "final_roc_auc": round(float(bench["auc"].iloc[-1]), 4),
            "final_fp_rate_on_legit": round(float(bench["fp_rate_on_legit"].iloc[-1]), 4),
        },
        "fidelity": result.fidelity,
        "iterations": cfg.iterations,
        "runtime_seconds": round(dt, 1),
        "config": result.config,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== BENCHMARK: blue generalisation on the fixed held-out adversarial set ===")
    print("    recall at a fixed share of REAL legitimate traffic flagged")
    print(bench[["iteration"] + op_cols + ["pr_auc", "alert_rate"]].to_string(index=False))
    print("\nblue generalisation (fixed held-out benchmark), at each operating point:")
    for name, v in summary["operating_points"].items():
        print(f"  {name:<22} {v['baseline']:.2%} -> {v['final']:.2%}")
    print(f"  {'PR AUC':<22} {summary['baseline_pr_auc']:.4f} -> {summary['final_pr_auc']:.4f}")
    print(f"  {'alert rate':<22} {summary['final_alert_rate']:.3%} of all traffic")
    nv = summary["naive_threshold_0.5"]
    print(f"\n  for comparison, the naive 0.5 cutoff: F1 {nv['baseline_f1']} -> {nv['final_f1']}, "
          f"ROC AUC {nv['final_roc_auc']} (flattered by 3.5% prevalence), "
          f"FP on legit {nv['final_fp_rate_on_legit']:.2%}")
    fid = result.fidelity
    print(f"\nfidelity: on-manifold rate={fid['on_manifold_rate']:.2%} (guardrail held); "
          f"mimicry vector KS vs legit={fid['mimicry_mean_ks_vs_legit']:.3f} (lower=stealthier)")
    print(result.fidelity_per_vector.to_string(index=False))
    print(f"\ndone in {dt:.1f}s -> results/ (curve, per_vector_recall, "
          f"fidelity_per_vector.csv, fidelity_mimicry_ks.csv, fidelity.png)")


if __name__ == "__main__":
    main()
