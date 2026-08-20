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
    summary = {
        "baseline_benchmark_f1": round(float(bench["f1"].iloc[0]), 4),
        "baseline_benchmark_recall": round(float(bench["recall"].iloc[0]), 4),
        "final_benchmark_f1": round(float(bench["f1"].iloc[-1]), 4),
        "final_benchmark_recall": round(float(bench["recall"].iloc[-1]), 4),
        "final_benchmark_auc": round(float(bench["auc"].iloc[-1]), 4),
        "final_fp_rate_on_legit": round(float(bench["fp_rate_on_legit"].iloc[-1]), 4),
        "fidelity": result.fidelity,
        "iterations": cfg.iterations,
        "runtime_seconds": round(dt, 1),
        "config": result.config,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== BENCHMARK: blue generalisation on the fixed held-out adversarial set ===")
    print(bench[["iteration", "f1", "auc", "recall", "fp_rate_on_legit"]].to_string(index=False))
    print(f"\nblue generalisation (fixed held-out benchmark): "
          f"recall {summary['baseline_benchmark_recall']:.2%} -> "
          f"{summary['final_benchmark_recall']:.2%}, "
          f"F1 {summary['baseline_benchmark_f1']} -> {summary['final_benchmark_f1']}  "
          f"(FP on legit={summary['final_fp_rate_on_legit']:.2%})")
    fid = result.fidelity
    print(f"\nfidelity: on-manifold rate={fid['on_manifold_rate']:.2%} (guardrail held); "
          f"mimicry vector KS vs legit={fid['mimicry_mean_ks_vs_legit']:.3f} (lower=stealthier)")
    print(result.fidelity_per_vector.to_string(index=False))
    print(f"\ndone in {dt:.1f}s -> results/ (curve, per_vector_recall, "
          f"fidelity_per_vector.csv, fidelity_mimicry_ks.csv, fidelity.png)")


if __name__ == "__main__":
    main()
