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

from chakravyuh.loop import LoopConfig, run_loop  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--fast", action="store_true", help="4 iters, fewer attacks")
    args = ap.parse_args()

    cfg = LoopConfig(iterations=4, attacks_per_vector=300) if args.fast \
        else LoopConfig(iterations=args.iterations)

    print(f"Running Chakravyuh loop: {cfg}")
    t0 = time.time()
    result = run_loop(cfg)
    dt = time.time() - t0

    RESULTS.mkdir(exist_ok=True)
    result.curve.to_csv(RESULTS / "curve.csv", index=False)
    result.per_vector_recall.to_csv(RESULTS / "per_vector_recall.csv", index=False)
    result.sample_attacks.to_csv(RESULTS / "sample_attacks.csv", index=False)

    bench = result.curve[result.curve["phase"] == "benchmark"]
    summary = {
        "baseline_benchmark_f1": round(float(bench["f1"].iloc[0]), 4),
        "baseline_benchmark_recall": round(float(bench["recall"].iloc[0]), 4),
        "final_benchmark_f1": round(float(bench["f1"].iloc[-1]), 4),
        "final_benchmark_recall": round(float(bench["recall"].iloc[-1]), 4),
        "final_benchmark_auc": round(float(bench["auc"].iloc[-1]), 4),
        "final_fp_rate_on_legit": round(float(bench["fp_rate_on_legit"].iloc[-1]), 4),
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
    print(f"done in {dt:.1f}s -> results/")


if __name__ == "__main__":
    main()
