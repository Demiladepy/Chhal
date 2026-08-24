"""Latency and throughput — the "real-world feasibility" number, measured.

Everything else here is measured in recall and money. Neither matters if the decision
cannot be made inside an authorization. A card authorization is a synchronous
round trip with a budget of roughly 100-300ms end to end, most of which belongs to the
network and the issuer's own systems; the risk decision gets a slice of it, tens of
milliseconds at best. So the question is not "is the model fast" but "does the whole
path — anomaly score, detector, calibration, action decision — fit in that slice at
n=1, one transaction at a time, which is how authorizations actually arrive."

Batch numbers are reported too, because they are what a nightly rescoring or a queue
drain would see, and they are much better per transaction. They are NOT the number to
quote for live auth.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN        # noqa: E402
from chhal.data import load_base_data                           # noqa: E402
from chhal.detector import Detector                             # noqa: E402
from chhal.ensemble import AnomalyArm, StackedDetector          # noqa: E402
from chhal.mitigation import ActionPolicy, Calibrator           # noqa: E402

SEED = 7
BATCH_SIZES = (1, 10, 100, 1_000, 10_000)
AUTH_BUDGET_MS = 50.0     # the slice of an authorization a risk decision can reasonably take
RESULTS = Path(__file__).resolve().parents[1] / "results"


def bench(fn, data, repeats: int) -> dict:
    fn(data)                                            # warm up JIT/threads/caches
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(data)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99)), "mean_ms": float(t.mean())}


def main() -> None:
    base = load_base_data(source="ieee")
    print(f"[data] {base.describe()}")
    train, test = base.train, base.test

    det = Detector(seed=SEED).fit(train, LABEL_COLUMN)
    anom = AnomalyArm().fit(train, LABEL_COLUMN)
    stacked = StackedDetector(anomaly=anom, seed=SEED).fit(train, LABEL_COLUMN)
    calib = Calibrator().fit(det.score(train[FEATURE_COLUMNS]), train[LABEL_COLUMN].to_numpy())
    policy = ActionPolicy()

    stages = {
        "detector only": lambda d: det.score(d),
        "+ anomaly arm (stacked)": lambda d: stacked.score(d),
        "+ calibration": lambda d: calib(det.score(d)),
        "FULL PATH (score -> action)": lambda d: policy.decide(
            calib(stacked.score(d)), d["amount"].to_numpy()),
    }

    rows = []
    for n in BATCH_SIZES:
        chunk = test[FEATURE_COLUMNS].head(n).copy()
        repeats = 200 if n <= 100 else (50 if n <= 1_000 else 10)
        for name, fn in stages.items():
            r = bench(fn, chunk, repeats)
            rows.append({"stage": name, "batch": n, **{k: round(v, 4) for k, v in r.items()},
                         "us_per_txn": round(r["p50_ms"] * 1000 / n, 2),
                         "txns_per_sec": int(n / (r["p50_ms"] / 1000)) if r["p50_ms"] else 0})

    df = pd.DataFrame(rows)
    print("\n=== per-call latency by batch size ===")
    print(df.pivot(index="stage", columns="batch", values="p50_ms").round(3).to_string())
    print("\n=== microseconds per transaction (p50) ===")
    print(df.pivot(index="stage", columns="batch", values="us_per_txn").round(2).to_string())

    live = df[(df.batch == 1) & (df.stage == "FULL PATH (score -> action)")].iloc[0]
    bulk = df[(df.batch == 10_000) & (df.stage == "FULL PATH (score -> action)")].iloc[0]
    print("\n=== live authorization, one transaction at a time ===")
    print(f"  full path p50 {live.p50_ms:.3f} ms | p95 {live.p95_ms:.3f} ms | p99 {live.p99_ms:.3f} ms")
    headroom = AUTH_BUDGET_MS / live.p99_ms
    print(f"  against a {AUTH_BUDGET_MS:.0f}ms risk-decision budget: {headroom:.0f}x headroom at p99")
    print(f"  batch throughput at 10k: {bulk.txns_per_sec:,} txns/sec "
          f"({bulk.us_per_txn:.1f} us/txn)")

    footprint = {
        "detector_kb": round(len(pickle.dumps(det.model)) / 1024, 1),
        "anomaly_kb": round(len(pickle.dumps(anom.model)) / 1024, 1),
        "calibrator_kb": round(len(pickle.dumps(calib.iso)) / 1024, 1),
    }
    footprint["total_kb"] = round(sum(footprint.values()), 1)
    print(f"\n=== model footprint === {footprint}")
    print("  no external lookups, no feature store, no network calls on the scoring path")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latency.json").write_text(json.dumps(
        {"auth_budget_ms": AUTH_BUDGET_MS, "measurements": rows, "footprint_kb": footprint,
         "live_single_txn": {"p50_ms": round(float(live.p50_ms), 4),
                             "p95_ms": round(float(live.p95_ms), 4),
                             "p99_ms": round(float(live.p99_ms), 4),
                             "headroom_at_p99": round(float(headroom), 1)},
         "batch_throughput_txns_per_sec": int(bulk.txns_per_sec)}, indent=2))
    df.to_csv(RESULTS / "latency.csv", index=False)
    print(f"-> {RESULTS/'latency.json'}")


if __name__ == "__main__":
    main()
