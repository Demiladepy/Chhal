"""Chhal — the 3-panel live demo (Streamlit).

Red Team | Live Stream | Blue Team. The dashboard REPLAYS precomputed loop results
(results/*.csv) — it never trains live, so it cannot stall on stage. Run the loop
first:  python scripts/run_loop.py

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

st.set_page_config(page_title="Chhal — adversarial fraud loop", layout="wide")


@st.cache_data
def load():
    curve = pd.read_csv(RESULTS / "curve.csv")
    pv = pd.read_csv(RESULTS / "per_vector_recall.csv")
    sample = pd.read_csv(RESULTS / "sample_attacks.csv")
    summary = json.loads((RESULTS / "summary.json").read_text())
    fid_path = RESULTS / "fidelity_per_vector.csv"
    fidelity = pd.read_csv(fid_path) if fid_path.exists() else pd.DataFrame()
    return curve, pv, sample, summary, fidelity


if not (RESULTS / "curve.csv").exists():
    st.error("No results yet. Run:  python scripts/run_loop.py")
    st.stop()

curve, per_vector, sample, summary, fidelity = load()

st.title("Chhal — a closed-loop adversarial engine for GenAI payment fraud")
st.caption(
    "Every attack the red team invents becomes training ground for a stronger defence. "
    "The chart tracks detection on **held-out novel attacks the detector never trained on** "
    "— generalisation, not memorisation."
)

# headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Static detector vs adaptive attacks", f"{summary['baseline_benchmark_recall']:.1%}",
          help="Baseline recall on the fixed adversarial benchmark — attacks optimised to evade it.")
c2.metric("After the loop (benchmark recall)", f"{summary['final_benchmark_recall']:.1%}",
          delta=f"{summary['final_benchmark_recall'] - summary['baseline_benchmark_recall']:+.1%}")
c3.metric("Final benchmark F1", f"{summary['final_benchmark_f1']:.3f}")
c4.metric("FP rate on legit", f"{summary['final_fp_rate_on_legit']:.2%}")

left, center, right = st.columns([1, 1.1, 1.3])

# ---- LEFT: Red Team --------------------------------------------------------
with left:
    st.subheader("🔴 Red Team")
    st.caption("Adaptive attack vectors, tuned to evade the current detector.")
    for vid in sorted(sample["vector"].unique()):
        rows = sample[sample["vector"] == vid]
        st.markdown(f"**{vid}** · {len(rows)} adapted txns")
    st.dataframe(
        sample[["vector", "amount", "velocity_24h", "amount_to_avg_ratio",
                "is_new_beneficiary", "channel_code"]].round(2),
        height=280, use_container_width=True,
    )

# ---- CENTER: Live Stream ---------------------------------------------------
with center:
    st.subheader("🟡 Live Stream")
    st.caption("Transactions flowing; the retrained detector flags adaptive fraud.")
    final_iter = int(per_vector["iteration"].max())
    latest = per_vector[per_vector["iteration"] == final_iter]
    fig = go.Figure(go.Bar(
        x=latest["recall"], y=latest["vector"], orientation="h",
        marker_color="#c0392b",
        text=[f"{r:.0%}" for r in latest["recall"]], textposition="auto",
    ))
    fig.update_layout(
        title=f"Recall by vector — iteration {final_iter}",
        xaxis_title="caught", xaxis_range=[0, 1], height=340, margin=dict(l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Threshold-hugging (the hero vector) stays the hardest to catch — it "
               "mimics normal behaviour, so it sits lowest even after the loop learns.")

# ---- RIGHT: Blue Team — the money chart ------------------------------------
with right:
    st.subheader("🔵 Blue Team — the arms race")
    bench = curve[curve["phase"] == "benchmark"]
    pressure = curve[curve["phase"] == "pressure"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bench["iteration"], y=bench["f1"], name="blue generalisation (fixed benchmark)",
        mode="lines+markers", line=dict(color="#2980b9", width=3)))
    fig.add_trace(go.Scatter(
        x=pressure["iteration"], y=pressure["f1"], name="red pressure (newest evasion)",
        mode="lines+markers", line=dict(color="#e67e22", dash="dot")))
    fig.update_layout(
        title="Held-out F1 over loop iterations",
        xaxis_title="iteration", yaxis_title="F1 (novel held-out attacks)",
        yaxis_range=[0, 1.02], height=340, legend=dict(y=-0.35), margin=dict(l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A static detector catches almost none of the adaptive attacks (iteration 0 "
               "— they're tuned to look normal). One loop pass and blue holds the fixed "
               "benchmark near the top. The dotted line is red's ongoing pressure: each "
               "iteration it finds a fresh evasion the just-retrained model partly misses. "
               "That gap is the honest, unfinished arms race.")

with st.expander("Why this curve is defensible (the judge's question)"):
    st.markdown(
        "- **The blue line is a FIXED benchmark the detector never trains on.** Hard "
        "adaptive attacks, built once against the baseline. Improving on them proves "
        "**generalisation to unseen adaptive fraud**, not memorisation of specific attacks.\n"
        "- **Attacks are split, not just rows.** Each iteration's fresh attacks are split "
        "into `train` (detector may learn them) and held-out (the dotted pressure line).\n"
        "- **No leakage:** the base train/test split is frozen before any attack is "
        "injected; the benchmark and pressure attacks touch neither.\n"
        "- **The gap between the two lines is the point** — blue holds the known shape, "
        "red keeps probing new ones. An arms race, not a solved problem."
    )

# ---- Fidelity: a metric, not a claim (judged criterion) --------------------
st.markdown("---")
st.subheader("🎯 Fidelity of simulation — measured, not claimed")
fid = summary.get("fidelity", {})
if fid:
    f1, f2, f3 = st.columns(3)
    f1.metric("On-manifold rate", f"{fid['on_manifold_rate']:.1%}",
              help="Share of fully-optimised attack feature values still inside the realistic "
                   "manifold bounds. Every candidate is hard-clipped to these exact bounds, so "
                   "this is ~100% by construction — it proves the clip is wired correctly, not "
                   "that the guardrail did meaningful work. See 'guardrail binding rate' for that.")
    if "frac_off_manifold_pre_clip" in fid:
        f2.metric("Guardrail binding rate", f"{fid['frac_off_manifold_pre_clip']:.1%}",
                  help="Share of proposed perturbations that landed outside the manifold "
                       "BEFORE clipping and had to be pulled back. This is the non-tautological "
                       "evidence the guardrail actually does work.")
    f3.metric(f"Mimicry KS vs legit ({fid.get('mimicry_vector','')})",
              f"{fid['mimicry_mean_ks_vs_legit']:.3f}",
              help="Distribution distance of the hero vector from legitimate traffic. Low = it "
                   "genuinely mimics normal behaviour, which is why it's the hardest to catch.")

fcol1, fcol2 = st.columns([1.3, 1])
with fcol1:
    img = RESULTS / "fidelity.png"
    if img.exists():
        st.image(str(img), caption="Legit traffic (blue) vs the mimicry vector (red) — "
                                    "overlapping mass = the attack hides inside normal behaviour.")
with fcol2:
    if not fidelity.empty:
        st.caption("KS distance from legit, per vector (lower = more legit-like). "
                   "`threshold_hugging` sits closest to normal — the stealth that makes it the "
                   "hero vector; overt vectors sit further out **by design** (that's the fraud "
                   "signal, not a defect).")
        st.dataframe(fidelity.round(3), height=240, use_container_width=True)

st.caption("Base data is generated (see `chhal/data.py`); point it at real PaySim / "
           "IEEE-CIS features and every fidelity number becomes a real-data report.")
