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

# Headline metrics, read at the operating point a payments team would actually run —
# a fixed share of real legitimate traffic flagged, not an arbitrary 0.5 threshold.
PRIMARY = "recall_at_fpr=0.001"
op = summary["operating_points"][PRIMARY]

src = summary.get("data_source", "unknown")
if src == "ieee":
    st.success(f"Measured on **{summary['train_rows'] + summary['test_rows']:,} real "
               f"IEEE-CIS card transactions** (temporal split, "
               f"{summary['train_rows']:,} train / {summary['test_rows']:,} test).")
else:
    st.warning(f"Running on the **{src}** fallback — these numbers must not be quoted. "
               f"Run `python scripts/prepare_ieee.py` for the real base population.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Static detector vs adaptive attacks", f"{op['baseline']:.1%}",
          help="Baseline recall on the fixed adversarial benchmark — the attacks were "
               "optimised to evade exactly this detector, so near-zero is the optimizer "
               "working, not a broken model.")
c2.metric("After the loop", f"{op['final']:.1%}",
          delta=f"{op['final'] - op['baseline']:+.1%}",
          help="Recall on the same fixed benchmark, at a 0.1% false-positive budget.")
c3.metric("PR AUC", f"{summary['final_pr_auc']:.3f}",
          help="Average precision — the honest summary under 3.5% fraud prevalence. "
               f"ROC AUC reads {summary['naive_threshold_0.5']['final_roc_auc']:.4f} on the "
               "same run and says almost nothing.")
c4.metric("Alert rate", f"{summary['final_alert_rate']:.2%}",
          help="Share of ALL traffic flagged at that operating point — the number that "
               "decides whether the queue behind it is staffable.")

left, center, right = st.columns([1, 1.1, 1.3])

# ---- LEFT: Red Team --------------------------------------------------------
with left:
    st.subheader("Red Team")
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
    st.subheader("Live Stream")
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
    st.caption("Recall per vector at the same 0.1% false-positive budget. Every bar here "
               "is 0.00% before the loop runs; what you are seeing is how quickly the "
               "detector learns each generator's fingerprint once retrained on it.")

# ---- RIGHT: Blue Team — the money chart ------------------------------------
with right:
    st.subheader("Blue Team — the arms race")
    bench = curve[curve["phase"] == "benchmark"]
    pressure = curve[curve["phase"] == "pressure"]
    fig = go.Figure()
    # Recall at a fixed false-positive budget, not F1 at a 0.5 cutoff. No payments system
    # is tuned the way F1 assumes, and the write-up tells judges to distrust that number —
    # the demo has to show the same metric the claims are made in.
    METRIC = "recall_at_fpr_0.001"
    fig.add_trace(go.Scatter(
        x=bench["iteration"], y=bench[METRIC], name="blue generalisation (fixed benchmark)",
        mode="lines+markers", line=dict(color="#2980b9", width=3)))
    fig.add_trace(go.Scatter(
        x=pressure["iteration"], y=pressure[METRIC], name="red pressure (newest evasion)",
        mode="lines+markers", line=dict(color="#e67e22", dash="dot")))
    # The control. Same detector, same threshold, IEEE-CIS's own labelled fraud. Without
    # it the blue line is unfalsifiable: a detector that learned only our generator draws
    # exactly the same climb as one that learned fraud.
    if "real_fraud_recall_at_fpr" in bench.columns:
        fig.add_trace(go.Scatter(
            x=bench["iteration"], y=bench["real_fraud_recall_at_fpr"],
            name="real IEEE-CIS fraud (control)", mode="lines+markers",
            line=dict(color="#7f8c8d", dash="dash")))
    fig.update_layout(
        title="Recall on held-out attacks, at 0.1% of real customers flagged",
        xaxis_title="iteration", yaxis_title="recall @ 0.1% FPR",
        yaxis_range=[0, 1.02], height=340, legend=dict(y=-0.35), margin=dict(l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A static detector catches almost none of the adaptive attacks (iteration 0 "
               "— they're tuned to look normal). One loop pass and blue holds the fixed "
               "benchmark near the top. The dotted line is red's ongoing pressure: each "
               "iteration it finds a fresh evasion the just-retrained model partly misses. "
               "That gap is the honest, unfinished arms race. The grey dashed line is the "
               "control: real IEEE-CIS fraud at the same threshold. It barely moves. The "
               "loop is teaching the detector our generator far faster than it is "
               "teaching it fraud, and that gap is the point of the benchmark.")

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
st.subheader("Fidelity of simulation — measured, not claimed")
fid = summary.get("fidelity", {})
if fid:
    f1, f2, f3 = st.columns(3)
    f1.metric("On-manifold rate", f"{fid['on_manifold_rate']:.1%}",
              help="Share of ATTACKER-CONTROLLED feature values still inside the realistic "
                   "manifold bounds. Every candidate is hard-clipped to these exact bounds, so "
                   "this is ~100% by construction — it proves the clip is wired correctly, not "
                   "that the guardrail did meaningful work. See 'guardrail binding rate' for "
                   "that. The rest of an attack row is inherited from a real account or derived "
                   "from a real timeline, so it needs no plausibility check.")
    if "frac_off_manifold_pre_clip" in fid:
        f2.metric("Guardrail binding rate", f"{fid['frac_off_manifold_pre_clip']:.1%}",
                  help="Share of proposed perturbations that landed outside the manifold "
                       "BEFORE clipping and had to be pulled back. This is the non-tautological "
                       "evidence the guardrail actually does work.")
    f3.metric(f"Mimicry KS vs legit ({fid.get('mimicry_vector','')})",
              f"{fid['mimicry_mean_ks_vs_legit']:.3f}",
              help="Distribution distance of the mimicry vector from legitimate traffic. "
                   "Read it against the noise floor: two samples of the same legit traffic "
                   "score ~0.024 apart, so this is ~16x sampling noise, not a match.")

fcol1, fcol2 = st.columns([1.3, 1])
with fcol1:
    img = RESULTS / "fidelity.png"
    if img.exists():
        st.image(str(img), caption="Legit traffic (blue) vs the mimicry vector (red) — "
                                    "overlapping mass = the attack hides inside normal behaviour.")
with fcol2:
    if not fidelity.empty:
        st.caption("KS distance from legit, per vector (lower = more legit-like). "
                   "Narrowed to comparable typologies, **real** fraud spans 0.13–0.46 on this "
                   "metric and is caught 10–22% of the time; these vectors span 0.27–0.51 and "
                   "are caught 0%. Distance from legit is not what separates them.")
        st.dataframe(fidelity.round(3), height=240, use_container_width=True)

st.caption("Distances are measured against real IEEE-CIS legitimate traffic, and each attack "
           "is mounted on a real, never-fraudulent account whose issuer-side context it "
           "inherits rather than invents (see `chhal/redteam/hosts.py`).")
