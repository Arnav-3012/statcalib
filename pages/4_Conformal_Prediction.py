import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Conformal Prediction",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Conformal Prediction")
st.subheader("Mathematical Guarantees Beyond Calibration")

st.info(
    "**What is Conformal Prediction?**\n\n"
    "Calibration fixes the numbers. "
    "Conformal Prediction provides **mathematical guarantees.**\n\n"
    "Instead of saying '41% probability of Effusion' — it says:\n\n"
    "*'The true answer is in {No Effusion} and I mathematically guarantee "
    "this system will be correct at least 90% of the time across all patients.'*"
)

st.markdown("---")

# ── The four outcomes ─────────────────────────────────────────────────────────
st.markdown("### The Four Possible Outputs")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.success("**✅ {No Disease}**\n\nAI is confident. No disease present.\n\n**Act on this safely.**")
with col2:
    st.error("**🚨 {Disease}**\n\nAI is confident. Disease present.\n\n**Act on this urgently.**")
with col3:
    st.warning("**⚠️ {Both}**\n\nAI is uncertain. Could be either.\n\n**Radiologist review.**")
with col4:
    st.info("**🚩 {Empty}**\n\nAI cannot decide. Maximum uncertainty.\n\n**Flag immediately.**")

st.markdown("---")

# ── Interactive conformal predictor ──────────────────────────────────────────
st.markdown("### 🎮 Interactive Conformal Predictor")
st.markdown("Adjust the calibrated score to see what prediction set you would receive:")

col1, col2 = st.columns([1, 2])
with col1:
    cal_score       = st.slider("Calibrated probability score", 0.0, 1.0, 0.35, 0.01)
    coverage_choice = st.radio("Coverage level:",
                               ["90% (standard)", "95% (conservative)"])
    q_hat = 0.385 if "90" in coverage_choice else 0.806

with col2:
    nc_pos      = 1 - cal_score
    nc_neg      = cal_score
    include_pos = nc_pos <= q_hat
    include_neg = nc_neg <= q_hat

    pred_set = []
    if include_neg: pred_set.append("No Effusion")
    if include_pos: pred_set.append("Effusion")

    if len(pred_set) == 0:
        st.info("**🚩 { } — Empty Set**\n\nModel cannot make any prediction with sufficient confidence.\n\n**Immediate radiologist review required.**")
    elif pred_set == ["No Effusion"]:
        st.success("**✅ { No Effusion } — Confident Negative**\n\nModel is confident there is no Effusion.\n\n**Safe to act on this result.**")
    elif pred_set == ["Effusion"]:
        st.error("**🚨 { Effusion } — Confident Positive**\n\nModel is confident Effusion is present.\n\n**Urgent clinical action recommended.**")
    else:
        st.warning("**⚠️ { No Effusion, Effusion } — Uncertain**\n\nBoth outcomes are plausible.\n\n**Radiologist review needed.**")

    st.markdown("<br>", unsafe_allow_html=True)
    computation_df = pd.DataFrame({
        "": ["q̂ threshold", "NC(Effusion)", "NC(No Effusion)"],
        "Value": [
            f"{q_hat:.3f}  ({coverage_choice.split()[0]} coverage)",
            f"1 − {cal_score:.3f} = {nc_pos:.3f}  →  {'Include ✓' if include_pos else 'Exclude ✗'}",
            f"{cal_score:.3f}  →  {'Include ✓' if include_neg else 'Exclude ✗'}",
        ],
    })
    st.caption("How this was computed:")
    st.table(computation_df.set_index(""))

st.markdown("---")

# ── Results across all diseases ───────────────────────────────────────────────
st.markdown("### 📊 Coverage Results — All 12 Diseases")


@st.cache_data
def load_cp():
    return pd.read_csv("data/conformal_results.csv")


try:
    cp_df = load_cp()

    n_pass  = int(cp_df["Guarantee_met"].sum())
    avg_c   = cp_df["Confident_pct"].mean()
    avg_u   = cp_df["Uncertain_pct"].mean()
    n_empty = int(cp_df["Empty_sets"].sum()) if "Empty_sets" in cp_df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Guarantee Met", f"{n_pass}/{len(cp_df)}")
    with col2: st.metric("Avg Confident Predictions", f"{avg_c:.1f}%")
    with col3: st.metric("Avg Uncertain", f"{avg_u:.1f}%")
    with col4: st.metric("Total Empty Sets", str(n_empty),
                         "0 = guarantee holds" if n_empty == 0 else "flag for review",
                         delta_color="normal" if n_empty == 0 else "inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    def color_cov(val):
        if not isinstance(val, float): return ""
        if val >= 0.90: return "background-color:#d4edda;color:#155724"
        if val >= 0.88: return "background-color:#fff3cd;color:#856404"
        return "background-color:#ffcccc;color:#DC3545"

    show_cols = [c for c in [
        "Disease", "N_test", "Coverage_actual", "Guarantee_met",
        "q_hat", "Confident_pct", "Uncertain_pct", "Empty_sets", "Avg_set_size"
    ] if c in cp_df.columns]

    st.dataframe(
        cp_df[show_cols].style.map(color_cov, subset=["Coverage_actual"]),
        use_container_width=True
    )

    st.markdown("---")

    # Stacked bar + coverage line — Plotly
    st.caption("Hover over bars for exact values. Click legend to toggle series.")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Confident single predictions",
        x=cp_df["Disease"], y=cp_df["Confident_pct"],
        marker_color="#28A745",
        hovertemplate="<b>%{x}</b><br>Confident: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Uncertain (radiologist needed)",
        x=cp_df["Disease"], y=cp_df["Uncertain_pct"],
        marker_color="#FD7E14",
        hovertemplate="<b>%{x}</b><br>Uncertain: %{y:.1f}%<extra></extra>",
    ))
    if "Empty_sets" in cp_df.columns and "N_test" in cp_df.columns:
        empty_pct = cp_df["Empty_sets"] / cp_df["N_test"] * 100
        fig.add_trace(go.Bar(
            name="Empty (flag immediately)",
            x=cp_df["Disease"], y=empty_pct,
            marker_color="#DC3545",
            hovertemplate="<b>%{x}</b><br>Empty: %{y:.2f}%<extra></extra>",
        ))

    # Coverage line on secondary axis
    fig.add_trace(go.Scatter(
        name="Actual coverage %",
        x=cp_df["Disease"],
        y=cp_df["Coverage_actual"] * 100,
        mode="lines+markers",
        marker=dict(symbol="diamond", size=9, color="navy"),
        line=dict(color="navy", width=2, dash="dash"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=90, line_dash="dot", line_color="navy",
                  annotation_text="90% target", yref="y2")

    fig.update_layout(
        barmode="stack",
        title="Prediction Set Composition and Coverage by Disease (90% target)",
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="% of patients"),
        yaxis2=dict(title="Coverage (%)", overlaying="y", side="right",
                    range=[80, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.success(
        f"**✅ Coverage Guarantee**\n\n"
        f"{n_pass}/{len(cp_df)} diseases meet the ≥90% coverage target. "
        "Conformal prediction provides a mathematically provable safety net — "
        "when the AI is uncertain it says so explicitly, rather than outputting "
        "a false confident number."
    )

except FileNotFoundError:
    st.error("Run notebook 10 first to generate data/conformal_results.csv.")
