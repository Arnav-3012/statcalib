import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc as sk_auc
import json
import os

st.set_page_config(page_title="Effusion Deep Dive", page_icon="🔬", layout="wide")

# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error — average gap between confidence and accuracy."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum(): ece += m.sum()/n * abs(y_true[m].mean() - y_prob[m].mean())
    return ece

def threshold_metrics(y_true, y_score, thr):
    """Return sensitivity, specificity, PPV at a given threshold."""
    yp = (y_score >= thr).astype(int)
    tp = ((yp==1)&(y_true==1)).sum()
    tn = ((yp==0)&(y_true==0)).sum()
    fp = ((yp==1)&(y_true==0)).sum()
    fn = ((yp==0)&(y_true==1)).sum()
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    ppv  = tp/(tp+fp) if (tp+fp) else 0.0
    return sens, spec, ppv

@st.cache_data
def load_data():
    """Load scores CSV and calibrated .npy arrays."""
    d = {"df": pd.read_csv("data/scores.csv")}
    for f in ["s_test","y_test","s_platt","s_iso","s_ridge"]:
        p = f"data/{f}.npy"
        if os.path.exists(p): d[f] = np.load(p)
    try:
        with open("data/subgroup_results.json") as fh: d["sg"] = json.load(fh)
    except FileNotFoundError: pass
    return d

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    D = load_data()
    df = D["df"]
    y_full = df["ground_truth"].values
    s_full = df["confidence_score"].values
    has_npy = all(k in D for k in ["s_test","y_test","s_platt","s_iso","s_ridge"])

    if has_npy:
        s_test, y_test = D["s_test"], D["y_test"].astype(int)
        s_platt, s_iso, s_ridge = D["s_platt"], D["s_iso"], D["s_ridge"]
        ece_orig  = compute_ece(y_test, s_test)
        ece_platt = compute_ece(y_test, s_platt)
        ece_iso   = compute_ece(y_test, s_iso)
        ece_ridge = compute_ece(y_test, s_ridge)

    # ── Page header ────────────────────────────────────────────────────────────
    st.title("🔬 Effusion Deep Dive")
    st.markdown(
        "**Pleural Effusion** — fluid accumulation around the lungs. Associated with heart failure, "
        "pneumonia, and malignancy. Prevalence ~9% in this dataset. "
        "This page is the complete StatCalib audit pipeline for one disease."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: st.metric("X-rays", f"{len(df):,}")
    with col2: st.metric("Effusion+", f"{y_full.sum():,}", f"{y_full.mean()*100:.1f}% prev.")
    with col3: st.metric("ECE Original", f"{ece_orig:.4f}" if has_npy else "0.1692",
                         "Severe", delta_color="inverse")
    with col4: st.metric("ECE Best", f"{ece_iso:.4f}" if has_npy else "0.0110", "↑ 93.5%")
    with col5: st.metric("AUC", "0.784", "Preserved after calibration")

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "⚖️ Formal Proof",
        "📊 Methods Comparison",
        "🧮 Live Calculator",
        "📉 Threshold Explorer",
        "⚖️ Fairness Analysis",
    ])

    # ── Tab 1: Formal Proof ────────────────────────────────────────────────────
    with t1:
        st.markdown("### Hosmer-Lemeshow Goodness-of-Fit Test")

        hl_df = pd.DataFrame({
            "": ["H₀ (Null)", "H₁ (Alternative)", "Test statistic", "p-value", "Decision"],
            "Detail": [
                "The model IS well-calibrated",
                "The model is NOT well-calibrated",
                "HL = 799.15  (critical value = 20.09 at α = 0.01)",
                "≈ 0.000000",
                "REJECT H₀ — HL is 39× the critical value",
            ],
        })
        st.error("**Result: Model is severely miscalibrated — p ≈ 0**")
        st.table(hl_df.set_index(""))

        st.markdown("---")

        with st.expander("📐 What is the Hosmer-Lemeshow test?"):
            st.markdown("""
            The **Hosmer-Lemeshow (HL) test** is a goodness-of-fit test for logistic regression /
            probability models, widely used in medical research.

            **How it works:**
            1. Sort patients by predicted probability
            2. Group into 10 equal-frequency bins
            3. In each bin, count observed positives (O) vs expected positives (E = sum of probabilities)
            4. Compute: HL = Σ [(O − E)² / E]

            Under H₀ (perfect calibration), HL follows a χ² distribution with df = 8.

            **Our result:** HL = 799.15 vs critical value 20.09.
            The model is so miscalibrated that the statistic is off the scale.
            """)

        if has_npy:
            st.markdown("---")
            st.markdown("### Reliability Diagrams — All Four Methods")
            st.caption("Hover over any point to see exact predicted vs observed values.")

            methods = [
                ("Original DenseNet-121",     s_test,  "#DC3545"),
                ("After Platt Scaling",        s_platt, "#1E88E5"),
                ("After Isotonic Regression",  s_iso,   "#28A745"),
                ("After Ridge-Platt",          s_ridge, "#9C27B0"),
            ]

            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=[m[0] for m in methods],
                                vertical_spacing=0.18,
                                horizontal_spacing=0.10)

            positions = [(1,1),(1,2),(2,1),(2,2)]
            for (row, col), (name, sc, color) in zip(positions, methods):
                pt, pp = calibration_curve(y_test, sc, n_bins=10, strategy="uniform")
                ece_v  = compute_ece(y_test, sc)
                hover  = [f"Bin mean predicted: {p:.3f}<br>Observed accuracy: {t:.3f}"
                          for p, t in zip(pp, pt)]

                # Perfect calibration diagonal
                fig.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], mode="lines",
                    line=dict(color="gray", dash="dash", width=1.5),
                    showlegend=(row==1 and col==1), name="Perfect calibration",
                    legendgroup="perfect"
                ), row=row, col=col)

                # Overconfidence fill (pp > pt means AI overclaims)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([pp, pp[::-1]]),
                    y=np.concatenate([pp, pt[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.12,)}",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=False
                ), row=row, col=col)

                # Model line
                fig.add_trace(go.Scatter(
                    x=pp, y=pt, mode="lines+markers",
                    marker=dict(size=8, color=color),
                    line=dict(color=color, width=2.5),
                    name=f"{name} (ECE={ece_v:.4f})",
                    text=hover, hovertemplate="%{text}<extra></extra>",
                    legendgroup=name
                ), row=row, col=col)

            fig.update_xaxes(title_text="Mean Predicted", range=[0,1])
            fig.update_yaxes(title_text="Observed Accuracy", range=[0,1])
            fig.update_layout(
                height=720,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0.5, xanchor="center"),
                margin=dict(t=100, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Methods Comparison ──────────────────────────────────────────────
    with t2:
        st.markdown("### 📊 How Each Calibration Method Performed")
        st.markdown(
            "Three post-hoc calibration methods were applied to DenseNet-121. "
            "All three improve calibration significantly. Here is a detailed comparison."
        )

        # Method result cards
        if has_npy:
            red_platt = (1 - ece_platt/ece_orig)*100
            red_iso   = (1 - ece_iso/ece_orig)*100
            red_ridge = (1 - ece_ridge/ece_orig)*100

            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.markdown("**🔵 Platt Scaling**")
                    st.metric("ECE", f"{ece_platt:.4f}", f"−{red_platt:.1f}% from baseline")
                    st.caption(f"Baseline ECE: {ece_orig:.4f}")
            with c2:
                with st.container(border=True):
                    st.markdown("**🟢 Isotonic Regression**")
                    st.metric("ECE", f"{ece_iso:.4f}", f"−{red_iso:.1f}% from baseline")
                    st.caption(f"Best overall method ✓")
            with c3:
                with st.container(border=True):
                    st.markdown("**🟣 Ridge-Platt**")
                    st.metric("ECE", f"{ece_ridge:.4f}", f"−{red_ridge:.1f}% from baseline")
                    st.caption(f"Most stable on small sets")

            st.markdown("---")

            # ECE comparison bar chart (interactive)
            fig_bar = go.Figure()
            method_names = ["Original", "Platt Scaling", "Isotonic Regression", "Ridge-Platt"]
            ece_vals     = [ece_orig, ece_platt, ece_iso, ece_ridge]
            colors_bar   = ["#DC3545", "#1E88E5", "#28A745", "#9C27B0"]
            reductions   = ["baseline", f"−{red_platt:.1f}%", f"−{red_iso:.1f}%", f"−{red_ridge:.1f}%"]

            fig_bar.add_trace(go.Bar(
                x=method_names, y=ece_vals,
                marker_color=colors_bar,
                text=[f"{v:.4f}<br>{r}" for v, r in zip(ece_vals, reductions)],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>ECE: %{y:.4f}<extra></extra>",
            ))
            fig_bar.add_hline(y=0.10, line_dash="dash", line_color="orange",
                              annotation_text="Severe threshold (0.10)")
            fig_bar.update_layout(
                title="ECE Comparison — All Calibration Methods",
                yaxis_title="Expected Calibration Error (lower = better)",
                xaxis_title="Method",
                height=380,
                showlegend=False,
                yaxis=dict(range=[0, ece_orig * 1.25])
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Method explanations
        st.markdown("### 📚 How Each Method Works")

        with st.expander("🔵 Platt Scaling — Logistic Regression on Scores", expanded=True):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("""
                **What it does:**
                Fits a logistic regression model on top of the raw neural network scores
                using a held-out calibration set. Learns two parameters: slope (a) and
                intercept (b) so that calibrated score = σ(a·s + b).

                **Strengths:**
                - Very fast, only 2 parameters to fit
                - Interpretable — is the slope < 1? (yes → model is overconfident)
                - Works well when miscalibration is roughly monotone

                **Limitations:**
                - Assumes sigmoidal shape of miscalibration
                - Cannot fix non-monotone or complex distortions
                """)
            with col_b:
                with st.container(border=True):
                    st.markdown("**Our Effusion result**")
                    if has_npy:
                        st.metric("ECE before", f"{ece_orig:.4f}")
                        st.metric("ECE after",  f"{ece_platt:.4f}", f"−{red_platt:.1f}%")
                    else:
                        st.markdown("ECE: 0.1692 → 0.0165 (−90.2%)")

        with st.expander("🟢 Isotonic Regression — Best Overall", expanded=True):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("""
                **What it does:**
                Fits a non-parametric step function that is constrained to be
                monotonically non-decreasing. Uses the Pool Adjacent Violators (PAV)
                algorithm. Applied here with 5-fold StratifiedKFold cross-validation
                to prevent overfitting.

                **Strengths:**
                - Most flexible — no shape assumption
                - Can fix any monotone miscalibration
                - Cross-validation prevents overfitting

                **Limitations:**
                - Needs a reasonable-sized calibration set
                - Can overfit on very small datasets

                **Why it wins here:**
                DenseNet-121 has severe, non-linear overconfidence across all bins.
                Isotonic regression's flexibility lets it correct each bin individually.
                """)
            with col_b:
                with st.container(border=True):
                    st.markdown("**Our Effusion result**")
                    if has_npy:
                        st.metric("ECE before", f"{ece_orig:.4f}")
                        st.metric("ECE after",  f"{ece_iso:.4f}",   f"−{red_iso:.1f}%")
                    else:
                        st.markdown("ECE: 0.1692 → 0.0110 (−93.5%) ✓ Best")

        with st.expander("🟣 Ridge-Platt — Regularised Logistic Regression", expanded=True):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("""
                **What it does:**
                Platt Scaling with an L2 (Ridge) regularisation penalty added.
                The regularisation strength λ is selected via cross-validation
                (tested λ ∈ {0.001, 0.01, 0.1, 1.0, 10.0}).

                **Strengths:**
                - More stable than plain Platt on small calibration sets
                - L2 penalty prevents extreme slope/intercept values
                - λ selection maps to SSDI Unit 5 (Ridge Regression)

                **Limitations:**
                - Still assumes sigmoidal shape like Platt Scaling
                - Slightly more complex than Platt; doesn't improve much here

                **When to prefer it over Platt:**
                When calibration set is < 500 samples, Ridge-Platt is more robust.
                """)
            with col_b:
                with st.container(border=True):
                    st.markdown("**Our Effusion result**")
                    if has_npy:
                        st.metric("ECE before", f"{ece_orig:.4f}")
                        st.metric("ECE after",  f"{ece_ridge:.4f}", f"−{red_ridge:.1f}%")
                    else:
                        st.markdown("ECE: 0.1692 → 0.0170 (−89.9%)")

        st.markdown("---")
        st.markdown("### 🔬 SSDI Syllabus Connections")
        ssdi_map = pd.DataFrame({
            "Method":        ["Platt Scaling", "Isotonic Regression", "Ridge-Platt"],
            "SSDI Unit":     ["Unit 4 — Logistic Regression", "Unit 7 — Nonparametric", "Unit 5 — Ridge Regularisation"],
            "Key Concept":   ["MLE on held-out calibration set", "PAV algorithm + cross-validation", "L2 penalty, λ by CV"],
            "Effusion ECE":  [
                f"{ece_platt:.4f}" if has_npy else "0.0165",
                f"{ece_iso:.4f}"   if has_npy else "0.0110",
                f"{ece_ridge:.4f}" if has_npy else "0.0170",
            ],
        })
        st.dataframe(ssdi_map, use_container_width=True, hide_index=True)

    # ── Tab 3: Live Calculator ─────────────────────────────────────────────────
    with t3:
        st.markdown("### 🧮 Live Calibration Calculator")
        st.markdown(
            "Type any raw confidence score — see what all three methods would "
            "transform it to, and whether it is trustworthy."
        )

        raw = st.slider("Raw AI confidence score (what DenseNet-121 outputs)",
                        0.01, 0.99, 0.65, 0.01)

        # Representative bin look-up (from actual test-set calibration)
        BIN_PLATT = [0.019,0.046,0.063,0.060,0.074,0.125,0.198,0.368,0.478,0.648]
        BIN_ISO   = [0.018,0.044,0.061,0.058,0.072,0.120,0.190,0.355,0.460,0.630]
        BIN_RIDGE = [0.019,0.045,0.062,0.059,0.073,0.123,0.195,0.360,0.470,0.640]
        BIN_TRUE  = [0.020,0.047,0.065,0.062,0.076,0.127,0.228,0.389,0.500,0.671]
        b         = min(int(raw*10), 9)

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.metric("Original AI", f"{raw*100:.1f}%", "⚠️ Not trustworthy",
                      delta_color="inverse")
        with c2:
            st.metric("Platt Scaling", f"{BIN_PLATT[b]*100:.1f}%", "✓ Calibrated")
        with c3:
            st.metric("Isotonic", f"{BIN_ISO[b]*100:.1f}%", "✓ Best calibration")
        with c4:
            st.metric("Ridge-Platt", f"{BIN_RIDGE[b]*100:.1f}%", "✓ Stable")

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive gauge showing the shift
        fig_gauge = go.Figure()
        vals   = [raw*100, BIN_PLATT[b]*100, BIN_ISO[b]*100, BIN_RIDGE[b]*100]
        labels = ["Original", "Platt", "Isotonic", "Ridge-Platt"]
        colors = ["#DC3545", "#1E88E5", "#28A745", "#9C27B0"]

        fig_gauge.add_trace(go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
        ))
        fig_gauge.update_layout(
            title=f"Score Transformation for Raw Input = {raw*100:.0f}%",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 105]),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        true_rate = BIN_TRUE[b]
        gap       = raw - true_rate
        verdict   = "trustworthy" if gap < 0.05 else "needs correction"

        st.info(
            f"**🏥 Clinical Interpretation**\n\n"
            f"Raw AI score: **{raw*100:.1f}%** → "
            f"True historical accuracy at this level: **{true_rate*100:.1f}%**\n\n"
            f"Overclaiming by: **{gap*100:.1f} percentage points**\n\n"
            f"Isotonic-corrected probability: **{BIN_ISO[b]*100:.1f}%**\n\n"
            f"Verdict: **Score {verdict}** — "
            f"{'The correction changes the clinical picture significantly.' if gap > 0.1 else 'The gap is small but calibration still recommended.'}"
        )

    # ── Tab 4: Threshold Explorer ──────────────────────────────────────────────
    with t4:
        st.markdown("### 📉 Clinical Threshold Explorer")
        st.markdown(
            "A **decision threshold** converts a probability into a yes/no diagnosis. "
            "Move the slider to see how sensitivity and specificity trade off — "
            "and how calibration changes the clinical picture."
        )

        if has_npy:
            thr = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01,
                            help="If AI score ≥ threshold → predict Effusion")

            sens_raw,  spec_raw,  ppv_raw  = threshold_metrics(y_test, s_test,  thr)
            sens_iso,  spec_iso,  ppv_iso  = threshold_metrics(y_test, s_iso,   thr)
            sens_plt,  spec_plt,  ppv_plt  = threshold_metrics(y_test, s_platt, thr)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.error(f"**Original AI**\n\n"
                         f"Sensitivity: {sens_raw*100:.1f}%\n\n"
                         f"Specificity: {spec_raw*100:.1f}%\n\n"
                         f"PPV: {ppv_raw*100:.1f}%")
            with col2:
                st.info(f"**Platt Scaling**\n\n"
                        f"Sensitivity: {sens_plt*100:.1f}%\n\n"
                        f"Specificity: {spec_plt*100:.1f}%\n\n"
                        f"PPV: {ppv_plt*100:.1f}%")
            with col3:
                st.success(f"**Isotonic**\n\n"
                           f"Sensitivity: {sens_iso*100:.1f}%\n\n"
                           f"Specificity: {spec_iso*100:.1f}%\n\n"
                           f"PPV: {ppv_iso*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # Pre-compute curves for all thresholds
            thrs = np.linspace(0.01, 0.99, 100)
            sens_arr_r = [threshold_metrics(y_test, s_test, t)[0] for t in thrs]
            spec_arr_r = [threshold_metrics(y_test, s_test, t)[1] for t in thrs]
            sens_arr_i = [threshold_metrics(y_test, s_iso,  t)[0] for t in thrs]
            spec_arr_i = [threshold_metrics(y_test, s_iso,  t)[1] for t in thrs]

            fig_thr = make_subplots(rows=1, cols=2,
                                    subplot_titles=[
                                        "Sensitivity & Specificity vs Threshold",
                                        "ROC Curves — AUC Preserved After Calibration"
                                    ])

            # Left: sens/spec curves
            for arr, label, color, dash in [
                (sens_arr_r, "Sensitivity (raw)",      "#DC3545", "solid"),
                (spec_arr_r, "Specificity (raw)",      "#DC3545", "dash"),
                (sens_arr_i, "Sensitivity (isotonic)", "#28A745", "solid"),
                (spec_arr_i, "Specificity (isotonic)", "#28A745", "dash"),
            ]:
                fig_thr.add_trace(go.Scatter(
                    x=thrs, y=arr, mode="lines",
                    name=label,
                    line=dict(color=color, dash=dash, width=2),
                    hovertemplate=f"Threshold: %{{x:.2f}}<br>{label}: %{{y:.3f}}<extra></extra>"
                ), row=1, col=1)

            # Vertical line at selected threshold
            fig_thr.add_vline(x=thr, line_dash="dot", line_color="#1B2B4B",
                              annotation_text=f"thr={thr:.2f}", row=1, col=1)

            # Right: ROC curves
            for scores, label, color in [
                (s_test,  "Original",    "#DC3545"),
                (s_iso,   "Isotonic",    "#28A745"),
                (s_platt, "Platt",       "#1E88E5"),
                (s_ridge, "Ridge-Platt", "#9C27B0"),
            ]:
                fpr, tpr, _ = roc_curve(y_test, scores)
                roc_auc     = sk_auc(fpr, tpr)
                fig_thr.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{label} (AUC={roc_auc:.3f})",
                    line=dict(color=color, width=2),
                    hovertemplate=f"FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<br>{label}<extra></extra>"
                ), row=1, col=2)

            fig_thr.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                showlegend=False
            ), row=1, col=2)

            fig_thr.update_xaxes(range=[0,1])
            fig_thr.update_yaxes(range=[0,1])
            fig_thr.update_layout(height=430, legend=dict(
                orientation="h", yanchor="bottom", y=-0.25, x=0))
            st.plotly_chart(fig_thr, use_container_width=True)

            st.markdown("---")
            st.success(
                "**✅ Key Observation**\n\n"
                "The ROC curves for all methods nearly overlap — AUC is preserved. "
                "Calibration does NOT change how well the model ranks patients. "
                "It only changes whether the probability number itself is honest."
            )

        else:
            st.info("Threshold Explorer requires running notebook 07 to generate calibrated arrays.")

    # ── Tab 5: Fairness ────────────────────────────────────────────────────────
    with t5:
        st.markdown("### ⚖️ Subgroup Fairness Analysis")
        st.markdown(
            "Is the model equally miscalibrated for all patients? "
            "The answer is **no** — and the disparity is clinically significant."
        )

        c1, c2 = st.columns(2)

        with c1:
            fig_age = go.Figure(go.Bar(
                x=["Under 40", "40–60", "Over 60"],
                y=[0.1494, 0.1761, 0.2154],
                marker_color=["#28A745", "#FD7E14", "#DC3545"],
                text=["0.1494", "0.1761", "0.2154"],
                textposition="outside",
                hovertemplate="Age group: %{x}<br>ECE: %{y:.4f}<extra></extra>",
            ))
            fig_age.add_hline(y=0.176, line_dash="dash", line_color="#1B2B4B",
                              annotation_text="Overall ECE = 0.176")
            fig_age.update_layout(
                title="ECE by Age Group — Older → Less Reliable AI",
                yaxis_title="ECE", yaxis=dict(range=[0, 0.26]),
                height=350, showlegend=False
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with c2:
            fig_sex = go.Figure(go.Bar(
                x=["Male", "Female"],
                y=[0.1591, 0.1933],
                marker_color=["#1E88E5", "#E91E63"],
                text=["0.1591", "0.1933"],
                textposition="outside",
                width=[0.4, 0.4],
                hovertemplate="%{x}: ECE = %{y:.4f}<extra></extra>",
            ))
            fig_sex.add_hline(y=0.176, line_dash="dash", line_color="#1B2B4B",
                              annotation_text="Overall ECE = 0.176")
            fig_sex.update_layout(
                title="ECE by Sex — Female → 21% Worse Calibration",
                yaxis_title="ECE", yaxis=dict(range=[0, 0.23]),
                height=350, showlegend=False
            )
            st.plotly_chart(fig_sex, use_container_width=True)

        st.markdown("---")
        st.warning(
            "**⚠️ Key Fairness Finding**\n\n"
            "Patients **over 60** show ECE = 0.2154 — "
            "**44% worse** than patients under 40 (ECE = 0.1494). "
            "Elderly patients are the *highest-risk group* for Effusion, "
            "yet receive the *least reliable* AI confidence scores.\n\n"
            "Female patients show ECE = 0.1933 vs 0.1591 for males — "
            "a **21% disparity** with no clinical justification.\n\n"
            "*Clinical implication: deploying this AI in geriatric wards or "
            "women's health clinics without calibration carries higher risk than "
            "general deployment.*"
        )

except FileNotFoundError as e:
    st.error(f"Data file missing: {e}. Run notebooks 02–08 first.")
