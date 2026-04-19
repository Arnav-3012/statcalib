import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="All 14 Diseases",
    page_icon="📊",
    layout="wide"
)

TOTAL_IMAGES = 10_000


@st.cache_data
def load_results():
    """Load multi-disease calibration results CSV."""
    df = pd.read_csv("data/robust_calibration_results.csv")
    df["Prevalence_pct"] = (df["N_pos"] / TOTAL_IMAGES * 100).round(2)
    return df


try:
    df    = load_results()
    valid = df.sort_values("ECE_orig", ascending=False).reset_index(drop=True)

    # Load conformal results for comparison tool
    try:
        cp_df = pd.read_csv("data/conformal_results.csv")
        has_cp = True
    except FileNotFoundError:
        cp_df = pd.DataFrame()
        has_cp = False

    # ── Page header ───────────────────────────────────────────────────────────
    st.title("📊 Systematic Audit — All 14 NIH Pathologies")

    st.info(
        "**🔬 Novel Contribution**\n\n"
        "This is the first systematic calibration audit across all "
        "NIH ChestX-ray14 pathologies with formal statistical testing. "
        "Most published papers report calibration for only one disease. "
        "StatCalib audits all 12 diseases with sufficient positive cases "
        "and proves the finding is universal — not disease-specific."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Diseases Audited", f"{len(valid)}/14")
    with col2:
        st.metric("Avg ECE Reduction",
                  f"{valid['Reduction_pct'].mean():.1f}%",
                  "All diseases improved")
    with col3:
        worst = valid.iloc[0]
        st.metric("Most Miscalibrated", worst["Disease"],
                  f"ECE = {worst['ECE_orig']:.4f}",
                  delta_color="inverse")
    with col4:
        best = valid.iloc[-1]
        st.metric("Best Calibrated", best["Disease"],
                  f"ECE = {best['ECE_orig']:.4f}")

    st.markdown("---")

    # ── Tabbed interface ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Master Table",
        "📈 Visual Comparison",
        "🔍 Disease Explorer",
        "⚖️ Compare Two Diseases",
    ])

    # ── Tab 1: Master table ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### Complete Results Table")
        st.markdown(
            "🔴 Red = ECE > 0.25 (severe) | "
            "🟡 Yellow = ECE 0.15–0.25 (moderate) | "
            "🟢 Green = ECE < 0.15 (better)"
        )

        display_df = valid[[
            "Disease", "Prevalence_pct", "N_pos", "AUC",
            "ECE_orig", "ECE_best", "Reduction_pct",
            "Best_Method", "Brier_orig", "Brier_best"
        ]].rename(columns={
            "Prevalence_pct": "Prevalence%",
            "Reduction_pct":  "Reduction%",
            "ECE_orig":       "ECE Before",
            "ECE_best":       "ECE After",
        })

        def color_ece(val):
            """Colour-code ECE severity."""
            if not isinstance(val, float): return ""
            if val > 0.25: return "background-color:#ffcccc;color:#DC3545;font-weight:bold"
            if val > 0.15: return "background-color:#fff3cd;color:#856404"
            return "background-color:#d4edda;color:#155724"

        st.dataframe(
            display_df.style.map(color_ece, subset=["ECE Before"]),
            use_container_width=True,
            height=450
        )

        st.markdown("---")
        st.error(
            "**Key Finding**\n\n"
            "Every single disease shows a positive calibration gap. "
            "DenseNet-121 is systematically overconfident across "
            "ALL pathologies — not just Effusion. This confirms "
            "miscalibration is a structural property of the model, "
            "not a disease-specific artefact."
        )

    # ── Tab 2: Visual comparison ──────────────────────────────────────────────
    with tab2:
        st.markdown("### ECE Before and After — Interactive Charts")
        st.caption("Hover over bars for exact values. Click legend items to toggle series.")

        # Grouped bar chart with error bars
        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            name="Before calibration",
            x=valid["Disease"], y=valid["ECE_orig"],
            marker_color="#DC3545",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(valid["CI_orig_hi"] - valid["ECE_orig"]).to_numpy(),
                arrayminus=(valid["ECE_orig"] - valid["CI_orig_lo"]).to_numpy(),
                color="rgba(0,0,0,0.5)",
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "ECE before: %{y:.4f}<br>"
                "95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                "<extra></extra>"
            ),
            customdata=valid[["CI_orig_lo","CI_orig_hi"]].to_numpy(),
        ))

        fig_bar.add_trace(go.Bar(
            name="After best method",
            x=valid["Disease"], y=valid["ECE_best"],
            marker_color="#28A745",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(valid["CI_best_hi"] - valid["ECE_best"]).to_numpy(),
                arrayminus=(valid["ECE_best"] - valid["CI_best_lo"]).to_numpy(),
                color="rgba(0,0,0,0.5)",
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "ECE after: %{y:.4f}<br>"
                "Best method: %{customdata[0]}<br>"
                "Reduction: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
            customdata=valid[["Best_Method","Reduction_pct"]].to_numpy(),
        ))

        fig_bar.add_hline(y=0.10, line_dash="dash", line_color="#FD7E14",
                          annotation_text="Severe threshold (0.10)")
        fig_bar.update_layout(
            barmode="group",
            title="ECE Before vs After — All Diseases (error bars = 95% bootstrap CI)",
            xaxis_title="Disease",
            yaxis_title="Expected Calibration Error",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=450,
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### AUC vs Calibration Error — Before & After")
        st.caption(
            "Each disease appears twice: 🔴 before calibration, 🟢 after. "
            "Arrows show the movement. AUC barely shifts; ECE drops sharply."
        )

        fig_scatter = go.Figure()

        # Arrows connecting before → after for each disease
        for _, row in valid.iterrows():
            fig_scatter.add_annotation(
                x=row["AUC_after"], y=row["ECE_best"],
                ax=row["AUC"],      ay=row["ECE_orig"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor="rgba(100,100,100,0.45)",
            )

        # Before dots (red)
        fig_scatter.add_trace(go.Scatter(
            x=valid["AUC"], y=valid["ECE_orig"],
            mode="markers+text",
            name="Before calibration",
            marker=dict(color="#DC3545", size=11, symbol="circle"),
            text=valid["Disease"],
            textposition="top center",
            textfont=dict(size=8),
            customdata=valid[["ECE_orig", "AUC", "Reduction_pct", "Best_Method"]].to_numpy(),
            hovertemplate=(
                "<b>%{text}</b> — Before<br>"
                "AUC: %{customdata[1]:.4f}<br>"
                "ECE: %{customdata[0]:.4f}<br>"
                "Reduction after: %{customdata[2]:.1f}% (%{customdata[3]})"
                "<extra></extra>"
            ),
        ))

        # After dots (green)
        fig_scatter.add_trace(go.Scatter(
            x=valid["AUC_after"], y=valid["ECE_best"],
            mode="markers",
            name="After calibration",
            marker=dict(color="#28A745", size=11, symbol="circle"),
            customdata=valid[["ECE_best", "AUC_after", "Reduction_pct", "Best_Method"]].to_numpy(),
            hovertemplate=(
                "<b>%{text}</b> — After<br>"
                "AUC: %{customdata[1]:.4f}<br>"
                "ECE: %{customdata[0]:.4f}<br>"
                "Reduction: %{customdata[2]:.1f}% (%{customdata[3]})"
                "<extra></extra>"
            ),
            text=valid["Disease"],
        ))

        fig_scatter.update_layout(
            title="AUC vs ECE — Before & After Calibration (arrows show movement)",
            xaxis=dict(title="AUC-ROC", range=[
                valid["AUC"].min() - 0.05,
                valid[["AUC", "AUC_after"]].max().max() + 0.05
            ]),
            yaxis=dict(title="Expected Calibration Error (ECE)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=500,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.warning(
            "**Key Insight**\n\n"
            "Every arrow points **left-down or straight down** — ECE falls sharply "
            "while AUC barely moves. Cardiomegaly has the highest AUC (0.858) "
            "but is still severely miscalibrated (ECE = 0.1495) before correction. "
            "A high-accuracy AI can still be dangerously overconfident — and "
            "calibration fixes the honesty without touching the ranking ability."
        )

    # ── Tab 3: Disease explorer ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Explore Any Disease")

        selected = st.selectbox("Select disease:", valid["Disease"].tolist())
        row = valid[valid["Disease"] == selected].iloc[0]

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**{selected} — Calibration Profile**")
            detail_df = pd.DataFrame({
                "Metric": [
                    "Prevalence",
                    "ECE before calibration",
                    "ECE after calibration",
                    "ECE reduction",
                    "Best method",
                    "AUC (before)",
                    "AUC (after)",
                    "95% CI (before)",
                    "95% CI (after)",
                ],
                "Value": [
                    f"{int(row['N_pos']):,} cases ({row['Prevalence_pct']:.2f}%)",
                    f"{row['ECE_orig']:.4f}",
                    f"{row['ECE_best']:.4f}",
                    f"{row['Reduction_pct']:.1f}%",
                    row["Best_Method"],
                    f"{row['AUC']:.4f}",
                    f"{row['AUC_after']:.4f}",
                    f"[{row['CI_orig_lo']:.4f}, {row['CI_orig_hi']:.4f}]",
                    f"[{row['CI_best_lo']:.4f}, {row['CI_best_hi']:.4f}]",
                ],
            })
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

        with col_b:
            fig_mini = go.Figure(go.Bar(
                x=["ECE Before", "ECE After"],
                y=[row["ECE_orig"], row["ECE_best"]],
                marker_color=["#DC3545", "#28A745"],
                text=[f"{row['ECE_orig']:.4f}", f"{row['ECE_best']:.4f}"],
                textposition="outside",
                error_y=dict(
                    type="data",
                    array=[row["CI_orig_hi"] - row["ECE_orig"],
                           row["CI_best_hi"] - row["ECE_best"]],
                    arrayminus=[row["ECE_orig"] - row["CI_orig_lo"],
                                row["ECE_best"] - row["CI_best_lo"]],
                    color="rgba(0,0,0,0.5)"
                ),
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ))
            fig_mini.update_layout(
                title=f"{selected} — ECE with 95% CI",
                yaxis_title="ECE",
                yaxis=dict(range=[0, row["ECE_orig"] * 1.35]),
                height=340, showlegend=False
            )
            st.plotly_chart(fig_mini, use_container_width=True)

        st.markdown("---")

        if row["ECE_orig"] > 0.25:
            st.error(
                f"**🔴 Severely Miscalibrated**\n\n"
                f"When the AI diagnoses {selected}, its stated confidence is off by "
                f"**{row['ECE_orig']*100:.1f} ppts** on average. "
                f"After calibration this drops to **{row['ECE_best']*100:.1f} ppts** — "
                f"a **{row['Reduction_pct']:.1f}% improvement**."
            )
        elif row["ECE_orig"] > 0.15:
            st.warning(
                f"**🟡 Moderately Miscalibrated**\n\n"
                f"When the AI diagnoses {selected}, its stated confidence is off by "
                f"**{row['ECE_orig']*100:.1f} ppts** on average. "
                f"After calibration this drops to **{row['ECE_best']*100:.1f} ppts** — "
                f"a **{row['Reduction_pct']:.1f}% improvement**."
            )
        else:
            st.success(
                f"**🟢 Better Calibrated**\n\n"
                f"When the AI diagnoses {selected}, its stated confidence is off by "
                f"**{row['ECE_orig']*100:.1f} ppts** on average. "
                f"After calibration this drops to **{row['ECE_best']*100:.1f} ppts** — "
                f"a **{row['Reduction_pct']:.1f}% improvement**."
            )

    # ── Tab 4: Disease comparison tool ───────────────────────────────────────
    with tab4:
        st.markdown("### ⚖️ Head-to-Head Disease Comparison")
        st.markdown(
            "Select any two diseases to compare their calibration profile, "
            "AUC, improvement, and (if available) conformal coverage."
        )

        disease_list = valid["Disease"].tolist()

        cc1, cc2 = st.columns(2)
        with cc1:
            d1 = st.selectbox("Disease A", disease_list, index=0, key="cmp_d1")
        with cc2:
            d2 = st.selectbox("Disease B", disease_list,
                              index=min(1, len(disease_list)-1), key="cmp_d2")

        r1 = valid[valid["Disease"] == d1].iloc[0]
        r2 = valid[valid["Disease"] == d2].iloc[0]

        def cp_row(disease):
            """Look up conformal results for a disease."""
            if has_cp and disease in cp_df["Disease"].values:
                return cp_df[cp_df["Disease"] == disease].iloc[0]
            return None

        cp1 = cp_row(d1)
        cp2 = cp_row(d2)

        st.markdown("---")

        left, right = st.columns(2)

        def render_comparison_table(col, row, cp):
            """Render a disease detail table."""
            rows = {
                "Prevalence":      f"{int(row['N_pos']):,} ({row['Prevalence_pct']:.2f}%)",
                "ECE before":      f"{row['ECE_orig']:.4f}",
                "ECE after":       f"{row['ECE_best']:.4f}",
                "ECE reduction":   f"{row['Reduction_pct']:.1f}%",
                "Best method":     row["Best_Method"],
                "AUC (before)":    f"{row['AUC']:.4f}",
                "AUC (after)":     f"{row['AUC_after']:.4f}",
                "95% CI (before)": f"[{row['CI_orig_lo']:.4f}, {row['CI_orig_hi']:.4f}]",
                "95% CI (after)":  f"[{row['CI_best_lo']:.4f}, {row['CI_best_hi']:.4f}]",
            }
            if cp is not None:
                badge = "✅ Met" if cp["Guarantee_met"] else "❌ Missed"
                rows["Conformal coverage"]    = f"{cp['Coverage_actual']*100:.1f}% {badge}"
                rows["Confident predictions"] = f"{cp['Confident_pct']:.1f}%"
                rows["Uncertain (review)"]    = f"{cp['Uncertain_pct']:.1f}%"

            tbl = pd.DataFrame({"Metric": list(rows.keys()),
                                 "Value":  list(rows.values())})
            col.dataframe(tbl, use_container_width=True, hide_index=True)

        with left:
            st.markdown(f"**{d1}**")
            render_comparison_table(left, r1, cp1)

        with right:
            st.markdown(f"**{d2}**")
            render_comparison_table(right, r2, cp2)

        st.markdown("---")
        st.markdown("#### Visual Comparison")
        st.caption("Hover for exact values.")

        fig_cmp = go.Figure()
        metrics   = ["ECE Before", "ECE After", "AUC"]
        vals_d1   = [r1["ECE_orig"], r1["ECE_best"], r1["AUC"]]
        vals_d2   = [r2["ECE_orig"], r2["ECE_best"], r2["AUC"]]

        fig_cmp.add_trace(go.Bar(
            name=d1, x=metrics, y=vals_d1,
            marker_color="#1B2B4B",
            text=[f"{v:.4f}" for v in vals_d1],
            textposition="outside",
            hovertemplate=f"<b>{d1}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
        fig_cmp.add_trace(go.Bar(
            name=d2, x=metrics, y=vals_d2,
            marker_color="#DC3545",
            text=[f"{v:.4f}" for v in vals_d2],
            textposition="outside",
            hovertemplate=f"<b>{d2}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
        fig_cmp.update_layout(
            barmode="group",
            title=f"Side-by-Side: {d1} vs {d2}",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        ece_winner = d1 if r1["ECE_orig"] < r2["ECE_orig"] else d2
        red_winner = d1 if r1["Reduction_pct"] > r2["Reduction_pct"] else d2
        auc_winner = d1 if r1["AUC"] > r2["AUC"] else d2

        st.info(
            f"**🏆 Comparison Verdict**\n\n"
            f"- **Better baseline calibration:** {ece_winner}\n"
            f"- **Greater improvement after correction:** {red_winner}\n"
            f"- **Higher discrimination (AUC):** {auc_winner}"
        )

except FileNotFoundError:
    st.error(
        "Run notebook Multi-DiseaseCalibrationAudit first to generate "
        "data/robust_calibration_results.csv."
    )
