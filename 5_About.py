import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="About StatCalib",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About StatCalib")
st.subheader("Statistical Calibration Audit Framework for Medical AI Diagnostics")
st.caption("SSDI Course Project · MBATech Data Science Semester IV · NMIMS MPSTME Mumbai · Arnavv")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Data and Model")
    st.markdown("""
    - **Dataset:** NIH ChestX-ray14 (Wang et al. 2017)
    - **Images:** 10,000 real chest X-rays
    - **Model:** DenseNet-121 (torchxrayvision)
    - **Source:** Direct NIH download (nihcc.app.box.com)
    - **No simulation** — all scores from real inference
    """)

    st.markdown("### 🏥 Clinical Context")
    st.markdown("""
    - **Qure.ai** — Mumbai, $87M funded, NIH TB programme
    - **Niramai** — breast cancer screening, 22+ countries
    - **5C Network** — 3,000+ hospitals in India
    - **TRIPOD+AI 2024** — explicitly requires calibration reporting
    - **FDA data** — 43% of AI device recalls within year 1
    """)

    st.markdown("### 🔬 What StatCalib Proves")
    st.markdown("""
    1. **Miscalibration exists** — HL test: HL=799.15, p≈0
    2. **Every bin is miscalibrated** — Bonferroni Z-tests: all 10 significant
    3. **Correction works** — ECE reduced 93.5% by Isotonic Regression
    4. **Improvement is real** — Wilcoxon: p≈0, bootstrap CIs non-overlapping
    5. **AUC is preserved** — no loss of discrimination ability
    6. **Universal finding** — 12/14 diseases all show same pattern
    7. **Guarantees exist** — conformal prediction at ≥90% coverage
    """)

with col2:
    st.markdown("### 📚 SSDI Syllabus Mapping")
    ssdi = pd.DataFrame({
        "Unit": [
            "1 — Statistical Inference",
            "2 — Hypothesis Testing",
            "3 — Chi-Square",
            "4 — Regression",
            "5 — Ridge/Lasso",
            "6 — GLMs",
            "7 — Nonparametric",
        ],
        "Implementation": [
            "Bootstrap CI on ECE",
            "Z-tests + Bonferroni",
            "Hosmer-Lemeshow HL=799",
            "Platt Scaling",
            "Ridge-Platt (CV lambda)",
            "Logistic GLM framework",
            "Wilcoxon + Isotonic",
        ],
        "Result": [
            "ECE CI [0.161, 0.177]",
            "All 10 bins p < 0.001",
            "REJECT H₀ — p ≈ 0",
            "ECE: 0.1692 → 0.0165",
            "ECE: 0.1692 → 0.0170",
            "GLM interpretation",
            "Wilcoxon p ≈ 0",
        ],
    })
    st.dataframe(ssdi, use_container_width=True, hide_index=True)

st.markdown("---")

st.markdown("### 📈 Key Results Summary")

results_data = pd.DataFrame({
    "Metric": [
        "ECE before calibration (Effusion)",
        "ECE after Isotonic Regression",
        "ECE reduction (Effusion)",
        "HL test statistic",
        "HL p-value",
        "Wilcoxon p-value",
        "AUC preserved",
        "Avg ECE reduction (12 diseases)",
        "Conformal coverage @ 90% target",
        "Diseases meeting coverage guarantee",
    ],
    "Value": [
        "0.1692",
        "0.0110",
        "93.5%",
        "799.15",
        "≈ 0",
        "≈ 0",
        "0.7839 (unchanged)",
        "98.4%",
        "≥ 90%",
        "11/12",
    ],
    "Significance": [
        "Severe (threshold > 0.10)",
        "Well calibrated",
        "Statistically proven via Wilcoxon",
        "39× critical value",
        "Indistinguishable from zero",
        "Indistinguishable from zero",
        "Calibration preserves ranking",
        "Universal across all pathologies",
        "Mathematical guarantee",
        "Fibrosis: marginal miss (rare disease)",
    ]
})
st.dataframe(results_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 📖 References")
st.markdown("""
1. Wang et al. (2017). ChestX-ray8: Hospital-Scale Chest X-Ray Database. *CVPR*.
2. Guo et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
3. Hosmer & Lemeshow (2000). *Applied Logistic Regression*. Wiley.
4. Platt (1999). Probabilistic Outputs for Support Vector Machines. *Advances in Large Margin Classifiers*.
5. Zadrozny & Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. *KDD*.
6. Angelopoulos & Bates (2021). A Gentle Introduction to Conformal Prediction. *arXiv*.
7. TRIPOD+AI (2024). Transparent Reporting of AI Prediction Models. *BMJ*.
""")
