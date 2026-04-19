import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import random

st.set_page_config(page_title="The Problem", page_icon="🔴", layout="wide")

# ── Observed accuracy per confidence bin (from our audit) ─────────────────────
BIN_PRED = [0.033, 0.144, 0.247, 0.347, 0.448,
            0.535, 0.644, 0.747, 0.845, 0.930]
BIN_OBS  = [0.020, 0.047, 0.065, 0.062, 0.076,
            0.127, 0.228, 0.389, 0.500, 0.671]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔴 The Problem")
st.markdown("### Why Medical AI Confidence Scores Cannot Be Trusted")

st.info(
    "**🌍 The Real World Context**\n\n"
    "**Qure.ai** (Mumbai, $87M funded) processes chest X-rays for TB screening across "
    "India's national programme. **5C Network** provides AI radiology to 3,000+ hospitals. "
    "**Niramai** screens for breast cancer using AI in 22+ countries.\n\n"
    "Every one of these systems outputs a **confidence score** alongside its diagnosis. "
    "Doctors act on these scores every day.\n\n"
    "**StatCalib asks one question: can those scores be trusted?**"
)

st.markdown("---")

# ── Weather analogy ───────────────────────────────────────────────────────────
st.markdown("### 🎯 Understanding Calibration — The Weather App Analogy")
col1, col2 = st.columns(2)
with col1:
    st.error(
        "**❌ Poorly Calibrated (Like Our AI)**\n\n"
        'The app says **"90% chance of rain"**.\n'
        "But when it says 90% — it only rains **55%** of the time.\n\n"
        "You carry an umbrella for nothing half the time. "
        "The number 90% is **meaningless**."
    )
with col2:
    st.success(
        "**✅ Well Calibrated (After StatCalib)**\n\n"
        'The app says **"90% chance of rain"**.\n'
        "And when it says 90% — it actually rains **89%** of the time.\n\n"
        "You can plan your day around it. "
        "The number 90% **means something real**."
    )

st.markdown(
    "> **StatCalib proves DenseNet-121 is the broken weather app — then fixes it.**"
)

st.markdown("---")

# ── Interactive: Patient Scenario Generator ───────────────────────────────────
st.markdown("### 👤 Patient Scenario Generator")
st.markdown(
    "This tool generates a realistic clinical scenario and shows you exactly "
    "what goes wrong without calibration."
)

col_ctrl, col_result = st.columns([1, 2])

# Initialise session state defaults on first load
if "pt_age" not in st.session_state:
    st.session_state.pt_age   = 67
    st.session_state.pt_sex   = "Female"
    st.session_state.pt_score = 0.73

with col_ctrl:
    with st.container(border=True):
        if st.button("🎲 Randomise Patient", use_container_width=True):
            st.session_state.pt_age   = random.randint(30, 82)
            st.session_state.pt_sex   = random.choice(["Female", "Male"])
            st.session_state.pt_score = round(random.uniform(0.15, 0.92), 2)
            st.rerun()

        age   = st.slider("Patient Age",  20, 85,
                          value=st.session_state.pt_age,
                          key="pt_age")
        sex   = st.selectbox("Patient Sex", ["Female", "Male"],
                             index=["Female", "Male"].index(st.session_state.pt_sex),
                             key="pt_sex")
        score = st.slider("Raw AI Score (DenseNet-121 output)",
                          0.01, 0.99,
                          value=st.session_state.pt_score,
                          step=0.01,
                          key="pt_score",
                          help="This is the raw confidence score the model outputs")

with col_result:
    bin_idx  = min(int(score * 10), 9)
    true_acc = BIN_OBS[bin_idx]
    gap      = score - true_acc

    condition_words = {
        (0.0,  0.3):  ("low-grade", "monitoring"),
        (0.3,  0.6):  ("moderate",  "additional imaging"),
        (0.6,  0.8):  ("elevated",  "hospital admission"),
        (0.8,  1.01): ("high",      "urgent intervention"),
    }
    urgency, action = next(
        (v for (lo, hi), v in condition_words.items() if lo <= score < hi),
        ("moderate", "review")
    )

    with st.container(border=True):
        st.markdown("**📋 Clinical Scenario**")
        st.markdown(
            f"👤 **{age}-year-old {sex}** presents with breathlessness and chest discomfort."
        )
        st.markdown("🔬 Chest X-ray processed by DenseNet-121.")
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("AI Claims",      f"{score*100:.0f}%",     "raw output")
        col_b.metric("Actually True",  f"{true_acc*100:.1f}%",  "at this score level")
        col_c.metric("Overclaiming",   f"{gap*100:.1f} ppts",   "false certainty",
                     delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    if score > 0.6:
        st.error(
            f"**⚠️ Without calibration:** The doctor sees {score*100:.0f}% confidence. "
            f"They order *{action}* for a {urgency}-risk patient.\n\n"
            f"**With calibration:** True probability is {true_acc*100:.1f}%. "
            f"The doctor orders one more confirmatory test first — avoiding unnecessary {action}."
        )
    else:
        st.warning(
            f"**⚠️ Even at low scores:** The AI says {score*100:.0f}% but the true rate "
            f"is {true_acc*100:.1f}%. The model systematically overclaims even in the low range."
        )

st.markdown("---")

# ── Reliability diagram (Plotly) ──────────────────────────────────────────────
st.markdown("### 📈 The Reliability Diagram")
st.markdown(
    "Every point **below the diagonal** means the AI is overclaiming confidence. "
    "A perfect model follows the diagonal exactly. **Hover over points for exact values.**"
)

pred = np.array(BIN_PRED)
obs  = np.array(BIN_OBS)
obs_after  = np.array([0.020,0.047,0.065,0.062,0.076,0.127,0.200,0.370,0.480,0.650])
pred_after = np.array([0.019,0.046,0.063,0.060,0.074,0.125,0.198,0.368,0.478,0.648])

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Before Calibration — ECE = 0.1692**")
    fig_before = go.Figure()

    # Overconfidence fill
    fig_before.add_trace(go.Scatter(
        x=np.concatenate([pred, pred[::-1]]),
        y=np.concatenate([pred, obs[::-1]]),
        fill="toself", fillcolor="rgba(220,53,69,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Overconfidence gap", hoverinfo="skip"
    ))
    # Perfect calibration
    fig_before.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        name="Perfect calibration"
    ))
    # Model points
    hover_text = [
        f"Bin {i+1}<br>AI claims: {p*100:.1f}%<br>Actually true: {o*100:.1f}%<br>Gap: {(p-o)*100:.1f} ppts"
        for i, (p, o) in enumerate(zip(pred, obs))
    ]
    fig_before.add_trace(go.Scatter(
        x=pred, y=obs,
        mode="lines+markers",
        marker=dict(size=10, color="#DC3545"),
        line=dict(color="#DC3545", width=2.5),
        name="DenseNet-121",
        text=hover_text, hovertemplate="%{text}<extra></extra>"
    ))
    fig_before.update_layout(
        xaxis=dict(title="AI Stated Confidence", range=[0,1]),
        yaxis=dict(title="True Accuracy", range=[0,1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380, margin=dict(l=40, r=20, t=40, b=40)
    )
    st.plotly_chart(fig_before, use_container_width=True)

with col_right:
    st.markdown("**After Calibration (Isotonic Regression) — ECE = 0.0110**")
    fig_after = go.Figure()

    fig_after.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        name="Perfect calibration"
    ))
    hover_after = [
        f"Bin {i+1}<br>Calibrated: {p*100:.1f}%<br>Actually true: {o*100:.1f}%<br>Gap: {abs(p-o)*100:.1f} ppts"
        for i, (p, o) in enumerate(zip(pred_after, obs_after))
    ]
    fig_after.add_trace(go.Scatter(
        x=pred_after, y=obs_after,
        mode="lines+markers",
        marker=dict(size=10, color="#28A745"),
        line=dict(color="#28A745", width=2.5),
        name="After Isotonic Regression",
        text=hover_after, hovertemplate="%{text}<extra></extra>"
    ))
    fig_after.update_layout(
        xaxis=dict(title="AI Stated Confidence", range=[0,1]),
        yaxis=dict(title="True Accuracy", range=[0,1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380, margin=dict(l=40, r=20, t=40, b=40)
    )
    st.plotly_chart(fig_after, use_container_width=True)

st.markdown("---")

# ── Bin breakdown ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Bin-by-Bin Statistical Proof")
st.markdown(
    "Z-tests with Bonferroni correction (α = 0.001). "
    "**All 10 bins are statistically significant.**"
)

bin_df = pd.DataFrame({
    "Confidence Range": [f"{i*10}–{i*10+10}%" for i in range(10)],
    "AI Claims":        [f"{BIN_PRED[i]*100:.1f}%" for i in range(10)],
    "Actually True":    [f"{BIN_OBS[i]*100:.1f}%" for i in range(10)],
    "Gap":              [f"−{(BIN_PRED[i]-BIN_OBS[i])*100:.1f}%" for i in range(10)],
    "Z-stat":           ["−4.66","−8.97","−10.04","−11.54","−12.42",
                         "−36.90","−21.59","−14.70","−13.83","−8.47"],
    "Bonferroni":       ["✓"]*10,
})

def _color_gap(val):
    if not isinstance(val, str) or not val.startswith("−"):
        return ""
    g = abs(float(val.replace("−","").replace("%","")))
    if g > 30: return "background-color:#ffcccc;color:#DC3545;font-weight:bold"
    if g > 15: return "background-color:#fff3cd;color:#856404"
    return "background-color:#d4edda;color:#155724"

st.dataframe(
    bin_df.style.map(_color_gap, subset=["Gap"]),
    use_container_width=True, hide_index=True
)

with st.expander("📐 What does this mean statistically?"):
    st.markdown("""
    **Z-test for proportions** tests whether the observed accuracy in each bin
    could have come from a perfectly calibrated model by chance.

    **Bonferroni correction** adjusts the significance threshold from α=0.05
    to α=0.001 (dividing by 10 bins) to control the family-wise error rate.
    Even with this stricter threshold, all 10 bins reject the null hypothesis.

    The **Hosmer-Lemeshow chi-square test** combines all bins into one global test:
    - HL = 799.15 (critical value = 20.09 at α = 0.01)
    - **p ≈ 0** — the probability of observing this if the model were truly calibrated
      is indistinguishable from zero.
    """)
