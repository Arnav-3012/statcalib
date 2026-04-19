import streamlit as st

st.set_page_config(
    page_title="StatCalib — Medical AI Audit",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.title("🏥 StatCalib")
st.subheader("Statistical Calibration Audit Framework for Medical AI Diagnostics")
st.caption(
    "Real NIH ChestX-ray14 Data  ·  DenseNet-121  ·  "
    "10,000 Chest X-rays  ·  12 Diseases Audited"
)

st.markdown("---")

# ── One-line problem ──────────────────────────────────────────────────────────
st.error(
    "**⚠️ The Problem in One Sentence**\n\n"
    'When this medical AI says **"I am 80% confident this patient has Effusion"** '
    "— it is actually only correct **about 39% of the time** in that confidence range. "
    "That gap is what StatCalib detects and fixes."
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Key metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:  st.metric("X-rays Analysed",   "10,000",  "Real NIH images")
with col2:  st.metric("Original ECE",       "0.1692",  "↓ Severe",   delta_color="inverse")
with col3:  st.metric("After Correction",   "0.0110",  "↑ 93.5%")
with col4:  st.metric("Diseases Audited",   "12 / 14", "98.4% avg ECE reduction")
with col5:  st.metric("Conformal Coverage", "≥ 90%",   "Mathematically guaranteed")

st.markdown("---")

# ── Navigation grid ───────────────────────────────────────────────────────────
st.markdown("### 🗺️ Navigate the Dashboard")

nav_pages = [
    ("🔴", "The Problem",        "Why AI confidence scores are dangerous",      "Sidebar → Page 1", "#DC3545"),
    ("🔬", "Effusion Deep Dive", "Full audit + live calibration calculator",    "Sidebar → Page 2", "#1E88E5"),
    ("📊", "All 14 Diseases",    "Systematic audit + disease comparison tool",  "Sidebar → Page 3", "#28A745"),
    ("🎯", "Conformal",          "Mathematical guarantees on coverage",         "Sidebar → Page 4", "#9C27B0"),
    ("ℹ️",  "About",              "Methods, SSDI mapping, references",          "Sidebar → Page 5", "#5C6BC0"),
]

cols = st.columns(5)
for col, (icon, title, desc, hint, color) in zip(cols, nav_pages):
    with col:
        st.markdown(
            f"""
            <div style="
                background:{color};
                border-radius:10px;
                padding:20px 16px;
                height:160px;
                display:flex;
                flex-direction:column;
                justify-content:space-between;
            ">
                <div style="font-size:26px;line-height:1">{icon}</div>
                <div style="font-weight:700;color:#fff;font-size:15px;margin-top:8px">{title}</div>
                <div style="color:rgba(255,255,255,0.85);font-size:12px;flex-grow:1;margin-top:6px">{desc}</div>
                <div style="color:rgba(255,255,255,0.60);font-size:11px;margin-top:8px">{hint}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "SSDI Course Project · MBATech Data Science Sem IV · "
    "NMIMS MPSTME Mumbai · Arnavv"
)
