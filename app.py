import streamlit as st
import pandas as pd
from pathlib import Path

from modules.data_loader import load_data
from modules.model import train_model
from modules.dashboard import show_dashboard
from modules.prediction import show_prediction
from modules.risk_analysis import show_risk_analysis
from modules.records import show_records

st.set_page_config(
    page_title="Student Dropout Radar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d19; --panel:#111a2b; --panel2:#151f33; --muted:#91a0b8; --text:#edf3ff; --blue:#5b8cff; --cyan:#43d5e8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #14264a 0%, #080d19 42%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(8,13,25,.72); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101a2d,#0b1220); border-right:1px solid #263550; }
[data-testid="stSidebar"] * { color:#e8efff; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { color:#f3f7ff !important; letter-spacing:-.03em; }
p, label, .stCaption { color:#aab8ce; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#17243a,#101a2b); border:1px solid #2a3b59; padding:18px 20px; border-radius:16px; box-shadow:0 10px 30px #0002; }
div[data-testid="stMetricLabel"] { color:#9fb0ca; }
div[data-testid="stMetricValue"] { color:#f4f7ff; font-weight:800; }
.stButton>button, .stDownloadButton>button { background:linear-gradient(90deg,#477cf5,#6b63e8); color:white; border:0; border-radius:10px; padding:.65rem 1.1rem; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { border:1px solid #8baaff; color:white; }
.stSelectbox>div>div, .stMultiSelect>div>div, .stTextInput>div>div, .stSlider { background:#111d30; border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid #2a3b59; border-radius:12px; overflow:hidden; }
hr { border-color:#293954; }
div[data-testid="stRadio"] label { background:#142138; padding:10px 12px; border-radius:10px; margin:4px 0; border:1px solid #263954; }
div[data-testid="stRadio"] label:hover { border-color:#5b8cff; }
.hero { padding:24px 28px; border:1px solid #2a3e60; border-radius:20px; background:linear-gradient(110deg,#172b4d 0%,#142039 58%,#17213b 100%); margin-bottom:24px; box-shadow:0 16px 40px #0003; }
.eyebrow { color:#74c9ff; font-size:12px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.hero-title { color:#f5f8ff; font-size:32px; font-weight:800; margin:6px 0; }
.hero-sub { color:#a9bad4; font-size:14px; margin:0; }
.section-note { color:#91a4c0; font-size:13px; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "students.csv"

try:
    data = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

required = {"student_name", "target", "attendance_percentage"}
missing = required - set(data.columns)
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    st.stop()
if data.empty:
    st.error("The dataset is empty. Add student records to data/students.csv.")
    st.stop()

@st.cache_resource
def get_trained_model(frame):
    return train_model(frame)

try:
    model, metrics, feature_columns = get_trained_model(data)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🎓 Dropout Radar")
    st.caption("STUDENT SUCCESS INTELLIGENCE")
    st.divider()
    module = st.radio(
        "WORKSPACE",
        ["Overview", "Student Prediction", "Risk Analysis", "Student Records"],
        format_func=lambda x: {
            "Overview":"◈  Dashboard",
            "Student Prediction":"◎  Predict outcome",
            "Risk Analysis":"◉  Risk analysis",
            "Student Records":"▤  Student records"
        }[x]
    )
    st.divider()
    st.metric("Students in dataset", f"{len(data):,}")
    st.markdown('<div class="section-note">Model engine<br><b>Random Forest Classifier</b></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Academic decision-support tool • v1.1")

st.markdown("""
<div class="hero">
  <div class="eyebrow">STUDENT SUCCESS • EARLY WARNING SYSTEM</div>
  <div class="hero-title">Student Dropout Radar</div>
  <p class="hero-sub">Understand student outcomes, identify potential risk, and support better academic decisions.</p>
</div>
""", unsafe_allow_html=True)

if module == "Overview":
    show_dashboard(data, metrics)
elif module == "Student Prediction":
    show_prediction(data, model, feature_columns)
elif module == "Risk Analysis":
    show_risk_analysis(data, model, feature_columns)
else:
    show_records(data)
