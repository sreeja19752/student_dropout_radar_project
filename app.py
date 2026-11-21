import streamlit as st
import pandas as pd
from pathlib import Path

from modules.data_loader import load_data
from modules.model import train_model, predict_one, get_feature_columns
from modules.dashboard import show_dashboard
from modules.prediction import show_prediction
from modules.risk_analysis import show_risk_analysis
from modules.records import show_records

st.set_page_config(page_title="Student Dropout Radar", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "students.csv"

st.title("🎓 Student Dropout Radar")
st.caption("A student outcome prediction and early-warning system")

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
    st.header("Navigation")
    module = st.radio(
        "Choose a module",
        ["1. Dashboard", "2. Student Prediction", "3. Risk Analysis", "4. Student Records"]
    )
    st.divider()
    st.metric("Total students", f"{len(data):,}")
    st.caption("Model: Random Forest Classifier")

if module == "1. Dashboard":
    show_dashboard(data, metrics)
elif module == "2. Student Prediction":
    show_prediction(data, model, feature_columns)
elif module == "3. Risk Analysis":
    show_risk_analysis(data, model, feature_columns)
else:
    show_records(data)
