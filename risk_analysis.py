import streamlit as st
import pandas as pd
from modules.model import predict_one

def show_risk_analysis(data, model, feature_columns):
    st.header("Module 3 — Risk Analysis")
    st.write("Students predicted as Dropout are listed first. Risk is based on the model's predicted Dropout probability.")

    results = []
    for idx, row in data.iterrows():
        prediction, probs = predict_one(model, row[feature_columns].to_dict(), feature_columns)
        dropout_probability = float(probs.get("Dropout", 0.0))
        results.append({
            "Student": row.get("student_name", f"Student {idx + 1}"),
            "Attendance (%)": row.get("attendance_percentage"),
            "Actual outcome": row.get("target"),
            "Predicted outcome": prediction,
            "Dropout probability (%)": round(dropout_probability * 100, 1)
        })

    result_df = pd.DataFrame(results).sort_values("Dropout probability (%)", ascending=False)
    threshold = st.slider("Show students with dropout probability at least (%)", 0, 100, 40)
    filtered = result_df[result_df["Dropout probability (%)"] >= threshold]

    high = (result_df["Dropout probability (%)"] >= 70).sum()
    medium = ((result_df["Dropout probability (%)"] >= 40) & (result_df["Dropout probability (%)"] < 70)).sum()
    low = (result_df["Dropout probability (%)"] < 40).sum()
    a, b, c = st.columns(3)
    a.metric("High risk (70–100%)", int(high))
    b.metric("Medium risk (40–69.9%)", int(medium))
    c.metric("Low risk (below 40%)", int(low))

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download risk report (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="student_risk_report.csv",
        mime="text/csv"
    )
    st.caption("Risk bands are project thresholds, not clinical or official institutional categories.")
