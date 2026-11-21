import streamlit as st
import pandas as pd

def show_dashboard(data, metrics):
    st.header("Module 1 — Dashboard")
    total = len(data)
    counts = data["target"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", f"{total:,}")
    c2.metric("Dropout", f"{counts.get('Dropout', 0):,}")
    c3.metric("Enrolled", f"{counts.get('Enrolled', 0):,}")
    c4.metric("Graduated", f"{counts.get('Graduate', 0):,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Student outcome distribution")
        st.bar_chart(counts)
    with right:
        st.subheader("Average attendance by outcome")
        attendance = data.groupby("target")["attendance_percentage"].mean().sort_values(ascending=False)
        st.bar_chart(attendance)

    st.subheader("Model performance")
    st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    report = pd.DataFrame(metrics["report"]).T
    visible = [c for c in ["precision", "recall", "f1-score", "support"] if c in report.columns]
    st.dataframe(report[visible].round(3), use_container_width=True)
    st.caption("Accuracy is measured on a held-out 20% test split. It is not a guarantee for future students.")
