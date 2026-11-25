import streamlit as st
import pandas as pd

def show_dashboard(data, metrics):
    st.header("Your student overview")
    st.caption("A live snapshot of academic outcomes and model performance.")
    total = len(data)
    counts = data["target"].value_counts()
    dropout = int(counts.get("Dropout", 0))
    enrolled = int(counts.get("Enrolled", 0))
    graduate = int(counts.get("Graduate", 0))
    rate = dropout / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", f"{total:,}", help="Total records in the current dataset")
    c2.metric("Dropout records", f"{dropout:,}", f"{rate:.1%} of dataset")
    c3.metric("Currently enrolled", f"{enrolled:,}")
    c4.metric("Graduated", f"{graduate:,}")

    st.divider()
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Student outcome distribution")
        st.bar_chart(counts.rename_axis("Outcome"), color="#648cff")
    with right:
        st.subheader("Average attendance by outcome")
        attendance = data.groupby("target")["attendance_percentage"].mean().sort_values(ascending=False)
        st.bar_chart(attendance.rename_axis("Outcome"), color="#38c9d8")

    st.divider()
    st.subheader("Model performance")
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        st.caption("Measured on a held-out 20% test split.")
    with m2:
        report = pd.DataFrame(metrics["report"]).T
        visible = [c for c in ["precision", "recall", "f1-score", "support"] if c in report.columns]
        st.dataframe(report[visible].round(3), use_container_width=True)
    st.info("Predictions are decision-support estimates, not proof of a student's future outcome. Review each case with appropriate staff.")
