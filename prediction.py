import streamlit as st
import pandas as pd
from modules.model import predict_one

def show_prediction(data, model, feature_columns):
    st.header("Module 2 — Student Prediction")
    st.write("Choose a student from the uploaded dataset. The model predicts one of the three outcomes: Dropout, Enrolled, or Graduate.")

    options = data["student_name"].fillna("Unnamed student").astype(str)
    selected_index = st.selectbox(
        "Select student",
        options=data.index.tolist(),
        format_func=lambda i: f"{data.loc[i, 'student_name']} (record {i + 1})"
    )
    row = data.loc[selected_index, feature_columns].to_dict()

    st.subheader("Student details used by the model")
    display_cols = [c for c in ["attendance_percentage", "age_at_enrollment",
                    "curricular_units_1st_sem_approved", "curricular_units_2nd_sem_approved",
                    "curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade",
                    "tuition_fees_up_to_date", "debtor", "scholarship_holder"] if c in row]
    st.dataframe(pd.DataFrame([row])[display_cols], use_container_width=True)

    if st.button("Predict student outcome", type="primary"):
        prediction, probabilities = predict_one(model, row, feature_columns)
        st.success(f"Predicted outcome: {prediction}")
        st.subheader("Prediction probabilities")
        probability_frame = pd.DataFrame({
            "Outcome": list(probabilities.keys()),
            "Probability": [float(v) for v in probabilities.values()]
        }).set_index("Outcome")
        st.bar_chart(probability_frame)
        st.caption("This is a decision-support estimate. Review the student's situation before taking action.")
