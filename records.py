import streamlit as st

def show_records(data):
    st.header("Module 4 — Student Records")
    st.write("Search, filter, and download student information from the dataset.")

    search = st.text_input("Search by student name")
    outcomes = sorted(data["target"].dropna().astype(str).unique().tolist())
    selected = st.multiselect("Filter by outcome", outcomes, default=outcomes)

    filtered = data[data["target"].astype(str).isin(selected)].copy()
    if search.strip():
        filtered = filtered[filtered["student_name"].astype(str).str.contains(search.strip(), case=False, na=False)]

    st.caption(f"Showing {len(filtered):,} of {len(data):,} records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered student records (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_student_records.csv",
        mime="text/csv"
    )
