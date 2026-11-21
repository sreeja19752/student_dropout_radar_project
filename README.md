# Student Dropout Radar

A Streamlit project built around the supplied student dataset.

## Four modules
1. **Dashboard** — student totals, outcome distribution, attendance averages, and model performance.
2. **Student Prediction** — select a student and predict Dropout, Enrolled, or Graduate with class probabilities.
3. **Risk Analysis** — rank students by predicted dropout probability and export a risk report.
4. **Student Records** — search/filter records and download the filtered CSV.

## Dataset notes
- The supplied CSV has 4,424 rows and 29 columns.
- The target contains three classes: `Dropout`, `Enrolled`, and `Graduate`.
- `student_name` is kept for display only and excluded from model training.
- The model uses the remaining columns as predictors.
- No missing values or duplicate rows were found in the supplied file at build time.

## Run on Windows
1. Open Anaconda Prompt or Command Prompt in this project folder.
2. Create and activate an environment (optional but recommended):
   ```bash
   conda create -n dropout-radar python=3.11 -y
   conda activate dropout-radar
   ```
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

The browser will open the local Streamlit application.

## Important
This is an academic decision-support project. A prediction is not proof that a student will drop out. Use it to identify students who may need supportive follow-up, and review predictions with relevant staff.
