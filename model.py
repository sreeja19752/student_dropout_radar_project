import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TARGET = "target"
IGNORE = ["student_name", TARGET]

# These columns are coded categories in the supplied dataset, not continuous measurements.
CATEGORICAL_COLUMNS = [
    "marital_status", "application_mode", "application_order", "course",
    "daytime_evening_attendance", "displaced", "educational_special_needs",
    "debtor", "tuition_fees_up_to_date", "gender", "scholarship_holder",
    "international"
]

def get_feature_columns(data):
    return [c for c in data.columns if c not in IGNORE]

def build_pipeline(X):
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])

def train_model(data):
    feature_columns = get_feature_columns(data)
    X = data[feature_columns].copy()
    y = data[TARGET].astype(str)

    # Stratify keeps all three outcome classes represented in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_test, predictions, labels=sorted(y.unique())),
        "labels": sorted(y.unique()),
        "test_size": len(y_test)
    }
    return model, metrics, feature_columns

def predict_one(model, row, feature_columns):
    X = pd.DataFrame([row])[feature_columns]
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    probability_map = dict(zip(model.classes_, probabilities))
    return prediction, probability_map
