import pandas as pd

def load_data(path):
    """Load and validate the student CSV file."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().copy()

    if "target" not in data.columns:
        raise ValueError("A 'target' column is required.")
    data = data.dropna(subset=["target"])
    return data
