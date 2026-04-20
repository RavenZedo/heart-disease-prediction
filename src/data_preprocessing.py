import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path="data/raw/heart.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def preprocess_data(df):
    df = df.drop_duplicates().copy()

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # Outlier handling using simple threshold filtering
    if "chol" in df.columns:
        df = df[df["chol"] < 500]

    if "trestbps" in df.columns:
        df = df[df["trestbps"] < 200]

    target_col = "target"
    if target_col not in df.columns:
        raise ValueError("Target column 'target' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, df


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler