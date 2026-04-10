import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_data, preprocess_data, split_and_scale
from evaluate_model import evaluate_model


def print_results(name, results):
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Accuracy   : {results['accuracy']:.4f}")
    print(f"Recall     : {results['recall']:.4f}")
    print(f"Precision  : {results['precision']:.4f}")
    print(f"F1 Score   : {results['f1_score']:.4f}")
    if results["roc_auc"] is not None:
        print(f"ROC-AUC    : {results['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])


def main():
    df = load_data("data/raw/heart.csv")
    X, y, clean_df = preprocess_data(df)

    # save cleaned data
    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv("data/processed/clean_heart.csv", index=False)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale(X, y)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_train)
    log_results = evaluate_model(log_model, X_test_scaled, y_test)
    print_results("Logistic Regression", log_results)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    print_results("Random Forest", rf_results)

    # choose best model by ROC-AUC, fallback to accuracy
    log_score = log_results["roc_auc"] if log_results["roc_auc"] is not None else log_results["accuracy"]
    rf_score = rf_results["roc_auc"] if rf_results["roc_auc"] is not None else rf_results["accuracy"]

    os.makedirs("models", exist_ok=True)

    if rf_score >= log_score:
        joblib.dump(rf_model, "models/best_model.pkl")
        print("\nSaved best model: Random Forest -> models/best_model.pkl")
    else:
        joblib.dump({"model": log_model, "scaler": scaler}, "models/best_model.pkl")
        print("\nSaved best model: Logistic Regression + scaler -> models/best_model.pkl")


if __name__ == "__main__":
    main()