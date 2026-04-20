import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from data_preprocessing import load_data, preprocess_data, split_and_scale
from evaluate_model import evaluate_model


def print_results(name, results):
    print(f"\n{'=' * 55}")
    print(name)
    print(f"{'=' * 55}")
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
    print("Loading dataset...")
    df = load_data("data/raw/heart.csv")

    print("Preprocessing dataset...")
    X, y, clean_df = preprocess_data(df)

    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv("data/processed/clean_heart.csv", index=False)
    print("Saved cleaned dataset to data/processed/clean_heart.csv")

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale(X, y)

    # -----------------------------
    # 1. Logistic Regression
    # -----------------------------
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_train)
    log_results = evaluate_model(log_model, X_test_scaled, y_test)
    print_results("Logistic Regression", log_results)

    # -----------------------------
    # 2. Random Forest with Hyperparameter Tuning
    # -----------------------------
    print("\nRunning GridSearchCV for Random Forest...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    print("\nBest Random Forest Parameters:", grid.best_params_)

    rf_results = evaluate_model(best_rf, X_test, y_test)
    print_results("Tuned Random Forest", rf_results)

    # -----------------------------
    # 3. XGBoost
    # -----------------------------
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test)
    print_results("XGBoost", xgb_results)

    # -----------------------------
    # Model Selection
    # -----------------------------
    log_score = log_results["roc_auc"] if log_results["roc_auc"] is not None else log_results["accuracy"]
    rf_score = rf_results["roc_auc"] if rf_results["roc_auc"] is not None else rf_results["accuracy"]
    xgb_score = xgb_results["roc_auc"] if xgb_results["roc_auc"] is not None else xgb_results["accuracy"]

    os.makedirs("models", exist_ok=True)

    best_model_name = ""
    best_model_object = None

    if log_score >= rf_score and log_score >= xgb_score:
        best_model_name = "Logistic Regression"
        best_model_object = {"model": log_model, "scaler": scaler}
    elif rf_score >= log_score and rf_score >= xgb_score:
        best_model_name = "Tuned Random Forest"
        best_model_object = best_rf
    else:
        best_model_name = "XGBoost"
        best_model_object = xgb_model

    joblib.dump(best_model_object, "models/best_model.pkl")
    print(f"\nSaved best model: {best_model_name} -> models/best_model.pkl")


if __name__ == "__main__":
    main()