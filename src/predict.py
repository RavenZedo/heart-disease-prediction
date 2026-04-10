import joblib
import pandas as pd


def predict_single(input_data, model_path="models/best_model.pkl"):
    saved_obj = joblib.load(model_path)

    input_df = pd.DataFrame([input_data])

    if isinstance(saved_obj, dict):
        model = saved_obj["model"]
        scaler = saved_obj["scaler"]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        model = saved_obj
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    return prediction, probability