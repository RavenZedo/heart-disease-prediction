# Heart Disease Prediction

This project predicts the likelihood of heart disease using machine learning.

## Features
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Logistic Regression and Random Forest models
- Model evaluation using accuracy, recall, F1-score, ROC-AUC
- Streamlit web app for prediction

## Folder Structure
- `data/raw/` -> raw dataset
- `data/processed/` -> cleaned dataset
- `notebooks/` -> EDA and training notebooks
- `src/` -> preprocessing, training, evaluation, prediction scripts
- `models/` -> saved model
- `app/` -> Streamlit app

## Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app/streamlit_app.py