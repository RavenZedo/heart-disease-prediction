import os
import sys
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

sys.path.append(SRC_DIR)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("❤️ Heart Disease Analytics Dashboard")
st.markdown("### Healthcare Predictive Analytics System")
st.write("Interactive dashboard for dataset exploration and heart disease risk prediction.")

# -----------------------------
# FILE CHECKS
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.info("Run this first in terminal: python src/train_model.py")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

saved_obj = joblib.load(MODEL_PATH)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Dashboard Filters")
st.sidebar.write("These filters affect only the charts and KPIs.")

min_age = int(df["age"].min())
max_age = int(df["age"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

sex_filter = st.sidebar.selectbox("Sex", ["All", "Male", "Female"])

filtered_df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

if sex_filter == "Male":
    filtered_df = filtered_df[filtered_df["sex"] == 1]
elif sex_filter == "Female":
    filtered_df = filtered_df[filtered_df["sex"] == 0]

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visual Analysis", "Prediction", "Dataset"])

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab1:
    st.subheader("Dashboard Overview")

    total_patients = len(filtered_df)
    heart_cases = int(filtered_df["target"].sum())
    healthy_cases = total_patients - heart_cases
    avg_age = round(filtered_df["age"].mean(), 2) if total_patients > 0 else 0
    avg_chol = round(filtered_df["chol"].mean(), 2) if total_patients > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Patients", total_patients)
    k2.metric("Heart Disease Cases", heart_cases)
    k3.metric("Healthy Patients", healthy_cases)
    k4.metric("Average Age", avg_age)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="target", data=filtered_df, ax=ax)
        ax.set_xlabel("Target")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with c2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df["age"], kde=True, ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Heart Disease Distribution")
        target_counts = filtered_df["target"].value_counts().sort_index()
        labels = ["No Disease", "Disease"] if len(target_counts) == 2 else target_counts.index.astype(str)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(target_counts, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Disease vs No Disease")
        st.pyplot(fig)

    with c4:
        st.subheader("Average Cholesterol")
        st.metric("Average Cholesterol", avg_chol)

        if total_patients > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(filtered_df["chol"], kde=True, ax=ax)
            ax.set_xlabel("Cholesterol")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

# =========================================================
# TAB 2: VISUAL ANALYSIS
# =========================================================
with tab2:
    st.subheader("Visual Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(filtered_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    with c2:
        st.subheader("Feature Importance")

        if isinstance(saved_obj, dict):
            st.info("Feature importance is shown best when Random Forest is the saved model.")
        else:
            if hasattr(saved_obj, "feature_importances_"):
                feature_cols = df.drop("target", axis=1).columns
                feature_importance = pd.Series(saved_obj.feature_importances_, index=feature_cols)
                feature_importance = feature_importance.sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax)
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
            else:
                st.warning("Saved model does not support feature importance.")

    st.markdown("---")

    b1, b2 = st.columns(2)

    with b1:
        st.subheader("Cholesterol vs Target")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="target", y="chol", data=filtered_df, ax=ax)
        ax.set_xlabel("Target")
        ax.set_ylabel("Cholesterol")
        st.pyplot(fig)

    with b2:
        st.subheader("Max Heart Rate vs Target")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="target", y="thalach", data=filtered_df, ax=ax)
        ax.set_xlabel("Target")
        ax.set_ylabel("Max Heart Rate")
        st.pyplot(fig)

    b3, b4 = st.columns(2)

    with b3:
        st.subheader("Age vs Target")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="target", y="age", data=filtered_df, ax=ax)
        ax.set_xlabel("Target")
        ax.set_ylabel("Age")
        st.pyplot(fig)

    with b4:
        st.subheader("Chest Pain Type vs Target")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="cp", hue="target", data=filtered_df, ax=ax)
        ax.set_xlabel("Chest Pain Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# =========================================================
# TAB 3: PREDICTION
# =========================================================
with tab3:
    st.subheader("Heart Disease Risk Prediction")
    st.caption("Enter details for a single patient to predict risk.")

    with st.container(border=True):
        st.markdown("### Patient Input Panel")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Basic Information")
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic"
            )
            trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with right:
            st.markdown("#### Clinical Details")
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
            exang = st.selectbox(
                "Exercise Induced Angina",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thal", [0, 1, 2, 3])

        st.markdown("### Quick Patient Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Age", age)
        s2.metric("Sex", "Male" if sex == 1 else "Female")
        s3.metric("BP", trestbps)
        s4.metric("Cholesterol", chol)

    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    p1, p2, p3 = st.columns([1, 1, 2])

    with p1:
        predict_clicked = st.button("Predict Risk", use_container_width=True)

    with p2:
        clear_clicked = st.button("Refresh View", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if predict_clicked:
        input_df = pd.DataFrame([input_data])

        try:
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

            st.markdown("---")
            st.markdown("## Prediction Result")

            r1, r2 = st.columns([1, 2])

            with r1:
                st.metric("Risk Probability", f"{probability * 100:.2f}%")
                st.progress(float(probability))

            with r2:
                if probability < 0.30:
                    st.success("✅ Low Risk Category")
                elif probability < 0.70:
                    st.warning("⚠️ Moderate Risk Category")
                else:
                    st.error("🚨 High Risk Category")

                if prediction == 1:
                    st.error("Model Prediction: Heart disease risk detected")
                else:
                    st.success("Model Prediction: Low heart disease risk detected")

            st.markdown("### Patient vs Dataset Average")
            avg1, avg2, avg3 = st.columns(3)
            avg1.metric("Age vs Avg", age, f"{age - df['age'].mean():.1f}")
            avg2.metric("Chol vs Avg", chol, f"{chol - df['chol'].mean():.1f}")
            avg3.metric("Heart Rate vs Avg", thalach, f"{thalach - df['thalach'].mean():.1f}")

            st.markdown("### Input Review")
            result_df = pd.DataFrame([{
                **input_data,
                "prediction": int(prediction),
                "probability": float(probability)
            }])

            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Report",
                csv,
                "prediction_report.csv",
                "text/csv",
                use_container_width=False
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with st.expander("Medical Term Guide"):
        st.write("**cp**: Chest pain type")
        st.write("**trestbps**: Resting blood pressure")
        st.write("**chol**: Serum cholesterol")
        st.write("**fbs**: Fasting blood sugar")
        st.write("**restecg**: Resting electrocardiographic result")
        st.write("**thalach**: Maximum heart rate achieved")
        st.write("**exang**: Exercise induced angina")
        st.write("**oldpeak**: ST depression induced by exercise")
        st.write("**slope**: Slope of peak exercise ST segment")
        st.write("**ca**: Number of major vessels")
        st.write("**thal**: Thalassemia-related test value")

# =========================================================
# TAB 4: DATASET
# =========================================================
with tab4:
    st.subheader("Dataset Explorer")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("### Dataset Shape")
        st.write(filtered_df.shape)

        st.markdown("### Column Names")
        st.write(list(filtered_df.columns))

    with d2:
        st.markdown("### Missing Values")
        st.write(filtered_df.isnull().sum())

    st.markdown("---")

    st.markdown("### Dataset Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.markdown("### Statistical Summary")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    with st.expander("Show Full Filtered Dataset"):
        st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.subheader("About")
st.write("This dashboard combines exploratory analytics with machine learning-based heart disease prediction.")
st.info("Educational use only. Not for real medical diagnosis.")