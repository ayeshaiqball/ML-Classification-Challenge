import streamlit as st
import pandas as pd
import pickle

# --- Page setup ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "This app uses a **Tuned Random Forest** model (trained on the UCI Heart Disease dataset) "
    "to estimate whether a patient is likely to have heart disease, based on clinical measurements."
)

# --- Load model and feature list ---
@st.cache_resource
def load_model():
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_order = load_model()

st.divider()
st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)

    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical angina",
            1: "Atypical angina",
            2: "Non-anginal pain",
            3: "Asymptomatic"
        }[x]
    )

    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)

    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    fasting_blood_sugar_high = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    resting_ecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy"
        }[x]
    )

with col2:
    max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

    exercise_induced_angina = st.selectbox(
        "Exercise-Induced Angina?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    st_depression = st.number_input(
        "ST Depression (exercise vs. rest)", min_value=0.0, max_value=7.0, value=1.0, step=0.1
    )

    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]
    )

    num_major_vessels = st.selectbox("Number of Major Vessels Colored (0–4)", options=[0, 1, 2, 3, 4])

    thalassemia = st.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "Normal",
            2: "Fixed defect",
            3: "Reversible defect"
        }[x]
    )

st.divider()

# --- Predict ---
if st.button("Predict", type="primary"):
    input_dict = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar_high": fasting_blood_sugar_high,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina": exercise_induced_angina,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "num_major_vessels": num_major_vessels,
        "thalassemia": thalassemia,
    }

    # Build the input row in the exact column order the model was trained on
    input_df = pd.DataFrame([input_dict])[feature_order]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Higher risk of heart disease — model confidence: {probability * 100:.1f}%")
    else:
        st.success(f"✅ Lower risk of heart disease — model confidence: {(1 - probability) * 100:.1f}%")

    st.progress(float(probability))
    st.caption(
        "This is a probability estimate from a machine learning model trained on a small dataset "
        "(302 patients). It is **not** a medical diagnosis — always consult a doctor for real clinical decisions."
    )

st.divider()
st.caption("Model: Tuned Random Forest | Test AUC: 0.903 | Dataset: UCI Heart Disease (post deduplication, 302 rows)")