# src/predict_calories_app.py

import streamlit as st
import pandas as pd
import joblib
from features import preprocess_features

# Load model and encoder
model = joblib.load("../models/calorie_rf.pkl")
encoder = joblib.load("../models/workout_encoder.pkl")

# Streamlit UI
st.title("🔥 Calorie Burn Estimator")
st.write("Enter your workout details to estimate calories burned.")

# User Inputs
workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "HIIT", "Yoga", "Mobility"])
duration = st.slider("Duration (minutes)", 10, 120, 45)
intensity = st.slider("Intensity (RPE 1–10)", 1, 10, 7)
avg_hr = st.slider("Average Heart Rate (bpm)", 60, 190, 140)
weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=72.0)
age = st.number_input("Age", min_value=10, max_value=100, value=22)
sex = st.radio("Sex", ["M", "F"])

# Predict
if st.button("Estimate Calories"):
    input_df = pd.DataFrame([{
        "Workout Type": workout_type,
        "Duration (mins)": duration,
        "Intensity (RPE)": intensity,
        "Avg Heart Rate": avg_hr,
        "Weight": weight,
        "Age": age,
        "Sex": sex
    }])

    # Preprocess
    X_input, _ = preprocess_features(input_df, encoder=encoder, fit_encoder=False)

    # Predict
    calories = model.predict(X_input)[0]
    st.success(f" Estimated Calories Burned: **{calories:.2f} kcal**")
