# src/tracker_app.py

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from features import preprocess_features

# -------------------- Paths --------------------
MODEL_PATH = "../models/calorie_rf.pkl"
ENCODER_PATH = "../models/workout_encoder.pkl"
LOG_FILE = "../data/workout_log.csv"


# -------------------- Utilities --------------------
@st.cache_data
def load_log(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Make sure Date is a datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    return pd.DataFrame()


def append_log(entry: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([entry])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def safe_load_model(model_path, encoder_path):
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder, None
    except Exception as e:
        return None, None, str(e)


# -------------------- App --------------------
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🏋️‍♂️", layout="wide")
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.caption("Predict calories, log workouts, and view progress. (Forecasting removed as requested.)")

# Load model + encoder
model, encoder, load_err = safe_load_model(MODEL_PATH, ENCODER_PATH)
if load_err:
    st.error(f"Could not load model/encoder. Train them first. Details: {load_err}")

# Tabs (Forecast removed)
tab1, tab2 = st.tabs(["📋 Log Workout", "📈 View Progress"])

# -------------------- TAB 1: Log Workout --------------------
with tab1:
    st.subheader("Log a Workout (with calorie prediction)")
    col1, col2 = st.columns(2)

    with col1:
        workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "HIIT", "Yoga", "Mobility"])
        duration = st.slider("Duration (minutes)", 10, 180, 45)
        intensity = st.slider("Intensity (RPE 1–10)", 1, 10, 7)
        avg_hr = st.slider("Average Heart Rate (bpm)", 60, 200, 140)

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=72.0, step=0.1)
        age = st.number_input("Age", min_value=10, max_value=100, value=22)
        sex = st.radio("Sex", ["M", "F"], horizontal=True)
        date_input = st.date_input("Date", value=datetime.now())

    predict_and_log = st.button("🔥 Predict & Log Workout", disabled=(model is None or encoder is None))

    if predict_and_log:
        input_df = pd.DataFrame([{
            "Workout Type": workout_type,
            "Duration (mins)": duration,
            "Intensity (RPE)": intensity,
            "Avg Heart Rate": avg_hr,
            "Weight": weight,
            "Age": age,
            "Sex": sex
        }])

        # Preprocess & predict
        try:
            X_input, _ = preprocess_features(input_df, encoder=encoder, fit_encoder=False)
            predicted_calories = float(model.predict(X_input)[0])
            st.success(f"🔥 Estimated Calories Burned: **{predicted_calories:.2f} kcal**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            predicted_calories = None

        # Prepare log entry
        if predicted_calories is not None:
            log_entry = {
                "Date": pd.to_datetime(date_input).strftime("%Y-%m-%d"),
                "Workout Type": workout_type,
                "Duration (mins)": duration,
                "Calories Burned": round(predicted_calories, 2),
                "Intensity (RPE)": intensity,
                "Avg Heart Rate": avg_hr,
                "Weight": weight,
                "Age": age,
                "Sex": sex
            }

            append_log(log_entry, LOG_FILE)
            st.info("✅ Workout logged successfully!")
            st.cache_data.clear()  # clear cached history

# -------------------- TAB 2: View Progress --------------------
with tab2:
    st.subheader("Workout History & Quick Analytics")
    df = load_log(LOG_FILE)

    if df.empty:
        st.warning("No workout history found. Log your first session in the 'Log Workout' tab.")
    else:
        # Show table
        st.dataframe(df.sort_values("Date", ascending=False), use_container_width=True)

        # Daily aggregates for quick charts
        daily = df.groupby("Date", as_index=False).agg({
            "Calories Burned": "sum",
            "Duration (mins)": "sum"
        }).sort_values("Date")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔥 Daily Calories Burned")
            st.line_chart(daily.set_index("Date")["Calories Burned"])
        with c2:
            st.markdown("#### 🕒 Daily Duration (mins)")
            st.line_chart(daily.set_index("Date")["Duration (mins)"])

        # Basic stats
        st.markdown("### 📊 Summary Stats")
        st.write(daily.describe()[["Calories Burned", "Duration (mins)"]])

st.caption("Tip: retrain your model as you accumulate more personal data to improve accuracy.")
