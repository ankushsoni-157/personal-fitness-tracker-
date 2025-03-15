import streamlit as st
import pandas as pd
import base64

# Function to encode local image to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Provide correct local path of the image
image_path = "fitness_2.webp"  # Ensure this file is in the same directory or provide a full path
base64_image = get_base64_of_image(image_path)

# Custom CSS for Background Image & Styling
page_bg_img = f'''
<style>
.stApp {{
    background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                url("data:image/png;base64,{base64_image}") no-repeat center center fixed;
    background-size: cover;
    color: white;
}}
.stApp {{
    background-color: rgba(0, 0, 0, 0.6);
    padding: 2rem;
    border-radius: 10px;
}}
.stSuccess {{
    color: white !important;
    padding: 10px;
    border-radius: 5px;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load fitness dataset
dataset_path = "dataset-2.csv"
diet_chart_path = "diet_dataset_2.csv"
try:
    df = pd.read_csv(dataset_path)
    diet_df = pd.read_csv(diet_chart_path)
except FileNotFoundError:
    df = None  # Fallback if dataset isn't available
    diet_df = None

st.title("🏋 Personal Fitness Tracker")
st.write("### Track your health & stay fit!")

# Sidebar for User Input
st.sidebar.header("🔹 Enter Your Details")
age = st.sidebar.number_input("Age:", min_value=5, max_value=100, step=1)
height = st.sidebar.number_input("Height (cm):", min_value=50, max_value=250, step=1)
weight = st.sidebar.number_input("Weight (kg):", min_value=10, max_value=300, step=1)
steps = st.sidebar.number_input("Daily Steps:", min_value=0, step=100)
workout = st.sidebar.number_input("Workout Minutes:", min_value=0, step=5)
sleep = st.sidebar.number_input("Sleep Hours:", min_value=0.0, step=0.5)

# BMI Calculator
def calculate_bmi(weight, height):
    if height > 0:
        return round(weight / ((height / 100) ** 2), 2)
    return None

# Calories Burned Calculator
def calculate_calories_burned(steps, workout, weight):
    return round((steps * 0.04) + (workout * 7), 2)

# Fitness Recommendation
def recommend_workout(steps, workout, sleep, bmi):
    step_advice = "Great job! Maintain your activity level." if steps >= 10000 else "Try to hit 10,000 steps daily."
    workout_advice = "You're working out enough!" if workout >= 30 else "Increase your workout time to at least 30 minutes."
    sleep_advice = "Your sleep is good!" if sleep >= 6 else "Try to get 7-8 hours of sleep."
    bmi_advice = f"Your BMI is '{bmi}', which is in the "
    if bmi < 18.5:
        bmi_advice += "Underweight category. Consider gaining some weight."
        color = "orange"
    elif 18.5 <= bmi < 25:
        bmi_advice += "Healthy Weight range. Keep up the good work!"
        color = "green"
    elif 25 <= bmi < 30:
        bmi_advice += "Overweight category. Consider adjusting your diet & exercise."
        color = "red"
    else:
        bmi_advice += "Obese category. Focus on fitness & a balanced diet."
        color = "red"
    return f"{step_advice} {workout_advice} {sleep_advice} {bmi_advice}", color

# Generate Diet Chart based on BMI
def generate_diet_chart(bmi):
    if diet_df is not None:
        diet_info = diet_df[diet_df['BMI_Category'] == ('Underweight' if bmi < 18.5 else 'Healthy Weight' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese')]
        if not diet_info.empty:
            available_columns = [col for col in ['Diet_Recommended', 'Calorie_Intake', 'Exercise_Recommendation'] if col in diet_info.columns]
            return diet_info.iloc[0][available_columns].to_dict()
    return "No diet data available."

# Process User Data
bmi = calculate_bmi(weight, height)
calories_burned = calculate_calories_burned(steps, workout, weight)
diet_chart = generate_diet_chart(bmi)

if st.sidebar.button("Get Recommendation"):
    st.subheader("📊 Your Fitness Summary")
    st.write(f"🔥 *Calories Burned Today:* {calories_burned} kcal")
    st.write(f"⚖ *Your BMI:* {bmi}")
    
    result, color = recommend_workout(steps, workout, sleep, bmi)
    st.markdown(f'<div class="stSuccess" style="background-color:{color};">{result}</div>', unsafe_allow_html=True)
    
    st.subheader("🥗 Recommended Diet Plan")
    if isinstance(diet_chart, dict):
        for key, value in diet_chart.items():
            st.write(f"**{key}:** {value}")
    else:
        st.write(diet_chart)
