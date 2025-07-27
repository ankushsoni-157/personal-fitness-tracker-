# src/train_calorie_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from features import preprocess_features

# Load data
df = pd.read_csv("../data/fitness_data_realistic.csv")

# Preprocess features
X, encoder = preprocess_features(df)
y = df["Calories Burned"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE on test set: {mae:.2f} kcal")

# Save model and encoder
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/calorie_rf.pkl")
joblib.dump(encoder, "../models/workout_encoder.pkl")
