# src/features.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_features(df, encoder=None, fit_encoder=True):
    df = df.copy()
    
    # Encode 'Sex' as binary
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

    X = df[[
        "Workout Type", "Duration (mins)", "Intensity (RPE)",
        "Avg Heart Rate", "Weight", "Age", "Sex"
    ]]

    if fit_encoder:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        workout_encoded = encoder.fit_transform(X[["Workout Type"]])
    else:
        workout_encoded = encoder.transform(X[["Workout Type"]])
    
    workout_df = pd.DataFrame(
        workout_encoded, columns=encoder.get_feature_names_out(["Workout Type"])
    )

    X_final = pd.concat([X.drop(columns=["Workout Type"]).reset_index(drop=True), 
                         workout_df.reset_index(drop=True)], axis=1)
    
    return X_final, encoder
