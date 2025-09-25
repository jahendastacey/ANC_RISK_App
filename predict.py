import joblib
import numpy as np

# Load model and feature names
model = joblib.load("anc_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Create a dictionary with patient data (you only fill in what you have)
patient_data = {
    "Age": 25,
    "Gravidity": 2,
    "Parity": 1,
    "Weight": 65,
    "Height": 160,
    "Blood Pressure": 120,
    "Diabetes": 0,
    "Hypertension": 0,
    "Previous complications": 0,
    "Smoking": 0
}

# Build full feature vector with 85 features
new_patient = []
for feature in feature_names:
    if feature in patient_data:
        new_patient.append(patient_data[feature])
    else:
        new_patient.append(0)  # default 0 for missing features

# Convert to numpy array with correct shape
new_patient = np.array(new_patient).reshape(1, -1)

# Predict
prediction = model.predict(new_patient)
print("Prediction:", prediction)