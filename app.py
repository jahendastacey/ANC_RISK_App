# app.py - Streamlit ANC risk app (minimal, safe)
import streamlit as st
import joblib
import numpy as np
import os
import glob

st.title("🤰 ANC Risk Prediction (local)")

# --- Load model file (try common names) ---
model_path = None
candidates = ["anc_risk_model.pkl"] + glob.glob("anc_risk_model*.pkl")
for c in candidates:
    if os.path.exists(c):
        model_path = c
        break

if model_path is None:
    st.error("Model file not found. Put anc_risk_model.pkl (or anc_risk_model(1).pkl) in the same folder as this app.")
    st.stop()

model = joblib.load(model_path)

# --- Load feature names (feature_names.pkl expected) ---
feat_path = None
if os.path.exists("feature_names.pkl"):
    feat_path = "feature_names.pkl"
else:
    # try any file with feature_names*.pkl
    g = glob.glob("feature_names*.pkl")
    if g:
        feat_path = g[0]

if feat_path is None:
    st.error("feature_names.pkl not found in this folder. Place the file exported from training here.")
    st.stop()

feature_names = joblib.load(feat_path)
# ensure list
if not isinstance(feature_names, list):
    try:
        feature_names = list(feature_names)
    except:
        st.error("feature_names.pkl format not recognized.")
        st.stop()

st.write("Model loaded from:", model_path)
st.write("Number of features expected:", len(feature_names))

# --- Inputs (you can add more later) ---
st.subheader("Patient details (fill what you know)")
gestation_days = st.number_input("Gestation days", min_value=0, max_value=300, value=200)
bp_systolic = st.number_input("Blood Pressure (Systolic)", min_value=50, max_value=220, value=120)
bp_diastolic = st.number_input("Blood Pressure (Diastolic)", min_value=30, max_value=140, value=80)
age = st.number_input("Age", min_value=10, max_value=60, value=25)
bmi = st.number_input("BMI / weight", min_value=8.0, max_value=80.0, value=22.0)
haemoglobin = st.number_input("Haemoglobin level (g/dL)", min_value=2.0, max_value=20.0, value=12.0)

history_csection = st.selectbox("History of C-section", ("No", "Yes"))
history_miscarriage = st.selectbox("History of miscarriage", ("No", "Yes"))
diabetes = st.selectbox("Diabetes history", ("No", "Yes"))
hypertension = st.selectbox("Hypertension history", ("No", "Yes"))
multiple_gestation = st.selectbox("Multiple gestation", ("No", "Yes"))

# convert to numeric
binary = {"No": 0, "Yes": 1}
history_csection = binary[history_csection]
history_miscarriage = binary[history_miscarriage]
diabetes = binary[diabetes]
hypertension = binary[hypertension]
multiple_gestation = binary[multiple_gestation]

# --- Helper to set a feature by checking multiple possible column names ---
def set_feature_by_candidates(new_vec, names_list, value):
    candidates = names_list
    for cand in candidates:
        if cand in feature_names:
            idx = feature_names.index(cand)
            new_vec[idx] = value
            return True
    return False

# Create feature vector of correct length, default 0
x = np.zeros(len(feature_names))

# Try to place values into the right columns (checks a few possible column name variants)
set_feature_by_candidates(x, ["Gestation_days", "gestation_days", "estation_days"], gestation_days)
set_feature_by_candidates(x, ["Blood pressure(Systolic)", "Blood pressure (Systolic)", "Systolic", "bp_systolic"], bp_systolic)
set_feature_by_candidates(x, ["Blood pressure (Diastolic)", "Blood pressure(Diastolic)", "Diastolic", "bp_diastolic"], bp_diastolic)
set_feature_by_candidates(x, ["AGE", "Age", "age"], age)
set_feature_by_candidates(x, ["BMI/ weight", "BMI", "bmi", "BMI/weight"], bmi)
set_feature_by_candidates(x, ["Haemoglobin level", "Haemoglobin", "Haemoglobin_level", "haemoglobin"], haemoglobin)
set_feature_by_candidates(x, ["History of c-section", "History of c-section", "history_of_c_section"], history_csection)
set_feature_by_candidates(x, ["History of miscarriage", "History of miscarriage", "history_of_miscarriage"], history_miscarriage)
set_feature_by_candidates(x, ["Diabetes History", "Diabetes History", "Diabetes", "diabetes_history"], diabetes)
set_feature_by_candidates(x, ["Hpertension history", "Hypertension history", "Hypertension", "Hypertension history"], hypertension)
set_feature_by_candidates(x, ["Multiple gestation(YES/NO)", "Multiple gestation(YES/NO)", "Multiple gestation", "multiple_gestation"], multiple_gestation)

# --- Predict button ---
if st.button("Predict"):
    try:
        pred = model.predict([x])[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([x])[0]
        if pred == 1:
            st.error(f"⚠️ HIGH RISK (pred={pred})" + (f" — prob={prob[1]:.2f}" if prob is not None else ""))
        else:
            st.success(f"✅ LOW RISK (pred={pred})" + (f" — prob={prob[1]:.2f}" if prob is not None else ""))
    except Exception as e:
        st.error("Prediction failed: " + str(e))
