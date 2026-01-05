import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Earnings Manipulation Predictor",
    layout="centered"
)

st.title("Earnings Manipulation Prediction App")
st.write("Enter financial ratios to predict manipulation risk.")

# -----------------------------
# Load model and features
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, feature_names


model, feature_names = load_artifacts()

# -----------------------------
# User inputs (dynamic)
# -----------------------------
st.subheader("Financial Inputs")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        format="%.4f"
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        st.write(f"**Manipulation Probability:** {prob:.2%}")

    if prediction == 1:
        st.error("⚠ High Risk of Earnings Manipulation")
    else:
        st.success("✅ Low Risk of Earnings Manipulation")

# -----------------------------
# Disclaimer
# -----------------------------
st.caption(
    "Disclaimer: This tool is for academic and analytical purposes only."
)

