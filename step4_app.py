import streamlit as st
import joblib
import numpy as np

# Load model and feature list
model = joblib.load('best_model_fraud.pkl')
features = joblib.load('model_features.pkl')

st.title("Credit Card Fraud Detection App")

st.write("Enter the transaction details below to check if it's fraudulent:")

# Collect user inputs dynamically
user_input = []
for feature in features:
    value = st.number_input(f"Enter value for {feature}:", step=0.01, format="%.2f")
    user_input.append(value)

# Predict
if st.button("Check Fraud"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This transaction is likely **FRAUDULENT** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This transaction is **NOT fraudulent** (Probability: {probability:.2f})")
