import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/model.pkl")

# Load dataset sample for dropdown examples
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard_small.csv")
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df.drop("Time", axis=1, inplace=True)
    return df[df['Class'] == 0].sample(5, random_state=42).reset_index(drop=True)

example_data = load_data()

st.title("💳 Credit Card Fraud Detector")
st.markdown("Select an example or manually enter transaction details to check for fraud risk.")

# Dropdown to choose example
example_index = st.selectbox("Choose example input (optional):", options=[f"Example #{i+1}" for i in range(len(example_data))])
use_example = st.checkbox("Use selected example")

# Inputs: V1 - V28
input_values = []
for i in range(1, 29):
    col_name = f"V{i}"
    default_val = example_data.loc[int(example_index.split('#')[-1]) - 1, col_name] if use_example else 0.0
    input_val = st.number_input(f"{col_name}", value=float(default_val), format="%.4f")
    input_values.append(input_val)

# Input: Amount
default_amt = example_data.loc[int(example_index.split('#')[-1]) - 1, "Amount"] if use_example else 0.0
amount = st.number_input("Amount", value=float(default_amt), format="%.2f")
input_values.append(amount)

# Predict
if st.button("🔍 Predict Fraud"):
    # Use DataFrame instead of array to avoid warning
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_df = pd.DataFrame([input_values], columns=feature_names)

    prediction = model.predict(X_df)[0]
    proba = model.predict_proba(X_df)[0][1]

    st.success(f"Prediction: {'🚨 FRAUD' if prediction == 1 else '✅ NOT FRAUD'}")
    #st.info(f"Confidence: {proba * 100:.2f}%")
