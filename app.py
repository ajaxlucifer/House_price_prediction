

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page configuration
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

# Load trained model and feature list
model = joblib.load("house_price_rf_model.pkl")
feature_names = joblib.load("model_features.pkl")

# App title
st.title("🏠 House Price Prediction")
st.write("Enter house details below and click **Predict Price**")

st.divider()

# Function to create numeric inputs
def num_input(label, value=0.0, min_val=0.0, step=1.0):
    return st.number_input(label, value=float(value), min_value=float(min_val), step=float(step))

# Input fields (generated dynamically from model features)
inputs = {}

for feature in feature_names:
    if feature == "bedrooms":
        inputs[feature] = num_input("Bedrooms", value=3, min_val=0)
    elif feature == "bathrooms":
        inputs[feature] = st.number_input("Bathrooms", value=2.0, min_value=0.0, step=0.5)
    elif feature == "floors":
        inputs[feature] = st.number_input("Floors", value=1.0, min_value=0.0, step=0.5)
    elif feature == "waterfront":
        inputs[feature] = st.selectbox("Waterfront", options=[0, 1])
    elif feature == "view":
        inputs[feature] = num_input("View (0–4 scale)", value=0, min_val=0)
    elif feature == "condition":
        inputs[feature] = num_input("Condition (1–5 scale)", value=3, min_val=1)
    elif feature == "yr_built":
        inputs[feature] = num_input("Year Built", value=1970, min_val=0)
    elif feature == "yr_renovated":
        inputs[feature] = num_input("Year Renovated (0 if never)", value=0, min_val=0)
    else:
        # For sqft-related and other numeric features
        inputs[feature] = num_input(feature.replace("_", " ").title(), value=0, min_val=0)

# Convert inputs to DataFrame
input_df = pd.DataFrame([[inputs[f] for f in feature_names]], columns=feature_names)

st.divider()

# Prediction button
if st.button("Predict Price ✅"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
    st.caption("This prediction is generated using a trained Random Forest regression model.")

st.divider()

# Footer note
st.caption("© House Price Prediction Prototype – Machine Learning Project")
