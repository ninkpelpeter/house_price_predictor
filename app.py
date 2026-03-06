import joblib
import pandas as pd
import numpy as np
import streamlit as st


# Load trained model
model = joblib.load("house_price_model.pkl")

# App title
st.title("House Price Prediction App")
st.write("Enter house details below to predict the price.")

# User inputs
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
size = st.number_input("Size of House (m²)", min_value=50, max_value=1000, value=150)
age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=10)
location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])

# Convert location into the same format used during training
location_suburban = 1 if location == "Suburban" else 0
location_urban = 1 if location == "Urban" else 0

# Predict button
if st.button("Predict Price"):
    features = np.array([[rooms, size, age, location_suburban, location_urban]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated House Price: {prediction:,.2f}")
  

