import streamlit as st
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("House Price Prediction")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathroom = st.number_input("Bathroom")
stories = st.number_input("Stories")
parking = st.number_input("Parking")

if st.button("Predict"):

    data = pd.DataFrame(
        [[area, bedrooms, bathroom, stories, parking]],
        columns=["Area", "Bedrooms", "Bathroom", "Stories", "Parking"]
    )

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
