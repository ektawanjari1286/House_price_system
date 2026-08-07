import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

FEATURES = ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Parking']


@st.cache_resource
def load_artifacts():
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


model, scaler = load_artifacts()

st.title("🏠 House Price Prediction")
st.write("Enter the details of the house below to get an estimated price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=1500, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)

    with col2:
        stories = st.number_input("Stories", min_value=1, max_value=5, value=1, step=1)
        parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1, step=1)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [[area, bedrooms, bathrooms, stories, parking]],
        columns=FEATURES
    )

    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)[0]

    st.success(f"### Estimated Price: ₹{prediction:,.2f}")

    with st.expander("Input summary"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Model: Linear Regression | Trained on house price dataset")
