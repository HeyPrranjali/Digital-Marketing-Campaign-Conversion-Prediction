# for customer conversion app
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('xgb_model_top5_features.joblib')

# Set the title of the app
st.title('Digital Marketing Campaign Prediction')

# Define the input fields
st.sidebar.header('Input Features')
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
income = st.sidebar.number_input('Income', min_value=1000, max_value=1000000, value=50000)
ad_spend = st.sidebar.number_input('Ad Spend', min_value=0, max_value=10000, value=1000)
click_through_rate = st.sidebar.number_input('Click Through Rate', min_value=0.0, max_value=1.0, value=0.05)
conversion_rate = st.sidebar.number_input('Conversion Rate', min_value=0.0, max_value=1.0, value=0.02)

# Collect input features into a numpy array
features = np.array([age, income, ad_spend, click_through_rate, conversion_rate]).reshape(1, -1)

# Button to make predictions
if st.sidebar.button('Predict'):
    # Make predictions using the model
    prediction = model.predict(features)

    # Show the prediction result
    if prediction == 0:
        st.write('Prediction: Class 0 (Not Converted)')
    else:
        st.write('Prediction: Class 1 (Converted)')
    
    # Show prediction probabilities (if needed)
    prediction_proba = model.predict_proba(features)
    st.write(f"Prediction Probability: Class 0: {prediction_proba[0][0]:.2f}, Class 1: {prediction_proba[0][1]:.2f}")

# Display a message
st.markdown("""
    This application allows you to input various features related to marketing campaigns, 
    and it will predict whether a user will convert or not based on the trained XGBoost model.
""")

