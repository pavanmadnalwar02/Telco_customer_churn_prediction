# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 01:12:28 2025

@author: pavan
"""




import streamlit as st
import joblib
import numpy as np

# Load the trained churn prediction model
model = joblib.load(r"C:\Users\pavan\Documents\Data science\Telco Customer Churn\Telco_Customer_Churn_model.joblib")

# Title and description
st.title("Customer Churn Prediction")
st.write("Provide the details below to predict if the customer will churn or not.")

# Input fields
gender = st.selectbox("Gender", options=["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", options=["Yes", "No"])
partner = st.selectbox("Partner", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, step=1)
phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", options=["Yes", "No"])
online_backup = st.selectbox("Online Backup", options=["Yes", "No"])
tech_support = st.selectbox("Tech Support", options=["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", options=["Yes", "No"])
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1)

# Map categorical inputs to numerical values
gender_mapping = {"Male": 0, "Female": 1}
senior_citizen_mapping = {"Yes": 1, "No": 0}
partner_mapping = {"Yes": 1, "No": 0}
dependents_mapping = {"Yes": 1, "No": 0}
phone_service_mapping = {"Yes": 1, "No": 0}
multiple_lines_mapping = {"Yes": 1, "No": 0}
internet_service_mapping = {"DSL": 0, "Fiber optic": 1, "No": 2}
online_security_mapping = {"Yes": 1, "No": 0}
online_backup_mapping = {"Yes": 1, "No": 0}
device_protection_mapping = {"Yes": 1, "No": 0}
tech_support_mapping = {"Yes": 1, "No": 0}
streaming_tv_mapping = {"Yes": 1, "No": 0}
streaming_movies_mapping = {"Yes": 1, "No": 0}
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
paperless_billing_mapping = {"Yes": 1, "No": 0}
payment_method_mapping = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}

# Convert categorical values to numerical
gender_num = gender_mapping[gender]
senior_citizen_num = senior_citizen_mapping[senior_citizen]
partner_num = partner_mapping[partner]
dependents_num = dependents_mapping[dependents]
phone_service_num = phone_service_mapping[phone_service]
multiple_lines_num = multiple_lines_mapping[multiple_lines]
internet_service_num = internet_service_mapping[internet_service]
online_security_num = online_security_mapping[online_security]
online_backup_num = online_backup_mapping[online_backup]
tech_support_num = tech_support_mapping[tech_support]
streaming_tv_num = streaming_tv_mapping[streaming_tv]

contract_num = contract_mapping[contract]
paperless_billing_num = paperless_billing_mapping[paperless_billing]
payment_method_num = payment_method_mapping[payment_method]

# Predict churn
if st.button("Predict Churn"):
    # Combine inputs into a single array
    features = np.array([
        [gender_num, senior_citizen_num, partner_num, dependents_num, tenure, phone_service_num,
         multiple_lines_num, internet_service_num, online_security_num, online_backup_num,
          tech_support_num, streaming_tv_num, contract_num,
         paperless_billing_num, payment_method_num, monthly_charges, total_charges]
    ])
    
    # Predict using the model
    churn_prediction = model.predict(features)
    
    # Display the prediction
    if churn_prediction[0] == 1:
        st.success("Yes, the customer is likely to churn.")
    else:
        st.success("No,the customer is likely to stay.")

# Footer
st.write("This app uses a machine learning model to predict customer churn based on input features.")
