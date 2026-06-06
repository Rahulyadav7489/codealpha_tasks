import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Scoring System")

st.write("Enter applicant details")

# Inputs
checking_account = st.number_input("Checking Account Status", min_value=0, value=1)

loan_duration = st.number_input("Loan Duration (Months)", min_value=1, value=24)

credit_history = st.number_input("Credit History", min_value=0, value=1)

loan_amount = st.number_input("Loan Amount", min_value=0, value=5000)

savings_account = st.number_input("Savings Account Status", min_value=0, value=1)

employment_duration = st.number_input("Employment Duration", min_value=0, value=2)

installment_rate = st.number_input(
    "Installment Rate Percent",
    min_value=1,
    max_value=4,
    value=2
)

personal_status = st.number_input("Personal Status Sex", min_value=0, value=2)

guarantors = st.number_input("Guarantors", min_value=0, value=1)

property_type = st.number_input("Property", min_value=0, value=2)

age = st.number_input("Age", min_value=18, max_value=100, value=35)

other_installments = st.number_input(
    "Other Installment Plans",
    min_value=0,
    value=1
)

housing = st.number_input("Housing", min_value=0, value=2)

existing_credits = st.number_input(
    "Number Of Existing Credits",
    min_value=1,
    value=1
)

job = st.number_input("Job", min_value=0, value=2)

telephone = st.number_input("Telephone", min_value=0, value=1)

foreign_worker = st.number_input("Foreign Worker", min_value=0, value=1)

if st.button("Predict Credit Score"):

    customer = pd.DataFrame([{
        "Checking_Account_Status": checking_account,
        "Loan_Duration_Months": loan_duration,
        "Credit_History": credit_history,
        "Loan_Amount": loan_amount,
        "Savings_Account_Status": savings_account,
        "Employment_Duration": employment_duration,
        "Installment_Rate_Percent": installment_rate,
        "Personal_Status_Sex": personal_status,
        "Guarantors": guarantors,
        "Property": property_type,
        "Age": age,
        "Other_Installment_Plans": other_installments,
        "Housing": housing,
        "Number_of_Existing_Credits": existing_credits,
        "Job": job,
        "Telephone": telephone,
        "Foreign_Worker": foreign_worker
    }])
    customer = scaler.transform(customer)
    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0, 1]

    credit_score = int(300 + probability * 600)

    if credit_score >= 800:
        risk = "Excellent"
    elif credit_score >= 700:
        risk = "Good"
    elif credit_score >= 600:
        risk = "Fair"
    elif credit_score >= 500:
        risk = "Poor"
    else:
        risk = "High Risk"

    st.success(f"Credit Score: {credit_score}")
    st.info(f"Risk Category: {risk}")

    st.write(
        f"Probability of Good Credit: {probability:.2%}"
    )

    st.write(
        "Prediction:",
        "Good Credit" if prediction == 1 else "Bad Credit"
    )