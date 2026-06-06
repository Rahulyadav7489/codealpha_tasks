import pandas as pd
from split import scaler
from random_forest import model

low_risk_customer = {
    'Checking_Account_Status': 4,
    'Loan_Duration_Months': 12,
    'Credit_History': 4,
    'Loan_Amount': 2000,
    'Savings_Account_Status': 5,
    'Employment_Duration': 5,
    'Installment_Rate_Percent': 2,
    'Personal_Status_Sex': 2,
    'Guarantors': 1,
    'Property': 4,
    'Age': 45,
    'Other_Installment_Plans': 3,
    'Housing': 2,
    'Number_of_Existing_Credits': 1,
    'Job': 4,
    'Telephone': 2,
    'Foreign_Worker': 1
}

high_risk_customer= {
    'Checking_Account_Status': 1,
    'Loan_Duration_Months': 60,
    'Credit_History': 0,
    'Loan_Amount': 15000,
    'Savings_Account_Status': 1,
    'Employment_Duration': 1,
    'Installment_Rate_Percent': 4,
    'Personal_Status_Sex': 1,
    'Guarantors': 2,
    'Property': 1,
    'Age': 22,
    'Other_Installment_Plans': 1,
    'Housing': 1,
    'Number_of_Existing_Credits': 4,
    'Job': 1,
    'Telephone': 1,
    'Foreign_Worker': 1
}

def find_credit_score(new_customer):
    new_df = pd.DataFrame([new_customer])
    # print("Expected columns:")
    # print(list(scaler.feature_names_in_))

    # print("\nPrediction columns:")
    # print(list(new_df.columns))

    # print("\nMissing:")
    # print(set(scaler.feature_names_in_) - set(new_df.columns))

    # print("\nExtra:")
    # print(set(new_df.columns) - set(scaler.feature_names_in_))
    new_df_scaled = scaler.transform(new_df)

    pred = model.predict(new_df_scaled)[0]

    predict = "good" if pred == 1 else "bad"

    prob = model.predict_proba(new_df_scaled)
    good_prob = prob[:, 1]

    credit_score = int(300 + good_prob[0] * 600)

    return {
        "prediction": predict,
        "credit_score": credit_score
    }

predictions= find_credit_score(low_risk_customer)
print(f"low risk customer", predictions)

predictions= find_credit_score(high_risk_customer)
print(f"high risk customer", predictions)
