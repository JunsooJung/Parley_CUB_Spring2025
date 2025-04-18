import pandas as pd
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
nn_model = load_model("nn_504_model.keras")
xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")
meta_model = joblib.load("meta_model.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("feature_names.pkl")

st.title("504 Loan Eligibility Type Prediction")

# Define your input fields
base_features = ['Business Structure',
 'Country',
 'NAICS',
 'Business Ownership (1)',
 'Business Ownership (2)',
 'Business Ownership (3)',
 'Business Ownership (4)',
 'Business Ownership (5)',
 'Personal Credit Score',
 'Business Credit Score',
 'Annual Revenue (2 years ago)',
 'Annual Revenue (1 year ago)',
 'Annual Revenue (latest year)',
 'Net Profit Margin',
 'Business Debt (2 years ago)',
 'Business Debt (1 year ago)',
 'Business Debt (latest year)',
 'NOI (2 years ago)',
 'NOI (1 year ago)',
 'NOI (latest year)',
 'DSCR (2 years ago)',
 'DSCR (1 year ago)',
 'DSCR (latest year)',
 'Industry Experience',
 'Managerial Experience',
 'Years in Business',
 'Collateral Availability',
 'Loan Amount',
 'Fast Approval',
 'For Profit',
 'Acquisition Request',
 'Working Capital',
 'Business Expansion',
 'Equipment Purchase or Leasing',
 'Inventory Purchase',
 'Real Estate Acquisition or Improvement',
 'Business Acquisition or Buyout',
 'Refinancing Existing Debt',
 'Emergency Funds',
 'Franchise Financing',
 'Contract Financing',
 'Licensing or Permits',
 'Line of Credit Establishment']



input_dict = {}
for feat in base_features:  # base features: original column names before get_dummies()
    input_dict[feat] = st.text_input(f"Enter value for {feat}")


if st.button("Check Eligibility"):
    input_df = pd.DataFrame([input_dict])

    # You may need to apply the same scaling as during training
    # For now, assuming it's already scaled or not needed

    # NN needs scaled input
    # If scaler is needed, load and apply:
    # input_array_scaled = scaler.transform(input_array)

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    model_features = joblib.load("feature_names.pkl")

    input_encoded_aligned = pd.DataFrame(columns=model_features)
    input_encoded_aligned = pd.concat([input_encoded_aligned, input_encoded], ignore_index=True)
    input_encoded_aligned = input_encoded_aligned.fillna(0)
    
    scaler = joblib.load("scaler.pkl")
    numeric_cols = [col for col in input_encoded_aligned.columns if col in scaler.feature_names_in_]
    input_encoded_aligned[numeric_cols] = scaler.transform(input_encoded_aligned[numeric_cols])
    input_encoded_aligned = input_encoded_aligned.astype('float32')

    pred_nn = nn_model.predict(input_encoded_aligned).flatten()[0]
    pred_xgb = xgb_model.predict_proba(input_encoded_aligned)[:, 1][0]
    pred_lgb = lgb_model.predict_proba(input_encoded_aligned)[:, 1][0]

    # Meta-prediction
    meta_input = np.array([[pred_nn, pred_xgb, pred_lgb]])
    final_pred = meta_model.predict(meta_input)[0]

    if (int(final_pred) == 0):
        st.success(f"Eligibility: Not Eligible for 504.")
    else:
        st.success(f"Eligibility: Eligible for 504.")
        

    # st.success(f"Eligibility: {int(final_pred)}")