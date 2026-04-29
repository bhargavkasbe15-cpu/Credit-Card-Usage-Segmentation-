import os
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="wide")

# 1. Load Assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cluster_model (1).pkl")
scaler_path = os.path.join(BASE_DIR, "scaler (1).pkl")

@st.cache_resource
def load_assets():
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

st.title("💳 Credit Card Customer Segmentation")

if model is None:
    st.error("Model files not found! Ensure 'cluster_model (1).pkl' and 'scaler (1).pkl' are in your GitHub repo.")
else:
    st.write("Enter the customer usage details below to determine their segment.")
    
    # 2. Input Fields - Organized into 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        balance = st.number_input("Balance", value=0.0)
        balance_freq = st.slider("Balance Frequency (0-1)", 0.0, 1.0, 1.0)
        purchases = st.number_input("Total Purchases", value=0.0)
        oneoff_purch = st.number_input("One-Off Purchases", value=0.0)
        inst_purch = st.number_input("Installment Purchases", value=0.0)
        cash_adv = st.number_input("Cash Advance", value=0.0)

    with col2:
        purch_freq = st.slider("Purchases Frequency (0-1)", 0.0, 1.0, 0.5)
        oneoff_purch_freq = st.slider("One-Off Purchases Frequency (0-1)", 0.0, 1.0, 0.5)
        purch_inst_freq = st.slider("Purchases Inst. Frequency (0-1)", 0.0, 1.0, 0.5)
        cash_adv_freq = st.slider("Cash Advance Frequency (0-1)", 0.0, 1.0, 0.0)
        cash_adv_trx = st.number_input("Cash Advance Transactions", value=0)
        purch_trx = st.number_input("Purchases Transactions", value=0)

    with col3:
        credit_limit = st.number_input("Credit Limit", value=1000.0)
        payments = st.number_input("Total Payments", value=0.0)
        min_payments = st.number_input("Minimum Payments", value=0.0)
        prc_full_pay = st.slider("PRC Full Payment (0-1)", 0.0, 1.0, 0.0)
        tenure = st.number_input("Tenure (Months)", min_value=6, max_value=12, value=12)

    # 3. Prediction Button
    if st.button("Predict Customer Segment", key="predict_btn"):
        # Create feature array in the EXACT order your model was trained on
        features = np.array([[
            balance, balance_freq, purchases, oneoff_purch, inst_purch, 
            cash_adv, purch_freq, oneoff_purch_freq, purch_inst_freq, 
            cash_adv_freq, cash_adv_trx, purch_trx, credit_limit, 
            payments, min_payments, prc_full_pay, tenure
        ]])

        try:
            # Scale and Predict
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            
            st.success(f"### Result: This customer belongs to **Cluster {prediction[0]}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
