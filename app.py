import os
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Credit Card Segmentation", layout="wide")

# 1. File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cluster_model (1).pkl")
scaler_path = os.path.join(BASE_DIR, "scaler (1).pkl")

# 2. Load Assets
@st.cache_resource
def load_assets():
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

st.title("💳 Credit Card Customer Segmentation")

if model is None:
    st.error("Model files not found. Please check your GitHub repository.")
else:
    st.success("Model loaded! Enter details below:")

    # 3. Input Fields (Add all the columns your model needs here)
    col1, col2 = st.columns(2)
    
    with col1:
        balance = st.number_input("Balance", value=0.0)
        purchases = st.number_input("Total Purchases", value=0.0)
        # Add more here...

    with col2:
        cash_advance = st.number_input("Cash Advance", value=0.0)
        credit_limit = st.number_input("Credit Limit", value=0.0)
        # Add more here...

    # 4. The Button (With a unique key to prevent your error)
    if st.button("Predict Segment", key="final_prediction_btn"):
        # Make sure the list below contains ALL features in the correct order!
        features = np.array([[balance, purchases, cash_advance, credit_limit]])
        
        # Scale and Predict
        scaled_features = scaler.transform(features)
        cluster = model.predict(scaled_features)
        
        st.metric(label="Assigned Cluster", value=f"Cluster {cluster[0]}")
