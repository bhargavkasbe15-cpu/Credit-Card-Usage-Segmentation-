import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Segment Predictor", layout="centered")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Ensure these files are in the same folder as app.py
    model = pickle.load(open('cluster_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    kmeans, scaler = load_models()
except FileNotFoundError:
    st.error("Model or Scaler files not found! Please ensure 'cluster_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# --- UI DESIGN ---
st.title("👥 Customer Segmentation Tool")
st.markdown("""
Predict which customer segment a user belongs to based on their **Balance** and **Purchase** history.
""")

st.sidebar.header("User Input Features")

def user_input_features():
    balance = st.sidebar.number_input("Balance ($)", min_value=0.0, value=1000.0)
    purchases = st.sidebar.number_input("Purchases ($)", min_value=0.0, value=500.0)
    
    # Note: Your original script used full scaled data for the final model.
    # If your model was trained on all features, you must provide inputs for all of them.
    # This example assumes a 2-feature input matching your main visualization.
    data = {'BALANCE': balance,
            'PURCHASES': purchases}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- PREDICTION ---
st.subheader("Customer Data")
st.write(input_df)

if st.button("Predict Cluster"):
    # 1. Scale the input using the loaded scaler
    # Note: The scaler expects the same number of features it was trained on.
    scaled_input = scaler.transform(input_df)
    
    # 2. Predict
    prediction = kmeans.predict(scaled_input)
    
    # 3. Display Result
    st.success(f"This customer belongs to **Cluster {prediction[0]}**")
    
    # Optional: Logic to describe the clusters based on your analysis
    if prediction[0] == 0:
        st.info("💡 **Insight:** This group typically represents Low Balance / Low Spenders.")
    elif prediction[0] == 1:
        st.info("💡 **Insight:** This group typically represents High Balance / High Spenders.")
    else:
        st.info("💡 **Insight:** This group represents Moderate Spenders.")

st.divider()
st.caption("Powered by Scikit-Learn and Streamlit")
import os
import pickle
import streamlit as st

# This code tells Python to look in the SAME folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "cluster_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Debugging: This will print the path to your Streamlit screen so you can see it
# st.write(f"Looking for model at: {model_path}")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
else:
    st.error(f"Files still not found! Make sure they are in: {BASE_DIR}")
    import os
import streamlit as st

st.write("Files in current folder:", os.listdir())
