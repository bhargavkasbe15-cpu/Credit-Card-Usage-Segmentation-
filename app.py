import os
import pickle
import streamlit as st

# 1. Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the paths (Ensure these names match your GitHub files exactly)
model_path = os.path.join(BASE_DIR, "cluster_model (1).pkl")
scaler_path = os.path.join(BASE_DIR, "scaler (1).pkl")

# 3. Load the model and scaler
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model or Scaler files not found!")
    st.write("Files found in directory:", os.listdir(BASE_DIR))
