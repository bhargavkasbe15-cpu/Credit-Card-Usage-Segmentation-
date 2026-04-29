import os
import pickle
import streamlit as st

# 1. Get the directory where THIS script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the full path to your files
model_path = os.path.join(BASE_DIR, "cluster_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# 3. Load the files with an error check
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error(f"ERROR: Files not found.")
    st.write("I am looking in this folder:", BASE_DIR)
    st.write("Files actually present in this folder:", os.listdir(BASE_DIR))
