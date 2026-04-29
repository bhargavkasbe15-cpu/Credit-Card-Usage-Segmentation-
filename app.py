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
    # Change these lines in your script to match the actual filenames on GitHub
model_path = os.path.join(BASE_DIR, "cluster_model (1).pkl")
scaler_path = os.path.join(BASE_DIR, "scaler (1).pkl")import os
import pickle
import streamlit as st

# Get the exact directory where your app is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths (Ensure these names match your files on GitHub exactly!)
model_path = os.path.join(BASE_DIR, "cluster_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_assets():
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

if model is None or scaler is None:
    st.error("Model or Scaler files not found!")
    st.write("Debug Info - Files found in directory:", os.listdir(BASE_DIR))
    st.stop() # Stops the app from crashing further down
