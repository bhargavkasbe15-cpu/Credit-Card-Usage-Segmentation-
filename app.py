import os
import pickle
import streamlit as st

# 1. Setup Page Config
st.set_page_config(page_title="Credit Card Segmentation", layout="centered")

# 2. Define the base directory and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cluster_model (1).pkl")
scaler_path = os.path.join(BASE_DIR, "scaler (1).pkl")

# 3. Load the model and scaler
@st.cache_resource
def load_models():
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_models()

# 4. User Interface
st.title("💳 Credit Card Customer Segmentation")
st.write("Enter customer details to predict their usage category.")

if model is None or scaler is None:
    st.error("Model files not found! Please check your GitHub repository.")
    st.write("Folder path:", BASE_DIR)
    st.write("Files seen by app:", os.listdir(BASE_DIR))
else:
    st.success("Model ready for prediction!")

    # --- ADD YOUR INPUT FIELDS BELOW ---
    # Example:
    balance = st.number_input("Customer Balance", min_value=0.0)
    purchases = st.number_input("Total Purchases", min_value=0.0)
    
    if st.button("Predict Segment"):
        # This is where your prediction logic goes
        # input_data = scaler.transform([[balance, purchases]])
        # prediction = model.predict(input_data)
        st.info("Prediction logic should be placed here based on your model's features.")# 1. Collect all inputs into a list (Must be in the SAME order as your training data)
    # Example: [Balance, Purchases, OneOff_Purchases, Installment_Purchases, etc.]
    feature_list = [balance, purchases] 

    if st.button("Predict Segment"):
        # 2. Reshape the data for the model
        import numpy as np
        input_data = np.array(feature_list).reshape(1, -1)
        
        # 3. Scale the data (very important for Clustering!)
        scaled_data = scaler.transform(input_data)
        
        # 4. Make the prediction
        prediction = model.predict(scaled_data)
        
        # 5. Show the result
        st.header(f"The Customer belongs to Cluster: {prediction[0]}")
