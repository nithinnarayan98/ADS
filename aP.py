import streamlit as st
import pickle
import pandas as pd
import os

# Load the trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    st.error("❌ Model file not found! Please check the file path.")
    st.stop()

# Streamlit UI
st.title("🎯 Advertising Sales Prediction App")
st.write("Enter ad spending details to predict sales.")

# Sidebar for user inputs
st.sidebar.header("📊 Input Advertising Budget")
tv_budget = st.sidebar.number_input("TV Budget ($)", min_value=0, value=100)
radio_budget = st.sidebar.number_input("Radio Budget ($)", min_value=0, value=50)
newspaper_budget = st.sidebar.number_input("Newspaper Budget ($)", min_value=0, value=25)

# Compute Total Budget (same as during model training)
total_budget = tv_budget + radio_budget + newspaper_budget

# Prepare input data for prediction
user_data = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget, total_budget]],
                         columns=["TV", "Radio", "Newspaper", "Total_Budget"])

# Predict Sales
if st.button("Predict"):
    try:
        prediction = model.predict(user_data)
        st.success(f"📈 Predicted Sales: {prediction[0]:,.2f} units")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
