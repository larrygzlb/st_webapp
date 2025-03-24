import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler from the files
loaded_model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize prediction history in session state # NEW
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
    st.session_state.input_index = []

# Define the range of values for each variable
variable_ranges = {
    "Hours_Studied": (0, 50),
    "Attendance": (0, 100),
    "Parental_Involvement": (1, 3),
    "Access_to_Resources": (1, 3),
    "Extracurricular_Activities": (0, 1),
    "Sleep_Hours": (0, 12),
    "Motivation_Level": (1, 3),
    "Internet_Access": (0, 1),
    "Teacher_Quality": (1, 3),
    "Peer_Influence": (1, 3),
    "Learning_Disabilities": (0, 1),
    "Distance_from_Home": (1, 3),
    "Physical_Activity": (0, 24),
}

# Streamlit app
st.title("Student Performance Prediction")
st.write("Enter the values for the following variables:")

# Input fields for each variable
input_data = {}
for variable, (min_val, max_val) in variable_ranges.items():
    if variable in [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Teacher_Quality",
        "Peer_Influence",
        "Distance_from_Home",
    ]:
        value = st.selectbox(variable, range(min_val, max_val + 1), key=variable)
    else:
        value = st.slider(variable, min_val, max_val, key=variable)
    input_data[variable] = [value]

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Predict button
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = loaded_model.predict(input_scaled)

    # Update history # NEW
    st.session_state.prediction_history.append(prediction[0])
    st.session_state.input_index.append(len(st.session_state.input_index) + 1)

    st.write(f"Predicted Exam Score: {prediction[0]:.2f}")

# Line chart for prediction history # NEW
if st.session_state.prediction_history:
    chart_data = pd.DataFrame(
        {
            "Prediction Index": st.session_state.input_index,
            "Predicted Score": st.session_state.prediction_history,
        }
    ).set_index("Prediction Index")

    st.line_chart(chart_data)
