import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
loaded_model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize session state for prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
    st.session_state.input_index = []

# App title
st.title("📈 Student Performance Prediction")
st.write("Enter the values for the following variables:")


pi_mapping = {"Low": 1, "Medium": 2, "High": 3}
boolean_map = {"No": 0, "Yes": 1}
# Layout: Form on the left, Chart on the right
layout_cols = st.columns([2, 1])  # 2:1 width ratio

with layout_cols[0]:  # Left column: Form
    with st.form("input_form"):
        cols = st.columns(3)

        with cols[0]:
            hours_studied = st.slider("Hours_Studied", 0, 50, key="Hours_Studied")
            parental_involvement_option = st.radio(
                "Parental Involvement", ["Low", "Medium", "High"], key="Parental_Involvement"
            )
            extracurricular_activities_option = st.radio(
                "Extracurricular_Activities", ["Yes", "No"], key="Extracurricular_Activities"
            )
            motivation_level_option = st.radio(
                "Motivation_Level", ["Low", "Medium", "High"], key="Motivation_Level"
            )
            teacher_quality = st.radio("Teacher_Quality", ["Low", "Medium", "High"], key="Teacher_Quality")

        with cols[1]:
            attendance = st.slider("Attendance", 0, 100, key="Attendance")
            access_to_resources_option = st.radio(
                "Access_to_Resources", ["Low", "Medium", "High"], key="Access_to_Resources"
            )
            sleep_hours = st.slider("Sleep_Hours", 0, 12, key="Sleep_Hours")
            internet_access = st.radio("Internet_Access", ["Yes", "No"], key="Internet_Access")
            peer_influence_option = st.radio(
                "Peer_Influence", ["Low", "Medium", "High"], key="Peer_Influence"
            )

        with cols[2]:
            learning_disabilities = st.radio(
                "Learning_Disabilities", ["Yes", "No"], key="Learning_Disabilities"
            )
            distance_from_home = st.radio(
                "Distance_from_Home", ["Low", "Medium", "High"], key="Distance_from_Home"
            )
            physical_activity = st.slider("Physical_Activity", 0, 24, key="Physical_Activity")

        submitted = st.form_submit_button("Predict")
        # map the radio button value to numerical value
        level_mapping = {"Low": 1, "Medium": 2, "High": 3}
        boolean_mapping = {"No": 0, "Yes": 1}

        parental_involvement = level_mapping[parental_involvement_option]
        access_to_resources = level_mapping[access_to_resources_option]
        motivation_level = level_mapping[motivation_level_option]
        teacher_quality = level_mapping[teacher_quality]
        peer_influence = level_mapping[peer_influence_option]
        distance_from_home = level_mapping[distance_from_home]
        internet_access = boolean_mapping[internet_access]
        learning_disabilities = boolean_mapping[learning_disabilities]

        extracurricular_activities = boolean_mapping[extracurricular_activities_option]
        if submitted:
            input_data = {
                "Hours_Studied": [hours_studied],
                "Attendance": [attendance],
                "Parental_Involvement": [parental_involvement],
                "Access_to_Resources": [access_to_resources],
                "Extracurricular_Activities": [extracurricular_activities],
                "Sleep_Hours": [sleep_hours],
                "Motivation_Level": [motivation_level],
                "Internet_Access": [internet_access],
                "Teacher_Quality": [teacher_quality],
                "Peer_Influence": [peer_influence],
                "Learning_Disabilities": [learning_disabilities],
                "Distance_from_Home": [distance_from_home],
                "Physical_Activity": [physical_activity],
            }

            input_df = pd.DataFrame(input_data)
            input_scaled = scaler.transform(input_df)
            prediction = loaded_model.predict(input_scaled)

            st.session_state.prediction_history.append(prediction[0])
            st.session_state.input_index.append(len(st.session_state.input_index) + 1)

            st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")

with layout_cols[1]:  # Right column: Chart
    if st.session_state.prediction_history:
        st.subheader("📊 Trend")
        chart_data = pd.DataFrame(
            {
                "Prediction #": st.session_state.input_index,
                "Predicted Score": st.session_state.prediction_history,
            }
        ).set_index("Prediction #")
        st.line_chart(chart_data)

    if st.button("Reset Prediction History"):
        st.session_state.prediction_history = []
        st.session_state.input_index = []
        st.info("Prediction history reset.")
