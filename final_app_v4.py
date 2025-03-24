import streamlit as st
import joblib
import pandas as pd

# 加载模型和scaler
loaded_model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# 初始化预测历史
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
    st.session_state.input_index = []

st.title("📈 Student Performance Prediction")
st.write("Enter the values for the following variables:")

# 创建左右两列布局：左侧放表单，右侧放趋势图
cols = st.columns([2, 1])

with cols[0]:
    with st.form("input_form"):
        # 第一行：4个滑块
        slider_cols = st.columns(4)
        with slider_cols[0]:
            hours_studied = st.slider("Hours Studied", 0, 50)
        with slider_cols[1]:
            attendance = st.slider("Attendance %", 0, 100)
        with slider_cols[2]:
            sleep_hours = st.slider("Sleep Hours", 0, 12)
        with slider_cols[3]:
            physical_activity = st.slider("Physical Activity", 0, 24)

        # 第二部分：Radios（第一行）
        radio_row1 = st.columns(3)
        with radio_row1[0]:
            parental_involvement_option = st.radio("Parental Involvement", ["Low", "Medium", "High"])
        with radio_row1[1]:
            extracurricular_activities_option = st.radio("Extracurricular Activities", ["Yes", "No"])
        with radio_row1[2]:
            motivation_level_option = st.radio("Motivation Level", ["Low", "Medium", "High"])

        # 第三部分：Radios（第二行）
        radio_row2 = st.columns(3)
        with radio_row2[0]:
            internet_access_option = st.radio("Internet Access", ["Yes", "No"])
        with radio_row2[1]:
            teacher_quality_option = st.radio("Teacher Quality", ["Low", "Medium", "High"])
        with radio_row2[2]:
            access_to_resources_option = st.radio("Access to Resources", ["Low", "Medium", "High"])

        # 第四部分：Radios（第三行）
        radio_row3 = st.columns(3)
        with radio_row3[0]:
            peer_influence_option = st.radio("Peer Influence", ["Low", "Medium", "High"])
        with radio_row3[1]:
            learning_disabilities_option = st.radio("Learning Disabilities", ["Yes", "No"])
        with radio_row3[2]:
            distance_from_home_option = st.radio("Distance from Home", ["Low", "Medium", "High"])

        # 表单内的提交按钮
        submitted = st.form_submit_button("🔍 Predict")

# 当表单提交后，处理预测逻辑
if submitted:
    level_mapping = {"Low": 1, "Medium": 2, "High": 3}
    boolean_mapping = {"No": 0, "Yes": 1}

    input_data = {
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Parental_Involvement": [level_mapping[parental_involvement_option]],
        "Access_to_Resources": [level_mapping[access_to_resources_option]],
        "Extracurricular_Activities": [boolean_mapping[extracurricular_activities_option]],
        "Sleep_Hours": [sleep_hours],
        "Motivation_Level": [level_mapping[motivation_level_option]],
        "Internet_Access": [boolean_mapping[internet_access_option]],
        "Teacher_Quality": [level_mapping[teacher_quality_option]],
        "Peer_Influence": [level_mapping[peer_influence_option]],
        "Learning_Disabilities": [boolean_mapping[learning_disabilities_option]],
        "Distance_from_Home": [level_mapping[distance_from_home_option]],
        "Physical_Activity": [physical_activity],
    }
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    prediction = loaded_model.predict(input_scaled)

    st.session_state.prediction_history.append(prediction[0])
    st.session_state.input_index.append(len(st.session_state.input_index) + 1)

    st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")

with cols[1]:
    st.subheader("📊 Trend")
    if st.session_state.prediction_history:
        chart_data = pd.DataFrame(
            {
                "Prediction #": st.session_state.input_index,
                "Predicted Score": st.session_state.prediction_history,
            }
        ).set_index("Prediction #")
        st.line_chart(chart_data)
