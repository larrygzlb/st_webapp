import streamlit as st
import joblib
import pandas as pd
import altair as alt
# 要删除internet access
# 要增加family income, school type, previous score, tutoring sessions

st.set_page_config(layout="wide")
# 加载模型和scaler
loaded_model = joblib.load("linear_regression_model_0418.pkl")
scaler = joblib.load("scaler_0418.pkl")

# 初始化预测历史
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
    st.session_state.input_index = []

st.title("📈 Exam Score Prediction")
st.write("Enter the values for the following variables:")

# 创建左右两列布局：左侧放表单，右侧放趋势图
cols = st.columns([2, 3])

with cols[0]:

    with st.form("input_form"):
    
        # 先写第一个bubble
        # 原来的文字改成 Markdown + HTML，使用 h3
        st.image("checklist.png", width=50)
        st.markdown(
    "<div style='font-size:22px; font-weight:bold; color:#006400;'> Controllable Factors:</div>",
    unsafe_allow_html=True
)
        # 第一行：2个滑块
        slider_cols = st.columns(2)
        with slider_cols[0]:
            hours_studied = st.slider("**Hours spent on studying/week**", 0, 50)
        with slider_cols[1]:
            attendance = st.slider("**Percentage of classes attended %**", min_value=0, max_value=100, step=1)

        # 第二行：2个滑块
        slider_cols = st.columns(2)
        with slider_cols[0]:
            sleep_hours = st.slider("**Average hours of sleep/night**", 0, 12)
        with slider_cols[1]:
            physical_activity = st.slider("**Average hours of physical activity/week**", 0, 24)

        
        #第三行：2个选择框
        radio_row1= st.columns(3)
        with radio_row1[0]:
            # st.write("**Motivation Level**")
            motivation_level_option = st.radio("**Motivation Level**", ["Low", "Medium", "High"])
        with radio_row1[1]:
            # st.write("**Access to Resources**")
            access_to_resources_option = st.radio("**Access to Resources**", ["Low", "Medium", "High"], key="3")
        with radio_row1[2]:
            st.write("**Peer Influence**")
            peer_influence_option = st.radio("", ["Negative", "Neutral", "Positive"])
       

        st.markdown("</div>", unsafe_allow_html=True)  # Close Internal Factors box

        # 第二个bubble
        st.image("social-media.png", width=50)
        st.markdown(
    "<div style='font-size:22px; font-weight:bold; color:#003366;'> Uncontrollable Factors:</div>",
    unsafe_allow_html=True
)
        
        # 第一个小节：school factors
        st.markdown("<div style='font-size:18px; color:#3399FF;'>🎓 Self Factor</span>", unsafe_allow_html=True)
        
        # 第一行：1个滑块
        radio_row1 = st.columns(1)
        with radio_row1[0]:
            previous_score = st.slider("**Previous score**", min_value=0, max_value=100, step=1, key="1")

        # 第二个小节：school factors
        st.markdown("<div style='font-size:18px; color:#3399FF;'>🏫 School Factors</span>", unsafe_allow_html=True)
        
        # 第一行：3个选项框）
        radio_row2 = st.columns(2)
        with radio_row2[0]:
            st.write("**School type**")
            peer_influence_option = st.radio("", ["Public", "Private"])
        with radio_row2[1]:
            st.write("**Teacher Quality**")
            teacher_quality_option = st.radio("", ["Low", "Medium", "High"], key="2")
        # with radio_row2[2]:
        #     st.write("**Peer Influence**")
        #     peer_influence_option = st.radio("", ["Negative", "Neutral", "Positive"])

        # 第二行：1个滑块
        slider_cols = st.columns(1)
        with slider_cols[0]:
            tutoring_sessions = st.slider("**Tutoring sessions**", min_value=0, max_value=8, step=1)
            
        
        # 第三个小节：Family factors
        st.markdown("<div style='font-size:18px; color:#3399FF;'>👨‍👩‍👧‍👦 Family Factors</span>", unsafe_allow_html=True)

        # 第一行：3个选项框
        radio_row3 = st.columns(3)
        with radio_row3[0]:
            st.write("**Parental Involvement**")
            parental_involvement_option = st.radio("", ["Low", "Medium", "High"], key="4")
        with radio_row3[1]:
            st.write("**Extracurricular activities**")
            extracurricular_activities_option = st.radio("", ["Yes", "No"], key="5")
        with radio_row3[2]:
            st.write("**Distance from Home**")
            distance_from_home_option = st.radio("", ["Near", "Moderate", "Far"])

        # 第二行：2个选项框
        radio_row4 = st.columns(2)
        with radio_row4[0]:
            st.write("**Parental education level**")
            parental_education_level_option = st.radio("", ["High school", "College", "Postgraduate"])
        with radio_row4[1]:
            st.write("**Family income**")
            family_income_option = st.radio("", ["Low", "Medium", "High"], key="6")

        st.markdown("</div>", unsafe_allow_html=True) 

        submitted = st.form_submit_button("🔍 Predict")
        # st.markdown("</div>", unsafe_allow_html=True)  # Close External Factors box

        # 在表单最后添加一行，用于放置提交按钮在右侧
        # radio_row4 = st.columns(3)
        # button_cols = st.columns([7, 3])
        # with button_cols[1]:
        #     submitted = st.form_submit_button("🔍 Predict")
        # with button_cols[1]:
        # with radio_row4[1]:
        #     reset_button = st.button("Reset Prediction History")
        # with radio_row4[2]:
        #     submitted = st.form_submit_button("🔍 Predict")
        # submitted = st.form_submit_button("🔍 Predict")

# 当表单提交后，处理预测逻辑
if submitted:
    peer_mapping = {"Negative": 1, "Neutral": 2, "Positive": 3}
    level_mapping = {"Low": 1, "Medium": 2, "High": 3}
    boolean_mapping = {"No": 0, "Yes": 1}
    distance_mapping = {"Near": 1, "Moderate": 2, "Far": 3} 
    education_mapping = {"High school": 1, "College": 2, "Postgraduate": 3}

    input_data = {
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Parental_Involvement": [level_mapping[parental_involvement_option]],
        "Access_to_Resources": [level_mapping[access_to_resources_option]],
        "Extracurricular_Activities": [boolean_mapping[extracurricular_activities_option]],
        "Sleep_Hours": [sleep_hours],
        # previou_score
        "Previous_Scores": [previous_score],

        "Motivation_Level": [level_mapping[motivation_level_option]],
        # tutorial session
        "Tutoring_Sessions": [tutoring_sessions],

        # family income
        "Family_Income": [level_mapping[family_income_option]],


        "Teacher_Quality": [level_mapping[teacher_quality_option]],
        # school type
        "School_Type": [peer_mapping[peer_influence_option]],

        "Peer_Influence": [peer_mapping[peer_influence_option]],
        # pyhsical activity
        "Physical_Activity": [physical_activity],

        # parent educaiton level
        "Parental_Education_Level": [education_mapping[parental_education_level_option]],

        "Distance_from_Home": [distance_mapping[distance_from_home_option]],
    }
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    prediction = loaded_model.predict(input_scaled)[0]
    if prediction > 100:
        prediction = 100
    

    st.session_state.prediction_history.append(prediction)
    st.session_state.input_index.append(len(st.session_state.input_index) + 1)

    st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")

with cols[1]:
    chart_placeholder = st.empty()

# 在占位符内绘制趋势图
with chart_placeholder.container():
    st.subheader("📊 Score predictions")
    chart_data = pd.DataFrame(
        {
            "Prediction": st.session_state.input_index,
            "Predicted Score": st.session_state.prediction_history,
        }
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Prediction:Q", axis=alt.Axis(format="d", title="Prediction #")),
            y=alt.Y("Predicted Score", title="Predicted Score"),
        )
    )
    st.altair_chart(chart, use_container_width=True)
    reset_button = st.button("Reset Prediction History")

# 重置按钮


if reset_button:
    st.session_state.prediction_history = []
    st.session_state.input_index = []
    st.info("Prediction history reset.")
    # 更新同一个占位符，清空趋势图（或显示新的空数据图表）
    chart_placeholder.empty()
    with chart_placeholder.container():
        st.subheader("📊 Trend")
        chart_data = pd.DataFrame(
            {
                "Prediction": st.session_state.input_index,
                "Predicted Score": st.session_state.prediction_history,
            }
        )
        chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("Prediction:Q", axis=alt.Axis(format="d", title="Prediction #")),
                y=alt.Y("Predicted Score", title="Predicted Score"),
            )
        )
        st.altair_chart(chart, use_container_width=True)
        # reset_button = st.button("Reset Prediction History")
