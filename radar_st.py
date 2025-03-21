import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="交互式雷达预测", layout="wide")

hide_menu_style = """
    <style>
        /* 隐藏 Fork 按钮（如果存在） */
        [data-testid="stActionButtonIcon-fork"] {
            display: none !important;
        }
        /* 隐藏右上角的菜单（三个点） */
        header [data-testid="stToolbar"] {display: none !important;}

        /* 隐藏 Streamlit 默认的 "Made with Streamlit" */
        footer {visibility: hidden;}

        /* 隐藏 “Deploy” 按钮（如果有的话） */
        .stDeployButton {display: none !important;}
        
        /* 调整滑块的整体宽度 */
        div[data-testid="stSlider"] {
            width: 30% !important; 
        }
    </style>
    """

st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("🎯 人物属性 & 预测模型 联动展示")

# ---- 1. 属性输入（用 slider 控制） ----
st.sidebar.header("🔧 调整人物属性")
attributes = {
    "Strength": st.sidebar.slider("Strength", 0, 100, 80),
    "Agility": st.sidebar.slider("Agility", 0, 100, 65),
    "Intelligence": st.sidebar.slider("Intelligence", 0, 100, 90),
    "Charisma": st.sidebar.slider("Charisma", 0, 100, 70),
    "Endurance": st.sidebar.slider("Endurance", 0, 100, 85),
}

labels = list(attributes.keys())
values = list(attributes.values())
values.append(values[0])  # 闭合雷达图
labels.append(labels[0])

# ---- 2. 模拟预测模型 ----
weights = {"Strength": 0.3, "Agility": 0.2, "Intelligence": 0.25, "Charisma": 0.15, "Endurance": 0.1}
# 计算预测分数
score = sum([attributes[k] * weights[k] for k in attributes])
score = round(score, 2)

# ---- 3. 展示雷达图 ----
fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=values, theta=labels, fill="toself", name="当前属性", line=dict(color="royalblue")
        )
    ]
)
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)

# ---- 4. 展示结果 ----
col1, col2 = st.columns([2, 1])
col1.plotly_chart(fig, use_container_width=True)
col2.metric("🎯 模型预测分数", f"{score}")
