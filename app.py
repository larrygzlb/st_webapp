import streamlit as st


def main():
    st.title("滑动模块值并计算总和")

    # 创建三个滑块
    value1 = st.slider("模块1值", 0, 100, 50)
    value2 = st.slider("模块2值", 0, 100, 50)
    value3 = st.slider("模块3值", 0, 100, 50)

    # 计算总和
    total = value1 + value2 + value3

    # 显示总和
    st.write(f"### 总和: {total}")


if __name__ == "__main__":
    main()
