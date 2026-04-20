import streamlit as st
import pandas as pd
import numpy as np
from model import detect_bias

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Bias Detection Tool", layout="wide")

st.title("AI Bias Detection & Mitigation Tool")

# -------------------------------
# DATA INPUT
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

df = None

# -------------------------------
# SAMPLE DATA (FIXED + STRONGER)
# -------------------------------
if st.button("Use Sample Data"):

    np.random.seed(42)
    size = 500   # 🔥 larger dataset

    gender = np.random.choice(["Male", "Female"], size=size)
    experience = np.random.randint(1, 10, size=size)

    # 🔴 STRONG BIAS (needed to show mitigation)
    hired = []
    for g, exp in zip(gender, experience):
        if g == "Male":
            hired.append("Yes" if (exp + np.random.randint(0,3)) > 5 else "No")
        else:
            hired.append("Yes" if (exp - np.random.randint(0,3)) > 6 else "No")
    df = pd.DataFrame({
        "gender": gender,
        "experience": experience,
        "hired": hired
    })

# -------------------------------
# USER UPLOAD
# -------------------------------
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if len(df) < 50:
        st.warning("Dataset too small → results may be unreliable")
        # ❌ DO NOT duplicate anymore

# -------------------------------
# MAIN
# -------------------------------
if df is not None:

    st.subheader("Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10))

    target = st.selectbox("Target Column", df.columns)
    sensitive = st.selectbox("Sensitive Column", df.columns)

    st.write("Shape:", df.shape)

    st.write("Class distribution:")
    st.write(df[target].value_counts())

    st.write("Group distribution:")
    st.write(df[sensitive].value_counts())

    threshold = st.slider("Bias Threshold", 0.0, 1.0, 0.2)

    if st.button("Run Analysis"):

        result = detect_bias(df, target, sensitive, threshold)

        # ✅ ADD THIS BLOCK HERE
        if result["before"]["bias"] < threshold:
            st.warning("Baseline model is already fair. Mitigation not required.")
            st.write("Bias Difference:", round(result["before"]["bias"] - result["after"]["bias"], 3))

        # -------------------------------
        # RESULTS
        # -------------------------------
        st.subheader("Before vs After")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔴 Before Model")
            st.metric("Accuracy", result["before"]["accuracy"])
            st.metric("Bias", result["before"]["bias"])
            st.write(result["before"]["result"])

        with col2:
            st.markdown("### 🟢 After Model")
            st.metric("Accuracy", result["after"]["accuracy"])
            st.metric("Bias", result["after"]["bias"])
            st.write(result["after"]["result"])

        # -------------------------------
        # GROUP COMPARISON TABLE
        # -------------------------------
        st.subheader("Group-wise Prediction Rates")

        col3, col4 = st.columns(2)

        with col3:
            st.write("Before")
            st.dataframe(result["before"]["group_rates"])

        with col4:
            st.write("After")
            st.dataframe(result["after"]["group_rates"])

        # -------------------------------
        # GRAPH
        # -------------------------------
        st.subheader("Bias Comparison")

        before = result["before"]["group_rates"]
        after = result["after"]["group_rates"]

        all_groups = sorted(set(before.index).union(set(after.index)))

        df_plot = pd.DataFrame({
            "Before": before.reindex(all_groups),
            "After": after.reindex(all_groups)
        })

        st.bar_chart(df_plot)

        # -------------------------------
        # INSIGHT
        # -------------------------------
        diff = result["before"]["bias"] - result["after"]["bias"]

        if diff > 0.05:
            st.success("Bias reduced successfully 🎯")
        elif diff > -0.05:
            st.info("Bias change is minimal — dataset may be too simple")
        else:
            st.warning("Bias increased — mitigation ineffective")