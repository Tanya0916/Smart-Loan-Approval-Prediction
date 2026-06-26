import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Loan Approval Prediction",
    page_icon="",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color:#f8f9fa;
}
.stButton>button {
    width:100%;
    background:#2563eb;
    color:white;
    border-radius:8px;
}
.metric-card{
    background:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 2px 8px rgba(0,0,0,.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_prediction.csv")

df = load_data()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Smart Loan Prediction")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "EDA",
        "Prediction",
        "Model Accuracy",
        "About"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "Home":

    st.title(" Smart Loan Approval Prediction")

    st.write("""
    Welcome to the Loan Approval Prediction System.

    This application predicts whether a loan application is likely
    to be approved using Machine Learning.

    The model has been trained on historical loan data using
    Scikit-Learn.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric(
            "Approved Loans",
            int((df["Loan_Status"] == "Y").sum())
        )

    with col3:
        st.metric(
            "Rejected Loans",
            int((df["Loan_Status"] == "N").sum())
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

# -----------------------------
# DATASET PAGE
# -----------------------------
elif menu == "Dataset":

    st.title(" Dataset Overview")

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

    st.subheader("First 10 Rows")

    st.dataframe(df.head(10))
    # -----------------------------
# EDA PAGE
# -----------------------------
elif menu == "EDA":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Loan Status Distribution")

    fig1 = px.pie(
        df,
        names="Loan_Status",
        title="Loan Approval Distribution",
        color_discrete_sequence=["#ef4444", "#22c55e"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Applicant Income Distribution")

    fig2 = px.histogram(
        df,
        x="ApplicantIncome",
        color="Loan_Status",
        barmode="overlay"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Credit History vs Loan Status")

    fig3 = px.histogram(
        df,
        x="Credit_History",
        color="Loan_Status",
        barmode="group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Property Area Impact")

    fig4 = px.histogram(
        df,
        x="Property_Area",
        color="Loan_Status",
        barmode="group"
    )
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
if st.button("Predict Loan Status"):
    
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # 🔥 IMPORTANT: enforce correct column order
    input_data = input_data[
        [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area"
        ]
    ]

    # Handle missing numeric values safely
    input_data["LoanAmount"] = pd.to_numeric(input_data["LoanAmount"], errors="coerce")
    input_data["Loan_Amount_Term"] = pd.to_numeric(input_data["Loan_Amount_Term"], errors="coerce")

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")

    st.write(f"Approval Probability: {probability:.2f}")
    st.progress(float(probability))

# -----------------------------
# MODEL ACCURACY PAGE
# -----------------------------
elif menu == "Model Accuracy":

    st.title("📈 Model Performance")

    try:
        acc_df = pd.read_csv("model_accuracy.csv")

        fig = px.bar(
            acc_df,
            x="Model",
            y="Accuracy",
            color="Accuracy",
            text="Accuracy"
        )

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.warning("Run train_model.py first to generate accuracy file.")

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif menu == "About":

    st.title(" About Project")

    st.write("""
    This Smart Loan Approval Prediction system uses Machine Learning
    to predict loan eligibility.

    ### Models Used:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - KNN

    ### Tech Stack:
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - Plotly

    ### Developer:
    Built as a Machine Learning Project for deployment.
    """)

    st.success("Project is ready for Streamlit Cloud deployment 🚀")