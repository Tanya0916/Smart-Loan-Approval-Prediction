import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Smart Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main{
background-color:#F8F9FA;
}

.title{
font-size:40px;
font-weight:bold;
color:#0E76A8;
text-align:center;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
}

.card{
background:#ffffff;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

.metric{
font-size:22px;
font-weight:bold;
color:#0066cc;
}

.footer{
text-align:center;
font-size:15px;
color:gray;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

model = joblib.load("model.pkl")

df = pd.read_csv("loan_prediction.csv")

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.image(
"https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
width=120
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

"Go To",

[
"  Home",
" 📊 Dashboard",
" 📈 Visualizations",
" Prediction",
"About"
]

)

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------

if page== "  Home":

    st.markdown(
    "<div class='title'>🏦 Smart Loan Approval Prediction System</div>",
    unsafe_allow_html=True
    )

    st.markdown(
    "<div class='subtitle'>Predict whether a loan application is approved using Machine Learning.</div>",
    unsafe_allow_html=True
    )

    st.write("")

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Rows",df.shape[0])
    col2.metric("Columns",df.shape[1])
    col3.metric("Model","KNN")
    col4.metric("Accuracy","86.99%")

    st.write("")

    st.markdown("##  Project Overview")

    st.info("""

This application predicts loan approval based on:

✔ Gender

✔ Married Status

✔ Education

✔ Income

✔ Credit History

✔ Property Area

✔ Loan Amount

using Machine Learning.

""")

    st.write("")

    st.markdown("## 📂 Dataset Preview")

    st.dataframe(df.head(10),use_container_width=True)

    st.write("")

    st.markdown("##  Dataset Shape")

    st.success(f"Rows : {df.shape[0]}")

    st.success(f"Columns : {df.shape[1]}")

    st.write("")

    st.markdown("## 📋 Column Names")

    st.write(df.columns.tolist())
    

# DASHBOARD


elif page == " 📊 Dashboard":

    st.title("Loan Dataset Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Applicants", len(df))

    with col2:
        approved = (df["Loan_Status"] == "Y").sum()
        st.metric("Loans Approved", approved)

    st.divider()

    st.subheader("Dataset Information")

    st.write(df.describe())

    st.divider()

    st.subheader("Missing Values")

    missing = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
    st.dataframe(missing, use_container_width=True)

    st.divider()

    st.subheader("Data Types")

    dtype = pd.DataFrame(df.dtypes, columns=["Datatype"])
    st.dataframe(dtype, use_container_width=True)


# VISUALIZATION


elif page == " 📈 Visualizations":

    st.title(" 📈 Data Visualizations")

    chart = st.selectbox(

        "Choose Chart",

        [

            "Loan Status",

            "Gender",

            "Education",

            "Property Area",

            "Applicant Income",

            "Loan Amount",

            "Credit History"

        ]

    )

    # Loan Status 

    if chart == "Loan Status":

        fig, ax = plt.subplots(figsize=(6,4))

        df["Loan_Status"].value_counts().plot(

            kind="bar",

            ax=ax

        )

        ax.set_title("Loan Approval Distribution")

        st.pyplot(fig)

    # Gender

    elif chart == "Gender":

        fig, ax = plt.subplots(figsize=(6,4))

        df["Gender"].value_counts().plot(

            kind="pie",

            autopct="%1.1f%%",

            ax=ax

        )

        st.pyplot(fig)

    #  Education 

    elif chart == "Education":

        fig, ax = plt.subplots(figsize=(6,4))

        df["Education"].value_counts().plot(

            kind="bar",

            ax=ax

        )

        ax.set_title("Education")

        st.pyplot(fig)

    # Property Area 

    elif chart == "Property Area":

        fig, ax = plt.subplots(figsize=(6,4))

        df["Property_Area"].value_counts().plot(

            kind="bar",

            ax=ax

        )

        st.pyplot(fig)

    # Income

    elif chart == "Applicant Income":

        fig, ax = plt.subplots(figsize=(8,4))

        ax.hist(df["ApplicantIncome"], bins=30)

        ax.set_title("Applicant Income Distribution")

        st.pyplot(fig)

    #  Loan Amount 

    elif chart == "Loan Amount":

        fig, ax = plt.subplots(figsize=(8,4))

        ax.hist(df["LoanAmount"].dropna(), bins=30)

        ax.set_title("Loan Amount Distribution")

        st.pyplot(fig)

    #  Credit History 

    elif chart == "Credit History":

        fig, ax = plt.subplots(figsize=(6,4))

        df["Credit_History"].value_counts().plot(

            kind="bar",

            ax=ax

        )

        st.pyplot(fig)
        # ==========================================================
# PREDICTION
# ==========================================================

elif page == " Prediction":

    st.title(" Loan Approval Prediction")

    st.markdown("Fill all the applicant details below.")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox("Gender", ["Male", "Female"])

        married = st.selectbox("Married", ["Yes", "No"])

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

    with col2:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=1500
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=150
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [360,180,240,300,120,84,60]
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0,0.0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban","Semiurban","Rural"]
        )

    st.write("")

    if st.button("🔍 Predict Loan Status", use_container_width=True):
    
     input_df = pd.DataFrame({

        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [float(applicant_income)],
        "CoapplicantIncome": [float(coapplicant_income)],
        "LoanAmount": [float(loan_amount)],
        "Loan_Amount_Term": [float(loan_term)],
        "Credit_History": [float(credit_history)],
        "Property_Area": [property_area]

    })

    try:

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:

            confidence = probability[0][1] * 100

            st.success("✅ Congratulations! Loan Approved")
            st.progress(int(confidence))

            st.metric(
                "Approval Probability",
                f"{confidence:.2f}%"
            )

            st.balloons()

        else:

            confidence = probability[0][0] * 100

            st.error("❌ Loan Rejected")
            st.progress(int(confidence))

            st.metric(
                "Rejection Probability",
                f"{confidence:.2f}%"
            )

        st.divider()

        st.subheader("Applicant Details")

        st.dataframe(input_df)

        csv = input_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Report",
            csv,
            "prediction.csv",
            "text/csv"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)
# ==========================================================
# ABOUT
# ==========================================================

elif page == "About":
    
    st.title("About This Project")

    st.write("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            width=180
        )

    with col2:
        st.subheader("Smart Loan Approval Prediction System")

        st.write("""
This project predicts whether a loan application will be **Approved** or **Rejected**
using Machine Learning.

It is built using Python, Scikit-Learn, Pandas and Streamlit.
        """)

    st.write("---")

    st.subheader(" Technologies")

    st.markdown("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
    """)

    st.write("---")

    st.subheader(" Features")

    st.success("✔ Loan Prediction")
    st.success("✔ Interactive Dashboard")
    st.success("✔ Data Visualization")
    st.success("✔ Model Accuracy")
    st.success("✔ Download Prediction Report")

    st.write("---")

    st.subheader(" Machine Learning Models")

    st.table({
        "Model":[
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN"
        ],
        "Status":[
            "Compared",
            "Compared",
            "Compared",
            "Compared"
        ]
    })

    st.write("---")

    st.info("Developed for educational purposes using Streamlit.")
    
    st.write("---")

st.markdown(
"""
<center>

© 2025 Smart Loan Approval Prediction System

Made with using Streamlit

</center>
""",
unsafe_allow_html=True
)