import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("pipeline_randomforest.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="HR Attrition App", layout="centered")

st.title("HR Attrition Prediction App")
st.write("Enter employee details to predict attrition risk.")

# Input fields
Age = st.number_input("Age", min_value=18, max_value=60, value=30)
MonthlyIncome = st.number_input("Monthly Income", value=30000)
NumCompaniesWorked = st.number_input("Number of Companies Worked", value=1)
YearsAtCompany = st.number_input("Years at Company", value=2)
YearsInCurrentRole = st.number_input("Years in Current Role", value=1)

OverTime = st.selectbox("OverTime", ["Yes", "No"])
JobSatisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
WorkLifeBalance = st.slider("Work Life Balance (1–4)", 1, 4, 3)
EnvironmentSatisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)

Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
EducationField = st.selectbox("Education Field", ["Engineering", "Marketing", "Life Sciences", "Other"])
JobRole = st.text_input("Job Role", "Manufacturing Director")
Department = st.text_input("Department", "Operations")
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Prediction button
if st.button("Predict Attrition"):
    new_data = pd.DataFrame([{
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "OverTime": OverTime,
        "JobSatisfaction": JobSatisfaction,
        "WorkLifeBalance": WorkLifeBalance,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Education": Education,
        "EducationField": EducationField,
        "JobRole": JobRole,
        "Department": Department,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus
    }])

    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Employee is likely to Leave | Probability: {probability:.2f}")
    else:
        st.success(f"✅ Employee is likely to Stay | Probability: {probability:.2f}")


