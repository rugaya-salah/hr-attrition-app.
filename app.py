import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# تحميل النموذج
model = joblib.load("xgb_model.pkl")
model_features = joblib.load("model_features.pkl")

# إعداد الصفحة
st.set_page_config(
    page_title="HR Analytics",
    layout="wide",
    page_icon="🧑‍💼"
)

# العنوان الرئيسي
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🔍 HR Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>نموذج تحليل استقالة الموظفين باستخدام الذكاء الاصطناعي</p>", unsafe_allow_html=True)

# 📷 صورة رأسية
image = Image.open("employee_banner.png")
st.image(image, width=600)

# نموذج الإدخال
with st.form("prediction_form"):
    st.subheader("📋 بيانات الموظف")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.slider("العمر", 18, 60, 30)
        JobSatisfaction = st.selectbox("رضا الوظيفة", [1, 2, 3, 4])
        DistanceFromHome = st.slider("المسافة من المنزل", 1, 30, 10)
        OverTime = st.selectbox("العمل الإضافي", ["Yes", "No"])

    with col2:
        MonthlyIncome = st.number_input("الدخل الشهري", 1000, 20000, 5000)
        NumCompaniesWorked = st.slider("عدد الشركات السابقة", 0, 10, 2)
        TotalWorkingYears = st.slider("إجمالي سنوات الخبرة", 0, 40, 10)
        Gender = st.selectbox("الجنس", ["Male", "Female"])

    with col3:
        JobRole = st.selectbox("الوظيفة", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        MaritalStatus = st.selectbox("الحالة الاجتماعية", ["Single", "Married", "Divorced"])
        BusinessTravel = st.selectbox("السفر في العمل", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        EducationField = st.selectbox("مجال التعليم", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])

    submitted = st.form_submit_button("🔍 تحليل")

# ✅ التحليل
if submitted:
    user_input = {
        "Age": Age,
        "DistanceFromHome": DistanceFromHome,
        "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "TotalWorkingYears": TotalWorkingYears,
        "OverTime": OverTime,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "BusinessTravel": BusinessTravel,
        "EducationField": EducationField,
    }

    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

    with result_col2:
        st.subheader("🔎 النتيجة")
        st.metric(label="⚠️ احتمالية الاستقالة", value=f"{prediction_proba * 100:.2f} %")

        if prediction == 1:
            st.error("الموظف معرض للاستقالة 🚨")
            sad_img = Image.open("sad_employee.png")
            st.image(sad_img, caption="موظف غير سعيد", width=200)
        else:
            st.success("الموظف غير معرض للاستقالة ✅")
            happy_img = Image.open("happy_employee.png")
            st.image(happy_img, caption="موظف سعيد", width=200)

        # 📊 رسم بياني أفقي
        fig, ax = plt.subplots()
        labels = ["باقٍ", "سيستقيل"]
        values = [1 - prediction_proba, prediction_proba]
        colors = ["#4CAF50", "#F44336"]
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("الاحتمال")
        st.pyplot(fig)

# تذييل
st.markdown("---")
st.caption("تم التطوير باستخدام Streamlit و XGBoost")
