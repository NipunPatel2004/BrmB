import streamlit as st
import pandas as pd
from Utils.model_loader import load_models
import pickle




model, scaler, accuracy, report, cm = load_models()

def prediction_page():
    st.title("heart disase risk prediction")
    st.divider()


    col1, col2 = st.columns(2)

    # collect inputdata 
    with col1:

        st.subheader("personal information")

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=45,
            help="Enter your age in years."
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "Typical angina",
                "Atypical angina",
                "Non-anginal Pain",
                "No Chest Pain"
            ],
            help="""
    Typical Angina = Chest pain commonly linked to heart disease.

    Atypical Angina = Unusual chest discomfort.

    Non-Anginal Pain = Chest pain usually not related to the heart.

    No Chest Pain = No chest pain symptoms.
    """
        )

        blood_pressure = st.number_input(
            "Resting blood bressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120,
            help="Your blood pressure while resting."
        )

        cholesterol = st.number_input(
            "Cholesterol level (mg/dL)",
            min_value=100,
            max_value=600,
            value=200,
            help="Amount of cholesterol present in your blood."
        )

        blood_sugar = st.selectbox(
            "Fasting blood sugar",
            [
                "120 mg or Less",
                "More Than 120 mg"
            ],
            help="Blood sugar level measured after fasting."
        )

    with col2:

        st.subheader("Medical information")

        ecg = st.selectbox(
            "Heart Rhythm Test (ECG)",
            [
                "Normal",
                "Minor Changes",
                "Significant Changes"
            ],
            help="""
    Normal = No unusual findings.

    Minor Changes = Small irregularities detected.

    Significant Changes = More noticeable heart-related changes.
    """
        )

        max_heart_rate = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150,
            help="Highest heart rate reached during exercise or activity."
        )

        exercise_pain = st.selectbox(
            "Chest Pain During Exercise",
            [
                "No",
                "Yes"
            ],
            help="Do you experience chest pain during physical activity?"
        )

        oldpeak = st.number_input(
            "ECG Stress Score",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="A value calculated from a heart stress test."
        )

        slope_option = st.selectbox(
            "Heart Stress Test Pattern",
            [
                "Upsloping",
                "Flat",
                "Downsloping"
            ],
            help="Pattern observed during a heart stress test."
        )

        vessels = st.selectbox(
            "Major blood vessels detected",
            [0, 1, 2, 3],
            help="Number of major blood vessels visible during a heart imaging test."
        )

        blood_flow = st.selectbox(
            "Heart Blood Flow Test",
            [
                "Normal",
                "Fixed Defect",
                "Reversible Defect"
            ],
            help="""
    Normal = Blood flow appears normal.

    Fixed Defect = Reduced blood flow detected.

    Reversible Defect = Blood flow decreases during activity.
    """
        )


    sex = 1 if gender == "Male" else 0

    cp_map = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "No chest pain": 3
    }
    cp = cp_map[chest_pain]

    fbs = 1 if blood_sugar == "More than 120 mg" else 0

    ecg_map = {
        "Normal": 0,
        "Minor changes": 1,
        "Significant changes": 2
    }   
    restecg = ecg_map[ecg]

    exang = 1 if exercise_pain == "Yes" else 0

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = slope_map[slope_option]

    thal_map = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal = thal_map[blood_flow]



    #  button work


    st.divider()

    if st.button("Predict Heart Disease Risk", width="stretch"):

        input_data = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                blood_pressure,
                cholesterol,
                fbs,
                restecg,
                max_heart_rate,
                exang,
                oldpeak,
                slope,
                vessels,
                thal
            ]],
            columns=[
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]
        )

        # Scale input data
        inp_scl = scaler.transform(input_data)

        # Predict class
        prediction = model.predict(inp_scl)

        # Predict probability
        pro = model.predict_proba(inp_scl)

        risk_percentage = pro[0][1] * 100

        st.divider()

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error("Higher Risk of Heart Disease")

            st.metric(
                "Estimated Risk",
                f"{risk_percentage:.2f}%"
            )

            st.warning(
                "The model suggests a higher likelihood of heart disease. Consider discussing these results with a healthcare professional."
            )

            st.markdown("""
            Maintain Good Heart Health

            - Continue regular physical activity.
            - Eat a balanced diet rich in fruits and vegetables.
            - Maintain a healthy weight.
            - Drink sufficient water daily.
            - Get 7-9 hours of sleep each night.
            - Avoid smoking and excessive alcohol consumption.
            - Schedule routine health checkups.
            - Monitor blood pressure and cholesterol periodically.
            - Continue healthy lifestyle habits to reduce future risk.
            """)
            st.subheader("Recommendations")


        else:

            st.success("Lower Risk of Heart Disease")

            st.metric(
                "Estimated Risk",
                f"{risk_percentage:.2f}%"
            )

            st.info(
                "The model suggests a lower likelihood of heart disease."
            )
            
            st.subheader("Recommendations")

            st.markdown("""
            Maintain Good Heart Health

            - Continue regular physical activity.
            - Eat a balanced diet rich in fruits and vegetables.
            - Maintain a healthy weight.
            - Drink sufficient water daily.
            - Get 7-9 hours of sleep each night.
            - Avoid smoking and excessive alcohol consumption.
            - Schedule routine health checkups.
            - Monitor blood pressure and cholesterol periodically.
            - Continue healthy lifestyle habits to reduce future risk.
            """)




    st.divider()

 
    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.subheader("Model Performance")
    st.text("Classification Report")
    st.code(report)

    st.subheader("Confusion Matrix")
    st.code(cm)

    comparison = pickle.load(
    open("Models/model_comparison.pkl", "rb")
)

    st.divider()

    st.subheader(" Model Comparison")

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }
        ),
        use_container_width=True
    )

    