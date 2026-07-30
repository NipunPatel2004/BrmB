import streamlit as st


def home():

# Title

    st.title("Heart Disease Prediction System")
    st.markdown(
        """
        ### Predict Heart Disease Risk Using Machine Learning
        
        This application uses a trained Machine Learning model to analyze
        patient health parameters and predict the likelihood of heart disease.
        It also provides dataset exploration and interactive visualizations
        to better understand the factors affecting heart disease.
        """
    )

    st.divider()

# Project Overview

    st.header(" Project Overview")

    st.write("""
Heart disease is one of the leading causes of death worldwide. Early prediction
can help healthcare professionals identify high-risk patients and recommend
timely medical intervention.

This project uses patient clinical information such as age, cholesterol level,
blood pressure, heart rate, chest pain type, and ECG results to predict whether
a patient is likely to have heart disease.
""")

    st.divider()

    # ===========================
    # Key Features
    # ===========================

    st.header("Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("\Dataset Overview")
        st.write("""
- View complete dataset
- Dataset statistics
- Missing values
- Duplicate records
- Feature descriptions
""")

        st.success("Interactive Visualizations")
        st.write("""
- Histogram
- Box Plot
- Scatter Plot
- Pie Chart
- Bar Chart
- Correlation Heatmap
""")

    with col2:

        st.success("Heart Disease Prediction")
        st.write("""
- Enter patient information
- Predict heart disease risk
- Instant prediction result
- Model performance metrics
""")

        st.success("Machine Learning")
        st.write("""
- Trained Classification Model
- Data Preprocessing
- Feature Scaling
- High Prediction Accuracy
""")

    st.divider()
# Input Features

    st.header("Patient Health Parameters")

    st.write("""
The prediction model uses the following medical attributes:
""")

    features = [
        "Age",
        "Gender",
        "Chest Pain Type",
        "Resting Blood Pressure",
        "Serum Cholesterol",
        "Fasting Blood Sugar",
        "Resting ECG Results",
        "Maximum Heart Rate Achieved",
        "Exercise Induced Angina",
        "ST Depression",
        "Slope of ST Segment",
        "Number of Major Vessels",
        "Thalassemia"
    ]

    for feature in features:
        st.write(f"• {feature}")

    st.divider()


# Workflow

    st.header("System Workflow")

    st.markdown("""
1. Explore the dataset using the Data Overview page.
2. Analyze relationships through visualizations.
3. Enter patient health information.
4. The trained Machine Learning model processes the data.
5. View the predicted result instantly.
""")

    st.divider()

# Technologies Used

    st.header("Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    tech1.info("""
**Frontend**

-Streamlit

- HTML

- CSS
""")

    tech2.info("""
**Backend**

- Python

- Pandas

- NumPy

- Pickle
""")

    tech3.info("""
**Machine Learning**

- Scikit-Learn

- StandardScaler

-  Classification Model
""")

    st.divider()

    # How to Use

    st.header("How to Use")

    st.info("""
1. Open the **Data Overview** tab to understand the dataset.

2. Explore patterns in the **Visualization** tab.

3. Navigate to the **Prediction** tab.

4. Enter patient medical details.

5. Click **Predict** to view the prediction result.
""")

    st.divider()

# Footer

    st.caption("Heart Disease Prediction System | Machine Learning Project")