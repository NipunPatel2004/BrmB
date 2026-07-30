import streamlit as st
import pandas as pd


def overview():

# Load Dataset

    df = pd.read_csv("Data/heart_cleveland_upload.csv")


# Title
    st.title("Heart Disease Dataset Overview")

    st.write("""
    This page provides a complete overview of the Heart Disease dataset used
    for building the Machine Learning prediction model.
    """)

    st.divider()

    # Dataset summary
    st.subheader(" Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Features", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())

    st.divider()

  
# Dataset Information
    st.subheader("ℹ Dataset Information")

    info = pd.DataFrame({
        "Property": [
            "Dataset Name",
            "Problem Type",
            "Target Variable",
            "Target Values",
            "Source"
        ],
        "Details": [
            "Heart Disease Cleveland Dataset",
            "Binary Classification",
            "condition",
            "0 = No Heart Disease, 1 = Heart Disease",
            "UCI Machine Learning Repository"
        ]
    })

    st.dataframe(info, use_container_width=True, hide_index=True)

    st.divider()

# Dataset Preview

    st.subheader("Dataset Preview")

    rows = st.slider(
        "Number of rows",
        5,
        min(297, len(df)),
        10
    )

    st.dataframe(
        df.head(rows),
        use_container_width=True
    )

    st.divider()

  
    # Dataset Shape
    st.subheader("Dataset Shape")

    shape = pd.DataFrame({
        "Property": ["Rows", "Columns"],
        "Value": [df.shape[0], df.shape[1]]
    })

    st.dataframe(shape, use_container_width=True, hide_index=True)

    st.divider()

# Column Information

    st.subheader("Feature Description")

    feature_info = pd.DataFrame({

        "Feature": [
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
            "thal",
            "condition"
        ],

        "Description": [
            "Age of patient",
            "Gender",
            "Chest pain type",
            "Resting blood pressure",
            "Serum cholesterol",
            "Fasting blood sugar",
            "Resting ECG",
            "Maximum heart rate",
            "Exercise induced angina",
            "ST depression",
            "Slope of ST segment",
            "Number of major vessels",
            "Thalassemia",
            "Heart disease"
        ],

        "Medical Importance": [
            "Older people have higher heart disease risk.",
            "Risk differs between males and females.",
            "Chest pain is an important symptom.",
            "High BP increases heart workload.",
            "High cholesterol can block arteries.",
            "High blood sugar indicates diabetes risk.",
            "Detects electrical abnormalities.",
            "Shows heart performance during exercise.",
            "Chest pain while exercising is a major indicator.",
            "Measures heart stress.",
            "Useful for coronary artery diagnosis.",
            "Shows blocked blood vessels.",
            "Detects blood flow abnormalities.",
            "Prediction target."
        ]
    })

    st.dataframe(
        feature_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

# Data Types
    st.subheader("Data Types")

    datatype = pd.DataFrame({
        "Feature": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(
        datatype,
        use_container_width=True,
        hide_index=True
    )

    st.divider()


# Missing Values
    st.subheader("Missing Value Report")

    missing = pd.DataFrame({
        "Feature": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Duplicate Rows
    st.subheader("Duplicate Rows")

    duplicate = df.duplicated().sum()

    st.write(f"Total Duplicate Rows : **{duplicate}**")

    st.divider()

    # Statistical Summary
    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.divider()

# Target Variable
    st.subheader("Target Variable")

    st.info("""
### Heart Disease Condition

- **0** → No Heart Disease

- **1** → Heart Disease Present
""")

    target = (
        df["condition"]
        .value_counts()
        .rename_axis("Condition")
        .reset_index(name="Count")
    )

    st.dataframe(
        target,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

# Dataset Insights
    st.subheader("Dataset Insights")

    st.success(f"""
✔ Total Patients : {df.shape[0]}

✔ Total Features : {df.shape[1]}

✔ Missing Values : {df.isnull().sum().sum()}

✔ Duplicate Rows : {df.duplicated().sum()}

✔ Problem Type : Binary Classification

✔ Target Variable : condition
""")