import streamlit as st
import pandas as pd
import plotly.express as px



def visualization():

# Load Dataset

    df = pd.read_csv("Data/heart_cleveland_upload.csv")

    st.title("Exploratory Data Analysis (EDA)")
    st.write("Analyze the Heart Disease dataset using different visualization techniques.")

    st.divider()

# Dataset summary


    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    st.divider()

# Feature Lists

    histogram_features = {
        "Age": "age",
        "Resting Blood Pressure": "trestbps",
        "Serum Cholesterol": "chol",
        "Maximum Heart Rate": "thalach",
        "ST Depression": "oldpeak"
    }

    categorical_features = {
        "Gender": "sex",
        "Chest Pain Type": "cp",
        "Fasting Blood Sugar": "fbs",
        "Resting ECG": "restecg",
        "Exercise Induced Angina": "exang",
        "Slope of ST Segment": "slope",
        "Major Vessels": "ca",
        "Thalassemia": "thal",
        "Heart Disease": "condition"
    }

    pie_features = {
        "Gender": "sex",
        "Heart Disease": "condition",
        "Chest Pain Type": "cp",
        "Exercise Induced Angina": "exang",
        "Thalassemia": "thal"
    }
# Tabs

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Histogram",
        "Box Plot",
        "Scatter Plot",
        "Bar Chart",
        "Pie Chart",
        "Correlation"
    ])

# Histogram
    
    
    with tab1:

        st.subheader("Histogram")

        feature = st.selectbox(
            "Select Feature",
            list(histogram_features.keys()),
            key="histogram_feature"
        )

        column = histogram_features[feature]

        fig = px.histogram(
            df,
            x=column,
            nbins=20,
            title=f"{feature} Distribution"
        )

        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="histogram_chart"
        )

        st.info("Histogram shows how the values are distributed.")
        
# Box Plot

    with tab2:

        st.subheader("Box Plot")

        feature = st.selectbox(
            "Select Feature",
            list(histogram_features.keys()),
            key="box_feature"
        )

        column = histogram_features[feature]

        fig = px.box(
            df,
            y=column,
            title=feature
        )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="box_chart"
        )

        st.info("Box Plot helps identify outliers and data spread.")

# Scatter Plot


    with tab3:

        st.subheader("Scatter Plot")

        x_feature = st.selectbox(
            "Select X-axis",
            list(histogram_features.keys()),
            key="scatter_x"
        )

        y_feature = st.selectbox(
            "Select Y-axis",
            list(histogram_features.keys()),
            index=1,
            key="scatter_y"
        )

        fig = px.scatter(
            df,
            x=histogram_features[x_feature],
            y=histogram_features[y_feature],
            color="condition",
            title=f"{x_feature} vs {y_feature}"
        )

        fig.update_layout(
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="scatter_chart"
        )

        st.info("Scatter Plot shows the relationship between two numerical features.")

        with tab4:

            st.subheader("Bar Chart")

            feature = st.selectbox(
                "Select Feature",
                list(categorical_features.keys()),
                key="bar_feature"
            )

            column = categorical_features[feature]

            counts = df[column].value_counts().reset_index()
            counts.columns = [feature, "Count"]

            fig = px.bar(
                counts,
                x=feature,
                y="Count",
                title=feature
            )

            fig.update_layout(
                height=550
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="bar_chart"
            )

            st.info("Bar Chart compares the frequency of different categories.")

 # Pie Chart


    with tab5:

        st.subheader("Pie Chart")

        feature = st.selectbox(
            "Select Feature",
            list(categorical_features.keys()),
            key="pie_feature"
        )

        column = categorical_features[feature]

        fig = px.pie(
            df,
            names=column,
            title=feature
        )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="pie_chart"
        )

        st.info("Pie Chart displays the proportion of each category.")

# Correlation Heatmap

    with tab6:

        st.subheader("Correlation Heatmap")

        st.write("""
        This heatmap shows the relationship between numerical features.

        - **+1** → Strong Positive Correlation
        - **0** → No Correlation
        - **-1** → Strong Negative Correlation
        """)

        corr = df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )

        fig.update_layout(
            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="heatmap_chart"
        )

        st.success(
            "This heatmap helps identify relationships between different numerical features."
        )