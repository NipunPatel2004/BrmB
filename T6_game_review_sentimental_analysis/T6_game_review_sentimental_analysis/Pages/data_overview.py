import streamlit as st
import pandas as pd

from Utils.css_loader import load_css


def data_overview_page():

    # Load CSS
    load_css()

    # Load Dataset


    df = pd.read_csv("Data/balanced_game_reviews_80k.csv")



    # Header


    st.title("Data Overview")

    st.caption(
        "Detailed overview of the gaming review dataset, "
        "including structure, quality, sentiment labels and records."
    )

    st.divider()


 
    # Dataset Summary


    st.subheader("Dataset Summary")

    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Reviews",
            f"{total_rows:,}"
        )

    with col2:
        st.metric(
            "Total Columns",
            total_columns
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}"
        )


    # Dataset Shape

    st.subheader(" Dataset Dimensions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Rows**")
        st.write(f"{df.shape[0]:,}")

    with col2:
        st.write("**Columns**")
        st.write(f"{df.shape[1]}")

    with col3:
        memory = df.memory_usage(deep=True).sum() / (1024 ** 2)

        st.write("**Memory Usage**")
        st.write(f"{memory:.2f} MB")


  
    # Dataset Preview - 100 Rows

    st.subheader("Dataset Records")

    st.write(
        "The table below displays the first **100 records**. "
        "Use the scrollbar to explore the rows."
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=500,
        hide_index=True
    )


   
    # Column Information
 

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )



    # Data Types


    st.subheader("Data Type Summary")

    dtype_counts = (
        df.dtypes
        .astype(str)
        .value_counts()
        .reset_index()
    )

    dtype_counts.columns = [
        "Data Type",
        "Number of Columns"
    ]

    st.dataframe(
        dtype_counts,
        use_container_width=True,
        hide_index=True
    )


    # Missing Value Analysis


    st.subheader(" Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().sum().values / len(df) * 100
        ).round(2)
    })

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if missing_df.empty:

        st.success(
            "Excellent! No missing values were found."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )


    # Duplicate Analysis


    st.subheader("Duplicate Records")

    if duplicate_rows == 0:

        st.success(
            "No duplicate records were found."
        )

    else:

        st.warning(
            f" {duplicate_rows:,} duplicate records "
            "were found in the dataset."
        )


    # Sentiment Analysis


    st.subheader("Sentiment Distribution")

    if "review_score" in df.columns:

        positive_count = int(
            (df["review_score"] == 1).sum()
        )

        negative_count = int(
            (df["review_score"] == -1).sum()
        )

        total_labeled = positive_count + negative_count

        positive_percentage = (
            positive_count / total_labeled * 100
            if total_labeled > 0 else 0
        )

        negative_percentage = (
            negative_count / total_labeled * 100
            if total_labeled > 0 else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Positive Reviews",
                f"{positive_count:,}"
            )

        with col2:
            st.metric(
                "Negative Reviews",
                f"{negative_count:,}"
            )

        with col3:
            st.metric(
                "Positive %",
                f"{positive_percentage:.2f}%"
            )

        with col4:
            st.metric(
                "Negative %",
                f"{negative_percentage:.2f}%"
            )

        st.write("")

        sentiment_table = pd.DataFrame({
            "Sentiment": [
                "Positive",
                "Negative"
            ],
            "Label": [
                1,
                -1
            ],
            "Reviews": [
                positive_count,
                negative_count
            ],
            "Percentage": [
                f"{positive_percentage:.2f}%",
                f"{negative_percentage:.2f}%"
            ]
        })

        st.dataframe(
            sentiment_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "The 'review_score' column was not found."
        )



    # Text Data Information
 

    st.subheader("Review Text Information")

    if "review_text" in df.columns:

        text_length = df["review_text"].astype(str).str.len()

        word_count = (
            df["review_text"]
            .astype(str)
            .str.split()
            .str.len()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Average Characters",
                f"{text_length.mean():.0f}"
            )

        with col2:
            st.metric(
                "Average Words",
                f"{word_count.mean():.1f}"
            )

        with col3:
            st.metric(
                "Shortest Review",
                f"{text_length.min():,} chars"
            )

        with col4:
            st.metric(
                "Longest Review",
                f"{text_length.max():,} chars"
            )

    else:

        st.info(
            "The 'review_text' column was not found."
        )



    # Statistical Information

    st.subheader("Numerical Column Statistics")

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numerical_columns) > 0:

        statistics = df[numerical_columns].describe().T

        statistics = statistics.round(2)

        st.dataframe(
            statistics,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns are available."
        )



    # Dataset Quality


    st.subheader(" Dataset Quality")

    quality_score = 100

    if missing_values > 0:
        quality_score -= 20

    if duplicate_rows > 0:
        quality_score -= 10

    quality_score = max(quality_score, 0)

    if quality_score == 100:

        st.success(
            f"Dataset Quality Score: **{quality_score}/100**"
        )

    elif quality_score >= 70:

        st.warning(
            f"Dataset Quality Score: **{quality_score}/100**"
        )

    else:

        st.error(
            f"Dataset Quality Score: **{quality_score}/100**"
        )