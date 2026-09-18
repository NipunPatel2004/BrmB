import streamlit as st
import pandas as pd

from Utils.loader import load_css


st.set_page_config(
    page_title="Dataset Overview",
    layout="wide"
)

load_css()


# load data
@st.cache_data
def load_data():

    df = pd.read_csv(
        "Data/Restaurant_Reviews.tsv",
        delimiter="\t"
    )

    return df


df = load_data()


st.markdown(
    '<div class="page-title dataset-title">Dataset Overview</div>',
    unsafe_allow_html=True
)

st.write(
    "Explore the restaurant review dataset."
)


# dataset summary
st.divider()
st.markdown(
    '<div class="section-title">Dataset Summary</div>',
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)


with col1:
        st.metric(
            "Total Reviews",
            len(df)
        )

with col2:
        st.metric(
            "Columns",
            len(df.columns)
        )

with col3:
        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )

with col4:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )


# dataset preview
st.divider()
st.markdown(
    '<div class="section-title">Dataset Preview</div>',
    unsafe_allow_html=True
)

row_count = st.slider(
        "Number of rows",
        min_value=5,
        max_value=100,
        value=10,
        step=5
    )

st.dataframe(
        df.head(row_count),
        use_container_width=True
    )


# column information
st.divider()
st.markdown(
    '<div class="section-title">Column Information</div>',
    unsafe_allow_html=True
)
col_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

st.dataframe(
        col_df,
        use_container_width=True
    )

st.divider()
# duplicate reviews

st.markdown(
    '<div class="section-title">Duplicate Reviews</div>',
    unsafe_allow_html=True
)

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:

        dup_df = df[
            df.duplicated(
                keep=False
            )
        ]

        review_count = (
            dup_df.groupby(
                ["Review", "Liked"]
            )
            .size()
            .reset_index(
                name="Times Repeated"
            )
            .sort_values(
                "Times Repeated",
                ascending=False
            )
        )

        st.dataframe(
            review_count,
            use_container_width=True
        )

else:

        st.success(
            "No duplicate reviews found."
        )


# sentiment distribution
st.divider()
st.markdown(
    '<div class="section-title">Sentiment Distribution</div>',
    unsafe_allow_html=True
)
sentiment_df = (
        df["Liked"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

sentiment_df.columns = [
        "Sentiment",
        "Count"
    ]

sentiment_df["Sentiment"] = (
        sentiment_df["Sentiment"]
        .map({
            0: "Negative",
            1: "Positive"
        })
    )

sentiment_df["Percentage"] = (
        sentiment_df["Count"]
        / len(df)
        * 100
    ).round(2)

st.dataframe(
        sentiment_df,
        use_container_width=True
    )