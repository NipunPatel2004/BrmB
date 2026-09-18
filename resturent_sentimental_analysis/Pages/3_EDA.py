import streamlit as st
import pandas as pd
import plotly.express as px

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from Utils.text_preprocessor import TextPreprocessor
from Utils.loader import load_css, style_fig


st.set_page_config(
    page_title="EDA Analysis",
    layout="wide"
)

load_css()


# LOAD DATA

@st.cache_data
def load_data():

    df = pd.read_csv(
        "Data/Restaurant_Reviews.tsv",
        delimiter="\t"
    )

    return df


# CLEAN REVIEW TEXT


@st.cache_data
def clean_reviews(reviews):

    cleaner = TextPreprocessor()

    return cleaner.transform(reviews)


df = load_data()

df["Sentiment"] = df["Liked"].map(
    {
        0: "Negative",
        1: "Positive"
    }
)

# Raw length is kept as-is on purpose: it measures how much the
# reviewer actually typed, which preprocessing (stopword/punctuation
# removal) would distort.
df["Review_Length"] = (
    df["Review"]
    .astype(str)
    .apply(len)
)

df["Cleaned_Review"] = clean_reviews(
    df["Review"].astype(str).tolist()
)


st.markdown(
    '<div class="page-title eda-title">Sentiment Analysis EDA</div>',
    unsafe_allow_html=True
)


# SENTIMENT DISTRIBUTION

with st.expander(
    "Sentiment Distribution",
    expanded=True
):

    sentiment_counts = (
        df["Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.5,
        title="Positive vs Negative Reviews"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# REVIEW LENGTH DISTRIBUTION

with st.expander(
    "Review Length Distribution"
):

    fig = px.histogram(
        df,
        x="Review_Length",
        nbins=30,
        title="Review Length Distribution"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# REVIEW LENGTH BY SENTIMENT

with st.expander(
    "Review Length by Sentiment"
):

    fig = px.box(
        df,
        x="Sentiment",
        y="Review_Length",
        title="Review Length by Sentiment"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# POSITIVE WORDS

positive_text = " ".join(
    df[
        df["Liked"] == 1
    ]["Cleaned_Review"]
)

positive_words = Counter(
    positive_text.split()
)

top_positive = pd.DataFrame(
    positive_words.most_common(15),
    columns=[
        "Word",
        "Frequency"
    ]
)


with st.expander(
    "Top Positive Words"
):

    fig = px.bar(
        top_positive,
        x="Frequency",
        y="Word",
        orientation="h",
        title="Most Frequent Positive Words"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        top_positive,
        use_container_width=True
    )


# NEGATIVE WORDS

negative_text = " ".join(
    df[
        df["Liked"] == 0
    ]["Cleaned_Review"]
)

negative_words = Counter(
    negative_text.split()
)

top_negative = pd.DataFrame(
    negative_words.most_common(15),
    columns=[
        "Word",
        "Frequency"
    ]
)


with st.expander(
    "Top Negative Words"
):

    fig = px.bar(
        top_negative,
        x="Frequency",
        y="Word",
        orientation="h",
        title="Most Frequent Negative Words"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        top_negative,
        use_container_width=True
    )


# POSITIVE WORD CLOUD

with st.expander(
    "Positive Word Cloud"
):

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(
        positive_text
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(fig)


# NEGATIVE WORD CLOUD

with st.expander(
    "Negative Word Cloud"
):

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="red"
    ).generate(
        negative_text
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(fig)


# POSITIVE BIGRAMS

with st.expander(
    "Top Positive Bigrams"
):

    vectorizer = CountVectorizer(
        ngram_range=(2, 2)
    )

    X = vectorizer.fit_transform(
        df[
            df["Liked"] == 1
        ]["Cleaned_Review"]
    )

    freq = X.sum(axis=0).A1

    words = (
        vectorizer
        .get_feature_names_out()
    )

    bigram_df = pd.DataFrame({
        "Bigram": words,
        "Frequency": freq
    })

    bigram_df = (
        bigram_df
        .sort_values(
            by="Frequency",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        bigram_df,
        x="Frequency",
        y="Bigram",
        orientation="h",
        title="Top Positive Bigrams"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        bigram_df,
        use_container_width=True
    )


# NEGATIVE BIGRAMS

with st.expander(
    "Top Negative Bigrams"
):

    vectorizer = CountVectorizer(
        ngram_range=(2, 2)
    )

    X = vectorizer.fit_transform(
        df[
            df["Liked"] == 0
        ]["Cleaned_Review"]
    )

    freq = X.sum(axis=0).A1

    words = (
        vectorizer
        .get_feature_names_out()
    )

    bigram_df = pd.DataFrame({
        "Bigram": words,
        "Frequency": freq
    })

    bigram_df = (
        bigram_df
        .sort_values(
            by="Frequency",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        bigram_df,
        x="Frequency",
        y="Bigram",
        orientation="h",
        title="Top Negative Bigrams"
    )

    fig = style_fig(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        bigram_df,
        use_container_width=True
    )