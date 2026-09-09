import streamlit as st
import pandas as pd
import numpy as np
import re

from collections import Counter

import plotly.express as px


from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud

from Utils.css_loader import load_css



# HELPER FUNCTIONS


def get_words(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    return words


def get_word_counts(text_series):

    counter = Counter()

    for text in text_series:

        words = get_words(text)

        counter.update(words)

    return counter


def get_ngrams(text_series, n=2, top_n=20):

    texts = text_series.fillna("").astype(str)

    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        stop_words="english"
    )

    try:

        matrix = vectorizer.fit_transform(texts)

        counts = np.asarray(
            matrix.sum(axis=0)
        ).ravel()

        terms = vectorizer.get_feature_names_out()

        result = pd.DataFrame({
            "N-Gram": terms,
            "Count": counts
        })

        result = result.sort_values(
            "Count",
            ascending=False
        ).head(top_n)

        return result

    except ValueError:

        return pd.DataFrame(
            columns=["N-Gram", "Count"]
        )


def create_wordcloud(text_series):

    text = " ".join(
        text_series.fillna("").astype(str)
    )

    if not text.strip():
        return None

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=150,
        collocations=False
    ).generate(text)

    return wordcloud



# MAIN PAGE


def visualization_page():

    load_css()


    # LOAD DATA
  

    df = pd.read_csv(
        "Data/balanced_game_reviews_80k.csv"
    )

    # HEADER


    st.title(
        "Visualization & Exploratory Data Analysis"
    )

    st.caption(
        "Interactive analysis of gaming reviews, "
        "sentiment patterns and text characteristics."
    )

    st.divider()


    # CHECK REVIEW COLUMN
   

    if "review_text" not in df.columns:

        st.error(
            " 'review_text' column was not found."
        )

        return

    review_text = (
        df["review_text"]
        .fillna("")
        .astype(str)
    )

  
    # CREATE TEXT FEATURES
    

    df["character_count"] = (
        review_text.str.len()
    )

    df["word_count"] = (
        review_text
        .str.split()
        .str.len()
    )

    df["sentence_count"] = (
        review_text
        .str.count(r"[.!?]")
    )

    df["uppercase_count"] = (
        review_text.apply(
            lambda x: sum(
                1 for c in x
                if c.isupper()
            )
        )
    )

    df["exclamation_count"] = (
        review_text.str.count("!")
    )

    df["question_count"] = (
        review_text.str.count(r"\?")
    )

  
    # 1. SENTIMENT ANALYSIS

    with st.expander("1. Sentiment Analysis", expanded=False):
        st.header(
            "1. Sentiment Analysis"
        )

        if "review_score" in df.columns:

            positive_count = int(
                (df["review_score"] == 1).sum()
            )

            negative_count = int(
                (df["review_score"] == -1).sum()
            )

            total_labeled = (
                positive_count +
                negative_count
            )

            positive_percentage = (
                positive_count /
                total_labeled * 100
                if total_labeled > 0
                else 0
            )

            negative_percentage = (
                negative_count /
                total_labeled * 100
                if total_labeled > 0
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    " Positive",
                    f"{positive_count:,}"
                )

            with col2:
                st.metric(
                    " Negative",
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

            sentiment_data = pd.DataFrame({
                "Sentiment": [
                    "Positive",
                    "Negative"
                ],
                "Reviews": [
                    positive_count,
                    negative_count
                ]
            })

        
            # Interactive Bar Chart
    

            fig = px.bar(
                sentiment_data,
                x="Sentiment",
                y="Reviews",
                text="Reviews",
                title="Gaming Review Sentiment Distribution"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Number of Reviews",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            
            # Interactive Pie Chart


            fig = px.pie(
                sentiment_data,
                names="Sentiment",
                values="Reviews",
                hole=0.4,
                title="Positive vs Negative Review Percentage"
            )

            fig.update_traces(
                textinfo="label+percent"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        else:

            st.warning(
                "The 'review_score' column was not found."
            )

    with st.expander("2. Review Length Analysis", expanded=False): 
        # 2. REVIEW LENGTH ANALYSIS
    

        st.header(
            " 2. Review Length Analysis"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Average Characters",
                f"{df['character_count'].mean():.0f}"
            )

        with col2:
            st.metric(
                "Average Words",
                f"{df['word_count'].mean():.1f}"
            )

        with col3:
            st.metric(
                "Shortest Review",
                f"{df['character_count'].min():,}"
            )

        with col4:
            st.metric(
                "Longest Review",
                f"{df['character_count'].max():,}"
            )

        # Character Distribution
        

        st.subheader(
            " Character Length Distribution"
        )

        fig = px.histogram(
            df,
            x="character_count",
            nbins=50,
            title="Distribution of Review Character Length"
        )

        fig.update_layout(
            xaxis_title="Number of Characters",
            yaxis_title="Number of Reviews",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Word Distribution
    

        st.subheader(
            "Words per Review"
        )

        fig = px.histogram(
            df,
            x="word_count",
            nbins=40,
            title="Distribution of Words per Review"
        )

        fig.update_layout(
            xaxis_title="Number of Words",
            yaxis_title="Number of Reviews",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        
        # REVIEW LENGTH BY SENTIMENT


        if "review_score" in df.columns:

            st.subheader(
                "Review Length by Sentiment"
            )

            length_df = df[
                df["review_score"].isin([1, -1])
            ].copy()

            length_df["Sentiment"] = (
                length_df["review_score"]
                .map({
                    1: "Positive",
                    -1: "Negative"
                })
            )

            fig = px.box(
                length_df,
                x="Sentiment",
                y="word_count",
                points=False,
                title="Word Count Comparison by Sentiment"
            )

            fig.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Number of Words",
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # 3. TEXT QUALITY
    
    with st.expander("3. Text Quality Analysis", expanded=False):
        st.header(
            " 3. Text Quality Analysis"
        )

        empty_reviews = int(
            (review_text.str.strip() == "").sum()
        )

        very_short_reviews = int(
            (df["word_count"] <= 3).sum()
        )

        long_reviews = int(
            (df["word_count"] >= 100).sum()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Empty Reviews",
                f"{empty_reviews:,}"
            )

        with col2:
            st.metric(
                "Very Short Reviews",
                f"{very_short_reviews:,}"
            )

        with col3:
            st.metric(
                "100+ Word Reviews",
                f"{long_reviews:,}"
            )

        # Punctuation

        st.subheader(
            "❗ Punctuation & Capitalization"
        )

        punctuation_data = pd.DataFrame({
            "Type": [
                "Exclamation (!)",
                "Question (?)",
                "Uppercase Characters"
            ],
            "Count": [
                int(df["exclamation_count"].sum()),
                int(df["question_count"].sum()),
                int(df["uppercase_count"].sum())
            ]
        })

        fig = px.bar(
            punctuation_data,
            x="Type",
            y="Count",
            text="Count",
            title="Punctuation and Uppercase Character Usage"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # 4. WORD FREQUENCY
    with st.expander("4. Word Frequency Analysis", expanded=False):
        st.header(
            " 4. Word Frequency Analysis"
        )

        all_word_counts = get_word_counts(
            review_text
        )

        common_stopwords = {
            "the", "and", "a", "to", "of",
            "is", "in", "it", "for", "this",
            "that", "on", "with", "was",
            "but", "are", "as", "be",
            "have", "has", "i", "you",
            "my", "game"
        }

        filtered_all_words = Counter({
            word: count
            for word, count in all_word_counts.items()
            if word not in common_stopwords
            and len(word) > 2
        })

        top_words = filtered_all_words.most_common(20)

        if top_words:

            word_df = pd.DataFrame(
                top_words,
                columns=[
                    "Word",
                    "Frequency"
                ]
            )

            fig = px.bar(
                word_df.sort_values("Frequency"),
                x="Frequency",
                y="Word",
                orientation="h",
                text="Frequency",
                title="Top 20 Most Frequent Words"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=650,
                yaxis_title="Word",
                xaxis_title="Frequency"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                word_df,
                use_container_width=True,
                hide_index=True
            )

    # 5. POSITIVE / NEGATIVE VOCABULARY
    with st.expander("5. Positive vs Negative Vocabulary", expanded=False):
        if "review_score" in df.columns:

            st.header(
                " 5. Positive vs Negative Vocabulary"
            )

            positive_reviews = df.loc[
                df["review_score"] == 1,
                "review_text"
            ]

            negative_reviews = df.loc[
                df["review_score"] == -1,
                "review_text"
            ]

            positive_counts = get_word_counts(
                positive_reviews
            )

            negative_counts = get_word_counts(
                negative_reviews
            )

            positive_counts = Counter({
                word: count
                for word, count in positive_counts.items()
                if word not in common_stopwords
                and len(word) > 2
            })

            negative_counts = Counter({
                word: count
                for word, count in negative_counts.items()
                if word not in common_stopwords
                and len(word) > 2
            })

            col1, col2 = st.columns(2)

        
            # Positive Words
            
            with col1:

                st.subheader(
                    " Top Positive Words"
                )

                positive_top = positive_counts.most_common(15)

                if positive_top:

                    positive_df = pd.DataFrame(
                        positive_top,
                        columns=[
                            "Word",
                            "Frequency"
                        ]
                    )

                    fig = px.bar(
                        positive_df.sort_values(
                            "Frequency"
                        ),
                        x="Frequency",
                        y="Word",
                        orientation="h",
                        title="Top Positive Words"
                    )

                    fig.update_layout(
                        height=550
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            # Negative Words
        

            with col2:

                st.subheader(
                    " Top Negative Words"
                )

                negative_top = negative_counts.most_common(15)

                if negative_top:

                    negative_df = pd.DataFrame(
                        negative_top,
                        columns=[
                            "Word",
                            "Frequency"
                        ]
                    )

                    fig = px.bar(
                        negative_df.sort_values(
                            "Frequency"
                        ),
                        x="Frequency",
                        y="Word",
                        orientation="h",
                        title="Top Negative Words"
                    )

                    fig.update_layout(
                        height=550
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

    # 6. N-GRAM ANALYSIS

    with st.expander("6. N-Gram Analysis", expanded=False):
        st.header(
            "6. N-Gram Analysis"
        )

        st.write(
            "N-grams identify frequently occurring combinations of words."
        )


        # BIGRAMS


        st.subheader(
            " Top 20 Bigrams"
        )

        bigram_df = get_ngrams(
            review_text,
            n=2,
            top_n=20
        )

        if not bigram_df.empty:

            fig = px.bar(
                bigram_df.sort_values("Count"),
                x="Count",
                y="N-Gram",
                orientation="h",
                text="Count",
                title="Top 20 Bigrams"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=650
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # TRIGRAMS
        

        st.subheader(
            " Top 20 Trigrams"
        )

        trigram_df = get_ngrams(
            review_text,
            n=3,
            top_n=20
        )

        if not trigram_df.empty:

            fig = px.bar(
                trigram_df.sort_values("Count"),
                x="Count",
                y="N-Gram",
                orientation="h",
                text="Count",
                title="Top 20 Trigrams"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=650
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # 7. WORD CLOUDS
    with st.expander("7. Word Cloud Analysis", expanded=False):
        st.header(
            " 7. Word Cloud Analysis"
        )

        st.write(
            "Visual representation of frequently occurring words."
        )

    
        # All Reviews
        

        st.subheader(
            " All Reviews"
        )

        all_cloud = create_wordcloud(
            review_text
        )

        if all_cloud is not None:

            st.subheader("All Reviews")

            all_cloud = create_wordcloud(review_text)

        if all_cloud is not None:

            st.image(
                all_cloud.to_array(),
                use_container_width=True
            )

        # Positive / Negative

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                " Positive Reviews"
            )

            positive_cloud = create_wordcloud(
                positive_reviews
            )

            if positive_cloud is not None:

                st.image(
                    positive_cloud.to_array(),
                    use_container_width=True
                )

        with col2:

            st.subheader(
                " Negative Reviews"
            )

            negative_cloud = create_wordcloud(
                negative_reviews
            )

            if negative_cloud is not None:

                st.image(
                    negative_cloud.to_array(),
                    use_container_width=True
                )

    
        # 8. VOCABULARY ANALYSIS
    
    with st.expander("8. Vocabulary Analysis", expanded=False):
        st.header(
            " 8. Vocabulary Analysis"
        )

        unique_words = len(
            filtered_all_words
        )

        total_words = sum(
            filtered_all_words.values()
        )

        vocabulary_ratio = (
            unique_words / total_words
            if total_words > 0
            else 0
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Unique Words",
                f"{unique_words:,}"
            )

        with col2:

            st.metric(
                "Total Words",
                f"{total_words:,}"
            )

        with col3:

            st.metric(
                "Vocabulary Ratio",
                f"{vocabulary_ratio:.4f}"
            )
        
        # 9. REVIEW STATISTICS
    with st.expander("9. Review Statistics", expanded=False):
            st.header(
                " 9. Review Statistics"
            )

            statistics = pd.DataFrame({
                "Metric": [
                    "Average characters",
                    "Median characters",
                    "Minimum characters",
                    "Maximum characters",
                    "Average words",
                    "Median words",
                    "Minimum words",
                    "Maximum words",
                    "Average sentences"
                ],

                "Value": [
                    round(
                        df["character_count"].mean(),
                        2
                    ),

                    round(
                        df["character_count"].median(),
                        2
                    ),

                    int(
                        df["character_count"].min()
                    ),

                    int(
                        df["character_count"].max()
                    ),

                    round(
                        df["word_count"].mean(),
                        2
                    ),

                    round(
                        df["word_count"].median(),
                        2
                    ),

                    int(
                        df["word_count"].min()
                    ),

                    int(
                        df["word_count"].max()
                    ),

                    round(
                        df["sentence_count"].mean(),
                        2
                    )
                ]
            })

            st.dataframe(
                statistics,
                use_container_width=True,
                hide_index=True
            )

        # 10. SENTIMENT-BASED STATISTICS
    with st.expander("10. Sentiment-Based Text Statistics Analysis", expanded=False):
            if "review_score" in df.columns:

                st.header(
                    "10. Sentiment-Based Text Statistics"
                )

                sentiment_stats = (
                    df.groupby("review_score")
                    .agg(
                        Reviews=("review_text", "count"),
                        Avg_Words=("word_count", "mean"),
                        Median_Words=("word_count", "median"),
                        Avg_Characters=(
                            "character_count",
                            "mean"
                        ),
                        Median_Characters=(
                            "character_count",
                            "median"
                        )
                    )
                    .reset_index()
                )

                sentiment_stats["Sentiment"] = (
                    sentiment_stats["review_score"]
                    .map({
                        1: "Positive",
                        -1: "Negative"
                    })
                )

                sentiment_stats = sentiment_stats[
                    [
                        "Sentiment",
                        "Reviews",
                        "Avg_Words",
                        "Median_Words",
                        "Avg_Characters",
                        "Median_Characters"
                    ]
                ]

                numeric_columns = [
                    "Avg_Words",
                    "Median_Words",
                    "Avg_Characters",
                    "Median_Characters"
                ]

                sentiment_stats[numeric_columns] = (
                    sentiment_stats[numeric_columns]
                    .round(2)
                )

                st.dataframe(
                    sentiment_stats,
                    use_container_width=True,
                    hide_index=True
                )

                # Interactive comparison
                comparison_df = sentiment_stats[
                    [
                        "Sentiment",
                        "Avg_Words",
                        "Avg_Characters"
                    ]
                ]

                fig = px.bar(
                    comparison_df,
                    x="Sentiment",
                    y=[
                        "Avg_Words",
                        "Avg_Characters"
                    ],
                    barmode="group",
                    title="Average Review Size by Sentiment"
                )

                fig.update_layout(
                    height=500
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

