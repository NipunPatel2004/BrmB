
import streamlit as st
import pandas as pd
import plotly.express as px

from Utils.loader import (
    load_pipeline,
    load_labels,
    load_css,
    style_fig
)

# PAGE CONFIG

st.set_page_config(
    page_title="Restaurant Sentiment Analyzer",
    layout="wide"
)

load_css()

# LOAD FILES

pipeline = load_pipeline()
labels = load_labels()

@st.cache_data
def load_data():
    return pd.read_csv(
        "Data/Restaurant_Reviews.tsv",
        sep="\t"
    )

df = load_data()

# HEADER

st.markdown(
    '<div class="page-title single-pred-title">Restaurant Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Predict customer review sentiment using Machine Learning."
)

# RESTAURANT DROPDOWN

restaurants = sorted(
    df["Restaurant"].unique()
)

selected_restaurant = st.selectbox(
    "Select Restaurant",
    restaurants
)

# REVIEW INPUT

review = st.text_area(
    "Enter Review",
    height=180,
    placeholder="Write your review here..."
)

# PREDICT BUTTON

if st.button(
    "Predict Sentiment",
    use_container_width=True
):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        prediction = pipeline.predict(
            [review]
        )[0]

        probabilities = (
            pipeline.predict_proba(
                [review]
            )[0]
        )

        positive_prob = (
            probabilities[1] * 100
        )

        negative_prob = (
            probabilities[0] * 100
        )

        confidence = (
            max(probabilities) * 100
        )

        sentiment = labels[prediction]

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.success(
                f"{sentiment}"
            )

        else:

            st.error(
                f"{sentiment}"
            )

    
        # METRICS
    

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Positive %",
                f"{positive_prob:.2f}%"
            )

        with col2:

            st.metric(
                "Negative %",
                f"{negative_prob:.2f}%"
            )

        with col3:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.progress(
            int(confidence)
        )

    
        # WORD IMPACT ANALYSIS
    

        try:

            preprocessor = (
                pipeline.named_steps[
                    "preprocessor"
                ]
            )

            tfidf = (
                pipeline.named_steps[
                    "tfidf"
                ]
            )

            model = (
                pipeline.named_steps[
                    "model"
                ]
            )

            cleaned_text = (
                preprocessor.transform(
                    [review]
                )[0]
            )

            vector = tfidf.transform(
                [cleaned_text]
            )

            feature_names = (
                tfidf.get_feature_names_out()
            )

            coefficients = (
                model.coef_[0]
            )

            word_impacts = []

            non_zero = (
                vector.nonzero()[1]
            )

            for idx in non_zero:

                word = (
                    feature_names[idx]
                )

                tfidf_score = (
                    vector[0, idx]
                )

                impact = (
                    tfidf_score
                    * coefficients[idx]
                )

                word_impacts.append(
                    (
                        word,
                        float(impact)
                    )
                )

            impact_df = pd.DataFrame(
                word_impacts,
                columns=[
                    "Word",
                    "Impact"
                ]
            )

            impact_df = (
                impact_df
                .sort_values(
                    "Impact",
                    ascending=False
                )
            )

            st.markdown(
                '<div class="section-title">Word Contribution Analysis</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                impact_df,
                use_container_width=True
            )

        
            # HORIZONTAL BAR CHART
        

            chart_df = (
                impact_df
                .sort_values(
                    "Impact"
                )
            )

            fig = px.bar(
                chart_df,
                x="Impact",
                y="Word",
                orientation="h",
                title="Word Contribution Impact"
            )

            fig.update_layout(
                height=500
            )

            fig = style_fig(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                "Word contribution analysis is not available for the current model."
            )

