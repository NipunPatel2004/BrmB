import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from Utils.css_loader import load_css
from Utils.model_loader import load_models



# FILE PATH

DATA_PATH = "balanced_game_reviews_80k.csv"

# PREPROCESSING

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove emojis
    try:
        import emoji
        text = emoji.replace_emoji(text, replace="")
    except:
        pass

    # Tokenization
    try:

        tokens = word_tokenize(text)

    except LookupError:

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        tokens = word_tokenize(text)

    # Stopwords
    try:

        stop_words = set(
            stopwords.words("english")
        )

    except LookupError:

        nltk.download("stopwords", quiet=True)

        stop_words = set(
            stopwords.words("english")
        )

    # Keep negative words
    stop_words = stop_words - {
        "not",
        "no",
        "nor",
        "never"
    }


    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]


    # Lemmatization
     
    lemmatizer = WordNetLemmatizer()

    try:

        tokens = [
            lemmatizer.lemmatize(
                word,
                pos="v"
            )
            for word in tokens
        ]

    except LookupError:

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

        tokens = [
            lemmatizer.lemmatize(
                word,
                pos="v"
            )
            for word in tokens
        ]


    # Remove short words
    tokens = [
        word
        for word in tokens
        if len(word) > 2
    ]


    # Join tokens
    return " ".join(tokens)

# LOAD GAME DATA

@st.cache_data
def load_game_data():

    df = pd.read_csv(DATA_PATH)

    games = sorted(
        df["app_name"]
        .dropna()
        .astype(str)
        .unique()
    )

    return df, games

# PREDICTION PAGE
def prediction_page():

    load_css()


    
    # LOAD MODELS
    

    try:

        tfidf, model = load_models()

    except Exception as e:

        st.error(
            " Model files could not be loaded."
        )

        st.exception(e)

        return


    
    # LOAD DATASET
    

    try:

        df, games = load_game_data()

    except Exception as e:

        st.error(
            "Game dataset could not be loaded."
        )

        st.exception(e)

        return
 # HEADER
    
    st.title(
        "Gaming Review Sentiment Prediction"
    )

    st.caption(
        "Select a game, enter your review, and predict "
        "whether the review is Positive or Negative."
    )

    st.divider()

    # GAME SELECTION
    
    st.subheader(
        "Select Game"
    )

    selected_game = st.selectbox(
        "Game Name",
        games,
        index=None,
        placeholder="Select a game..."
    )


    # REVIEW INPUT
    
    st.subheader(
        "Enter Gaming Review"
    )

    review = st.text_area(
        "Gaming Review",
        placeholder=(
            "Example: The gameplay is amazing, "
            "the graphics are excellent and "
            "I really enjoyed this game!"
        ),
        height=180
    )


    # PREDICT BUTTON    

    if st.button(
        "Predict Sentiment",
        use_container_width=True
    ):

        # CHECK GAME
        
        if selected_game is None:

            st.warning(
                "Please select a game first."
            )

            return

  
        # CHECK REVIEW        

        if not review.strip():

            st.warning(
                " Please enter a review."
            )

            return
        try:

            
            # PREPROCESS REVIEW
            
            processed_review = preprocess_text(
                review
            )


            # TF-IDF
            
            X_input = tfidf.transform(
                [processed_review]
            )


            # PREDICTION
            
            prediction = model.predict(
                X_input
            )[0]

            
            # PROBABILITY            

            probabilities = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    X_input
                )[0]


            
            # SENTIMENT LABEL
            

            if prediction == 1:

                sentiment = "Positive"
               

            elif prediction == -1:

                sentiment = "Negative"
                

            else:

                sentiment = str(prediction)
                
            # CONFIDENCE
            

            if probabilities is not None:

                confidence = (
                    np.max(probabilities) * 100
                )

            else:

                confidence = 0


          # RESULT
        
            st.divider()

            st.subheader(
                "Prediction Result"
            )


            st.write(
                f" **Game:** {selected_game}"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Predicted Sentiment",
                    f"{"sentiment_icon"} {sentiment}"
                )


            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


            
            # MESSAGE
            

            if sentiment == "Positive":

                st.success(
                    f" This is a **Positive** review "
                    f"for **{selected_game}**."
                )

            elif sentiment == "Negative":

                st.error(
                    f" This is a **Negative** review "
                    f"for **{selected_game}**."
                )


            
            # PROBABILITY
            

            if probabilities is not None:

                st.subheader(
                    " Sentiment Probability"
                )


                probability_data = []


                for class_value, probability in zip(
                    model.classes_,
                    probabilities
                ):

                    if class_value == 1:

                        label = "Positive"

                    elif class_value == -1:

                        label = "Negative"

                    else:

                        label = str(class_value)


                    probability_data.append({

                        "Sentiment": label,

                        "Probability": round(
                            probability * 100,
                            2
                        )

                    })


                probability_df = pd.DataFrame(
                    probability_data
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True
                    )


                with col2:

                    fig = px.bar(
                        probability_df,
                        x="Sentiment",
                        y="Probability",
                        text="Probability",
                        title="Prediction Probability"
                    )


                    fig.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside"
                    )


                    fig.update_layout(
                        yaxis_title="Probability (%)",
                        xaxis_title="Sentiment",
                        yaxis_range=[0, 100],
                        height=350
                    )


                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


            
            # PREPROCESSED REVIEW
            

            with st.expander(
                " View Preprocessed Review"
            ):

                st.write(
                    processed_review
                )


            
            # WORD CONTRIBUTION
            

            if hasattr(
                model,
                "coef_"
            ):

                st.divider()

                st.subheader(
                    "Words Affecting the Prediction"
                )


                feature_names = np.array(
                    tfidf.get_feature_names_out()
                )


                input_values = (
                    X_input.toarray()[0]
                )


                coefficients = model.coef_[0]


                contributions = (
                    input_values *
                    coefficients
                )


                indices = np.where(
                    input_values > 0
                )[0]


                if len(indices) == 0:

                    st.info(
                        " None of the words in this review "
                        "were recognized by the trained TF-IDF vocabulary."
                    )

                else:

                    contribution_data = []


                    for index in indices:

                        contribution = (
                            contributions[index]
                        )


                        if contribution > 0:

                            direction = "Positive"

                        else:

                            direction = "Negative"


                        contribution_data.append({

                            "Word":
                                feature_names[index],

                            "Contribution":
                                contribution,

                            "Direction":
                                direction

                        })


                    contribution_df = pd.DataFrame(
                        contribution_data
                    )


                    contribution_df["Impact"] = (
                        contribution_df[
                            "Contribution"
                        ].abs()
                    )


                    contribution_df = (
                        contribution_df
                        .sort_values(
                            "Impact",
                            ascending=False
                        )
                        .head(15)
                    )


                    
                    # CHART
                    

                    chart_df = (
                        contribution_df
                        .sort_values(
                            "Contribution"
                        )
                    )


                    fig = px.bar(
                        chart_df,
                        x="Contribution",
                        y="Word",
                        orientation="h",
                        color="Direction",
                        title="Top Words Influencing the Prediction"
                    )


                    fig.update_layout(
                        height=600,
                        xaxis_title="Contribution",
                        yaxis_title="Word"
                    )


                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


                    
                    # TABLE
                    

                    st.subheader(
                        "Word Contribution Details"
                    )


                    display_df = (
                        contribution_df[
                            [
                                "Word",
                                "Contribution",
                                "Direction"
                            ]
                        ]
                        .sort_values(
                            "Contribution",
                            ascending=False
                        )
                    )


                    display_df[
                        "Contribution"
                    ] = (
                        display_df[
                            "Contribution"
                        ].round(4)
                    )


                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )


        except Exception as e:

            st.error(
                "Something went wrong during prediction."
            )

            st.exception(e)