import pickle
import os


def load_models():

    base_path = "Models"


    # TF-IDF
    with open(
        os.path.join(
            base_path,
            "tfidf_vectorizer.pkl"
        ),
        "rb"
    ) as file:

        tfidf = pickle.load(file)


    # Model
    with open(
        os.path.join(
            base_path,
            "sentiment_model.pkl"
        ),
        "rb"
    ) as file:

        model = pickle.load(file)


    return tfidf, model