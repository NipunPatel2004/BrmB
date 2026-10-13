import pickle
import os


from pathlib import Path


def load_models():

    base_path = Path(__file__).parent.parent / "models"


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