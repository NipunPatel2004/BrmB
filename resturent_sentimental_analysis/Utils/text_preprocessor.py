"""
TextPreprocessor

This class must exist somewhere importable at unpickling time because it was
used as a step inside the scikit-learn Pipeline that was saved to
Models/sentiment_pipeline.pkl. It was originally defined inside the training
notebook (Untitled1.ipynb) where it lived in the `__main__` module. When the
pickle was created there, the reference stored inside the pickle became
`__main__.TextPreprocessor`. Streamlit's page scripts are not that same
`__main__` module, so a plain `pickle.load(...)` fails with:

    AttributeError: Can't get attribute 'TextPreprocessor' on <module '__main__' ...>

Keeping the class definition here (byte-for-byte the same as training) and
registering it onto `__main__` in Utils/loader.py before unpickling fixes
the error without needing to retrain or re-pickle the model.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


def _ensure_nltk_data():
    """Download the small NLTK corpora this class needs, if missing."""
    required = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, package in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()


class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.stop_words = set(
            stopwords.words("english")
        )

        negation_words = {
            "not",
            "no",
            "nor",
            "never"
        }

        self.stop_words = (
            self.stop_words
            - negation_words
        )

        self.lemmatizer = (
            WordNetLemmatizer()
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        cleaned_reviews = []

        for text in X:

            text = str(text)

            text = text.lower()

            text = re.sub(
                r"<.*?>",
                " ",
                text
            )

            text = re.sub(
                r"http\S+|www\S+|https\S+",
                " ",
                text
            )

            text = text.encode(
                "ascii",
                "ignore"
            ).decode("ascii")

            text = re.sub(
                r"\d+",
                " ",
                text
            )

            text = text.translate(
                str.maketrans(
                    "",
                    "",
                    string.punctuation
                )
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            tokens = word_tokenize(text)

            tokens = [
                word
                for word in tokens
                if word not in self.stop_words
            ]

            tokens = [
                self.lemmatizer.lemmatize(
                    word,
                    pos="v"
                )
                for word in tokens
            ]

            tokens = [
                word
                for word in tokens
                if len(word) > 2
            ]

            cleaned_reviews.append(
                " ".join(tokens)
            )

        return cleaned_reviews
