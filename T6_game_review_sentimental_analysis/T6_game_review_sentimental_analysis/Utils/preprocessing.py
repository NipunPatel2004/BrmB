import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# # Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")   # Required for newer NLTK versions
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

def preprocessing(text):

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
    tokens = word_tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words("english"))

    # Keep negation words
    stop_words = stop_words - {
        "not",
        "no",
        "nor",
        "never"
    }

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(word, pos="v")
        for word in tokens
    ]

    # Remove short words
    tokens = [
        word for word in tokens
        if len(word) > 2
    ]

    # Join
    return " ".join(tokens)