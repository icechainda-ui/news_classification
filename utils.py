import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("russian")).union(set(stopwords.words("english")))


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text):
    text = clean_text(text)
    tokens = text.split()
    result = []

    for token in tokens:
        if token in stop_words or len(token) < 3:
            continue
        result.append(token)

    return " ".join(result)