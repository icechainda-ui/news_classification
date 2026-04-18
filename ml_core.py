import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import preprocess

MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoder.pkl"


def train_model(csv_path):
    df = pd.read_csv(csv_path)

    df = df.dropna()
    X = df["text"].astype(str)
    y = df["category"].astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess)),
        ("clf", LinearSVC())
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    return acc


def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder