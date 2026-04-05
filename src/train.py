import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import TRAIN_PATH, VAL_PATH, MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH


def build_vectorizer():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True
    )

    return FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print("\n===== LogisticRegression =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    return acc, macro_f1


def main():
    print("Dang doc du lieu...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train_text = train_df["text"].astype(str)
    y_train = train_df["label"]

    X_val_text = val_df["text"].astype(str)
    y_val = val_df["label"]

    print("Dang build vectorizer...")
    vectorizer = build_vectorizer()

    print("Dang fit vectorizer...")
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    model = LogisticRegression(
        C=2.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    )

    print("Dang train LogisticRegression...")
    model.fit(X_train, y_train)

    evaluate_model(model, X_val, y_val)
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(VECTORIZER_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    joblib.dump(model, MODEL_SAVE_PATH)

    print("\nDa luu model va vectorizer.")


if __name__ == "__main__":
    main()