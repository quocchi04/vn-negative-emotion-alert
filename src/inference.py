import joblib

try:
    from src.config import MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH
    from src.preprocessing import clean_text
except ModuleNotFoundError:
    from config import MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH
    from preprocessing import clean_text


try:
    GLOBAL_VECTORIZER = joblib.load(VECTORIZER_SAVE_PATH)
    GLOBAL_MODEL = joblib.load(MODEL_SAVE_PATH)
except Exception:
    GLOBAL_VECTORIZER = None
    GLOBAL_MODEL = None


def load_model():
    return GLOBAL_VECTORIZER, GLOBAL_MODEL


def predict_text(text, vectorizer=None, model=None):
    v = vectorizer if vectorizer is not None else GLOBAL_VECTORIZER
    m = model if model is not None else GLOBAL_MODEL

    if m is None or v is None:
        raise RuntimeError("Model chưa được load. Hãy train lại hoặc kiểm tra đường dẫn.")

    cleaned = clean_text(text)
    x = v.transform([cleaned])

    label = int(m.predict(x)[0])

    if hasattr(m, "predict_proba"):
        probs = m.predict_proba(x)[0].tolist()
        confidence = float(max(probs))
    else:
        probs = None
        confidence = None

    return {
        "predicted_label": label,
        "probabilities": probs,
        "confidence": confidence,
        "cleaned_text": cleaned
    }