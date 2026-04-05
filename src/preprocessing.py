import pandas as pd
import re
import unicodedata
from pathlib import Path
from underthesea import word_tokenize
from src.config import RAW_TRAIN_PATH, RAW_VAL_PATH, RAW_TEST_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH


def normalize_emoji_and_teencode(text: str) -> str:
    text = str(text)

    emoji_map = {
        "😭": " emoji_buon ",
        "😢": " emoji_buon ",
        "😞": " emoji_buon ",
        "💔": " emoji_dau_long ",
        "😊": " emoji_vui ",
        "🙂": " emoji_vui ",
        "😂": " emoji_cuoi ",
        ":)": " emoji_vui ",
        ":(": " emoji_buon ",
        "=))": " emoji_cuoi ",
        "^^": " emoji_vui ",
    }
    for k, v in emoji_map.items():
        text = text.replace(k, v)

    teencode_map = {
        " ko ": " không ",
        " k ": " không ",
        " hok ": " không ",
        " khum ": " không ",
        " dc ": " được ",
        " đc ": " được ",
    }
    text = f" {text} "
    for k, v in teencode_map.items():
        text = text.replace(k, v)

    return text.strip()


def attach_negation(text: str) -> str:
    patterns = [
        r"\b(không)\s+(\w+)",
        r"\b(chẳng)\s+(\w+)",
        r"\b(chưa)\s+(\w+)",
        r"\b(chả)\s+(\w+)",
        r"\b(đâu_có)\s+(\w+)",
    ]
    for pattern in patterns:
        text = re.sub(pattern, r"\1_\2", text)
    return text


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = unicodedata.normalize("NFKC", text)

    text = normalize_emoji_and_teencode(text)

    # giữ chữ, số, khoảng trắng và một số dấu cơ bản
    text = re.sub(r"[^\w\s!?.,]", " ", text)

    # rút ký tự lặp về 2 thay vì 1
    text = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ])\1{2,}", r"\1\1", text)

    text = re.sub(r"\s+", " ", text).strip().lower()

    text = word_tokenize(text, format="text")

    # gộp phủ định sau tokenize
    text = text.replace("đâu có", "đâu_có")
    text = attach_negation(text)

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Sentence" in df.columns:
        df = df.rename(columns={"Sentence": "text"})
    if "score" in df.columns:
        df = df.rename(columns={"score": "label"})

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)

    # drop duplicate sau clean
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df


def run_preprocessing():
    print("Đang đọc dữ liệu thô...")
    train_df = pd.read_csv(RAW_TRAIN_PATH)
    val_df = pd.read_csv(RAW_VAL_PATH)
    test_df = pd.read_csv(RAW_TEST_PATH)

    print("Đang làm sạch và tách từ...")
    train_clean = clean_dataframe(train_df)
    val_clean = clean_dataframe(val_df)
    test_clean = clean_dataframe(test_df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train_clean.to_csv(TRAIN_PATH, index=False)
    val_clean.to_csv(VAL_PATH, index=False)
    test_clean.to_csv(TEST_PATH, index=False)

    print("Đã lưu các file làm sạch:")
    print(f"- Train: {TRAIN_PATH} {train_clean.shape}")
    print(f"- Val:   {VAL_PATH} {val_clean.shape}")
    print(f"- Test:  {TEST_PATH} {test_clean.shape}")


if __name__ == "__main__":
    run_preprocessing()