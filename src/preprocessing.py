import pandas as pd
import re
import unicodedata
from pathlib import Path
from src.config import RAW_TRAIN_PATH, RAW_VAL_PATH, RAW_TEST_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH

# Làm sạch 1 câu
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  #xóa các ký tự ẩn
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["Sentence"]).reset_index(drop=True)     #xóa các dòng trùng lặp
    df["Sentence"] = df["Sentence"].apply(clean_text)
    return df

# Chạy toàn bộ quy trình làm sạch và lưu kết quả
def run_preprocessing():
    train_df = pd.read_csv(RAW_TRAIN_PATH)
    val_df = pd.read_csv(RAW_VAL_PATH)
    test_df = pd.read_csv(RAW_TEST_PATH)

    train_clean = clean_dataframe(train_df)
    val_clean = clean_dataframe(val_df)
    test_clean = clean_dataframe(test_df)

    #Lưu các file đã làm sạch:
    Path("data/processed").mkdir(parents=True, exist_ok=True)   

    train_clean.to_csv(TRAIN_PATH, index=False)
    val_clean.to_csv(VAL_PATH, index=False)
    test_clean.to_csv(TEST_PATH, index=False)

    print("Saved cleaned files:")
    print(TRAIN_PATH, train_clean.shape)
    print(VAL_PATH, val_clean.shape)
    print(TEST_PATH, test_clean.shape)

if __name__ == "__main__":
    run_preprocessing()