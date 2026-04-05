import pandas as pd
from src.config import TRAIN_PATH, VAL_PATH, TEST_PATH

def build_dataset_dict():
    """
    Đọc các file CSV đã làm sạch và trả về Dictionary 
    chứa text và label dưới dạng list (danh sách) chuẩn cho Machine Learning.
    """
    # 1. Đọc dữ liệu
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2. Xóa các dòng rỗng (Phòng hờ lỗi sau khi preprocessing)
    train_df = train_df.dropna(subset=['text', 'label'])
    val_df = val_df.dropna(subset=['text', 'label'])
    test_df = test_df.dropna(subset=['text', 'label'])

    # 3. Trả về cấu trúc Dictionary đơn giản
    dataset = {
        "train": {
            "text": train_df["text"].tolist(),
            "label": train_df["label"].tolist()
        },
        "validation": {
            "text": val_df["text"].tolist(),
            "label": val_df["label"].tolist()
        },
        "test": {
            "text": test_df["text"].tolist(),
            "label": test_df["label"].tolist()
        }
    }
    
    return dataset