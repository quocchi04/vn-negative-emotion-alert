import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from src.config import MODEL_NAME, MAX_LENGTH, TRAIN_PATH, VAL_PATH, TEST_PATH

def load_dataframes():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df

def build_dataset_dict():
    train_df, val_df, test_df = load_dataframes()

    # Chuyển từng DataFrame sang Hugging Face Dataset
    train_hf = Dataset.from_pandas(train_df[["Sentence", "score"]])
    val_hf = Dataset.from_pandas(val_df[["Sentence", "score"]])
    test_hf = Dataset.from_pandas(test_df[["Sentence", "score"]])

    dataset = DatasetDict({
        "train": train_hf,
        "validation": val_hf,
        "test": test_hf
    })
    return dataset

def tokenize_dataset(dataset):     #Mục tiêu: biến text thành dạng số mà PhoBERT hiểu được
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["Sentence"],
            truncation=True,        # Nếu câu dài hơn MAX_LENGTH thì cắt bớt
            padding="max_length",   # Tất cả câu sẽ được thêm padding để có cùng độ dài MAX_LENGTH
            max_length=MAX_LENGTH   # 128 
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["Sentence"])      #xóa cột "Sentence" vì sau khi token hóa không cần cột này nữa
    tokenized_dataset = tokenized_dataset.rename_column("score", "labels")  #đổi tên cột "score" thành "labels"
    
    # chuyển dataset sang định dạng PyTorch
    # Các mẫu dữ liệu sẽ được trả về dưới dạng tensor, giúp cho việc xử lý và tính toán hiệu quả
    tokenized_dataset.set_format("torch")

    return tokenized_dataset, tokenizer