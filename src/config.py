MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128
NUM_LABELS = 5  # lớp dự đoán từ 0 đến 4 : tổng cộng 5 lớp

TRAIN_PATH = "data/processed/train_uit_vsmec_clean.csv"
VAL_PATH = "data/processed/val_uit_vsmec_clean.csv"
TEST_PATH = "data/processed/test_uit_vsmec_clean.csv"

RAW_TRAIN_PATH = "data/raw/train_uit_vsmec.csv"
RAW_VAL_PATH = "data/raw/val_uit_vsmec.csv"
RAW_TEST_PATH = "data/raw/test_uit_vsmec.csv"

MODEL_SAVE_PATH = "QuocChio/vn-negative-emotion-phobert"
OUTPUT_DIR = "outputs/phobert_training"