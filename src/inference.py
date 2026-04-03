import torch
import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_SAVE_PATH, MAX_LENGTH

# làm sạch văn bản đầu vào
def clean_input_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)  # Load tokenizer 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.eval() # chuyển model sang chế độ đánh giá, tắt dropout đảm bảo model dự đoán ổn định hơn
    return tokenizer, model

# dự đoán nhãn cho một văn bản đầu vào
def predict_text(text: str, tokenizer, model):
    cleaned_text = clean_input_text(text)

    # tokenize 
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt", # Trả về dữ liệu dạng tensor PyTorch
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**inputs) # Đưa dữ liệu đã được tokenize vào model để nhận dự đoán
        probs = torch.softmax(outputs.logits, dim=-1)[0] # Chuyển logits thành xác suất bằng hàm softmax, lấy phần tử đầu tiên vì chỉ có một văn bản đầu vào
        pred = torch.argmax(probs).item() # Lấy nhãn có xác suất cao nhất làm dự đoán cuối cùng

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "predicted_label": pred,
        "probabilities": probs.cpu().numpy().tolist()
    }