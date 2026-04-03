import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from src.inference import predict_text

def compute_metrics(eval_pred):
    logits, labels = eval_pred # logits là đầu ra của model, labels là nhãn thật
    preds = np.argmax(logits, axis=-1) #Lấy vị trí có giá trị lớn nhất trong logits, đó chính là lớp model dự đoán

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "mae": mean_absolute_error(labels, preds)
    }
def evaluate_model_on_test(model, tokenizer, test_df):
    y_true = test_df['score'].tolist()
    y_pred = []
    
    # Chạy dự đoán cho từng dòng trong tập test
    for text in test_df['Sentence']:
        res = predict_text(text, tokenizer, model)
        y_pred.append(int(res['predicted_label']))
    
    # Tính toán các chỉ số
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "cm": confusion_matrix(y_true, y_pred)
    }
    return metrics, y_true, y_pred