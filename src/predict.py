from src.inference import load_model, predict_text
from src.utils import get_label_name, get_risk_level

def main():
    tokenizer, model = load_model()

    while True:
        text = input("\nNhập câu (gõ 'exit' để thoát): ")
        if text.lower() == "exit":
            break

        result = predict_text(text, tokenizer, model)
        pred = result["predicted_label"]

        print("\n=== Kết quả ===")
        print("Câu gốc:", result["original_text"])
        print("Sau clean:", result["cleaned_text"])
        print("Cấp:", pred)
        print("Nhãn:", get_label_name(pred))
        print("Cảnh báo:", get_risk_level(pred))
        print("Xác suất:", result["probabilities"])

if __name__ == "__main__":
    main()