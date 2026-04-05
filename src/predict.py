from src.inference import predict_text
from src.utils import get_label_name, get_risk_level


def main():
    print("🤖 Đang khởi động công cụ dự đoán (Phiên bản Machine Learning)...")

    while True:
        text = input("\nNhập câu (gõ 'exit' để thoát): ")

        if text.strip().lower() == "exit":
            print("Tạm biệt!")
            break

        if not text.strip():
            continue

        try:
            res = predict_text(text)

            label = res["predicted_label"]
            confidence = res["confidence"] if res["confidence"] is not None else 0.0

            print(f"👉 [Debug của Dev] Câu mô hình thực tế đọc: '{res['cleaned_text']}'")

            print("\n=== Kết quả ===")
            print(f"Câu gốc:    {text}")
            print(f"Câu sạch:   {res['cleaned_text']}")
            print(f"Cấp (ID):   {label}")
            print(f"Nhãn:       {get_label_name(label)}")
            print(f"Cảnh báo:   {get_risk_level(label)}")
            print(f"Độ tin cậy: {confidence:.2%}")

            if res["probabilities"] is not None:
                print(f"Xác suất:   {res['probabilities']}")

        except Exception as e:
            print(f"❌ Lỗi dự đoán: {e}")


if __name__ == "__main__":
    main()