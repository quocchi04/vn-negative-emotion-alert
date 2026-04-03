from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from src.config import MODEL_NAME, NUM_LABELS, MODEL_SAVE_PATH, OUTPUT_DIR
from src.dataset import build_dataset_dict, tokenize_dataset
from src.trainer_utils import compute_metrics

def main():
    dataset = build_dataset_dict()
    tokenized_dataset, tokenizer = tokenize_dataset(dataset) 

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS   # số lớp
    )

    # Tạo thư mục nếu chưa tồn tại
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",  #Sau mỗi epoch, model sẽ evaluate trên validation set
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,
        num_train_epochs=3,             # model đi qua toàn bộ tập train 3 lần
        weight_decay=0.01,              # giúp giảm overfitting bằng cách phạt các trọng số lớn
        load_best_model_at_end=True,    # train xong -> load lại model tốt nhất dựa trên macro_f1
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2, # chỉ giữ lại 2 checkpoint gần nhất để tiết kiệm dung lượng lưu trữ
        report_to="none"
    )

    trainer = Trainer(    # quản lý toàn bộ quá trình train/evaluate 
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],   # dùng để đánh giá sau mỗi epoch và chọn model tốt nhất
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("\n=== Evaluate on test ===")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(test_results)

    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Saved model to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
