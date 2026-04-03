# Phân loại bài đăng tiếng Việt theo mức độ cảm xúc tiêu cực

Ứng dụng Streamlit dùng mô hình PhoBERT để phân loại bài đăng tiếng Việt theo 5 mức độ cảm xúc tiêu cực:

- 0: Không/ít tiêu cực
- 1: Lo lắng/Sợ nhẹ
- 2: Buồn
- 3: Khó chịu/Phẫn nộ nhẹ
- 4: Tức giận mạnh

## Chạy local

~~~bash
pip install -r requirements.txt
streamlit run app.py
~~~
app.py
Là file chạy Streamlit. Đây là giao diện web để bạn nhập câu và nó trả ra cấp 0–4.

requirements.txt
Danh sách thư viện để máy khác hoặc Streamlit Cloud biết cần cài gì.

README.md
Mô tả project để nộp, đưa GitHub, hoặc giải thích cách chạy.

src/inference.py
Chứa code load model và dự đoán.
Tách riêng file này để sau này:

app dùng lại được
terminal dùng lại được
không nhét hết logic vào app.py

src/utils.py
Chứa hàm phụ như:

clean text đầu vào
map từ số 0–4 sang tên mức
map sang mức cảnh báo

src/__init__.py
Cho Python biết src là package. File này có thể để trống.

models/phobert_final/
Chứa model bạn đã fine-tune xong và tokenizer. Đây là phần quan trọng nhất để app chạy.

data/processed/
Chứa file clean. Không bắt buộc để app chạy, nhưng nên có để chứng minh pipeline dữ liệu.

#### 1. src/preprocessing.py
Chứa phần:

đọc file gốc
xóa trùng
clean text
lưu file clean

Tức là phần bạn đã làm trong notebook.

#### 2. src/dataset.py
Chứa phần:

đọc file clean
convert sang Hugging Face DatasetDict
tokenize bằng PhoBERT
Tức là đoạn từ Dataset.from_pandas(...) trở xuống.
Chuẩn bị dữ liệu cho PhoBERT:
đọc clean csv
Hugging Face Dataset
tokenize

src/trainer_utils.py
Chứa compute_metrics() như:

accuracy
macro_f1
weighted_f1
mae

Để file train gọn hơn.

#### 3.train.py
Là script train model, thay cho việc mở notebook để train.
Nó sẽ:

đọc dataset
tokenize
load PhoBERT
train
evaluate
save model vào models/phobert_final

predict.py
Là file chạy test trong terminal.
Ví dụ bạn gõ 1 câu, nó in ra cấp dự đoán luôn.