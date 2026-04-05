# Phân loại bài đăng tiếng Việt theo mức độ cảm xúc tiêu cực bằng Logistic Regression nhằm hỗ trợ cảnh báo sớm

Mục tiêu của bài toán là nhận diện mức độ cảm xúc tiêu cực từ nội dung văn bản tiếng Việt trên mạng xã hội, từ đó hỗ trợ theo dõi và phát hiện sớm các trường hợp có dấu hiệu bất ổn về cảm xúc.

## Mục tiêu bài toán
Đầu vào của hệ thống là một câu hoặc bài đăng tiếng Việt.  
Đầu ra là:
- **Mức độ cảm xúc tiêu cực**
- **Nhãn dự đoán**
- **Trạng thái cảnh báo**
- **Xác suất theo từng mức**

Các mức nhãn được chia từ mức ít hoặc không tiêu cực đến các mức tiêu cực cao hơn.

## Pipeline thực hiện
Quy trình xây dựng mô hình gồm các bước chính:

1. Chuẩn bị dữ liệu
2. Tiền xử lý văn bản:
* chuẩn hóa Unicode
* xóa ký tự ẩn
* chuẩn hóa khoảng trắng
* chuyển về chữ thường
* chuẩn hóa emoji và teencode
* loại bỏ giá trị thiếu
* xóa trùng lặp
* Tách từ tiếng Việt bằng **Underthesea**
* Xử lý phủ định:
   - `không_buồn`
   - `không_vui`
   - `không_ổn`
5. Trích xuất đặc trưng bằng **TF-IDF**
   - word n-gram
~~~
 word_tfidf = TfidfVectorizer(
       analyzer="word",
       ngram_range=(1, 3),
       min_df=1,
       max_df=0.95,
       sublinear_tf=True
   )
~~~
- character n-gram
~~~
   char_tfidf = TfidfVectorizer(
       analyzer="char_wb",
       ngram_range=(3, 5),
       min_df=1,
       sublinear_tf=True
   )
~~~
   
6. Huấn luyện mô hình **Logistic Regression**
~~~
    model = LogisticRegression(
        C=2.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    )
~~~
7. Đánh giá mô hình bằng:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Macro F1-score
   - Weighted F1-score
8. Phân tích lỗi bằng:
   - confusion matrix
   - các trường hợp dự đoán sai

Ứng dụng Streamlit dùng mô hình **Logistic Regression** để phân loại bài đăng tiếng Việt theo 5 mức độ cảm xúc tiêu cực:
~~~
- 0: Không/ít tiêu cực
- 1: Lo lắng/Sợ nhẹ
- 2: Buồn
- 3: Khó chịu/Phẫn nộ nhẹ
- 4: Tức giận mạnh
~~~

### Cấu trúc thư mục

~~~
VN-NEGATIVE-EMOTION-ALERT/
├── .venv/                         # Môi trường ảo Python
├── data/
│   ├── processed/                 # Dữ liệu sau tiền xử lý
│   │   ├── train_uit_vsmec_clean.csv
│   │   ├── val_uit_vsmec_clean.csv
│   │   └── test_uit_vsmec_clean.csv
│   └── raw/                       # Dữ liệu gốc chưa xử lý
│       ├── train_uit_vsmec.csv
│       ├── val_uit_vsmec.csv
│       └── test_uit_vsmec.csv
├── models/
│   └── ml_version/                # Nơi lưu model và vectorizer đã train
│       ├── lr_model.pkl
│       └── tfidf_vectorizer.pkl
├── src/
│   ├── __pycache__/               # File cache do Python tự sinh
│   ├── __init__.py                # Đánh dấu thư mục src là package
│   ├── config.py                  # Khai báo đường dẫn file, tham số cấu hình
│   ├── dataset.py                 # Xử lý đọc/chia dữ liệu nếu cần
│   ├── inference.py               # Load model và dự đoán văn bản mới
│   ├── predict.py                 # Chạy dự đoán từ terminal
│   ├── preprocessing.py           # Làm sạch dữ liệu, tách từ, xử lý phủ định/emoji
│   ├── train.py                   # Huấn luyện mô hình Logistic Regression
│   ├── trainer_utils.py           # Các hàm hỗ trợ train/đánh giá
│   └── utils.py                   # Các hàm tiện ích như map nhãn, trạng thái cảnh báo
├── .gitignore                     # Khai báo file/thư mục không đẩy lên Git
├── app.py                         # Ứng dụng Streamlit giao diện web
├── README.md                      # Tài liệu mô tả dự án
├── requirements.txt               # Danh sách thư viện cần cài đặt
└── sodo.md                        # Ghi chú/sơ đồ mô tả dự án (nếu có)
~~~