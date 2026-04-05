# Phân loại bài đăng tiếng Việt theo mức độ cảm xúc tiêu cực

Ứng dụng Streamlit dùng mô hình PhoBERT để phân loại bài đăng tiếng Việt theo 5 mức độ cảm xúc tiêu cực:
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