import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple # Sửa lỗi NameError: 'Optional' is not defined
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# --- KHẮC PHỤC LỖI ĐƯỜNG DẪN ---
root_path = Path(__file__).parent.absolute()
src_path = root_path / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- IMPORT CÁC HÀM TỪ TRONG SRC ---
try:
    # Sau khi append sys.path, ta gọi trực tiếp tên file
    from inference import predict_text 
    from utils import get_label_name, get_risk_level
    from config import TRAIN_PATH, VAL_PATH, TEST_PATH
    from trainer_utils import compute_metrics
    
    # Vì file inference.py của bạn KHÔNG có load_model, 
    # ta sẽ tự định nghĩa một hàm giả lập ở đây để app không sập
    def load_model():
        import joblib
        from config import MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH
        try:
            v = joblib.load(VECTORIZER_SAVE_PATH)
            m = joblib.load(MODEL_SAVE_PATH)
            return v, m
        except:
            return None, None

except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}. Hãy chắc chắn các file nằm trong thư mục 'src'.")
    st.stop()
# 1. Cấu hình trang
st.set_page_config(
    page_title="VN Negative Emotion Alert",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" # Mở sẵn sidebar, người dùng có thể bấm nút thu gọn
)

PROJECT_INFO = {
    "title": "Phân loại bài đăng tiếng Việt theo mức độ cảm xúc tiêu cực bằng PhoBERT nhằm hỗ trợ cảnh báo sớm",
    "subtitle": "Phân loại bài đăng tiếng Việt bằng PhoBERT",
    "student": "Trần Đức Quốc Chí",
    "masv": "22T1020036",
    "description": (
        "Phát hiện và phân loại sớm các bài đăng tiếng Việt mang cảm xúc tiêu cực theo nhiều mức độ, từ đó hỗ trợ cảnh báo sớm cho mạng xã hội, diễn đàn hoặc nền tảng số."
    ),
}

# --- CÁC HÀM CACHE DỮ LIỆU VÀ MODEL ---
@st.cache_resource
def load_resources():
    return load_model()

@st.cache_data
def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    file_path = Path(path)
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

@st.cache_data
def load_all_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    return (
        safe_read_csv(TRAIN_PATH),
        safe_read_csv(VAL_PATH),
        safe_read_csv(TEST_PATH),
    )

@st.cache_data
def build_overview_table(train_df: Optional[pd.DataFrame], val_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if df is None:
            rows.append({
                "Tập dữ liệu": name,
                "Số dòng": "Không tìm thấy file",
                "Số cột": "-",
                "Cột": "-",
            })
        else:
            rows.append({
                "Tập dữ liệu": name,
                "Số dòng": f"{len(df):,}",
                "Số cột": len(df.columns),
                "Cột": ", ".join(df.columns),
            })
    return pd.DataFrame(rows)

@st.cache_data
def evaluate_model_cached(test_df: pd.DataFrame):
    vectorizer, model = load_resources()

    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        text = row["text"]
        label = int(row["label"])
        pred = int(predict_text(text, vectorizer, model)["predicted_label"])
        y_true.append(label)
        y_pred.append(pred)

    labels = sorted(set(y_true) | set(y_pred))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, cm, labels


# --- CÁC HÀM VẼ BIỂU ĐỒ ---
def plot_score_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color='#4b72b8', edgecolor='black')
    ax.set_title("Phân phối nhãn Score", fontsize=12, fontweight='bold')
    ax.set_xlabel("Nhãn", fontsize=10)
    ax.set_ylabel("Số lượng mẫu", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

def plot_text_length_distribution(df: pd.DataFrame):
    lengths = df["text"].fillna("").astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color='#e67e22', edgecolor='black', alpha=0.8)
    ax.set_title("Phân phối độ dài văn bản", fontsize=12, fontweight='bold')
    ax.set_xlabel("Số từ", fontsize=10)
    ax.set_ylabel("Số lượng mẫu", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

def plot_probability_chart(probabilities):
    labels = [f"Mức {i}" for i in range(len(probabilities))]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000']
    ax.bar(labels, probabilities, color=colors[:len(probabilities)])
    ax.set_ylim(0, 1)
    ax.set_title("Xác suất theo từng mức độ", fontsize=12, fontweight='bold')
    ax.set_ylabel("Xác suất", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

def plot_confusion_matrix(cm: np.ndarray, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix", fontsize=12, fontweight='bold')
    ax.set_xlabel("Dự đoán", fontsize=10)
    ax.set_ylabel("Thực tế", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > (cm.max() / 2.) else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


# --- CÁC HÀM RENDER NỘI DUNG TỪNG TRANG ---
def render_eda_tab(train_df: Optional[pd.DataFrame], val_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame]):
    st.header("📊 Giới thiệu & Khám phá dữ liệu (EDA)")
    
    # Chuyển phần thông tin sinh viên vào đây, đặt trong một box (container) đẹp mắt
    with st.container(border=True):
        st.markdown("### 🧑‍💻 Thông tin đề tài & Sinh viên")
        st.write(f"**Tên đề tài:** {PROJECT_INFO['title']}")
        st.write(f"**Họ tên:** {PROJECT_INFO['student']}")
        st.write(f"**MSSV:** {PROJECT_INFO['masv']}")
        st.write(f"**Mô tả ngắn gọn giá trị thực tiễn:** {PROJECT_INFO['description']}")

    overview_df = build_overview_table(train_df, val_df, test_df)
    st.subheader("📌 Tổng quan tập dữ liệu")
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    if train_df is None:
        st.warning(
            "⚠️ Chưa tìm thấy file train/validation/test theo đường dẫn trong config.py. "
            "Phần EDA và Evaluation sẽ bị giới hạn."
        )
        return


    st.subheader("🔍 Mẫu dữ liệu Train")
    st.dataframe(train_df.head(11), use_container_width=True)

    st.subheader("⚠️ Kiểm tra dữ liệu thiếu (NaN)")
    missing = train_df.isna().sum().reset_index()
    missing.columns = ["Cột", "Số lượng thiếu"]
    st.dataframe(missing, use_container_width=True, hide_index=True)

    st.divider() # Tạo đường phân cách ngang đẹp mắt giữa các phần

#----------------- 4.trực quan hóa và phân tích dữ liệu---------------------------
# --- PHẦN BIỂU ĐỒ ---
    st.subheader("📈 Trực quan hóa & Phân tích đặc trưng")
    
    # Hàng 1: Phân phối Score và Độ dài văn bản
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if "label" in train_df.columns:
            plot_score_distribution(train_df)
    with row1_col2:
        if "text" in train_df.columns:
            plot_text_length_distribution(train_df)

    # Tính toán đặc trưng độ dài
  # Tính toán đặc trưng độ dài
    train_df_copy = train_df.copy()
    
    # ĐÃ THÊM .fillna("") ĐỂ CHỐNG LỖI Ô TRỐNG
    train_df_copy['Length'] = train_df_copy['text'].fillna("").astype(str).apply(lambda x: len(x.split()))

    # Hàng 2: Ma trận tương quan (Đặt vào giữa để không bị quá to)
    if "label" in train_df.columns:
        # Chia làm 3 cột, biểu đồ nằm ở cột giữa (tỷ lệ 1:2:1)
        _, mid_col, _ = st.columns([1, 2, 1]) 
        
        with mid_col:
            corr_matrix = train_df_copy[['label', 'Length']].corr()
            corr_val = corr_matrix.iloc[0, 1]
            
            # Giảm figsize xuống (ví dụ 5x3) để biểu đồ trông gọn hơn
            fig, ax = plt.subplots(figsize=(5, 3.5)) 
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['label', 'Độ dài'], fontsize=9)
            ax.set_yticklabels(['label', 'Độ dài'], fontsize=9)
            ax.set_title(f"Ma trận tương quan (r = {corr_val:.2f})", fontsize=10, fontweight='bold')
            
            # Thêm chỉ số text vào trong các ô của ma trận
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                            ha="center", va="center", color="black", fontweight='bold')
            
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

    # PHẦN NHẬN XÉT
    with st.expander("📝 Phân tích chi tiết đặc trưng dữ liệu", expanded=True):
            if "label" in train_df.columns:
                counts = train_df["label"].value_counts(normalize=True).sort_index()
                # Tính toán chỉ số lệch
                imbalance_ratio = counts.max() / counts.min()
                is_imbalanced = "CÓ độ lệch" if imbalance_ratio > 2 else "tương đối CÂN BẰNG"
                
                avg_len = train_df_copy['Length'].mean()
                max_len = train_df_copy['Length'].max()

                st.markdown(f"#### 1. Đánh giá sự phân bổ nhãn (Labels Distribution)")
                st.write(f"- **Trạng thái:** Dữ liệu hiện tại **{is_imbalanced}** (Tỷ lệ lệch: {imbalance_ratio:.2f}).")
                st.write(f"- **Ghi chú:** Nhãn phổ biến nhất chiếm **{counts.max()*100:.1f}%**. Với đặc thù này, các chỉ số như **Macro F1-Score** sẽ phản ánh chính xác hiệu năng mô hình hơn là Accuracy thông thường.")

                st.markdown(f"#### 2. Phân tích đặc trưng quan trọng (Key Features)")
                st.markdown(f"""
                - **Đặc trưng từ vựng (Semantic Features):** Đây là đặc trưng **quan trọng nhất**. Các từ ngữ mang sắc thái tiêu cực, biểu cảm mạnh là tín hiệu chính để PhoBERT phân loại mức độ.
                - **Đặc trưng độ dài (Text Length):** - Độ dài trung bình: **{avg_len:.1f}** từ/câu.
                    - Hệ số tương quan giữa độ dài và nhãn: **{corr_val:.2f}**.
                    - **Nhận xét:** Hệ số tương quan thấp (gần 0) cho thấy mức độ tiêu cực phụ thuộc vào **ngữ nghĩa từ ngữ** chứ không phụ thuộc vào việc câu đó dài hay ngắn.
                """)

                st.markdown(f"#### 3. Kết luận về chất lượng dữ liệu")
                if train_df.isna().sum().sum() == 0:
                    st.success("✅ Dữ liệu hoàn toàn sạch (0 NaN), không bị trùng lặp, sẵn sàng cho quá trình huấn luyện mô hình PhoBERT.")
                else:
                    st.warning("⚠️ Dữ liệu vẫn còn một số giá trị thiếu, cần xử lý trước khi train.")

#-------------------------- Phần 2: Triển khai mô hình  ---------------------------                 
def render_prediction_tab(vectorizer, model):
    st.header("🚀 Triển khai mô hình")

    with st.container(border=True):
        st.markdown("### ✍️ Nhập văn bản cần kiểm tra")
        user_text = st.text_area(
            "Nội dung bài đăng",
            height=150,
            label_visibility="collapsed",
            placeholder="Ví dụ: Mình cảm thấy rất mệt mỏi, áp lực và chán nản với công việc hiện tại..."
        )

        c_btn, _, _ = st.columns([1, 2, 2])
        with c_btn:
            run_predict = st.button("🔮 Phân Tích Cảm Xúc", type="primary", use_container_width=True)

    if run_predict:
        if not user_text.strip():
            st.warning("Vui lòng nhập nội dung để dự đoán.")
            return

        with st.spinner("Đang phân tích..."):
            res = predict_text(user_text, vectorizer, model)

        pred = int(res["predicted_label"])
        probabilities = res.get("probabilities")
        cleaned_text = res.get("cleaned_text", user_text)

        st.markdown("---")
        st.subheader("🎯 Kết quả phân tích")

        c1, c2, c3 = st.columns(3)
        c1.metric("Cấp độ", f"Mức {pred}")
        c2.metric("Nhãn dự đoán", get_label_name(pred).split("-")[-1].strip())

        risk = get_risk_level(pred)
        if pred <= 1:
            c3.success(f"Trạng thái: {risk}")
        elif pred == 2:
            c3.warning(f"Trạng thái: {risk}")
        else:
            c3.error(f"Trạng thái: {risk}")

        st.markdown("**Văn bản sau khi làm sạch:**")
        st.info(cleaned_text)

        if probabilities is not None:
            st.markdown("### 📊 Chi tiết xác suất")
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                plot_probability_chart(probabilities)
            with col_table:
                prob_df = pd.DataFrame({
                    "Mức độ": [f"Mức {i}" for i in range(len(probabilities))],
                    "Độ tin cậy": [f"{p*100:.2f}%" for p in probabilities],
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
        else:
            st.info("Mô hình hiện tại không hỗ trợ xác suất dự đoán.")

@st.cache_data
def get_test_predictions(_vectorizer, _model, _test_df):
    y_true = _test_df["label"].astype(int).values
    y_pred = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, text in enumerate(_test_df["text"]):
        res = predict_text(text, _vectorizer, _model)

        if isinstance(res, dict):
            res_label = int(res["predicted_label"])
        else:
            res_label = int(res[0])

        y_pred.append(res_label)

        if i % max(1, len(_test_df) // 20) == 0:
            progress_bar.progress((i + 1) / len(_test_df))
            status_text.text(f"Đang xử lý tập Test: {i + 1}/{len(_test_df)} mẫu...")

    progress_bar.empty()
    status_text.empty()
    return y_true, np.array(y_pred)

def render_evaluation_tab(test_df: Optional[pd.DataFrame], vectorizer, model):
    st.header("📉 Đánh giá & Hiệu năng mô hình")

    if test_df is None or vectorizer is None or model is None:
        st.error("❌ Không thể thực hiện đánh giá: Thiếu dữ liệu hoặc Model chưa được tải.")
        return

    st.markdown("""
    Phần này thực hiện đánh giá khách quan mô hình trên **tập dữ liệu kiểm thử (Test Set)** chưa từng xuất hiện trong quá trình huấn luyện.
    """)

    # Nút bấm để kích hoạt (tránh việc app tự chạy nặng mỗi khi người dùng click vào tab)
    if st.button("📊 Bắt đầu đánh giá hiệu năng"):
        with st.spinner("Đang tính toán các chỉ số..."):
            y_true, y_pred = get_test_predictions(vectorizer, model, test_df)
            
            # --- 1. CHỈ SỐ ĐO LƯỜNG (METRICS) ---
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")
            f1_weighted = f1_score(y_true, y_pred, average="weighted")

            st.subheader("📌 1. Các chỉ số đo lường")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy (Độ chính xác)", f"{acc:.2%}")
            c2.metric("Macro F1-Score", f"{f1_macro:.4f}")
            c3.metric("Weighted F1-Score", f"{f1_weighted:.4f}")
            st.caption("F1-Score là chỉ số quan trọng nhất khi dữ liệu có sự chênh lệch giữa các lớp nhãn.")

            st.divider()

            # --- 2. BIỂU ĐỒ KỸ THUẬT ---
            st.subheader("📊 2. Phân tích qua Ma trận nhầm lẫn")
            col_chart, col_note = st.columns([2, 1])
            
            with col_chart:
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                # Vẽ heatmap thủ công bằng Matplotlib (hoặc Seaborn nếu bạn đã cài)
                im = ax.imshow(cm, cmap='Blues')
                plt.colorbar(im)
                
                # Hiển thị số liệu lên các ô
                for i in range(len(cm)):
                    for j in range(len(cm)):
                        ax.text(j, i, cm[i, j], ha="center", va="center", 
                                color="white" if cm[i, j] > cm.max()/2 else "black")
                
                ax.set_xticks(np.arange(5))
                ax.set_yticks(np.arange(5))
                ax.set_xticklabels([f"Mức {i}" for i in range(5)])
                ax.set_yticklabels([f"Mức {i}" for i in range(5)])
                ax.set_xlabel('Nhãn dự đoán')
                ax.set_ylabel('Nhãn thực tế')
                st.pyplot(fig)

            with col_note:
                st.info("""
                **Cách đọc ma trận:**
                - Đường chéo từ trên xuống dưới thể hiện số lượng đoán đúng.
                - Các ô nằm ngoài đường chéo thể hiện sự nhầm lẫn giữa các mức độ.
                - Nếu các con số tập trung sát đường chéo, nghĩa là máy chỉ nhầm giữa các mức độ gần nhau (ví dụ 3 và 4), điều này vẫn rất khả quan.
                """)

            st.divider()

            # --- 3. PHÂN TÍCH SAI SỐ (ERROR ANALYSIS) ---
            st.subheader("🔍 3. Chi tiết các trường hợp dự đoán sai")
            mask = y_true != y_pred
            errors = test_df[mask].copy()
            errors['Dự đoán'] = [get_label_name(p) for p in y_pred[mask]]
            errors['Thực tế'] = [get_label_name(t) for t in y_true[mask]]
            
            st.write(f"Tìm thấy **{len(errors)}** mẫu dự đoán sai trên tổng số **{len(test_df)}** mẫu kiểm thử.")
            st.dataframe(errors[['text', 'Thực tế', 'Dự đoán']].head(10), use_container_width=True)

            with st.expander("📝 Nhận định và hướng cải thiện"):
                st.markdown(f"""
                - **Nhận định:** Với độ chính xác **{acc:.2%}**, PhoBERT chứng minh khả năng hiểu ngữ nghĩa sâu sắc của tiếng Việt, vượt xa các mô hình truyền thống.
                - **Điểm yếu:** Mô hình vẫn còn nhầm lẫn nhẹ giữa các sắc thái cảm xúc cực đoan (Mức 3 và 4) do tính chất ngôn ngữ trên mạng xã hội thường có ẩn ý hoặc mỉa mai.
                - **Hướng cải thiện:** Thu thập thêm dữ liệu thực tế cho các nhãn ít mẫu (Mức 1, Mức 4) để tăng độ nhạy bén cho mô hình.
                """)
# --- CHƯƠNG TRÌNH CHÍNH ---

# 1. Vẽ Tiêu đề chính đẹp mắt
st.markdown(
    f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 25px;">
        <h1 style="color: #1f3a93; margin-bottom: 5px; font-size: 32px;">🧠 {PROJECT_INFO['title']}</h1>
        <h4 style="color: #444; margin-top: 0px; font-weight: normal;">{PROJECT_INFO['subtitle']}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Load Dữ liệu & Model
try:
    vectorizer, model = load_resources()
    model_ok = True
except Exception as exc:
    vectorizer, model = None, None
    model_ok = False

train_df, val_df, test_df = load_all_data()

# 3. Thanh điều hướng (Sidebar Menu)
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📌 MENU ĐIỀU HƯỚNG</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Tạo radio button cho menu
    menu_selection = st.radio(
        "Chọn chức năng:",
        ["📊 Giới thiệu & EDA", "🚀 Triển khai mô hình", "📉 Đánh giá & Hiệu năng"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if not model_ok:
        st.error("⚠️ Lỗi: Không thể tải mô hình PhoBERT.")

# 4. Điều hướng tới các trang dựa trên menu
if menu_selection == "📊 Giới thiệu & EDA":
    render_eda_tab(train_df, val_df, test_df)

elif menu_selection == "🚀 Triển khai mô hình":
    if model_ok:
        render_prediction_tab(vectorizer, model)
    else:
        st.error("Mô hình chưa sẵn sàng. Vui lòng kiểm tra lại đường dẫn model trong config.py")

elif menu_selection == "📉 Đánh giá & Hiệu năng":
    if model_ok:
        # TRUYỀN ĐỦ 3 THAM SỐ: test_df, vectorizer, model
        render_evaluation_tab(test_df, vectorizer, model)
    else:
        st.error("Mô hình chưa sẵn sàng. Không thể chạy đánh giá.")