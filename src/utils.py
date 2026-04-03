def get_label_name(label_id: int) -> str:
    id2label = {
        0: "Mức 0 - Không/ít tiêu cực",
        1: "Mức 1 - Lo lắng/Sợ nhẹ",
        2: "Mức 2 - Buồn",
        3: "Mức 3 - Khó chịu/Phẫn nộ",
        4: "Mức 4 - Tức giận mạnh"
    }
    return id2label.get(label_id, "Không xác định")

def get_risk_level(score: int) -> str:
    if score <= 1:
        return "Bình thường"
    elif score == 2:
        return "Cần theo dõi"
    return "Cảnh báo sớm"