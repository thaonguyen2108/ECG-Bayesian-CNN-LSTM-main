from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.utils import load_json


def _markdown_table(path: Path) -> str:
    if not path.exists():
        return "Chưa có bảng metrics. Cần chạy train/evaluate thật trước khi kết luận kết quả."
    data = pd.read_csv(path)
    if data.empty:
        return "File metrics đang trống. Cần chạy lại evaluate trước khi nhận xét."
    columns = list(data.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _honest_observation(path: Path) -> str:
    if not path.exists():
        return "Chưa có metrics nên chưa kết luận mô hình nào tốt nhất."
    data = pd.read_csv(path)
    if data.empty or "accuracy" not in data.columns or "f1_macro" not in data.columns:
        return "Metrics chưa đủ cột accuracy/F1 macro để so sánh."
    best_acc = data.loc[data["accuracy"].idxmax()]
    best_f1 = data.loc[data["f1_macro"].idxmax()]
    return (
        f"Mô hình có accuracy cao nhất hiện tại là `{best_acc['model_name']}` "
        f"({best_acc['accuracy']:.4f}). Mô hình có F1 macro cao nhất là "
        f"`{best_f1['model_name']}` ({best_f1['f1_macro']:.4f}). "
        "Nhận xét này chỉ dựa trên lần chạy thực nghiệm đã lưu trong outputs/metrics."
    )


def _metadata_summary() -> str:
    path = config.REPORTS_DIR / "data_metadata.json"
    if not path.exists():
        return "Chưa có metadata dữ liệu. Metadata sẽ được tạo khi chạy pipeline có dataset."
    metadata = load_json(path)
    samples = metadata.get("samples", {})
    distribution = metadata.get("class_distribution", {})
    return (
        f"- Số feature: {metadata.get('num_features', config.INPUT_LENGTH)}\n"
        f"- Số mẫu train/validation/test: {samples.get('train')}/"
        f"{samples.get('validation')}/{samples.get('test')}\n"
        f"- Phân bố lớp train: `{distribution.get('train', {})}`"
    )


def generate_experiment_summary() -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = config.METRICS_DIR / "model_comparison.csv"
    figures = sorted(path.name for path in config.FIGURE_DIR.glob("*.png"))
    figures_text = "\n".join(f"- `{name}`" for name in figures) if figures else "- Chưa có biểu đồ."

    content = f"""# Tóm tắt thực nghiệm

## 1. Tên đề tài

Xây dựng hệ thống dự đoán rối loạn nhịp tim có ước lượng độ tin cậy sử dụng mô hình Bayesian CNN-LSTM.

## 2. Mục tiêu project

Project đọc tín hiệu ECG heartbeat từ bộ MIT-BIH đã chuẩn hóa, huấn luyện 3 mô hình PyTorch, đánh giá bằng các chỉ số phân loại và xây dựng app Streamlit để demo dự đoán.

## 3. Dataset

- Kaggle: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- Nguồn gốc: MIT-BIH Arrhythmia Database - PhysioNet
- Mỗi mẫu có 187 điểm tín hiệu ECG và 1 nhãn thuộc 5 lớp N, S, V, F, Q.

## 4. Mô tả dữ liệu

{_metadata_summary()}

Dữ liệu MIT-BIH thường mất cân bằng lớp, vì vậy project dùng class weights trong CrossEntropyLoss, tính riêng từ train set.

## 5. Tiền xử lý

CSV được đọc với `header=None`, tách 187 feature và nhãn cuối, ép feature về `float32`, nhãn về `int64`, kiểm tra NaN/inf và chia train/validation bằng stratified split. File `mitbih_test.csv` được giữ làm test set cuối.

## 6. Ba mô hình

- CNN 1D baseline: mô hình truyền thống trong phạm vi bài, dùng Conv1D và pooling để trích đặc trưng cục bộ.
- CNN-BiLSTM-Attention: mô hình tiên tiến hơn, kết hợp CNN, BiLSTM và attention pooling.
- Bayesian CNN-BiLSTM-Attention: mô hình cải tiến của nhóm, thêm MC Dropout để trả confidence và uncertainty.

## 7. Cải tiến của nhóm

Nhóm kết hợp CNN + BiLSTM + Attention để học cả đặc trưng cục bộ và quan hệ chuỗi của tín hiệu ECG. Phần Bayesian dùng MC Dropout để hệ thống không chỉ dự đoán nhãn mà còn ước lượng độ bất định, hữu ích trong bối cảnh y tế vì các mẫu có uncertainty cao cần được kiểm tra lại.

## 8. Bảng metrics

{_markdown_table(comparison_path)}

## 9. Nhận xét trung thực

{_honest_observation(comparison_path)}

Bayesian model có thể đánh đổi một phần tốc độ inference vì phải chạy nhiều lần MC Dropout, nhưng đổi lại cung cấp confidence/uncertainty.

## 10. Biểu đồ đã xuất

{figures_text}

## 11. Hạn chế

- Hệ thống chỉ phục vụ học tập và thực nghiệm, không thay thế bác sĩ.
- Dataset là các đoạn heartbeat đã trích và chuẩn hóa, chưa phải toàn bộ quy trình chẩn đoán lâm sàng.
- Cần kiểm thử thêm trên dữ liệu đa nguồn nếu muốn áp dụng thực tế.

## 12. Gợi ý đưa vào Word/PPT

- Một slide mô tả dataset và 5 lớp nhịp tim.
- Một slide so sánh kiến trúc 3 mô hình.
- Một slide bảng metrics và confusion matrix.
- Một slide giải thích MC Dropout, confidence và uncertainty.
- Một slide hạn chế và hướng phát triển.
"""
    output_path = config.REPORTS_DIR / "experiment_summary.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path
