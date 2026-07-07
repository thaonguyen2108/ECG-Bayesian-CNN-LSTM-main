from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_raw_data
from src.evaluate import load_checkpoint_model
from src.model_factory import checkpoint_path
from src.uncertainty import predict_with_uncertainty, uncertainty_level


def plot_signal(signal) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(signal)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Amplitude")
    ax.set_title("Tín hiệu ECG")
    st.pyplot(fig)


@st.cache_data(show_spinner=False)
def cached_test_data():
    _, _, x_test, y_test = load_raw_data()
    return x_test, y_test


def load_user_csv(uploaded_file):
    frame = pd.read_csv(uploaded_file, header=None)
    values = frame.to_numpy().reshape(-1)
    if len(values) != config.INPUT_LENGTH:
        raise ValueError(f"File upload cần đúng {config.INPUT_LENGTH} giá trị ECG.")
    return values.astype("float32")


def predict_signal(model_name: str, signal):
    device = config.DEVICE
    model = load_checkpoint_model(model_name, device)
    tensor = torch.tensor(signal, dtype=torch.float32).view(1, 1, config.INPUT_LENGTH).to(device)
    if model_name == "bayesian_cnn_bilstm_attention":
        result = predict_with_uncertainty(model, tensor.cpu(), device, mc_samples=config.MC_SAMPLES)
        probabilities = result["mean_probs"][0]
        prediction = int(result["pred_class"][0])
        return probabilities, prediction, result
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return probabilities, int(probabilities.argmax()), None


def render_prediction_tab() -> None:
    st.subheader("Dự đoán ECG")
    display_to_name = {v: k for k, v in config.MODEL_DISPLAY_NAMES.items()}
    selected_display = st.selectbox("Chọn mô hình", list(display_to_name.keys()))
    model_name = display_to_name[selected_display]
    ckpt = checkpoint_path(model_name)

    if not ckpt.exists():
        st.warning(
            "Chưa tìm thấy checkpoint. Vui lòng chạy "
            "`python main.py --mode all` hoặc train model trước."
        )
        return

    source = st.radio("Nguồn dữ liệu", ["Mẫu test set", "Upload CSV 1 dòng"], horizontal=True)
    signal = None
    if source == "Mẫu test set":
        try:
            x_test, y_test = cached_test_data()
            index = st.number_input("Index mẫu test", min_value=0, max_value=len(y_test) - 1, value=0, step=1)
            if st.button("Chọn random"):
                index = int(pd.Series(range(len(y_test))).sample(1).iloc[0])
            signal = x_test[int(index)]
            label = int(y_test[int(index)])
            st.caption(f"Nhãn thật: {config.CLASS_NAMES[label]} - {config.CLASS_DESCRIPTIONS[label]}")
        except Exception as exc:
            st.error(f"Không đọc được test set: {exc}")
    else:
        uploaded = st.file_uploader("Upload CSV gồm 187 giá trị ECG", type=["csv"])
        if uploaded is not None:
            try:
                signal = load_user_csv(uploaded)
            except Exception as exc:
                st.error(str(exc))

    if signal is None:
        return

    plot_signal(signal)
    try:
        probabilities, prediction, uncertainty = predict_signal(model_name, signal)
        st.metric("Class dự đoán", f"{config.CLASS_NAMES[prediction]} - {config.CLASS_DESCRIPTIONS[prediction]}")
        st.metric("Confidence", f"{float(probabilities[prediction]):.4f}")
        st.dataframe(
            pd.DataFrame(
                {
                    "class": [config.CLASS_NAMES[i] for i in range(config.NUM_CLASSES)],
                    "description": [config.CLASS_DESCRIPTIONS[i] for i in range(config.NUM_CLASSES)],
                    "probability": probabilities,
                }
            ),
            use_container_width=True,
        )
        if uncertainty is not None:
            score = float(uncertainty["uncertainty_score"][0])
            st.metric("Uncertainty score", f"{score:.4f}", uncertainty_level(score))
        else:
            st.info("Mô hình này chỉ trả xác suất dự đoán, không thực hiện MC Dropout uncertainty.")
    except Exception as exc:
        st.error(f"Không thể dự đoán: {exc}")


def render_comparison_tab() -> None:
    st.subheader("So sánh mô hình")
    comparison_path = config.METRICS_DIR / "model_comparison.csv"
    figure_path = config.FIGURE_DIR / "model_metrics_comparison.png"
    if comparison_path.exists():
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
    else:
        st.info("Chưa có kết quả thực nghiệm. Vui lòng chạy `python main.py --mode all`.")
    if figure_path.exists():
        st.image(str(figure_path))


def render_figures_tab() -> None:
    st.subheader("Biểu đồ thực nghiệm")
    figures = sorted(config.FIGURE_DIR.glob("*.png"))
    if not figures:
        st.info("Chưa có hình trong outputs/figures. Cần chạy `python main.py --mode visualize` sau khi train/evaluate.")
    for figure in figures:
        st.caption(figure.name)
        st.image(str(figure))


def render_info_tab() -> None:
    st.subheader("Thông tin mô hình")
    st.markdown(
        """
Dataset sử dụng ECG Heartbeat Categorization Dataset trên Kaggle, có nguồn gốc từ MIT-BIH Arrhythmia Database.

Nhãn gồm 5 lớp: N, S, V, F, Q.

Ba mô hình được so sánh:

- CNN 1D baseline.
- CNN-BiLSTM-Attention.
- Bayesian CNN-BiLSTM-Attention.

Bayesian model dùng MC Dropout: khi inference chỉ bật các lớp Dropout nhiều lần, lấy trung bình xác suất và entropy dự đoán để ước lượng độ bất định.

Hệ thống chỉ phục vụ mục đích học tập và thực nghiệm, không thay thế kết luận chẩn đoán của bác sĩ.
        """
    )


def main() -> None:
    st.set_page_config(page_title="ECG Bayesian CNN-LSTM", layout="wide")
    st.title("N7 - ECG Bayesian CNN-LSTM")
    tabs = st.tabs(["Dự đoán ECG", "So sánh mô hình", "Biểu đồ thực nghiệm", "Thông tin mô hình"])
    with tabs[0]:
        render_prediction_tab()
    with tabs[1]:
        render_comparison_tab()
    with tabs[2]:
        render_figures_tab()
    with tabs[3]:
        render_info_tab()


if __name__ == "__main__":
    main()
