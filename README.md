# ECG_Bayesian_CNN_LSTM

## 1. Giới thiệu

Project xây dựng hệ thống dự đoán rối loạn nhịp tim từ tín hiệu ECG heartbeat. Nhóm triển khai và so sánh 3 mô hình: CNN 1D baseline, CNN-BiLSTM-Attention và Bayesian CNN-BiLSTM-Attention dùng MC Dropout để ước lượng độ tin cậy.

## 2. Dataset

- Kaggle: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- Nguồn gốc học thuật: MIT-BIH Arrhythmia Database - PhysioNet: https://www.physionet.org/physiobank/database/mitdb/
- Đặt file vào `data/raw/`:
  - `mitbih_train.csv`
  - `mitbih_test.csv`

File CSV không có header, gồm 187 cột tín hiệu ECG đã chuẩn hóa và 1 cột nhãn cuối cùng. Code không tự tải dataset và không tạo dữ liệu giả.

## 3. Cấu trúc project

```text
ECG_Bayesian_CNN_LSTM/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   ├── README_data.txt
│   │   ├── mitbih_train.csv
│   │   └── mitbih_test.csv
│   └── processed/
├── notebooks/
│   └── experiment_summary.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluate.py
│   ├── uncertainty.py
│   ├── visualize.py
│   ├── report_utils.py
│   ├── utils.py
│   └── models/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── reports/
├── main.py
├── requirements.txt
└── README.md
```

## 4. Cài đặt môi trường

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 5. Cách chạy project

Train, evaluate và tạo biểu đồ:

```powershell
.\.venv\Scripts\python.exe main.py --mode all
```

Smoke test 1 epoch:

```powershell
.\.venv\Scripts\python.exe main.py --mode train --model cnn_baseline --epochs 1
.\.venv\Scripts\python.exe main.py --mode evaluate --model cnn_baseline
.\.venv\Scripts\python.exe main.py --mode visualize
```

Train riêng từng mô hình:

```powershell
.\.venv\Scripts\python.exe main.py --mode train --model cnn_baseline
.\.venv\Scripts\python.exe main.py --mode train --model cnn_bilstm_attention
.\.venv\Scripts\python.exe main.py --mode train --model bayesian_cnn_bilstm_attention
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe main.py --mode evaluate --model all
```

Chạy app Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

## 6. Kiểm tra GPU/CUDA

Kiểm tra môi trường hiện tại:

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Nếu `torch.__version__` có `+cpu` hoặc `torch.version.cuda` là `None`, PyTorch trong `.venv` đang là bản CPU-only. Project vẫn chạy được bằng CPU, nhưng full training sẽ lâu hơn đáng kể.

Nếu máy có GPU NVIDIA và `nvidia-smi` hoạt động, có thể cài lại PyTorch CUDA trong `.venv` bằng lệnh đã dùng cho project này:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Nếu wheel CUDA 12.6 không phù hợp, có thể thử CUDA 11.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Sau khi cài, chạy lại `scripts/check_environment.py`. Khi CUDA hoạt động, output sẽ có `torch` dạng `+cu...`, `torch.version.cuda` khác `None`, `CUDA available: True` và tên GPU NVIDIA.

## 7. Kiểm tra dữ liệu

Kiểm tra file MIT-BIH trong `data/raw/`:

```powershell
.\.venv\Scripts\python.exe scripts/check_data.py
```

Script kiểm tra sự tồn tại của `mitbih_train.csv`, `mitbih_test.csv`, số cột 188, label trong 0..4, NaN/inf và in phân bố lớp train/test. Script không tự sửa dữ liệu và không tạo dữ liệu giả.

## 8. Mô hình sử dụng

- CNN 1D baseline: mô hình truyền thống, dùng Conv1D, BatchNorm, pooling và classifier tuyến tính.
- CNN-BiLSTM-Attention: mô hình tiên tiến hơn, CNN trích đặc trưng cục bộ, BiLSTM học quan hệ theo chuỗi, Attention chọn bước thời gian quan trọng.
- Bayesian CNN-BiLSTM-Attention: mô hình cải tiến của nhóm, thêm Dropout và MC Dropout inference để trả confidence/uncertainty.

## 9. Chỉ số đánh giá

- Loss
- Accuracy
- Precision macro
- Recall macro
- F1 macro
- Confusion matrix
- Training time
- Inference time
- Confidence và uncertainty với Bayesian model

Dataset MIT-BIH mất cân bằng lớp, project xử lý bằng class weights trong `CrossEntropyLoss`. Class weights chỉ tính từ train set để tránh data leakage.

## 10. Ghi chú y tế

Hệ thống chỉ dùng cho mục đích học tập và thực nghiệm, không thay thế kết luận chẩn đoán của bác sĩ.
