from pathlib import Path

import torch

RANDOM_SEED = 42
NUM_CLASSES = 5
INPUT_LENGTH = 187
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
MC_SAMPLES = 30
VALIDATION_SIZE = 0.15
NUM_WORKERS = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_CSV = RAW_DATA_DIR / "mitbih_train.csv"
TEST_CSV = RAW_DATA_DIR / "mitbih_test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

CLASS_DESCRIPTIONS = {
    0: "Normal beat",
    1: "Supraventricular premature beat",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal beat",
    4: "Unclassifiable beat",
}

MODEL_DISPLAY_NAMES = {
    "cnn_baseline": "CNN 1D baseline",
    "cnn_bilstm_attention": "CNN-BiLSTM-Attention",
    "bayesian_cnn_bilstm_attention": "Bayesian CNN-BiLSTM-Attention",
}

ALL_MODEL_NAMES = tuple(MODEL_DISPLAY_NAMES.keys())
