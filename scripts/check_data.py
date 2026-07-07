from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = RAW_DIR / "mitbih_train.csv"
TEST_CSV = RAW_DIR / "mitbih_test.csv"
EXPECTED_COLUMNS = 188
VALID_LABELS = {0, 1, 2, 3, 4}


def validate_file(path: Path, name: str) -> dict[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Thiếu file {path}. Vui lòng đặt {path.name} vào data/raw/ rồi chạy lại."
        )

    row_count = 0
    label_counts = {label: 0 for label in sorted(VALID_LABELS)}
    for chunk in pd.read_csv(path, header=None, chunksize=20000):
        if chunk.shape[1] != EXPECTED_COLUMNS:
            raise ValueError(
                f"{name} phải có đúng {EXPECTED_COLUMNS} cột, hiện có {chunk.shape[1]} cột."
            )

        features = chunk.iloc[:, :187]
        labels = chunk.iloc[:, 187]
        values = features.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} có NaN hoặc inf trong 187 cột feature.")

        if labels.isna().any():
            raise ValueError(f"{name} có label bị NaN.")
        label_values = labels.astype(int)
        invalid = sorted(set(label_values.unique()) - VALID_LABELS)
        if invalid:
            raise ValueError(f"{name} có label ngoài 0..4: {invalid}.")

        counts = label_values.value_counts().to_dict()
        for label, count in counts.items():
            label_counts[int(label)] += int(count)
        row_count += len(chunk)

    print(f"{name}: {row_count} dòng, {EXPECTED_COLUMNS} cột.")
    print(f"Phân bố lớp {name}: {label_counts}")
    return label_counts


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    validate_file(TRAIN_CSV, "train")
    validate_file(TEST_CSV, "test")
    print("Kết luận: Dữ liệu hợp lệ, không phát hiện NaN/inf hoặc label sai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
