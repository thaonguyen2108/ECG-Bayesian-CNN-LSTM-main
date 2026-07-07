from __future__ import annotations

import sys

import torch


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        print("Kết luận: PyTorch đang nhận CUDA, có thể train bằng GPU.")
    else:
        print("Kết luận: PyTorch chưa nhận CUDA, project sẽ chạy bằng CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
