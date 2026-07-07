from __future__ import annotations

import argparse
import sys

MODEL_NAMES = ("cnn_baseline", "cnn_bilstm_attention", "bayesian_cnn_bilstm_attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG Bayesian CNN-LSTM project pipeline")
    parser.add_argument("--mode", default="all", choices=["all", "train", "evaluate", "visualize", "predict-sample"])
    parser.add_argument("--model", default="all", choices=["all", *MODEL_NAMES])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def selected_models(model_arg: str) -> tuple[str, ...]:
    return MODEL_NAMES if model_arg == "all" else (model_arg,)


def prepare_data(batch_size: int):
    from src.data_loader import create_data_splits, create_dataloaders
    from src.preprocessing import save_data_metadata

    print("Đang đọc dữ liệu ECG...")
    x_train, y_train, x_val, y_val, x_test, y_test = create_data_splits()
    save_data_metadata(x_train, y_train, x_val, y_val, x_test, y_test)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=batch_size
    )
    return (x_train, y_train, x_val, y_val, x_test, y_test), train_loader, val_loader, test_loader, class_weights


def run_train(model_names: tuple[str, ...], train_loader, val_loader, class_weights, device, epochs: int) -> None:
    from src.model_factory import build_model
    from src.training import train_model

    for model_name in model_names:
        print(f"\nHuấn luyện {model_name} trên {device}...")
        model = build_model(model_name)
        train_model(model_name, model, train_loader, val_loader, device, class_weights=class_weights, epochs=epochs)


def run_evaluate(model_names: tuple[str, ...], test_loader, device) -> None:
    from src.evaluate import compare_models, evaluate_model, load_checkpoint_model

    for model_name in model_names:
        print(f"\nĐánh giá {model_name}...")
        model = load_checkpoint_model(model_name, device)
        metrics = evaluate_model(model_name, model, test_loader, device)
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, loss={metrics['loss']:.4f}"
        )
    compare_models(model_names)


def run_predict_sample(index: int, model_name: str, x_test, y_test, device) -> None:
    import torch

    from src import config
    from src.evaluate import load_checkpoint_model
    from src.visualize import plot_sample_ecg

    model_name = "bayesian_cnn_bilstm_attention" if model_name == "all" else model_name
    index = max(0, min(index, len(y_test) - 1))
    model = load_checkpoint_model(model_name, device)
    tensor = torch.tensor(x_test[index], dtype=torch.float32).view(1, 1, config.INPUT_LENGTH).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    prediction = int(probabilities.argmax())
    plot_sample_ecg(x_test[index], y_test[index], prediction)
    print(
        f"Sample {index}: true={config.CLASS_NAMES[int(y_test[index])]}, "
        f"prediction={config.CLASS_NAMES[prediction]}, confidence={probabilities[prediction]:.4f}"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        from src import config
        from src.report_utils import generate_experiment_summary
        from src.utils import ensure_dirs, set_seed
        from src.visualize import generate_all_figures
    except ModuleNotFoundError as exc:
        print(
            f"Thiếu thư viện: {exc.name}. Hãy cài bằng `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    ensure_dirs()
    set_seed()
    device = config.DEVICE
    print(f"Thiết bị sử dụng: {device}")

    needs_data = args.mode in {"all", "train", "evaluate", "predict-sample"}
    try:
        if needs_data:
            data, train_loader, val_loader, test_loader, class_weights = prepare_data(args.batch_size)
        else:
            data = train_loader = val_loader = test_loader = class_weights = None

        model_names = selected_models(args.model)
        if args.mode in {"all", "train"}:
            run_train(model_names, train_loader, val_loader, class_weights, device, args.epochs)
        if args.mode in {"all", "evaluate"}:
            run_evaluate(model_names, test_loader, device)
        if args.mode in {"all", "visualize"}:
            generate_all_figures(model_names if args.model != "all" else config.ALL_MODEL_NAMES)
            print(f"Biểu đồ lưu tại: {config.FIGURE_DIR}")
        summary_path = generate_experiment_summary()
        print(f"Báo cáo tóm tắt lưu tại: {summary_path}")
        if args.mode == "predict-sample":
            _, _, _, _, x_test, y_test = data
            run_predict_sample(args.index, args.model, x_test, y_test, device)

        print("\nHoàn tất.")
        print(f"Checkpoints: {config.CHECKPOINT_DIR}")
        print(f"Metrics: {config.METRICS_DIR}")
        print(f"Figures: {config.FIGURE_DIR}")
        return 0
    except FileNotFoundError as exc:
        print(
            "\nLỗi dữ liệu/checkpoint: "
            f"{exc}\nVui lòng tải ECG Heartbeat Categorization Dataset từ Kaggle và đặt "
            "mitbih_train.csv, mitbih_test.csv vào data/raw/ rồi chạy lại nếu lỗi do thiếu dữ liệu.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
