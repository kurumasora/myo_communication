from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nn.model import model
from nn.util import save_model, to_categorical


CLASS_LABELS = {
    0: "no_motion",
    1: "a",
    2: "i",
    3: "u",
}
NUM_CLASSES = len(CLASS_LABELS)
RESULTS_DIR = ROOT_DIR / "model_evaluation" / "results"
DATA_DIR = ROOT_DIR / "teacher_data"


@dataclass
class Dataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    onehot: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the model with teacher_data train/val/test splits."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--output-prefix", default="teacher_eval")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_labels(path: Path) -> np.ndarray:
    labels: list[int] = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            value = line.strip()
            if value:
                labels.append(int(value))
    return np.array(labels, dtype=np.int64)


def load_split(name: str) -> Dataset:
    split_dir = DATA_DIR / name
    data = np.loadtxt(split_dir / "data.csv", delimiter=",", comments="#", dtype=np.float64)
    labels = load_labels(split_dir / "label.csv")
    samples = data.reshape(labels.shape[0], -1) / 255.0
    return Dataset(
        name=name,
        x=samples,
        y=labels,
        onehot=to_categorical(labels, NUM_CLASSES),
    )


def build_model(input_size: int, hidden_size: int) -> model:
    m = model()
    m.add(input_size, hidden_size, "ReLU")
    m.add(hidden_size, NUM_CLASSES, "softmax")
    return m


def predict_classes(m: model, x: np.ndarray) -> np.ndarray:
    probs = m.predict(x)
    return np.argmax(probs, axis=1)


def compute_loss(m: model, dataset: Dataset) -> float:
    probs = m.predict(dataset.x)
    return float(m.get_error(probs, dataset.onehot, dataset.x.shape[0]))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def classification_report(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for class_id, class_name in CLASS_LABELS.items():
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - tp)
        fn = float(matrix[class_id, :].sum() - tp)
        support = float(matrix[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return report


def evaluate_dataset(m: model, dataset: Dataset) -> dict[str, object]:
    y_pred = predict_classes(m, dataset.x)
    matrix = confusion_matrix(dataset.y, y_pred, NUM_CLASSES)
    accuracy = float((y_pred == dataset.y).mean())
    return {
        "loss": compute_loss(m, dataset),
        "accuracy": accuracy,
        "sample_count": int(dataset.x.shape[0]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(matrix),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *CLASS_LABELS.values()])
        for class_id, row in enumerate(matrix):
            writer.writerow([CLASS_LABELS[class_id], *row])


def render_summary(metrics: dict[str, object], args: argparse.Namespace) -> str:
    lines = [
        "# Teacher Data Evaluation Summary",
        "",
        "## Configuration",
        "",
        f"- epochs: {args.epochs}",
        f"- batch_size: {args.batch_size}",
        f"- learning_rate: {args.lr}",
        f"- hidden_size: {args.hidden_size}",
        f"- seed: {args.seed}",
        "",
        "## Metrics",
        "",
    ]
    for split in ("train", "val", "test"):
        result = metrics[split]
        lines.append(
            f"- {split}: accuracy={result['accuracy']:.4f}, loss={result['loss']:.4f}, samples={result['sample_count']}"
        )
    lines.extend(
        [
            "",
            "## Test Classification Report",
            "",
            "| class | precision | recall | f1 | support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_name, row in metrics["test"]["classification_report"].items():
        lines.append(
            f"| {class_name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Test Confusion Matrix",
            "",
            "| actual/predicted | no_motion | a | i | u |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_id, row in enumerate(metrics["test"]["confusion_matrix"]):
        lines.append(
            f"| {CLASS_LABELS[class_id]} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    m = build_model(train.x.shape[1], args.hidden_size)
    history = m.fit(
        input_train=train.x,
        correct_train=train.onehot,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss="cross_entropy_error",
        lr=args.lr,
    )

    metrics = {
        "train": evaluate_dataset(m, train),
        "val": evaluate_dataset(m, val),
        "test": evaluate_dataset(m, test),
        "history": history,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "class_labels": CLASS_LABELS,
        },
    }

    model_path = RESULTS_DIR / f"{args.output_prefix}.model"
    metrics_path = RESULTS_DIR / f"{args.output_prefix}_metrics.json"
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.md"
    confusion_path = RESULTS_DIR / f"{args.output_prefix}_test_confusion_matrix.csv"

    save_model(model_path, m)
    write_json(metrics_path, metrics)
    summary_path.write_text(render_summary(metrics, args), encoding="utf-8")
    write_confusion_csv(confusion_path, metrics["test"]["confusion_matrix"])

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    print(
        "Final metrics: "
        f"train_acc={metrics['train']['accuracy']:.4f}, "
        f"val_acc={metrics['val']['accuracy']:.4f}, "
        f"test_acc={metrics['test']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
