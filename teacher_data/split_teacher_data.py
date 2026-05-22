from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "teacher_data"
RAW_DIR = DATA_DIR / "raw"
STEP_ROWS = 50
TRAIN_COUNT = 4480
VAL_COUNT = 560
TEST_COUNT = 560


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split teacher_data/raw into train, val, and test directories."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling.",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[int]:
    with path.open(encoding="utf-8-sig") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return labels


def load_data_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    header_lines: list[str] = []
    data_rows: list[list[str]] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].startswith("#"):
                header_lines.append(",".join(row))
                continue
            if row:
                data_rows.append(row)

    return header_lines, data_rows


def validate_dataset(labels: list[int], data_rows: list[list[str]]) -> None:
    expected_rows = len(labels) * STEP_ROWS
    if len(labels) != TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        raise ValueError(f"Expected 5600 labels, found {len(labels)}.")
    if len(data_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} data rows from {len(labels)} labels, found {len(data_rows)}."
        )


def write_split(
    split_dir: Path,
    header_lines: list[str],
    labels: list[int],
    samples: list[list[list[str]]],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    label_path = split_dir / "label.csv"
    with label_path.open("w", encoding="utf-8", newline="") as f:
        for label in labels:
            f.write(f"{label}\n")

    data_path = split_dir / "data.csv"
    with data_path.open("w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(f"{line}\n")

        writer = csv.writer(f)
        for sample in samples:
            writer.writerows(sample)


def main() -> None:
    args = parse_args()

    labels = load_labels(RAW_DIR / "label.csv")
    header_lines, data_rows = load_data_rows(RAW_DIR / "data.csv")
    validate_dataset(labels, data_rows)

    samples = [
        data_rows[index * STEP_ROWS : (index + 1) * STEP_ROWS]
        for index in range(len(labels))
    ]

    indices = list(range(len(labels)))
    random.Random(args.seed).shuffle(indices)

    split_specs = {
        "train": indices[:TRAIN_COUNT],
        "val": indices[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT],
        "test": indices[TRAIN_COUNT + VAL_COUNT :],
    }

    for split_name, split_indices in split_specs.items():
        split_labels = [labels[index] for index in split_indices]
        split_samples = [samples[index] for index in split_indices]
        write_split(DATA_DIR / split_name, header_lines, split_labels, split_samples)

    print(
        f"Created train={TRAIN_COUNT}, val={VAL_COUNT}, test={TEST_COUNT} "
        f"from {RAW_DIR} with seed={args.seed}."
    )


if __name__ == "__main__":
    main()
