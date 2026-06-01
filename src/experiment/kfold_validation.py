"""
k-fold Cross Validation: ベースライン vs データ拡張 vs 特徴量抽出
"""
import os
import sys
import contextlib
import numpy as np
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from nn.model import model
from nn.util import to_categorical

DATA_DIR = SRC_DIR.parent / "teacher_data" / "raw"
N_FOLDS = 5
N_CLASSES = 4
BATCH_SIZE = 20
EPOCHS = 30
LR = 0.001
LOSS = "cross_entropy_error"


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def load_raw_data():
    data = pd.read_csv(DATA_DIR / "data.csv", comment="#", header=None).to_numpy().ravel()
    with open(DATA_DIR / "label.csv", encoding="utf-8") as f:
        raw = f.read().replace("﻿", "")
    labels = np.array([int(s) for s in raw.split() if s.strip()])
    data = data.reshape(len(labels), -1)  # (N, 100)
    return data, labels


def extract_features(X_raw):
    """各チャンネルから6統計量を抽出し (N, 12) を返す。"""
    ch1 = X_raw[:, 0::2].astype(float)  # (N, 50)
    ch2 = X_raw[:, 1::2].astype(float)  # (N, 50)
    cols = []
    for ch in [ch1, ch2]:
        cols += [
            np.max(ch, axis=1) / 255.0,
            np.min(ch, axis=1) / 255.0,
            np.mean(ch, axis=1) / 255.0,
            np.std(ch, axis=1) / 127.5,       # 0-255 範囲での最大 std ≈ 127.5
            np.sum(ch, axis=1) / (255.0 * 50), # 積分値 (合計値)
            np.argmax(ch, axis=1) / 49.0,      # 最大値インデックス
        ]
    return np.column_stack(cols)  # (N, 12)


def build_and_train(X_train, y_train):
    input_size = X_train.shape[1]
    m = model()
    m.add(input_size, 50, "ReLU")
    m.add(50, N_CLASSES, "softmax")
    y_oh = to_categorical(y_train, N_CLASSES)
    with suppress_stdout():
        m.fit(X_train, y_oh, batch_size=BATCH_SIZE, epochs=EPOCHS, loss=LOSS, lr=LR)
    return m


def evaluate(m, X_test, y_test):
    pred = np.argmax(m.predict(X_test), axis=1)
    return np.mean(pred == y_test)


def augment(X, y, sigma=0.02, n_augment=4):
    """fold 分割後の学習データにのみ適用する（データリーク防止）。"""
    aug_X = [X] + [X + np.random.randn(*X.shape) * sigma for _ in range(n_augment)]
    aug_y = [y] * (n_augment + 1)
    return np.vstack(aug_X), np.concatenate(aug_y)


def kfold_indices(n, k=5, seed=42):
    idx = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(idx, k)
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def run_kfold(pattern_name, data_raw, labels, use_augment=False, use_features=False):
    print(f"\n=== {pattern_name} ===")
    splits = kfold_indices(len(labels), k=N_FOLDS)
    accs = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr_raw, X_te_raw = data_raw[train_idx], data_raw[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        if use_features:
            X_tr = extract_features(X_tr_raw)
            X_te = extract_features(X_te_raw)
        else:
            X_tr = X_tr_raw / 255.0
            X_te = X_te_raw / 255.0

        if use_augment:
            # fold 分割後に水増し → テストデータには手を加えない
            X_tr, y_tr = augment(X_tr, y_tr, sigma=0.02, n_augment=4)

        m = build_and_train(X_tr, y_tr)
        acc = evaluate(m, X_te, y_te)
        print(f"  fold{fold_i + 1}: {acc:.3f}")
        accs.append(acc)

    mean, std = float(np.mean(accs)), float(np.std(accs))
    print(f"  平均精度: {mean:.3f}")
    print(f"  標準偏差: {std:.3f}")
    return mean, std, accs


def print_comparison(results):
    labels_ja = ["ベースライン        ", "データ拡張          ", "特徴量抽出          ", "特徴量抽出+データ拡張"]
    print("\n=== 最終比較 ===")
    print(f"| {'手法':<20} | {'平均精度':^8} | {'標準偏差':^8} |")
    print(f"|{'-'*22}|{'-'*10}|{'-'*10}|")
    for name, result in zip(labels_ja, results):
        mean, std = result[0], result[1]
        print(f"| {name} | {mean:.3f}    | {std:.3f}    |")

    best_idx = int(np.argmax([r[0] for r in results]))
    recommend_labels = [
        "ベースライン",
        "データ拡張（Gaussian Noise）",
        "特徴量抽出（統計量）",
        "特徴量抽出 + データ拡張",
    ]
    reasons = [
        "水増しや特徴変換なしでも最高精度が得られており、シンプルさと精度を両立できるため",
        "ガウシアンノイズによるデータ水増しが汎化性能の向上に最も寄与したため",
        "統計的特徴量による次元圧縮が筋電ジェスチャの識別に最も有効だったため",
        "特徴量抽出とデータ拡張の組み合わせが精度・安定性の両面で最優位だったため",
    ]
    print(f"\n推奨手法: {recommend_labels[best_idx]}（理由: {reasons[best_idx]}）")


if __name__ == "__main__":
    print("データを読み込み中...")
    data_raw, labels = load_raw_data()
    print(f"データ形状: {data_raw.shape}, ラベル数: {len(labels)}")

    results = []
    results.append(run_kfold("パターンA：ベースライン", data_raw, labels))
    results.append(run_kfold("パターンB：データ拡張（Gaussian Noise）", data_raw, labels, use_augment=True))
    results.append(run_kfold("パターンC：特徴量抽出（統計量）", data_raw, labels, use_features=True))
    results.append(run_kfold("パターンD：特徴量抽出 + データ拡張", data_raw, labels, use_features=True, use_augment=True))

    print_comparison(results)
