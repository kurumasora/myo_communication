# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Python 3.11. Install dependencies:

```bash
pip install matplotlib numpy opencv-python pillow pyserial pandas
```

## Running

Train the model (outputs `src/train.model`):
```bash
cd src && python train.py
```

Run the main gesture recognition loop (requires Myo device on serial port):
```bash
cd src && python main.py
```

Run the alphabet matrix UI (keyboard-navigable, no hardware required):
```bash
cd src && python matrix.py
```

Split raw teacher data into train/val/test:
```bash
python teacher_data/split_teacher_data.py [--seed 42]
```

## Architecture

Myoアームバンドから取得した2チャンネルのEMG（筋電）信号をシリアル通信で受け取り、カスタム実装のDNNでジェスチャを分類して日本語文字を出力するシステム。

### システム全体の流れ

```
[Myo アームバンド]
      | USB シリアル (/dev/cu.usbserial-58550230311, 115200 baud)
      ↓
[main.py: シリアル受信ループ]
  各行のフォーマット: "x,ch1,ch2"  (1フィールド目は破棄、ch1/ch2 が 0–254)
  50行分 = 2チャンネル × 50ステップ = 100 値 を蓄積
      ↓
[predict()]
  data / 255.0 で正規化 → nn/model.py の forwards() → softmax 出力
  argmax → クラスラベル (0=無動作, 1=あ, 2=い, 3=う)
      ↓
[PIL + OpenCV 表示ウィンドウ 1000×600]
  認識された文字を character_1[] に蓄積して描画 (最大5文字で自動クリア)
```

予測後は `count_4 = 1` をセットして次の1サイクルをスキップする（腕の戻り動作による誤認識を防ぐ）。`train.model` は推論サイクルごとにディスクから再ロードするため、プロセスを再起動せずにモデルを差し替えられる。

### 学習パイプライン

```
teacher_data/raw/data.csv   (コメント行 # をスキップ、280,000 行)
teacher_data/raw/label.csv  (BOM 除去が必要、1 行 = 1 ラベル)
      ↓ train.py
reshape → (N, 100)、/255.0 正規化
to_categorical → one-hot (4 クラス)
      ↓ model.fit()
  層1: 100 → 50 (ReLU)
  層2:  50 →  4 (softmax)
  mini-batch SGD: batch=20, lr=0.001, epochs=30, loss=cross_entropy
      ↓
src/train.model  (pickle)
```

### カスタム NN モジュール (`src/nn/`)

外部 ML フレームワークを使わないスクラッチ実装。

| ファイル | 役割 |
|---|---|
| `layer.py` | 全結合層。`forward()` で行列積＋活性化（ReLU / softmax）、`backward()` で勾配計算、`update()` で重み更新 |
| `model.py` | 層のリストを管理する DNN コンテナ。`fit()` でミニバッチ SGD、`predict()` で推論 |
| `util.py` | `to_categorical`（one-hot 変換）、`save_model`/`load_model`（pickle） |

誤差関数は `cross_entropy_error`（デフォルト）と `squared_error` を選択可能。softmax 層の逆伝播は `δ = y − t`（出力と正解の差）を使う簡略形。

### matrix.py（代替 UI）

ハードウェア不要のスタンドアロン UI。5×6 のアルファベットグリッド（A–Z、SPC/BS/CLR/End）をキーボード（矢印キーまたは WASD）で操作し、Enter で文字を確定する。Myo システムとは独立しており、文字入力インターフェースの単体テストに使える。

## データ構造

- `teacher_data/raw/`: 全データ — 4クラス × 1400サンプル × 50ステップ = 280,000行。`data.csv` は2列（ch1, ch2）、`label.csv` は1行1ラベル（0–3）
- `teacher_data/{train,val,test}/`: `split_teacher_data.py` で生成した分割データ（4480 / 560 / 560 サンプル）
- `train.py` は現在 `raw/` を直接参照。`train/val/test` はバリデーション・テスト評価への移行が想定されている
- CSV のコメント行は `#` で始まり、pandas の `comment="#"` または手動スキップで除外する
