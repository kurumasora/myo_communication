# -*- coding: utf-8 -*-
import serial
import numpy as np
from nn.util import load_model
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2

# シリアルポートcom9,シリアルポーレート115200
ser = serial.Serial("/dev/cu.usbserial-58550230311", 115200, timeout=1)

# 1. 2次元アルファベット配列の定義（5行×6列）
CHAR_MATRIX = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "SPC", "BS", "CLR", "End"],
]

MAX_ROWS = len(CHAR_MATRIX)
MAX_COLS = len(CHAR_MATRIX[0])

# カーソルの初期位置
current_row = 0
current_col = 0

# 確定された文字を蓄積するバッファ
text_buffer = []

# データを格納する配列
data_1 = []

# カウント
count = 0
count_1 = 0
count_2 = 0
count_4 = 0


# PIL->OpenCVに変換する関数を呼び出し
def pil_to_cv2(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 3 and new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    return new_image


def draw_matrix_ui(row_idx, col_idx, buffer_text):
    """2次元配列、カーソル、入力文字バッファを画面に描画する関数"""
    # 画面サイズ (論文の仕様に合わせて 横1000 × 縦600)
    win_w, win_h = 1000, 600
    img = PIL.Image.new("RGB", (win_w, win_h), (30, 30, 30))
    draw = PIL.ImageDraw.Draw(img)

    try:
        font_main = PIL.ImageFont.truetype("Arial.ttf", 36)
        font_sub = PIL.ImageFont.truetype("Arial.ttf", 24)
        font_key = PIL.ImageFont.truetype("Arial.ttf", 22)
    except Exception:
        font_main = PIL.ImageFont.load_default()
        font_sub = PIL.ImageFont.load_default()
        font_key = PIL.ImageFont.load_default()

    # グリッドの配置設定（6列に合わせてサイズを調整）
    start_x, start_y = 70, 150
    cell_w, cell_h = 140, 70

    # マトリックスの描画ループ
    for r in range(MAX_ROWS):
        for c in range(MAX_COLS):
            x = start_x + c * cell_w
            y = start_y + r * cell_h
            char_text = CHAR_MATRIX[r][c]

            current_font = font_key if len(char_text) > 1 else font_main
            offset_x = 40 if len(char_text) > 1 else 50
            offset_y = 18 if len(char_text) > 1 else 10

            if r == row_idx and c == col_idx:
                draw.rectangle(
                    [x, y, x + cell_w - 15, y + cell_h - 15],
                    fill=(255, 255, 255),
                )
                draw.text(
                    (x + offset_x, y + offset_y),
                    char_text,
                    fill=(0, 0, 0),
                    font=current_font,
                )
            else:
                draw.rectangle(
                    [x, y, x + cell_w - 15, y + cell_h - 15],
                    outline=(100, 100, 100),
                    width=2,
                )
                draw.text(
                    (x + offset_x, y + offset_y),
                    char_text,
                    fill=(255, 255, 255),
                    font=current_font,
                )

    display_text = "TEXT BUFFER: " + "".join(buffer_text)
    draw.text((70, 50), display_text, fill=(0, 255, 128), font=font_sub)

    instruction_text = "[MYO] 1: Down | 2: Right | 3: Select | Q: Quit"
    draw.text((70, 530), instruction_text, fill=(180, 180, 180), font=font_sub)

    return pil_to_cv2(img)


def select_current_char():
    """現在のカーソル位置の文字・機能キーを確定する関数"""
    global is_running

    selected_char = CHAR_MATRIX[current_row][current_col]

    if selected_char == "SPC":
        text_buffer.append(" ")
    elif selected_char == "BS":
        if len(text_buffer) > 0:
            text_buffer.pop()
    elif selected_char == "CLR":
        text_buffer.clear()
    elif selected_char == "End":
        is_running = False
    else:
        text_buffer.append(selected_char)


def apply_prediction(pred_1):
    """予測結果をmatrix.pyの操作に変換する関数"""
    global current_row
    global current_col
    global count_4

    # 0は手を動かしていないときのラベル
    if pred_1 == 0:
        return

    # 元の「あ/い/う」出力を、行列UIの「下/右/選択」に置換する
    if pred_1 == 1:
        if current_row < MAX_ROWS - 1:
            current_row += 1
    elif pred_1 == 2:
        if current_col < MAX_COLS - 1:
            current_col += 1
    else:
        select_current_char()

    # 次は1回待機させるためのカウント
    count_4 = 1


# 予測関数
def predict(data, model):
    # 入力データが教師データのどれに近いのか
    pred = model.predict([data / 255.0])
    pred_classes = np.argmax(pred, axis=1)
    pred_1 = int(pred_classes[0])
    apply_prediction(pred_1)


is_running = True
while is_running:
    cv2_image = draw_matrix_ui(current_row, current_col, text_buffer)
    cv2.imshow("Alphabet UI Simulator", cv2_image)

    key = cv2.waitKeyEx(1)
    if key == ord("q") or key == ord("Q") or key == 27:
        break

    # 数値の読み取り
    s = ser.readline()
    val = s.decode("utf-8").rstrip(",\r")
    val_1 = val.replace("\r", "")

    for i in range(len(val_1)):
        # val_1[i] == ','になるまでのカウントをとる。
        # 格納した数値が2桁なのか3桁なのかを知りたい。
        count = count + 1

        if val_1[i] == ",":
            # ','で区切られるデータが3つ取得するが必要なのは2つ目と3つ目
            count_1 = count_1 + 1

            if count_1 == 2:
                data_1.append(int(val_1[i + 1 - count : i]))

            if count_1 == 3:
                data_1.append(int(val_1[i + 1 - count : i]))

                # 0に戻す。データは3個までしか取れない
                count_1 = 0

                # 教師データは100だから入力データが100個取れるまでカウントする
                count_2 = count_2 + 1

            # 0に戻す。上限254だから3桁以上になることはない
            count = 0

        # 入力データが100個集ったら。2つ目と3つ目のデータをとってカウントするからcount_2が50で入力データ100
        if count_2 == 50:
            data = np.array(data_1[0:100])

            # 0に戻す。入力データが100以上だと学習できない
            count_2 = 0

            # 初期化。次の入力データを格納するのに前の入力データを消去する
            data_1 = []

            # 学習モデルの呼び出し
            m = load_model("train.model")

            # 手を動かしたときに戻す動作が必要な場合があるため、1回学習をせず間を作る
            if count_4 == 0:
                predict(data, m)
            else:
                count_4 = 0

cv2.destroyAllWindows()


