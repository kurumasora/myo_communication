# -*- coding: utf-8 -*-
import time
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import serial

from nn.util import load_model


SERIAL_PORT = "/dev/cu.usbserial-575E0797941"
SERIAL_BAUDRATE = 115200
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "train.model"
WINDOW_NAME = "Myo Matrix Input"
DEBOUNCE_SECONDS = 0.4

# 現在の学習ラベルが 0..3 の場合:
# 0: 待機, 1: 右, 2: 下, 3: 決定
#
# 上/左も使うように再学習した場合は、例えば以下のように増やせます。
# 4: 左, 5: 上
ACTION_BY_LABEL = {
    1: "right",
    2: "down",
    3: "select",
    4: "left",
    5: "up",
}

CHAR_MATRIX = [
    ["あ", "か", "さ", "た", "な"],
    ["い", "き", "し", "ち", "に"],
    ["う", "く", "す", "つ", "ぬ"],
    ["え", "け", "せ", "て", "ね"],
    ["お", "こ", "そ", "と", "の"],
]


def load_font(size):
    font_candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "Arial.ttf",
    ]
    for font_path in font_candidates:
        try:
            return PIL.ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return PIL.ImageFont.load_default()


def pil_to_cv2(image):
    """PILイメージをOpenCV形式に変換する。"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 3 and new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    return new_image


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


class MatrixInputController:
    def __init__(self, matrix, debounce_seconds=0.4):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.row = 0
        self.col = 0
        self.text_buffer = []
        self.previous_label = 0
        self.last_trigger_time = 0.0
        self.debounce_seconds = debounce_seconds

    def process_label(self, predicted_label):
        """推論ラベルをエッジトリガー + 不応期つきで操作に変換する。"""
        now = time.time()
        current_label = int(predicted_label)
        should_trigger = (
            self.previous_label == 0
            and current_label != 0
            and now - self.last_trigger_time >= self.debounce_seconds
        )

        if should_trigger:
            action = ACTION_BY_LABEL.get(current_label)
            if action is not None:
                self.apply_action(action)
                self.last_trigger_time = now

        self.previous_label = current_label

    def apply_action(self, action):
        if action == "right":
            self.col = (self.col + 1) % self.cols
        elif action == "left":
            self.col = (self.col - 1) % self.cols
        elif action == "down":
            self.row = (self.row + 1) % self.rows
        elif action == "up":
            self.row = (self.row - 1) % self.rows
        elif action == "select":
            self.select_current_char()

    def select_current_char(self):
        selected_char = self.matrix[self.row][self.col]
        self.text_buffer.append(selected_char)

    def draw(self):
        win_w, win_h = 1000, 600
        img = PIL.Image.new("RGB", (win_w, win_h), (20, 20, 20))
        draw = PIL.ImageDraw.Draw(img)

        font_cell = load_font(42)
        font_buffer = load_font(34)
        font_status = load_font(22)

        start_x, start_y = 110, 145
        cell_w, cell_h = 155, 70
        gap = 8

        display_text = "".join(self.text_buffer[-20:])
        draw.text((70, 45), "入力: " + display_text, fill=(0, 255, 128), font=font_buffer)

        for r, row_values in enumerate(self.matrix):
            for c, char_text in enumerate(row_values):
                x1 = start_x + c * cell_w
                y1 = start_y + r * cell_h
                x2 = x1 + cell_w - gap
                y2 = y1 + cell_h - gap

                is_cursor = r == self.row and c == self.col
                fill = (245, 245, 245) if is_cursor else (20, 20, 20)
                outline = (255, 255, 255) if is_cursor else (100, 100, 100)
                text_fill = (0, 0, 0) if is_cursor else (255, 255, 255)

                draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=3)
                tw, th = text_size(draw, char_text, font_cell)
                tx = x1 + ((x2 - x1) - tw) / 2
                ty = y1 + ((y2 - y1) - th) / 2 - 3
                draw.text((tx, ty), char_text, fill=text_fill, font=font_cell)

        selected = self.matrix[self.row][self.col]
        status = "0:待機  1:右  2:下  3:決定  /  現在位置: " + selected
        draw.text((70, 535), status, fill=(190, 190, 190), font=font_status)

        return pil_to_cv2(img)


def predict_label(model, data):
    pred = model.predict([data / 255.0])
    pred_classes = np.argmax(pred, axis=1)
    return int(pred_classes[0])


def main():
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    model = load_model(str(MODEL_PATH))
    controller = MatrixInputController(CHAR_MATRIX, DEBOUNCE_SECONDS)

    data_buffer = []
    digit_count = 0
    comma_count = 0
    sample_count = 0

    while True:
        s = ser.readline()
        try:
            val = s.decode("utf-8").rstrip(",\r").replace("\r", "")
        except UnicodeDecodeError:
            continue

        for i in range(len(val)):
            digit_count += 1

            if val[i] == ",":
                comma_count += 1

                if comma_count == 2:
                    data_buffer.append(int(val[i + 1 - digit_count : i]))

                if comma_count == 3:
                    data_buffer.append(int(val[i + 1 - digit_count : i]))
                    comma_count = 0
                    sample_count += 1

                digit_count = 0

            if sample_count == 50:
                data = np.array(data_buffer[0:100])
                data_buffer = []
                sample_count = 0

                predicted_label = predict_label(model, data)
                controller.process_label(predicted_label)

                cv2.imshow(WINDOW_NAME, controller.draw())
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    ser.close()
                    cv2.destroyAllWindows()
                    return


if __name__ == "__main__":
    main()
