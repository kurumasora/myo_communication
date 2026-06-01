# -*- coding: utf-8 -*-
import cv2
import numpy as np

CHAR_MATRIX = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "SPC", "BS", "CLR", "End"]
]

MAX_ROWS = len(CHAR_MATRIX)
MAX_COLS = len(CHAR_MATRIX[0])

current_row = 0
current_col = 0
text_buffer = []

FONT           = cv2.FONT_HERSHEY_SIMPLEX
SCALE_MAIN     = 1.1   # A-Z
SCALE_KEY      = 0.65  # SPC / BS / CLR / End
THICKNESS_MAIN = 2
THICKNESS_KEY  = 1

WIN_W, WIN_H   = 1000, 600
START_X        = 70
START_Y        = 150
CELL_W         = 140
CELL_H         = 70
GAP            = 15    # セル間の余白（内寸 = CELL_W-GAP × CELL_H-GAP）


def draw_matrix_ui(row_idx, col_idx, buffer_text):
    img = np.full((WIN_H, WIN_W, 3), 30, dtype=np.uint8)

    for r in range(MAX_ROWS):
        for c in range(MAX_COLS):
            char_text = CHAR_MATRIX[r][c]
            x1 = START_X + c * CELL_W
            y1 = START_Y + r * CELL_H
            x2 = x1 + CELL_W - GAP
            y2 = y1 + CELL_H - GAP

            is_key   = len(char_text) > 1
            scale    = SCALE_KEY      if is_key else SCALE_MAIN
            thick    = THICKNESS_KEY  if is_key else THICKNESS_MAIN

            (tw, th), _ = cv2.getTextSize(char_text, FONT, scale, thick)
            tx = x1 + ((x2 - x1) - tw) // 2
            ty = y1 + ((y2 - y1) + th) // 2

            if r == row_idx and c == col_idx:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                cv2.putText(img, char_text, (tx, ty), FONT, scale, (0, 0, 0), thick)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 2)
                cv2.putText(img, char_text, (tx, ty), FONT, scale, (255, 255, 255), thick)

    buf_str = "TEXT BUFFER: " + "".join(buffer_text)
    cv2.putText(img, buf_str, (70, 80), FONT, 0.75, (128, 255, 0), 2)

    inst = "[CONTROLS] Arrow Keys or WASD: Move | Enter: Select | Q: Quit"
    cv2.putText(img, inst, (70, 555), FONT, 0.55, (180, 180, 180), 1)

    return img


is_running = True
while is_running:
    cv2.imshow("Alphabet UI Simulator", draw_matrix_ui(current_row, current_col, text_buffer))
    key = cv2.waitKeyEx(50)

    if key in (ord('q'), ord('Q'), 27):
        break
    elif key in (63232, ord('w'), ord('W')):
        if current_row > 0: current_row -= 1
    elif key in (63233, ord('s'), ord('S')):
        if current_row < MAX_ROWS - 1: current_row += 1
    elif key in (63234, ord('a'), ord('A')):
        if current_col > 0: current_col -= 1
    elif key in (63235, ord('d'), ord('D')):
        if current_col < MAX_COLS - 1: current_col += 1
    elif key in (13, 3):
        selected = CHAR_MATRIX[current_row][current_col]
        if selected == "SPC":
            text_buffer.append(" ")
        elif selected == "BS":
            if text_buffer: text_buffer.pop()
        elif selected == "CLR":
            text_buffer.clear()
        elif selected == "End":
            is_running = False
        else:
            text_buffer.append(selected)

cv2.destroyAllWindows()
