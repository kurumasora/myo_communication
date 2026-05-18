# -*- coding: utf-8 -*-
import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# 1. 2次元アルファベット配列の定義（5行×6列）
CHAR_MATRIX = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "SPC", "BS", "CLR", "End"]
]

MAX_ROWS = len(CHAR_MATRIX)
MAX_COLS = len(CHAR_MATRIX[0])

# カーソルの初期位置
current_row = 0
current_col = 0

# 確定された文字を蓄積するバッファ
text_buffer = []

def pil_to_cv2(image):
    """PILイメージをOpenCV形式に変換する関数"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 3 and new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]  # RGBからBGRへ
    return new_image

def draw_matrix_ui(row_idx, col_idx, buffer_text):
    """2次元配列、カーソル、入力文字バッファを画面に描画する関数"""
    # 画面サイズ (論文の仕様に合わせて 横1000 × 縦600)
    win_w, win_h = 1000, 600
    img = PIL.Image.new("RGB", (win_w, win_h), (30, 30, 30))  # ダークグレー背景
    draw = PIL.ImageDraw.Draw(img)
    
    # Mac/Windows共通で確実にロードできるデフォルトフォント（サイズ指定不可のため擬似サイズアップ対応）
    # 代わりにシステム標準のSans-Serif（Arial等）をロード（Macで確実に入っている英字フォント）
    try:
        font_main = PIL.ImageFont.truetype("Arial.ttf", 36)
        font_sub = PIL.ImageFont.truetype("Arial.ttf", 24)
        font_key = PIL.ImageFont.truetype("Arial.ttf", 22)  # 機能キー用（SPCなど）
    except:
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

            # 機能キー（3文字以上のラベル）か、通常のアルファベットかでフォントサイズを切り替え
            current_font = font_key if len(char_text) > 1 else font_main
            
            # 文字をセルの真ん中に寄せるための簡易座標微調整
            offset_x = 40 if len(char_text) > 1 else 50
            offset_y = 18 if len(char_text) > 1 else 10

            if r == row_idx and c == col_idx:
                # 現在選択されているカーソル位置（白背景、黒文字）
                draw.rectangle([x, y, x + cell_w - 15, y + cell_h - 15], fill=(255, 255, 255))
                draw.text((x + offset_x, y + offset_y), char_text, fill=(0, 0, 0), font=current_font)
            else:
                # 通常状態（暗い枠線、白文字）
                draw.rectangle([x, y, x + cell_w - 15, y + cell_h - 15], outline=(100, 100, 100), width=2)
                draw.text((x + offset_x, y + offset_y), char_text, fill=(255, 255, 255), font=current_font)

    # 確定したテキストを表示するエリア（上部）
    display_text = "TEXT BUFFER: " + "".join(buffer_text)
    draw.text((70, 50), display_text, fill=(0, 255, 128), font=font_sub)
    
    # 下部の操作案内案内
    instruction_text = "[CONTROLS] Arrow Keys or WASD: Move | Enter: Select | Q: Quit"
    draw.text((70, 530), instruction_text, fill=(180, 180, 180), font=font_sub)

    return pil_to_cv2(img)

# メインループ
is_running = True
while is_running:
    # 画面を描画して表示
    cv2_image = draw_matrix_ui(current_row, current_col, text_buffer)
    cv2.imshow('Alphabet UI Simulator', cv2_image)
    
    # キー入力を待機 (Macのキーコードのバグを回避するため waitKeyEx を使用)
    key = cv2.waitKeyEx(50)
    
    # 終了コマンド (Qキー または Escキー)
    if key == ord('q') or key == ord('Q') or key == 27:
        break
        
    # --- 十字キー および WASD 移動ロジック ---
    elif key == 63232 or key == ord('w') or key == ord('W'):  # 上
        if current_row > 0: current_row -= 1
    elif key == 63233 or key == ord('s') or key == ord('S'):  # 下
        if current_row < MAX_ROWS - 1: current_row += 1
    elif key == 63234 or key == ord('a') or key == ord('A'):  # 左
        if current_col > 0: current_col -= 1
    elif key == 63235 or key == ord('d') or key == ord('D'):  # 右
        if current_col < MAX_COLS - 1: current_col += 1
            
    # --- 文字確定・機能キー処理 (Enter / Return キー) ---
    elif key == 13 or key == 3:
        selected_char = CHAR_MATRIX[current_row][current_col]
        
        if selected_char == "SPC":
            text_buffer.append(" ")
        elif selected_char == "BS":
            if len(text_buffer) > 0:
                text_buffer.pop()  # 末尾の一文字を削除
        elif selected_char == "CLR":
            text_buffer.clear()  # バッファを全消去
        elif selected_char == "End":
            is_running = False  # アプリ終了
        else:
            text_buffer.append(selected_char)  # 通常のアルファベットをバッファに追加

cv2.destroyAllWindows()