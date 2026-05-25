import serial
import numpy as np
from nn.util import load_model
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import os

#シリアルポートcom9,シリアルポーレート115200
vtty_path = os.path.expanduser('~/vtty')
ser = serial.Serial(vtty_path, 115200, timeout = 1)

#データを格納する配列
data_1 = []

#キャラクタを格納する配列
character_1 = []

#カウント
count = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0

#mv2に文字を出力する
def draw_text_at_center(img, text_1):
    global count_5
    draw = PIL.ImageDraw.Draw(img)

    # フォントの設定
    font_ttf ='C:/Windows/Fonts/msgothic.ttc'
    draw.font = PIL.ImageFont.truetype(font_ttf, 200)

    # テキストの描画(BGRの順)
    #(x,y)の場所に,textを白(255, 255, 255)で出力する
    draw.text((0,200), text_1, (255, 255, 255))

#PIL->OpenCVに変換する関数を呼び出し
def pil_to_cv2(image):
    # numpy.ndarrayに変換
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    return new_image

#予測関数
def predict():
    #globalで関数内で変数が変化しても関数外にも適応させる
    global count_4
    global count_5

    #入力データが教師データのどれに近いのか
    #正規化
    pred = m.predict([data/255.0])
    pred_classes = np.argmax(pred, axis=1)

    #ラベルがpred_1に格納される
    pred_1= int(pred_classes[0])
    #pred_1が0じゃなければ文字を出力
    #0は手を動かしていないときのラベル
    if (pred_1 != 0):
        #count_5が1なら character_1に文字を格納し1を足す
        if (pred_1 == 1):
            character_1.append("あ")
        elif (pred_1 == 2):
            character_1.append("い")
        else:
            character_1.append("う")

        #出力画面はキャラクタを5個までしかできないからそのカウント
        count_5 = count_5+1
        #次は1回待機させるためのカウント
        count_4 = 1

while True:
    #数値の読み取り
    s = ser.readline()
    #print(s)
    #\rの削除．読み取った数値には/rや,などいらんもんがある
    val = s.decode("utf-8").rstrip(",\r")
    val_1 = val.replace("\r", "")

    for i in range(len(val_1)):
        #val_1[i] == ','になるまでのカウントをとる．
        #格納した数値が2桁なのか3桁なのかを知りたい．
        count = count + 1

        if (val_1[i] == ','):
            #val_1[i] == ','になるたびにカウントする
            #','で区切られるデータが3つ取得するが必要なのは2つ目と3つ目
            count_1 = count_1 + 1

            if(count_1 == 2):
                #取得したデータの2つ目のデータをdata_1配列に追加
                data_1.append(int(val_1[i+1-count:i]))

            if(count_1 == 3):
                #取得したデータの3つ目のデータをdata_1配列に追加
                data_1.append(int(val_1[i+1-count:i]))

                #0に戻す．データは3個までしか取れない
                count_1 = 0

                #教師データは100だから入力データが100個取れるまでカウントする
                count_2 = count_2 + 1
            
            #0に戻す．上限254だから3桁以上になることはない
            count = 0
        
        #入力データが100個集ったら．2つ目と3つ目のデータをとってカウントするからcount_2が50で入力データ100
        if (count_2 == 50):
            #data_1は','で区切ってあるから','をなくし，配列化する
            data = np.array(data_1[0:100])

            #0に戻す．入力データが100以上だと学習できない
            count_2 = 0

            #初期化．次の入力データを格納するのに前の入力データを消去する
            data_1 = []

            #学習モデルの呼び出し
            m = load_model("train.model")

            #手を動かしたときに戻す動作が必要な場合があるためキャラクタを出力したら1回学習をせず間を作る
            if(count_4 == 0):
                #学習（予測関数）
                predict()

                #character_1に格納されたキャラクタをstr_list_1に格納
                str_list_1 = [str(i) for i in character_1]
                character_1_string = "".join(str_list_1)
                
                # 出力画面の生成
                size = (1000, 600)
                image = PIL.Image.new("RGB", size)

                # テキストを描画する関数を呼び出し
                draw_text_at_center(image, character_1_string)

                # PIL->OpenCVに変換する関数を呼び出し
                cv2_image = pil_to_cv2(image)

                # 描画
                cv2.imshow('test', cv2_image)
                cv2.waitKey(1000)

                # 画面を消去
                cv2.destroyAllWindows()

                #出力画面には5個までしか表示できないから5個表示したらcharacter_1を初期化してカウントを0に戻す
                if (count_5 == 5):
                    character_1=[]
                    count_5 =0

            #動作後の待機
            else:
                count_4 = 0








