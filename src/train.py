import numpy as np
from pathlib import Path
from nn.model import model
from nn.util import to_categorical, save_model

#CSSのデータを読み込む
#教師データ:train_data
#教師ラベル:train_truth
#正解データ:validation_data
#正解ラベル:validation_truth

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "teacher_data"

with open(DATA_DIR / "data.csv", encoding="utf-8") as f:
    train_data_a = f.read()
    train_data_1 = []
    count = 0
    for i in range(len(train_data_a)):
        count = count + 1
        if (train_data_a[i] == ','):
            train_data_1.append(int(train_data_a[i+1-count:i]))
            count = 0
        if (train_data_a[i] == '\n'):
            train_data_1.append(int(train_data_a[i+1-count:i]))
            count = 0
    train_data = np.array(train_data_1)

with open(DATA_DIR / "label.csv", encoding="utf-8") as f:
    train_label_a = f.read()
    train_label_b = train_label_a.replace("\ufeff", "")
    train_label_c = train_label_b.replace("\n", "")
    train_label_1 = []
    for i in range (len(train_label_c)):
        train_label_1.append(int(train_label_c[i]))
    train_label = np.array(train_label_1)
#データの形状を変更
train_data = train_data.reshape(train_label.shape[0], -1)
#データの正規化
train_data = train_data/255.0

#onehot表現
#ラベルを数字にする．if文で1なら'あ', 2なら'い'みたいな感じにする．
train_truth = to_categorical(train_label, 4)

#3層のDNNを作成する
m = model()
#入力層と中間層の追加
m.add(100, 50, "ReLU")
#出力層の追加
m.add(50, 4, "softmax")

history =m.fit(input_train=train_data,
                correct_train=train_truth,
                batch_size = 20, 
                epochs=30,
                loss = "cross_entropy_error", 
                lr = 0.001,
                )

#学習モデルの保存
save_model('train.model', m)