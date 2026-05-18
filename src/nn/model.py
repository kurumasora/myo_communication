from nn.layer import layer
import numpy as np

#DNNを表現するmodelクラス
class model:
    def __init__(self):
        self.layer_lists = [] #層を保持するリスト
        self.batch_size = 0 #バッチサイズ
        self.epochs = 0 #学習回数
        self.lr = 0.01 #学習係数
        self.interval = 10 #学習の経過を定期的に表示する

    #層の追加
    def add(self, input_node, output_node, activation_function):
        #layerクラスのオブジェクトを作成
        l= layer(input_node, output_node, activation_function)
        self.layer_lists.append(l)
    
    
    def summary(self):
        for l in self.layer_lists: 
            print("I:{}, O:{}, A:{}".format(l.input_node, 
                                            l.output_node, 
                                            l.activation_function))
    
    #順伝番
    def forwards(self, x):
        for l in self.layer_lists:
            x = l.forward(x)
        return x

    #逆伝番
    def backwards(self, y, x):
        for l in self.layer_lists[::-1]:
            x = l.backward(y, x)
        return x

    #誤差の計算
    def get_error(self, y, t, data_count):
        #交差エントロピー誤差
        if self.loss == "cross_entropy_error":
            return -np.sum(t*np.log(y+1e-7))/data_count
        elif self.loss == "squared_error":
            return 1.0/2.0 * np.sum(np.square(y-t))/data_count
        

    #重みとバイアスの更新
    def update_wb(self):
        for l in self.layer_lists:
            l.update(self.lr)

    #学習
    def fit(self, input_train, correct_train, batch_size,
            epochs, loss, lr):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.loss = loss

        #データの数
        n_train = input_train.shape[0]

        #誤差の記録用
        train_error_x = []
        train_error_y = []

        #学習と経過の記録
        n_batch = n_train // self.batch_size

        for i in range(self.epochs):
            index_random = np.arange(n_train)
            np.random.shuffle(index_random)
            print("Epoch:{}/{}".format(i+1, self.epochs,), end="")
            for j in range(n_batch):
                if j%100==0:
                    print(".", end="")
                #ミニバッチの取り出し
                mb_index = index_random[j*batch_size:(j+1)*batch_size]
                x=input_train[mb_index, :]
                t=correct_train[mb_index, :]

                #順伝番
                y = self.forwards(x)

                #逆伝番
                self.backwards(y, t)

                #重みとバイアスの更新
                self.update_wb()

            y = self.forwards(input_train)
            error_train = self.get_error(y, correct_train, n_train)

            #誤差の記録
            train_error_x.append(i)
            train_error_y.append(error_train)

            #経過の表示
            print("Error_train:{:5f}".format(error_train))
        return {"train_error_x":train_error_x,
                "train_error_y":train_error_y}
    # 推論
    def predict(self, x):
        y = self.forwards(x)
        return y
