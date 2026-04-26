import numpy as np

#層を表現するlayerクラス
class layer:
    def __init__(self, input_node, output_node, activation_function):
        self.input_node =input_node
        self.output_node = output_node
        self.wb_width = 0.1
        self.w =self.wb_width * np.random.randn(input_node, output_node)
        self.b = self.wb_width * np.random.randn(output_node)
        self.activation_function = activation_function

    #順伝番
    def forward(self, x):
        #入力値
        self.x = x
        #ノード値
        self.nout = np.dot(x, self.w) + self.b
        #活性化関数
        if self.activation_function == "ReLU":
            y = np.where(self.nout <= 0, 0, self.nout)
        elif self.activation_function == "softmax":
            y = np.exp(self.nout)/np.sum(np.exp(self.nout), axis=1, keepdims=True)
        return y

    #逆伝番
    def backward(self, y, grad_y):
        if self.activation_function == "ReLU":
            delta = grad_y * np.where(self.nout <= 0, 0, 1)
        elif self.activation_function == "softmax":
            delta = y - grad_y
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
        return self.grad_x

    #重みとバイアスの更新
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b





