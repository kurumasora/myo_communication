import numpy as np
import pickle

#onehot表現
def to_categorical(label_data, num_classes):
    onehot = np.zeros((len(label_data), num_classes))

    for i in range(len(label_data)):
        onehot[i, label_data[i]]=1
    return onehot

def save_model(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        m = pickle.load(f)
    return m