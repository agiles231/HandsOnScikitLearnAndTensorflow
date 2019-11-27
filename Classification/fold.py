import numpy as np
import pullMnist as pm
import split_train_test as stt

def foldMnistData(X_train, y_train):
    shuffle_index = np.random.permutation(60000)
    return X_train[shuffle_index], y_train[shuffle_index]

mnist = pm.pull_mnist()
X_train, X_test, y_train, y_test = stt.split_train_test(mnist)
X_train, y_train = foldMnistData(X_train, y_train)
