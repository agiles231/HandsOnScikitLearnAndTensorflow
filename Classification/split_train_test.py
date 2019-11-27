
def split_train_test(mnist):
    X, y = mnist["data"], mnist["target"]
    return X[:60000], X[60000:], y[:60000], y[60000:]