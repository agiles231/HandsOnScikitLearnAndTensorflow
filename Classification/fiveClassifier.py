from sklearn.linear_model import SGDClassifier
import pullMnist as pm
import fold
import split_train_test as stt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
import numpy as np

mnist = pm.pull_mnist()
X_train, X_test, y_train, y_test = stt.split_train_test(mnist)
X_train, y_train = fold.foldMnistData(X_train, y_train)
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
some_digit = mnist["data"][36000]

print(sgd_clf.predict([some_digit]))

cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cvs)

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cvs = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cvs)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))