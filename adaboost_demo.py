import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

class adaboost:
    def __init__(self, T):
        self.T = T
        self.alphas = []
        self.errors = []
        self.weak_learn = []
    def calculate_error(self,y, y_pred, weight_t):
        return sum(weight_t * y * y_pred)
    def calculate_alpha(self,error):
        return np.log((1 + error) / (1 - error)) / 2
    def new_weights_t(self, alpha, weight_t, y, y_pred):
        normalization_function = sum(weight_t * np.exp(-1 * alpha * y * y_pred))
        return weight_t * np.exp(-1 * alpha * y * y_pred)/normalization_function

    def fit(self, X, y):
        for t in range (self.T):
            if t == 0:
                weight_t = np.ones(len(y))/len(y) # waga 1/T
            else:
                weight_t = self.new_weights_t(alpha, weight_t, y, y_pred)

            weak_learn = DecisionTreeClassifier(max_depth=1)
            weak_learn.fit(X,y,weight_t)
            y_pred = weak_learn.predict(X)
            self.weak_learn.append(weak_learn)

            error = self.calculate_error(y, y_pred, weight_t)
            self.errors.append(error)
            alpha = self.calculate_alpha(error)
            self.alphas.append(alpha)

    #demo
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for t in range(self.T):
            y_pred_t = self.weak_learn[t].predict(X) * self.alphas[t]
            y_pred += y_pred_t
        y_pred = np.sign(y_pred).astype(int)
        return y_pred

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)

for T in [50, 100]:
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

    accuracies = []

    for test_index, train_index in rskf.split(X, y):
        X_test, X_train = X[test_index], X[train_index]
        y_test, y_train = y[test_index], y[train_index]

        clf = adaboost(T=T)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_true=y_test, y_pred=y_pred))

    print()
    print(f"T={T}")
    for i, acc in enumerate(accuracies, start=1):
        print(f"Fold {i}: {acc}")

    print(f"Wartość oczekiwana: {np.mean(accuracies):.3f}")
    print(f"Odchylenie standardowe: {np.std(accuracies):.3f}")
