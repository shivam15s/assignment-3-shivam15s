import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from mlp.mlp import MLP
from metrics import *

np.random.seed(42)

data = load_digits()
X = pd.DataFrame(data.data, columns=data.feature_names)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
y = pd.Series(data.target)


print("\n----------Digits Dataset----------")
cv = KFold(n_splits=3, shuffle=True, random_state=42)
digits_acc = []
ind = 0
for train_ix, test_ix, in cv.split(X):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]

    X_train = X_train.to_numpy()
    m1 = MLP(X_train, y_train.reset_index(drop=True), [20], ['sigmoid'], 10)
    m1.fit(X_train.shape[0], n_iter=100, lr=10)

    y_hat = m1.predict(X_test.reset_index(drop=True))
    curr_acc = accuracy(y_hat, y_test)
    digits_acc.append(curr_acc)
    print("Fold {} accuracy: ".format(ind+1), curr_acc)
    ind += 1
print("Average accuracy: ", np.mean(digits_acc))


data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
y = pd.Series(data.target)

print("\n----------Boston Dataset----------")
cv = KFold(n_splits=3, shuffle=True, random_state=42)
boston_err = []
ind = 0
for train_ix, test_ix, in cv.split(X):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]

    X_train = X_train.to_numpy()
    m1 = MLP(X_train, y_train.reset_index(drop=True), [6, 4], ['sigmoid', 'sigmoid'], 1, regression=True)
    m1.fit(X_train.shape[0], n_iter=200, lr=0.1)

    y_hat = m1.predict(X_test.reset_index(drop=True))
    curr_err = rmse(y_hat, y_test)
    boston_err.append(curr_err)
    print("Fold {} RMSE: ".format(ind+1), curr_err)
    ind += 1
print("Average RMSE: ", np.mean(boston_err))
#X = X.to_numpy()
#m1 = MLP(X, y, [6, 4], ['sigmoid', 'sigmoid'], 1, regression=True)
#m1.fit(X.shape[0], n_iter=200, lr=0.1)
#y_hat = m1.predict(X)
##print(y_hat)
#print(rmse(y_hat, y))
