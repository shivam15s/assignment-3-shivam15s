import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *

np.random.seed(42)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
y = pd.Series(data.target)

folds = 3
fold_size = X.shape[0]//folds
datasets = [X.iloc[fold_size*i: fold_size*(i+1)] for i in range(folds)]


print("\n----------Gradient Descent (Formula vs Autograd)----------")
for fit_intercept in [True]:
    LR = LogisticRegression(fit_intercept=fit_intercept)
    #LR = LogisticRegression(fit_intercept=fit_:intercept, regularization='L1', reg_lambda=5)
    #LR = LogisticRegression(fit_intercept=fit_intercept, regularization='L2', reg_lambda=0.5)
    
    LR.fit_vectorised(X, y, X.shape[0], n_iter = 100, lr=1)
    y_hat = LR.predict(X)
    print("Gradient Descent using Formula: ", accuracy(y_hat, y))

    LR.fit_autograd(X, y, X.shape[0], n_iter = 100, lr=1) 
    y_hat = LR.predict(X)
    print("Gradient Descent using Autograd", accuracy(y_hat, y))


#3 folds cross validation
print("\n----------3 Folds Accuracy----------")
fold_acc = []
for itr1 in range(folds):
    test = datasets[itr1]
    frames1 = []
    for j in range(folds):
        if j!=itr1:
            frames1.append(datasets[j])
    """
    Divides dataset into folds_outer parts. Takes folds - 1 parts for Training
    and 1 part for Testing
    """
    #Creates dataset of fold_outer - 1 sub-parts of data
    X_t = pd.concat(frames1).sort_index()
    y_t = y[X_t.index].reset_index(drop=True)
    X_t = X_t.reset_index(drop=True)

    LR = LogisticRegression()
    LR.fit_vectorised(X_t, y_t, X_t.shape[0], n_iter = 100, lr=2)

    y_hat = LR.predict(datasets[itr1])
    
    curr_acc = accuracy(y_hat, y[datasets[itr1].index])
    fold_acc.append(curr_acc)

    print("Test fold {}: ".format(itr1+1), curr_acc)

print("Average accuracy: ", np.mean(fold_acc))


print("\n----------Decision Boundary----------")
X_small = X.iloc[:, :2]
LR = LogisticRegression()
LR.fit_vectorised(X_small, y, X_small.shape[0], n_iter=200, lr=2)
y_hat = LR.predict(X_small)
print("Accuracy:", accuracy(y_hat, y))
fig = LR.plot_decision_boundary()
fig.savefig("plots/q1_d.png")