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
folds_outer = 5
folds_inner = 5
fold_size = X.shape[0]//folds_outer
datasets = [X.iloc[fold_size*i: fold_size*(i+1)] for i in range(folds_outer)]
#5 folds cross validation
for itr1 in range(folds_outer):
    X_test = datasets[itr1]
    y_test = y[X_test.index].reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    frames1 = []
    for j in range(folds_outer):
        if j!=itr1:
            frames1.append(datasets[j])
    """
    Divides dataset into folds_outer parts. Takes fold_outer - 1 parts for Training + Validation
    and 1 part for Testing
    """
    #Creates dataset of fold_outer - 1 sub-parts of data
    X_t = pd.concat(frames1).sort_index()
    y_t = y[X_t.index].reset_index(drop=True)
    X_t = X_t.reset_index(drop=True)
    ff_size = X_t.shape[0]//folds_inner
    dd = [X_t.iloc[ff_size*i: ff_size*(i+1)] for i in range(folds_inner)]
    acc = {}
    print("Outer Fold {}: ".format(itr1 + 1))

    reg_lambda_l2 = 0.2
    reg_lambda_l2_inc = 0.2

    reg_lambda_l1 = 0.1
    reg_lambda_l1_inc = 0.1

    acc_l2 = []
    acc_l1 = []
    for itr2 in range(folds_inner):
        X_val = dd[itr2]
        y_val = y_t[X_val.index].reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        frames2 = []
        for j in range(folds_inner):
            if j!=itr2:
                frames2.append(dd[j])
        """
        Divides data into fold_inner parts. Take fold_inner - 1 parts for training.
        1 part of the data is taken as the validation set
        """
        X_tt = pd.concat(frames2).sort_index()
        y_tt = y_t[X_tt.index].reset_index(drop=True)
        X_tt = X_tt.reset_index(drop=True)

        #LR = LogisticRegression()
        LR1 = LogisticRegression(regularization='L1', reg_lambda=reg_lambda_l1)

        LR1.fit_autograd(X_tt, y_tt, X_tt.shape[0], n_iter = 100, lr=2) 
        y_hat1 = LR1.predict(X_val)

        LR2 = LogisticRegression(regularization='L2', reg_lambda=reg_lambda_l2)
        LR2.fit_autograd(X_tt, y_tt, X_tt.shape[0], n_iter = 100, lr=2) 
        y_hat2 = LR2.predict(X_val)

        curr_acc1 = accuracy(y_hat1, y_val)
        curr_acc2 = accuracy(y_hat2, y_val)

        print("\tInner Fold {}: ".format(itr2+1))
        print("\t\tL1 Lambda = {:.2f}: ".format(reg_lambda_l1), curr_acc1)
        print("\t\tL2 Lambda = {:.2f}: ".format(reg_lambda_l2), curr_acc2)
        acc_l1.append(curr_acc1)
        acc_l2.append(curr_acc2)

        reg_lambda_l1 += reg_lambda_l1_inc
        reg_lambda_l2 += reg_lambda_l2_inc

    
    best_l1_lambda = (np.argmax(acc_l1) + 1) * reg_lambda_l1_inc
    best_l2_lambda = (np.argmax(acc_l2) + 1) * reg_lambda_l2_inc

    print("\tBest L1 Lambda for Outer Fold {}: {:.2f}".format(itr1+1, best_l1_lambda))
    LR1 = LogisticRegression(regularization='L1', reg_lambda=best_l1_lambda)
    LR1.fit_autograd(X_t, y_t, X_t.shape[0], n_iter = 100, lr=2) 
    y_hat = LR1.predict(X_test)
    print("\tTest accuracy using L1 Lambda = {:.2f}: ".format(best_l1_lambda), accuracy(y_hat, y_test))

    print("\tBest L2 Lambda for Outer Fold {}: {:.2f}".format(itr1+1, best_l2_lambda))
    LR2 = LogisticRegression(regularization='L2', reg_lambda=best_l2_lambda)
    LR2.fit_autograd(X_t, y_t, X_t.shape[0], n_iter = 100, lr=2) 
    y_hat = LR2.predict(X_test)
    print("\tTest accuracy using L2 Lambda = {:.2f}: ".format(best_l2_lambda), accuracy(y_hat, y_test))