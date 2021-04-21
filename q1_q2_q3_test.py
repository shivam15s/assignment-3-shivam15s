
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import MinMaxScaler
from logisticRegression.logisticRegression import LogisticRegression
from logisticRegression.kClassLogisticRegression import kClassLogisticRegression
from mlp.mlp import MLP
from metrics import *

np.random.seed(42)

N = 30
P = 5
#data = load_breast_cancer()
data = load_digits()
X = pd.DataFrame(data.data, columns=data.feature_names)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = pd.Series(data.target)

print(X.shape, y.shape)
m1 = MLP(X, y, [20], ['sigmoid'], 10)
m1.fit(X.shape[0], n_iter=400, lr=10)

y_hat = m1.predict(X)
print(y_hat)
print(accuracy(y_hat, y))
#for fit_intercept in [True, False]:
    ##LR = LogisticRegression(fit_intercept=fit_intercept)
    ##LR = LogisticRegression(fit_intercept=fit_intercept, regularization='L2', reg_lambda=0.5)
    #LR = kClassLogisticRegression(fit_intercept=fit_intercept)
    ##LR.fit_non_vectorised(X, y, X.shape[0], n_iter = 100) # here you can use fit_non_vectorised / fit_autograd methods
    
    #LR.fit_vectorised(X, y, X.shape[0], n_iter = 100)

    ##LR.fit_autograd(X, y, X.shape[0], n_iter = 100) 
    ##LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    #y_hat = LR.predict(X)
    ##for i in range(len(y_hat)):
        ##print(y_hat[i], y[i])
    #print(accuracy(y_hat, y))
    ##print('RMSE: ', rmse(y_hat, y))
    ##print('MAE: ', mae(y_hat, y))
