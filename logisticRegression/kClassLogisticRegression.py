import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from autograd import grad
import autograd.numpy as anp
# Import Autograd modules here

class kClassLogisticRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.X = None
        self.y = None
        self.t_0_list = None
        self.t_1_list = None



    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        alpha = lr
        N = X.shape[0]
        d = X.shape[1]+self.fit_intercept
        k = np.unique(y).shape[0]
        self.coef_ = np.ones((d, k))
        if self.fit_intercept:
            X = np.c_[np.ones((N, 1)), np.array(X)]
        else:
            X = np.array(X)
        y = np.array(y).reshape(N, 1)
        n_batches = (X.shape[0] + batch_size - 1)//batch_size
        X_batches = [X[(i)*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        y_batches = [y[(i)*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        for i in range(n_iter):
            if (lr_type == "inverse"):
                alpha = lr/(i+1)

            self.X = X_batches[i%n_batches]
            self.y = y_batches[i%n_batches]

            y_hat_t = np.dot(self.X, self.coef_)
            y_hat_t -= np.max(y_hat_t, axis=1)[:, np.newaxis]
            y_hat = np.exp(y_hat_t)/np.sum(np.exp(y_hat_t), axis=1)[:, np.newaxis]

            for j in range(k):
                self.coef_[:, j] += alpha * np.dot(self.X.T, (self.y==j).astype(float)[:, 0] - y_hat[:, j])

        self.X = X
        self.y = y


    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        N = X.shape[0]
        d = X.shape[1]+self.fit_intercept
        k = np.unique(y).shape[0]
        alpha = lr
        self.coef_ = np.ones((d, k))
        if self.fit_intercept:
            X = np.c_[np.ones((N, 1)), np.array(X)]
        else:
            X = np.array(X)
        y = np.array(y).reshape(N, 1)
        n_batches = (X.shape[0] + batch_size - 1)//batch_size
        X_batches = [X[(i)*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        y_batches = [y[(i)*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        for i in range(n_iter):
            if (lr_type == "inverse"):
                alpha = lr/(i+1)

            self.X = X_batches[i%n_batches]
            self.y = y_batches[i%n_batches]
            grad_xent = grad(self.xentropy)
            grad_i = grad_xent(self.coef_)
            self.coef_ -= alpha*grad_i
        
        self.X = X
        self.y = y


    def sigmoid(self, logy):
        return 1/(1+np.exp(-logy))

    def xentropy(self, coef):
        y_hat_t = anp.dot(self.X, coef)
        y_hat_t -= anp.max(y_hat_t, axis=1)[:, anp.newaxis]
        y_hat = anp.exp(y_hat_t)/anp.sum(anp.exp(y_hat_t), axis=1)[:, anp.newaxis]

        cost = 0
        for i in range(self.X.shape[0]):
            cost -= anp.log(y_hat[i, self.y[i]])
        return cost


    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        N = X.shape[0]
        if self.fit_intercept:
            X = np.c_[np.ones((N, 1)), np.array(X)]
        else:
            X = np.array(X)

        y_hat_t = np.dot(self.X, self.coef_)
        y_hat_t -= np.max(y_hat_t, axis=1)[:, np.newaxis]
        y_hat = np.exp(y_hat_t)/np.sum(np.exp(y_hat_t), axis=1)[:, np.newaxis]

        return np.argmax(y_hat, axis=1)


