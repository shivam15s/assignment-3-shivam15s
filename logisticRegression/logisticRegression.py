import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from autograd import grad
import autograd.numpy as anp
# Import Autograd modules here

class LogisticRegression():
    def __init__(self, fit_intercept=True, regularization=None, reg_lambda = 0.1):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.X = None
        self.y = None



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
        self.coef_ = np.ones((d, 1))
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

            y_hat = self.sigmoid(np.dot(X_batches[i%n_batches], self.coef_))
            self.coef_ -= (alpha/X_batches[i%n_batches].shape[0]) * np.dot(X_batches[i%n_batches].T, (y_hat - y_batches[i%n_batches]))
            #self.coef_ -= 2 * alpha / (X_batches[i%n_batches].shape[0]) * X_batches[i%n_batches].T @ (y_hat -  y_batches[i%n_batches])
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
        alpha = lr
        self.coef_ = np.ones((d, 1))
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
            self.coef_ -= (alpha/self.X.shape[0])*grad_i
        
        self.X = X
        self.y = y


    def sigmoid(self, logy):
        return 1/(1+np.exp(-logy))

    def xentropy(self, coef):
        y_hat_t = anp.dot(self.X, coef)
        y_hat = 1/(1+anp.exp(-y_hat_t))

        cost = -1 * (anp.dot(self.y.T, anp.log(y_hat)) + anp.dot((1 - self.y).T, anp.log(1 - y_hat)))
        if self.regularization == 'L1':
            cost += self.reg_lambda * anp.abs(anp.sum(coef))
        elif self.regularization == 'L2':
            cost += self.reg_lambda * anp.dot(coef.T, coef)
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
        y_hat = np.dot(X, self.coef_)
        y_hat[y_hat>0] = 1
        y_hat[y_hat<0] = 0

        return y_hat.reshape((N, ))

    def plot_decision_boundary(self):
        # [1 x1 x2] * [w1 w2 w3].T = 0 => w1 + w2*x1 + w3*x2 = 0
        figure = plt.figure(figsize=(10, 7))
        w1, w2, w3 = self.coef_
        slope = -w2/w3
        intercept = -w1/w3
        x_min, x_max = -0.2, 1.2
        y_min, y_max = -1.8, 2.0

        xx = np.arange(x_min, x_max, 0.05)
        yy = slope*xx + intercept

        plt.plot(xx, yy, 'k', ls='--')
        plt.fill_between(xx, yy, y_min, color='tab:blue', alpha=0.2)
        plt.fill_between(xx, yy, y_max, color='tab:green', alpha=0.2)

        X_df = pd.DataFrame(self.X)
        label0 = X_df[self.y==0]
        label1 = X_df[self.y==1]

        plt.scatter(label0[1], label0[2], color='green', label='0')
        plt.scatter(label1[1], label1[2], color='blue', label='1')

        plt.xlabel("Attribute x1")
        plt.ylabel("Attribute x2")
        plt.title("Logistic Regression on Breast Cancer Dataset")
        #plt.show()
        return figure
