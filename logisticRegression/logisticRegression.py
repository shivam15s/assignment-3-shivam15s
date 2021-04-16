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
            self.coef_ -= alpha * np.dot(X_batches[i%n_batches].T, (y_hat - y_batches[i%n_batches]))
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
            grad_mse = grad(self.xentropy)
            grad_i = grad_mse(self.coef_)
            self.coef_ -= alpha*grad_i
        
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

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        if self.fit_intercept:
            X = np.c_[np.ones((X.shape[0], 1)), np.array(X)]
        else:
            X = np.array(X)

        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y


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

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        figure = plt.figure(figsize=(10, 7))
        x_min, x_max = self.coef_[0] - 10, self.coef_[0] + 10
        y_min, y_max = self.coef_[1] - 10, self.coef_[1] + 10
        h = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
        Z = np.apply_along_axis(self.rss, 1, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax = figure.add_subplot(111, projection='3d')

        print(t_0, t_1, self.rss(np.array([t_0, t_1])))

        if self.t_0_list == None:
            self.t_0_list = []
            self.t_1_list = []
        self.t_0_list.append(t_0)
        self.t_1_list.append(t_1)
        for i in range(len(self.t_0_list)):
            ax.scatter(self.t_0_list[i], self.t_1_list[i], self.rss(np.array([self.t_0_list[i], self.t_1_list[i]]))+100, s=50, color="red")
            
        #ax.scatter(t_0, t_1, self.rss(np.array([t_0, t_1]))+100, s=100, color="red")
        surf = ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.4)
        figure.colorbar(surf, shrink=0.5, aspect=10, pad=0.15)
        ax.set_xlabel("b", fontsize=13, labelpad=6)
        ax.set_ylabel("m", fontsize=13, labelpad=6)
        ax.set_zlabel("RSS", fontsize=13, labelpad=12)
        ax.set_title("RSS = {}".format(self.rss(np.array([t_0, t_1]))))
        return figure


    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        figure = plt.figure(figsize=(10, 7))
        plt.scatter(X, y, color='blue')
        plt.plot(X, t_0 + X*t_1, color='orange')
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(y)-1, np.max(y)+1)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("m = {}   b = {}".format(t_1[0], t_0[0]))
        return figure

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        figure = plt.figure(figsize=(10, 7))
        x_min, x_max = self.coef_[0] - 10, self.coef_[0] + 10
        y_min, y_max = self.coef_[1] - 10, self.coef_[1] + 10
        h = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
        Z = np.apply_along_axis(self.rss, 1, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax = figure.add_subplot(111)

        print(t_0, t_1, self.rss(np.array([t_0, t_1])))

        if self.t_0_list == None:
            self.t_0_list = []
            self.t_1_list = []
        self.t_0_list.append(t_0)
        self.t_1_list.append(t_1)
        for i in range(len(self.t_0_list)-1):
            plt.arrow(self.t_0_list[i][0], self.t_1_list[i][0], self.t_0_list[i+1][0] - self.t_0_list[i][0], self.t_1_list[i+1][0] - self.t_1_list[i][0], width=0.1)
            #ax.scatter(self.t_0_list[i], self.t_1_list[i], self.rss(np.array([self.t_0_list[i], self.t_1_list[i]]))+100, s=50, color="red")
            
        #ax.scatter(t_0, t_1, self.rss(np.array([t_0, t_1]))+100, s=100, color="red")
        surf = ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.4)
        figure.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel("b", fontsize=13, labelpad=6)
        ax.set_ylabel("m", fontsize=13, labelpad=6)
        ax.set_title("RSS = {}".format(self.rss(np.array([t_0, t_1]))))
        return figure

