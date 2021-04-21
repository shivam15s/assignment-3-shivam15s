#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import math
from autograd import grad
import autograd.numpy as np
from tqdm import tqdm
from autograd.misc.optimizers import adam
# Import Autograd modules here

class MLP():
    def __init__(self, X, y, hl_sizes, activations, labels):
        self.X = X
        self.y = y
        self.params = []
        k = labels
        if len(hl_sizes)==0:
            self.params.append([np.ones(shape=(X.shape[1], k)), np.ones(shape=(k))])
        else:
            self.params.append([np.random.normal(size=(X.shape[1], hl_sizes[0])), np.random.normal(size=(hl_sizes[0]))])
            for i in range(1, len(hl_sizes)):
                self.params.append([np.random.normal(size=(hl_sizes[i-1], hl_sizes[i])), np.random.normal(size=(hl_sizes[i]))])
            self.params.append([np.random.normal(size=(hl_sizes[-1], k)), np.random.normal(size=(k))])


        #if len(hl_sizes)==0:
            #self.params.append([np.ones(shape=(X.shape[1], k)), np.ones(shape=(k))])
        #else:
            #self.params.append([np.ones(shape=(X.shape[1], hl_sizes[0])), np.ones(shape=(hl_sizes[0]))])
            #for i in range(1, len(hl_sizes)):
                #self.params.append([np.ones(shape=(hl_sizes[i-1], hl_sizes[i])), np.ones(shape=(hl_sizes[i]))])
            #self.params.append([np.ones(shape=(hl_sizes[-1], k)), np.ones(shape=(k))])

        self.activations = []
        for i in range(len(activations)):
            if activations[i] == 'relu':
                self.activations.append(self.ReLU)
            elif activations[i] == 'sigmoid':
                self.activations.append(self.sigmoid)
            else:
                self.activations.append(self.identity)

    
    def ReLU(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def identity(self, x):
        return x

    def softmax(self, x):
        x -= np.max(x, axis=1)[:, np.newaxis]
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:, np.newaxis]


    def forward_pass(self, params):
        computed = self.X

        for i in range(len(params)-1):
            computed = np.dot(computed, params[i][0]) + params[i][1].T
            computed = self.activations[i](computed)

        computed = np.dot(computed, params[-1][0]) + params[-1][1].T
        return self.softmax(computed)

    def xentropy(self, params):
        pred = self.forward_pass(params)
        cost = 0.0
        for i in range(self.X.shape[0]):
            cost -= np.log(pred[i, self.y.iloc[i]])
        print(cost)
        return cost

    def update(self, alpha):
        grad_xent = grad(self.xentropy)
        grad_curr = grad_xent(self.params)
            
        for i in range(len(self.params)):
            self.params[i][0] -= (alpha/self.X.shape[0]) * grad_curr[i][0]
            self.params[i][1] -= (alpha/self.X.shape[0]) * grad_curr[i][1]

    def fit(self, batch_size, n_iter=100, lr=0.01):
        self.batch_size = batch_size
        self.n_iter = n_iter
        X = self.X
        y = self.y
        n_batches = (self.X.shape[0] + batch_size - 1)//batch_size
        X_batches = [self.X[(i)*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        y_batches = [self.y[(i)*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        for i in tqdm(range(n_iter)):
            alpha = lr
            self.X = X_batches[i%n_batches]
            self.y = y_batches[i%n_batches]

            self.update(alpha)

        self.X = X
        self.y = y

    def predict(self, X):
        #print(self.params[0][0])
        computed = X
        for i in range(len(self.params)-1):
            computed = np.dot(computed, self.params[i][0]) + self.params[i][1].T
            computed = self.activations[i](computed)

        computed = np.dot(computed, self.params[-1][0]) + self.params[-1][1].T
        computed = self.softmax(computed)

        return np.argmax(computed, axis=1)



        
    
