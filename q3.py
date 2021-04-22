import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from logisticRegression.kClassLogisticRegression import kClassLogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from metrics import *

np.random.seed(42)

#data = load_breast_cancer()
data = load_digits()
X = pd.DataFrame(data.data, columns=data.feature_names)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=4)

y = pd.Series(data.target)

print("\n----------Stratified 4 Folds Accuracy----------")
ind = 0
fold_acc = []
best_acc_y_hat = -1
best_acc_y = -1
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    LR = kClassLogisticRegression()
    LR.fit_vectorised(X_train, y_train.reset_index(drop=True), X.shape[0], n_iter = 100)

    y_hat = LR.predict(X_test)
    curr_acc = accuracy(y_hat, y_test)
    if len(fold_acc)!=0 and curr_acc > np.max(fold_acc):
        best_acc_y_hat = y_hat
        best_acc_y = y_test
    fold_acc.append(curr_acc)
    print("Test fold {}: ".format(ind+1), curr_acc)
    ind+=1

print("Average accuracy: ", np.mean(fold_acc))

print("\n----------Confusion Matrix----------")
print(confusion_matrix(best_acc_y, best_acc_y_hat))

print("\n----------PCA on Digits Dataset----------")
figure = plt.figure(figsize=(10, 7))
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
cm = plt.cm.tab10
plt.scatter(X_transformed[:,0], X_transformed[:,1], c=data.target, cmap=cm)
plt.colorbar()
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")
plt.title("Digits Data in 2 dimensions")
#plt.show()
figure.savefig('plots/q3_d.png')
