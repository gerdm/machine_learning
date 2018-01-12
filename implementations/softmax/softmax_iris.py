# Test of the SoftmaxRegression class
# to learn the separations between flowers 
# inside the iris dataset

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from softmax import SoftmaxRegression
from pydataset import data
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# One-Hot encoding the categorical values
iris = get_dummies(data("iris"))
X, y = iris.iloc[:,:4].values, iris.iloc[:,4:].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
# Matrix transpose to have the data as requested
# by the model definition. Namely, rows are features
# and columns are examples for the 'X' matrix and
# rows are classes are columns examples for the 'y', or 
# 'labels' matrix
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T

# Stochastic Grdient Descent
model = SoftmaxRegression(X_train, y_train)
model.train(batch_size=1, epochs=1000, alpha=0.03, verbose=True, save_cost_hist=True)
stochasticdg = model.cost_hist

# Mini-Batch Gradient Descent
model = SoftmaxRegression(X_train, y_train)
model.train(batch_size=20, epochs=1000, verbose=True, save_cost_hist=True)
minibatchdg = model.cost_hist

# Batch Gradient Descent
model = SoftmaxRegression(X_train, y_train)
model.train(epochs=5000, verbose=True, save_cost_hist=True)
batchdg = model.cost_hist

# Making Figure
fig = plt.figure(figsize=(10,6), dpi=150)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax1.plot(stochasticdg, linewidth=0.5)
ax2.plot(minibatchdg, linewidth=0.5)
ax3.plot(batchdg, linewidth=0.5)
ax1.set_title("SGD")
ax2.set_title("Mini-Batch GD")
ax3.set_title("Batch GD")
plt.savefig("iris_training.pdf")
plt.show()

yhat = model.predict(X_test)
ytrue = np.arange(3).reshape(1,-1) @ y_test
sns.heatmap(confusion_matrix(ytrue.ravel(), yhat.ravel()), annot=True)
plt.title("Iris Confusion Matrix\non Test Set")
plt.savefig("iris_confusion_matrix.pdf")
plt.show()
