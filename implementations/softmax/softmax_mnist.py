# Test of the SoftmaxRegression class
# to learn the separations between flowers 
# inside the iris dataset

from softmax import SoftmaxRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder, Normalizer
import numpy as np

mnist = fetch_mldata("mnist original")
ohe = OneHotEncoder(sparse=False)
normalize = Normalizer()
X = normalize.fit_transform(mnist["data"])
# One-Hot encoding the categorical variables
y = ohe.fit_transform(mnist["target"].reshape(-1, 1))

X_train, X_test = X[:60000,:], X[60000:,:]
y_train, y_test = y[:60000,:], y[60000:,:]

# Matrix transpose to have the data as required
# by the model definition. Namely, rows are features
# and columns are examples for the 'X' matrix and
# rows are classes are columns examples for the 'y', or 
# 'labels' matrix
X_train = X_train.T
y_train = y_train.T
X_test = X_test.T
y_test = y_test.T

# Mini-Batch Gradient Descent
model = SoftmaxRegression(X_train, y_train)
model.train(batch_size=1000, epochs=1000, verbose=True, save_cost_hist=True)
minibatchdg = model.cost_hist

fig = plt.figure(figsize=(10,6), dpi=150)
plt.plot(minibatchdg, linewidth=0.5)
plt.savefig("training_mnist.pdf")
yhat = model.predict(X_train)
ytrue = np.arange(10).reshape(1, -1) @ y_train
sns.heatmap(confusion_matrix(ytrue.ravel(), yhat.ravel()), annot=True)
plt.savefig("confusion_mnist.pdf")
plt.show()
