from numpy.random import randn
import numpy as np
from numpy import exp
from sklearn.preprocessing import OneHotEncoder

class SoftmaxRegression:
    """
    Implementation of the softmax learning algorithm
    using batch gradient descent
    """
    def __init__(self, X_train, y_train):
        """
        Initialize the model setting the training dataset

        Parameters
        ----------
        X_train: Mat(R, mxn)
            Matrix of real numbers with 'm' training examples and
            'n' parameters
        y_train: Mat(R, mx1)
            Vector of integer numbers with 'm' training examples. Each integer
            represents a class
        """
        self.X_train = X_train.T 
        self.y_train = y_train
        self.y_train_ohe = self.make_y_train_ohe()
        self.nclasses = np.unique(y_train).shape[0]
        self.nparams = X_train.shape[1]
        self.theta = self.init_theta()

    def make_y_train_ohe(self):
        ohe = OneHotEncoder(sparse=False)
        return ohe.fit_transform(self.y_train).T


    def minibatch_train(self, X_train, y_train):
        for _ in range(50):
            grads = np.mean(self.scores() - self.y_train_ohe, axis=0) @ self.X_train.T
            for k in self.nclasses:
                grads =  (self.scores() -  self.y_train_ohe)[k,:] @ self.X_train.T


    def scores(self, X=None):
        """
        Compute the softmax score for a given
        X sample.

        Parameters
        ----------
        X: Mat(R, mxn)
            Matrix of real numbers with 'm' training examples and
            'n' parameters. If no example is provided, X taken is
            the training dataset

        Returns
        -------
        Mat(R, mxn) with scores for each training dataset
        """
        if X is None:
            X = self.X_train
        else:
            X = X.T
        Z = self.theta.T @ X
        score = exp(Z) / exp(Z).sum(0)

        return score

    def predict(self, X=None):
        """
        Predict the model for each training example 
        as the index with the highest score
        """
        return np.argmax(self.scores(X), axis=0)

    def loss(self, X=None):
        """
        Return 
        """
        loss = np.sum(self.y_train_ohe * np.log(self.scores(X)), axis=0)
        loss = np.mean(loss)
        return -loss

    def init_theta(self):
        """
        Initialize the parameters for the model
        """
        theta = randn(self.nparams, self.nclasses)
        return theta

if __name__ == "__main__":
    from pydataset import data
    from sklearn.model_selection import train_test_split
    
    iris = data("iris")
    species = iris.Species.unique()
    map_labels = {species: val for val, species
                  in enumerate(species)}

    y = iris.Species.apply(lambda sp: map_labels[sp]).values.reshape(-1,1)
    X = iris.drop("Species", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = SoftmaxRegression(X_train, y_train)
    print(model.predict().shape)
    print(model.y_train.ravel().shape)
