from numpy.random import randn
import numpy as np

class Softmax:
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
        self.X_train = X_train
        self.y_train = y_train
        self.nclasses = np.unique(y_train).shape[0]
        self.nparams = X_train.shape[0]
        self.theta = self.init_theta()

    def train(self, X_train, y_train):
        pass

    def score(self, k):
        """
        Compute the softmax score for class K
        """
        thetak = self.theta[:,k]
        pass

    def init_theta(self):
        """
        Initialize the parameters for the model
        """
        theta = randn(self.nparams, self.nclasses) * 0.1
        return theta
