import numpy as np
import matplotlib.pyplot as plt

class SoftmaxRegression:
    """
    Implementation of the Softmax Regression model
    for multiple class classification.
    """
    def __init__(self, X_train, y_train):
        """
        Parameters
        ----------
        X_train: matrix of nfeatures X nexamples
        y_train: matrix of nclasses x nexamples
        """
        self.X_train = X_train
        self.y_train = y_train
        self.nfeatures = X_train.shape[0]
        self.nclasses = y_train.shape[0]
        self.theta = np.zeros((self.nfeatures,
                               self.nclasses))

    def predict(self, X=None):
        pass

    def softmax_mat(self, X=None, theta=None):
        """
        Method to compute the probability for each
        training example to belong in any of the nclasses

        Parameters
        ----------
        X:  Matrix of size nfeatures X nexamples. If nothing is provided
            it defaults to use self.X_train
        theta: Matrix of size nfeaturesXnclasses with the parameters to consider
            If no matrix is provided, we consider self.theta

        Returns
        -------
        Transposed stochastic Matrix of size nclassesXnexamples
        with the probability of a training example 'x' belonging to class 'k'
        """
        X = self.X_train if X is None else X
        theta = self.theta if theta is None else theta
        # Numerator for each class
        softmax_vals = np.exp(theta.T @ X)
        # Normalization factor
        onesvect = np.ones((1, self.nclasses))
        softmax_norm = onesvect @ softmax_vals

        return softmax_vals / softmax_norm

    def cost(self, X=None, y=None, theta=None):
        """
        Compute the cost function using the cross entropy
        cost function

        Parameters
        ----------
        X:  Matrix of size nfeatures X nexamples. If nothing is provided
            it defaults to self.X_train
        y:  Matrix of size nfeatures y nexamples. If nothing is provided
            it defaults to self.y_train
        theta: Matrix of size nfeaturesXnclasses
            If no matrix is provided, we consider self.theta
        """
        X = self.X_train if X is None else X
        y = self.y_train if y is None else y
        m = X.shape[1]

        prob_mat = self.softmax_mat(X, theta)
        cost = -(y * np.log(prob_mat)).sum() / m
        return cost

    def compute_grads(self):
        """
        Compute the gradient of the cost function w.r.t. the
        feature matrix self.theta
        Parameters
        ----------
        X:  Matrix of size nfeatures X nexamples. If nothing is provided
            it defaults to use self.X_train
        y:  Matrix of size nfeatures y nexamples. If nothing is provided
            it defaults to use self.y_train
        """
        grads = np.zeros_like(self.theta)
        sigsoftk = self.softmax_mat()
        m = self.X_train.shape[1]
        for k in range(self.nclasses):
            # To compute the gradient we sum over every training
            # example, i.e., sum(axis=1)
            grad = ((sigsoftk - self.y_train)[k,:] * self.X_train).sum(axis=1) / m
            grads[:,k] = grad
        return grads

    def train(self, alpha=0.1, iterations=5000, verbose=False, graph_cost=False):
        """
        Train the model by means of stochastic gradient descent
        """
        cost_hist = []
        for it in range(iterations):
            grads = self.compute_grads()
            self.theta = self.theta - alpha * grads

            cost = self.cost()
            cost_hist.append(cost)
            if verbose and it % 100 == 0:
                print(f"At iteration {it}, cost: {cost}")

        if graph_cost:
            plt.plot(cost_hist, linewidth=0.5, color="tab:red")
            plt.show()
        
if __name__ == "__main__":
    from pydataset import data
    from pandas import get_dummies
    from sklearn.model_selection import train_test_split
    
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
    
    model = SoftmaxRegression(X_train, y_train)
    model.train(verbose=True, graph_cost=False)
    print(np.argmax(model.theta.T @ X_train, axis=0))
    print(np.arange(3).reshape(1,-1) @ y_train)
