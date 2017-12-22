import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle, seed

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
        X = self.X_train if X is None else X
        prediction = np.argmax(self.theta.T @ X, axis=0)
        return prediction

    def softmax_mat(self, X, theta=None):
        """
        Method to compute the probability for each
        training example to belong in any of the nclasses

        Parameters
        ----------
        X:  Matrix of size nfeatures X nexamples
        theta: Matrix of size nfeaturesXnclasses with the parameters to consider
            If no matrix is provided, we consider self.theta

        Returns
        -------
        Transposed stochastic Matrix of size nclassesXnexamples
        with the probability of a training example 'x' belonging to class 'k'
        """
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

    def compute_grads(self, X, y):
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
        sigsoftk = self.softmax_mat(X)
        m = X.shape[1]
        for k in range(self.nclasses):
            # To compute the gradient we sum over every training
            # example, i.e., sum(axis=1)
            grad = ((sigsoftk - y)[k,:] * X).sum(axis=1) / m
            grads[:,k] = grad
        return grads

    def batch_train(self, alpha=0.1, iterations=5000, verbose=False, graph_cost=False):
        """
        Train the model using batch gradient descent
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

    def shuffle_dataset(self, seed=None):
        """
        Copy and shuffle the training dataset (self.X_train, self.y_train)
        along the columns (training examples)
        """
        stacked = np.r_[self.X_train.copy(),
                        self.y_train.copy()]
        shuffle(stacked.T)
        X_shuffled = stacked[:self.nfeatures,:]
        y_shuffled = stacked[self.nfeatures:,:]
        return X_shuffled, y_shuffled

    def train(self, batch_size=None, alpha=0.1, epochs=100, verbose=False, save_cost_hist=False):
        """
        Train the model using mini-batch gradient descent. If no batch is
        performed, the model defaults to batch gradient descent

        Parameters
        ----------
        batches: integer
            Number of elements to consider per iteration in training
            the model. if batches=1, mini-batch GD becomes SGD. On
            the other hand, batches=X_train.shape[1] is equivalent to
            batch GD. If no number of batches is provided, the train
            becomes batch gradient descent
        alpha: double
            Learning rate factor
        epochs: integer
            Number of times to sweep the data after training over the whole
            dataset
        """
        cost_hist = []
        nexamples = X_train.shape[1]
        batch_size = nexamples if batch_size is None else batch_size
        number_batches = int(np.floor(nexamples / batch_size))
        for epoch in range(epochs):
            X_batch, y_batch = self.shuffle_dataset()
            for batch in range(number_batches):
                fact = 0 if number_batches - 1 != batch else nexamples % batch_size
                columns = range(batch * batch_size,
                                (batch + 1) * batch_size + fact)
                X_batch_train = X_batch[:, columns]
                y_batch_train = y_batch[:, columns]
                grads = self.compute_grads(X_batch_train, y_batch_train)
                self.theta = self.theta - alpha * grads

                if save_cost_hist:
                    cost = self.cost()
                    cost_hist.append(cost)

            if verbose and epoch % 10 == 0:
                cost = self.cost()
                print(f"Total cost after epoch {epoch + 1}: {cost}")

        if save_cost_hist:
            self.cost_hist = cost_hist
        
if __name__ == "__main__":
    from pydataset import data
    from pandas import get_dummies
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
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
    model.train(batch_size=1, epochs=1000, verbose=True, save_cost_hist=True)
    stochasticdg = model.cost_hist

    model = SoftmaxRegression(X_train, y_train)
    model.train(batch_size=20, epochs=1000, verbose=True, save_cost_hist=True)
    minibatchdg = model.cost_hist
    
    model = SoftmaxRegression(X_train, y_train)
    model.train(epochs=5000, verbose=True, save_cost_hist=True)
    batchdg = model.cost_hist

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
    plt.savefig("/Users/gerardoduran/Desktop/training.pdf")
    plt.show()
    #yhat = model.predict(X_train)
    #ytrue = np.arange(3).reshape(1,-1) @ y_train
    #sns.heatmap(confusion_matrix(ytrue.ravel(), yhat.ravel()), annot=True)
    #plt.show()
