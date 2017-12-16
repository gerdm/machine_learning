import numpy as np

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
        
if __name__ == "__main__":
    from pydataset import data
    from pandas import get_dummies
    from sklearn.model_selection import train_test_split
    
    iris = get_dummies(data("iris"))
    X, y = iris.iloc[:,:4].values, iris.iloc[:,4:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)
    
    # Matrix transpose to have the data as requested
    # by the model definition. Namely, rows are features
    # and columns are examples for the 'X' matrix and
    # rows are classes are columns examples for the 'y', or 
    # 'labels' matrix
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    
    model = So