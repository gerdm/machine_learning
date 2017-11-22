import numpy as np
from numpy.random import seed, randn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def distance(x0, X): return np.sqrt(np.sum((x0 - X) ** 2, axis=0))

class KNN:
    def __init__(self, Xtrain, Ytrain, nneighbors):
        """
        Xtrain: nXm numpy array with n datapoints and m
            examples
        Ytrain: 1xm numpy array with m different classes
        """
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.nneighbors = nneighbors
    
    def knn_ix(self, xpred):
        """
        A simple implementation of the K-nearest-neighbors
        algorithm that returns the indices witht the closest
        k-neighbors

        Params
        ------
        """
        distances = distance(xpred, self.Xtrain)
        kclosest = np.argsort(distances)[:self.nneighbors]

        return kclosest

    def pred(self, xpred):
        
        xpred = xpred.reshape(*xpred.shape)
        closest_ix = self.knn_ix(xpred)
        return np.round(np.mean(self.Ytrain[closest_ix]))
    

    def plot_closest_neigh(self, xpred):
        """
        Plot the actual classes and the prediction of a single
        point based on its closest n neighbors
        """
        miny = np.min(self.Ytrain)
        maxy = np.max(self.Ytrain)
        normalized = Normalize(miny, maxy, clip=True)
        colormap = cm.ScalarMappable(normalized, cmap=cm.coolwarm)

        ypred = self.pred(xpred)
        colors=[colormap.to_rgba(y) for y in self.Ytrain]
        knnix = self.knn_ix(xpred)
        notknnix = [x for x in range(Y.shape[0]) if x not in self.knn_ix(xpred)]

        plt.title(f"KNN with {self.nneighbors} neighbors")
        plt.scatter(*self.Xtrain, c=colors)
        for ix in knnix:
            plt.plot(*np.c_[xpred, self.Xtrain[:,ix]], linewidth=1,
                     c="black", alpha=0.5)
        plt.scatter(*xpred, c=colormap.to_rgba(ypred) , s=90)

if __name__ == "__main__":
    from matplotlib import animation
    seed(314)
    nvals = 30
    X1 = randn(2,nvals) + 0.6
    X2 = randn(2,nvals) - 0.6
    X = np.c_[X1, X2]
    Y = np.r_[np.ones(nvals), np.zeros(nvals)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    def update_neighbors(ix):
        x0 = np.array([[0.5, -0.5]]).T
        knmodel = KNN(X, Y, ix)
        knmodel.plot_closest_neigh(x0)
        return

    Xk = np.r_[np.linspace(-1,1, 10), np.linspace(1,-1,10)] 
    Yk = np.arange(1,-11, -0.1)
    def update_position(ix):
        x0 = np.array([[Xk[ix], Yk[ix]]]).T
        knmodel = KNN(X, Y, 30)
        knmodel.plot_closest_neigh(x0)
        return


    animate = animation.FuncAnimation(fig, update_neighbors,
            frames=30,interval=10, blit=False)

    animate.save('knn_neigh.gif', fps=10, writer='imagemagick')
    #plt.show()
