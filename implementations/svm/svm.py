import numpy as np
from numpy.random import choice

class SupportVectorMachine:
    def __init__(self, C, Xtrain, ytrain):
        self.C = C
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.ntrain = Xtrain.shape[1]
        self.alpha = np.zeros((self.ntrain, 1))

    def _select_target_alpha(self, i, j):
        """
        Choose the target index i, j to update i the
        """
        #i, j = choice(range(self.ntrain), size=2, replace=False)
        self.i, self.j = i, j
        self.xi = self.Xtrain[:, [i]]
        self.xj = self.Xtrain[:, [j]]
        self.yi = self.ytrain[i][0]
        self.yj = self.ytrain[j][0]

        # every X, y element but indices i, j
        self.xrem = np.delete(self.Xtrain, [i, j], axis=1)
        self.yrem = np.delete(self.ytrain, [i, j])
        self.alpharem = np.delete(self.alpha, [i, j])
        # The constat term to help compute yi
        self.zeta = - (self.alpharem * self.yrem).sum()

        self.aj = self.alpha[j, 0]
        self.ai = self.yi * (self.zeta - self.aj * self.yj)

    def _compute_coeficients(self):
        """
        Compute the terms A1, A2, A3 that are in terms
        of aj.
        """
        A1 = np.asscalar((self.xi - self.xj).T @ (self.xi - self.xj))

        A2 = 2 * (-(self.alpharem * (self.xi.T @ self.xrem).ravel() * self.yi * self.yrem).sum() + \
             (self.alpharem * (self.xj.T @ self.xrem).ravel() * self.yj * self.yrem).sum() + \
             - self.zeta * self.xi.T @ self.xj * self.yj + self.yi * self.yj - 1); A2 = np.asscalar(A2)
                
        A3 = (self.alpharem * self.xrem * self.yrem).sum(axis=0) @ \
            (self.alpharem * self.xrem * self.yrem).sum(axis=0) - \
             2 * self.alpharem.sum()\
             + 2 * self.zeta * (self.alpharem * (self.xi.T @ self.xrem) * self.yrem).sum() + \
             self.zeta ** 2 * self.xi.T @ self.xi - 2 * self.zeta * self.yi; A3 = np.asscalar(A3)

        return A1, A2, A3

    def _update_aij(self):
        A1, A2, A3 = self._compute_coeficients()

        if self.yi != self.yj:
            L = np.maximum(0, self.aj - self.ai)
            H = np.minimum(self.C, self.C + self.aj - self.ai)
        else:
            L = np.maximum(0, self.aj + self.ai - self.C)
            H = np.minimum(self.C, self.aj + self.ai)

        aj_unclipped = - A2 / (2 * A1)
        self.aj = np.clip(aj_unclipped, L, H)
        self.ai = (self.zeta - self.aj * self.yj) * self.yi

        self.alpha[self.i, 0] = self.ai
        self.alpha[self.j, 0] = self.aj

    def _compute_cost(self):
        A1, A2, A3 = self._compute_coeficients()
        return -(self.aj ** 2 * A1 + self.aj * A2 + A3) / 2

