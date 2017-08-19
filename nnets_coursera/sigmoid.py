import numpy as np
from numpy.random import normal
from numpy import ones, exp

class Sigmoid(object):
    def __init__(self, inputs, outputs, hidden_layers,  weights=None):
        """
        Attributes
        ----------
        inputs: nxm matrix
                A matrix of n observations and m input units
        outputs:nxq matrix
                A matrix of n observations and q output units 
        hidden_layers:  tuple
                        A tuple of length 'p' where each entry denotes the elments
                        in each hidden layer
        """
        self.ishape = inputs.shape
        self.oshape = outputs.shape
        self.inputs = np.c_[inputs, ones(self.ishape[0])]
        self.sigmoid = lambda z: 1 /(1 + exp(-z))
        self.architecture = (self.ishape[1], *hidden_layers, self.oshape[1])

    def init_inputs(self, inputs):
        """
        Initialize the input units by asigning a
        bias parameter of 1

        Parameters
        ----------
        inputs
        """
        return
