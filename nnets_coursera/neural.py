import numpy as np
from numpy.random import randn
from numpy import ones, exp

sigmoid = lambda z: 1 / (1+ exp(-z))

class NNet:
    def __init__(self, layers):
        self.layers_len = len(layers)
        self.layers = layers
        self.biases = self.init_biases()
        self.weights = self.init_weights()

    def init_weights(self):
        layers_map = zip(self.layers[1:], self.layers[:-1])
        return [randn(t0, t1) for t0, t1 in layers_map]

    def init_biases(self):
        return [randn(b, 1) for b in self.layers[1:]]

    def forwardpropagate(self, inputs):
        """
        Forward propagate the neural net where 'inputs'
        represent the values given in the input layer
        """
        for w, b in zip(self.weights, self.biases):
            inputs = sigmoid(w @ inputs + b)
        
        return inputs
