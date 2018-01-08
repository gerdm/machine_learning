# Implementation of the softmax regression using
# TensorFlow. This becomes a 0-hiden-layer neural 
# network with C (number of classes) outputs.

import tensorflow as tf
from sklearn.datasets import fetch_mldata
import numpy as np
from numpy.random import randint

class MNIST:
    def __init__(self):
        self.data = fetch_mldata("mnist")
        self.train_size = 60000
        self.X_train = self.data["data"][:60_000]
        self.y_train = self.data["target"][:60_000]
        self.X_test = self.data["data"][60_000:]
        self.y_test = self.data["target"][60_000:]

    def random_batch(self, batch_size):
        """
        Return a random selection of batch_size
        elements
        """
        batch_indices = randint(0, self.train_size, batch_size)
        X_batch = self.X_train[batch_indices, :]
        y_batch = self.y_train[batch_indices]
        return X_batch, y_batch

if __name__ == "__main__":
    mnist = MNIST()

    # Initializing the Softmax Regression model:
    # y_hat = softmax(XW + b) in R(nexamples X nclasses)
    # where softmax is applied rowise

    # Input variables, None is where the minibatch will
    # be place to train the model
    X = tf.placeholder([None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    init = tf.initialize_all_variables()

    # Predicted output
    y_hat = tf.nn.softmax(X @ W + b)
    # Actual output
    y = tf.placeholder(tf.float32, [None, 10])

    # Loss function
    cross_entropy = -tf.reduce_sum(y * tf.log(y_hat), name="cross_entropy")

