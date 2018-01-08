# Implementation of the softmax regression using
# TensorFlow. This becomes a 0-hiden-layer neural 
# network with C (number of classes) outputs.
# Based on the talk by Martin Gorner: 
# TensorFlow and deep learning without a PhD

import tensorflow as tf
from sklearn.datasets import fetch_mldata
import numpy as np
from numpy.random import randint, permutation

class MNIST:
    def __init__(self):
        self.data = fetch_mldata("mnist original")
        self.train_size = 60000
        self.X_train = self.data["data"][:self.train_size]
        self.y_train = self.data["target"][:self.train_size]
        self.X_test = self.data["data"][self.train_size:]
        self.y_test = self.data["target"][self.train_size:]
        self.untrained_indices = permutation(range(self.train_size))
        self.current_index = 0

    def one_hot_class(self, y):
        """
        One hot encoded targets along columns
        (axis=1)

        Parameters
        ----------
        y: np.array of size (nexamples,)
            Targets to one-hot encode into individual buckets

        Returns
        -------
        np.array with size nexamples X nclasses
        """
        y_hot_var = tf.one_hot(y, depth=10, axis=1)
        with tf.Session() as sess:
            y_hot = sess.run(y_hot_var)
        return y_hot

    def test_data(self):
        """
        Return a tupple with X_test and one-hot encoded
        y_test
        """
        one_hot_test = self.one_hot_class(self.y_test)
        return self.X_test, one_hot_test

    def random_batch(self, batch_size=100):
        """
        Return a random selection of batch_size
        elements.

        Parameters
        ----------
        batch_size: integer
            Size of the random batch to train

        Returns
        -------
        tuple with 'batch_size' examples to train from,
        together withthe equivalent one-hot encoded labels
        """
        batch_indices = self.untrained_indices[batch_size * self.current_index:
                                               batch_size * (self.current_index + 1)]
        self.current_index += 1
        X_batch = self.X_train[batch_indices, :]
        y_batch = self.y_train[batch_indices]

        # Once iterating over all indices, start a new epoch:
        # permute once again
        if batch_size * (self.current_index + 1) >= self.train_size:
            self.untrained_indices = permutation(range(self.train_size))
            self.current_index = 0
            print("Starting new epoch")

        return X_batch, self.one_hot_class(y_batch)

if __name__ == "__main__":
    mnist = MNIST()

    # Initializing the Softmax Regression model

    # Input variables, None is where the minibatch will
    # be place to train the model
    X = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    init = tf.global_variables_initializer()

    # Predicted output. The input to the softmax is a matrix
    # of dimensions 'nexamples X nclasses'. The output is the
    # predicted probability for each class (rowise)
    y_hat = tf.nn.softmax(X @ W + b)
    # Actual output
    y = tf.placeholder(tf.float32, [None, 10])

    # Loss function
    cross_entropy = -tf.reduce_sum(y * tf.log(y_hat + 1e-10), name="cross_entropy")
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)

    # Tracking the ratio of errors in each batch
    # We predict the class considering the index with
    # highest probability. The array of correct predictions
    # is 'casted' to integers
    correct_pred = tf.cast(tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_pred)

    # Selecting the optimization method
    gdesc = tf.train.GradientDescentOptimizer(0.003)
    train_step = gdesc.minimize(cross_entropy)

    nepochs = 10_000
    batch_size = 100
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        for epoch in range(nepochs):
            X_batch, y_batch = mnist.random_batch(batch_size)
            train_batch = {X: X_batch, y: y_batch}

            # We perform a single pass of feedforward, bakprop and
            # weight update
            sess.run(train_step, feed_dict=train_batch)

            # Compute the accuracy, and cross entropy after
            # training the minibatch
            acc, loss = sess.run([accuracy, cross_entropy],
                                 feed_dict=train_batch)

            if epoch % 100 == 0:
                print(f"Accuracy at epoch {epoch}: {acc:0.4f}")

        test_data = mnist.test_data()
        test_dict = {X: test_data[0], y: test_data[1]}
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict=test_dict)
        print(f"\nFinal accuracy on test: {acc:0.4f}\nFinal cross-entropy on test: {loss:0.4f}")
