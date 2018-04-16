import tensorflow as tf
import numpy as np
from numpy.random import seed, shuffle
from math import ceil

class LinearRegression:
	"""
	A linear regression implementation using the TensorFlow framework
	"""
	def __init__(self, learning_rate, include_bias=True):
		self.learning_rate = learning_rate
		self.include_bias = include_bias

	def train(self, Xtrain, ytrain, n_epochs=50, batch_size=None):
		"""
		Train the Linear Regression model via
		mini-batch gradient descent

		Parameters
		----------
		Xtrain: numpy array with 'n' examples and 'm' features
			i.e., ndarray of shape (n X m)
		ytrain: numpy array with 'n' examples, i.e., ndarray of shape
			(n X 1)
		batch_size: the number of elements to pass into the training batch
		"""
		self.n, self.m = Xtrain.shape
		if self.include_bias:
			Xtrain = np.c_[np.ones((n, 1)), Xtrain]
			# Plus one feature: the bias
			self.m += 1
		self.n_batches = ceil(n / batch_size)
		training_op = self._initialize_graph()

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(n_epochs):
				for Xbatch, ybatch in self._fetch_batch(epoch):
					sess.run(training_op, feed_dict={self.X:Xbatch, self.y:ybatch})
				if epoch % 100 == 0:
					print(f"@Epoch {epoch};  MSE={self.mse.eval(feed_dict={self.X:Xbatch, self.y:ybatch}): 0.2f}")

	def _initialize_graph(self):
		"""
		Define the TF computation graph
		"""
		self.X = tf.placeholder(tf.float32, shape=[None, self.m], name="X")
		self.y = tf.placeholder(tf.float32, shape=[None, self.m], name="y")
		self.theta = tf.Variable(tf.random_uniform([self.m, 1], minval=-1, maxval=1), name="theta")

		self.y_pred = tf.matmul(self.X, self.theta)
		self.err = self.y_pred - self.y
		self.mse = tf.reduce_mean(tf.square(self.err), name="MSE")
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		return optimizer.minimize(mse)

	def _fetch_batch(self, epoch):
		"""
		Shuffle the dataset and load chunks of the data to pass
		as batches
		"""
		seed(epoch)
		batches = np.c_[self.Xtrain, self.ytrain]
		shuffle(batches)
		batches = np.split(batches, self.n_batches)
		for batch in batches:
			# Return Xbatch, ybatch
			yield batch[:, :-1], batch[:, -1]

if __name__ == "__main__":
	pass
