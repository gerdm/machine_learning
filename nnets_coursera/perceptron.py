import numpy as np

class Perceptron(object):
    """
    Python class for the "Perceptron learning algorithm" based on
    the course on Neural Networks for Machine Learning on Coursera
    """
    def __init__(self, neg_examples, pos_examples, w_init=None, w_gen_feas=None): 
        """
        neg_examples: A Mat(neg_examples x 2) for examples with target 0
        pos_examples: A Mat(pos_examples x 2) for examples with target 1
        w_init: an R3 vector with initial weights --the third element is the bias--
        w_feas: A 'generously' feasible weight vector
        num_err_history: An array of total number of errors per iteration
        w_dist_history: The distance from the current weight vector to the 'generously'
                        feasible weight vector (if it exists)

        **NOTE** The examples given do not contain a bias, they are given
                 at the initialization of an instance of a class
        """
        self.w_gen_feas = w_gen_feas
        self.w_init = self.init_weights(w_init) 
        self.num_neg_examples = len(neg_examples)
        self.num_pos_examples = len(pos_examples)
        self.neg_examples = np.c_[neg_examples, np.ones(self.num_neg_examples)]
        self.pos_examples = np.c_[pos_examples, np.ones(self.num_pos_examples)]
        self.num_err_history = np.array([])
        self.w_dist_history = np.array([])
        self.learned_weights = None
    
    def init_weights(self, weights):
        """
        Initialize the weights appropriately. If no
        initial weights are given, create 3D vector
        initializing the weights sampled  from a normal distribution.
        """
        if weights is None:
            return np.random.randn(3, 1)
        else:
            return weights

    def learn_perceptron(self, err_threshold=0, iter_threshold=100):
        iter = 0
        current_weights = self.w_init
  
        while True:
            iter += 1
            print("At iteration {iter}\r".format(iter=iter))
            mistakes0, mistakes1 = self.eval_perceptron(current_weights)
            total_errors = np.size(mistakes0) + np.size(mistakes1)
            self.num_err_history = np.append(self.num_err_history, total_errors)
            
            # Update the distance from the current weight
            # to the feasible weight vector
            if self.w_gen_feas is not None:
                error_dist = np.abs(weight - self.w_gen_feas)
                sellf.w_dist_history = np.append(self.w_dist_history)

            current_weights = self.update_weights(current_weights)

            if (iter > iter_threshold) or (total_errors <= err_threshold):
                self.learned_weights = current_weights
                return True

    def update_weights(self, weights):
        """
        Updates the weights of the perceptron for incorrectly classified points
        using the preceptron update algorithm. This method takes one sweep 
        over the dataset

        Parameters
        ----------
        weights: 3x1 vector
            A 3-dimensional weight vector, wehre the last element is the bias

        Returns
        -------
        3x1 vector 
        """
        wprime = weights.reshape((1, 3))
        neg, pos = self.eval_perceptron(weights)

        # If it is a unlearned positive weight,
        # we must increment the weight by the input vector
        for row in self.pos_examples[pos, :]:
            wprime += row 

        # If it is a unlearned negative weight,
        # we must decrement the weight by the input vector
        for row in self.neg_examples[neg, :]:
            wprime -= row

        return wprime.reshape((3,1))

    def eval_perceptron(self, weights):
        """
        Evaluates the perceptron using a given weight vector. Here, evaluation 
        refers to finding the data points that the perceptron incorrectly classifies
        
        Parameters
        ----------
        weights: 3x1 vector
            A 3-dimensional weight vector, where the last element is the bias

        Returns
        ------
        3x1 vector mistakes0
            A vector containing the indices of the negative examples that have been
            incorrectly classified as positive
        3x1 vector mistakes1
            A vector containing the indices of the positive examples that have been
            incorrectly classified as negative
        """
        predict0 = self.neg_examples @ weights
        predict1 = self.pos_examples @ weights

        mistakes0 = np.flatnonzero(predict0 >= 0)
        mistakes1 = np.flatnonzero(predict1 < 0)
        
        return mistakes0, mistakes1
