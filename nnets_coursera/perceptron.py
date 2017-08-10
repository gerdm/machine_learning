import numpy as np

class Perceptron(object):
    """
    Perceptron class for an R2 dataset.
    """
    def __init__(self, neg_examples, pos_examples, w_init=None, w_gen_feas=None): 
        """
        neg_examples: A Mat(neg_examples x 2) for examples with target 0
        pos_examples: A Mat(pos_examples x 2) for examples with target 1
        w_init: an R3 vector with initial weights --the third element is the bias--
        w_feas: A 'generously' feasible weight vector

        **NOTE** The examples given do not contain a bias, they are given
                 at the initialization of an instance of a class
        """
        self.w_gen_feas = w_gen_feas
        self.w_init = self.init_weights(w_init) 
        self.num_neg_examples = len(neg_examples)
        self.num_pos_examples = len(pos_examples)
        self.neg_examples = np.c[neg_example, np.ones(self.num_neg_examples)]
        self.pos_examples = np.c[pos_example, np.ones(self.num_pos_examples)]
    
    def init_weights(self, weights):
        """
        Initialize the weights appropriately. If no
        initial weights are given, create an R3 vector
        initializing the weights sampling from a normal distribution.
        """
        if weights is None:
            return np.random.randn(3, 1)
        else:
            return weights

    def learn_perceptron(self):
        # Find the data points that the perceptron has incorrectly classified
        # and record the number of error it makes
        num_err_history = np.array([])
        w_dist_history = np.array([])
        iter = 0
        mistakes0, mistakes1 = self.eval_perceptron(self.w_init)

    def update_weights(self):
        pass

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
        pass

