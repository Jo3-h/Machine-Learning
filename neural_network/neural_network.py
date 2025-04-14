import numpy as np

'''
----- > Auxiliary functions < -----
'''
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

'''
----- > Network class < -----

Architecture Hyperparameters:
- input_nodes: number of nodes in the input layers
- output_nodes: number of nodes in the output layer
- hidden_layers: number of hidden layers
- hidden_neurons: number of neurons in each hidden layer
- activation_function: function applied to activation at node to compress into range [0,1]

Training Hyperparameters:
- learning_rate: the size of the steps taken during gradient decent 
- batch_size: number of samples per gradient update
- epochs: how many full passes over the training data
- optimizer: optimization algorithm
- loss_function: metric by which the model is measured 
'''
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a