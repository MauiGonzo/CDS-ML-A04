import numpy as np
# assignment 4: make a perceptron
# by Remco Volmer and  Maurice Verbrugge
# comments refer to slides (41 etc) from the course
# class defenition

# define perceptron class
# dim: nr of dimsensions
# weigths: adaptable weights (as mentioned in slide 42)
# learning rate: we take one
class Perceptron():
    def __init__(self, dim):
        # Initialize perceptron, set dim, initialize weights
        self.dim = dim
        self.weights = self.__initialize_weights(self.dim)

    def __initialize_weights(self, dim):
        # function to initialize the weights with random data
        # returns dim size float or ndarray of floats
        weights = np.random.random(dim+1)
        return weights

    def predict(self, x):
        # function that makes a prediction of the label with current weights
        x = self.__adjust_input_x(x)
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = self.__softmax(np.sum(w*x[i]))
        return y

    def fit(self, x, t, learning = None):
        if len(x[0]) != self.dim:
            print('Error: dimensions of input ({}) do not match to dimensions of Perceptron ({})'.format(len(x[0]), self.dim))
            return
        x = self.__adjust_input_x(x)

        if learning == 'gradient':
            learning_function = self.__learning_gradient
        elif learning == 'momentum':
            learning_function = self.__learning_momentum
        elif learning == 'decay':
            learning_function = self.__learning_decay
        elif learning == 'netwon':
            learning_function = self.__learning_newton
        elif learning == 'line':
            learning_function = self.__learning_line
        elif learning == 'conjugate':
            learning_function = self.__learning_conjugate
        else:
            learning_function = self.__learning_gradient
        E = []
        while self.stopping_condition: 
            learning_function()
        return 

    def __adjust_input_x(self, x):
        return np.insert(x,0,1, axis=1)

    def __cost(self, x, t, w):
        # Calculates cost, given input x, weights w and target t
        # Returns cost
        N = len(x)
        y = np.zeros(N)
        for i in range(N):
            y[i] = self.__softmax(np.sum(w*x[i]))
        return -1/N * np.sum(t * np.log10(y) + (1-t) * np.log10(1-y))


    def __softmax(self, x):
        # Calculates softmax of input x
        return (1 + np.exp(-x))**(-1)

    def __learning_gradient(self):
        # Write gradient desent method for updating weights

        pass
        # return delta_weights

    def __learning_momentum(self):
        # Write momentum for updating weights
        pass
        # return delta_weights

    def __learning_decay(self):
        pass
        # return delta_weights

    def __learning_newton(self):
        pass
        # return delta_weights

    def __learning_line(self):
        pass
        # return delta_weights

    def __learning_conjugate(self):
        pass
        # return delta_weights