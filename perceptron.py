import numpy as np
import progressbar
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
        self.stopping_condition = False

    def __initialize_weights(self, dim):
        # function to initialize the weights with random data
        # returns dim size float or ndarray of floats
        weights = np.random.normal(loc = 1/(dim+1), scale = 1/np.sqrt(dim+1), size=dim+1)
        return weights


    def predict(self, x):
        # function that makes a prediction of the label with current weights
        if len(x[0]) == self.dim:
            x = self.__adjust_input_x(x)
        elif len(x[0]) != self.dim+1:
            print('Error: dimensions of input ({}) do not match to dimensions of Perceptron ({})'.format(len(x[0]), self.dim))
        y = np.zeros(len(x), dtype='float32')
        for i in range(len(x)):
            y[i] = self.__softmax(np.sum(self.weights*x[i]))
        return y

    def fit(self, x, t, learning = None, epochs = 1000):
        if len(x[0]) != self.dim:
            print('Error: dimensions of input ({}) do not match to dimensions of Perceptron ({})'.format(len(x[0]), self.dim))
            return
        x = self.__adjust_input_x(x)
        self.learning = learning
        learning_function = self.__get_learning_method(self.learning)
        E = []
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while not self.stopping_condition:
            y = self.predict(x)
            E.append(self.__cost(x, y, t))
            delta_weights = learning_function(x, y, t)
            self.weights += delta_weights
            bar.update(len(E))
            if len(E) == epochs:
                self.stopping_condition = True
        return E

    def __adjust_input_x(self, x):
        return np.insert(x,0,1, axis=1)

    def __get_learning_method(self, learning):
        if learning == 'gradient':
            learning_function = self.__learning_gradient
        elif learning == 'momentum':
            learning_function = self.__learning_momentum
        elif learning == 'decay':
            learning_function = self.__learning_decay
            self.lam = 0.1
        elif learning == 'netwon':
            learning_function = self.__learning_newton
        elif learning == 'line':
            learning_function = self.__learning_line
        elif learning == 'conjugate':
            learning_function = self.__learning_conjugate
        else:
            print('Error: No learning method given to fit function.')
        return learning_function

    def __cost(self, x, y, t):
        # Calculates cost, given input x, prediction y and target t
        # Returns cost
        N = len(x)
        if self.learning == 'decay':
            decay = self.lam * np.sum(weights**2) / (2*(self.dim+1))
        else:
            decay = 0
        return -1/N * np.sum(t * np.log10(y) + (1-t) * np.log10(1 - y)) + decay


    def __softmax(self, x):
        # Calculates softmax of input x
        return (1 + np.exp(-x))**(-1)

    def __learning_gradient(self, x, y, t, dW = None):
        learning_rate = 1
        delta_w = - learning_rate * self.__gradient(x, y, t)
        if np.all(delta_w == 0):
            print('Stop')
            self.stopping_condition = True
        return delta_w

    def __learning_momentum(self, x, t):
        # momentum learning method for updating weights
        learning_rate = 1
        #choose momentum factor alpha - assignment says you have to try values
        alpha = 0.1
        #calculates gradient and add the momentum
        delta_w = - learning_rate * self._gradient(x, y, t) + alpha * self._gradient(x, y, t)
        # return delta_weights
        return delta_w

    def __learning_decay(self):
        # now add weight decay term and use the momentum learning method
        labda = 0.1
        learning_rate = 1
        # choose momentum factor alpha - assignment says you have to try values
        alpha = 0.1
        # calculates gradient and add the momentum
        delta_w = - learning_rate * self._gradient(x, y, t) + alpha * self._gradient(x, y, t)
        # return delta_weights
        return delta_w

    def __learning_newton(self):
        pass
        # return delta_weights

    def __learning_line(self):
        pass
        # return delta_weights

    def __learning_conjugate(self):
        pass
        # return delta_weights

    def __gradient(self, x, y, t):
        g = np.zeros((self.dim+1))
        for i in range(self.dim+1):
            g[i] = self.__gradient_i(i, x, y, t)
        return g

    def __gradient_i(self, i, x, y, t):
        # inputs dim i of x, predict y, true label t
        N = len(x) # Number of samples
        if self.learning == 'decay':
            decay = self.lam * self.weights[i] / (self.dim + 1)
        else:
            decay = 0
        return np.sum((y-t)*x[:,i])/N + decay

    def __hessian(self, x, y):
        h = np.zeros((self.dim+1, self.dim+1))
        for i in range(self.dim+1):
            for j in range(self.dim+1):
                h[i,j] = self.__hessian_ij(i, j, x, y)
        return h


    def __hessian_ij(self, i, j, x, y):
        N = len(x) # Number of samples
        # Test commit shared branch
        if self.learning == 'decay' and i == j:
            decay = self.lam / (self.dim + 1)
        else:
            decay = 0
        return np.sum(x[:,i] * y (1 - y) * x[:,j])/N + decay
