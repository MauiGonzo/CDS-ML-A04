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
        y = np.zeros(len(x), dtype='float64') #Set dtype to float64 otherwizie y[i] might be rounded up to 1.0 or down to 0.0 then log will break
        for i in range(len(x)):
            y[i] = self.__softmax(np.sum(self.weights*x[i]))
        return y

    def fit(self, x, t, x_val, t_val, learning = None, epochs = 1000):
        if len(x[0]) != self.dim:
            print('Error: dimensions of input ({}) do not match to dimensions of Perceptron ({})'.format(len(x[0]), self.dim))
            return
        x = self.__adjust_input_x(x)
        x_val = self.__adjust_input_x(x_val)
        self.learning = learning
        learning_function = self.__get_learning_method(self.learning)
        E = []
        E_val = []
        # bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while not self.stopping_condition:
            y = self.predict(x)
            y_val = self.predict(x_val)
            E.append(self.__cost(x, y, t))
            print('E: {}, y: min: {}, max {}'.format(E[-1], np.amin(y), np.amax(y)))
            E_val.append(self.__cost(x_val, y_val, t_val))
            delta_weights = learning_function(x, y, t)
            self.weights += delta_weights
            # bar.update(len(E))
            if len(E) == epochs:
                self.stopping_condition = True
        return E, E_val

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
        elif learning == 'newton':
            learning_function = self.__learning_newton
            self.lam = 0.1
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
            decay = self.lam * np.sum(self.weights**2) / (2*(self.dim+1))
        else:
            decay = 0
        return -1/N * np.sum(np.where(t == 1, np.log(y + 1e-50), np.log(1-y+1e-50))) + decay


    def __softmax(self, x):
        # Calculates softmax of input x
        return (1 + np.exp(-x))**(-1)

    def __learning_gradient(self, x, y, t, dW = None):
        learning_rate = 1
        delta_w = - learning_rate * self.__gradient(x, y, t)
        if np.all(delta_w == 0):
            self.stopping_condition = True
        return delta_w

    def __learning_momentum(self, x, y, t, dW = None):
        # momentum learning method for updating weights
        learning_rate = 1
        #choose momentum factor alpha - assignment says you have to try values
        alpha = 0.1
        #calculates gradient and add the momentum
        delta_w = - learning_rate * self.__gradient(x, y, t) + alpha * self.__gradient(x, y, t)
        # return delta_weights
        return delta_w

    def __learning_decay(self, x, y, t, dW = None):
        # now add weight decay term and use the momentum learning method
        labda = 0.1
        learning_rate = 1
        # choose momentum factor alpha - assignment says you have to try values
        alpha = 0.1
        # calculates gradient and add the momentum
        delta_w = - learning_rate * self.__gradient(x, y, t) + alpha * self.__gradient(x, y, t)
        # return delta_weights
        return delta_w

    def __learning_newton(self, x, y, t):
        hessian = self.__hessian(x, y)
        gradient = self.__gradient(x, y, t)
        return -1*np.linalg.inv(hessian).dot(gradient)

    def __learning_line(self,x ,y, t, dW = None):
        #write gradient method with line search
        pass
        # return delta_weights

    def __learning_conjugate(self, x, y, t):
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
        if self.learning == 'decay' or self.learning == 'newton':
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
        if (self.learning == 'decay' or self.learning == 'newton') and i == j:
            decay = self.lam / (self.dim + 1)
        else:
            decay = 0
        return np.sum(x[:,i] * y * (1 - y) * x[:,j])/N + decay
