import numpy as np
import progressbar
from scipy import optimize
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
            E_val.append(self.__cost(x_val, y_val, t_val))
            print('Epoch: {:3}, E train: {}, E test: {}'.format(len(E), np.round(E[-1],6), np.round(E_val[-1],6)))
            self.weights += learning_function(x, y, t)
            # bar.update(len(E))
            if len(E) == epochs:
                self.stopping_condition = True
        class_error_train = self.__class_error(y, t)
        class_error_test = self.__class_error(y_val, t_val)
        print('Train classification error: {}%, Test classification error: {}%'.format(class_error_train, class_error_test))
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

    def __class_error(self, y, t):
        # Calculates the classification error, given prediction y and target t
        # return the classification error
        N = len(y)
        n_faults = 0
        for i in range(N):
            if np.abs(np.round(y[i]) - t[i]) > 0:
                n_faults += 1
        class_error_percentage = np.round(n_faults / N * 100, 2)
        return class_error_percentage


    def __softmax(self, x):
        # Calculates softmax of input x
        return (1 + np.exp(-x))**(-1)

    def __learning_gradient(self, x, y, t):
        learning_rate = 1
        delta_w = - learning_rate * self.__gradient(x, y, t)
        if np.all(delta_w == 0):
            self.stopping_condition = True
        return delta_w

    def __learning_momentum(self, x, y, t):
        # momentum learning method for updating weights
        learning_rate = 1
        alpha = 0.1
        # A) use rough approximation method to calculate gradient with momentum 
        # delta_w = - learning_rate/(1 - alpha) * self.__gradient(x, y, t)
        # B) add delta-w-min-1 to gradient
        if hasattr(self, 'delta_w_min_one' ):
            # delta_w_min_one has been created
            delta_w = - learning_rate * self.__gradient(x, y, t) + alpha * self.delta_w_min_one
        else:
            # else delta_w_min_one, has not been created
            delta_w = - learning_rate * self.__gradient(x, y, t)       # the delta_w_min_one-part during first epoch =0
        self.delta_w_min_one = np.copy(delta_w)
        # return delta_weights
        return delta_w

    def __learning_decay(self, x, y, t):
        # now add weight decay term and use the momentum learning method
        learning_rate = 1
        alpha = 0.1
        # A) use rough approximation method to calculate gradient with momentum 
        # delta_w = - learning_rate/(1 - alpha) * self.__gradient(x, y, t)
        # B) add delta-w-min-1 to gradient
        if hasattr(self, 'delta_w_min_one' ):
            # delta_w_min_one has been created
            delta_w = - learning_rate * self.__gradient(x, y, t) + alpha * self.delta_w_min_one
        else:
            # else delta_w_min_one, has not been created
            delta_w = - learning_rate * self.__gradient(x, y, t)       # the delta_w_min_one-part during first epoch =0
        self.delta_w_min_one = np.copy(delta_w)
        # return delta_weights
        # return delta_weights
        return delta_w

    def __learning_newton(self, x, y, t):
        hessian = self.__hessian(x, y)
        gradient = self.__gradient(x, y, t)
        return -1*np.linalg.inv(hessian).dot(gradient)

    def __learning_line(self, x, y, t):
        d = -1 * self.__gradient(x, y, t)
        gamma = 1
        res = optimize.minimize_scalar(self.__cost_line_search, args=[x,t,d])
        return res.x * d

    def __cost_line_search(self, gamma, args):
        x, t, d = args
        N = len(x)
        y = np.zeros(N, dtype='float64')
        for i in range(len(x)):
            y[i] = self.__softmax(np.sum((self.weights + gamma*d)*x[i]))
        return -1/N * np.sum(np.where(t == 1, np.log(y + 1e-50), np.log(1-y+1e-50)))


    def __learning_conjugate(self, x, y, t):
        if 'self.d' not in globals():
            self.d = -1 * self.__gradient(x, y, t)
        else:
            self.d = -1 * self.__gradient(x, y, t) + self.__polak_ribiere() * self.d
        gamma = 1
        res = optimize.minimize_scalar(self.__cost_line_search, args=[x,t,self.d])
        self.weights_old = np.copy(self.weights)
        return res.x * self.d

    def __polak_ribiere(self):
        weights_temp = np.copy(self.weights)
        self.weights = self.weights_old
        g_w0 = self.__gradient(x,y,t)
        self.weights = np.copy(weights_temp)
        g_w1 = self.__gradient(x,y,t)
        return (g_w1 - g_w0).dot(g_w1)/np.sum(g_w1*g_w1)

    def __learning_stochastic(self, x, y, t):
        pass

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
