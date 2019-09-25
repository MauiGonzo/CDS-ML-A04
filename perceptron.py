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
        self.dim = dim
        self.weights = self.__initialize_weights(self.dim)
        self.learning_rate = 1

# function to initialize the weights with random data
# returns dim size float or ndarray of floats
    def __initialize_weights(self, dim):
        weights = np.random.random(dim+1)
        return weights

# function that returns update for weights#
    def predict(self, x_u):
        x_u = x_u.insert(0,1)
        return self.__sign(sum(self.weights*x_u))

# function does the fitting
# M is nr of updates or learning cycles

    def fit(self, x, t):
        

        if len(x[0]) != self.dim:
            print('Error: dimensions of input ({}) do not match to dimensions of Perceptron ({})'.format(len(x[0]), self.dim))
        x = np.insert(x,0,1, axis=1)
        z = x
        for i in range(len(x)):
            z[i] = x[i]*t[i]
        M = [0] * len(x) # List for counting how many times item in x has been updated
        M_old = []
        while np.any(M_old != M): # If the update counting matrix remains the same it has converges
            if np.sum(M) > 10000: break
            M_old = np.copy(M)
            for i in range(len(z)):
                if np.sum(self.weights*z[i])<0:
                    self.__update_weights(z[i])
                    M[i] += 1
        return np.sum(M)


    def __softmax(self, a):
        #Write softmax function
        pass
        return y

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