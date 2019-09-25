import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron
import import_data


trainX = import_data.loadX("data/train-images-idx3-ubyte.gz")
trainY = import_data.loadY("data/train-labels-idx1-ubyte.gz")
testX = import_data.loadX("data/t10k-images-idx3-ubyte.gz")
testY = import_data.loadY("data/t10k-labels-idx1-ubyte.gz")

# Select 3 and 7 numbers X,Y

model = perceptron.Perceptron()

# for i in methods:
        #fit, for different methods, record time

#test

#plot different methods

