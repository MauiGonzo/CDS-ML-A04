import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron
import import_data


trainX = import_data.loadX("data/train-images-idx3-ubyte.gz")
trainLabel = import_data.loadY("data/train-labels-idx1-ubyte.gz")
testX = import_data.loadX("data/t10k-images-idx3-ubyte.gz")
testLabel = import_data.loadY("data/t10k-labels-idx1-ubyte.gz")

)
# Select 3 and 7 numbers X,Y
trainLabelT = np.zeros(len(trainLabel))
trainLabelS = np.zeros(len(trainLabel)
for i in len(trainLabel):
    if trainLabel[i] == 3:
        trainLabelT[i] = 1
    if trainLabel[i] == 7:
        trainLabelS[i] = 1


model = perceptron.Perceptron()

# for i in methods:
        #fit, for different methods, record time

#test

#plot different methods

