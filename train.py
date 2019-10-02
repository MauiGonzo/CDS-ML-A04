import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron
import import_data


trainX = import_data.loadX("data/train-images-idx3-ubyte.gz")
trainLabel = import_data.loadY("data/train-labels-idx1-ubyte.gz")
testX = import_data.loadX("data/t10k-images-idx3-ubyte.gz")
testLabel = import_data.loadY("data/t10k-labels-idx1-ubyte.gz")


# Select 3 and 7 numbers X,Y
trainLabelT = np.zeros(len(trainLabel))
trainLabelS = np.zeros(len(trainLabel))
for i in range(len(trainLabel)):
        if trainLabel[i] == 3:
                trainLabelT[i] = 1
        if trainLabel[i] == 7:
                trainLabelS[i] = 1

trainX_t = []
trainLabel_t = []
for i in range(len(trainLabel)):
        if trainLabel[i] == 3:
                trainX_t.append(trainX[i])
                trainLabel_t.append(0)
        elif trainLabel[i] == 7:
                trainX_t.append(trainX[i])
                trainLabel_t.append(1)

trainX_t = np.array(trainX_t)/np.amax(trainX_t)
trainLabel_t = np.array(trainLabel_t)

model = perceptron.Perceptron(len(trainX_t[0]))

#E = model.fit(trainX_t, trainLabel_t, learning = 'gradient', epochs=100)
#E = model.fit(trainX_t, trainLabel_t, learning = 'momentum', epochs=100)
E = model.fit(trainX_t, trainLabel_t, learning = 'decay', epochs=100)

plt.plot(range(len(E)), E)
plt.title('Lowest cost: {}'.format(min(E)))
plt.show()


# for i in methods:
        #fit, for different methods, record time

#test

#plot different methods
