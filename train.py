import numpy as np
import matplotlib.pyplot as plt
import time
import perceptron
import import_data


trainX = import_data.loadX("data/train-images-idx3-ubyte.gz")
trainLabel = import_data.loadY("data/train-labels-idx1-ubyte.gz")
testX = import_data.loadX("data/t10k-images-idx3-ubyte.gz")
testLabel = import_data.loadY("data/t10k-labels-idx1-ubyte.gz")


'Select 3 and 7 from training set'
trainX_t = []
trainLabel_t = []
for i in range(len(trainLabel)):
        if trainLabel[i] == 3:
                trainX_t.append(trainX[i])
                trainLabel_t.append(0)
        elif trainLabel[i] == 7:
                trainX_t.append(trainX[i])
                trainLabel_t.append(1)

'Select 3 and 7 from test set'
testX_t = []
testLabel_t = []
for i in range(len(testLabel)):
        if testLabel[i] == 3:
                testX_t.append(testX[i])
                testLabel_t.append(0)
        elif testLabel[i] == 7:
                testX_t.append(testX[i])
                testLabel_t.append(1)

'Normalize data sets'
trainX_t = np.array(trainX_t)/np.amax(trainX_t)
trainLabel_t = np.array(trainLabel_t)

testX_t = np.array(testX_t)/np.amax(testX_t)
testLabel_t = np.array(testLabel_t)


'Make models'
model_gradient = perceptron.Perceptron(len(trainX_t[0]))
model_momentum = perceptron.Perceptron(len(trainX_t[0]))
model_decay = perceptron.Perceptron(len(trainX_t[0]))
model_newton = perceptron.Perceptron(len(trainX_t[0]))
model_line = perceptron.Perceptron(len(trainX_t[0]))
model_conjugate = perceptron.Perceptron(len(trainX_t[0]))


print('Training gradient:')
E_train_gradient, E_test_gradient = model_gradient.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = 'gradient', epochs=10)

print('Training momentum:')
E_train_momentum, E_test_momentum = model_momentum.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = 'momentum', epochs=10)

print('Training decay:')
E_train_decay, E_test_decay = model_decay.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = 'decay', epochs=10)


'Make plots'
fig, axs = plt.subplots(2, 2)

axs[0,0].plot(range(len(E_train_gradient)), E_train_gradient, label='Train')
axs[0,0].plot(range(len(E_test_gradient)), E_test_gradient, label='Test')
axs[0,0].set_title('Gradient: Lowest cost train: {}, lowest cost test: {}'.format(np.round(min(E_train_gradient), 4), np.round(min(E_test_gradient), 4)))
axs[0,0].legend()


axs[0,1].plot(range(len(E_train_momentum)), E_train_momentum, label='Train')
axs[0,1].plot(range(len(E_test_momentum)), E_test_momentum, label='Test')
axs[0,1].set_title('Momentum: Lowest cost train: {}, lowest cost test: {}'.format(np.round(min(E_train_momentum), 4), np.round(min(E_test_momentum), 4)))
axs[0,1].legend()

axs[1,0].plot(range(len(E_train_decay)), E_train_decay, label='Train')
axs[1,0].plot(range(len(E_test_decay)), E_test_decay, label='Test')
axs[1,0].set_title('Decay: Lowest cost train: {}, lowest cost test: {}'.format(np.round(min(E_train_decay), 4), np.round(min(E_test_decay), 4)))
axs[1,0].legend()
plt.show()
