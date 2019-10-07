import numpy as np
import matplotlib.pyplot as plt
import time
from time import perf_counter
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

def grad_algo_testing(learning_type, nr_epochs, lr=1, alpha=0.1):
    model = perceptron.Perceptron(len(trainX_t[0]))
    t1_start = perf_counter()
    E, E_val = model.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = learning_type, learning_rate=lr, epochs=nr_epochs)
    t1_stop = perf_counter()
    print("Running time for {} = {} seconds".format(learning_type, np.round(t1_stop - t1_start, 2)))
    plt.plot(E, label='Train, lr={}, min cost={}'.format(lr, np.round(min(E), 4)))
    plt.plot(E_val, label='Test, lr={}, min cost={}'.format(lr, np.round(min(E_val), 4)))
    plt.title('Training with learning method: {}'.format(learning_type))
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.yscale('log', nonposy='clip')
    plt.legend()
    plt.savefig('cost_{}.png'.format(learning_type))
    plt.clf()
    plt.close()


'Training with gradient descent learning method'
'Search for best parameters'
for lr in [0.5,0.9,1,1.1,2]:
    model = perceptron.Perceptron(len(trainX_t[0]))
    t1_start = perf_counter()
    E, E_val = model.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = 'gradient', learning_rate=lr, epochs=500)
    t1_stop = perf_counter()
    print("Running time for {} = {} seconds".format('gradient', np.round(t1_stop - t1_start, 2)))
    plt.plot(E, label='Train, lr={}, alpha={}, min cost={}'.format(lr, alpha, np.round(min(E), 4)))
    plt.plot(E_val, label='Test, lr{}, alpha={}, min cost={}'.format(lr, alpha, np.round(min(E_val), 4)))
plt.title('Training with learning method: {}'.format('gradient'))
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.yscale('log', nonposy='clip')
plt.legend()
plt.savefig('cost_{}.png'.format('gradient'))
plt.clf()
plt.close()

'Training with momentum learning method'
'Search for best parameters'
for lr in [0.5, 1, 2]:
    for alpha in [0.05, 0.1, 0.2]:
        model = perceptron.Perceptron(len(trainX_t[0]))
        t1_start = perf_counter()
        E, E_val = model.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = 'momentum', learning_rate=lr, alpha=alpha, epochs=500)
        t1_stop = perf_counter()
        print("Running time for {} = {} seconds".format('momentum', np.round(t1_stop - t1_start, 2)))
        plt.plot(E, label='Train, lr={}, alpha={}, min cost={}'.format(lr, alpha, np.round(min(E), 4)))
        plt.plot(E_val, label='Test, lr{}, alpha={}, min cost={}'.format(lr, alpha, np.round(min(E_val), 4)))
plt.title('Training with learning method: {}'.format('momentum'))
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.yscale('log', nonposy='clip')
plt.legend()
plt.savefig('cost_{}.png'.format('momentum'))
plt.clf()
plt.close()

'Training with weight decay learning method'
grad_algo_testing('decay',    1000)

'Training with weight decay learning method'
grad_algo_testing('newton', 10)

'Training with line search learning method'
grad_algo_testing('line', 100)

'Training with conjugate gradient descent learning method'
grad_algo_testing('conjugate', 50)

'Training with stochastic gradient descent learning method'
'Search for best parameters'
for lr in [0.5,0.9,1,1.1,2]:
    model = perceptron.Perceptron(len(trainX_t[0]))
    t1_start = perf_counter()
    E, E_val = model.fit(trainX_t, trainLabel_t, testX_t, testLabel_t, learning = 'stochastic', learning_rate=lr, epochs=500)
    t1_stop = perf_counter()
    print("Running time for {} = {} seconds".format('stochastic', np.round(t1_stop - t1_start, 2)))
    plt.plot(E, label='Train, lr={}, min cost={}'.format(lr, np.round(min(E), 4)))
    plt.plot(E_val, label='Test, lr={}, min cost={}'.format(lr, np.round(min(E_val), 4)))
plt.title('Training with learning method: {}'.format('stochastic'))
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.yscale('log', nonposy='clip')
plt.legend()
plt.savefig('cost_{}.png'.format('stochastic'))
plt.clf()
plt.close()
