import numpy as np
import matplotlib.pyplot as plt
import perceptron

dim = 50
steps = 4*dim
runs = 100

M = np.zeros((runs,steps-1))

for i in range(runs):
    print('Starting run {}'.format(i))
    for N in range(1,steps):
        model = perceptron.Perceptron(dim)
        x = np.random.randint(2, size=(N,dim), dtype='int')
        t = np.random.choice([-1,1], size=N)
        M[i,N-1] = model.fit(x,t)
        y = model.predict(x)

M_avg = np.average(M, axis=0)
M_max = np.amax(M, axis=0)
M_min = np.amin(M, axis=0)

C = np.where(M > 10000, 0, 1)

frac = np.sum(C, axis=0)/runs

plt.plot(range(1,steps), M_min)
plt.plot(range(1,steps), M_max)
plt.plot(range(1,steps), M_avg)
plt.xlabel('p')
plt.ylabel('M: number of updates till convergence')
plt.title('Implementation of preceptron learning rule')
plt.show()

plt.plot(range(1,steps), frac)
plt.xlabel('p')
plt.ylabel('C(p,50)')
plt.title('Convergence fraction per p after {} training runs per p'.format(runs))
plt.show()
