#this is the import function
#more text and more
#github daniel-e

import struct, gzip
import scipy.io
import numpy as np

def loadY(fnlabel):
	f = gzip.open(fnlabel, 'rb')
	f.read(8)
	return np.frombuffer(f.read(), dtype = np.uint8)

def loadX(fnimg):
	f = gzip.open(fnimg, 'rb')
	f.read(16)
	return np.frombuffer(f.read(), dtype = np.uint8).reshape((-1, 28*28))


if __name__ == "__main__":
	trainX = loadX("data/train-images-idx3-ubyte.gz")
	trainY = loadY("data/train-labels-idx1-ubyte.gz")
	testX = loadX("data/t10k-images-idx3-ubyte.gz")
	testY = loadY("data/t10k-labels-idx1-ubyte.gz")
