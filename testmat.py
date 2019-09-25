#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:35:08 2019

@author: mauriceanno2019
"""

#test mat-file
import scipy.io as sio
import matplotlib.pyplot as plt

d = sio.loadmat("mnist.mat")

k = d['trainX']  # shape = (60000, 768)
l = d['trainY']  # shape = (1, 60000)

print(k.shape)

for i in range(10):
	print(l[0, i])
	plt.imshow(k[i].reshape((28, 28)) / 255.0)
	plt.show()