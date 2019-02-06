#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:09:42 2019

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt

# Create vectors for each state's connectivity
 
state_1 = np.asarray([0,1,1,1,0,0,0,1,0,0,0,0,0,0])
state_2 = np.asarray([1,0,1,1,0,0,0,0,0,0,0,0,0,0])
state_3 = np.asarray([1,1,0,0,0,0,0,1,0,0,0,0,0,0])
state_4 = np.asarray([1,1,0,0,0,0,0,1,0,0,0,0,0,0])
state_5 = np.asarray([1,0,0,0,0,1,1,0,0,0,0,0,0,0])
state_6 = np.asarray([1,0,0,0,1,0,0,1,0,0,0,0,0,0])
state_7 = np.asarray([1,0,0,0,1,0,0,1,0,0,0,0,0,0])
state_8 = np.asarray([0,0,1,1,0,1,1,0,1,0,0,1,0,0])
state_9 = np.asarray([0,0,0,0,0,0,0,1,0,1,1,0,0,0])
state_10 = np.asarray([1,0,0,0,0,0,0,1,1,0,0,0,0,0])
state_11 = np.asarray([1,0,0,0,0,0,0,1,1,0,0,0,0,0])
state_12 = np.asarray([0,0,0,0,0,0,0,1,0,0,0,0,1,1])
state_13 = np.asarray([1,0,0,0,0,0,0,1,0,0,0,1,0,0])
state_14 = np.asarray([1,0,0,0,0,0,0,1,0,0,0,1,0,0])


adj_m = np.vstack((state_1,state_2,state_3, state_4, state_5,state_6, state_7, state_8, state_9,state_10, state_11,\
                   state_12, state_13, state_14))

w,v = np.linalg.eig(adj_m)
#x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]

fig, axes = plt.subplots(14, 1)

for i in range(w.shape[0]):
    eig = v[:,i]
    axes[i].plot(eig)
    