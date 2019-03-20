#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:06:08 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt

A = [[1, 4, 5], 
    [2, 8, 9],
    [3, 6, 10],
    [8,1,1]]
    

#B = [[1, 0, 0], 
#    [0, 2, 0],
#    [0, 0, 3],
#    [0,0,0]]


C = [[1,34, 5], 
    [2, 88, 9],
    [3, 6, 10],
    [48,15,18]]
    



A = np.asarray(A)
#B = np.asarray(B)

u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(A, full_matrices = False)
u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(C, full_matrices = False)

t_u = np.transpose(u_t1_1)  
t_v = np.transpose(vh_t1_1)

x = np.linalg.multi_dot([t_u, B])

var_rows = np.sum(x**2, axis = 1)
cum_var_rows = np.cumsum(var_rows)

#cum_var_rows =cum_var_rows/cum_var_rows[-1]

s = np.linalg.multi_dot([t_u, B, t_v])

s_var = np.sum(s**2, axis = 1)
s_var = np.cumsum(s_var)
#s_var =s_var/s_var[-1]

plt.plot(cum_var_rows)
plt.plot(s_var)