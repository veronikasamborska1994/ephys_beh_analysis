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
    [4,15,18]]
    

B = [[2, 4, 5], 
    [-7, 4, 9],
    [8, 3, 9],
    [5, 11, 9]]

A = np.asarray(A)
B = np.asarray(B)

u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(A, full_matrices = True)

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