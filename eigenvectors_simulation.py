#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:32:50 2018

@author: veronikasamborska
"""

from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import MinMaxScaler

# Loop degree, adjacency, laplacian eigenvectors 

degree_m = np.ones((8,8))
degree_m.fill(0.0)

a = np.ones((8,8))
a.fill(0.0)
b = np.ones(8)
connections =  np.ones(8)
connections.fill(2)

np.fill_diagonal(a[1:], b)
np.fill_diagonal(a[:,1:], b)
np.fill_diagonal(degree_m,connections)

#Transistion matrix
adj_loop = np.ones((8,8))
adj_loop.fill(0.0)
adj_connections = np.ones(8)
adj_connections.fill(0.5)

np.fill_diagonal(adj_loop[:,1:], adj_connections)
np.fill_diagonal(adj_loop[1:], adj_connections)
adj_loop[0,7] = 0.5
adj_loop[7,0] = 0.5

a[0,7] = 1
a[7,0] = 1
laplacian_m = degree_m - a
w,v = np.linalg.eig(laplacian_m)


#Adjacency matrix
c = np.ones((8,8))
c.fill(0.0)
d = np.ones(8)
np.fill_diagonal(c[1:], d)
np.fill_diagonal(c[:,1:], d)

#Transistion matrix
adj_line = np.ones((8,8))
adj_line.fill(0.0)
adj_connections = np.ones(8)
adj_connections.fill(0.5)
np.fill_diagonal(adj_line[:,1:], adj_connections)
#np.fill_diagonal(adj_line[1:], adj_connections)
adj_line[0,1] = 1
#adj_line[7,6] = 1

# Line degree, adjacency, laplacian eigenvectors 
degree_m_line =  np.ones((8,8))
degree_m_line.fill(0.0)
connections_line =  np.ones(8)
connections_line.fill(2)
np.fill_diagonal(degree_m_line,connections_line)
degree_m_line[0,0] = 1
degree_m_line[7,7] = 1

laplacian_m_line = degree_m_line - c

lw,lv = np.linalg.eig(laplacian_m_line)

#Eigendecomposition
fig, axes = plt.subplots(8, 1)
fig = figsize(3,8)

x = [0,1,2,3,4,5,6,7]
y = [1,1,1,1,1,1,1,1]


for i in range(w.shape[0]):
    eig = v[:,i]
    axes[i].scatter(x, y, c=eig, s=200)
    
#Eigendecomposition
fig_line, axes_line = plt.subplots(8, 1)
fig_line = figsize(3,8)


#normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
for i in range(lw.shape[0]):
    eig = lv[:,i]
    axes_line[i].scatter(x, y, c=eig, s=200)
    

imshow(adj_loop, vmax = 1)
colorbar()