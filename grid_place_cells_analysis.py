#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:54:21 2019

@author: veronikasamborska
"""

import scipy.io
import numpy as np 
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter
mat_place_cells = scipy.io.loadmat('/Users/veronikasamborska/Desktop/smthRm_grid_all_animals.mat')

#smthRm_grid_all_animals.mat
place_cells = mat_place_cells['smthRm_grid_all_animals']
place_cells = place_cells[:41]

#smthRm_place_all_animals.mat

place_cells_trial_1 = place_cells[:,0]
place_cells_trial_2 = place_cells[:,1]
place_cells_trial_3 = place_cells[:,2]
place_cells_trial_4 = place_cells[:,3]
place_cells_trial_5 = place_cells[:,4]

place_cells_trial_1_int = []
for i in range(len(place_cells_trial_1)):
    image = place_cells_trial_1[i]
    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    #filled = gaussian_filter(filled, 2)
    place_cells_trial_1_int.append(filled)
    
place_cells_trial_2_int = []
for i in range(len(place_cells_trial_2)):
    image = place_cells_trial_2[i]
    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    #filled = gaussian_filter(filled, 2)
    place_cells_trial_2_int.append(filled)

place_cells_trial_3_int = []
for i in range(len(place_cells_trial_3)):
    image = place_cells_trial_3[i]
    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    #filled = gaussian_filter(filled, 2)
    place_cells_trial_3_int.append(filled)
 
place_cells_trial_4_int = []
for i in range(len(place_cells_trial_4)):
    image = place_cells_trial_4[i]
    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    #filled = gaussian_filter(filled, 2)
    place_cells_trial_4_int.append(filled)
    
place_cells_trial_5_int = []
for i in range(len(place_cells_trial_5)):
    image = place_cells_trial_5[i]
    valid_mask = ~np.isnan(image)
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(image.shape))).reshape(image.shape)
    #filled = gaussian_filter(filled, 2)
    place_cells_trial_5_int.append(filled)
    
    

    
place_cells_trial_1_vector = []
for i in range(place_cells_trial_1.shape[0]):
    place_cells_trial_1_vector.append(place_cells_trial_1_int[i].flatten())

place_cells_trial_2_vector = []
for i in range(place_cells_trial_2.shape[0]):
    place_cells_trial_2_vector.append(place_cells_trial_2_int[i].flatten())
    
place_cells_trial_3_vector = []
for i in range(place_cells_trial_3.shape[0]):
    place_cells_trial_3_vector.append(place_cells_trial_3_int[i].flatten())
    
place_cells_trial_4_vector = []
for i in range(place_cells_trial_4.shape[0]):
    place_cells_trial_4_vector.append(place_cells_trial_4_int[i].flatten())
    
place_cells_trial_5_vector = []
for i in range(place_cells_trial_5.shape[0]):
    place_cells_trial_5_vector.append(place_cells_trial_5_int[i].flatten())
    
place_cells_trial_1_vector = np.asarray(place_cells_trial_1_vector)
place_cells_trial_2_vector = np.asarray(place_cells_trial_2_vector)
place_cells_trial_3_vector = np.asarray(place_cells_trial_3_vector)
place_cells_trial_4_vector = np.asarray(place_cells_trial_4_vector)
place_cells_trial_5_vector = np.asarray(place_cells_trial_5_vector)


u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(place_cells_trial_1_vector, full_matrices = True)
        
u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(place_cells_trial_2_vector, full_matrices = True)
    
u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(place_cells_trial_3_vector, full_matrices = True)
    
u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(place_cells_trial_4_vector, full_matrices = True)


#Finding variance explained in second half of task 1 using the Us and Vs from the first half
t_u = np.transpose(u_t1_1)  
t_v = np.transpose(vh_t1_1)  

t_u_t_1_2 = np.transpose(u_t1_2)   
t_v_t_1_2 = np.transpose(vh_t1_2)  

t_u_t_2_1 = np.transpose(u_t2_1)   
t_v_t_2_1 = np.transpose(vh_t2_1)  

t_u_t_2_2 = np.transpose(u_t2_2)  
t_v_t_2_2 = np.transpose(vh_t2_2)  

#Compare task 1 Second Half 
x_task_2_from_task_1 = np.linalg.multi_dot([t_u, place_cells_trial_4_vector])
var_x_task_2_from_task_1 = np.sum(x_task_2_from_task_1**2, axis = 1)
cum_var_x_task_2_from_task_1 = np.cumsum(var_x_task_2_from_task_1)/np.sqrt(place_cells_trial_4_vector.shape[0])
cum_var_x_task_2_from_task_1 = cum_var_x_task_2_from_task_1/cum_var_x_task_2_from_task_1[-1]


#Compare task 1 Second Half 
x_task_1_from_task_1 = np.linalg.multi_dot([t_u, place_cells_trial_5_vector])
var_x_task_1_from_task_1 = np.sum(x_task_1_from_task_1**2, axis = 1)
cum_var_x_task_1_from_task_1 = np.cumsum(var_x_task_1_from_task_1)/np.sqrt(place_cells_trial_5_vector.shape[0])
cum_var_x_task_1_from_task_1 = cum_var_x_task_1_from_task_1/cum_var_x_task_1_from_task_1[-1]
 

plt.plot(cum_var_x_task_2_from_task_1, label = 'Between EC', color = 'green', linestyle = '--', alpha = 0.7)
plt.plot(cum_var_x_task_1_from_task_1, label = 'Within EC', color = 'green',alpha = 0.7)

plt.legend()


