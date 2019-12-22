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
from scipy import stats

file_HP = 'smthRm_place_all_animals.mat'
file_EC = 'smthRm_grid_all_animals.mat'
dict_HP = 'smthRm_place_all_animals'
dict_EC = 'smthRm_grid_all_animals'

def place_grid_cells_SVDs(file, dictionary, c = 'green'):
    
    mat_place_cells = scipy.io.loadmat('/Users/veronikasamborska/Desktop/grid_and_place_cells/'+ file)
    
    
    place_cells = mat_place_cells[dictionary]
    
    place_cells = place_cells[:41]
    
    
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
        filled = gaussian_filter(filled, 2)
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
    
    place_cell_all = np.concatenate([place_cells_trial_1_vector,place_cells_trial_2_vector,place_cells_trial_3_vector, place_cells_trial_4_vector, place_cells_trial_5_vector], axis =1 )
    
    place_cell_mean = np.mean(place_cell_all, axis = 1)
    
    place_cells_trial_1_vector = np.transpose(place_cells_trial_1_vector)- place_cell_mean
    place_cells_trial_1_vector = np.transpose(place_cells_trial_1_vector)
    
    place_cells_trial_2_vector = np.transpose(place_cells_trial_2_vector)- place_cell_mean
    place_cells_trial_2_vector = np.transpose(place_cells_trial_2_vector)
    
    place_cells_trial_3_vector = np.transpose(place_cells_trial_3_vector)- place_cell_mean
    place_cells_trial_3_vector = np.transpose(place_cells_trial_3_vector)
    
    place_cells_trial_4_vector = np.transpose(place_cells_trial_4_vector)- place_cell_mean
    place_cells_trial_4_vector = np.transpose(place_cells_trial_4_vector)
    
    place_cells_trial_5_vector = np.transpose(place_cells_trial_5_vector)- place_cell_mean
    place_cells_trial_5_vector = np.transpose(place_cells_trial_5_vector)
    
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
    full =  np.linalg.multi_dot([t_u, place_cells_trial_4_vector, t_v])
    full_diagonal = np.diagonal(full)
    cum_sum = np.cumsum(full_diagonal)/place_cells_trial_4_vector.shape[0]
    
    x_task_2_from_task_1 = np.linalg.multi_dot([t_u, place_cells_trial_4_vector])
    var_x_task_2_from_task_1 = np.sum(x_task_2_from_task_1**2, axis = 1)
    cum_var_x_task_2_from_task_1 = np.cumsum(var_x_task_2_from_task_1)/np.sqrt(place_cells_trial_4_vector.shape[0])
    cum_var_x_task_2_from_task_1 = cum_var_x_task_2_from_task_1/cum_var_x_task_2_from_task_1[-1]
    
    y_task_2_from_task_1 = np.linalg.multi_dot([place_cells_trial_4_vector,t_v])
    var_y_task_2_from_task_1 = np.sum(y_task_2_from_task_1**2, axis = 1)
    cum_var_y_task_2_from_task_1 = np.cumsum(var_y_task_2_from_task_1)/np.sqrt(place_cells_trial_4_vector.shape[0])
    cum_var_y_task_2_from_task_1 = cum_var_y_task_2_from_task_1/cum_var_y_task_2_from_task_1[-1]
    
    
    #Compare task 1 Second Half 
    full_between =  np.linalg.multi_dot([t_u, place_cells_trial_5_vector, t_v])
    full_diagonal_between = np.diagonal(full_between)
    cum_sum_within = np.cumsum(full_diagonal_between)/place_cells_trial_5_vector.shape[0]
    
    
    x_task_1_from_task_1 = np.linalg.multi_dot([t_u, place_cells_trial_5_vector])
    var_x_task_1_from_task_1 = np.sum(x_task_1_from_task_1**2, axis = 1)
    cum_var_x_task_1_from_task_1 = np.cumsum(var_x_task_1_from_task_1)/np.sqrt(place_cells_trial_5_vector.shape[0])
    cum_var_x_task_1_from_task_1 = cum_var_x_task_1_from_task_1/cum_var_x_task_1_from_task_1[-1]
    
    y_task_1_from_task_1 = np.linalg.multi_dot([place_cells_trial_5_vector, t_v])
    var_y_task_1_from_task_1 = np.sum(y_task_1_from_task_1**2, axis = 1)
    cum_var_y_task_1_from_task_1 = np.cumsum(var_y_task_1_from_task_1)/np.sqrt(place_cells_trial_5_vector.shape[0])
    cum_var_y_task_1_from_task_1 = cum_var_y_task_1_from_task_1/cum_var_y_task_1_from_task_1[-1]
    
    plt.figure()

    plt.plot(cum_var_x_task_2_from_task_1, label = 'Between EC Left', color = c, linestyle = '--', alpha = 0.7)
    plt.plot(cum_var_x_task_1_from_task_1, label = 'Within EC Left', color = c,alpha = 0.7)
    #plt.plot(cum_var_y_task_2_from_task_1, label = 'Between Right Eigenvectors HP', color = 'black', linestyle = '--', alpha = 0.7)
    #plt.plot(cum_var_y_task_1_from_task_1, label = 'Within Right Eigenvectors HP', color = 'black',alpha = 0.7)
    
    plt.legend()
    
    plt.figure()
    plt.plot(cum_sum_within, label = 'Full Within HP', color = c)
    plt.plot(cum_sum, label = 'Full Between HP', color = c,linestyle = '--')
    
    plt.figure()
    plt.imshow(t_v[:,5].reshape(50, 50).T,cmap = 'jet')
    plt.figure()
    plt.plot(t_v[:,5])
    
    fig = plt.figure(num = 11, figsize=(5,10))
    fig.add_subplot(10,2,1)
    plt.imshow(t_v[:,0].reshape(50, 50).T, cmap = 'jet')
    plt.title('Arena 1')
    plt.ylabel('Eig 1')
    
    fig.add_subplot(10,2,2)
    plt.imshow(t_v_t_2_1[:,0].reshape(50, 50).T, cmap = 'jet')
    plt.title('Arena 2')
    
    
    fig.add_subplot(10,2,3)
    plt.imshow(t_v[:,1].reshape(50, 50).T, cmap = 'jet')
    plt.ylabel('Eig 2')
    
    
    fig.add_subplot(10,2,4)
    plt.imshow(t_v_t_2_1[:,1].reshape(50, 50).T,cmap = 'jet')

    
    fig.add_subplot(10,2,5)
    plt.imshow(t_v[:,2].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 3')
    
    
    fig.add_subplot(10,2,6)
    plt.imshow(t_v_t_2_1[:,2].reshape(50, 50).T, cmap = 'jet')
    
    
    fig.add_subplot(10,2,7)
    plt.imshow(t_v[:,3].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 4')
    
    fig.add_subplot(10,2,8)
    plt.imshow(t_v_t_2_1[:,3].reshape(50, 50).T, cmap = 'jet')
    
    
    
    
    fig.add_subplot(10,2,9)
    plt.imshow(t_v[:,4].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 5')
    
    
    fig.add_subplot(10,2,10)
    plt.imshow(t_v_t_2_1[:,4].reshape(50, 50).T,cmap = 'jet')
    
    
    
    fig.add_subplot(10,2,11)
    plt.imshow(t_v[:,5].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 6')
    
    
    fig.add_subplot(10,2,12)
    plt.imshow(t_v_t_2_1[:,5].reshape(50, 50).T,cmap = 'jet')
    
    
    fig.add_subplot(10,2,13)
    plt.imshow(t_v[:,6].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 7')
    
    
    fig.add_subplot(10,2,14)
    plt.imshow(t_v_t_2_1[:,6].reshape(50, 50).T,cmap = 'jet')
    
    fig.add_subplot(10,2,15)
    plt.imshow(t_v[:,7].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 8')
    
    
    fig.add_subplot(10,2,16)
    plt.imshow(t_v_t_2_1[:,7].reshape(50, 50).T,cmap = 'jet')
    
    
    fig.add_subplot(10,2,17)
    plt.imshow(t_v[:,8].reshape(50, 50).T,cmap = 'jet')
    plt.ylabel('Eig 9')
    
    
    fig.add_subplot(10,2,18)
    plt.imshow(t_v_t_2_1[:,8].reshape(50, 50).T,cmap = 'jet')
    
    
    
    fig.add_subplot(10,2,19)
    plt.imshow(t_v[:,9].reshape(50, 50).T,cmap = 'jet')
    
    
    plt.ylabel('Eig 10')
    fig.add_subplot(10,2,20)
    plt.imshow(t_v_t_2_1[:,9].reshape(50, 50).T,cmap = 'jet')
    
    
    
    return cum_var_x_task_2_from_task_1, cum_var_x_task_1_from_task_1



def area_under_curve():
    between_HP,within_HP = place_grid_cells_SVDs(file_HP, dict_HP)
    
    between_EC,within_EC = place_grid_cells_SVDs(file_EC, dict_EC)
    HP_area = (np.trapz(within_HP) - np.trapz(between_HP))#/within_HP.shape[0]
    EC_area = (np.trapz(within_EC) - np.trapz(between_EC))#/within_EC.shape[0]
    sns.barplot(data=[HP_area,EC_area], capsize=.1, ci="sd",  palette="Blues_d")
    
def plot():
    

    
    
    place_cells_trial_1_vector = np.asarray(place_cells_trial_1_vector)
    place_cells_trial_2_vector = np.asarray(place_cells_trial_2_vector)
    place_cells_trial_3_vector = np.asarray(place_cells_trial_3_vector)
    place_cells_trial_4_vector = np.asarray(place_cells_trial_4_vector)
    place_cells_trial_5_vector = np.asarray(place_cells_trial_5_vector)  
        
    
    place_cells_trial_1_vector_mean = np.mean(place_cells_trial_1_vector, axis = 1)
    place_cells_trial_1_vector_std = np.std(place_cells_trial_1_vector, axis = 1)
    
    demeaned_place_cells_trial_1_vector = np.transpose(place_cells_trial_1_vector)- place_cells_trial_1_vector_mean/place_cells_trial_1_vector_std
    demeaned_place_cells_trial_1_vector = np.transpose(demeaned_place_cells_trial_1_vector)
    
    
    place_cells_trial_2_vector_mean = np.mean(place_cells_trial_2_vector, axis = 1)
    place_cells_trial_2_vector_std = np.std(place_cells_trial_2_vector, axis = 1)
    
    demeaned_place_cells_trial_2_vector = np.transpose(place_cells_trial_2_vector)- place_cells_trial_2_vector_mean/place_cells_trial_2_vector_std
    demeaned_place_cells_trial_2_vector = np.transpose(demeaned_place_cells_trial_2_vector)
    
    
    place_cells_trial_3_vector_mean = np.mean(place_cells_trial_3_vector, axis = 1)
    place_cells_trial_3_vector_std = np.std(place_cells_trial_3_vector, axis = 1)
    
    demeaned_place_cells_trial_3_vector = np.transpose(place_cells_trial_3_vector)- place_cells_trial_3_vector_mean/place_cells_trial_3_vector_std
    demeaned_place_cells_trial_3_vector = np.transpose(demeaned_place_cells_trial_3_vector)
    
    
    place_cells_trial_4_vector_mean = np.mean(place_cells_trial_4_vector, axis = 1)
    place_cells_trial_4_vector_std = np.std(place_cells_trial_4_vector, axis = 1)
    
    demeaned_place_cells_trial_4_vector = np.transpose(place_cells_trial_4_vector)- place_cells_trial_4_vector_mean/place_cells_trial_4_vector_std
    demeaned_place_cells_trial_4_vector = np.transpose(demeaned_place_cells_trial_4_vector)
    
    
    place_cells_trial_5_vector_mean = np.mean(place_cells_trial_5_vector, axis = 1)
    place_cells_trial_5_vector_std = np.std(place_cells_trial_5_vector, axis = 1)
    
    demeaned_place_cells_trial_5_vector = np.transpose(place_cells_trial_5_vector)- place_cells_trial_5_vector_mean/place_cells_trial_5_vector_std
    demeaned_place_cells_trial_5_vector = np.transpose(demeaned_place_cells_trial_5_vector)
    
    
    correlation_task_1 = np.linalg.multi_dot([demeaned_place_cells_trial_1_vector, np.transpose(demeaned_place_cells_trial_1_vector)])/demeaned_place_cells_trial_5_vector.shape[0]
    correlation_task_2 = np.linalg.multi_dot([demeaned_place_cells_trial_2_vector, np.transpose(demeaned_place_cells_trial_2_vector)])/demeaned_place_cells_trial_5_vector.shape[0]
    correlation_task_3 = np.linalg.multi_dot([demeaned_place_cells_trial_3_vector, np.transpose(demeaned_place_cells_trial_3_vector)])/demeaned_place_cells_trial_5_vector.shape[0]
    correlation_task_4 = np.linalg.multi_dot([demeaned_place_cells_trial_4_vector, np.transpose(demeaned_place_cells_trial_4_vector)])/demeaned_place_cells_trial_5_vector.shape[0]
    correlation_task_5 = np.linalg.multi_dot([demeaned_place_cells_trial_5_vector, np.transpose(demeaned_place_cells_trial_5_vector)])/demeaned_place_cells_trial_5_vector.shape[0]
    
    #correlation_task_1 = np.corrcoef(demeaned_place_cells_trial_1_vector,demeaned_place_cells_trial_1_vector)
    #correlation_task_2 = np.corrcoef(demeaned_place_cells_trial_2_vector,demeaned_place_cells_trial_2_vector)
    #correlation_task_3 = np.corrcoef(demeaned_place_cells_trial_3_vector,demeaned_place_cells_trial_3_vector)
    #correlation_task_4 = np.corrcoef(demeaned_place_cells_trial_4_vector,demeaned_place_cells_trial_4_vector)
    #correlation_task_5 = np.corrcoef(demeaned_place_cells_trial_5_vector,demeaned_place_cells_trial_5_vector)
    
    correlation_task_1 = np.triu(correlation_task_1)
    correlation_task_1 = correlation_task_1.flatten()
    
    correlation_task_2= np.triu(correlation_task_2)
    correlation_task_2 = correlation_task_2.flatten()
    
    correlation_task_3 = np.triu(correlation_task_3)
    correlation_task_3 = correlation_task_3.flatten()
    
    correlation_task_4 = np.triu(correlation_task_4)
    correlation_task_4 = correlation_task_4.flatten()
    
    correlation_task_5 = np.triu(correlation_task_5)
    correlation_task_5 = correlation_task_5.flatten()
    
    mean_correlation_within = np.mean([correlation_task_1, correlation_task_5], axis = 0)
    mean_correlation_between = np.mean([correlation_task_2, correlation_task_3, correlation_task_4], axis = 0)
    
    mean_w = np.mean(mean_correlation_within)
    mean_b = np.mean(mean_correlation_between)
    std_w = np.std(mean_correlation_within)
    std_b = np.std(mean_correlation_between)
    mean_w_5_above = mean_w+(std_w*5)
    mean_b_5_above = mean_w+(std_b*5)
    
    mean_correlation_within_exclude = (np.where(mean_correlation_within < mean_w_5_above) and np.where(mean_correlation_between < mean_b_5_above))
    
    plt.figure()
    plt.scatter(mean_correlation_within[mean_correlation_within_exclude],mean_correlation_between[mean_correlation_within_exclude], s =1, color = 'black')
    
    gradient, intercept, r_value, p_value, std_err = stats.linregress(mean_correlation_within[mean_correlation_within_exclude],mean_correlation_between[mean_correlation_within_exclude])
    
    mn=np.min(mean_correlation_within[mean_correlation_within_exclude])
    mx=np.max(mean_correlation_within[mean_correlation_within_exclude])
    x1=np.linspace(mn,mx,500)
    y1=gradient*x1+intercept
    plt.plot(x1,y1,'black')
    plt.show()
    plt.xlabel('Arena 1')
    plt.ylabel('Arena 2')
    plt.title('Covariance EC')
    
    color = 'black'
    label = 'HP'
    
    fig = plt.figure(num = 11, figsize=(5,10))
    fig.add_subplot(10,2,1)
    plt.imshow(t_v[:,0].reshape(50, 50).T)
    plt.title('Arena 1')
    plt.ylabel('Eig 1')
    
    fig.add_subplot(10,2,2)
    plt.imshow(t_v_t_2_1[:,0].reshape(50, 50).T)
    plt.title('Arena 2')
    
    
    fig.add_subplot(10,2,3)
    plt.imshow(t_v[:,1].reshape(50, 50).T)
    plt.ylabel('Eig 2')
    
    
    fig.add_subplot(10,2,4)
    plt.imshow(t_v_t_2_1[:,1].reshape(50, 50).T)
    
    
    
    
    
    fig.add_subplot(10,2,5)
    plt.imshow(t_v[:,2].reshape(50, 50).T)
    plt.ylabel('Eig 3')
    
    
    fig.add_subplot(10,2,6)
    plt.imshow(t_v_t_2_1[:,2].reshape(50, 50).T)
    
    
    fig.add_subplot(10,2,7)
    plt.imshow(t_v[:,3].reshape(50, 50).T)
    plt.ylabel('Eig 4')
    
    fig.add_subplot(10,2,8)
    plt.imshow(t_v_t_2_1[:,3].reshape(50, 50).T)
    
    
    
    
    fig.add_subplot(10,2,9)
    plt.imshow(t_v[:,4].reshape(50, 50).T)
    plt.ylabel('Eig 5')
    
    
    fig.add_subplot(10,2,10)
    plt.imshow(t_v_t_2_1[:,4].reshape(50, 50).T)
    
    
    
    fig.add_subplot(10,2,11)
    plt.imshow(t_v[:,5].reshape(50, 50).T)
    plt.ylabel('Eig 6')
    
    
    fig.add_subplot(10,2,12)
    plt.imshow(t_v_t_2_1[:,5].reshape(50, 50).T)
    
    
    fig.add_subplot(10,2,13)
    plt.imshow(t_v[:,6].reshape(50, 50).T)
    plt.ylabel('Eig 7')
    
    
    fig.add_subplot(10,2,14)
    plt.imshow(t_v_t_2_1[:,6].reshape(50, 50).T)
    
    fig.add_subplot(10,2,15)
    plt.imshow(t_v[:,7].reshape(50, 50).T)
    plt.ylabel('Eig 8')
    
    
    fig.add_subplot(10,2,16)
    plt.imshow(t_v_t_2_1[:,7].reshape(50, 50).T)
    
    
    fig.add_subplot(10,2,17)
    plt.imshow(t_v[:,8].reshape(50, 50).T)
    plt.ylabel('Eig 9')
    
    
    fig.add_subplot(10,2,18)
    plt.imshow(t_v_t_2_1[:,8].reshape(50, 50).T)
    
    
    
    fig.add_subplot(10,2,19)
    plt.imshow(t_v[:,9].reshape(50, 50).T)
    
    
    plt.ylabel('Eig 10')
    fig.add_subplot(10,2,20)
    plt.imshow(t_v_t_2_1[:,9].reshape(50, 50).T)
    
