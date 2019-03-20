#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:46:13 2019

@author: veronikasamborska
"""
import SVDs as sv
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

def svd_plotting(experiment, tasks_unchanged = True, HP = False, plot_a = False, plot_b = False, average_reward = False):    
    if tasks_unchanged == True:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
    
        all_data = np.concatenate([flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_first_half,\
                                   flattened_all_clusters_task_2_second_half, flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half], axis = 1)
        
        z_score = zscore(all_data, axis = 1)
        
        flattened_all_clusters_task_1_first_half = z_score[:,:flattened_all_clusters_task_1_first_half.shape[1]]
        flattened_all_clusters_task_1_second_half = z_score[:,flattened_all_clusters_task_1_first_half.shape[1]:flattened_all_clusters_task_1_first_half.shape[1]*2]
            
        flattened_all_clusters_task_2_first_half = z_score[:,flattened_all_clusters_task_1_first_half.shape[1]*2:flattened_all_clusters_task_1_first_half.shape[1]*3]
        flattened_all_clusters_task_2_second_half = z_score[:,flattened_all_clusters_task_1_first_half.shape[1]*3:flattened_all_clusters_task_1_first_half.shape[1]*4]
        
        flattened_all_clusters_task_3_first_half = z_score[:,flattened_all_clusters_task_1_first_half.shape[1]*4:flattened_all_clusters_task_1_first_half.shape[1]*5]
        flattened_all_clusters_task_3_second_half = z_score[:,flattened_all_clusters_task_1_first_half.shape[1]*5:flattened_all_clusters_task_1_first_half.shape[1]*6]
                
    else:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = sv.flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
   
    
    
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = True)
    
    if tasks_unchanged == True:
        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = True)
    
        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = True)


    #Finding eigenvectors of correlation matrix DxDt
    corr_t1_1 = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half, np.transpose(flattened_all_clusters_task_1_first_half)])

    eig_t1_1,vect_t1_1 = np.linalg.eig(corr_t1_1)
    
    corr_t1_2 = np.linalg.multi_dot([flattened_all_clusters_task_1_second_half, np.transpose(flattened_all_clusters_task_1_second_half)])

    eig_t1_2,vect_t1_2 = np.linalg.eig(corr_t1_2)
    
    corr_t2_1 = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half, np.transpose(flattened_all_clusters_task_2_first_half)])
    
    eig_t2_1,vect_t2_1 = np.linalg.eig(corr_t2_1)
    
    corr_t2_2 = np.linalg.multi_dot([flattened_all_clusters_task_2_second_half, np.transpose(flattened_all_clusters_task_2_second_half)])
    
    eig_t2_2,vect_t2_2 = np.linalg.eig(corr_t2_2)
    
    corr_t3_1 = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half, np.transpose(flattened_all_clusters_task_3_first_half)])
    
    eig_t3_1,vect_t3_1 = np.linalg.eig(corr_t3_1)
    
   
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    
    t_u = np.transpose(u_t1_1)  
    t_u_t_2_1 = np.transpose(u_t2_1)   
    
    if tasks_unchanged == True:
        t_u_t_3_1 = np.transpose(u_t3_1)
 
    var_t_1_on_t_1 =  (np.linalg.multi_dot([np.transpose(vect_t1_1), u_t1_2]))**2
    var_t_1_on_t_1 = np.linalg.multi_dot([eig_t1_1, np.transpose(var_t_1_on_t_1)])  
    
    var_t_2_on_t_1 =  (np.linalg.multi_dot([np.transpose(vect_t2_1), u_t1_2]))**2
    var_t_2_on_t_1 =  np.linalg.multi_dot([eig_t2_1,np.transpose(var_t_2_on_t_1)])
#
    var_t_2_on_t_2 =  (np.linalg.multi_dot([np.transpose(vect_t2_1), u_t2_2]))**2
    var_t_2_on_t_2 =  np.linalg.multi_dot([eig_t2_1, np.transpose(var_t_2_on_t_2)])
#    
    var_t_3_on_t_2 =  (np.linalg.multi_dot([np.transpose(vect_t3_1), u_t2_2]))**2
    var_t_3_on_t_2 =  np.linalg.multi_dot([eig_t3_1, np.transpose(var_t_3_on_t_2)])

#    var_t_1_on_t_1 =  (np.linalg.multi_dot([t_u, vect_t1_2]))**2
#    var_t_1_on_t_1 = np.linalg.multi_dot([eig_t1_2, np.transpose(var_t_1_on_t_1)])  
#    
#    var_t_2_on_t_1 =  (np.linalg.multi_dot([t_u_t_2_1, vect_t1_2]))**2
#    var_t_2_on_t_1 =  np.linalg.multi_dot([eig_t1_2,np.transpose(var_t_2_on_t_1)])
#
#    var_t_2_on_t_2 =  (np.linalg.multi_dot([t_u_t_2_1, vect_t2_2]))**2
#    var_t_2_on_t_2 =  np.linalg.multi_dot([eig_t2_2, np.transpose(var_t_2_on_t_2)])
#    
#    var_t_3_on_t_2 =  (np.linalg.multi_dot([t_u_t_3_1, vect_t2_2]))**2
#    var_t_3_on_t_2 =  np.linalg.multi_dot([eig_t2_2, np.transpose(var_t_3_on_t_2)])

    sum_var_t_2_on_t_1 = np.cumsum(var_t_2_on_t_1, axis = 0)/flattened_all_clusters_task_1_second_half.shape[0]
    
    sum_var_t_3_on_t_2 = np.cumsum(var_t_3_on_t_2, axis = 0)/flattened_all_clusters_task_1_second_half.shape[0]

    sum_var_t_1_on_t_1 = np.cumsum(var_t_1_on_t_1, axis = 0)/flattened_all_clusters_task_1_second_half.shape[0]

    sum_var_t_2_on_t_2 = np.cumsum(var_t_2_on_t_2, axis = 0)/flattened_all_clusters_task_1_second_half.shape[0]

    average_within = np.mean([sum_var_t_1_on_t_1,sum_var_t_2_on_t_2],axis = 0)
    average_within = average_within/average_within[-1]
    average_between = np.mean([sum_var_t_2_on_t_1,sum_var_t_3_on_t_2],axis = 0)
    average_between = average_between/average_between[-1]
    
    if HP == False:
        plt.plot(average_within, color = 'green', label = 'PFC Within')
        plt.plot(average_between, color = 'green',linestyle = '--', label = 'PFC Between')
    else:
        plt.plot(average_within, color = 'black', label = 'HP Within')
        plt.plot(average_between, color = 'black', linestyle = '--', label = 'HP Between')
    plt.legend()