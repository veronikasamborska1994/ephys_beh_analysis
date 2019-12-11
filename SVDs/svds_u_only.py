#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:29:01 2019

@author: veronikasamborska
"""

import SVDs as sv 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import SVDs_replicate_Tim as svd_tim

def demean_data(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = True, z_score = False):
    
    
    if  tasks_unchanged == True:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
    else:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = sv.flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)

    if  demean_all_tasks == True:
        if tasks_unchanged == True:
            all_data = np.concatenate([flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_first_half,\
                                       flattened_all_clusters_task_2_second_half, flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half], axis = 1)
            all_data_mean = np.mean(all_data, axis = 1)
        
            if z_score == True:
                all_data_std = np.std(all_data, axis = 1)
                z_scored = np.transpose(all_data)- all_data_mean/all_data_std
                demeaned = np.transpose(z_scored)
            else:
            
                demeaned = np.transpose(all_data)- all_data_mean
                demeaned = np.transpose(demeaned)            
            
            demean_all_clusters_task_1_first_half = demeaned[:,:flattened_all_clusters_task_1_first_half.shape[1]]
            demean_all_clusters_task_1_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]:flattened_all_clusters_task_1_first_half.shape[1]*2]
            
            demean_all_clusters_task_2_first_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*2:flattened_all_clusters_task_1_first_half.shape[1]*3]
            demean_all_clusters_task_2_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*3:flattened_all_clusters_task_1_first_half.shape[1]*4]
        
            demean_all_clusters_task_3_first_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*4:flattened_all_clusters_task_1_first_half.shape[1]*5]
            demean_all_clusters_task_3_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*5:flattened_all_clusters_task_1_first_half.shape[1]*6]
                            
        else:
            all_data = np.concatenate([flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_first_half,\
                                       flattened_all_clusters_task_2_second_half], axis = 1)
            all_data_mean = np.mean(all_data, axis = 1)
        
            demeaned = np.transpose(all_data)- all_data_mean
            demeaned = np.transpose(demeaned)
            
            if z_score == True:
                all_data_std = np.std(all_data, axis = 1)
                z_scored = np.transpose(all_data)- all_data_mean/all_data_std
                demeaned = np.transpose(z_scored)
            else:
            
                demeaned = np.transpose(all_data)- all_data_mean
                demeaned = np.transpose(demeaned)
            
            demean_all_clusters_task_1_first_half = demeaned[:,:flattened_all_clusters_task_1_first_half.shape[1]]
            demean_all_clusters_task_1_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]:flattened_all_clusters_task_1_first_half.shape[1]*2]
            
            demean_all_clusters_task_2_first_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*2:flattened_all_clusters_task_1_first_half.shape[1]*3]
            demean_all_clusters_task_2_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*3:flattened_all_clusters_task_1_first_half.shape[1]*4]
           
    else:
        if tasks_unchanged == True:               
            mean_all_clusters_task_1_first_half = np.mean(flattened_all_clusters_task_1_first_half, axis = 1)
            demean_all_clusters_task_1_first_half = np.transpose(flattened_all_clusters_task_1_first_half)- mean_all_clusters_task_1_first_half 
            demean_all_clusters_task_1_first_half = np.transpose(demean_all_clusters_task_1_first_half)
           
            mean_all_clusters_task_1_second_half = np.mean(flattened_all_clusters_task_1_second_half, axis = 1)
            demean_all_clusters_task_1_second_half = np.transpose(flattened_all_clusters_task_1_second_half) - mean_all_clusters_task_1_second_half
            demean_all_clusters_task_1_second_half = np.transpose(demean_all_clusters_task_1_second_half)
        
            mean_all_clusters_task_2_first_half = np.mean(flattened_all_clusters_task_2_first_half, axis = 1)
            demean_all_clusters_task_2_first_half =  np.transpose(flattened_all_clusters_task_2_first_half) - mean_all_clusters_task_2_first_half 
            demean_all_clusters_task_2_first_half = np.transpose(demean_all_clusters_task_2_first_half)
        
            mean_all_clusters_task_2_second_half = np.mean(flattened_all_clusters_task_2_second_half, axis = 1)
            demean_all_clusters_task_2_second_half =  np.transpose(flattened_all_clusters_task_2_second_half) - mean_all_clusters_task_2_second_half
            demean_all_clusters_task_2_second_half = np.transpose(demean_all_clusters_task_2_second_half)
        
            mean_all_clusters_task_3_first_half = np.mean(flattened_all_clusters_task_3_first_half, axis = 1)
            demean_all_clusters_task_3_first_half =  np.transpose(flattened_all_clusters_task_3_first_half) - mean_all_clusters_task_3_first_half 
            demean_all_clusters_task_3_first_half = np.transpose(demean_all_clusters_task_3_first_half)
        
            mean_all_clusters_task_3_second_half = np.mean(flattened_all_clusters_task_3_second_half, axis = 1)
            demean_all_clusters_task_3_second_half =  np.transpose(flattened_all_clusters_task_3_second_half) - mean_all_clusters_task_3_second_half
            demean_all_clusters_task_3_second_half = np.transpose(demean_all_clusters_task_3_second_half)
        else: 
            mean_all_clusters_task_1_first_half = np.mean(flattened_all_clusters_task_1_first_half, axis = 1)
            demean_all_clusters_task_1_first_half = np.transpose(flattened_all_clusters_task_1_first_half)- mean_all_clusters_task_1_first_half 
            demean_all_clusters_task_1_first_half = np.transpose(demean_all_clusters_task_1_first_half)
           
            mean_all_clusters_task_1_second_half = np.mean(flattened_all_clusters_task_1_second_half, axis = 1)
            demean_all_clusters_task_1_second_half = np.transpose(flattened_all_clusters_task_1_second_half) - mean_all_clusters_task_1_second_half
            demean_all_clusters_task_1_second_half = np.transpose(demean_all_clusters_task_1_second_half)
        
            mean_all_clusters_task_2_first_half = np.mean(flattened_all_clusters_task_2_first_half, axis = 1)
            demean_all_clusters_task_2_first_half =  np.transpose(flattened_all_clusters_task_2_first_half) - mean_all_clusters_task_2_first_half 
            demean_all_clusters_task_2_first_half = np.transpose(demean_all_clusters_task_2_first_half)
        
            mean_all_clusters_task_2_second_half = np.mean(flattened_all_clusters_task_2_second_half, axis = 1)
            demean_all_clusters_task_2_second_half =  np.transpose(flattened_all_clusters_task_2_second_half) - mean_all_clusters_task_2_second_half
            demean_all_clusters_task_2_second_half = np.transpose(demean_all_clusters_task_2_second_half)
        
    if tasks_unchanged == True:
        return demean_all_clusters_task_1_first_half,demean_all_clusters_task_1_second_half, demean_all_clusters_task_2_first_half,\
        demean_all_clusters_task_2_second_half, demean_all_clusters_task_3_first_half, demean_all_clusters_task_3_second_half
    else:
        return  demean_all_clusters_task_1_first_half,demean_all_clusters_task_1_second_half, demean_all_clusters_task_2_first_half,\
        demean_all_clusters_task_2_second_half


def correlation_analysis(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = True, z_score = False):
    
    
    if tasks_unchanged == True:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = demean_data(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, demean_all_tasks = demean_all_tasks, z_score = False)
    else:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = demean_data(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, demean_all_tasks = demean_all_tasks, z_score = False)
   
        
    correlation_task_1 = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half, np.transpose(flattened_all_clusters_task_1_first_half)])
    correlation_task_1_2 = np.linalg.multi_dot([flattened_all_clusters_task_1_second_half, np.transpose(flattened_all_clusters_task_1_second_half)])
    
    correlation_task_2_1 = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half, np.transpose(flattened_all_clusters_task_2_first_half)])

    correlation_task_2_second_half = np.linalg.multi_dot([flattened_all_clusters_task_2_second_half, np.transpose(flattened_all_clusters_task_2_second_half)])
    
    correlation_task_3 = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half, np.transpose(flattened_all_clusters_task_3_first_half)])

    correlation_task_1 = np.triu(correlation_task_1)
    correlation_task_1 = correlation_task_1.flatten()

    correlation_task_1_2 = np.triu(correlation_task_1_2)
    correlation_task_1_2 = correlation_task_1_2.flatten()

    correlation_task_2_1 = np.triu(correlation_task_2_1)
    correlation_task_2_1 = correlation_task_2_1.flatten()

    correlation_task_2_2 = np.triu(correlation_task_2_second_half)
    correlation_task_3 = np.triu(correlation_task_3)
    correlation_task_2_2 = correlation_task_2_2.flatten()/flattened_all_clusters_task_1_first_half.shape[0]
    correlation_task_3 = correlation_task_3.flatten()/flattened_all_clusters_task_1_first_half.shape[0]

    mean_correlation_t1_t2 = np.corrcoef(correlation_task_1_2, correlation_task_2_1)
  
    plt.scatter(correlation_task_1_2,correlation_task_2_1, s =1, color = 'red')

    gradient, intercept, r_value, p_value, std_err = stats.linregress(correlation_task_2_2,correlation_task_3)

    mn=np.min(correlation_task_1_2)
    mx=np.max(correlation_task_1_2)
    x1=np.linspace(mn,mx,500)
    y1=gradient*x1+intercept
    plt.plot(x1,y1,'-r')
    plt.show()
    plt.title('HP')
    
    return mean_correlation_t1_t2, p_value

def svd_u_and_v_separately(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = True, z_score = False):
    
    if tasks_unchanged == True:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
    else:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = sv.flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, demean_all_tasks = demean_all_tasks)
    task_1_len = flattened_all_clusters_task_1_first_half.shape[1]/4
    session =experiment[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward = initiate_choice_t[-2] +250
    
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward)).argmin()
    
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = False)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)
    
    if tasks_unchanged == True:
        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = False)
        
        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = False)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half   
    t_u = np.transpose(u_t1_1)  
    t_v = np.transpose(vh_t1_1)  

    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_1 = np.transpose(u_t2_1)   
    t_v_t_2_1 = np.transpose(vh_t2_1)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  
    
    if tasks_unchanged == True:
        t_u_t_3_1 = np.transpose(u_t3_1)
        t_v_t_3_1 = np.transpose(vh_t3_1)  
        
        t_u_t_3_2 = np.transpose(u_t3_2)
        t_v_t_3_2 = np.transpose(vh_t3_2)  

# =============================================================================
#     Between Task 1 and Task 2
# =============================================================================
      
    # Variance task 2 First Half from Task 1 Second Half
    # Using U
    x_task_2_from_task_1 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half])
    # Using V
    y_task_2_from_task_1 = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half,t_v_t_1_2])
    
    var_x_task_2_from_task_1 = np.sum(x_task_2_from_task_1**2, axis = 1)
    cum_var_x_task_2_from_task_1 = np.cumsum(var_x_task_2_from_task_1)/np.sqrt(flattened_all_clusters_task_2_first_half.shape[0])
    cum_var_x_task_2_from_task_1 = cum_var_x_task_2_from_task_1/cum_var_x_task_2_from_task_1[-1]
    
    var_y_task_2_from_task_1 = np.sum(y_task_2_from_task_1**2, axis = 0)
    cum_var_y_task_2_from_task_1 = np.cumsum(var_y_task_2_from_task_1)/flattened_all_clusters_task_2_first_half.shape[0]
    cum_var_y_task_2_from_task_1 = cum_var_y_task_2_from_task_1/cum_var_y_task_2_from_task_1[-1]
    
    # Variance task 2 First Half from First Half
    # Using U
    x_1_task_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half])
    # Using V
    y_1_task_2 = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half,t_v_t_2_2])
    
    var_rows_1_task_2 = np.sum(x_1_task_2**2, axis = 1)
    cum_x_1_task_2 = np.cumsum(var_rows_1_task_2)/np.sqrt(flattened_all_clusters_task_2_first_half.shape[0])
    cum_x_1_task_2 = cum_x_1_task_2/cum_x_1_task_2[-1]
    
    var_rows_y_1_task_2 = np.sum(y_1_task_2**2, axis = 0)
    cum_y_1_task_2 = np.cumsum(var_rows_y_1_task_2)/flattened_all_clusters_task_2_first_half.shape[0]    
    cum_y_1_task_2 = cum_y_1_task_2/cum_y_1_task_2[-1]
    
    area_x_task_2_from_task_1 = []
    for i in cum_var_x_task_2_from_task_1:
        if i == 0:
            area_x_task_2_from_task_1.append(np.trapz([0,i]))
        else:
            area_x_task_2_from_task_1.append(np.trapz([0,i]))
    
    #Calculating area proportion within  Task 1 and 2
    area_x_1_task_2 = []
    for i in cum_x_1_task_2:
        if i == 0:
            area_x_1_task_2.append(np.trapz([0,i]))
        else:
            area_x_1_task_2.append(np.trapz([0,i]))
            

# =============================================================================
#     Between Task 2 and Task 3
# =============================================================================
    if tasks_unchanged == True:
        # Variance task 3 First Half from Task 2 Second Half
        # Using U
        x_task_3_from_task_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_3_first_half])
        # Using V
        y_task_3_from_task_2 = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half,t_v_t_2_2])
        
        var_x_task_3_from_task_2 = np.sum(x_task_3_from_task_2**2, axis = 1)
        cum_var_x_task_3_from_task_2 = np.cumsum(var_x_task_3_from_task_2)/np.sqrt(flattened_all_clusters_task_3_first_half.shape[0])
        cum_var_x_task_3_from_task_2 = cum_var_x_task_3_from_task_2/cum_var_x_task_3_from_task_2[-1]
        
        var_y_task_3_from_task_2 = np.sum(y_task_3_from_task_2**2, axis = 0)
        cum_var_y_task_3_from_task_2 = np.cumsum(var_y_task_3_from_task_2)/flattened_all_clusters_task_3_first_half.shape[0]
        cum_var_y_task_3_from_task_2 = cum_var_y_task_3_from_task_2/cum_var_y_task_3_from_task_2[-1]

        # Variance task 3 First Half from First Half
        # Using U
        x_1_task_3 = np.linalg.multi_dot([t_u_t_3_2, flattened_all_clusters_task_3_first_half])
        # Using V
        y_1_task_3 = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half,t_v_t_3_2])
        
        var_rows_1_task_3 = np.sum(x_1_task_3**2, axis = 1)
        cum_x_1_task_3 = np.cumsum(var_rows_1_task_3)/np.sqrt(flattened_all_clusters_task_3_first_half.shape[0])
        cum_x_1_task_3 = cum_x_1_task_3/cum_x_1_task_3[-1]
    
        var_rows_y_1_task_3 = np.sum(y_1_task_3**2, axis = 0)
        cum_y_1_task_3 = np.cumsum(var_rows_y_1_task_3)/flattened_all_clusters_task_3_first_half.shape[0]
        cum_y_1_task_3 = cum_y_1_task_3/cum_y_1_task_3[-1]
        
        #Calculating area proportion between Task 1 and 2
        area_x_task_3_from_task_2 = []
        for i in cum_var_x_task_3_from_task_2:
            if i == 0:
                area_x_task_3_from_task_2.append(np.trapz([0,i]))
            else:
                area_x_task_3_from_task_2.append(np.trapz([0,i]))    
        #Calculating area proportion within  Task 1 and 2
        area_x_1_task_3 = []
        for i in cum_x_1_task_3:
            if i == 0:
                area_x_1_task_3.append(np.trapz([0,i]))
            else:
                area_x_1_task_3.append(np.trapz([0,i]))    
                
        area_x_task_3_from_task_2 = np.asarray(area_x_task_3_from_task_2)
        area_x_1_task_3 = np.asarray(area_x_1_task_3)
        area_x_task_2_from_task_1 = np.asarray(area_x_task_2_from_task_1)
        area_x_1_task_2 = np.asarray(area_x_1_task_2)

    if tasks_unchanged == True:
        average_within_x = np.mean([cum_x_1_task_2, cum_x_1_task_3], axis = 0)
        average_within_y = np.mean([cum_y_1_task_2, cum_y_1_task_3],axis = 0)

        average_between_x = np.mean([cum_var_x_task_2_from_task_1, cum_var_x_task_3_from_task_2],axis = 0)
        average_between_y = np.mean([cum_var_y_task_2_from_task_1, cum_var_y_task_3_from_task_2],axis = 0)
    
    else:
        average_within_x = cum_x_1_task_2
        average_between_x = cum_var_x_task_2_from_task_1
        average_within_y = cum_y_1_task_2
        average_between_y = cum_var_y_task_2_from_task_1
        
    if HP == True: 
        plt.figure(9)
        plt.plot(average_between_x, label = 'Between Task HP Reduced ', linestyle = '--', color = 'green')
        plt.plot(average_within_x, label = 'Within Task HP Reduced', color='green')
        plt.figure(10)
        plt.plot(average_between_y, label = 'Between Task HP Reduced ', linestyle = '--', color = 'green')
        plt.plot(average_within_y, label = 'Within Task HP Reduced', color='green')
     

    if HP == False:
        plt.figure(9)
        plt.plot(average_between_x, label = 'Between Task PFC Reduced ', linestyle = '--', color = 'black')
        plt.plot(average_within_x, label = 'Within Task PFC Reduced', color='black')
        plt.figure(10)
        plt.plot(average_between_y, label = 'Between Task PFC Reduced ', linestyle = '--', color = 'black')
        plt.plot(average_within_y, label = 'Within Task PFC Reduced', color='black')
     

    plt.figure(9)
    plt.legend()
    plt.figure(10)
    plt.legend()
    plt.title('Left only')
    
    if HP == True:
        color = 'green'
        label = 'HP'
    else:
        color = 'black'
        label = 'PFC'
        
        

    if tasks_unchanged == True:
        fig = plt.figure(num = 11, figsize=(10,8))
        fig.add_subplot(10,3,1)
        plt.plot(t_v[:,0], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.title('Task 1')
        plt.ylabel('Eig 1')
        
        fig.add_subplot(10,3,2)
        plt.plot(t_v_t_2_1[:,0], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
        plt.title('Task 2')
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,3)
        plt.plot(t_v_t_3_1[:,0], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
        plt.title('Task 3')
        plt.legend()
    
        plt.axvline(x=task_1_len, color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, color = 'black', alpha = 0.5)
        
        fig.add_subplot(10,3,4)
        plt.plot(t_v[:,1], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
        plt.ylabel('Eig 2')
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,5)
        plt.plot(t_v_t_2_1[:,1], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,6)
        plt.plot(t_v_t_3_1[:,1], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
      
        fig.add_subplot(10,3,7)
        plt.plot(t_v[:,2], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
        plt.ylabel('Eig 3')
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,8)
        plt.plot(t_v_t_2_1[:,2], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)

        fig.add_subplot(10,3,9)
        plt.plot(t_v_t_3_1[:,2], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,10)
        plt.plot(t_v[:,3], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 4')

        fig.add_subplot(10,3,11)
        plt.plot(t_v_t_2_1[:,3], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,12)
        plt.plot(t_v_t_3_1[:,3], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
    
        fig.add_subplot(10,3,13)
        plt.plot(t_v[:,4], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 5')

    
        fig.add_subplot(10,3,14)
        plt.plot(t_v_t_2_1[:,4], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,15)
        plt.plot(t_v_t_3_1[:,4], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        
        fig.add_subplot(10,3,16)
        plt.plot(t_v[:,5], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 6')

    
        fig.add_subplot(10,3,17)
        plt.plot(t_v_t_2_1[:,5], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,18)
        plt.plot(t_v_t_3_1[:,5], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,19)
        plt.plot(t_v[:,6], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 7')

    
        fig.add_subplot(10,3,20)
        plt.plot(t_v_t_2_1[:,6], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,21)
        plt.plot(t_v_t_3_1[:,6], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        
        fig.add_subplot(10,3,22)
        plt.plot(t_v[:,7], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 8')

    
        fig.add_subplot(10,3,23)
        plt.plot(t_v_t_2_1[:,7], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,24)
        plt.plot(t_v_t_3_1[:,7], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
    
        fig.add_subplot(10,3,25)
        plt.plot(t_v[:,8], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 9')

    
        fig.add_subplot(10,3,26)
        plt.plot(t_v_t_2_1[:,8], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,27)
        plt.plot(t_v_t_3_1[:,8], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)



        fig.add_subplot(10,3,28)
        plt.plot(t_v[:,9], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        plt.ylabel('Eig 9')

    
        fig.add_subplot(10,3,29)
        plt.plot(t_v_t_2_1[:,9], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
    
        fig.add_subplot(10,3,30)
        plt.plot(t_v_t_3_1[:,9], color = color, label = label)
        plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
    
        plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
        plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
        plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
        
        first = t_u[0,:]
        return first, average_between_x,average_between_y,average_within_x, average_within_y



#ind_loading_hp_1_eig = [38,  41,  86,  90, 206, 259, 261]
#flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
#flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
#flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_HP, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
#task_1_len = flattened_all_clusters_task_1_first_half.shape[1]/4
#session =experiment_aligned_HP[0]
#t_out = session.t_out
#initiate_choice_t = session.target_times 
#reward = initiate_choice_t[-2] +250
#
#ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
#ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
#ind_reward = (np.abs(t_out-reward)).argmin()
#   
#fig = plt.figure(num = 11, figsize=(10,8))
#fig.add_subplot(2,3,1)
#plt.plot(flattened_all_clusters_task_1_first_half[38])
#plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
#
#plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
#plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
#plt.ylabel('Cell 1')
#
#
#fig.add_subplot(2,3,2)
#plt.plot(flattened_all_clusters_task_1_first_half[41])
#plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
#
#plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
#plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
#plt.ylabel('Cell 2')
#
#fig.add_subplot(2,3,3)
#plt.plot(flattened_all_clusters_task_1_first_half[86])
#plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
#
#plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
#plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
#plt.ylabel('Cell 3')
#
#fig.add_subplot(2,3,4)
#plt.plot(flattened_all_clusters_task_1_first_half[206])
#plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
#
#plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
#plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
#plt.ylabel('Cell 4')
#
#fig.add_subplot(2,3,5)
#plt.plot(flattened_all_clusters_task_1_first_half[90])
#plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
#
#plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
#plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
#plt.ylabel('Cell 5')
#
#fig.add_subplot(2,3,6)
#plt.plot(flattened_all_clusters_task_1_first_half[259])
#plt.xticks([ind_init, ind_choice, ind_reward, ind_init+ task_1_len, ind_choice+task_1_len, ind_reward+task_1_len,  ind_init+(2*task_1_len), ind_choice+(2*task_1_len), ind_reward++(2*task_1_len), ind_init+(3*task_1_len), ind_choice+(3*task_1_len), ind_reward++(3*task_1_len)], ['I', 'A', 'R', 'I', 'A', 'N', 'I', 'B', 'R', 'I', 'B', 'N'])
#
#plt.axvline(x=task_1_len, label = 'A Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*2, label = 'A  No Reward', color = 'black',alpha = 0.5)
#plt.axvline(x=task_1_len*3, label = 'B Reward', color = 'black', alpha = 0.5)
#plt.axvline(x=task_1_len*4, label = 'B  No Reward', color = 'black', alpha = 0.5)
#plt.ylabel('Cell 6')

def use_hp_vs_pfc(HP_experiment,PFC_experiment, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False):
    
# =============================================================================
# HP
# =============================================================================
    flattened_all_clusters_task_1_first_half_HP, flattened_all_clusters_task_1_second_half_HP,\
    flattened_all_clusters_task_2_first_half_HP, flattened_all_clusters_task_2_second_half_HP,\
    flattened_all_clusters_task_3_first_half_HP,flattened_all_clusters_task_3_second_half_HP = demean_data(HP_experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
    
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1_hp, s_t1_1_hp, vh_t1_1_hp = np.linalg.svd(flattened_all_clusters_task_1_first_half_HP, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2_hp, s_t1_2_hp, vh_t1_2_hp = np.linalg.svd(flattened_all_clusters_task_1_second_half_HP, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1_hp, s_t2_1_hp, vh_t2_1_hp = np.linalg.svd(flattened_all_clusters_task_2_first_half_HP, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2_hp, s_t2_2_hp, vh_t2_2_hp = np.linalg.svd(flattened_all_clusters_task_2_second_half_HP, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1_hp, s_t3_1_hp, vh_t3_1_hp = np.linalg.svd(flattened_all_clusters_task_3_first_half_HP, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2_hp, s_t3_2_hp, vh_t3_2_hp = np.linalg.svd(flattened_all_clusters_task_3_second_half_HP, full_matrices = True)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_v_t_1_1_hp = np.transpose(vh_t1_1_hp)  

    t_v_t_1_2_hp = np.transpose(vh_t1_2_hp)  

    t_v_t_2_1_hp = np.transpose(vh_t2_1_hp)  

    t_v_t_2_2_hp = np.transpose(vh_t2_2_hp)  

    t_v_t_3_1_hp = np.transpose(vh_t3_1_hp)  
    
    t_v_t_3_2_hp = np.transpose(vh_t3_2_hp)  

# =============================================================================
# PFC
# =============================================================================
    
    flattened_all_clusters_task_1_first_half_PFC, flattened_all_clusters_task_1_second_half_PFC,\
    flattened_all_clusters_task_2_first_half_PFC, flattened_all_clusters_task_2_second_half_PFC,\
    flattened_all_clusters_task_3_first_half_PFC,flattened_all_clusters_task_3_second_half_PFC = demean_data(PFC_experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
    
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1_pfc, s_t1_1_pfc, vh_t1_1_pfc = np.linalg.svd(flattened_all_clusters_task_1_first_half_PFC, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2_pfc, s_t1_2_pfc, vh_t1_2_pfc = np.linalg.svd(flattened_all_clusters_task_1_second_half_PFC, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1_pfc, s_t2_1_pfc, vh_t2_1_pfc = np.linalg.svd(flattened_all_clusters_task_2_first_half_PFC, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2_pfc, s_t2_2_pfc, vh_t2_2_pfc = np.linalg.svd(flattened_all_clusters_task_2_second_half_PFC, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1_pfc, s_t3_1_pfc, vh_t3_1_pfc = np.linalg.svd(flattened_all_clusters_task_3_first_half_PFC, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2_pfc, s_t3_2_pfc, vh_t3_2_pfc = np.linalg.svd(flattened_all_clusters_task_3_second_half_PFC, full_matrices = True)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_v_t_1_1_pfc = np.transpose(vh_t1_1_hp)  

    t_v_t_1_2_pfc = np.transpose(vh_t1_2_pfc)  

    t_v_t_2_1_pfc = np.transpose(vh_t2_1_pfc)  

    t_v_t_2_2_pfc = np.transpose(vh_t2_2_pfc)  

    t_v_t_3_1_pfc = np.transpose(vh_t3_1_pfc)  
    
    t_v_t_3_2_pfc = np.transpose(vh_t3_2_pfc)  

# =============================================================================
# Explain pfc 
# =============================================================================
    
    task_1_first_half_pfc_from_hp = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half_PFC,t_v_t_1_1_hp])
    task_1_first_half_pfc_from_pfc = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half_PFC,t_v_t_1_2_pfc])

    task_1_first_half_pfc_from_hp_var = np.sum(task_1_first_half_pfc_from_hp**2, axis = 0)
    cum_task_1_pfc_from_hp = np.cumsum(task_1_first_half_pfc_from_hp_var)/flattened_all_clusters_task_1_first_half_PFC.shape[0]
    #cum_task_1_pfc_from_hp = (cum_task_1_pfc_from_hp-min(cum_task_1_pfc_from_hp))/(max(cum_task_1_pfc_from_hp)-min(cum_task_1_pfc_from_hp))

    task_1_first_half_pfc_from_pfc_var = np.sum(task_1_first_half_pfc_from_pfc**2, axis = 0)
    cum_task_1_pfc_from_pfc = np.cumsum(task_1_first_half_pfc_from_pfc_var)/flattened_all_clusters_task_1_first_half_PFC.shape[0]
    #cum_task_1_pfc_from_pfc = (cum_task_1_pfc_from_pfc-min(cum_task_1_pfc_from_pfc))/(max(cum_task_1_pfc_from_pfc)-min(cum_task_1_pfc_from_pfc))

    task_2_first_half_pfc_from_hp = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_PFC,t_v_t_2_1_hp])
    task_2_first_half_pfc_from_pfc = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_PFC,t_v_t_2_2_pfc])

    task_2_first_half_pfc_from_hp_var = np.sum(task_2_first_half_pfc_from_hp**2, axis = 0)
    cum_task_2_pfc_from_hp = np.cumsum(task_2_first_half_pfc_from_hp_var)/flattened_all_clusters_task_2_first_half_PFC.shape[0]
    #cum_task_2_pfc_from_hp = (cum_task_2_pfc_from_hp-min(cum_task_2_pfc_from_hp))/(max(cum_task_2_pfc_from_hp)-min(cum_task_2_pfc_from_hp))

    task_2_first_half_pfc_from_pfc_var = np.sum(task_2_first_half_pfc_from_pfc**2, axis = 0)
    cum_task_2_pfc_from_pfc = np.cumsum(task_2_first_half_pfc_from_pfc_var)/flattened_all_clusters_task_2_first_half_PFC.shape[0]
    #cum_task_2_pfc_from_pfc = (cum_task_2_pfc_from_pfc-min(cum_task_2_pfc_from_pfc))/(max(cum_task_2_pfc_from_pfc)-min(cum_task_2_pfc_from_pfc))
    
    task_3_first_half_pfc_from_hp = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half_PFC,t_v_t_3_1_hp])
    task_3_first_half_pfc_from_pfc = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half_PFC,t_v_t_3_2_pfc])

    task_3_first_half_pfc_from_hp_var = np.sum(task_3_first_half_pfc_from_hp**2, axis = 0)
    cum_task_3_pfc_from_hp = np.cumsum(task_3_first_half_pfc_from_hp_var)/flattened_all_clusters_task_3_first_half_PFC.shape[0]
    #cum_task_3_pfc_from_hp = (cum_task_3_pfc_from_hp-min(cum_task_3_pfc_from_hp))/(max(cum_task_3_pfc_from_hp)-min(cum_task_3_pfc_from_hp))

    task_3_first_half_pfc_from_pfc_var = np.sum(task_3_first_half_pfc_from_pfc**2, axis = 0)
    cum_task_3_pfc_from_pfc = np.cumsum(task_3_first_half_pfc_from_pfc_var)/flattened_all_clusters_task_3_first_half_PFC.shape[0]
    #cum_task_3_pfc_from_pfc = (cum_task_3_pfc_from_pfc-min(cum_task_3_pfc_from_pfc))/(max(cum_task_3_pfc_from_pfc)-min(cum_task_3_pfc_from_pfc))


# =============================================================================
# Explain hp 
# =============================================================================
    task_1_first_half_hp_from_pfc = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half_HP,t_v_t_1_1_pfc])
    task_1_first_half_hp_from_hp = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half_HP,t_v_t_1_2_hp])

    task_1_first_half_hp_from_pfc_var = np.sum(task_1_first_half_hp_from_pfc**2, axis = 0)
    cum_task_1_hp_from_pfc = np.cumsum(task_1_first_half_hp_from_pfc_var)/flattened_all_clusters_task_1_first_half_HP.shape[0]
   # cum_task_1_hp_from_pfc = (cum_task_1_hp_from_pfc-min(cum_task_1_hp_from_pfc))/(max(cum_task_1_hp_from_pfc)-min(cum_task_1_hp_from_pfc))

    task_1_first_half_hp_from_hp_var = np.sum(task_1_first_half_hp_from_hp**2, axis = 0)
    cum_task_1_hp_from_hp = np.cumsum(task_1_first_half_hp_from_hp_var)/flattened_all_clusters_task_1_first_half_HP.shape[0]
    #cum_task_1_hp_from_hp = (cum_task_1_hp_from_hp-min(cum_task_1_hp_from_hp))/(max(cum_task_1_hp_from_hp)-min(cum_task_1_hp_from_hp))
    
    task_2_first_half_hp_from_pfc = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_HP,t_v_t_2_1_pfc])
    task_2_first_half_hp_from_hp = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_HP,t_v_t_2_2_hp])

    task_2_first_half_hp_from_pfc_var = np.sum(task_2_first_half_hp_from_pfc**2, axis = 0)
    cum_task_2_hp_from_pfc = np.cumsum(task_2_first_half_hp_from_pfc_var)/flattened_all_clusters_task_2_first_half_HP.shape[0]
    #cum_task_2_hp_from_pfc = (cum_task_2_hp_from_pfc-min(cum_task_2_hp_from_pfc))/(max(cum_task_2_hp_from_pfc)-min(cum_task_2_hp_from_pfc))

    task_2_first_half_hp_from_hp_var = np.sum(task_2_first_half_hp_from_hp**2, axis = 0)
    cum_task_2_hp_from_hp = np.cumsum(task_2_first_half_hp_from_hp_var)/flattened_all_clusters_task_2_first_half_HP.shape[0]
    #cum_task_2_hp_from_hp = (cum_task_2_hp_from_hp-min(cum_task_2_hp_from_hp))/(max(cum_task_2_hp_from_hp)-min(cum_task_2_hp_from_hp))

    task_3_first_half_hp_from_pfc = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half_HP,t_v_t_3_1_pfc])
    task_3_first_half_hp_from_hp = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half_HP,t_v_t_3_2_hp])

    task_3_first_half_hp_from_pfc_var = np.sum(task_3_first_half_hp_from_pfc**2, axis = 0)
    cum_task_3_hp_from_pfc = np.cumsum(task_3_first_half_hp_from_pfc_var)/flattened_all_clusters_task_3_first_half_HP.shape[0]
    #cum_task_3_hp_from_pfc = (cum_task_3_hp_from_pfc-min(cum_task_3_hp_from_pfc))/(max(cum_task_3_hp_from_pfc)-min(cum_task_3_hp_from_pfc))

    task_3_first_half_hp_from_hp_var = np.sum(task_3_first_half_hp_from_hp**2, axis = 0)
    cum_task_3_hp_from_hp = np.cumsum(task_3_first_half_hp_from_hp_var)/flattened_all_clusters_task_3_first_half_HP.shape[0]
    #cum_task_3_hp_from_hp = (cum_task_3_hp_from_hp-min(cum_task_3_hp_from_hp))/(max(cum_task_3_hp_from_hp)-min(cum_task_3_hp_from_hp))
    
    mean_pfc_from_hp = np.mean([cum_task_1_pfc_from_hp,cum_task_2_pfc_from_hp, cum_task_3_pfc_from_hp],axis = 0)
    mean_pfc_from_hp = mean_pfc_from_hp/mean_pfc_from_hp[-1]
    mean_pfc_from_pfc = np.mean([cum_task_1_pfc_from_pfc,cum_task_2_pfc_from_pfc, cum_task_3_pfc_from_pfc], axis = 0)
    mean_pfc_from_pfc = mean_pfc_from_pfc/mean_pfc_from_pfc[-1]

    mean_hp_from_pfc = np.mean([cum_task_1_hp_from_pfc,cum_task_2_hp_from_pfc, cum_task_3_hp_from_pfc], axis = 0)
    mean_hp_from_pfc = mean_hp_from_pfc/mean_hp_from_pfc[-1]

    mean_hp_from_hp = np.mean([cum_task_1_hp_from_hp,cum_task_2_hp_from_hp, cum_task_3_hp_from_hp], axis = 0)
    mean_hp_from_hp = mean_hp_from_hp/mean_hp_from_hp[-1]
    
    
    plt.plot(mean_pfc_from_hp, color = 'blue', label = 'PFC from HP Within Task')
    plt.plot(mean_pfc_from_pfc, linestyle = '--',color = 'blue', label = 'PFC from PFC Within Task')
    plt.plot(mean_hp_from_pfc,color = 'red', label = 'HP from PFC Within Task ')
    plt.plot(mean_hp_from_hp, linestyle = '--',color = 'red', label = 'HP from HP Within Task')
    plt.legend()