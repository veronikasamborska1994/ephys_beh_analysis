#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:22:57 2019

@author: veronikasamborska
"""
# =============================================================================
# Tim's rewrittten from Matlab code for spectral clustering 
# =============================================================================

import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/modelling')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/SVDs')

import numpy as np
import SVDs as sv
import matplotlib.pyplot as plt
import svds_u_only as svdu



def spectral_clustering(experiment, demean = True): 
    if demean == True:
        # Check demeaning 
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = svdu.demean_data(experiment,tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)
    
        mean_task_1  = np.mean(flattened_all_clusters_task_1_second_half,axis = 1)
        mean_task_2  = np.mean(flattened_all_clusters_task_2_first_half,axis = 1)
        mean_task_3  = np.mean(flattened_all_clusters_task_2_second_half,axis = 1)
          
        corr_1_2 = np.corrcoef(mean_task_1,mean_task_2)        
        corr_2_3 = np.corrcoef(mean_task_2,mean_task_3)        
        corr_1_3 = np.corrcoef(mean_task_1,mean_task_3)    

    # Non-demeaned data
    elif demean == False:

        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)

    task_1 = flattened_all_clusters_task_1_first_half+flattened_all_clusters_task_1_second_half/2
    task_2 = flattened_all_clusters_task_2_first_half+flattened_all_clusters_task_2_second_half/2
    task_3 = flattened_all_clusters_task_3_first_half+flattened_all_clusters_task_3_second_half/2
    
    all_data = np.concatenate([task_1, task_2,task_3], axis = 1)
               
    corr_m = np.corrcoef(all_data)
    v,d = np.linalg.eig(corr_m)
    
    A = corr_m+1
    session = experiment[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
            
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()
    
    A =(A + np.transpose(A))/2
    
    G = np.power(A,1.5)
    Q = -G
    Q = np.triu(Q,1) + np.tril(Q,-1)
    Q = Q - np.diag(sum(Q))
     
    t = 1/np.sqrt(sum(G))
    Q = np.dot(np.diag(t),Q)
    Q = np.dot(Q,np.diag(t))
    
    v,d = np.linalg.eig(Q)
     
    v2 = d[:,1]
     
    v2scale = t*v2
     
    ind = np.argsort(v2scale)
    sorted_data = all_data[ind]
    
    plt.imshow(sorted_data)
    
    #Plot chunks of 10 neurons
    
    trial_length = 64
    n_neurons = 10
    n = 39
    
    HP_range = np.arange(40)
    HP_range = HP_range[1:]
    HP_cut = np.arange(n)
    
    PFC_range = np.arange(57)
    PFC_range = PFC_range[1:]
    PFC_cut = np.arange(n)
    
    fig = plt.figure(figsize=(30,60))
    
    for end, chunks in zip(HP_range,HP_cut):
        
    
        mean_task_1_a_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons,:trial_length],axis = 0)
        mean_task_1_a_nr =  np.mean(sorted_data[chunks*n_neurons:end*n_neurons,trial_length:trial_length*2],axis = 0)
        
        mean_task_1_b_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons,trial_length*2:trial_length*3],axis = 0)
        mean_task_1_b_nr =  np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*3:trial_length*4],axis = 0)
        
        mean_task_2_a_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*4: trial_length*5],axis = 0)
        mean_task_2_a_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*5: trial_length*6],axis = 0)
        
        mean_task_2_b_r= np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*6: trial_length*7],axis = 0)
        mean_task_2_b_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*7: trial_length*8],axis = 0)
        
        mean_task_3_a_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*8: trial_length*9],axis = 0)
        mean_task_3_a_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*9: trial_length*10],axis = 0)
        
        mean_task_3_b_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*10: trial_length*11],axis = 0)
        mean_task_3_b_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*11: trial_length*12],axis = 0)
    
        fig.add_subplot(5,8,end)
        plt.plot(mean_task_1_a_r,  color = 'darkblue')
        plt.plot(mean_task_1_a_nr,linestyle ='--', color = 'darkblue')
        
        plt.plot(mean_task_1_b_r, color = 'darkred')
        plt.plot(mean_task_1_b_nr,linestyle =  '--', color = 'darkred')
        
        plt.plot(mean_task_2_a_r, color = 'lightblue')
        plt.plot(mean_task_2_a_nr, linestyle = '--', color = 'lightblue')
        
        plt.plot(mean_task_2_b_r, color = 'pink')
        plt.plot(mean_task_2_b_nr, linestyle = '--', color = 'pink')
        
        plt.plot(mean_task_3_a_r, color = 'royalblue')
        plt.plot(mean_task_3_a_nr,linestyle = '--', color = 'royalblue')
        
        plt.plot(mean_task_3_b_r, color = 'coral')
        plt.plot(mean_task_3_b_nr,linestyle = '--', color = 'coral')
        plt.xticks([ind_init,ind_choice,ind_reward], ('I', 'C', 'O' ))  
    
        
        if end == n:     
            mean_task_1_a_r = np.mean(sorted_data[end*n_neurons:,:trial_length],axis = 0)
            mean_task_1_a_nr =  np.mean(sorted_data[end*n_neurons:,trial_length:trial_length*2],axis = 0)
            
            mean_task_1_b_r = np.mean(sorted_data[end*n_neurons:,trial_length*2:trial_length*3],axis = 0)
            mean_task_1_b_nr =  np.mean(sorted_data[end*n_neurons:, trial_length*3:trial_length*4],axis = 0)
            
            mean_task_2_a_r = np.mean(sorted_data[end*n_neurons:, trial_length*4: trial_length*5],axis = 0)
            mean_task_2_a_nr = np.mean(sorted_data[end*n_neurons:, trial_length*5: trial_length*6],axis = 0)
            
            mean_task_2_b_r= np.mean(sorted_data[end*n_neurons:, trial_length*6: trial_length*7],axis = 0)
            mean_task_2_b_nr = np.mean(sorted_data[end*n_neurons:, trial_length*7: trial_length*8],axis = 0)
            
            mean_task_3_a_r = np.mean(sorted_data[end*n_neurons:, trial_length*8: trial_length*9],axis = 0)
            mean_task_3_a_nr = np.mean(sorted_data[end*n_neurons:, trial_length*9: trial_length*10],axis = 0)
            
            mean_task_3_b_r = np.mean(sorted_data[end*n_neurons:, trial_length*10: trial_length*11],axis = 0)
            mean_task_3_b_nr = np.mean(sorted_data[end*n_neurons:, trial_length*11: trial_length*12],axis = 0)
        
            fig.add_subplot(5,8,n+1)
            plt.plot(mean_task_1_a_r, label = 'A T1 R', color = 'darkblue')
            plt.plot(mean_task_1_a_nr,label = 'A T1 NR',linestyle ='--', color = 'darkblue')
            
            plt.plot(mean_task_1_b_r, label = 'B T1 R', color = 'darkred')
            plt.plot(mean_task_1_b_nr, label = 'B T1 NR',linestyle =  '--', color = 'darkred')
            
            plt.plot(mean_task_2_a_r, label = 'A T2 R', color = 'lightblue')
            plt.plot(mean_task_2_a_nr, label = 'A T2 NR', linestyle = '--', color = 'lightblue')
            
            plt.plot(mean_task_2_b_r,label = 'B T2 R', color = 'pink')
            plt.plot(mean_task_2_b_nr, label = 'B T2 NR', linestyle = '--', color = 'pink')
            
            plt.plot(mean_task_3_a_r, label = 'A T3 R', color = 'royalblue')
            plt.plot(mean_task_3_a_nr, label = 'A T3 NR',linestyle = '--', color = 'royalblue')
            
            plt.plot(mean_task_3_b_r, label = 'B T3 R', color = 'coral')
            plt.plot(mean_task_3_b_nr, label = 'B T3 NR',linestyle = '--', color = 'coral')            
            plt.xticks([ind_init,ind_choice,ind_reward], ('I', 'C', 'O' ))  
    
    
    #plt.legend()