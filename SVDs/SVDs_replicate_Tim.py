#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:09:46 2019

@author: veronikasamborska
"""

import SVDs as sv 
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import svds_u_only as svdu

def replicate_Tim_interleave_trials(session):      
    spikes, aligned_rates_task_1_first_half_A_reward, aligned_rates_task_1_first_half_A_Nreward,\
    aligned_rates_task_1_second_half_A_reward,aligned_rates_task_1_second_half_A_Nreward,\
    aligned_rates_task_1_first_half_B_reward,aligned_rates_task_1_first_half_B_Nreward,\
    aligned_rates_task_1_second_half_B_reward,aligned_rates_task_1_second_half_B_Nreward,\
    aligned_rates_task_2_first_half_A_reward,aligned_rates_task_2_first_half_A_Nreward,\
    aligned_rates_task_2_second_half_A_reward,aligned_rates_task_2_second_half_A_Nreward,\
    aligned_rates_task_2_first_half_B_reward,aligned_rates_task_2_first_half_B_Nreward,\
    aligned_rates_task_2_second_half_B_reward,aligned_rates_task_2_second_half_B_Nreward,\
    aligned_rates_task_3_first_half_A_reward,aligned_rates_task_3_first_half_A_Nreward,\
    aligned_rates_task_3_second_half_A_reward,aligned_rates_task_3_second_half_A_Nreward,\
    aligned_rates_task_3_first_half_B_reward,aligned_rates_task_3_first_half_B_Nreward,\
    aligned_rates_task_3_second_half_B_reward,aligned_rates_task_3_second_half_B_Nreward = sv.extract_session_predictors_rates(session)
    
    aligned_rates_task_1_A_reward = np.concatenate((aligned_rates_task_1_first_half_A_reward,aligned_rates_task_1_second_half_A_reward), axis = 0)
    aligned_rates_task_1_A_Nreward = np.concatenate((aligned_rates_task_1_first_half_A_Nreward,aligned_rates_task_1_second_half_A_Nreward), axis = 0)

    aligned_rates_task_1_B_reward = np.concatenate((aligned_rates_task_1_first_half_B_reward,aligned_rates_task_1_second_half_B_reward), axis = 0)
    aligned_rates_task_1_B_Nreward = np.concatenate((aligned_rates_task_1_first_half_B_Nreward,aligned_rates_task_1_second_half_B_Nreward), axis = 0)

    aligned_rates_task_2_A_reward = np.concatenate((aligned_rates_task_2_first_half_A_reward,aligned_rates_task_2_second_half_A_reward), axis = 0)
    aligned_rates_task_2_A_Nreward = np.concatenate((aligned_rates_task_2_first_half_A_Nreward,aligned_rates_task_2_second_half_A_Nreward), axis = 0)

    aligned_rates_task_2_B_reward = np.concatenate((aligned_rates_task_2_first_half_B_reward,aligned_rates_task_2_second_half_B_reward), axis = 0)
    aligned_rates_task_2_B_Nreward = np.concatenate((aligned_rates_task_2_first_half_B_Nreward,aligned_rates_task_2_second_half_B_Nreward), axis = 0)
    
    aligned_rates_task_3_A_reward = np.concatenate((aligned_rates_task_3_first_half_A_reward,aligned_rates_task_3_second_half_A_reward), axis = 0)
    aligned_rates_task_3_A_Nreward = np.concatenate((aligned_rates_task_3_first_half_A_Nreward,aligned_rates_task_3_second_half_A_Nreward), axis = 0)

    aligned_rates_task_3_B_reward = np.concatenate((aligned_rates_task_3_first_half_B_reward,aligned_rates_task_3_second_half_B_reward), axis = 0)
    aligned_rates_task_3_B_Nreward = np.concatenate((aligned_rates_task_3_first_half_B_Nreward,aligned_rates_task_3_second_half_B_Nreward), axis = 0)
    
    aligned_rates_task_1_A_reward_interleaved_1 = aligned_rates_task_1_A_reward[::2]
    aligned_rates_task_1_A_reward_interleaved_2 = aligned_rates_task_1_A_reward[1::2]
    aligned_rates_task_1_B_reward_interleaved_1 = aligned_rates_task_1_B_reward[::2]
    aligned_rates_task_1_B_reward_interleaved_2 = aligned_rates_task_1_B_reward[1::2]
    
    aligned_rates_task_1_A_Nreward_interleaved_1 = aligned_rates_task_1_A_Nreward[::2]
    aligned_rates_task_1_A_Nreward_interleaved_2 = aligned_rates_task_1_A_Nreward[1::2]
    aligned_rates_task_1_B_Nreward_interleaved_1 = aligned_rates_task_1_B_Nreward[::2]
    aligned_rates_task_1_B_Nreward_interleaved_2 = aligned_rates_task_1_B_Nreward[1::2]
    
    aligned_rates_task_2_A_reward_interleaved_1 = aligned_rates_task_2_A_reward[::2]
    aligned_rates_task_2_A_reward_interleaved_2 = aligned_rates_task_2_A_reward[1::2]
    aligned_rates_task_2_B_reward_interleaved_1 = aligned_rates_task_2_B_reward[::2]
    aligned_rates_task_2_B_reward_interleaved_2 = aligned_rates_task_2_B_reward[1::2]
    
    aligned_rates_task_2_A_Nreward_interleaved_1 = aligned_rates_task_2_A_Nreward[::2]
    aligned_rates_task_2_A_Nreward_interleaved_2 = aligned_rates_task_2_A_Nreward[1::2]
    aligned_rates_task_2_B_Nreward_interleaved_1 = aligned_rates_task_2_B_Nreward[::2]
    aligned_rates_task_2_B_Nreward_interleaved_2 = aligned_rates_task_2_B_Nreward[1::2]

    aligned_rates_task_3_A_reward_interleaved_1 = aligned_rates_task_3_A_reward[::2]
    aligned_rates_task_3_A_reward_interleaved_2 = aligned_rates_task_3_A_reward[1::2]
    aligned_rates_task_3_B_reward_interleaved_1 = aligned_rates_task_3_B_reward[::2]
    aligned_rates_task_3_B_reward_interleaved_2 = aligned_rates_task_3_B_reward[1::2]
    
    aligned_rates_task_3_A_Nreward_interleaved_1 = aligned_rates_task_3_A_Nreward[::2]
    aligned_rates_task_3_A_Nreward_interleaved_2 = aligned_rates_task_3_A_Nreward[1::2]
    aligned_rates_task_3_B_Nreward_interleaved_1 = aligned_rates_task_3_B_Nreward[::2]
    aligned_rates_task_3_B_Nreward_interleaved_2 = aligned_rates_task_3_B_Nreward[1::2]
    
    return  aligned_rates_task_1_A_reward_interleaved_1, aligned_rates_task_1_A_reward_interleaved_2,\
            aligned_rates_task_1_B_reward_interleaved_1, aligned_rates_task_1_B_reward_interleaved_2,\
            aligned_rates_task_1_A_Nreward_interleaved_1, aligned_rates_task_1_A_Nreward_interleaved_2,\
            aligned_rates_task_1_B_Nreward_interleaved_1, aligned_rates_task_1_B_Nreward_interleaved_2,\
            aligned_rates_task_2_A_reward_interleaved_1, aligned_rates_task_2_A_reward_interleaved_2,\
            aligned_rates_task_2_B_reward_interleaved_1, aligned_rates_task_2_B_reward_interleaved_2,\
            aligned_rates_task_2_A_Nreward_interleaved_1, aligned_rates_task_2_A_Nreward_interleaved_2,\
            aligned_rates_task_2_B_Nreward_interleaved_1, aligned_rates_task_2_B_Nreward_interleaved_2,\
            aligned_rates_task_3_A_reward_interleaved_1, aligned_rates_task_3_A_reward_interleaved_2,\
            aligned_rates_task_3_B_reward_interleaved_1, aligned_rates_task_3_B_reward_interleaved_2,\
            aligned_rates_task_3_A_Nreward_interleaved_1, aligned_rates_task_3_A_Nreward_interleaved_2,\
            aligned_rates_task_3_B_Nreward_interleaved_1, aligned_rates_task_3_B_Nreward_interleaved_2 
    

def replicate_Tim_trial_selection(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, split_by_reward = False):       
    all_clusters_task_1_first_half = []
    all_clusters_task_1_second_half = []
    all_clusters_task_2_first_half = []
    all_clusters_task_2_second_half = []
    all_clusters_task_3_first_half = []
    all_clusters_task_3_second_half = []
    
  
    for s,session in enumerate(experiment):
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        if s == 15 or s ==31:
            print(session.file_name)
        elif s != 15 and s !=31:
            aligned_rates_task_1_A_reward_interleaved_1, aligned_rates_task_1_A_reward_interleaved_2,\
            aligned_rates_task_1_B_reward_interleaved_1, aligned_rates_task_1_B_reward_interleaved_2,\
            aligned_rates_task_1_A_Nreward_interleaved_1, aligned_rates_task_1_A_Nreward_interleaved_2,\
            aligned_rates_task_1_B_Nreward_interleaved_1, aligned_rates_task_1_B_Nreward_interleaved_2,\
            aligned_rates_task_2_A_reward_interleaved_1, aligned_rates_task_2_A_reward_interleaved_2,\
            aligned_rates_task_2_B_reward_interleaved_1, aligned_rates_task_2_B_reward_interleaved_2,\
            aligned_rates_task_2_A_Nreward_interleaved_1, aligned_rates_task_2_A_Nreward_interleaved_2,\
            aligned_rates_task_2_B_Nreward_interleaved_1, aligned_rates_task_2_B_Nreward_interleaved_2,\
            aligned_rates_task_3_A_reward_interleaved_1, aligned_rates_task_3_A_reward_interleaved_2,\
            aligned_rates_task_3_B_reward_interleaved_1, aligned_rates_task_3_B_reward_interleaved_2,\
            aligned_rates_task_3_A_Nreward_interleaved_1, aligned_rates_task_3_A_Nreward_interleaved_2,\
            aligned_rates_task_3_B_Nreward_interleaved_1, aligned_rates_task_3_B_Nreward_interleaved_2  = replicate_Tim_interleave_trials(session)
            
            unique_neurons  = np.unique(spikes[0])   
            for i in range(len(unique_neurons)):
                
                mean_firing_rate_task_1_first_half_A_reward  = np.mean(aligned_rates_task_1_A_reward_interleaved_1[:,i,:],0)
                mean_firing_rate_task_1_first_half_A_Nreward  = np.mean(aligned_rates_task_1_A_Nreward_interleaved_1[:,i,:],0)
                
                mean_firing_rate_task_1_second_half_A_reward  = np.mean(aligned_rates_task_1_A_reward_interleaved_2[:,i,:],0)
                mean_firing_rate_task_1_second_half_A_Nreward  = np.mean(aligned_rates_task_1_A_Nreward_interleaved_2[:,i,:],0)
                
                mean_firing_rate_task_2_first_half_A_reward  = np.mean(aligned_rates_task_2_A_reward_interleaved_1[:,i,:],0)
                mean_firing_rate_task_2_first_half_A_Nreward  = np.mean(aligned_rates_task_2_A_Nreward_interleaved_1[:,i,:],0)
                
                mean_firing_rate_task_2_second_half_A_reward  = np.mean(aligned_rates_task_2_A_reward_interleaved_2[:,i,:],0)
                mean_firing_rate_task_2_second_half_A_Nreward  = np.mean(aligned_rates_task_2_A_Nreward_interleaved_2[:,i,:],0)
                
                mean_firing_rate_task_1_first_half_B_reward  = np.mean(aligned_rates_task_1_B_reward_interleaved_1[:,i,:],0)
                mean_firing_rate_task_1_first_half_B_Nreward  = np.mean(aligned_rates_task_1_B_Nreward_interleaved_1[:,i,:],0)
                
                mean_firing_rate_task_1_second_half_B_reward  = np.mean(aligned_rates_task_1_B_reward_interleaved_2[:,i,:],0)     
                mean_firing_rate_task_1_second_half_B_Nreward  = np.mean(aligned_rates_task_1_B_Nreward_interleaved_2[:,i,:],0)
        
                mean_firing_rate_task_2_first_half_B_reward  = np.mean(aligned_rates_task_2_B_reward_interleaved_1[:,i,:],0)
                mean_firing_rate_task_2_first_half_B_Nreward  = np.mean(aligned_rates_task_2_B_Nreward_interleaved_1[:,i,:],0)
                
                mean_firing_rate_task_2_second_half_B_reward  = np.mean(aligned_rates_task_2_B_reward_interleaved_2[:,i,:],0)
                mean_firing_rate_task_2_second_half_B_Nreward  = np.mean(aligned_rates_task_2_B_Nreward_interleaved_2[:,i,:],0)
               
                if tasks_unchanged == True:
                    mean_firing_rate_task_3_first_half_A_reward  = np.mean(aligned_rates_task_3_A_reward_interleaved_1[:,i,:],0)
                    mean_firing_rate_task_3_first_half_A_Nreward  = np.mean(aligned_rates_task_3_A_Nreward_interleaved_1[:,i,:],0)
                
                    mean_firing_rate_task_3_second_half_A_reward  = np.mean(aligned_rates_task_3_A_reward_interleaved_2[:,i,:],0)
                    mean_firing_rate_task_3_second_half_A_Nreward  = np.mean(aligned_rates_task_3_A_Nreward_interleaved_2[:,i,:],0)
    
                    mean_firing_rate_task_3_first_half_B_reward  = np.mean(aligned_rates_task_3_B_reward_interleaved_1[:,i,:],0)
                    mean_firing_rate_task_3_first_half_B_Nreward  = np.mean(aligned_rates_task_3_B_Nreward_interleaved_1[:,i,:],0)
                   
                    mean_firing_rate_task_3_second_half_B_reward  = np.mean(aligned_rates_task_3_B_reward_interleaved_2[:,i,:],0)
                    mean_firing_rate_task_3_second_half_B_Nreward  = np.mean(aligned_rates_task_3_B_Nreward_interleaved_2[:,i,:],0)
                    
                
                if plot_a == False and plot_b == False and split_by_reward == False: 
                    mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward,\
                                                                         mean_firing_rate_task_1_first_half_B_reward,\
                                                                         mean_firing_rate_task_1_first_half_B_Nreward), axis = 0)
                    
                    mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward,\
                                                                         mean_firing_rate_task_1_second_half_B_reward,\
                                                                         mean_firing_rate_task_1_second_half_B_Nreward), axis = 0)
                    
                    mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward,\
                                                                         mean_firing_rate_task_2_first_half_B_reward,\
                                                                         mean_firing_rate_task_2_first_half_B_Nreward), axis = 0)
                    
                    mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward,\
                                                                         mean_firing_rate_task_2_second_half_B_reward,\
                                                                         mean_firing_rate_task_2_second_half_B_Nreward), axis = 0)
                    
                elif plot_a == True and plot_b == False:
                    mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward), axis = 0) 
                    mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward), axis = 0)
                    mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward), axis = 0)
                    mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward), axis = 0)
                  
                elif plot_a == False and plot_b == True:
                   mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward), axis = 0)
                   mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward), axis = 0)           
                   mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward), axis = 0)       
                   mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward), axis = 0)          
                
                elif plot_a == False and plot_b == False and split_by_reward == True: 
                    mean_firing_rate_task_1_first_half = np.mean([mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward], axis = 0) 
                    mean_firing_rate_task_1_second_half = np.mean([mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward], axis = 0)
                    mean_firing_rate_task_2_first_half = np.mean([mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward], axis = 0)
                    mean_firing_rate_task_2_second_half = np.mean([mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward], axis = 0)
                  
                    mean_firing_rate_task_1_first_half = np.mean([mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward], axis = 0)
                    mean_firing_rate_task_1_second_half = np.mean([mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward], axis = 0)           
                    mean_firing_rate_task_2_first_half = np.mean([mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward], axis = 0)       
                    mean_firing_rate_task_2_second_half = np.mean([mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward], axis = 0)          

                if tasks_unchanged == True: 
                    if plot_a == False and plot_b == False and split_by_reward == False: 
                        mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward,\
                                                                         mean_firing_rate_task_3_first_half_B_reward,\
                                                                         mean_firing_rate_task_3_first_half_B_Nreward), axis = 0)
                    
                        mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward,\
                                                                         mean_firing_rate_task_3_second_half_B_reward,\
                                                                         mean_firing_rate_task_3_second_half_B_Nreward), axis = 0)
                    elif plot_a == True and plot_b == False :
                        mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward), axis = 0)
                        mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward), axis = 0)
                    
                    elif plot_a == False and plot_b == True:
                        mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_B_reward, mean_firing_rate_task_3_first_half_B_Nreward), axis = 0)
                        mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_B_reward, mean_firing_rate_task_3_second_half_B_Nreward), axis = 0)
                    elif plot_a == False and plot_b == False and split_by_reward == True: 
                        mean_firing_rate_task_3_first_half = np.mean([mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward], axis = 0) 
                        mean_firing_rate_task_3_second_half = np.mean([mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward], axis = 0)
                       
                all_clusters_task_1_first_half.append(mean_firing_rate_task_1_first_half)
                all_clusters_task_1_second_half.append(mean_firing_rate_task_1_second_half)
                all_clusters_task_2_first_half.append(mean_firing_rate_task_2_first_half)
                all_clusters_task_2_second_half.append(mean_firing_rate_task_2_second_half)
                
                if tasks_unchanged == True: 
                    all_clusters_task_3_first_half.append(mean_firing_rate_task_3_first_half)
                    all_clusters_task_3_second_half.append(mean_firing_rate_task_3_second_half)
    if tasks_unchanged == True:        
        return all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
            all_clusters_task_2_first_half, all_clusters_task_2_second_half,\
            all_clusters_task_3_first_half,all_clusters_task_3_second_half
    else:    
        return all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
            all_clusters_task_2_first_half, all_clusters_task_2_second_half
        


def demean_data_tim(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, split_by_reward = False, diagonal = False):
    
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = replicate_Tim_trial_selection(experiment, tasks_unchanged = True, plot_a = plot_a, plot_b = plot_b, split_by_reward = split_by_reward )
    
    flattened_all_clusters_task_1_first_half  = np.asarray(flattened_all_clusters_task_1_first_half)
    flattened_all_clusters_task_1_second_half = np.asarray(flattened_all_clusters_task_1_second_half)
    flattened_all_clusters_task_2_first_half = np.asarray(flattened_all_clusters_task_2_first_half)
    flattened_all_clusters_task_2_second_half = np.asarray(flattened_all_clusters_task_2_second_half)
    flattened_all_clusters_task_3_first_half = np.asarray(flattened_all_clusters_task_3_first_half)
    flattened_all_clusters_task_3_second_half = np.asarray(flattened_all_clusters_task_3_second_half)
    
  
    
    if tasks_unchanged == True:
        
        all_data = np.concatenate([flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_first_half,\
                                   flattened_all_clusters_task_2_second_half, flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half], axis = 1)
        all_data_mean = np.mean(all_data, axis = 1)
    
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
        
        demean_all_clusters_task_1_first_half = demeaned[:,:flattened_all_clusters_task_1_first_half.shape[1]]
        demean_all_clusters_task_1_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]:flattened_all_clusters_task_1_first_half.shape[1]*2]
        
        demean_all_clusters_task_2_first_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*2:flattened_all_clusters_task_1_first_half.shape[1]*3]
        demean_all_clusters_task_2_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*3:flattened_all_clusters_task_1_first_half.shape[1]*4]
    if tasks_unchanged == True:
        return demean_all_clusters_task_1_first_half, demean_all_clusters_task_1_second_half,demean_all_clusters_task_2_first_half,demean_all_clusters_task_2_second_half,demean_all_clusters_task_3_first_half,\
    demean_all_clusters_task_3_second_half
    else:
        return demean_all_clusters_task_1_first_half, demean_all_clusters_task_1_second_half,demean_all_clusters_task_2_first_half,demean_all_clusters_task_2_second_half
    
def correlations_tim(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, split_by_reward = False):
    all_clusters_task_1_first_half, all_clusters_task_1_second_half,all_clusters_task_2_first_half,all_clusters_task_2_second_half,all_clusters_task_3_first_half,\
    all_clusters_task_3_second_half  = demean_data_tim(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, split_by_reward = split_by_reward )
    
    
    correlation_task_1_1 = np.linalg.multi_dot([all_clusters_task_1_first_half, np.transpose(all_clusters_task_1_first_half)])/all_clusters_task_3_first_half.shape[0]
    correlation_task_1_2 = np.linalg.multi_dot([all_clusters_task_1_second_half, np.transpose(all_clusters_task_1_second_half)])/all_clusters_task_3_first_half.shape[0]

    correlation_task_2_1 = np.linalg.multi_dot([all_clusters_task_2_first_half, np.transpose(all_clusters_task_2_first_half)])/all_clusters_task_3_first_half.shape[0]
    correlation_task_2_2 = np.linalg.multi_dot([all_clusters_task_2_second_half, np.transpose(all_clusters_task_2_second_half)])/all_clusters_task_3_first_half.shape[0]

    correlation_task_3_1 = np.linalg.multi_dot([all_clusters_task_3_first_half, np.transpose(all_clusters_task_1_first_half)])/all_clusters_task_3_first_half.shape[0]
    correlation_task_3_2 = np.linalg.multi_dot([all_clusters_task_3_second_half, np.transpose(all_clusters_task_3_second_half)])/all_clusters_task_3_first_half.shape[0]
 
    correlation_task_1_1 = np.triu(correlation_task_1_1)
    correlation_task_1_1 = correlation_task_1_1.flatten()
    
    correlation_task_1_2= np.triu(correlation_task_1_2)
    correlation_task_1_2 = correlation_task_1_2.flatten()
    
    correlation_task_2_1 = np.triu(correlation_task_2_1)
    correlation_task_2_1 = correlation_task_2_1.flatten()
    
    correlation_task_2_2 = np.triu(correlation_task_2_2)
    correlation_task_2_2 = correlation_task_2_2.flatten()
    
    correlation_task_3_1 = np.triu(correlation_task_3_1)
    correlation_task_3_1 = correlation_task_3_1.flatten()
    
    correlation_task_3_2 = np.triu(correlation_task_3_2)
    correlation_task_3_2 = correlation_task_3_2.flatten()
    
    
    plt.figure()
    plt.scatter(correlation_task_1_2,correlation_task_2_1, s =1, color = 'black')

    gradient, intercept, r_value, p_value, std_err = stats.linregress(correlation_task_1_2,correlation_task_2_1)

    mn=np.min(correlation_task_1_2)
    mx=np.max(correlation_task_1_2)
    x1=np.linspace(mn,mx,500)
    y1=gradient*x1+intercept
    plt.plot(x1,y1,'black')
    plt.show()
    plt.xlabel('Task 1')
    plt.ylabel('Task 2')
    plt.title('Covariance PFC')
    plt.savefig('/Users/veronikasamborska/Desktop/HP.pdf')
    
    return r_value,p_value



def svd_plotting_tim(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, split_by_reward = False, diagonal = False):
    #Calculating SVDs for trials split by A and B, reward/no reward (no block information) 
    if tasks_unchanged == True:
        
        all_clusters_task_1_first_half, all_clusters_task_1_second_half,all_clusters_task_2_first_half,all_clusters_task_2_second_half,all_clusters_task_3_first_half,\
        all_clusters_task_3_second_half = replicate_Tim_trial_selection(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a,\
                          plot_b = plot_b, split_by_reward = split_by_reward)
       
        all_clusters_task_1_first_half = np.asarray(all_clusters_task_1_first_half)
        all_clusters_task_1_second_half = np.asarray(all_clusters_task_1_second_half)
        all_clusters_task_2_first_half = np.asarray(all_clusters_task_2_first_half)
        all_clusters_task_2_second_half = np.asarray(all_clusters_task_2_second_half)
        all_clusters_task_3_first_half = np.asarray(all_clusters_task_3_first_half)
        all_clusters_task_3_second_half = np.asarray(all_clusters_task_3_second_half)
    else:
        all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
        all_clusters_task_2_first_half,all_clusters_task_2_second_half = replicate_Tim_trial_selection(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a,\
                                                                                         plot_b = plot_b,split_by_reward = split_by_reward)
    
    
        all_clusters_task_1_first_half  = np.asarray(all_clusters_task_1_first_half)
        all_clusters_task_1_second_half = np.asarray(all_clusters_task_1_second_half)
        all_clusters_task_2_first_half = np.asarray(all_clusters_task_2_first_half)
        all_clusters_task_2_second_half = np.asarray(all_clusters_task_2_second_half)
      
  
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(all_clusters_task_1_first_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(all_clusters_task_1_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(all_clusters_task_2_first_half, full_matrices = False)

    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(all_clusters_task_2_second_half, full_matrices = False)
    if tasks_unchanged == True:

        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(all_clusters_task_3_first_half, full_matrices = False)
    

    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)
    t_v = np.transpose(vh_t1_1)
    
    t_u_t_2_1 = np.transpose(u_t2_1)
    t_v_t_2_1 = np.transpose(vh_t2_1)
    if tasks_unchanged == True:

        t_u_t_3_1 = np.transpose(u_t3_1)
        t_v_t_3_1 = np.transpose(vh_t3_1)
    

    #Compare task 2 First Half 
    s_task_2_1 = np.linalg.multi_dot([t_u, all_clusters_task_2_first_half, t_v])
    if diagonal == True:
        s_2_1 = s_task_2_1.diagonal()
    else:
        s_2_1 = np.sum(s_task_2_1**2, axis = 1)
        #s_2_1 = s_2_1/s_2_1[-1]

    sum_c_task_2_1 = np.cumsum(abs(s_2_1))/all_clusters_task_2_first_half.shape[0]
    
    #Compare task 2 Second Half
    s_task_2_2 = np.linalg.multi_dot([t_u, all_clusters_task_2_second_half, t_v])
    if diagonal == True:
        s_2_2 = s_task_2_2.diagonal()
    else:
        s_2_2 = np.sum(s_task_2_2**2, axis = 1)
        #s_2_2 = s_2_2/s_2_2[-1]
        
    sum_c_task_2_2 = np.cumsum(abs(s_2_2))/all_clusters_task_2_second_half.shape[0]
    
    #Compare task 2 Second Half from first half
    s_task_2_2_from_t_2_1 = np.linalg.multi_dot([t_u_t_2_1, all_clusters_task_2_second_half, t_v_t_2_1])
    if diagonal == True:
        s_2_2_from_t_2_1 = s_task_2_2_from_t_2_1.diagonal()
    else:
        s_2_2_from_t_2_1 = np.sum(s_task_2_2_from_t_2_1**2, axis = 1)
        #s_2_2_from_t_2_1 = s_2_2_from_t_2_1/s_2_2_from_t_2_1[-1]
    
    sum_c_task_2_2_from_t_2_1 = np.cumsum(abs(s_2_2_from_t_2_1))/all_clusters_task_2_second_half.shape[0]
    
    
    if tasks_unchanged == True: 
        #Compare task 3 First Half from Task 1
        s_task_3_1 = np.linalg.multi_dot([t_u, all_clusters_task_3_first_half, t_v])
        if diagonal == True:
            s_3_1 = s_task_3_1.diagonal()
        else:
            s_3_1 = np.sum(s_task_3_1**2, axis = 1)
            #s_3_1 = s_3_1/s_3_1[-1]

        sum_c_task_3_1 = np.cumsum(abs(s_3_1))/all_clusters_task_3_first_half.shape[0]
        
        #Compare task 3 Second Half from Task 1
        s_task_3_2 = np.linalg.multi_dot([t_u, all_clusters_task_3_second_half, t_v])
        if diagonal == True:
            s_3_2 = s_task_3_2.diagonal()
        else:
            s_3_2 = np.sum(s_task_3_2**2, axis = 1)
            #s_3_2 = s_3_2/s_3_2[-1]

        sum_c_task_3_2 = np.cumsum(abs(s_3_2))/all_clusters_task_3_second_half.shape[0]
        
        #Compare task 3 First Half from Task 3 first Half 
        s_task_3_2_from_t_3_1 = np.linalg.multi_dot([t_u_t_3_1, all_clusters_task_3_second_half, t_v_t_3_1])
        if diagonal == True:
            s_3_2_from_t_3_1 = s_task_3_2_from_t_3_1.diagonal()
        else:
            s_3_2_from_t_3_1 = np.sum(s_task_3_2_from_t_3_1**2, axis = 1)
           # s_3_2_from_t_3_1 = s_3_2_from_t_3_1/s_3_2_from_t_3_1[-1]

        sum_c_task_3_2_from_t_3_1 = np.cumsum(abs(s_3_2_from_t_3_1))/all_clusters_task_3_second_half.shape[0]
    
        
        task_a_a = np.mean([sum_c_task_2_2_from_t_2_1 ,sum_c_task_3_2_from_t_3_1], axis = 0)
        #task_a_a = task_a_a/task_a_a[-1]
        
        task_a_b_c = np.mean([sum_c_task_2_1, sum_c_task_2_2,sum_c_task_3_1,sum_c_task_3_2], axis = 0)
        #task_a_b_c = task_a_b_c/task_a_b_c[-1]

        
    else:
        task_a_a = sum_c_task_2_2_from_t_2_1
        task_a_a = task_a_a/task_a_a[-1]
        
        task_a_b_c = np.mean([sum_c_task_2_1, sum_c_task_2_2], axis = 0)
        task_a_b_c = task_a_b_c/task_a_b_c[-1]

    if HP == True:
        plt.figure(13)
        plt.plot(task_a_b_c, label = 'Explain Task B/C from A HP', linestyle = '--', color='red')

        plt.plot(task_a_a, label = 'Explain Task A from A HP', color = 'red')

    if HP == False:
        plt.figure(13)

        plt.plot(task_a_b_c, label = 'Explain Task B/C from A PFC', linestyle = '--', color='blue')

        plt.plot(task_a_a, label = 'Explain Task A from A PFC', color = 'blue')



    plt.legend()