#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:09:46 2019

@author: veronikasamborska
"""

import SVDs as sv 
import numpy as np 

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
    

def replicate_Tim_trial_selection(experiment, tasks_unchanged = True, only_a = False, only_b = False, split_by_reward = False):       
    all_clusters_task_1_first_half = []
    all_clusters_task_1_second_half = []
    all_clusters_task_2_first_half = []
    all_clusters_task_2_second_half = []
    all_clusters_task_3_first_half = []
    all_clusters_task_3_second_half = []
    
  
    for s,session in enumerate(experiment):
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        
        if s != 15 and s !=31:
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
                    
                
                if only_a == False and only_b == False and split_by_reward == False: 
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
                    
                elif only_a == True and only_b == False:
                    mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward), axis = 0) 
                    mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward), axis = 0)
                    mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward), axis = 0)
                    mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward), axis = 0)
                  
                elif only_a == False and only_b == True:
                   mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward), axis = 0)
                   mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward), axis = 0)           
                   mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward), axis = 0)       
                   mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward), axis = 0)          
                
                elif only_a == False and only_b == False and split_by_reward == True: 
                    mean_firing_rate_task_1_first_half = np.mean([mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward], axis = 0) 
                    mean_firing_rate_task_1_second_half = np.mean([mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward], axis = 0)
                    mean_firing_rate_task_2_first_half = np.mean([mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward], axis = 0)
                    mean_firing_rate_task_2_second_half = np.mean([mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward], axis = 0)
                  
                    mean_firing_rate_task_1_first_half = np.mean([mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward], axis = 0)
                    mean_firing_rate_task_1_second_half = np.mean([mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward], axis = 0)           
                    mean_firing_rate_task_2_first_half = np.mean([mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward], axis = 0)       
                    mean_firing_rate_task_2_second_half = np.mean([mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward], axis = 0)          

                if tasks_unchanged == True: 
                    if only_a == False and only_b == False and split_by_reward == False: 
                        mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward,\
                                                                         mean_firing_rate_task_3_first_half_B_reward,\
                                                                         mean_firing_rate_task_3_first_half_B_Nreward), axis = 0)
                    
                        mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward,\
                                                                         mean_firing_rate_task_3_second_half_B_reward,\
                                                                         mean_firing_rate_task_3_second_half_B_Nreward), axis = 0)
                    elif only_a == True and only_b == False :
                        mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward), axis = 0)
                        mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward), axis = 0)
                    
                    elif only_a == False and only_b == True:
                        mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_B_reward, mean_firing_rate_task_3_first_half_B_Nreward), axis = 0)
                        mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_B_reward, mean_firing_rate_task_3_second_half_B_Nreward), axis = 0)
                    elif only_a == False and only_b == False and split_by_reward == True: 
                        mean_firing_rate_task_3_first_half = np.mean([mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward], axis = 0) 
                        mean_firing_rate_task_3_second_half = np.mean([mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward], axis = 0)
                       
                all_clusters_task_1_first_half.append(mean_firing_rate_task_1_first_half)
                all_clusters_task_1_second_half.append(mean_firing_rate_task_1_second_half)
                all_clusters_task_2_first_half.append(mean_firing_rate_task_2_first_half)
                all_clusters_task_2_second_half.append(mean_firing_rate_task_2_second_half)
                
                if tasks_unchanged == True: 
                    all_clusters_task_3_first_half.append(mean_firing_rate_task_3_first_half)
                    all_clusters_task_3_second_half.append(mean_firing_rate_task_3_second_half)
            
    return all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
        all_clusters_task_2_first_half, all_clusters_task_2_second_half,\
        all_clusters_task_3_first_half,all_clusters_task_3_second_half
        
        



def svd_plotting_tim(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, split_by_reward = False):
    #Calculating SVDs for trials split by A and B, reward/no reward (no block information) 

    all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
    all_clusters_task_2_first_half, all_clusters_task_2_second_half,\
    all_clusters_task_3_first_half,all_clusters_task_3_second_half = replicate_Tim_trial_selection(experiment, tasks_unchanged = True, only_a = plot_a, only_b = plot_b, split_by_reward = split_by_reward )
    
    all_clusters_task_1_first_half  = np.asarray(all_clusters_task_1_first_half)
    all_clusters_task_1_second_half = np.asarray(all_clusters_task_1_second_half)
    all_clusters_task_2_first_half = np.asarray(all_clusters_task_2_first_half)
    all_clusters_task_2_second_half = np.asarray(all_clusters_task_2_second_half)
    all_clusters_task_3_first_half = np.asarray(all_clusters_task_3_first_half)
    all_clusters_task_3_second_half = np.asarray(all_clusters_task_3_second_half)
    
    
  
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(all_clusters_task_1_first_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(all_clusters_task_1_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(all_clusters_task_2_first_half, full_matrices = False)

    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(all_clusters_task_2_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(all_clusters_task_3_first_half, full_matrices = False)


    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)
    t_v = np.transpose(vh_t1_1)
    
    t_u_t_2_1 = np.transpose(u_t2_1)
    t_v_t_2_1 = np.transpose(vh_t2_1)
    
    t_u_t_3_1 = np.transpose(u_t3_1)
    t_v_t_3_1 = np.transpose(vh_t3_1)
    
    #Compare task 1 Second Half 
    s_task_1_2 = np.linalg.multi_dot([t_u, all_clusters_task_1_second_half, t_v])
    s_1_2 = s_task_1_2.diagonal()
    sum_c_task_1_2 = np.cumsum(s_1_2)/all_clusters_task_1_second_half.shape[0]
    
    #Compare task 2 First Half 
    s_task_2_1 = np.linalg.multi_dot([t_u, all_clusters_task_2_first_half, t_v])
    s_2_1 = s_task_2_1.diagonal()
    sum_c_task_2_1 = np.cumsum(s_2_1)/all_clusters_task_2_first_half.shape[0]
    
    #Compare task 2 Second Half
    s_task_2_2 = np.linalg.multi_dot([t_u, all_clusters_task_2_second_half, t_v])
    s_2_2 = s_task_2_2.diagonal()
    sum_c_task_2_2 = np.cumsum(s_2_2)/all_clusters_task_2_second_half.shape[0]
    
    #Compare task 2 Second Half from first half
    s_task_2_2_from_t_2_1 = np.linalg.multi_dot([t_u_t_2_1, all_clusters_task_2_second_half, t_v_t_2_1])
    s_2_2_from_t_2_1 = s_task_2_2_from_t_2_1.diagonal()
    sum_c_task_2_2_from_t_2_1 = np.cumsum(s_2_2_from_t_2_1)/all_clusters_task_2_second_half.shape[0]
    
    
    if tasks_unchanged == True: 
        #Compare task 3 First Half from Task 1
        s_task_3_1 = np.linalg.multi_dot([t_u, all_clusters_task_3_first_half, t_v])
        s_3_1 = s_task_3_1.diagonal()
        sum_c_task_3_1 = np.cumsum(s_3_1)/all_clusters_task_3_first_half.shape[0]
        
        #Compare task 3 Second Half from Task 1
        s_task_3_2 = np.linalg.multi_dot([t_u, all_clusters_task_3_second_half, t_v])
        s_3_2 = s_task_3_2.diagonal()
        sum_c_task_3_2 = np.cumsum(s_3_2)/all_clusters_task_3_second_half.shape[0]
        
        #Compare task 3 First Half from Task 3 first Half 
        s_task_3_2_from_t_3_1 = np.linalg.multi_dot([t_u_t_3_1, all_clusters_task_3_second_half, t_v_t_3_1])
        s_3_2_from_t_3_1 = s_task_3_2_from_t_3_1.diagonal()
        sum_c_task_3_2_from_t_3_1 = np.cumsum(s_3_2_from_t_3_1)/all_clusters_task_3_second_half.shape[0]
    
        
        
        task_a_a = np.mean([sum_c_task_1_2,sum_c_task_2_2_from_t_2_1 ,sum_c_task_3_2_from_t_3_1], axis = 0)
        
        task_a_b_c = np.mean([sum_c_task_2_1, sum_c_task_2_2,sum_c_task_3_1,sum_c_task_3_2], axis = 0)
        
        std_within = np.std([sum_c_task_1_2,sum_c_task_2_2_from_t_2_1 ,sum_c_task_3_2_from_t_3_1], axis = 0)
        x_within = np.arange(len(task_a_a))
        std_between = np.std([sum_c_task_2_1, sum_c_task_2_2,sum_c_task_3_1,sum_c_task_3_2], axis = 0)
        x_between = np.arange(len(task_a_b_c))
        

    if HP == True:
        plot(task_a_b_c, label = 'Explain Task B/C from A HP', linestyle = '--', color='red')
        fill_between(x_within,task_a_a+std_within,task_a_a-std_within, color = 'red', alpha = 0.2)

        plot(task_a_a, label = 'Explain Task A from A HP', color = 'red')
        fill_between(x_between,task_a_b_c+std_between, task_a_b_c-std_between, color = 'red', alpha = 0.2)

    if HP == False:
        plot(task_a_b_c, label = 'Explain Task B/C from A PFC', linestyle = '--', color='blue')
        fill_between(x_within,task_a_a+std_within,task_a_a-std_within, color = 'blue', alpha = 0.2)

        plot(task_a_a, label = 'Explain Task A from A PFC', color = 'blue')

        fill_between(x_between,task_a_b_c+std_between, task_a_b_c-std_between, color = 'blue', alpha = 0.2)


    legend()