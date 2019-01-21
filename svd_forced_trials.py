#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:28:27 2019

@author: veronikasamborska
"""

import forced_trials_extract_data as ft
import numpy as np 

#PFC_forced = ha.all_sessions_aligment_forced(PFC)
#HP_forced = ha.all_sessions_aligment_forced(HP)


def extract_session_a_b_based_on_block_forced_trials(session, tasks_unchanged = False):
    spikes = session.ephys
    spikes = spikes[:,~np.isnan(spikes[1,:])] 
    aligned_rates = session.aligned_rates
    
    forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]
    
    task = session.trial_data['task']
    task_forced = task[forced_array]
    
    task_1 = len(np.where(task_forced == 1)[0])
    task_2 = len(np.where(task_forced == 2)[0])
    # Getting choice indices 
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = ft.predictors_forced(session)
    
    if aligned_rates.shape[0] != predictor_A_Task_1.shape[0]:
        predictor_A_Task_1 = predictor_A_Task_1[:aligned_rates.shape[0]] 
        predictor_A_Task_2 = predictor_A_Task_2[:aligned_rates.shape[0]] 
        predictor_A_Task_3 = predictor_A_Task_3[:aligned_rates.shape[0]] 
        predictor_B_Task_1 = predictor_B_Task_1[:aligned_rates.shape[0]] 
        predictor_B_Task_2 = predictor_B_Task_2[:aligned_rates.shape[0]] 
        predictor_B_Task_3 = predictor_B_Task_3[:aligned_rates.shape[0]] 
        reward = reward[:aligned_rates.shape[0]] 
        
   
    #Get firing rates for each task
    aligned_rates_task_1 = aligned_rates[:task_1]
    aligned_rates_task_2 = aligned_rates[task_1:task_1+task_2]
    aligned_rates_task_3 = aligned_rates[task_1+task_2:]
    
    #Indicies of A choices in each task (1s) and Bs are just 0s 
    predictor_A_Task_1_cut = predictor_A_Task_1[:task_1]
    reward_task_1_cut = reward[:task_1]
          
    predictor_A_Task_2_cut = predictor_A_Task_2[task_1:task_1+task_2]
    reward_task_2_cut = reward[task_1:task_1+task_2]
          
    predictor_A_Task_3_cut = predictor_A_Task_3[task_1+task_2:]
    reward_task_3_cut = reward[task_1+task_2:]
    
    
    # Make arrays with 1s and 0s to mark states in the task
    states_task_1 = np.zeros(len(predictor_A_Task_1_cut))
    states_task_1[predictor_a_good_task_1] = 1
    states_task_2 = np.zeros(len(predictor_A_Task_2_cut))
    states_task_2[predictor_a_good_task_2] = 1
    states_task_3 = np.zeros(len(predictor_A_Task_3_cut))
    states_task_3[predictor_a_good_task_3] = 1
    
    state_A_choice_A_t1 = aligned_rates_task_1[np.where((states_task_1 ==1) & (predictor_A_Task_1_cut == 1 ))]
    state_A_choice_B_t1 = aligned_rates_task_1[np.where((states_task_1 ==1) & (predictor_A_Task_1_cut == 0))]
    
    state_B_choice_A_t1 = aligned_rates_task_1[np.where((states_task_1 == 0) & (predictor_A_Task_1_cut == 1 ))]
    state_B_choice_B_t1 = aligned_rates_task_1[np.where((states_task_1 == 0) & (predictor_A_Task_1_cut == 0))]
    
    state_A_choice_A_t2 = aligned_rates_task_2[np.where((states_task_2 ==1) & (predictor_A_Task_2_cut == 1 ))]
    state_A_choice_B_t2 = aligned_rates_task_2[np.where((states_task_2 ==1) & (predictor_A_Task_2_cut == 0))]
    
    state_B_choice_A_t2 = aligned_rates_task_2[np.where((states_task_2 == 0) & (predictor_A_Task_2_cut == 1 ))]
    state_B_choice_B_t2 = aligned_rates_task_2[np.where((states_task_2 == 0) & (predictor_A_Task_2_cut == 0))]

    state_A_choice_A_t3 = aligned_rates_task_3[np.where((states_task_3 ==1) & (predictor_A_Task_3_cut == 1 ))]
    state_A_choice_B_t3 = aligned_rates_task_3[np.where((states_task_3 ==1) & (predictor_A_Task_3_cut == 0))]
    
    state_B_choice_A_t3 = aligned_rates_task_3[np.where((states_task_3 == 0) & (predictor_A_Task_3_cut == 1 ))]
    state_B_choice_B_t3 = aligned_rates_task_3[np.where((states_task_3 == 0) & (predictor_A_Task_3_cut == 0))]

    return state_A_choice_A_t1,state_A_choice_B_t1,state_B_choice_A_t1,state_B_choice_B_t1,\
        state_A_choice_A_t2, state_A_choice_B_t2,state_B_choice_A_t2,state_B_choice_B_t2,\
        state_A_choice_A_t3, state_A_choice_B_t3, state_B_choice_A_t3, state_B_choice_B_t3, spikes
        
        
def block_firings_rates_selection_forced_split_in_half(experiment):   
    cluster_list_task_1_a_good_1 = []
    cluster_list_task_1_b_good_1 = []   
    cluster_list_task_2_a_good_1 = []
    cluster_list_task_2_b_good_1 = []
    cluster_list_task_3_a_good_1 = []
    cluster_list_task_3_b_good_1 = []
    
    cluster_list_task_1_a_good_2 = []
    cluster_list_task_1_b_good_2 = []   
    cluster_list_task_2_a_good_2 = []
    cluster_list_task_2_b_good_2 = []
    cluster_list_task_3_a_good_2 = []
    cluster_list_task_3_b_good_2 = []
    
    for s,session in enumerate(experiment):
        if session.trial_data['block'][-1] >= 11:
            
            state_A_choice_A_t1,state_A_choice_B_t1,state_B_choice_A_t1,state_B_choice_B_t1,\
            state_A_choice_A_t2, state_A_choice_B_t2,state_B_choice_A_t2,state_B_choice_B_t2,\
            state_A_choice_A_t3, state_A_choice_B_t3, state_B_choice_A_t3, state_B_choice_B_t3, spikes = extract_session_a_b_based_on_block_forced_trials(session)
                   
            state_A_choice_A_t1_1 = state_A_choice_A_t1[:int(len(state_A_choice_A_t1)/2)]
            state_A_choice_B_t1_1 = state_A_choice_B_t1[:int(len(state_A_choice_B_t1)/2)]
            state_B_choice_A_t1_1 = state_B_choice_A_t1[:int(len(state_B_choice_A_t1)/2)]
            state_B_choice_B_t1_1 = state_B_choice_B_t1[:int(len(state_B_choice_B_t1)/2)]
            state_A_choice_A_t2_1 = state_A_choice_A_t2[:int(len(state_A_choice_A_t2)/2)]
            state_A_choice_B_t2_1 = state_A_choice_B_t2[:int(len(state_A_choice_B_t2)/2)]
            state_B_choice_A_t2_1 = state_B_choice_A_t2[:int(len(state_B_choice_A_t2)/2)]
            state_B_choice_B_t2_1 = state_B_choice_B_t2[:int(len(state_B_choice_B_t2)/2)]
            state_A_choice_A_t3_1 = state_A_choice_A_t3[:int(len(state_A_choice_A_t3)/2)]
            state_A_choice_B_t3_1 = state_A_choice_B_t3[:int(len(state_A_choice_B_t3)/2)]
            state_B_choice_A_t3_1 = state_B_choice_A_t3[:int(len(state_B_choice_A_t3)/2)]
            state_B_choice_B_t3_1 = state_B_choice_B_t3[:int(len(state_B_choice_B_t3)/2)]
            
            state_A_choice_A_t1_2 = state_A_choice_A_t1[int(len(state_A_choice_A_t1)/2):]
            state_A_choice_B_t1_2 = state_A_choice_B_t1[int(len(state_A_choice_B_t1)/2):]
            state_B_choice_A_t1_2 = state_B_choice_A_t1[int(len(state_B_choice_A_t1)/2):]
            state_B_choice_B_t1_2 = state_B_choice_B_t1[int(len(state_B_choice_B_t1)/2):]
            state_A_choice_A_t2_2 = state_A_choice_A_t2[int(len(state_A_choice_A_t2)/2):]
            state_A_choice_B_t2_2 = state_A_choice_B_t2[int(len(state_A_choice_B_t2)/2):]
            state_B_choice_A_t2_2 = state_B_choice_A_t2[int(len(state_B_choice_A_t2)/2):]
            state_B_choice_B_t2_2 = state_B_choice_B_t2[int(len(state_B_choice_B_t2)/2):]
            state_A_choice_A_t3_2 = state_A_choice_A_t3[int(len(state_A_choice_A_t3)/2):]
            state_A_choice_B_t3_2 = state_A_choice_B_t3[int(len(state_A_choice_B_t3)/2):]
            state_B_choice_A_t3_2 = state_B_choice_A_t3[int(len(state_B_choice_A_t3)/2):]
            state_B_choice_B_t3_2 = state_B_choice_B_t3[int(len(state_B_choice_B_t3)/2):]
                
            if (len(state_A_choice_A_t1_1) > 0) & (len(state_A_choice_B_t1_1) > 0) & (len(state_B_choice_A_t1_1) > 0) &\
            (len(state_B_choice_B_t1_1) > 0) & (len(state_A_choice_A_t2_1) > 0) & (len(state_A_choice_B_t2_1) > 0) &\
            (len(state_B_choice_A_t2_1) > 0) & (len(state_B_choice_B_t2_1) > 0) & (len(state_A_choice_A_t3_1) > 0) &\
            (len(state_A_choice_B_t3_1) > 0) & (len(state_B_choice_A_t3_1) > 0) & (len(state_B_choice_B_t3_1) > 0) &\
            (len(state_A_choice_A_t1_2) > 0) & (len(state_A_choice_B_t1_2) > 0) & (len(state_B_choice_A_t1_2) > 0) &\
            (len(state_B_choice_B_t1_2) > 0) & (len(state_A_choice_A_t2_2) > 0) & (len(state_A_choice_B_t2_2) > 0) &\
            (len(state_B_choice_A_t2_2) > 0) & (len(state_B_choice_B_t2_2) > 0) & (len(state_A_choice_A_t3_2) > 0) &\
            (len(state_A_choice_B_t3_2) > 0) & (len(state_B_choice_A_t3_2) > 0) & (len(state_B_choice_B_t3_2) > 0):    
                unique_neurons  = np.unique(spikes[0])  
                
                for i in range(len(unique_neurons)):                
                    mean_firing_rate_task_1_a_good_A_ch_1  = np.mean(state_A_choice_A_t1_1[:,i,:],0)
                    mean_firing_rate_task_1_a_good_B_ch_1  = np.mean(state_A_choice_B_t1_1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_B_ch_1  = np.mean(state_B_choice_B_t1_1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch_1  = np.mean(state_B_choice_A_t1_1[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch_1  = np.mean(state_A_choice_A_t2_1[:,i,:],0)
                    mean_firing_rate_task_2_a_good_B_ch_1  = np.mean(state_A_choice_B_t2_1[:,i,:],0)
                    mean_firing_rate_task_2_b_good_B_ch_1  = np.mean(state_B_choice_B_t2_1[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch_1  = np.mean(state_B_choice_A_t2_1[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch_1  = np.mean(state_A_choice_A_t3_1[:,i,:],0)
                    mean_firing_rate_task_3_a_good_B_ch_1  = np.mean(state_A_choice_B_t3_1[:,i,:],0)
                    mean_firing_rate_task_3_b_good_B_ch_1  = np.mean(state_B_choice_B_t3_1[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch_1  = np.mean(state_B_choice_A_t3_1[:,i,:],0)
                    
                    
                    mean_firing_rate_task_1_a_good_A_ch_2  = np.mean(state_A_choice_A_t1_2[:,i,:],0)
                    mean_firing_rate_task_1_a_good_B_ch_2  = np.mean(state_A_choice_B_t1_2[:,i,:],0)
                    mean_firing_rate_task_1_b_good_B_ch_2  = np.mean(state_B_choice_B_t1_2[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch_2  = np.mean(state_B_choice_A_t1_2[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch_2  = np.mean(state_A_choice_A_t2_2[:,i,:],0)
                    mean_firing_rate_task_2_a_good_B_ch_2  = np.mean(state_A_choice_B_t2_2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_B_ch_2  = np.mean(state_B_choice_B_t2_2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch_2  = np.mean(state_B_choice_A_t2_2[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch_2  = np.mean(state_A_choice_A_t3_2[:,i,:],0)
                    mean_firing_rate_task_3_a_good_B_ch_2  = np.mean(state_A_choice_B_t3_2[:,i,:],0)
                    mean_firing_rate_task_3_b_good_B_ch_2  = np.mean(state_B_choice_B_t3_2[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch_2  = np.mean(state_B_choice_A_t3_2[:,i,:],0)
                    
                    mean_firing_rate_a_task_1_1 = np.concatenate((mean_firing_rate_task_1_a_good_A_ch_1,mean_firing_rate_task_1_a_good_B_ch_1), axis = 0)
                    mean_firing_rate_a_task_2_1 = np.concatenate((mean_firing_rate_task_2_a_good_A_ch_1,mean_firing_rate_task_2_a_good_B_ch_1), axis = 0)
                    mean_firing_rate_a_task_3_1 = np.concatenate((mean_firing_rate_task_3_a_good_A_ch_1,mean_firing_rate_task_3_a_good_B_ch_1), axis = 0)
                        
                    mean_firing_rate_b_task_1_1 = np.concatenate((mean_firing_rate_task_1_b_good_A_ch_1,mean_firing_rate_task_1_b_good_B_ch_1), axis = 0)
                    mean_firing_rate_b_task_2_1 = np.concatenate((mean_firing_rate_task_2_b_good_A_ch_1,mean_firing_rate_task_2_b_good_B_ch_1), axis = 0)
                    mean_firing_rate_b_task_3_1 = np.concatenate((mean_firing_rate_task_3_b_good_A_ch_1,mean_firing_rate_task_3_b_good_B_ch_1), axis = 0)
                    
                    mean_firing_rate_a_task_1_2 = np.concatenate((mean_firing_rate_task_1_a_good_A_ch_2,mean_firing_rate_task_1_a_good_B_ch_2), axis = 0)
                    mean_firing_rate_a_task_2_2 = np.concatenate((mean_firing_rate_task_2_a_good_A_ch_2,mean_firing_rate_task_2_a_good_B_ch_2), axis = 0)
                    mean_firing_rate_a_task_3_2 = np.concatenate((mean_firing_rate_task_3_a_good_A_ch_2,mean_firing_rate_task_3_a_good_B_ch_2), axis = 0)
                        
                    mean_firing_rate_b_task_1_2 = np.concatenate((mean_firing_rate_task_1_b_good_A_ch_2,mean_firing_rate_task_1_b_good_B_ch_2), axis = 0)
                    mean_firing_rate_b_task_2_2 = np.concatenate((mean_firing_rate_task_2_b_good_A_ch_2,mean_firing_rate_task_2_b_good_B_ch_2), axis = 0)
                    mean_firing_rate_b_task_3_2 = np.concatenate((mean_firing_rate_task_3_b_good_A_ch_2,mean_firing_rate_task_3_b_good_B_ch_2), axis = 0)
                        
                    cluster_list_task_1_a_good_1.append(mean_firing_rate_a_task_1_1)
                    cluster_list_task_1_b_good_1.append(mean_firing_rate_b_task_1_1)   
                    cluster_list_task_2_a_good_1.append(mean_firing_rate_a_task_2_1)
                    cluster_list_task_2_b_good_1.append(mean_firing_rate_b_task_2_1)
                    cluster_list_task_3_a_good_1.append(mean_firing_rate_a_task_3_1)
                    cluster_list_task_3_b_good_1.append(mean_firing_rate_b_task_3_1)
                    
                    cluster_list_task_1_a_good_2.append(mean_firing_rate_a_task_1_2)
                    cluster_list_task_1_b_good_2.append(mean_firing_rate_b_task_1_2)   
                    cluster_list_task_2_a_good_2.append(mean_firing_rate_a_task_2_2)
                    cluster_list_task_2_b_good_2.append(mean_firing_rate_b_task_2_2)
                    cluster_list_task_3_a_good_2.append(mean_firing_rate_a_task_3_2)
                    cluster_list_task_3_b_good_2.append(mean_firing_rate_b_task_3_2)
                    
    cluster_list_task_1_a_good_1 = np.asarray(cluster_list_task_1_a_good_1)
    cluster_list_task_1_b_good_1 = np.asarray(cluster_list_task_1_b_good_1)   
    cluster_list_task_2_a_good_1 = np.asarray(cluster_list_task_2_a_good_1)
    cluster_list_task_2_b_good_1 = np.asarray(cluster_list_task_2_b_good_1)
    cluster_list_task_3_a_good_1 = np.asarray(cluster_list_task_3_a_good_1)
    cluster_list_task_3_b_good_1 = np.asarray(cluster_list_task_3_b_good_1)
    
    cluster_list_task_1_a_good_2 = np.asarray(cluster_list_task_1_a_good_2)
    cluster_list_task_1_b_good_2 = np.asarray(cluster_list_task_1_b_good_2) 
    cluster_list_task_2_a_good_2 = np.asarray(cluster_list_task_2_a_good_2)
    cluster_list_task_2_b_good_2 = np.asarray(cluster_list_task_2_b_good_2)
    cluster_list_task_3_a_good_2 = np.asarray(cluster_list_task_3_a_good_2)
    cluster_list_task_3_b_good_2 = np.asarray(cluster_list_task_3_b_good_2)
    
    return cluster_list_task_1_a_good_1, cluster_list_task_1_b_good_1,\
    cluster_list_task_2_a_good_1, cluster_list_task_2_b_good_1,\
    cluster_list_task_3_a_good_1, cluster_list_task_3_b_good_1, cluster_list_task_1_a_good_2,\
    cluster_list_task_1_b_good_2, cluster_list_task_2_a_good_2, cluster_list_task_2_b_good_2,\
    cluster_list_task_3_a_good_2, cluster_list_task_3_b_good_2 


def plot_blocks_split_intwo(experiment, HP = False):
    
    #Explain A state from A state vs B state within a task 
    
    cluster_list_task_1_a_good_1, cluster_list_task_1_b_good_1,\
    cluster_list_task_2_a_good_1, cluster_list_task_2_b_good_1,\
    cluster_list_task_3_a_good_1, cluster_list_task_3_b_good_1, cluster_list_task_1_a_good_2,\
    cluster_list_task_1_b_good_2, cluster_list_task_2_a_good_2, cluster_list_task_2_b_good_2,\
    cluster_list_task_3_a_good_2, cluster_list_task_3_b_good_2  = block_firings_rates_selection_forced_split_in_half(experiment)
    
    #A good task 1
    u_t1_a_good_1, s_t1_a_good_1, vh_t1_a_good_1 = np.linalg.svd(cluster_list_task_1_a_good_1, full_matrices = False)
    u_t1_a_good_2, s_t1_a_good_2, vh_t1_a_good_2 = np.linalg.svd(cluster_list_task_1_a_good_2, full_matrices = False)
    
    #B good task 1
    u_t1_b_good_1, s_t1_b_good_1, vh_t1_b_good_1 = np.linalg.svd(cluster_list_task_1_b_good_1, full_matrices = False)    
    u_t1_b_good_2, s_t1_b_good_2, vh_t1_b_good_2 = np.linalg.svd(cluster_list_task_1_b_good_2, full_matrices = False)
    
    #A good task 2
    u_t2_a_good_1, s_t2_a_good_1, vh_t2_a_good_1 = np.linalg.svd(cluster_list_task_2_a_good_1, full_matrices = False)
    u_t2_a_good_2, s_t2_a_good_2, vh_t2_a_good_2 = np.linalg.svd(cluster_list_task_2_a_good_2, full_matrices = False)
    
    #B good task 2
    u_t2_b_good_1, s_t2_b_good_1, vh_t2_b_good_1 = np.linalg.svd(cluster_list_task_2_b_good_1, full_matrices = False)    
    u_t2_b_good_2, s_t2_b_good_2, vh_t2_b_good_2 = np.linalg.svd(cluster_list_task_2_b_good_2, full_matrices = False)

    #A good task 3
    u_t3_a_good_1, s_t3_a_good_1, vh_t3_a_good_1 = np.linalg.svd(cluster_list_task_3_a_good_1, full_matrices = False)
    u_t3_a_good_2, s_t3_a_good_2, vh_t3_a_good_2 = np.linalg.svd(cluster_list_task_3_a_good_2, full_matrices = False)
    
    #B good task 3
    u_t3_b_good_1, s_t3_b_good_1, vh_t3_b_good_1 = np.linalg.svd(cluster_list_task_3_b_good_1, full_matrices = False)    
    u_t3_b_good_2, s_t3_b_good_2, vh_t3_b_good_2 = np.linalg.svd(cluster_list_task_3_b_good_2, full_matrices = False)
    
    t_u_t1_a_good_1 = np.transpose(u_t1_a_good_1)
    t_vh_t1_a_good_1 = np.transpose(vh_t1_a_good_1)
      
    t_u_t2_a_good_1 = np.transpose(u_t2_a_good_1)
    t_vh_t2_a_good_1 = np.transpose(vh_t2_a_good_1)
    
    t_u_t3_a_good_1 = np.transpose(u_t3_a_good_1)
    t_vh_t3_a_good_1 = np.transpose(vh_t3_a_good_1)
    
    t_u_t1_b_good_1 = np.transpose(u_t1_b_good_1)
    t_vh_t1_b_good_1 = np.transpose(vh_t1_b_good_1)
      
    t_u_t2_b_good_1 = np.transpose(u_t2_b_good_1)
    t_vh_t2_b_good_1 = np.transpose(vh_t2_b_good_1)
   
    t_u_t3_b_good_1 = np.transpose(u_t3_b_good_1)
    t_vh_t3_b_good_1 = np.transpose(vh_t3_b_good_1)
    
   
    #Predict within blocks 
    s1_t1_a_from_a = np.linalg.multi_dot([t_u_t1_a_good_1, cluster_list_task_1_a_good_2, t_vh_t1_a_good_1])
    d_t1_a_from_a = s1_t1_a_from_a.diagonal()
    sum_s1_t1_a_from_a = np.cumsum(d_t1_a_from_a)/cluster_list_task_1_a_good_2.shape[0]
    
    s1_t2_a_from_a = np.linalg.multi_dot([t_u_t2_a_good_1, cluster_list_task_2_a_good_2, t_vh_t2_a_good_1])
    d_t2_a_from_a = s1_t2_a_from_a.diagonal()
    sum_s1_t2_a_from_a = np.cumsum(d_t2_a_from_a)/cluster_list_task_2_a_good_2.shape[0]
    
    s1_t3_a_from_a = np.linalg.multi_dot([t_u_t3_a_good_1, cluster_list_task_3_a_good_2, t_vh_t3_a_good_1])
    d_t3_a_from_a = s1_t3_a_from_a.diagonal()
    sum_s1_t3_a_from_a = np.cumsum(d_t3_a_from_a)/cluster_list_task_3_a_good_2.shape[0]
    
    
    s1_t1_b_from_b = np.linalg.multi_dot([t_u_t1_b_good_1, cluster_list_task_1_b_good_2, t_vh_t1_b_good_1])
    d_t1_b_from_b = s1_t1_b_from_b.diagonal()
    sum_s1_t1_b_from_b = np.cumsum(d_t1_b_from_b)/cluster_list_task_1_b_good_2.shape[0]
    
    s1_t2_b_from_b = np.linalg.multi_dot([t_u_t2_b_good_1, cluster_list_task_2_b_good_2, t_vh_t2_b_good_1])
    d_t2_b_from_b = s1_t2_b_from_b.diagonal()
    sum_s1_t2_b_from_b = np.cumsum(d_t2_b_from_b)/cluster_list_task_2_b_good_2.shape[0]
    
    s1_t3_a_from_a = np.linalg.multi_dot([t_u_t3_b_good_1, cluster_list_task_3_b_good_2, t_vh_t3_b_good_1])
    d_t3_b_from_b = s1_t3_a_from_a.diagonal()
    sum_s1_t3_b_from_b = np.cumsum(d_t3_b_from_b)/cluster_list_task_3_b_good_2.shape[0]
        
    average_within = np.mean([sum_s1_t1_a_from_a,sum_s1_t2_a_from_a,sum_s1_t3_a_from_a, sum_s1_t1_b_from_b,sum_s1_t2_b_from_b, sum_s1_t3_b_from_b], axis = 0)
    
    #Predict between blocks 
    s1_t1_b_from_a = np.linalg.multi_dot([t_u_t1_a_good_1, cluster_list_task_1_b_good_2, t_vh_t1_a_good_1])
    d_t1_b_from_a = s1_t1_b_from_a.diagonal()
    sum_s1_t1_b_from_a = np.cumsum(d_t1_b_from_a)/cluster_list_task_1_b_good_2.shape[0]
    
    s1_t2_b_from_a = np.linalg.multi_dot([t_u_t2_a_good_1, cluster_list_task_2_b_good_2, t_vh_t2_a_good_1])
    d_t2_b_from_a = s1_t2_b_from_a.diagonal()
    sum_s1_t2_b_from_a = np.cumsum(d_t2_b_from_a)/cluster_list_task_2_b_good_2.shape[0]
    
    s1_t3_b_from_a = np.linalg.multi_dot([t_u_t3_a_good_1, cluster_list_task_3_b_good_2, t_vh_t3_a_good_1])
    d_t3_b_from_a = s1_t3_b_from_a.diagonal()
    sum_s1_t3_b_from_a = np.cumsum(d_t3_b_from_a)/cluster_list_task_3_b_good_2.shape[0]
    
    
    s1_t1_a_from_b = np.linalg.multi_dot([t_u_t1_b_good_1, cluster_list_task_1_a_good_2, t_vh_t1_b_good_1])
    d_t1_a_from_b = s1_t1_a_from_b.diagonal()
    sum_s1_t1_a_from_b = np.cumsum(d_t1_a_from_b)/cluster_list_task_1_a_good_2.shape[0]
    
    s1_t2_a_from_b = np.linalg.multi_dot([t_u_t2_b_good_1, cluster_list_task_2_a_good_2, t_vh_t2_b_good_1])
    d_t2_a_from_b = s1_t2_a_from_b.diagonal()
    sum_s1_t2_a_from_b = np.cumsum(d_t2_a_from_b)/cluster_list_task_2_a_good_2.shape[0]
    
    s1_t3_a_from_b = np.linalg.multi_dot([t_u_t3_b_good_1, cluster_list_task_3_a_good_2, t_vh_t3_b_good_1])
    d_t3_a_from_b = s1_t3_a_from_b.diagonal()
    sum_s1_t3_a_from_b = np.cumsum(d_t3_a_from_b)/cluster_list_task_3_a_good_2.shape[0]
   
    
    average_between = np.mean([sum_s1_t1_b_from_a,sum_s1_t2_b_from_a,sum_s1_t3_b_from_a, sum_s1_t1_a_from_b,sum_s1_t2_a_from_b,sum_s1_t3_a_from_b], axis = 0)
    
    if HP == True :
        plot(average_within, label = 'A from A HP', color='black')
        plot(average_between, label = 'A from B HP', linestyle = '--', color='black')
          
    elif HP == False:
        plot(average_within, label = 'A from A PFC', color='red')
        plot(average_between, label = 'A from B PFC', linestyle = '--', color='red')
        
    legend()
            
    
    
def block_firings_rates_selection_forced(experiment, compare_a_b = False):   
    cluster_list_task_1_a_good = []
    cluster_list_task_1_b_good = []   
    cluster_list_task_2_a_good = []
    cluster_list_task_2_b_good = []
    cluster_list_task_3_a_good = []
    cluster_list_task_3_b_good = []
    
    cluster_list_task_1_a_good_choice_a = []
    cluster_list_task_2_a_good_choice_a = []    
    cluster_list_task_3_a_good_choice_a = []
    
    cluster_list_task_1_b_good_choice_a = []
    cluster_list_task_2_b_good_choice_a = []       
    cluster_list_task_3_b_good_choice_a = [] 
                        
    cluster_list_task_1_a_good_choice_b = []
    cluster_list_task_2_a_good_choice_b = []        
    cluster_list_task_3_a_good_choice_b = []  

    cluster_list_task_1_b_good_choice_b = []
    cluster_list_task_2_b_good_choice_b = []        
    cluster_list_task_3_b_good_choice_b = []
        
    for s,session in enumerate(experiment):
        if session.trial_data['block'][-1] >= 11:
            
            state_A_choice_A_t1,state_A_choice_B_t1,state_B_choice_A_t1,state_B_choice_B_t1,\
            state_A_choice_A_t2, state_A_choice_B_t2,state_B_choice_A_t2,state_B_choice_B_t2,\
            state_A_choice_A_t3, state_A_choice_B_t3, state_B_choice_A_t3, state_B_choice_B_t3, spikes = extract_session_a_b_based_on_block_forced_trials(session)
            
            if (len(state_A_choice_A_t1) > 0) & (len(state_A_choice_B_t1) > 0) & (len(state_B_choice_A_t1) > 0) &\
                (len(state_B_choice_B_t1) > 0) & (len(state_A_choice_A_t2) > 0) & (len(state_A_choice_B_t2) > 0) &\
                (len(state_B_choice_A_t2) > 0) & (len(state_B_choice_B_t2) > 0) & (len(state_A_choice_A_t3) > 0) &\
                (len(state_A_choice_B_t3) > 0) & (len(state_B_choice_A_t3) > 0) & (len(state_B_choice_B_t3) > 0):
                
                
                unique_neurons  = np.unique(spikes[0])   
                for i in range(len(unique_neurons)):                
                    mean_firing_rate_task_1_a_good_A_ch  = np.mean(state_A_choice_A_t1[:,i,:],0)
                    mean_firing_rate_task_1_a_good_B_ch  = np.mean(state_A_choice_B_t1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_B_ch  = np.mean(state_B_choice_B_t1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch  = np.mean(state_B_choice_A_t1[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch  = np.mean(state_A_choice_A_t2[:,i,:],0)
                    mean_firing_rate_task_2_a_good_B_ch  = np.mean(state_A_choice_B_t2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_B_ch  = np.mean(state_B_choice_B_t2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch  = np.mean(state_B_choice_A_t2[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch  = np.mean(state_A_choice_A_t3[:,i,:],0)
                    mean_firing_rate_task_3_a_good_B_ch  = np.mean(state_A_choice_B_t3[:,i,:],0)
                    mean_firing_rate_task_3_b_good_B_ch  = np.mean(state_B_choice_B_t3[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch  = np.mean(state_B_choice_A_t3[:,i,:],0)
                    
                    if compare_a_b == True: 
                        

                        cluster_list_task_1_a_good_choice_a.append(mean_firing_rate_task_1_a_good_A_ch)
                        cluster_list_task_2_a_good_choice_a.append(mean_firing_rate_task_2_a_good_A_ch)        
                        cluster_list_task_3_a_good_choice_a.append(mean_firing_rate_task_3_a_good_A_ch)
                        
                        cluster_list_task_1_b_good_choice_a.append(mean_firing_rate_task_1_b_good_A_ch)
                        cluster_list_task_2_b_good_choice_a.append(mean_firing_rate_task_2_b_good_A_ch)        
                        cluster_list_task_3_b_good_choice_a.append(mean_firing_rate_task_2_b_good_A_ch) 
                        
                        cluster_list_task_1_a_good_choice_b.append(mean_firing_rate_task_1_a_good_B_ch)
                        cluster_list_task_2_a_good_choice_b.append(mean_firing_rate_task_2_a_good_B_ch)        
                        cluster_list_task_3_a_good_choice_b.append(mean_firing_rate_task_3_a_good_B_ch)  
                        
                        cluster_list_task_1_b_good_choice_b.append(mean_firing_rate_task_1_b_good_B_ch)
                        cluster_list_task_2_b_good_choice_b.append(mean_firing_rate_task_2_b_good_B_ch)        
                        cluster_list_task_3_b_good_choice_b.append(mean_firing_rate_task_3_b_good_B_ch)  
                    
                    else:
                       
                        mean_firing_rate_a_task_1= np.concatenate((mean_firing_rate_task_1_a_good_A_ch,mean_firing_rate_task_1_a_good_B_ch), axis = 0)
                        mean_firing_rate_a_task_2 = np.concatenate((mean_firing_rate_task_2_a_good_A_ch,mean_firing_rate_task_2_a_good_B_ch), axis = 0)
                        mean_firing_rate_a_task_3 = np.concatenate((mean_firing_rate_task_3_a_good_A_ch,mean_firing_rate_task_3_a_good_B_ch), axis = 0)
                        
                        mean_firing_rate_b_task_1 = np.concatenate((mean_firing_rate_task_1_b_good_A_ch,mean_firing_rate_task_1_b_good_B_ch), axis = 0)
                        mean_firing_rate_b_task_2 = np.concatenate((mean_firing_rate_task_2_b_good_A_ch,mean_firing_rate_task_2_b_good_B_ch), axis = 0)
                        mean_firing_rate_b_task_3 = np.concatenate((mean_firing_rate_task_3_b_good_A_ch,mean_firing_rate_task_3_b_good_B_ch), axis = 0)
                        
                    
                        cluster_list_task_1_a_good.append(mean_firing_rate_a_task_1)
                        cluster_list_task_1_b_good.append(mean_firing_rate_b_task_1)        
                        cluster_list_task_2_a_good.append(mean_firing_rate_a_task_2)
                        cluster_list_task_2_b_good.append(mean_firing_rate_b_task_2)       
                        cluster_list_task_3_a_good.append(mean_firing_rate_a_task_3)        
                        cluster_list_task_3_b_good.append(mean_firing_rate_b_task_3)
                    
    cluster_list_task_1_a_good = np.asarray(cluster_list_task_1_a_good)
    cluster_list_task_1_b_good = np.asarray(cluster_list_task_1_b_good)       
    cluster_list_task_2_a_good = np.asarray(cluster_list_task_2_a_good)
    cluster_list_task_2_b_good = np.asarray(cluster_list_task_2_b_good)        
    cluster_list_task_3_a_good = np.asarray(cluster_list_task_3_a_good)
    cluster_list_task_3_b_good = np.asarray(cluster_list_task_3_b_good)
    
    cluster_list_task_1_a_good_choice_a = np.asarray(cluster_list_task_1_a_good_choice_a)
    cluster_list_task_2_a_good_choice_a = np.asarray(cluster_list_task_2_a_good_choice_a)
    cluster_list_task_3_a_good_choice_a = np.asarray(cluster_list_task_3_a_good_choice_a)
    
    cluster_list_task_1_b_good_choice_a = np.asarray(cluster_list_task_1_b_good_choice_a)
    cluster_list_task_2_b_good_choice_a = np.asarray(cluster_list_task_2_b_good_choice_a) 
    cluster_list_task_3_b_good_choice_a = np.asarray(cluster_list_task_3_b_good_choice_a)
    
    cluster_list_task_1_a_good_choice_b = np.asarray(cluster_list_task_1_a_good_choice_b)
    cluster_list_task_2_a_good_choice_b = np.asarray(cluster_list_task_2_a_good_choice_b)
    cluster_list_task_3_a_good_choice_b = np.asarray(cluster_list_task_3_a_good_choice_b)
    
    cluster_list_task_1_b_good_choice_b = np.asarray(cluster_list_task_1_b_good_choice_b)
    cluster_list_task_2_b_good_choice_b = np.asarray(cluster_list_task_2_b_good_choice_b)
    cluster_list_task_3_b_good_choice_b = np.asarray(cluster_list_task_3_b_good_choice_b) 
    
    if compare_a_b == True: 
        return cluster_list_task_1_a_good_choice_a, cluster_list_task_2_a_good_choice_a ,\
        cluster_list_task_3_a_good_choice_a, cluster_list_task_1_b_good_choice_a,\
        cluster_list_task_2_b_good_choice_a, cluster_list_task_3_b_good_choice_a,\
        cluster_list_task_1_a_good_choice_b, cluster_list_task_2_a_good_choice_b,\
        cluster_list_task_3_a_good_choice_b, cluster_list_task_1_b_good_choice_b,\
        cluster_list_task_2_b_good_choice_b, cluster_list_task_3_b_good_choice_b
    else:
        return cluster_list_task_1_a_good, cluster_list_task_1_b_good,cluster_list_task_2_a_good,\
    cluster_list_task_2_b_good, cluster_list_task_3_a_good, cluster_list_task_3_b_good
        
def svd_plotting_block_analysis_forced(experiment, plot_HP = True, compare_a_b = False):
    
    if compare_a_b == False:
        # To check if a in task 1 is more similar a in task 2 than b in task 2 
        cluster_list_task_1_a_good, cluster_list_task_1_b_good, cluster_list_task_2_a_good,\
        cluster_list_task_2_b_good, cluster_list_task_3_a_good, cluster_list_task_3_b_good = block_firings_rates_selection_forced(experiment,compare_a_b = compare_a_b)
        
    
        #SVDsu.shape, s.shape, vh.shape for task 1
        u_t1_a_good, s_t1_a_good, vh_t1_a_good = np.linalg.svd(cluster_list_task_1_a_good, full_matrices = False)
        u_t1_b_good, s_t1_b_good, vh_t1_b_good = np.linalg.svd(cluster_list_task_1_b_good, full_matrices = False)
    
        #SVDsu.shape, s.shape, vh.shape for task 2
        u_t2_a_good, s_t2_a_good, vh_t2_a_good = np.linalg.svd(cluster_list_task_2_a_good, full_matrices = False)
        u_t2_b_good, s_t2_b_good, vh_t2_b_good = np.linalg.svd(cluster_list_task_2_b_good, full_matrices = False)
    
        #SVDsu.shape, s.shape, vh.shape for task 3 
        u_t3_a_good, s_t3_a_good, vh_t3_a_good = np.linalg.svd(cluster_list_task_3_a_good, full_matrices = False)
        u_t3_b_good, s_t3_b_good, vh_t3_b_good = np.linalg.svd(cluster_list_task_3_b_good, full_matrices = False)
    
        t_u_t1_a_good = np.transpose(u_t1_a_good)
        t_v_t1_a_good = np.transpose(vh_t1_a_good)
        
        t_u_t1_b_good = np.transpose(u_t1_b_good)
        t_v_t1_b_good = np.transpose(vh_t1_b_good)
        
        t_u_t2_a_good = np.transpose(u_t2_a_good)
        t_v_t2_a_good = np.transpose(vh_t2_a_good)
        
        t_u_t2_b_good = np.transpose(u_t2_b_good)
        t_v_t2_b_good = np.transpose(vh_t2_b_good)
        
        ## Predict A from B and B from A
        
        #Compare block a task 2 from block b task 1
        s_task_2_a_from_b = np.linalg.multi_dot([t_u_t1_b_good, cluster_list_task_2_a_good, t_v_t1_b_good])
        s_task_2_a_from_b_d = s_task_2_a_from_b.diagonal()
        sum_s_task_2_a_good_from_b_good_task_2 = np.cumsum(s_task_2_a_from_b_d)/cluster_list_task_2_a_good.shape[0]
        
        #Compare block a task 3 from block b task 2
        s_task_3_a_from_b = np.linalg.multi_dot([t_u_t2_b_good, cluster_list_task_3_a_good, t_v_t2_b_good])
        s_task_3_a_from_b_d = s_task_3_a_from_b.diagonal()
        sum_s_task_3_a_good_from_b_good_task_3 = np.cumsum(s_task_3_a_from_b_d)/cluster_list_task_3_a_good.shape[0]
        
        
        #Compare block b task 2 from block a task 1
        s_task_2_b_from_a = np.linalg.multi_dot([t_u_t1_a_good, cluster_list_task_1_b_good, t_v_t1_a_good])
        s_task_2_b_from_a_d = s_task_2_b_from_a.diagonal()
        sum_s_task_2_b_good_from_a_good_task_1 = np.cumsum(s_task_2_b_from_a_d)/cluster_list_task_1_b_good.shape[0]
        
        #Compare block b task 3 from block a task 3
        s_task_3_b_from_a = np.linalg.multi_dot([t_u_t2_a_good, cluster_list_task_3_b_good, t_v_t2_a_good])
        s_task_3_b_from_a_d = s_task_3_b_from_a.diagonal()
        sum_s_task_3_b_good_from_a_good_task_2 = np.cumsum(s_task_3_b_from_a_d)/cluster_list_task_3_b_good.shape[0]
        
        
        ## Predict A from A and B from B 
        
        #Compare block b task 2 from block b task 1
        s_task_2_b_from_b_t1 = np.linalg.multi_dot([t_u_t1_b_good, cluster_list_task_2_b_good, t_v_t1_b_good])
        s_task_2_b_from_b_d_t1 = s_task_2_b_from_b_t1.diagonal()
        sum_s_task_2_b_good_from_b_good_task_1 = np.cumsum(s_task_2_b_from_b_d_t1)/cluster_list_task_2_b_good.shape[0]
        
        #Compare block b task 3 from block b task 2
        s_task_3_b_from_b_t2= np.linalg.multi_dot([t_u_t2_b_good, cluster_list_task_3_b_good, t_v_t2_b_good])
        s_task_3_b_from_b_d_t2 = s_task_3_b_from_b_t2.diagonal()
        sum_s_task_3_b_good_from_b_good_task_2 = np.cumsum(s_task_3_b_from_b_d_t2)/cluster_list_task_3_b_good.shape[0]
        
        #Compare block a task 2 from block a task 1
        s_task_2_a_from_a_t1 = np.linalg.multi_dot([t_u_t1_a_good, cluster_list_task_2_a_good, t_v_t1_a_good])
        s_task_2_a_from_a_d_t1 = s_task_2_a_from_a_t1.diagonal()
        sum_s_task_2_a_good_from_a_good_task_1 = np.cumsum(s_task_2_a_from_a_d_t1)/cluster_list_task_2_a_good.shape[0]
        
        #Compare block a task 3 from block a task 2
        s_task_3_a_from_a_t2= np.linalg.multi_dot([t_u_t2_a_good, cluster_list_task_3_a_good, t_v_t2_a_good])
        s_task_3_a_from_a_d_t2 = s_task_3_a_from_a_t2.diagonal()
        sum_s_task_3_a_good_from_a_good_task_2 = np.cumsum(s_task_3_a_from_a_d_t2)/cluster_list_task_3_a_good.shape[0]
        
        average_a_to_a = np.mean([sum_s_task_2_a_good_from_a_good_task_1,sum_s_task_3_a_good_from_a_good_task_2], axis = 0)
        average_b_to_b = np.mean([sum_s_task_2_b_good_from_b_good_task_1,sum_s_task_3_b_good_from_b_good_task_2], axis = 0)
        average_within_block = np.mean([average_a_to_a,average_b_to_b], axis = 0) 
        
        average_a_to_b = np.mean([sum_s_task_2_b_good_from_a_good_task_1,sum_s_task_3_b_good_from_a_good_task_2], axis = 0)
        average_b_to_a = np.mean([sum_s_task_2_a_good_from_b_good_task_2,sum_s_task_3_a_good_from_b_good_task_3],axis = 0)
        average_between_block = np.mean([average_a_to_b,average_b_to_a], axis = 0) 
    
        #plot(average_within_block, label = 'Explain A block from A block PFC')
        #plot(average_between_block, label = 'Explain A block from B block PFC')
    
        if plot_HP == True :
            plot(average_a_to_a, label = 'Explain A from A HP', color='black')
            plot(average_a_to_b, label = 'Explain B from A HP', linestyle = '--', color='black')
        elif plot_HP == False:
            plot(average_a_to_a, label = 'Explain A from A PFC', color='red')
            plot(average_a_to_b, label = 'Explain B from A HP', linestyle = '--', color='red')

        legend()
    
    if compare_a_b == True:
        
       #SVDs to look whether a trials when a was good are more similar to a trials when a was bad than to b trials when b was good 
       cluster_list_task_1_a_good_choice_a, cluster_list_task_2_a_good_choice_a,\
       cluster_list_task_3_a_good_choice_a, cluster_list_task_1_b_good_choice_a,\
       cluster_list_task_2_b_good_choice_a, cluster_list_task_3_b_good_choice_a,\
       cluster_list_task_1_a_good_choice_b, cluster_list_task_2_a_good_choice_b,\
       cluster_list_task_3_a_good_choice_b, cluster_list_task_1_b_good_choice_b,\
       cluster_list_task_2_b_good_choice_b, cluster_list_task_3_b_good_choice_b = block_firings_rates_selection_forced(experiment, compare_a_b = compare_a_b)
        
       all_a_when_a_good = np.concatenate((cluster_list_task_1_a_good_choice_a, cluster_list_task_2_a_good_choice_a, cluster_list_task_3_a_good_choice_a), axis = 1)
       all_a_when_a_bad = np.concatenate((cluster_list_task_1_b_good_choice_a, cluster_list_task_2_b_good_choice_a, cluster_list_task_3_b_good_choice_a), axis = 1)

       all_b_when_b_good = np.concatenate((cluster_list_task_1_b_good_choice_b, cluster_list_task_2_b_good_choice_b, cluster_list_task_3_b_good_choice_b), axis = 1)
       all_b_when_a_bad = np.concatenate((cluster_list_task_1_a_good_choice_b, cluster_list_task_2_a_good_choice_b, cluster_list_task_3_a_good_choice_b), axis = 1)

       
       #SVDs A to A in another state vs A to B in the same state
       u_a_when_a_good, s_a_when_a_good, vh_a_when_a_good = np.linalg.svd(all_a_when_a_good, full_matrices = False)
       
       #SVDs B to B in another state vs A to B in the same state   
       u_b_when_b_good, s_b_when_b_good, vh_a_when_b_good = np.linalg.svd(all_b_when_b_good, full_matrices = False)
         
       t_u_a_when_a_good = np.transpose(u_a_when_a_good)
       t_vh_a_when_a_good = np.transpose(vh_a_when_a_good)
        
       
       
       #Explain A bad with A good
       s_a_bad_a_good = np.linalg.multi_dot([t_u_a_when_a_good, all_a_when_a_bad, t_vh_a_when_a_good])
       diag_a_bad_from_a_good = s_a_bad_a_good.diagonal()
       sum_a_bad_from_a_good = np.cumsum(diag_a_bad_from_a_good)/all_a_when_a_bad.shape[0]
       
       #Explain B good with A good
       s_b_good_a_good = np.linalg.multi_dot([t_u_a_when_a_good, all_b_when_b_good, t_vh_a_when_a_good])
       diag_b_good_from_a_good = s_b_good_a_good.diagonal()
       sum_b_good_from_a_good = np.cumsum(diag_b_good_from_a_good)/all_a_when_a_bad.shape[0]
       
       

       #SVDs for a in good and bad blocks in task 1
       u_t1_a_good_ch_A, s_t1_a_good_ch_A, vh_t1_a_good_ch_A = np.linalg.svd(cluster_list_task_1_a_good_choice_a, full_matrices = False)

       #SVDs for a in good and bad blocks in task 2
       u_t2_a_good_ch_A, s_t2_a_good_ch_A, vh_t2_a_good_ch_A = np.linalg.svd(cluster_list_task_2_a_good_choice_a, full_matrices = False)
       
       #SVDs for a in good and bad blocks in task 3
       u_t3_a_good_ch_A, s_t3_a_good_ch_A, vh_t3_a_good_ch_A = np.linalg.svd(cluster_list_task_3_a_good_choice_a, full_matrices = False)
       
       
       t_u_t1_a_good_ch_A = np.transpose(u_t1_a_good_ch_A)
       t_v_t1_a_good_ch_A = np.transpose(vh_t1_a_good_ch_A)
        
       t_u_t2_a_good_ch_A = np.transpose(u_t2_a_good_ch_A)
       t_v_t2_a_good_ch_A = np.transpose(vh_t2_a_good_ch_A)
        
       
       ## Predict A from A in a different task
       #Compare a choices in good a task 2 from task 1 in a good block 
       s_task_2_a_good_from_a_task_1 = np.linalg.multi_dot([t_u_t1_a_good_ch_A, cluster_list_task_2_a_good_choice_a, t_v_t1_a_good_ch_A])
       task_2_a_good_from_a_task_1 = s_task_2_a_good_from_a_task_1.diagonal()
       sum_s_task_2_a_good_from_a_good_task_1 = np.cumsum(task_2_a_good_from_a_task_1)/cluster_list_task_2_a_good_choice_a.shape[0]
       
       #Compare a choices in good a task 3 from task 2 in a good block 
       s_task_3_a_good_from_a_task_2 = np.linalg.multi_dot([t_u_t2_a_good_ch_A, cluster_list_task_3_a_good_choice_a, t_v_t2_a_good_ch_A])
       task_3_a_good_from_a_task_2 = s_task_3_a_good_from_a_task_2.diagonal()
       sum_s_task_3_a_good_from_a_good_task_2 = np.cumsum(task_3_a_good_from_a_task_2)/cluster_list_task_3_a_good_choice_a.shape[0]
       

       ## Predict A from A in a B block in the same task
       #Compare a choices in good a task 1 from task 1 in b good block 
       s_task_1_a_bad_from_a_task_1 = np.linalg.multi_dot([t_u_t1_a_good_ch_A, cluster_list_task_2_b_good_choice_a, t_v_t1_a_good_ch_A])
       task_1_a_bad_from_a_task_1 = s_task_1_a_bad_from_a_task_1.diagonal()
       sum_s_task_1_a_bad_from_a_good_task_1 = np.cumsum(task_1_a_bad_from_a_task_1)/cluster_list_task_1_b_good_choice_a.shape[0]
       
       #Compare a choices in good a task 2 from task 2 in b good block 
       s_task_2_a_bad_from_a_task_2 = np.linalg.multi_dot([t_u_t2_a_good_ch_A, cluster_list_task_3_b_good_choice_a, t_v_t2_a_good_ch_A])
       task_2_a_bad_from_a_task_2 = s_task_2_a_bad_from_a_task_2.diagonal()
       sum_s_task_2_a_bad_from_a_good_task_2 = np.cumsum(task_2_a_bad_from_a_task_2)/cluster_list_task_3_b_good_choice_a.shape[0]
       
            
       average_a_in_good_a = np.mean([sum_s_task_2_a_good_from_a_good_task_1,sum_s_task_3_a_good_from_a_good_task_2], axis = 0)
       average_a_in_bad_a = np.mean([sum_s_task_1_a_bad_from_a_good_task_1,sum_s_task_2_a_bad_from_a_good_task_2], axis = 0)
        
       if plot_HP == True :
            plot(average_a_in_good_a, label = 'A from A in A good Block HP', color='black')
            plot(average_a_in_bad_a, label = 'A from A in A bad Block HP', linestyle = '--', color='black')
          
       elif plot_HP == False:
            plot(average_a_in_good_a, label = 'A from A in A good Block PFC', color='red')
            plot(average_a_in_bad_a, label = 'A from A in A bad Block PFC', linestyle = '--', color='red')
            
           
       legend()
       
       
       
       
       
       
       
       
       
       