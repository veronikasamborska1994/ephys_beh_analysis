#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:28:27 2019

@author: veronikasamborska
"""
import forced_trials_extract_data as ft
import numpy as np 
import matplotlib.pyplot as plt



def extract_session_a_b_based_on_block_forced_trials(session, tasks_unchanged = False):
    spikes = session.ephys
    spikes = spikes[:,~np.isnan(spikes[1,:])] 
    aligned_rates = session.aligned_rates_forced
    
    forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]
    
    task = session.trial_data['task']
    task_forced = task[forced_array]
    
    task_1 = len(np.where(task_forced == 1)[0])
    task_2 = len(np.where(task_forced == 2)[0])
    #Getting choice indices 
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
    predictor_a_good_task_1_cut = predictor_a_good_task_1[:task_1]
    
    #reward_task_1_cut = reward[:task_1]
          
    predictor_A_Task_2_cut = predictor_A_Task_2[task_1:task_1+task_2]
    predictor_a_good_task_2_cut = predictor_a_good_task_2[task_1:task_1+task_2]

    #reward_task_2_cut = reward[task_1:task_1+task_2]
          
    predictor_A_Task_3_cut = predictor_A_Task_3[task_1+task_2:len(task_forced)]
    predictor_a_good_task_3_cut = predictor_a_good_task_3[task_1+task_2:len(task_forced)]

    #reward_task_3_cut = reward[task_1+task_2:]
    
   
    state_A_choice_A_t1 = aligned_rates_task_1[np.where((predictor_a_good_task_1_cut ==1) & (predictor_A_Task_1_cut == 1 ))]
    state_A_choice_B_t1 = aligned_rates_task_1[np.where((predictor_a_good_task_1_cut ==1) & (predictor_A_Task_1_cut == 0))]
    
    state_B_choice_A_t1 = aligned_rates_task_1[np.where((predictor_a_good_task_1_cut == 0) & (predictor_A_Task_1_cut == 1 ))]
    state_B_choice_B_t1 = aligned_rates_task_1[np.where((predictor_a_good_task_1_cut == 0) & (predictor_A_Task_1_cut == 0))]
    
    state_A_choice_A_t2 = aligned_rates_task_2[np.where((predictor_a_good_task_2_cut ==1) & (predictor_A_Task_2_cut == 1 ))]
    state_A_choice_B_t2 = aligned_rates_task_2[np.where((predictor_a_good_task_2_cut ==1) & (predictor_A_Task_2_cut == 0))]
    
    state_B_choice_A_t2 = aligned_rates_task_2[np.where((predictor_a_good_task_2_cut == 0) & (predictor_A_Task_2_cut == 1 ))]
    state_B_choice_B_t2 = aligned_rates_task_2[np.where((predictor_a_good_task_2_cut == 0) & (predictor_A_Task_2_cut == 0))]

    state_A_choice_A_t3 = aligned_rates_task_3[np.where((predictor_a_good_task_3_cut ==1) & (predictor_A_Task_3_cut == 1 ))]
    state_A_choice_B_t3 = aligned_rates_task_3[np.where((predictor_a_good_task_3_cut ==1) & (predictor_A_Task_3_cut == 0))]
    
    state_B_choice_A_t3 = aligned_rates_task_3[np.where((predictor_a_good_task_3_cut == 0) & (predictor_A_Task_3_cut == 1 ))]
    state_B_choice_B_t3 = aligned_rates_task_3[np.where((predictor_a_good_task_3_cut == 0) & (predictor_A_Task_3_cut == 0))]

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
            (len(state_A_choice_B_t3_2) > 0) & (len(state_B_choice_A_t3_2) > 0) & (len(state_B_choice_B_t3_2) > 0) &\
            (session.file_name != 'm479-2018-08-20-112813.txt') & (session.file_name != 'm483-2018-06-22-160006.txt') &\
            (session.file_name != 'm478-2018-08-09-120322.txt') &  (session.file_name != 'm486-2018-07-28-171910.txt') &\
            (session.file_name != 'm486-2018-07-16-170101.txt') & (session.file_name != 'm480-2018-08-01-164435.txt') &\
            (session.file_name != 'm480-2018-08-02-170827.txt') & (session.file_name != 'm480-2018-09-04-150501.txt') &\
            (session.file_name != 'm480-2018-08-22-111012.txt') & (session.file_name != 'm481-2018-06-28-160517.txt'):
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
            else:
                print(session.file_name)
                    
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
        plt.plot(average_within, label = 'A from A HP Within Tasks', color = 'black')
        plt.plot(average_between, label = 'A from B HP Between Tasks', linestyle = '--', color='black')
          
    elif HP == False:
        plt.plot(average_within, label = 'A from A PFC Within Tasks', color = 'red')
        plt.plot(average_between, label = 'A from B PFC Between Tasks', linestyle = '--', color='red')
        
    plt.legend()
            

def block_firings_rates_selection_forced_split_in_half_only_a(experiment):   
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
            state_B_choice_A_t1_1 = state_B_choice_A_t1[:int(len(state_B_choice_A_t1)/2)]
            state_A_choice_A_t2_1 = state_A_choice_A_t2[:int(len(state_A_choice_A_t2)/2)]
            state_B_choice_A_t2_1 = state_B_choice_A_t2[:int(len(state_B_choice_A_t2)/2)]
            state_A_choice_A_t3_1 = state_A_choice_A_t3[:int(len(state_A_choice_A_t3)/2)]
            state_B_choice_A_t3_1 = state_B_choice_A_t3[:int(len(state_B_choice_A_t3)/2)]
            
            state_A_choice_A_t1_2 = state_A_choice_A_t1[int(len(state_A_choice_A_t1)/2):]
            state_B_choice_A_t1_2 = state_B_choice_A_t1[int(len(state_B_choice_A_t1)/2):]
            state_A_choice_A_t2_2 = state_A_choice_A_t2[int(len(state_A_choice_A_t2)/2):]
            state_B_choice_A_t2_2 = state_B_choice_A_t2[int(len(state_B_choice_A_t2)/2):]
            state_A_choice_A_t3_2 = state_A_choice_A_t3[int(len(state_A_choice_A_t3)/2):]
            state_B_choice_A_t3_2 = state_B_choice_A_t3[int(len(state_B_choice_A_t3)/2):]
                
            if (len(state_A_choice_A_t1_1) > 0) & (len(state_B_choice_A_t1_1) > 0)& (len(state_A_choice_A_t2_1) > 0) &\
            (len(state_B_choice_A_t2_1) > 0) & (len(state_A_choice_A_t3_1) > 0) &\
            (len(state_B_choice_A_t3_1) > 0) & (len(state_A_choice_A_t1_2) > 0) &\
            (len(state_B_choice_A_t1_2) > 0) & (len(state_A_choice_A_t2_2) > 0) &\
            (len(state_B_choice_A_t2_2) > 0) & (len(state_A_choice_A_t3_2) > 0) &\
            (len(state_B_choice_A_t3_2) > 0):    
                unique_neurons  = np.unique(spikes[0])  
                
                for i in range(len(unique_neurons)):                
                    mean_firing_rate_task_1_a_good_A_ch_1  = np.mean(state_A_choice_A_t1_1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch_1  = np.mean(state_B_choice_A_t1_1[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch_1  = np.mean(state_A_choice_A_t2_1[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch_1  = np.mean(state_B_choice_A_t2_1[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch_1  = np.mean(state_A_choice_A_t3_1[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch_1  = np.mean(state_B_choice_A_t3_1[:,i,:],0)
                    
                    
                    mean_firing_rate_task_1_a_good_A_ch_2  = np.mean(state_A_choice_A_t1_2[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch_2  = np.mean(state_B_choice_A_t1_2[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch_2  = np.mean(state_A_choice_A_t2_2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch_2  = np.mean(state_B_choice_A_t2_2[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch_2  = np.mean(state_A_choice_A_t3_2[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch_2  = np.mean(state_B_choice_A_t3_2[:,i,:],0)
                    
                    
                    cluster_list_task_1_a_good_1.append(mean_firing_rate_task_1_a_good_A_ch_1)
                    cluster_list_task_1_b_good_1.append(mean_firing_rate_task_1_b_good_A_ch_1)   
                    cluster_list_task_2_a_good_1.append(mean_firing_rate_task_2_a_good_A_ch_1)
                    cluster_list_task_2_b_good_1.append(mean_firing_rate_task_2_b_good_A_ch_1)
                    cluster_list_task_3_a_good_1.append(mean_firing_rate_task_3_a_good_A_ch_1)
                    cluster_list_task_3_b_good_1.append(mean_firing_rate_task_3_b_good_A_ch_1)
                    
                    cluster_list_task_1_a_good_2.append(mean_firing_rate_task_1_a_good_A_ch_2)
                    cluster_list_task_1_b_good_2.append(mean_firing_rate_task_1_b_good_A_ch_2)   
                    cluster_list_task_2_a_good_2.append(mean_firing_rate_task_2_a_good_A_ch_2)
                    cluster_list_task_2_b_good_2.append(mean_firing_rate_task_2_b_good_A_ch_2)
                    cluster_list_task_3_a_good_2.append(mean_firing_rate_task_3_a_good_A_ch_2)
                    cluster_list_task_3_b_good_2.append(mean_firing_rate_task_3_b_good_A_ch_2)
                    
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
    
    
def plot_blocks_split_intwo_between_blocks(experiment, HP = False):
    
    #Explain A state from A state vs B state within a task 
    
    cluster_list_task_1_a_good_1, cluster_list_task_1_b_good_1,\
    cluster_list_task_2_a_good_1, cluster_list_task_2_b_good_1,\
    cluster_list_task_3_a_good_1, cluster_list_task_3_b_good_1, cluster_list_task_1_a_good_2,\
    cluster_list_task_1_b_good_2, cluster_list_task_2_a_good_2, cluster_list_task_2_b_good_2,\
    cluster_list_task_3_a_good_2, cluster_list_task_3_b_good_2  = block_firings_rates_selection_forced_split_in_half_only_a(experiment)
    
       
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
    
    t_u_t1_b_good_1 = np.transpose(u_t1_b_good_1)
    t_vh_t1_b_good_1 = np.transpose(vh_t1_b_good_1)
      
    t_u_t2_b_good_1 = np.transpose(u_t2_b_good_1)
    t_vh_t2_b_good_1 = np.transpose(vh_t2_b_good_1)
   
    t_u_t1_a_good_2 = np.transpose(u_t1_a_good_2)
    t_vh_t1_a_good_2 = np.transpose(vh_t1_a_good_2)
      
    t_u_t2_a_good_2 = np.transpose(u_t2_a_good_2)
    t_vh_t2_a_good_2 = np.transpose(vh_t2_a_good_2)
    
    t_u_t1_b_good_2 = np.transpose(u_t1_b_good_2)
    t_vh_t1_b_good_2 = np.transpose(vh_t1_b_good_2)
      
    t_u_t2_b_good_2 = np.transpose(u_t2_b_good_2)
    t_vh_t2_b_good_2 = np.transpose(vh_t2_b_good_2)
   
  
    
    #Predict within blocks between tasks A good   
    # Predict one half of a good in task 2 from a good in task 1
    s1_t2_a_from_a_t1 = np.linalg.multi_dot([t_u_t1_a_good_2, cluster_list_task_2_a_good_1, t_vh_t1_a_good_2])
    d_t2_a_from_a_t1 = s1_t2_a_from_a_t1.diagonal()
    sum_s1_t2_a_from_a_t1 = np.cumsum(d_t2_a_from_a_t1)/cluster_list_task_2_a_good_1.shape[0]
    
    #Predict one half of a good in task 3 from a good in task 2
    s1_t3_a_from_a_t2 = np.linalg.multi_dot([t_u_t2_a_good_2, cluster_list_task_3_a_good_1, t_vh_t2_a_good_2])
    d_t3_a_from_a_t2 = s1_t3_a_from_a_t2.diagonal()
    sum_s1_t3_a_from_a_t2 = np.cumsum(d_t3_a_from_a_t2)/cluster_list_task_3_a_good_1.shape[0]
    
    # Predict second half of a good in task 2 from a good in task 1
    s2_t2_a_from_a_t1 = np.linalg.multi_dot([t_u_t1_a_good_1, cluster_list_task_2_a_good_2, t_vh_t1_a_good_1])
    d2_t2_a_from_a_t1 = s2_t2_a_from_a_t1.diagonal()
    sum_s2_t2_a_from_a_t1 = np.cumsum(d2_t2_a_from_a_t1)/cluster_list_task_2_a_good_2.shape[0]
    
    #Predict second half of a good in task 3 from a good in task 2
    s2_t3_a_from_a_t2 = np.linalg.multi_dot([t_u_t2_a_good_1, cluster_list_task_3_a_good_2, t_vh_t2_a_good_1])
    d2_t3_a_from_a_t2 = s2_t3_a_from_a_t2.diagonal()
    sum_s2_t3_a_from_a_t2 = np.cumsum(d2_t3_a_from_a_t2)/cluster_list_task_3_a_good_2.shape[0]
    
    #Predict within blocks between tasks B good   

    # Predict one half of a good in task 2 from a good in task 1
    s1_t2_b_from_b_t1 = np.linalg.multi_dot([t_u_t1_b_good_2, cluster_list_task_2_b_good_1, t_vh_t1_b_good_2])
    d_t2_b_from_b_t1 = s1_t2_b_from_b_t1.diagonal()
    sum_s1_t2_b_from_b_t1 = np.cumsum(d_t2_b_from_b_t1)/cluster_list_task_2_b_good_1.shape[0]
    
    #Predict one half of a good in task 3 from a good in task 2
    s1_t3_b_from_b_t2 = np.linalg.multi_dot([t_u_t2_b_good_2, cluster_list_task_3_b_good_1, t_vh_t2_b_good_2])
    d_t3_b_from_b_t2 = s1_t3_b_from_b_t2.diagonal()
    sum_s1_t3_b_from_b_t2 = np.cumsum(d_t3_b_from_b_t2)/cluster_list_task_3_b_good_1.shape[0]
    
    # Predict second half of a good in task 2 from a good in task 1
    s2_t2_b_from_b_t1 = np.linalg.multi_dot([t_u_t1_b_good_1, cluster_list_task_2_b_good_2, t_vh_t1_b_good_1])
    d2_t2_b_from_b_t1 = s2_t2_b_from_b_t1.diagonal()
    sum_s2_t2_b_from_b_t1 = np.cumsum(d2_t2_b_from_b_t1)/cluster_list_task_2_b_good_2.shape[0]
    
    #Predict second half of a good in task 3 from a good in task 2
    s2_t3_b_from_b_t2 = np.linalg.multi_dot([t_u_t2_b_good_1, cluster_list_task_3_b_good_2, t_vh_t2_b_good_1])
    d2_t3_b_from_b_t2 = s2_t3_b_from_b_t2.diagonal()
    sum_s2_t3_b_from_b_t2 = np.cumsum(d2_t3_b_from_b_t2)/cluster_list_task_3_b_good_2.shape[0]
    
    average_within_block_between_task = np.mean([sum_s1_t2_a_from_a_t1,sum_s1_t3_a_from_a_t2,\
                                                 sum_s2_t2_a_from_a_t1,sum_s2_t3_a_from_a_t2,\
                                                 sum_s1_t2_b_from_b_t1,sum_s1_t3_b_from_b_t2,\
                                                 sum_s2_t2_b_from_b_t1,sum_s2_t3_b_from_b_t2], axis = 0 )
    
    #Predict between blocks between tasks A good  from B good  
    # Predict one half of a good in task 2 from b good in task 1
    s1_t2_a_from_b_t1 = np.linalg.multi_dot([t_u_t1_b_good_2, cluster_list_task_2_a_good_1, t_vh_t1_b_good_2])
    d_t2_a_from_b_t1 = s1_t2_a_from_b_t1.diagonal()
    sum_s1_t2_a_from_b_t1 = np.cumsum(d_t2_a_from_b_t1)/cluster_list_task_2_a_good_1.shape[0]
    
    #Predict one half of a good in task 3 from b good in task 2
    s1_t3_a_from_b_t2 = np.linalg.multi_dot([t_u_t2_b_good_2, cluster_list_task_3_a_good_1, t_vh_t2_b_good_2])
    d_t3_a_from_b_t2 = s1_t3_a_from_b_t2.diagonal()
    sum_s1_t3_a_from_b_t2 = np.cumsum(d_t3_a_from_b_t2)/cluster_list_task_3_a_good_1.shape[0]
    
    # Predict second half of a good in task 2 from b good in task 1
    s2_t2_a_from_b_t1 = np.linalg.multi_dot([t_u_t1_b_good_1, cluster_list_task_2_a_good_2, t_vh_t1_b_good_1])
    d2_t2_a_from_b_t1 = s2_t2_a_from_b_t1.diagonal()
    sum_s2_t2_a_from_b_t1 = np.cumsum(d2_t2_a_from_b_t1)/cluster_list_task_2_a_good_2.shape[0]
    
    #Predict second half of a good in task 3 from b good in task 2
    s2_t3_a_from_b_t2 = np.linalg.multi_dot([t_u_t2_b_good_1, cluster_list_task_3_a_good_2, t_vh_t2_b_good_1])
    d2_t3_a_from_b_t2 = s2_t3_a_from_b_t2.diagonal()
    sum_s2_t3_a_from_b_t2 = np.cumsum(d2_t3_a_from_b_t2)/cluster_list_task_3_a_good_2.shape[0]
    
    #Predict between blocks between tasks B good from A good   

    # Predict one half of b good in task 2 from a good in task 1
    s1_t2_b_from_a_t1 = np.linalg.multi_dot([t_u_t1_a_good_2, cluster_list_task_2_b_good_1, t_vh_t1_a_good_2])
    d_t2_b_from_a_t1 = s1_t2_b_from_a_t1.diagonal()
    sum_s1_t2_b_from_a_t1 = np.cumsum(d_t2_b_from_a_t1)/cluster_list_task_2_b_good_1.shape[0]
    
    #Predict one half of b good in task 3 from a good in task 2
    s1_t3_b_from_a_t2 = np.linalg.multi_dot([t_u_t2_a_good_2, cluster_list_task_3_b_good_1, t_vh_t2_a_good_2])
    d_t3_b_from_a_t2 = s1_t3_b_from_a_t2.diagonal()
    sum_s1_t3_b_from_a_t2 = np.cumsum(d_t3_b_from_a_t2)/cluster_list_task_3_b_good_1.shape[0]
    
    # Predict second half of b good in task 2 from a good in task 1
    s2_t2_b_from_a_t1 = np.linalg.multi_dot([t_u_t1_a_good_1, cluster_list_task_2_b_good_2, t_vh_t1_a_good_1])
    d2_t2_b_from_a_t1 = s2_t2_b_from_a_t1.diagonal()
    sum_s2_t2_b_from_a_t1 = np.cumsum(d2_t2_b_from_a_t1)/cluster_list_task_2_b_good_2.shape[0]
    
    #Predict second half of b good in task 3 from a good in task 2
    s2_t3_b_from_a_t2 = np.linalg.multi_dot([t_u_t2_a_good_1, cluster_list_task_3_b_good_2, t_vh_t2_a_good_1])
    d2_t3_b_from_a_t2 = s2_t3_b_from_a_t2.diagonal()
    sum_s2_t3_b_from_a_t2 = np.cumsum(d2_t3_b_from_a_t2)/cluster_list_task_3_b_good_2.shape[0]
    
    average_between_block_between_task = np.mean([sum_s1_t2_a_from_b_t1,sum_s1_t3_a_from_b_t2,\
                                                 sum_s2_t2_a_from_b_t1,sum_s2_t3_a_from_b_t2,\
                                                 sum_s1_t2_b_from_a_t1,sum_s1_t3_b_from_a_t2,\
                                                 sum_s2_t2_b_from_a_t1,sum_s2_t3_b_from_a_t2], axis = 0)
    
    if HP == True :
        plt.plot(average_within_block_between_task, label = 'A from A HP Within Tasks', color = 'grey')
        plt.plot(average_between_block_between_task, label = 'A from B HP Between Tasks', linestyle = '--', color='grey')
          
    elif HP == False:
        plt.plot(average_within_block_between_task, label = 'A from A PFC Within Tasks', color = 'pink')
        plt.plot(average_between_block_between_task, label = 'A from B PFC Between Tasks', linestyle = '--', color='pink')
        
    plt.legend()