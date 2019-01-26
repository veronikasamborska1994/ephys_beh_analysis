#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:06:37 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import ephys_beh_import as ep
import regressions as re

# Script for finding SVDs based on whether A and B happened in good or bad blocks

def extract_session_a_b_based_on_block(session, tasks_unchanged = True):
    # Extracta A and B trials based on what block and task it happened
    # Takes session as an argument and outputs A and B trials for when A and B ports were good in every task
    spikes = session.ephys
    spikes = spikes[:,~np.isnan(spikes[1,:])] 
    aligned_rates = session.aligned_rates
    
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = ep.extract_choice_pokes(session)
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3 = ep.initiation_and_trial_end_timestamps(session)
    task_1 = len(trial_сhoice_state_task_1)
    task_2 = len(trial_сhoice_state_task_2)
    
    
    # Getting choice indices 
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
    
    if aligned_rates.shape[0] != predictor_A_Task_1.shape[0]:
        predictor_A_Task_1 = predictor_A_Task_1[:aligned_rates.shape[0]] 
        predictor_A_Task_2 = predictor_A_Task_2[:aligned_rates.shape[0]] 
        predictor_A_Task_3 = predictor_A_Task_3[:aligned_rates.shape[0]] 
        predictor_B_Task_1 = predictor_B_Task_1[:aligned_rates.shape[0]] 
        predictor_B_Task_2 = predictor_B_Task_2[:aligned_rates.shape[0]] 
        predictor_B_Task_3 = predictor_B_Task_3[:aligned_rates.shape[0]] 
        reward = reward[:aligned_rates.shape[0]] 
        
    ## If you want to only look at tasks with a shared I port 
    if tasks_unchanged == False:
        if poke_I == poke_I_task_2: 
            aligned_rates_task_1 = aligned_rates[:task_1]
            predictor_A_Task_1 = predictor_A_Task_1[:task_1]
            predictor_B_Task_1 = predictor_B_Task_1[:task_1]
            reward_task_1 = reward[:task_1]
            aligned_rates_task_2 = aligned_rates[:task_1+task_2]
            predictor_A_Task_2 = predictor_A_Task_2[:task_1+task_2]
            predictor_B_Task_2 = predictor_B_Task_2[:task_1+task_2]
            reward_task_2 = reward[:task_1+task_2]
            predictor_a_good_task_1 = predictor_a_good_task_1
            predictor_a_good_task_2 = predictor_a_good_task_2
            
        elif poke_I == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1]
            predictor_A_Task_1 = predictor_A_Task_1[:task_1]
            predictor_B_Task_1 = predictor_B_Task_1[:task_1]
            #reward_task_1 = reward[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
            predictor_A_Task_2 = predictor_A_Task_3[task_1+task_2:]
            predictor_B_Task_2 = predictor_B_Task_3[task_1+task_2:]
            #reward_task_2 = reward[task_1+task_2:]
            
            predictor_a_good_task_1 = predictor_a_good_task_1
            predictor_a_good_task_2 = predictor_a_good_task_3
            
        elif poke_I_task_2 == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1+task_2]
            predictor_A_Task_1 = predictor_A_Task_2[:task_1+task_2]
            predictor_B_Task_1 = predictor_B_Task_2[:task_1+task_2]
            #reward_task_1 = reward[:task_1+task_2]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
            predictor_A_Task_2 = predictor_A_Task_3[task_1+task_2:]
            predictor_B_Task_2 = predictor_B_Task_3[task_1+task_2:]
            #reward_task_2 = reward[task_1+task_2:]
            
            predictor_a_good_task_1 = predictor_a_good_task_2
            predictor_a_good_task_2 = predictor_a_good_task_3
            
    #Get firing rates for each task
    aligned_rates_task_1 = aligned_rates[:task_1]
    aligned_rates_task_2 = aligned_rates[task_1:task_1+task_2]
    aligned_rates_task_3 = aligned_rates[task_1+task_2:]
    
    #Indicies of A choices in each task (1s) and Bs are just 0s 
    predictor_A_Task_1_cut = predictor_A_Task_1[:task_1]
    #reward_task_1_cut = reward[:task_1]
          
    predictor_A_Task_2_cut = predictor_A_Task_2[task_1:task_1+task_2]
    #reward_task_2_cut = reward[task_1:task_1+task_2]
          
    predictor_A_Task_3_cut = predictor_A_Task_3[task_1+task_2:]
    #reward_task_3_cut = reward[task_1+task_2:]
    
    
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
        
        
        
def block_firings_rates_selection(experiment):   
    cluster_list_task_1_a_good = []
    cluster_list_task_1_b_good = []   
    cluster_list_task_2_a_good = []
    cluster_list_task_2_b_good = []
    cluster_list_task_3_a_good = []
    cluster_list_task_3_b_good = []
        
    for s,session in enumerate(experiment):
        if session.trial_data['block'][-1] >= 11:
            
            state_A_choice_A_t1,state_A_choice_B_t1,state_B_choice_A_t1,state_B_choice_B_t1,\
            state_A_choice_A_t2, state_A_choice_B_t2,state_B_choice_A_t2,state_B_choice_B_t2,\
            state_A_choice_A_t3, state_A_choice_B_t3, state_B_choice_A_t3, state_B_choice_B_t3, spikes = extract_session_a_b_based_on_block(session)
            
            
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
                    
                    mean_firing_rate_a_task_1 = np.concatenate((mean_firing_rate_task_1_a_good_A_ch,mean_firing_rate_task_1_a_good_B_ch), axis = 0)
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
    
    return cluster_list_task_1_a_good, cluster_list_task_1_b_good,cluster_list_task_2_a_good,\
    cluster_list_task_2_b_good, cluster_list_task_3_a_good, cluster_list_task_3_b_good
        
def svd_plotting_block_analysis(experiment, tasks_unchanged = False, plot_HP = True, plot_a = False):
    
    #Calculating SVDs for trials split by blocks
    cluster_list_task_1_a_good, cluster_list_task_1_b_good, cluster_list_task_2_a_good,\
    cluster_list_task_2_b_good, cluster_list_task_3_a_good, cluster_list_task_3_b_good = block_firings_rates_selection(experiment)
    
    
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
    
    #Explain A from A 
    
    #Compare block a task 2 from block a task 1
    s_task_2_block_a_good_t1= np.linalg.multi_dot([t_u_t1_a_good, cluster_list_task_2_a_good, t_v_t1_a_good])
    s_task_2_a_good_from_a_good_task_1= s_task_2_block_a_good_t1.diagonal()
    sum_s_task_2_a_good_from_a_good_task_1 = np.cumsum(s_task_2_a_good_from_a_good_task_1)/cluster_list_task_2_a_good.shape[0]
     
    #Compare block b task 2 from block b task 1
    s_task_2_block_b_good_t1 = np.linalg.multi_dot([t_u_t1_b_good, cluster_list_task_2_b_good, t_v_t1_b_good])
    s_task_2_b_good_from_b_good_task_1= s_task_2_block_b_good_t1.diagonal()
    sum_s_task_2_b_good_from_b_good_task_1 = np.cumsum(s_task_2_b_good_from_b_good_task_1)/cluster_list_task_2_b_good.shape[0]
    
    #Compare block a task 3 from block a task 2
    s_task_3_block_a_good_t2 = np.linalg.multi_dot([t_u_t2_a_good, cluster_list_task_3_a_good, t_v_t2_a_good])
    s_task_3_a_good_from_a_good_task_2 = s_task_3_block_a_good_t2.diagonal()
    sum_s_task_3_a_good_from_a_good_task_2 = np.cumsum(s_task_3_a_good_from_a_good_task_2)/cluster_list_task_3_a_good.shape[0]
    
    #Compare block b task 3 from block b task 2
    s_task_3_block_b_good_t2 = np.linalg.multi_dot([t_u_t2_b_good, cluster_list_task_3_b_good, t_v_t2_b_good])
    s_task_3_b_good_from_b_good_task_2= s_task_3_block_b_good_t2.diagonal()
    sum_s_task_3_b_good_from_b_good_task_2 = np.cumsum(s_task_3_b_good_from_b_good_task_2)/cluster_list_task_3_b_good.shape[0]
    
    ### Explaining A from b
    
    #Compare block b task 2 from block a task 1
    s_task_2_block_b_good_t1_block_a= np.linalg.multi_dot([t_u_t1_a_good, cluster_list_task_2_b_good, t_v_t1_a_good])
    s_task_2_b_good_from_a_good_task_1= s_task_2_block_b_good_t1_block_a.diagonal()
    sum_s_task_2_b_good_from_a_good_task_1 = np.cumsum(s_task_2_b_good_from_a_good_task_1)/cluster_list_task_2_b_good.shape[0]
     
    #Compare block a task 2 from block b task 1
    s_task_2_block_a_good_t1_b = np.linalg.multi_dot([t_u_t1_b_good, cluster_list_task_2_a_good, t_v_t1_b_good])
    s_task_2_a_good_from_a_good_task_1= s_task_2_block_a_good_t1_b.diagonal()
    sum_s_task_2_a_good_from_b_good_task_1 = np.cumsum(s_task_2_a_good_from_a_good_task_1)/cluster_list_task_2_a_good.shape[0]
    
    #Compare block b task 3 from block a task 2
    s_task_3_block_b_good_t2_a = np.linalg.multi_dot([t_u_t2_a_good, cluster_list_task_3_b_good, t_v_t2_a_good])
    s_task_3_b_good_from_a_good_task_2= s_task_3_block_b_good_t2_a.diagonal()
    sum_s_task_3_b_good_from_a_good_task_2 = np.cumsum(s_task_3_b_good_from_a_good_task_2)/cluster_list_task_3_b_good.shape[0]
    
    #Compare block a task 3 from block b task 2
    s_task_3_block_a_good_t2_b = np.linalg.multi_dot([t_u_t2_b_good, cluster_list_task_3_a_good, t_v_t2_b_good])
    s_task_3_a_good_from_b_good_task_1 = s_task_3_block_a_good_t2_b.diagonal()
    sum_s_task_3_a_good_from_b_good_task_2 = np.cumsum(s_task_3_a_good_from_b_good_task_1)/cluster_list_task_3_a_good.shape[0]
    
    
    average_b_to_b = np.mean([sum_s_task_2_b_good_from_b_good_task_1,sum_s_task_3_b_good_from_b_good_task_2], axis  = 0 )
    average_a_to_a = np.mean([sum_s_task_2_a_good_from_a_good_task_1,sum_s_task_3_a_good_from_a_good_task_2], axis  = 0 )
    
    average_within = np.mean([average_b_to_b, average_a_to_a],axis = 0)

    average_b_to_a = np.mean([sum_s_task_2_a_good_from_b_good_task_1,sum_s_task_3_a_good_from_b_good_task_2], axis  = 0 )    
    average_a_to_b = np.mean([sum_s_task_2_b_good_from_a_good_task_1,sum_s_task_3_b_good_from_a_good_task_2], axis  = 0 )

    average_between = np.mean([average_b_to_a, average_a_to_b],axis = 0)
    
#    plot(average_within, label = 'Explain A block from A block PFC')
#    plot(average_between, label = 'Explain B block from A block PFC')
    
    if plot_a == False and plot_HP == True :
        plt.plot(average_within, label = 'Explain A from A HP', color='black')
        plt.plot(average_between, label = 'Explain B from A HP', linestyle = '--', color='black')
        
    elif plot_a == False and plot_HP == False:
        plt.plot(average_within, label = 'Explain A from A PFC', color='red')
        plt.plot(average_between, label = 'Explain B from A HP', linestyle = '--', color='red')
        
    if plot_a == True and plot_HP == True :
        plt.plot(average_a_to_a, label = 'Explain A from A HP', color='blue')
        plt.plot(average_b_to_b, label = 'Explain B from B HP', linestyle = '--', color='blue')
        plt.plot(average_b_to_a, label = 'Explain B from A HP', color='green')
        plt.plot(average_a_to_b, label = 'Explain A from B HP', linestyle = '--', color='green')
        
    elif plot_a == True and plot_HP == False:
        plt.plot(average_a_to_a, label = 'Explain A from A PFC', color='grey')
        plt.plot(average_b_to_b, label = 'Explain B from B PFC', linestyle = '--', color='grey')
        plt.plot(average_b_to_a, label = 'Explain B from A PFC', color='orange')
        plt.plot(average_a_to_b, label = 'Explain A from B PFC', linestyle = '--', color='orange')
#        
    plt.legend()
    
    