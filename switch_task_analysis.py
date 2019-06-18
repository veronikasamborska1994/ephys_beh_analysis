#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:38:03 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import regressions as re



def block_change(experiment, distribution):
    neuron_n = 0
    all_neurons = 0
    for session in experiment:
        if session.file_name != 'm486-2018-07-28-171910.txt' and session.file_name != 'm480-2018-08-14-145623.txt':
            aligned_spikes= session.aligned_rates 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape   
            all_neurons += n_neurons
            
    a1_fr_r_last = np.zeros([all_neurons,4]) 
    a1_fr_r_first = np.zeros([all_neurons,4]) 
    
    a1_fr_nr_last = np.zeros([all_neurons,4]) 
    a1_fr_nr_first = np.zeros([all_neurons,4]) 

    a2_fr_r_last = np.zeros([all_neurons,4]) 
    a2_fr_r_first = np.zeros([all_neurons,4]) 

    a2_fr_nr_last = np.zeros([all_neurons,4]) 
    a2_fr_nr_first = np.zeros([all_neurons,4]) 

    a3_fr_r_last = np.zeros([all_neurons,4]) 
    a3_fr_r_first = np.zeros([all_neurons,4]) 

    a3_fr_nr_last = np.zeros([all_neurons,4]) 
    a3_fr_nr_first = np.zeros([all_neurons,4]) 
    
    
    for i,session in enumerate(experiment):
        if session.file_name != 'm486-2018-07-28-171910.txt' and session.file_name != 'm480-2018-08-14-145623.txt':
            aligned_spikes= session.aligned_rates 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape   
            
            # Trial indices  of choices 
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
            t_out = session.t_out
          
            initiate_choice_t = session.target_times #Initiation and Choice Times
            
            ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
            ind_after_choice = ind_choice + 7 # 1 sec after choice
            spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
            aligned_spikes  = np.mean(spikes_around_choice,axis = 2)
        
            
            a_r_1 =  aligned_spikes[np.where((predictor_A_Task_1 == 1) & (reward == 1)),:]
            a_nr_1 = aligned_spikes[np.where((predictor_A_Task_1 == 1) & (reward == 0)),:]
            
            a_r_2 = aligned_spikes[np.where((predictor_A_Task_2 == 1) & (reward == 1)),:]
            a_nr_2 = aligned_spikes[np.where((predictor_A_Task_2 == 1) & (reward == 0)),:]
            
            a_r_3 = aligned_spikes[np.where((predictor_A_Task_3 == 1) & (reward == 1)),:]
            a_nr_3 = aligned_spikes[np.where((predictor_A_Task_3 == 1) & (reward == 0)),:]
            for neuron in range(n_neurons):
    
                a1_fr_r_last[neuron_n,:] = a_r_1[0,-4:,neuron]
                a1_fr_r_first[neuron_n,:] = a_r_1[0,:4,neuron]
        
                a1_fr_nr_last[neuron_n,:] = a_nr_1[0,-4:,neuron]
                a1_fr_nr_first[neuron_n,:] = a_nr_1[0,:4,neuron]
                
                a2_fr_r_last[neuron_n,:] = a_r_2[0,-4:,neuron]
                a2_fr_r_first[neuron_n,:] = a_r_2[0,:4,neuron]
        
                a2_fr_nr_last[neuron_n,:]= a_nr_2[0,-4:,neuron]
                a2_fr_nr_first[neuron_n,:] = a_nr_2[0,:4,neuron]
          
                a3_fr_r_last[neuron_n,:] = a_r_3[0,-4:,neuron]
                a3_fr_r_first[neuron_n,:] = a_r_3[0,:4,neuron]
        
                a3_fr_nr_last[neuron_n,:] = a_nr_3[0,-4:,neuron]
                a3_fr_nr_first[neuron_n,:] = a_nr_3[0,:4,neuron]
                
                neuron_n +=1
                
            a = np.concatenate([a1_fr_r_first,a1_fr_r_last,a1_fr_nr_first,a1_fr_nr_last,a2_fr_r_first,a2_fr_r_last,a2_fr_nr_first,a2_fr_nr_last,\
                                a3_fr_r_first,a3_fr_r_last,a3_fr_nr_first,a3_fr_nr_last], axis = 1)
    
            transitions = np.concatenate([a1_fr_r_last,a1_fr_nr_last,a2_fr_r_first,a2_fr_nr_first,a2_fr_nr_last,a2_fr_r_last,\
                                a3_fr_r_first,a3_fr_nr_first], axis = 1)
        
#transitions =  transitions.reshape(transitions.shape[0],transitions.shape[1]*transitions.shape[2])

    
corr_m = np.corrcoef(np.transpose(transitions))
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
sorted_data = transitions[:,ind]

plt.imshow(sorted_data)
