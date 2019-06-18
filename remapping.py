#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:20:38 2018

@author: veronikasamborska
"""
import numpy as np
import ephys_beh_import as ep
import heatmap_aligned as ha
from scipy.stats import pearsonr
import regressions as re
#experiment_aligned_PFC = ha.all_sessions_aligment(PFC)


def remapping_control(experiment):
    # Plotting the proportion of cells that were firing in Task 1 and stopped firing in Task 2 or started firing at Task 2 but 
    # were not firing in Task 1  and cells that were firing at Task 2 but stopped at Task 3 or were not firing at Task 2 
    # but started firing in Task 3 
    session_a1_a1 = []
    session_a2_a2 = []
    session_a1_a2 = []
    session_a2_a3 = []
   
    for i,session in enumerate(experiment):
        index_neuron = []
        
        aligned_spikes= session.aligned_rates 
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
        predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
        predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
        t_out = session.t_out
        initiate_choice_t = session.target_times #T Times of initiation and choice 
        #Find firing rates around choice
      
        initiate_choice_t = session.target_times #Initiation and Choice Times
        
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
        ind_after_choice = ind_choice + 7 # 1 sec after choice
        spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
        mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =2) # Mean firing rates around choice 
        baseline_mean_trial = np.mean(aligned_spikes, axis =2)
        std_trial = np.std(aligned_spikes, axis =2)
        baseline_mean_all_trials = np.mean(baseline_mean_trial, axis =0)
        std_all_trials = np.std(baseline_mean_trial, axis =1)
        index_no_reward = np.where(reward ==0)
        
        a_1 = np.where(predictor_A_Task_1 == 1) #Poke A task 1 idicies
        a_2 = np.where(predictor_A_Task_2 == 1) #Poke A task 2 idicies
        a_3 = np.where(predictor_A_Task_3 == 1) #Poke A task 3 idicies
        
        a1_nR = [a for a in a_1[0] if a in index_no_reward[0]]
        a2_nR = [a for a in a_2[0] if a in index_no_reward[0]]
        a3_nR = [a for a in a_3[0] if a in index_no_reward[0]]


        choice_a1 = mean_spikes_around_choice[a1_nR]
        
        if choice_a1.shape[0]%2 == 0:
            half = (choice_a1.shape[0])/2
            a_1_first_half = choice_a1[:int(half)]
            a_1_last_half = choice_a1[int(half):]
        else: # If number of trials is uneven 
            half = (choice_a1.shape[0]-1)/2
            a_1_first_half = choice_a1[:int(half)]
            a_1_last_half = choice_a1[int(half):]
            

        choice_a2 = mean_spikes_around_choice[a2_nR]

        if choice_a2.shape[0]%2 == 0:
            half = (choice_a2.shape[0])/2
            a_2_first_half = choice_a2[:int(half)]
            a_2_last_half = choice_a2[int(half):]
        else: #If number of trials is uneven 
            half = (choice_a2.shape[0]-1)/2
            a_2_first_half = choice_a2[:int(half)]
            a_2_last_half = choice_a2[int(half):]
        
        
        choice_a3 = mean_spikes_around_choice[a3_nR]  

        if choice_a3.shape[0]%2 == 0:
            half = (choice_a3.shape[0])/2
            a_3_first_half = choice_a3[:int(half)]
            a_3_last_half = choice_a3[int(half):]

        else: # If number of trials is uneven 
            half = (choice_a3.shape[0]-1)/2
            a_3_first_half = choice_a3[:int(half)]
            a_3_last_half = choice_a3[int(half):]
       

        a1_pokes_mean_1 = np.mean(a_1_first_half, axis = 0)
        a1_pokes_mean_2 = np.mean(a_1_last_half, axis = 0)

                     
        a2_pokes_mean_1 = np.mean(a_2_first_half, axis = 0)
        a2_pokes_mean_2 = np.mean(a_2_last_half, axis = 0)

        a3_pokes_mean_1 = np.mean(a_3_first_half, axis = 0)
        a3_pokes_mean_2 = np.mean(a_3_last_half, axis = 0)
        
        for neuron in range(n_neurons):

            if a1_pokes_mean_1[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or a1_pokes_mean_2[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or a2_pokes_mean_1[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or a2_pokes_mean_2[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or a3_pokes_mean_1[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or a3_pokes_mean_2[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron]:
                index_neuron.append(neuron)
        if len(index_neuron) > 0:
            a1_a1_angle = re.angle(a1_pokes_mean_1[index_neuron], a1_pokes_mean_2[index_neuron])            
            a2_a2_angle = re.angle(a2_pokes_mean_1[index_neuron], a2_pokes_mean_2[index_neuron])            
            a1_a2_angle = re.angle(a1_pokes_mean_2[index_neuron], a2_pokes_mean_1[index_neuron])
            a2_a3_angle = re.angle(a2_pokes_mean_2[index_neuron], a3_pokes_mean_1[index_neuron])
    
            session_a1_a1.append(a1_a1_angle)
            session_a2_a2.append(a2_a2_angle)
            session_a1_a2.append(a1_a2_angle)
            session_a2_a3.append(a2_a3_angle)
            
    mean_angle_a1_a1 = np.nanmean(session_a1_a1)
    mean_angle_a1_a2 = np.nanmean(session_a1_a2)
    mean_angle_a2_a2 = np.nanmean(session_a2_a2)
    mean_angle_a2_a3 = np.nanmean(session_a2_a3)
    mean_within = np.mean([mean_angle_a1_a1,mean_angle_a2_a2])
    std_within = np.nanstd([mean_angle_a1_a1,mean_angle_a2_a2])
    mean_between = np.mean([mean_angle_a1_a2,mean_angle_a2_a3])
    std_between = np.nanstd([mean_angle_a1_a2,mean_angle_a2_a3])
    
    
    return mean_within, mean_between, std_within, std_between


#plt.bar([1,2,3,4],[mean_within_HP,mean_within_PFC, mean_between_HP, mean_between_PFC], tick_label =['Within HP', 'Within PFC','Between HP','Between PFC',])

   
       