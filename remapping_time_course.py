#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:33:54 2018

@author: behrenslab
"""

import numpy as np
#import ephys_beh_import as ep
import heatmap_aligned as ha
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d as gs
#ephys_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons'
#beh_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys'
#HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path)
#experiment_aligned_HP = ha.all_sessions_aligment(HP)
#experiment_aligned_PFC = ha.all_sessions_aligment(PFC)


ephys_path = '/Users/veronikasamborska/Desktop/neurons'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

# Plotting timecourse of cells that either decreased or increased their firing rates. Selection bias so not a great analysis.
def remapping_timecourse(experiment):
    
    # Lists for appending the last 20 trials of task 1 and the first 20 trials of task 2 for neurons 
    # that either decreased or increased their firing rates between two tasks around choice time
    
    a1_a2_list_increase = []
    a2_a3_list_increase = []
    a1_a2_list_decrease = []
    a2_a3_list_decrease = []
    
    for session in experiment:
        
        aligned_spikes= session.aligned_rates 
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape
        
        # Numpy arrays to fill the firing rates of each neuron on the 40 trials where the A choice was made
        a1_a2_increase = np.ones(shape=(n_neurons,100))
        a1_a2_increase[:] = np.NaN
        a2_a3_increase= np.ones(shape=(n_neurons,100))
        a2_a3_increase[:] = np.NaN
        
        a1_a2_decrease = np.ones(shape=(n_neurons,100))
        a1_a2_decrease[:] = np.NaN
        a2_a3_decrease= np.ones(shape=(n_neurons,100))
        a2_a3_decrease[:] = np.NaN
        
        # Trial indices  of choices 
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = ha.predictors_f(session)
        t_out = session.t_out
      
        initiate_choice_t = session.target_times #Initiation and Choice Times
        
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
        ind_after_choice = ind_choice + 7 # 1 sec after choice
        spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice
        mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =0)
        for i in mean_spikes_around_choice:
            figure()
            plot(mean_spikes_around_choice[i,:])
        forced_trials = session.trial_data['forced_trial']
        non_forced_array = np.where(forced_trials == 0)[0]
        task = session.trial_data['task']
        task_non_forced = task[non_forced_array]
        task_2 = np.where(task_non_forced == 2)[0] 
        a_pokes = predictor_A_Task_1 + predictor_A_Task_2 + predictor_A_Task_3 # All A pokes across all three tasks
        
        mean_trial = np.mean(spikes_around_choice,axis = 2) # Mean firing rates around choice 
        
        a_1 = np.where(predictor_A_Task_1 == 1)
        a_2 = np.where(predictor_A_Task_2 == 1)
        a_3 = np.where(predictor_A_Task_3 == 1)
        task_2_start = task_2[0]
        task_3_start = task_2[-1]+1
        n_trials_of_interest = 50
        
        where_a_task_1_2= np.where(a_pokes[task_2_start - n_trials_of_interest: task_2_start + n_trials_of_interest] == 1) # Indices of A pokes in the 40 trials around 1 and 2 task switch
        where_a_task_2_3 = np.where(a_pokes[task_3_start - n_trials_of_interest:task_3_start + n_trials_of_interest] == 1) # Indices of A pokes in the 40 trials around 2 and 3 task switch

        for neuron in range(n_neurons):
            trials_firing = mean_trial[:,neuron]  # Firing rate of each neuron
           
            a1_fr = trials_firing[a_1]  # Firing rates on poke A choices in Task 1 
            a1_mean = np.mean(a1_fr, axis = 0)  # Mean rate on poke A choices in Task 1 
            a1_std = np.std(a1_fr) # Standard deviation on poke A choices in Task 1 
            a2_fr = trials_firing[a_2]
            a2_std = np.std(a2_fr)
            a2_mean = np.mean(a2_fr, axis = 0)
            a3_fr = trials_firing[a_3]
            a3_mean = np.mean(a3_fr, axis = 0)
            # If mean firing rate on a2 is higher than on a1 or mean firing rate on a3 
            #is higher than a2 find the firing rate on the trials for that neuron
            if a2_mean > (a1_mean+(3*a1_std)) or a3_mean > (a2_mean+(3*a2_std)):
                t1_t2 = trials_firing[task_2_start - n_trials_of_interest:task_2_start + n_trials_of_interest] 
                t2_t3 = trials_firing[task_3_start - n_trials_of_interest:task_3_start + n_trials_of_interest]
                
                a1_a2_increase[neuron,where_a_task_1_2] = t1_t2[where_a_task_1_2]
                a2_a3_increase[neuron,where_a_task_2_3] = t2_t3[where_a_task_2_3]
            # If mean firing rate on a2 is lower than on a1 or mean firing rate on a3 
            #is lower than a2 find the firing rate on the trials for that neuron
            elif a2_mean < (a1_mean+(3*a1_std)) or a3_mean < (a2_mean+(3*a2_std)):
                t1_t2 = trials_firing[task_2_start - n_trials_of_interest:task_2_start + n_trials_of_interest] 
                t2_t3 = trials_firing[task_3_start - n_trials_of_interest:task_3_start + n_trials_of_interest]
                a1_a2_decrease[neuron,where_a_task_1_2] = t1_t2[where_a_task_1_2]
                a2_a3_decrease[neuron,where_a_task_2_3] = t2_t3[where_a_task_2_3]
                
        a1_a2_list_increase.append(a1_a2_increase)
        a2_a3_list_increase.append(a2_a3_increase)
        a1_a2_list_decrease.append(a1_a2_decrease)
        a2_a3_list_decrease.append(a2_a3_decrease)
             
             
    a1_a2_list_increase = np.array(a1_a2_list_increase)
    a2_a3_list_increase = np.array(a2_a3_list_increase)
    
    a1_a2_list_decrease = np.array(a1_a2_list_decrease)
    a2_a3_list_decrease = np.array(a2_a3_list_decrease)
    
    a_list_increase = np.concatenate([a1_a2_list_increase,a2_a3_list_increase])
    a_list_decrease = np.concatenate([a1_a2_list_decrease,a2_a3_list_decrease])

    flattened_a_list_increase = []
    for x in a_list_increase:
        for y in x:
            index = np.isnan(y) 
            if np.all(index):
                continue
            else:
                flattened_a_list_increase.append(y)
                
    flattened_a_list_decrease = []
    for x in a_list_decrease:
        for y in x:
            index = np.isnan(y) 
            if np.all(index):
                continue
            else:
                flattened_a_list_decrease.append(y)
    flattened_a_list_increase = np.array(flattened_a_list_increase)
    flattened_a_list_decrease = np.array(flattened_a_list_decrease)
    x_array = np.arange(1,101)
    task_change = 50
    #imshow(flattened_a_list_increase, aspect='auto')
    mean_increase = np.nanmean(flattened_a_list_increase, axis = 0)
    mean_decrease = np.nanmean(flattened_a_list_decrease, axis = 0)
    plt.figure()
    smoothed_dec = gs(mean_decrease, 15)
    plt.plot(x_array, smoothed_dec)
    plt.axvline(task_change, color='k', linestyle=':')
    plt.ylabel('Number of Trials Before and After Task Change')
    plt.xlabel('Mean Firing Rate')
    plt.title('Cells that Decrease Firing Rates')
    
    plt.figure()
    smoothed_inc = gs(mean_increase, 15)
    plt.plot(x_array, smoothed_inc)
    plt.axvline(task_change, color='k', linestyle=':')
    plt.ylabel('Number of Trials Before and After Task Change')
    plt.xlabel('Mean Firing Rate')
    plt.title('Cells that Increase Firing Rates')


   
