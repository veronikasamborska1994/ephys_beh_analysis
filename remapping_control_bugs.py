#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:33:54 2018

@author: behrenslab
"""

import copy 
import numpy as np
from scipy.ndimage import gaussian_filter1d

import pylab as pl

import math
import ephys_beh_import as ep
import heatmap_aligned as ha

ephys_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons'
beh_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys'



#HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path)
#experiment_aligned_HP = all_sessions_aligment(HP)
#experiment_aligned_PFC = all_sessions_aligment(PFC)

def remapping_control(experiment_aligned_HP):
    mean_list_a1_1_half = []
    mean_list_a1_2_half = []
    mean_list_a2_1_half = []
    mean_list_a2_2_half = []
    mean_list_a3_1_half = []
    mean_list_a3_2_half = []
    for session in experiment_aligned_HP:
        aligned_spikes= session.aligned_rates 
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = ha.predictors_f(session)
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        #Find firinf rates around choice
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
        ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
        ind_after_choice = ind_choice + 7
        spikes_around_choice = aligned_spikes[:,:,ind_choice:ind_after_choice]
        baseline_firing = aligned_spikes[:,:,:ind_choice-2]
        #Finding a good and b good indicies 
        state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = ep.state_indices(session)
        ind_a_good_task_1 = state_a_good
        ind_a_good_task_2 = state_t2_a_good + [(len(state_a_good)+len(state_b_good))]
        ind_b_good_task_2 = state_t2_b_good + [(len(state_a_good)+len(state_b_good))]
        ind_a_good_task_3 = state_t3_a_good + [(len(state_a_good)+len(state_b_good))+(len(state_t2_a_good)+len(state_t2_b_good))]
        ind_b_good_task_3 = state_t3_b_good + [(len(state_a_good)+len(state_b_good))+(len(state_t2_a_good)+len(state_t2_b_good))]
        
        a_1 = np.where(predictor_A_Task_1 == 1)
        choice_a1 = spikes_around_choice[a_1,:,:]
        baseline_a1 = baseline_firing[a_1,:,:]
        if choice_a1.shape[1]%2 == 0:
            split_a1 = np.split(choice_a1, 2, axis = 1)
            baseline_split_a1 = np.split(baseline_a1,2,axis = 1)
        else:# If number of trials is uneven 
            split_a1 = np.split(choice_a1[:,:-1,:,:], 2, axis = 1)
            baseline_split_a1 = np.split(baseline_a1[:,:-1,:,:],2,axis = 1)

        a_1_first_half = split_a1[0]
        a_1_last_half = split_a1[1] 
        bsf_a1_first_half =baseline_split_a1[0]
        bsf_a1_last_half =baseline_split_a1[1]

        a_2 = np.where(predictor_A_Task_2 == 1)
        choice_a2 = spikes_around_choice[a_2,:,:]
        baseline_a2 = baseline_firing[a_2,:,:]
        if choice_a2.shape[1]%2 == 0:
            split_a2 = np.split(choice_a2, 2, axis = 1)
            baseline_split_a2 = np.split(baseline_a2,2,axis = 1)
        else:# If number of trials is uneven 
            split_a2 = np.split(choice_a2[:,:-1,:,:], 2, axis = 1)
            baseline_split_a2 = np.split(baseline_a2[:,:-1,:,:],2,axis = 1)

        a2_first_half = split_a2[0]
        a2_last_half = split_a2[1]
        bsf_a2_first_half =baseline_split_a2[0]
        bsf_a2_last_half =baseline_split_a2[1]
        
        a_3 = np.where(predictor_A_Task_3 == 1)
        choice_a3 = spikes_around_choice[a_3,:,:]      
        baseline_a3 = baseline_firing[a_3,:,:] 
        if choice_a3.shape[1]%2 == 0:
            split_a3 = np.split(choice_a3, 2, axis = 1)            
            baseline_split_a3 = np.split(baseline_a3,2,axis = 1)
        else: # If number of trials is uneven 
            split_a3 = np.split(choice_a3[:,:-1,:,:], 2, axis = 1)
            baseline_split_a3 = np.split(baseline_a3[:,:-1,:,:],2,axis = 1)

        a3_first_half = split_a3[0]
        a3_last_half = split_a3[1]
        bsf_a3_first_half =baseline_split_a3[0]
        bsf_a3_last_half =baseline_split_a3[1]

        choice_a3 = spikes_around_choice[a_3,:,:]
        for neuron in range(n_neurons):           
            mean_rate_around_choice_a1 = np.mean(a_1_first_half[:,:,neuron,:])
            baseline_mean_a1_1 = np.mean(bsf_a1_first_half[:,:,neuron,:]) # Baseline Mean
            std_choice_a1_1 = np.std(bsf_a1_first_half[:,:,neuron,:]) # Baseline SD
            mean_list_a1_1_half.append(mean_rate_around_choice_a1)
            
            mean_rate_around_choice_a1 = np.mean(a_1_last_half[:,:,neuron,:])
            baseline_mean_a1_2 = np.mean(bsf_a1_last_half[:,:,neuron,:])
            std_choice_a1_2= np.std(bsf_a1_last_half[:,:,neuron,:])
            mean_list_a1_2_half.append(mean_rate_around_choice_a1)
            
            mean_rate_around_choice_a2 = np.mean(a2_first_half[:,:,neuron,:])
            baseline_mean_a2_1 = np.mean(bsf_a2_first_half[:,:,neuron,:])
            std_choice_a2_1= np.std(bsf_a2_first_half[:,:,neuron,:])
            mean_list_a2_1_half.append(mean_rate_around_choice_a2)
            
            mean_rate_around_choice_a2 = np.mean(a2_last_half[:,:,neuron,:])
            baseline_mean_a2_2 = np.mean(bsf_a2_last_half[:,:,neuron,:])
            std_choice_a2_2= np.std(bsf_a2_last_half[:,:,neuron,:])
            mean_list_a2_2_half.append(mean_rate_around_choice_a2)
            
                        
            mean_rate_around_choice_a3 = np.mean(a3_first_half[:,:,neuron,:])
            baseline_mean_a3_1 = np.mean(bsf_a3_first_half[:,:,neuron,:])
            std_choice_a3_1 = np.std(bsf_a3_first_half[:,:,neuron,:])
            mean_list_a3_1_half.append(mean_rate_around_choice_a3)
            
            mean_rate_around_choice_a3 = np.mean(a3_last_half[:,:,neuron,:])
            baseline_mean_a3_2 = np.mean(bsf_a3_last_half[:,:,neuron,:])
            std_choice_a3_2= np.std(bsf_a3_last_half[:,:,neuron,:])
            mean_list_a3_2_half.append(mean_rate_around_choice_a3)
            
        mean_a1_tuned_1 = []
        mean_a1_tuned_2 = []
        mean_a2_tuned_1 = []
        mean_a2_tuned_2 = []
        mean_a3_tuned_1 = []
        mean_a3_tuned_2 = []
        index_list_a1_1 = []
        index_list_a1_2 = []
        index_list_a2_1 = []
        index_list_a2_2 = []
        index_list_a3_1 = []
        index_list_a3_2 = []
        
        for mean in mean_list_a1_1_half:
            if mean > (baseline_mean_a1_1 + (3*std_choice_a1_1)):
                index_neuron = np.where(mean_list_a1_1_half == mean)
                index_list_a1_1.append(index_neuron)
                mean_a1_tuned_1.append(mean)
                
        for mean in mean_list_a1_2_half:
            if mean > (baseline_mean_a1_2 +  (3*std_choice_a1_2)):
                index_neuron = np.where(mean_list_a1_2_half == mean)
                index_list_a1_2.append(index_neuron)
                mean_a1_tuned_2.append(mean)
                
        for mean in mean_list_a2_1_half:
            if mean > (baseline_mean_a2_1 + (3*std_choice_a2_1)):
                index_neuron= np.where(mean_list_a2_1_half == mean)
                index_list_a2_1.append(index_neuron)
                mean_a2_tuned_1.append(mean)
        
        for mean in mean_list_a2_2_half:
            if mean > (baseline_mean_a2_2 + (3*std_choice_a2_2)):
                index_neuron= np.where(mean_list_a2_2_half == mean)
                index_list_a2_2.append(index_neuron)
                mean_a2_tuned_2.append(mean)
        
                
        for mean in mean_list_a3_1_half:
            if mean > (baseline_mean_a3_1 + (3*std_choice_a3_1)):
                index_neuron= np.where(mean_list_a3_1_half == mean)
                index_list_a3_1.append(index_neuron)
                mean_a3_tuned_1.append(mean)
                
        for mean in mean_list_a3_2_half:
            if mean > (baseline_mean_a3_2 + (3*std_choice_a3_2)):
                index_neuron= np.where(mean_list_a3_2_half == mean)
                index_list_a3_2.append(index_neuron)
                mean_a3_tuned_2.append(mean)
                
    a1_neuron_list_1 =[]
    for x in index_list_a1_1:
        for y in x:
            for i in y:
                a1_neuron_list_1.append(i)
    
    a1_neuron_list_2 =[]
    for x in index_list_a1_2:
        for y in x:
            for i in y:
                a1_neuron_list_2.append(i)

    a2_neuron_list_1 =[]
    for x in index_list_a2_1:
        for y in x:
            for i in y:
                a2_neuron_list_1.append(i)
    
    a2_neuron_list_2 =[]
    for x in index_list_a2_2:
        for y in x:
            for i in y:
                a2_neuron_list_2.append(i)
    
    a3_neuron_list_1 =[]
    for x in index_list_a3_1:
        for y in x:
            for i in y:
                a3_neuron_list_1.append(i)
                
    a3_neuron_list_2 =[]
    for x in index_list_a3_2:
        for y in x:
            for i in y:
                a3_neuron_list_2.append(i)    
                
    neurons_modulated_change_a1_2_a2_1 = [neuron for neuron in a1_neuron_list_2 if neuron not in a2_neuron_list_1]
    neurons_modulated_change_a2_2_a3_1 = [neuron for neuron in a2_neuron_list_2 if neuron not in a3_neuron_list_1]
    
    mean_2_1 = np.mean([len(neurons_modulated_change_a1_2_a2_1), len(neurons_modulated_change_a2_2_a3_1)])
    std_2_1 = np.std([len(neurons_modulated_change_a1_2_a2_1), len(neurons_modulated_change_a2_2_a3_1)])
    neurons_modulated_change_a1_a1 = [neuron for neuron in a1_neuron_list_2 if not neuron in a1_neuron_list_1]
    neurons_modulated_change_a2_a2 = [neuron for neuron in a2_neuron_list_2 if not neuron in a2_neuron_list_1]
    neurons_modulated_change_a3_a3 = [neuron for neuron in a3_neuron_list_2 if not neuron in a3_neuron_list_1]

    mean_within_task = np.mean([len(neurons_modulated_change_a1_a1), len(neurons_modulated_change_a2_a2), len(neurons_modulated_change_a3_a3)])
    std_within_task = np.std([len(neurons_modulated_change_a1_a1), len(neurons_modulated_change_a2_a2), len(neurons_modulated_change_a3_a3)])
    bar([1,2],[mean_within_task, mean_2_1], tick_label =['Within Task', 'Between Tasks'], yerr = [std_within_task,std_2_1])

