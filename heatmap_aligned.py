#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:03:19 2018

@author: behrenslab
"""
import numpy as np
import align_activity as aa
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import ephys_beh_import as ep
import pylab as plt

## Target times for aligned rates of non-forced trials 
def target_times_f(all_experiments):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
    trial_times_all_trials  = []
    for session in all_experiments:
        init_times = session.times['choice_state']
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'choice_state'])
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  =np.asarray(trial_times_all_trials)
    target_times = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times_all_trials[0],1),0))))    
        
    return target_times

## Target times for aligned rates of non-forced trials 
def all_sessions_aligment(experiment, all_experiments,  fs=25):
    target_times  = target_times_f(all_experiments)
    experiment_aligned = []
    for session in experiment:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        init_times = session.times['choice_state']
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'choice_state'])
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times, spikes, fs = fs)
        session.aligned_rates = aligned_rates
        session.t_out = t_out
        session.target_times = target_times
        experiment_aligned.append(session)
        
    return experiment_aligned 


def heatplot_aligned(experiment_aligned): 
    all_clusters_task_1 = []
    all_clusters_task_2 = []
    for session in experiment_aligned:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        reward = initiate_choice_t[-2] +250
        cluster_list_task_1 = []
        cluster_list_task_2 = []
        aligned_rates = session.aligned_rates
        poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = ep.extract_choice_pokes(session)
        trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3 = ep.initiation_and_trial_end_timestamps(session)
        task_1 = len(trial_сhoice_state_task_1)
        task_2 = len(trial_сhoice_state_task_2)
        if poke_I == poke_I_task_2: 
            aligned_rates_task_1 = aligned_rates[:task_1]
            aligned_rates_task_2 = aligned_rates[:task_1+task_2]
        elif poke_I == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
        elif poke_I_task_2 == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1+task_2]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
        unique_neurons  = np.unique(spikes[0])
        for i in range(len(unique_neurons)):
            mean_firing_rate_task_1  = np.mean(aligned_rates_task_1[:,i,:],0)
            mean_firing_rate_task_2  = np.mean(aligned_rates_task_2[:,i,:],0)
            cluster_list_task_1.append(mean_firing_rate_task_1) 
            cluster_list_task_2.append(mean_firing_rate_task_2)
        all_clusters_task_1.append(cluster_list_task_1[:])
        all_clusters_task_2.append(cluster_list_task_2[:])
    all_clusters_task_1 = np.array(all_clusters_task_1)
    all_clusters_task_2 = np.array(all_clusters_task_2)
    same_shape_task_1 = []
    same_shape_task_2 = []
    for i in all_clusters_task_1:
        for ii in i:
            same_shape_task_1.append(ii)
    for i in all_clusters_task_2:
        for ii in i:
            same_shape_task_2.append(ii)
    same_shape_task_1 = np.array(same_shape_task_1)
    same_shape_task_2 = np.array(same_shape_task_2)
    peak_inds = np.argmax(same_shape_task_1,1)
    ordering = np.argsort(peak_inds)
    activity_sorted = same_shape_task_1[ordering,:]
    
    #not_normed = same_shape_task_1[ordering,:]
    #not_normed += 1
    #not_normed = np.log(not_normed)
    
    norm_activity_sorted = (activity_sorted - np.min(activity_sorted,1)[:, None]) / (np.max(activity_sorted,1)[:, None] - np.min(activity_sorted,1)[:, None])
    where_are_Nans = np.isnan(norm_activity_sorted)
    norm_activity_sorted[where_are_Nans] = 0
    plt.imshow(norm_activity_sorted, aspect='auto')  
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward)).argmin()
    
    plt.xticks([ind_init, ind_choice, ind_reward], ('I', 'C', 'R'))
    plt.title('HP')
    plt.colorbar()
    



