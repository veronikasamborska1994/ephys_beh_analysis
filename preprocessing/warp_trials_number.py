#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:39:22 2019

@author: veronikasamborska
"""

## Warping code for interpolating firing rates on neighbouring trials

import heatmap_aligned as ha 
import align_activity as aa
import numpy as np
import copy 
import ephys_beh_import as ep

def target_trials(DM):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
    trial_times_all_trials  = []
    for dm in DM:
        block = dm[:,4]
        
        state_change = np.where(np.diff(block)!=0)[0]+1
        state_change = np.append(state_change,0)
        state_change = np.sort(state_change)
           
        state_trial_n = np.diff(state_change)
        max_trial = np.max(state_trial_n)
        middle = max_trial/2
  
        trial_times = np.array([1, middle,max_trial ]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  = np.asarray(trial_times_all_trials)
    target_times = np.median(trial_times_all_trials, axis = 0)
        
    return target_times


# Target times for aligned rates of all trials 
def all_sessions_aligment_forced_unforced(DM, Data):
    
    target_times_forced_trials  = target_trials(DM)
    firing_aligned = []
    firing_aligned_t_out = []
    for dm, data in zip(DM, Data):
        trials, neurons, time = data.shape
        block = dm[:,4]

        state_change = np.where(np.diff(block)!=0)[0]+1
        state_change = np.append(state_change,0)
        state_change = np.sort(state_change)
        
        state_trial_n = np.diff(state_change)
        max_trial = np.max(state_trial_n)
        middle = int(max_trial/2)
  
        trial_times = np.array([1,middle,  max_trial]).T
        aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times_forced_trials, data)
        
        firing_aligned.append(aligned_rates)
        firing_aligned_t_out.append(t_out)
        
    return firing_aligned

firing_aligned = all_sessions_aligment_forced_unforced(DM, Data)



