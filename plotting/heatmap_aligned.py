#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:03:19 2018

@author: behrenslab
"""

# =============================================================================
# Script for trial time alignmentt and plotting a v simple heatplot to order 
# neurons by another task
# =============================================================================

import numpy as np
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import align_activity as aa

from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import ephys_beh_import as ep
import pylab as plt


def target_times_f(all_experiments):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
    trial_times_all_trials  = []
    for session in all_experiments:
        init_times = np.asarray([ev.time for ev in session.events if ev.name in ['choice_state', 'a_forced_state', 'b_forced_state']])
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'a_forced_state', 'b_forced_state','sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             (i>0 and inits_and_choices[i-1].name == 'choice_state') or (i>0  and inits_and_choices[i-1].name == 'a_forced_state')\
                                 or (i>0  and inits_and_choices[i-1].name == 'b_forced_state')])
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  =np.asarray(trial_times_all_trials)
    target_times = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times_all_trials[0],1),0))))    
        
    return target_times


def all_sessions_aligment(experiment, all_experiments,  fs=25):
    target_times  = target_times_f(all_experiments)
    experiment_aligned = []
    for session in experiment:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        init_times = np.asarray([ev.time for ev in session.events if ev.name in ['choice_state', 'a_forced_state', 'b_forced_state']])
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'a_forced_state', 'b_forced_state','sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]


        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             (i>0 and inits_and_choices[i-1].name == 'choice_state') or (i>0  and inits_and_choices[i-1].name == 'a_forced_state')\
                                 or (i>0  and inits_and_choices[i-1].name == 'b_forced_state')])
      
        
        if len(choice_times) != len(init_times):
            init_times  = (init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times, spikes, fs = fs)    
        session.aligned_rates = aligned_rates
        session.t_out = t_out
        session.target_times = target_times
        experiment_aligned.append(session)
        
    return experiment_aligned 

