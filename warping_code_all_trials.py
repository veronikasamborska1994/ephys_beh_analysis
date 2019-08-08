#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:35:27 2019

@author: veronikasamborska
"""
import heatmap_aligned as ha 
import align_activity as aa
import numpy as np
import copy 
import ephys_beh_import as ep

# Target times for aligned rates of all trials (forced + non-forced)
def all_sessions_aligment_forced_unforced(experiment,all_experiments):
    target_times_forced_trials  = ha.target_times_f(all_experiments)
    experiment_aligned_forced_unforced = []
    for session in experiment:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        init_times = np.concatenate((session.times['choice_state'],session.times['b_forced_state'], session.times['a_forced_state']), axis = 0)
        init_times = sorted(init_times)
        init_times = np.asarray(init_times)
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['a_forced_state','b_forced_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward', 'choice_state']]
        
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'a_forced_state' or inits_and_choices[i-1].name == 'b_forced_state' or inits_and_choices[i-1].name == 'choice_state'] )
            
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        aligned_rates_forced_unforced, t_out, min_max_stretch = aa.align_activity(trial_times, target_times_forced_trials, spikes)
        session.aligned_rates_forced_unforced = aligned_rates_forced_unforced
        session.t_out = t_out
        session.target_times_forced_trials = target_times_forced_trials
        experiment_aligned_forced_unforced.append(session)
        
    return experiment_aligned_forced_unforced




