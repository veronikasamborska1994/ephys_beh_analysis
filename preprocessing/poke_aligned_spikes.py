#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:43:25 2019

@author: veronikasamborska
"""
# =============================================================================
# Create data structures aligned to poke entry rather with a specified time window around
# =============================================================================

from scipy.ndimage import gaussian_filter1d
import numpy as np
import sys 
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_import/regressions/')
import regression_poke_based_include_a as re

def histogram_include_a(session,time_window_start = 50, time_window_end = 110):
    
    poke_identity,outcomes_non_forced,initation_choice,initiation_time_stamps,poke_list_A, poke_list_B,all_events,constant_poke_a,choices, trial_times  = re.extract_poke_times_include_a(session)
    
    neurons = np.unique(session.ephys[0])
    spikes = session.ephys[1]
    window_to_plot = 4000 #Ω 4 second window around poke
    smooth_sd_ms = 1
    bin_width_ms = 50
   
    bin_edges_trial = np.arange(-4050,window_to_plot, bin_width_ms)
    trial_length = 60
    aligned_rates = np.zeros([len(all_events), len(neurons), trial_length]) # Array to store trial aligned firing rates. 

    for i,neuron in enumerate(neurons):  
        spikes_ind = np.where(session.ephys[0] == neuron)
        spikes_n = spikes[spikes_ind]

        for e,event in enumerate(all_events):
            period_min = event - window_to_plot
            period_max = event + window_to_plot
            spikes_ind = spikes_n[(spikes_n >= period_min) & (spikes_n<= period_max)]

            spikes_to_save = (spikes_ind - event)   
            hist_task,edges_task = np.histogram(spikes_to_save, bins= bin_edges_trial)
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms)
            normalised_task = normalised_task*20
            if len(normalised_task) > 0:
                normalised_task = normalised_task[time_window_start:time_window_end]
                
            aligned_rates[e,i,:]  = normalised_task

    return aligned_rates


        
def raster_plot_save(experiment,time_window_start = 50, time_window_end = 110):
    all_sessions = []
    
    for s,session in enumerate(experiment):
        aligned_rates = histogram_include_a(session,time_window_start = time_window_start,time_window_end = time_window_end)
        aligned_rates = np.asarray(aligned_rates)
        all_sessions.append(aligned_rates)
        
    return all_sessions