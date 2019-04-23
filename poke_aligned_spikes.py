#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:43:25 2019

@author: veronikasamborska
"""
import regression_poke_based_include_a as re
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import numpy as np


def histogram_include_a(session,time_window = 1):
    
    # time window 1 == all_sessions_1s_500ms
    # time window 2 == all_sessions_500ms_0
    # time window 3 ==  all_sessions_0_500ms
    # time window 4 == all_sessions_500ms_1_sec
    # time window 5  == all_sessions_1_sec_500ms
    poke_identity,outcomes_non_forced,initation_choice,initiation_time_stamps,poke_list_A, poke_list_B,all_events,constant_poke_a,choices, trial_times  = re.extract_poke_times_include_a(session)
    
    neurons = np.unique(session.ephys[0])
    spikes = session.ephys[1]
    window_to_plot = 4000
    #all_neurons_all_spikes_raster_plot_task = []
    smooth_sd_ms = 1
    bin_width_ms = 50
   
    bin_edges_trial = np.arange(-4050,window_to_plot, bin_width_ms)
    # 10 for 0.5 second  
    trial_length = 10
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

                if time_window == 1:
                    normalised_task = normalised_task[60:70]
                elif time_window == 2:
                    normalised_task = normalised_task[70:80]
                elif time_window == 3:
                    normalised_task = normalised_task[80:90]
                elif time_window == 4:
                    normalised_task = normalised_task[90:100]
                elif time_window == 5:
                    normalised_task = normalised_task[100:110]
                
            aligned_rates[e,i,:]  = normalised_task

    return aligned_rates


        
def raster_plot_save(experiment,time_window = 1):
    all_sessions = []
    for s,session in enumerate(experiment):
        
        aligned_rates = histogram_include_a(session,time_window = time_window)
        aligned_rates = np.asarray(aligned_rates)
        all_sessions.append(aligned_rates)
        
    return all_sessions