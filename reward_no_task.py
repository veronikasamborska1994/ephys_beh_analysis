#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:35:10 2019

@author: veronikasamborska
"""

from random import randint
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.io
from sklearn import svm
from sklearn import metrics
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import remapping_count as rc 
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import utility as ut


    
def free_rewards(experiment):
    
    for s,session in enumerate(experiment):
        #neurons_aligned = session.aligned_rates
        #outcomes = session.trial_data['outcomes']  
        #outcomes_non_forced = outcomes[np.where(session.trial_data['forced_trial'] == 0)[0]]
        #reward_aligned = neurons_aligned[np.where(outcomes_non_forced==1)[0],:,:]
        #spikes_port = HP_port_alinged[s]
        #port = spikes_port[1::2]
        #reward_poke = port[np.where(outcomes_non_forced==1)[0],:,:]
        neurons = np.unique(session.ephys[0])
        spikes = session.ephys[1]
        window_to_plot = 4000 #Ω 1 second window around poke
        smooth_sd_ms = 1
        bin_width_ms = 50
       
        bin_edges_trial = np.arange(-4050,window_to_plot, bin_width_ms)
        # 10 for 0.5 second  
        events = [event for event in session.events if event.name in ['free_reward_trial', 'sound_b_reward','sound_a_reward']]

        free_reward = []
        poke_name = []
        poke_time = []
        events_rewards = [event.time for event in session.events if event.name in ['sound_a_reward','sound_b_reward']]
        events_no_rewards = [event.time for event in session.events if event.name in ['sound_a_no_reward','sound_b_no_reward']]

        for e,event in enumerate(events):
            if event.name == 'free_reward_trial':
                free_reward.append(event)
                poke_name.append(events[e+1].name)
                poke_time.append(events[e+1].time)

        fig = plt.figure()

        normalised_task_array = np.zeros((6,160))
        normalised_task_array[:] = np.nan
        for i,neuron in enumerate(neurons):  
            spikes_ind = np.where(session.ephys[0] == neuron)
            spikes_n = spikes[spikes_ind]
           
            for e,event in enumerate(poke_time):
                period_min = event - window_to_plot
                period_max = event + window_to_plot
                spikes_ind = spikes_n[(spikes_n >= period_min) & (spikes_n<= period_max)]
    
                spikes_to_save = (spikes_ind - event) 
                hist_task,edges_task = np.histogram(spikes_to_save, bins= bin_edges_trial)
                normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms)
                normalised_task_array[e,:]= normalised_task*20

            normalised_task_mean = np.mean(normalised_task_array, 0)
            normalised_reward_array = []
            
            for event in events_rewards:
                
                period_min = event - window_to_plot
                period_max = event + window_to_plot
                spikes_ind = spikes_n[(spikes_n >= period_min) & (spikes_n<= period_max)]
    
                spikes_to_save = (spikes_ind - event)   
                hist_task,edges_task = np.histogram(spikes_to_save, bins= bin_edges_trial)                
                normalised_reward = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms)
                normalised_reward_array.append(normalised_reward*20)
                
            normalised_reward_array = np.asarray(normalised_reward_array)
            normalised_reward_array_mean= np.mean(normalised_reward_array, axis = 0)
            normalised_no_reward_array = []
            
            for event in events_no_rewards:
                
                period_min = event - window_to_plot
                period_max = event + window_to_plot
                spikes_ind = spikes_n[(spikes_n >= period_min) & (spikes_n<= period_max)]
    
                spikes_to_save = (spikes_ind - event)   
                hist_task,edges_task = np.histogram(spikes_to_save, bins= bin_edges_trial)

                normalised_no_reward = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms)
                normalised_no_reward_array.append(normalised_no_reward*20)
                
            normalised_no_reward_array = np.asarray(normalised_no_reward_array)
            normalised_no_reward_array_mean= np.mean(normalised_no_reward_array, axis = 0)
            fig.add_subplot(5,6, i+1)
            plt.plot(normalised_reward_array_mean[50:110],color = 'green', label = 'Task Reward')
            plt.plot(normalised_no_reward_array_mean[50:110],color = 'green', linestyle = 'dotted', label = 'Task No Reward')
            plt.plot(normalised_task_mean[50:110],color = 'black', label = 'Off Task Reward')
            
        plt.legend()