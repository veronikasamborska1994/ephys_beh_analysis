#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:32:59 2018

@author: veronikasamborska
"""

import os
import numpy as np
import pandas as pd
import data_import as di
import OpenEphys as op 
import Esync as es
import funcs as fu 
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
     
    return z

recordings_to_extract = ['m481_2018-06-21_19-34-16']#,'m481_2018-06-27_15-54-04', 'm481_2018-06-29_15-32-56', 'm481_2018-07-01_16-53-43']
kilosort_folder= '/Users/veronikasamborska/Desktop/Ephys 3 Tasks Processed Spikes/m481'
sessions = ['m481-2018-06-21-193404.txt']#,'m481-2018-06-27-155357.txt','m481-2018-06-29-153244.txt','m481-2018-07-01-165322.txt']

recordings_to_extract_hp = ['m483_2018-06-07_16-15-43']
sessions_hp = ['m483-2018-06-07-161545.txt']

bin_width_ms = 1
smooth_sd_ms = 100
fr_convert = 1000
trial_duration = 2000
bin_edges_trial = np.arange(-1000,trial_duration, bin_width_ms)
max_list = []

for recording_to_extract,session in zip(recordings_to_extract,sessions):
    path_to_data = '/'.join([kilosort_folder, recording_to_extract])
    os.chdir(path_to_data)
    ephys_session = fu.load_data(recording_to_extract,kilosort_folder,'/',True )
    beh_session = di.Session('/Users/veronikasamborska/Desktop/data_3_tasks_ephys/{}'.format(session))
    
    #Trial Initiation Timestamps
    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    clusters = ephys_session['spike_cluster'].unique()
    new_matrix = np.ones(shape=(len(clusters),2999))
    cluster_list = []
    for i,cluster in enumerate(clusters):
        for choice in pyControl_choice:
            spikes_to_save = 0
            spikes_to_plot = []
            period_min = choice - 1000
            period_max = choice + 3000
            spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
            spikes_times = np.array(spikes['time'])
            spikes_ind = spikes_times[(spikes_times >= period_min) & (spikes_times<= period_max)]
            spikes_to_save = (spikes_ind - choice)          
            spikes_to_plot.append(spikes_to_save) 
            hist,edges = np.histogram(spikes_to_plot, bins= bin_edges_trial)
            normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
            #normalised = normalised*1000
            #bin_edges_trial, spikes_to_plot = np.meshgrid(bin_edges_trial, spikes_to_plot)

            #hist = hist/bin_width_ms 
        cluster_list.append(normalised)
    cluster_list = np.array(cluster_list)
    for cluster in cluster_list:
        max_list.append(np.argmax(cluster))
    for m in range(len(max_list)):
        ind = np.argmin(max_list)
        max_list[ind] = 1000000
        my_row = cluster_list[ind,:]
        normalized = (my_row-min(my_row))/(max(my_row) - min(my_row)+1e-10)
        new_matrix[m,:] = normalized
        
    a = sns.heatmap(new_matrix)
    a.set_title('PFC')
    a.set_xlabel('Time (ms)')
    a.set_ylabel(' Neuron #')
    #pl.xlim(+2000, +6000)
        
        
    
    