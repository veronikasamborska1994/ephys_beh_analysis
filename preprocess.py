#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:55:00 2018

@author: behrenslab
"""

import os
import numpy as np
import pandas as pd
import data_import as di
import OpenEphys as op 
import Esync as es


#mport NeuroTools.signals as nt

spikes_df_csv_out_folder =  '/Users/veronikasamborska'
sampling_rate = 30
recordings_to_extract = ['2018-06-07_16-15-43']
kilosort_folder = '/Volumes/My Passport/code'
beh_session = di.Session('/Volumes/My Passport/code/2018-06-05-reversal_learning_3_tasks_recording/m483-2018-06-07-161545.txt')

#Correct timestamps 
ephys_events = op.loadEvents('/Volumes/My Passport/code/2018-06-07_16-15-43/all_channels.events')
pyControl_events = [event.time for event in beh_session.events if event.name in ['Rsync']]
ephys_events = ephys_events['timestamps']
ephys_events= np.array(ephys_events)[::2]
pyControl_events = np.array(pyControl_events)
ephys_events =ephys_events/30
aligner = es.Rsync_aligner(pyControl_events, ephys_events, plot=True)


for recording_to_extract in recordings_to_extract:
    path_to_data = '/'.join([kilosort_folder, recording_to_extract])
    os.chdir(path_to_data)
    
    # load kilosort data files
    spike_clusters = np.load('spike_clusters.npy')
    spike_times = np.load('spike_times.npy')

    cluster_groups = pd.read_csv('cluster_groups.csv', sep='\t')
    channel_positions = np.load('channel_positions.npy')
    try:  # check data quality
        assert np.shape(spike_times.flatten()) == np.shape(spike_clusters)
    except:
        AssertionError('Array lengths do not match in recording {}'.format(recording_to_extract))

 
    #Find single unit clusters
    good_clusters_df = cluster_groups.loc[cluster_groups['group']=='good', :]
    good_cluster_numbers = good_clusters_df['cluster_id'].values

    # create spikes data frame 
    labels = ['spike_cluster', 'spike_time']
    data = [spike_clusters.flatten(), spike_times.flatten()]
    data_dict = dict(zip(labels, data))
    
    df_with_bad_spikes = pd.DataFrame(data)
    df_with_bad_spikes = df_with_bad_spikes.transpose()
    df_with_bad_spikes.rename(columns={0:labels[0], 1:labels[1]}, inplace=True)
    
    # filter only rows corresponding to spikes from single units
    good_spikes_df = df_with_bad_spikes.loc[df_with_bad_spikes['spike_cluster'].isin(good_cluster_numbers), :]
    good_spikes_df['time'] = good_spikes_df['spike_time'].divide(sampling_rate)
       #Convert TimeStamps to PyControl Time
    good_spikes_df['time'] = aligner.A_to_B(good_spikes_df['time'])
    # save good spikes df to csv
    good_spikes_df.to_csv('/'.join([spikes_df_csv_out_folder, recording_to_extract]) + '.csv')
    
  
