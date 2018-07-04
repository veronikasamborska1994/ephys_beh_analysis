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



spikes_df_csv_out_folder =  '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Processed Spikes/m481'
sampling_rate = 30
recordings_to_extract = ['m481_2018-06-29_15-32-56']
kilosort_folder = '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/m481'
beh_session = di.Session('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys/m481-2018-06-29-153244.txt')

# TTLs from OpenEphys
ephys_events = op.loadEvents('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/m481/m481_2018-06-29_15-32-56/all_channels.events')
pyControl_events = [event.time for event in beh_session.events if event.name in ['Rsync']]
ephys_events = ephys_events['timestamps']

# OpenEphys rising events in sampling time 
ephys_events= np.array(ephys_events)[::2]

# PyControl events
pyControl_events = np.array(pyControl_events)
pyControl_events_ephys_time = pyControl_events*30 
aligner_ephys_to_pyControl = es.Rsync_aligner(pyControl_events_ephys_time, ephys_events, plot = False)
#spike_times_A_poke =np.load('/Users/veronikasamborska/Desktop/poke_A_list.npy')
#cluster_100 = np.repeat(100,len(spike_times_A_poke))
#spike_times_A_poke = spike_times_A_poke*30
#spike_times_A_poke = aligner_ephys_to_pyControl.B_to_A(spike_times_A_poke)

    # Save good spikes df to csv
for recording_to_extract in recordings_to_extract:
    path_to_data = '/'.join([kilosort_folder, recording_to_extract])
    os.chdir(path_to_data)
    
    # Load kilosort data files
    spike_clusters = np.load('spike_clusters.npy')
    spike_times = np.load('spike_times.npy')
    #spike_times = np.append(spike_times,spike_times_A_poke)
    #cluster = np.repeat(100, len(spike_times_A_poke))
    #spike_clusters = np.append(spike_clusters, cluster)
    cluster_groups = pd.read_csv('cluster_groups.csv', sep='\t')
    channel_positions = np.load('channel_positions.npy')
    try:  # check data quality
        assert np.shape(spike_times.flatten()) == np.shape(spike_clusters)
    except:
        AssertionError('Array lengths do not match in recording {}'.format(recording_to_extract))

    # Create spikes data frame 
    labels = ['spike_cluster', 'spike_time']
    data = [spike_clusters.flatten(), spike_times.flatten()]
    data_dict = dict(zip(labels, data))
    
    df_with_bad_spikes = pd.DataFrame(data)
    df_with_bad_spikes = df_with_bad_spikes.transpose()
    df_with_bad_spikes.rename(columns={0:labels[0], 1:labels[1]}, inplace=True)
    
    # Find single unit clusters
    good_clusters_df = cluster_groups.loc[cluster_groups['group']=='good', :]
    #df_good = pd.DataFrame({'cluster_id':[100], 'group': ['good']})
    #good_clusters_df = pd.concat([good_clusters_df,df_good])
    good_cluster_numbers = good_clusters_df['cluster_id'].values
    
    # Filter only rows corresponding to spikes from single units
    good_spikes_df = df_with_bad_spikes.loc[df_with_bad_spikes['spike_cluster'].isin(good_cluster_numbers), :]
    good_spikes_df['time'] = good_spikes_df['spike_time']#.divide(sampling_rate)

    #Convert TimeStamps of A_Pokes to OpenEphys
    
    #Convert TimeStamps to PyControl Time
    good_spikes_df['time'] = aligner_ephys_to_pyControl.A_to_B(good_spikes_df['time'])
    good_spikes_df['time'] = good_spikes_df['time'].divide(sampling_rate)
    # Save good spikes df to csv
    good_spikes_df.to_csv('/'.join([spikes_df_csv_out_folder, recording_to_extract]) + '.csv')
    
  
