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



spikes_df_csv_out_folder =  '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/m486'

data_folder = 'm486_2018-07-10_16-47-52'

ephys_data_folder = '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/m486/m486_2018-07-10_16-47-52'
behaviour_filename = '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys/m486-2018-07-10-164717.txt'


beh_session = di.Session(behaviour_filename)
ephys_events = op.loadEvents(os.path.join(ephys_data_folder,'all_channels.events'))

# Get offset in samples of Kilosort spike data relative to TTL events.

with open(os.path.join(ephys_data_folder,'messages.events')) as f:
  message = f.read()

recording_start_sample, sampling_rate = [
    int(x) for x in message.split('start time: ')[1].split('Hz')[0].split('@')]

# Setup alignment.

ephys_sync_rise_fall_samples = ephys_events['timestamps']
ephys_pulse_samples = ephys_sync_rise_fall_samples[::2] # Ephys samples on which sync pulse rising edge occured.
ephys_pulse_times = ephys_pulse_samples * 1000/sampling_rate  # ms in ephys reference frame.
pycon_pulse_times = beh_session.times['Rsync']                # ms in pycon reference frame.

aligner = es.Rsync_aligner(pycon_pulse_times, ephys_pulse_times, plot=True)


spike_clusters = np.load(os.path.join(ephys_data_folder,'spike_clusters.npy'))
spike_samples = np.load(os.path.join(ephys_data_folder,'spike_times.npy'   ))[:,0] # Samples when spikes occured.
spike_samples = spike_samples + recording_start_sample # Correct for offset of Kilosort samples and OpenEphys samples.
cluster_groups = pd.read_csv(os.path.join(ephys_data_folder,'cluster_groups.csv'), sep='\t')
spike_times_ephys = spike_samples * 1000/sampling_rate # ms in ephys reference frame.
spike_times = aligner.B_to_A(spike_times_ephys) # ms in pycon reference frame.  

# Create spikes data frame 
labels = ['spike_cluster', 'spike_time']
data = [spike_clusters.flatten(), spike_times.flatten()]
data_dict = dict(zip(labels, data))

df_with_bad_spikes = pd.DataFrame(data)
df_with_bad_spikes = df_with_bad_spikes.transpose()
df_with_bad_spikes.rename(columns={0:labels[0], 1:labels[1]}, inplace=True)

# Find single unit clusters
good_clusters_df = cluster_groups.loc[cluster_groups['group']=='good', :]
good_cluster_numbers = good_clusters_df['cluster_id'].values

# Filter only rows corresponding to spikes from single units
good_spikes_df = df_with_bad_spikes.loc[df_with_bad_spikes['spike_cluster'].isin(good_cluster_numbers), :]
good_spikes_df['time'] = good_spikes_df['spike_time']

# Save good spikes df to csv
good_spikes_df.to_csv('/'.join([spikes_df_csv_out_folder, data_folder]) + '.csv')

  
