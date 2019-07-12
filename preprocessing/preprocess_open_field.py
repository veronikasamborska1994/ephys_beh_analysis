#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:07:41 2019

@author: veronika
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:55:00 2018

@author: behrenslab
"""

import os
import numpy as np
import pandas as pd
import OpenEphys as op 
import Esync as es
import re
import fnmatch
import datetime
from datetime import datetime
from scipy.signal import decimate
import matplotlib.pyplot as plt

def converttime(time):
    #offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds

def uncycle(time):
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128

# Script for synchronising timestamps for video in the openfield and animal behaviour
    
subject = 'm478'

ephys_data_folder ='/media/veronika/My Book/Ephys_Reversal_Learning/data/OpenField Data/m478'
video_dir = '/media/veronika/My Book/Ephys_Reversal_Learning/Video_data/m478/'
video_time = '/media/veronika/My Book/Ephys_Reversal_Learning/Video_data/m478/'

files_ephys = os.listdir(ephys_data_folder)
video_files = os.listdir(video_dir)

spikes_df_csv_out_folder =  '/media/veronika/My Book/Ephys_Reversal_Learning/open_field_neurons/'+ subject

for file_ephys in files_ephys:
    match_ephys = re.search(r'\d{4}-\d{2}-\d{2}', file_ephys)
    date_ephys = datetime.strptime(match_ephys.group(), '%Y-%m-%d').date()
    date_ephys = match_ephys.group()
    for file_behaviour in video_files:
        match_behaviour = re.search(r'\d{4}-\d{2}-\d{2}', file_behaviour)
        date_behaviour = datetime.strptime(match_behaviour.group(), '%Y-%m-%d').date()
        date_behaviour = match_behaviour.group()
        if date_ephys == date_behaviour:
            check_if_open_field =  os.listdir(video_dir+file_behaviour)
            for f in check_if_open_field:
                if 'openfield' in f:
                    dir_video = os.listdir(video_dir+file_behaviour+'/'+'openfield')
                    file_ttl = [i for i in dir_video if 'ttl' in i]
                    file_time = [i for i in dir_video if 'time' in i]

                    ttl_camera = np.loadtxt(open(video_dir+file_behaviour+'/'+'openfield'+'/'+ file_ttl[0]))
                    time_camera = np.loadtxt(open(video_dir+file_behaviour+'/'+'openfield'+'/'+ file_time[0]))
    
                    ephys_path = ephys_data_folder+'/'+file_ephys
                    
                    print(file_behaviour)
                    print(ephys_path)
                    
                    ephys_events = op.loadEvents(os.path.join(ephys_path,'all_channels.events'))  
                    data_folder = file_ephys
                    check_if_kilosort_exists = os.listdir(ephys_path)
                    check_if_npy_exists = os.listdir(spikes_df_csv_out_folder)
                    file_ephys_npy = file_ephys +'.npy'
                    if file_ephys_npy not in check_if_npy_exists:
                        # For multiple files in one session add the requirement of cluster_groups.csv file 
                        for file in check_if_kilosort_exists:
                            if fnmatch.fnmatch(file, '*.tsv'):
                                # Get offset in samples of Kilosort spike data relative to TTL events.
                                with open(os.path.join(ephys_path,'messages.events')) as f:
                                  message = f.read()
                    
                                recording_start_sample, sampling_rate = [
                                    int(x) for x in message.split('start time: ')[1].split('Hz')[0].split('@')]
                
                                # Setup alignment.
                                if data_folder.find(subject) == 5:
                                    box_2_ttls = np.where(ephys_events['channel'] == 3)
                                    if len(box_2_ttls[0]) == 0:
                                        box_2_ttls = np.where(ephys_events['channel'] == 1)
                                    box_2_ttls_fall = box_2_ttls[0][::2]
                                elif  data_folder.find(subject) == 0:
                                    box_2_ttls = np.where(ephys_events['channel'] == 0)
                                    box_2_ttls_fall = box_2_ttls[0][::2]
                                ephys_sync_rise_fall_samples = ephys_events['timestamps']
                                ephys_pulse_samples = ephys_sync_rise_fall_samples[box_2_ttls_fall]
                                ephys_pulse_times = ephys_pulse_samples * 1000/sampling_rate  # ms in ephys reference frame.
                                
                                # Video Timestamps
                                unique_events = np.unique(ttl_camera)
                                index_ttls = np.where(ttl_camera == unique_events[1])[0]
                                time_int = np.asarray(time_camera, dtype = 'int')
                                converted_time = converttime(time_int)
                                uncycled_time = uncycle(converted_time)
                                timestamps_ttls = uncycled_time[index_ttls]
                                timestamps_ttls = np.asarray(timestamps_ttls, dtype = 'int')
                                timestamps_ttls = np.unique(timestamps_ttls) 
                                timestamps_ttls = timestamps_ttls*1000                                     
                                
                                #PyControl
                                aligner = es.Rsync_aligner(timestamps_ttls, ephys_pulse_times, chunk_size = 10, plot=True)
        
                                                                
                                spike_clusters = np.load(os.path.join(ephys_path,'spike_clusters.npy'))
                                spike_samples = np.load(os.path.join(ephys_path,'spike_times.npy'   ))[:,0] # Samples when spikes occured.
                                spike_samples = spike_samples + recording_start_sample # Correct for offset of Kilosort samples and OpenEphys samples.
                                cluster_groups = pd.read_csv(os.path.join(ephys_path,'cluster_group.tsv'), sep='\t')
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
                                
                                #Find multiunit activity 
                                mua_clusters_df = cluster_groups.loc[cluster_groups['group']=='mua', :]
                                mua_cluster_numbers = mua_clusters_df['cluster_id'].values
                                
                                # Filter only rows corresponding to spikes from single units
                                good_spikes_df = df_with_bad_spikes.loc[df_with_bad_spikes['spike_cluster'].isin(good_cluster_numbers), :]
                                good_spikes_np = good_spikes_df['spike_cluster']
                                spike_times_np = good_spikes_df['spike_time']
                                good_spikes_df = np.vstack((good_spikes_np,spike_times_np))
                                
                                # Save good spikes df to np
                                np.save((spikes_df_csv_out_folder + '/'+ data_folder), good_spikes_df)

                      
