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
#import datefinder
import re
import fnmatch
import datetime
from datetime import datetime
from scipy.signal import decimate

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
#
## Timestamps for video
#ttls_video = np.loadtxt(open('//media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/Video_data/m484/03_08/openfield/openfield_m484_ttls2018-08-03T15_54_11.csv'))
#npttls_video = np.asarray(ttls_video,dtype=int32)
#unique_events = np.unique(npttls_video)
#index_ttls = np.where(npttls_video ==unique_events[1])[0]
#time_camera = np.loadtxt(open('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/Video_data/m484/03_08/openfield/openfield_m484_timestamps2018-08-03T15_54_11.csv'))
#time_camera = np.asarray(time_camera,dtype=uint32)
#converted_time = converttime(time_camera)
#uncycled_time = uncycle(converted_time)
#timestamps_ttls = uncycled_time[index_ttls]
#timestamps_ttls = np.asarray(timestamps_ttls, dtype = int32)
#timestamps_ttls = np.unique(timestamps_ttls) 
#timestamps_ttls = timestamps_ttls*1000

subject = 'm486'

m486 = ['m483_m486_2018-08-21_13-58-27']
ephys_data_folder ='/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/Multiple_Animals'

files_ephys = os.listdir(ephys_data_folder)

behaviour_filename = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys/'+ subject
files_behaviour = os.listdir(behaviour_filename)
spikes_df_csv_out_folder =  '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons/'+ subject

for file_ephys in files_ephys:
    if file_ephys in m486:
        match_ephys = re.search(r'\d{4}-\d{2}-\d{2}', file_ephys)
        date_ephys = datetime.strptime(match_ephys.group(), '%Y-%m-%d').date()
        date_ephys = match_ephys.group()
        for file_behaviour in files_behaviour:
            match_behaviour = re.search(r'\d{4}-\d{2}-\d{2}', file_behaviour)
            date_behaviour = datetime.strptime(match_behaviour.group(), '%Y-%m-%d').date()
            date_behaviour = match_behaviour.group()
            if date_ephys == date_behaviour:
                behaviour_path = behaviour_filename+'/'+file_behaviour
                behaviour_session = di.Session(behaviour_path)
                ephys_path = ephys_data_folder+'/'+file_ephys + '/' + subject
                print(behaviour_path)
                print(ephys_path)
                ephys_events = op.loadEvents(os.path.join(ephys_path,'all_channels.events'))  
                data_folder = file_ephys
                check_if_kilosort_exists = os.listdir(ephys_path)
                check_if_npy_exists = os.listdir(spikes_df_csv_out_folder)
                file_ephys_npy = file_ephys +'.npy'
                if file_ephys_npy not in check_if_npy_exists:
                    # For multiple files in one session add the requirement of cluster_groups.csv file 
                    for file in check_if_kilosort_exists:
                        if fnmatch.fnmatch(file, '*.csv'):
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
                            pycon_pulse_times = behaviour_session.times['Rsync']  # ms in pycon reference frame.
                            
                            #PyControl
                            aligner = es.Rsync_aligner(pycon_pulse_times, ephys_pulse_times, plot=True)
                            
                            LFP_dir = spikes_df_csv_out_folder + '/' + 'LFP'
                            MUA_dir = spikes_df_csv_out_folder + '/' + 'MUA'
                            if not os.path.exists(LFP_dir):
                                os.makedirs(LFP_dir)
                            
                            if not os.path.exists(MUA_dir):
                                os.makedirs(MUA_dir)
                                
                                
                            #Extract LFP 
                            n_channels = 32
                            all_channels = []
                            time = []
                            for file in os.listdirephys_path():
                                if file.endswith('.dat'):
                                    tmp = np.memmap(file, dtype = np.int16)
                                    shp = int(len(tmp)/n_channels)
                                    lfp = np.memmap(file, dtype = np.int16, shape = (shp,n_channels))
                                    
                            for channel in range(n_channels):
                                channel_lfp = lfp[:,channel]
                                downsample_3000 = decimate(channel_lfp, 10, zero_phase = True)
                                downsample_500 = decimate(downsample_3000, 6, zero_phase = True)
                                recording_offset_add = downsample_500 + recording_start_sample
                                recording_ms_frame = 1000/sampling_rate
                                # Convert LFP time
                                time_channel = aligner.B_to_A(recording_ms_frame)
                                all_channels.append(downsample_500)
                                time.append(time_channel)
                                
                            all_channels = np.asarray(all_channels)
                            time = np.asarray(time)
                            
                            lfp_to_save = np.vstack((time,all_channels))
                            
                            spike_clusters = np.load(os.path.join(ephys_path,'spike_clusters.npy'))
                            spike_samples = np.load(os.path.join(ephys_path,'spike_times.npy'   ))[:,0] # Samples when spikes occured.
                            spike_samples = spike_samples + recording_start_sample # Correct for offset of Kilosort samples and OpenEphys samples.
                            cluster_groups = pd.read_csv(os.path.join(ephys_path,'cluster_groups.csv'), sep='\t')
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
                            
                            # Filter only rows corresponding to spikes from multiunit

                            mua_spikes_df = df_with_bad_spikes.loc[df_with_bad_spikes['spike_cluster'].isin(mua_cluster_numbers), :]
                            mua_spikes_np = mua_spikes_df['spike_cluster']
                            mua_spike_times_np = mua_spikes_df['spike_time']
                            mua_spikes_df = np.vstack((mua_spikes_np,mua_spike_times_np))
                            
                            # Save good spikes df to np
                            np.save((spikes_df_csv_out_folder + '/'+ data_folder), good_spikes_df)
                            np.save((LFP_dir + '/'+ data_folder), lfp_to_save)
                            np.save((MUA_dir + '/'+ data_folder), mua_spikes_df)

                              
