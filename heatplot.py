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

# Change to m483 is HP, m483 if PFC
kilosort_folder= '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Processed Spikes/m483'

recordings_to_extract = ['m481_2018-06-20_19-09-08', 'm481_2018-06-21_19-34-16',
                         'm481_2018-06-27_15-54-04', 'm481_2018-06-29_15-32-56', 'm481_2018-07-01_16-53-43']



sessions = ['m481-2018-06-20-190858.txt','m481-2018-06-21-193404.txt'
            ,'m481-2018-06-27-155357.txt', 'm481-2018-06-29-153244.txt',
            'm481-2018-07-01-165322.txt']



recordings_to_extract_hp = ['m483_2018-06-07_16-15-43','m483_2018-06-08_15-55-29', 'm483_2018-06-09_18-21-19', 'm483_2018-06-11_18-54-05',
                            'm483_2018-06-12_18-31-25','m483_2018-06-14_17-24-44','m483_2018-06-15_16-47-24',
                            'm483_2018-06-18_17-18-55', 'm483_2018-06-20_17-25-20','m483_2018-06-21_17-39-58',
                            'm483_2018-06-22_16-00-17','m483_2018-06-25_16-40-47']

sessions_hp = ['m483-2018-06-07-161545.txt', 'm483-2018-06-08-155401.txt', 'm483-2018-06-09-182107.txt', 'm483-2018-06-11-185036.txt',
               'm483-2018-06-12-183143.txt', 'm483-2018-06-14-172430.txt','m483-2018-06-15-164724.txt',
               'm483-2018-06-18-171848.txt','m483-2018-06-20-172510.txt','m483-2018-06-21-173958.txt','m483-2018-06-22-160006.txt',
               'm483-2018-06-25-164041.txt']

bin_width_ms = 1
bin_width_ms_session = 50 
smooth_sd_ms = 100
fr_convert = 1000
trial_duration = 2000
bin_edges_trial = np.arange(0 ,trial_duration, bin_width_ms)
max_list = []
cluster_list = [] 
start_trial_list = []
end_trial_list = []
spikes_times_session_list =[]
for recording_to_extract,session in zip(recordings_to_extract_hp,sessions_hp):
    path_to_data = '/'.join([kilosort_folder, recording_to_extract])
    os.chdir(path_to_data)
    ephys_session = fu.load_data(recording_to_extract,kilosort_folder,'/',True )
    beh_session = di.Session('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys/{}'.format(session))
    forced_trials = beh_session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = beh_session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    #Trial Initiation Timestamps
    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    pyControl_end_trial = [event.time for event in beh_session.events if event.name in ['inter_trial_interval']][2:] #first two ITIs are free rewards
    pyControl_end_trial = np.array(pyControl_end_trial)
    task = beh_session.trial_data['task']
    task_1_end_trial = np.where(task == 1)[0]
    task_2_end_trial = np.where(task == 2)[0]
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    poke_A = 'poke_'+str(beh_session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(beh_session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(beh_session.trial_data['poke_A'][task_3_change[0]])
    poke_A_exit = 'poke_'+str(beh_session.trial_data['poke_A'][0]) + '_out'
    poke_B = 'poke_'+str(beh_session.trial_data['poke_B'][0])
    poke_B_exit = 'poke_'+str(beh_session.trial_data['poke_B'][0]) + '_out'
    poke_B_task_2  = 'poke_'+str(beh_session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(beh_session.trial_data['poke_B'][task_3_change[0]])
    
    poke_B_list = []
    poke_A_list = []
    choice_state = False 
    choice_state_count = 0

    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]
    
    events_and_times = [[event.name, event.time] for event in beh_session.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    for event in events_and_times:
        #choice_state_count = 0
        if 'choice_state' in event:
            choice_state_count +=1 
            choice_state = True   
        if choice_state_count <= len(task_1):
            if poke_B in event: 
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False
        elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
            if poke_B_task_2 in event:
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A_task_2 in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False     
        elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
            if poke_B_task_3 in event:
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A_task_3 in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False
    all_events = poke_A_list + poke_B_list            
    
    start_trial_list.append(pyControl_choice)
    end_trial_list.append(all_events)
    
    clusters = ephys_session['spike_cluster'].unique()
    neuron_spike_times = []
    bin_edges_session = np.arange(0, beh_session.events[-1].time,50)
    new_clusters = []
    for i,cluster in enumerate(clusters):
        spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
        spikes_times = np.array(spikes['time'])
        session_length = int((np.nanmax(spikes_times))/2)
        spikes_times_list =[~np.isnan(spikes_times)]
        mean_firing = (np.count_nonzero(spikes_times)/session_length)* 1000
        if mean_firing < 20:
            new_clusters.append(cluster)

    for i,cluster in enumerate(new_clusters):
        for choice in pyControl_choice:
            spikes_to_save = 0
            spikes_to_plot = []
            period_min = choice 
            period_max = choice + 2000
            spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
            spikes_times = np.array(spikes['time'])
#            session_length = int((np.nanmax(spikes_times))/2)
#            spikes_times_list =[~np.isnan(spikes_times)]
#            spikes_times_session = np.histogram(spikes_times_list, bins = bin_edges_session)            
            spikes_ind = spikes_times[(spikes_times >= period_min) & (spikes_times<= period_max)]
            spikes_to_save = (spikes_ind - choice)          
            spikes_to_plot.append(spikes_to_save) 
            hist,edges = np.histogram(spikes_to_plot, bins= bin_edges_trial)
            normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
        #neuron_spike_times.append(spikes_times_session)
        cluster_list.append(normalised)
    #spikes_times_session_list.append(neuron_spike_times)
        

spikes_times_session = np.array(spikes_times_session_list)

start_trial_list = np.array(start_trial_list)
end_trial_list = np.array(end_trial_list)
cluster_list = np.array(cluster_list)
peak_inds = np.argmax(cluster_list,1)
ordering = np.argsort(peak_inds)
activity_sorted = cluster_list[ordering,:]
norm_activity_sorted = activity_sorted / np.max(activity_sorted,1)[:, None]

np.save('/home/behrenslab/Desktop/spikes_times_session.npy', spikes_times_session)
np.save('/home/behrenslab/Desktop/start_trial_list.npy', start_trial_list)
np.save('/home/behrenslab/Desktop/end_trial_list.npy', end_trial_list)


plt.grid(False)
plt.imshow(norm_activity_sorted[144:][:], aspect='auto')
plt.title('HP')
plt.xlabel('Time (ms)')
plt.ylabel(' Neuron #')

        
        
    
    