#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:13:18 2018

@author: behrenslab
"""
import funcs as fu
import data_import as di
import numpy as np
import pylab as pl
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
import statsmodels
#Import Ephys and PyControl Data
ephys_session = fu.load_data('2018-06-07_16-15-43','/Users/veronikasamborska/Desktop/code','/',True)
beh_session = di.Session('/Volumes/My Passport/code/2018-06-05-reversal_learning_3_tasks_recording/m483-2018-06-07-161545.txt')
Iti = 1.75

forced_trials = beh_session.trial_data['forced_trial']
non_forced_array = np.where(forced_trials == 0)[0]
trials = beh_session.trial_data['trials']    
non_forced_trials = trials[non_forced_array]
print(len(non_forced_trials))
original_trials = beh_session.trial_data['trials'] 
configuration = beh_session.trial_data['configuration_i']
sessions_block = beh_session.trial_data['block']
Block_transitions = sessions_block[1:] - sessions_block[:-1] 
#reversal_trials = np.where(Block_transitions == 1)[0]
outcomes= beh_session.trial_data['outcomes']
task = beh_session.trial_data['task']
task_non_forced = task[non_forced_array]
trial_l =len(trials)

#Create timestamps for events within a trial
pyControl_trial_start = [event.time for event in beh_session.events if event.name in ['init_trial']][0:trial_l]
pyControl_trial_start = np.array(pyControl_trial_start)
pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
pyControl_choice = np.array(pyControl_choice)
pyControl_forced_trials = [event.time for event in beh_session.events if event.name in ['a_forced_state','b_forced_state']]
pyControl_b_poke = [event.time for event in beh_session.events if event.name in ['sound_b_no_reward', 'sound_b_reward']]
pyControl_b_poke =np.array(pyControl_b_poke)
pyControl_a_poke = [event.time for event in beh_session.events if event.name in ['sound_a_no_reward','sound_a_reward']]
pyControl_end_trial = [event.time for event in beh_session.events if event.name in ['inter_trial_interval']][2:] #first two ITIs are free rewards
pyControl_end_trial = np.array(pyControl_end_trial)
#Task 1 Time Events

task_non_forced = np.where(task == 1)[0]

trial_сhoice_state_task_NF = pyControl_choice[task_non_forced]

ITI_task_1 = pyControl_end_trial[non_forced_array]#[2:]
ITI_task_1 = ITI_task_1[task_non_forced]


clusters = ephys_session['spike_cluster'].unique()
clusters = clusters[10:]
print(clusters)
nclusters = len(clusters)
#for cluster in clusters:
spikes_to_plot = []
iti_spikes=[]
trial_l = len(trial_сhoice_state_task_NF)
#spikes = ephys_session.loc[ephys_session['spike_cluster'] == 53]
#spikes_times = np.array(spikes['spike_time'])
#spikes_times = spikes_times/30
spikes_to_plot = np.array([])
fig, axes = plt.subplots(figsize=(50,5), ncols=9, nrows=1, sharex=True)
plt.tight_layout()
for i,cluster in enumerate(clusters): 
    spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
    spikes_times = np.array(spikes['spike_time'])
    spikes_times = spikes_times/30
    spikes_to_save = 0
    spikes_to_plot = np.array([])
    for trial, iti in zip(trial_сhoice_state_task_NF,ITI_task_1):
        spikes_ind = spikes_times[(spikes_times >= (trial -6000)) & (spikes_times<=trial+6000)]
        spikes_to_save = (spikes_ind - trial)/1000
        spikes_to_plot= np.append(spikes_to_plot,spikes_to_save)
    sns.set(style="white", palette="muted", color_codes=True)
    #sns.kdeplot(spikes_to_plot,bw = 0.5, ax=axes[i])
    sns.distplot(spikes_to_plot, hist=True, bins = 36,color="g", kde = True, ax=axes[i])
    #n,bins,patches = axes[i].hist(spikes_to_plot, bins=36)
    print(bins)
    #sns.rugplot(spikes_to_plot)
    axes[i].set_title('{}'.format(cluster))
    pl.xlim(-3, +3)
    print(np.mean(n))
    
    #if i>10:
       # sns.set(style="white", palette="muted", color_codes=True)
        #sns.distplot(spikes_to_plot, hist=True, color="g", kde_kws={"shade": True}, ax=axes[i][1])
       # axes[i].set_title('{}'.format(cluster))
        #pl.title('{}'.format(cluster))
       # pl.xlim(-3, +3)

#print(spikes_to_plot)
#sns.distplot(spikes_to_plot, kde=False, hist = True)
#pl.xlim(-2, +2)
#pl.figure()
#print(spikes_to_plot)
#
