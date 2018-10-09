#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:03:19 2018

@author: behrenslab
"""
import copy 
import os
import numpy as np
import pandas as pd
import data_import as di
import OpenEphys as op 
import Esync as es
import datefinder
import re
import fnmatch
import datetime
from datetime import datetime
import align_activity as aa
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import pylab as pl

import ephys_beh_import as ep


def _CPD(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression()
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd

def plot_firing_rate_time_course(experiment):
    for session in experiment:
        predictor_A_Task_1,  predictor_A_Task_2,  predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3 = predictors_f(session)
        aligned_spikes= session.aligned_rates 
        n_neurons = aligned_spikes.shape[1]
        n_trials = aligned_spikes.shape[0]
        spikes_B_task_1 =aligned_spikes[np.where(predictor_B_Task_1 ==1)]
        spikes_A_task_1 =aligned_spikes[np.where(predictor_A_Task_1 ==1)]
        spikes_B_task_2 =aligned_spikes[np.where(predictor_B_Task_2 ==1)]
        spikes_A_task_2 =aligned_spikes[np.where(predictor_A_Task_2 ==1)]
        spikes_B_task_3 =aligned_spikes[np.where(predictor_B_Task_3 ==1)]
        spikes_A_task_3 =aligned_spikes[np.where(predictor_B_Task_3 ==1)]
        fig, axes = plt.subplots(figsize = (15,5), ncols = n_neurons , sharex=True, sharey = 'col')
        mean_spikes_B_task_1 = np.mean(spikes_B_task_1,axis = 0)
        mean_spikes_A_task_1 = np.mean(spikes_A_task_1,axis = 0)
        mean_spikes_B_task_2 = np.mean(spikes_B_task_2,axis = 0)
        mean_spikes_A_task_2 = np.mean(spikes_A_task_2,axis = 0)
        mean_spikes_B_task_3 = np.mean(spikes_B_task_3,axis = 0)
        mean_spikes_A_task_3 = np.mean(spikes_A_task_3,axis = 0)
        for neuron in range(n_neurons):
            axes[neuron].plot(mean_spikes_B_task_1[neuron], label = 'B Task 1')
            axes[neuron].plot(mean_spikes_A_task_1[neuron], label = 'A Task 1')
            axes[neuron].plot(mean_spikes_B_task_2[neuron], label = 'B Task 2')
            axes[neuron].plot(mean_spikes_A_task_2[neuron], label = 'A Task 2')
            axes[neuron].plot(mean_spikes_B_task_3[neuron], label = 'B Task 3')
            axes[neuron].plot(mean_spikes_A_task_3[neuron], label = 'A Task 3')
        axes[0].legend()
        plt.title('{}'.format(session.file_name))
        
def predictors_f(session):
    n_trials = session.aligned_rates.shape[0]
    print(n_trials)
    choices = session.trial_data['choices']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
    choice_non_forced = choices[non_forced_array]
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    poke_A = session.trial_data['poke_A']
    poke_B = session.trial_data['poke_B']
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3  = ep.extract_choice_pokes(session)
    
    #Task 1 
    choices_a = np.where(choice_non_forced == 1)
    choices_b = np.where(choice_non_forced == 0)
    
    predictor_a = np.zeros([1,n_trials])
    predictor_a[0][choices_a[0]] = 1
    predictor_b = np.zeros([1,n_trials])
    predictor_b[0][choices_b[0]] = 1
    
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)
    # Task 2
    # If Poke A in task 2 is the same as in task 1 keep it 
    predictor_a_1 = copy.copy(predictor_a)
    predictor_a_1[0][len(task_1):] = 0
    predictor_a_2 =  copy.copy(predictor_a)
    predictor_a_2[0][:len(task_1)] = 0
    predictor_a_2[0][len(task_1)+len(task_2):] = 0 
    predictor_a_3 =  copy.copy(predictor_a)
    predictor_a_3[0][:len(task_1)+len(task_2)] = 0 
    
    predictor_b_1 =  copy.copy(predictor_b)
    predictor_b_1[0][len(task_1):] = 0
    predictor_b_2 = copy.copy(predictor_b)
    predictor_b_2[0][:len(task_1)] = 0
    predictor_b_2[0][len(task_1)+len(task_2):] = 0 
    predictor_b_3 = copy.copy(predictor_b)
    predictor_b_3[0][:len(task_1)+len(task_2)] = 0
    
    if poke_A1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictorfiner_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
    elif poke_A1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
    elif poke_A1_B2_A3 == True: 
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
    elif poke_A1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
    elif poke_B1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
    elif poke_B1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
    elif poke_B1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
    elif poke_B1_B2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3
        
def regression(experiment):
    for session in experiment:
        print(session.file_name)
        C = []    # To strore predictor loadings for each session.
        cpd = []  # To strore cpd for each session.
        aligned_spikes= session.aligned_rates 
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
        
        predictor_A_Task_1,  predictor_A_Task_2,  predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3 = predictors_f(session)
        predictors = OrderedDict([
                                      ('a_task_1' , predictor_A_Task_1),
                                      ('a_task_2' , predictor_A_Task_2),
                                      ('a_task_3' , predictor_A_Task_3),
                                      ('b_task_1' , predictor_B_Task_1),
                                      ('b_task_2' , predictor_B_Task_2),
                                      ('b_task_3' , predictor_B_Task_3)])
            
        X = np.vstack(predictors.values()).T[:n_trials,:].astype(float)
        n_predictors = X.shape[1]
        y = aligned_spikes.reshape([n_trials,-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        ols = LinearRegression()
        ols.fit(X,y)
        C.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
        C = np.concatenate(C,0)
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
        cpd = np.mean(np.concatenate(cpd,0), axis = 0) # Population CPD is mean over neurons.
        plt.figure()
        for i, predictor in enumerate(predictors):
            plt.subplot(3,np.ceil(len(predictors)/3),i+1)
            plt.imshow(C[:,:,i])
            plt.colorbar()
        plt.figure()
        for i, predictor in enumerate(predictors):
            t = session.t_out
            plt.plot(t, 100*cpd[:,i], label=predictor)
            plt.title('{}'.format( session.file_name))
        
        plt.ylabel('Coef. partial determination (%)')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.xlim(t[0], t[-1])
        plt.ylim(ymin=0)

        
def target_times_f(experiment):
    trial_times_all_trials  = []
    for session in experiment:
        init_times = session.times['choice_state']
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'choice_state'])
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  =np.asarray(trial_times_all_trials)
    target_times = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times_all_trials[0],1),0))))
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are using the median timings from this session.
    return target_times

def all_sessions_aligment(experiment):
    target_times = target_times_f(experiment)
    experiment_aligned = []
    for session in experiment:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        init_times = session.times['choice_state']
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'choice_state'])
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times, spikes)
        session.aligned_rates = aligned_rates
        session.t_out = t_out
        experiment_aligned.append(session)
    return experiment_aligned

#ephys_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons'
#beh_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys'
#
#
HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path)
experiment_aligned = all_sessions_aligment(HP)

def heatplot_aligned(experiment_aligned): 
    all_clusters_task_1 = []
    all_clusters_task_2 = []
    all_clusters_task_3 = []
    for session in experiment_aligned:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        cluster_list_task_1 = []
        cluster_list_task_2 = []
        cluster_list_task_3 = []
        aligned_rates = session.aligned_rates
        trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3 = ep.initiation_and_trial_end_timestamps(session)
        task_1 = len(trial_сhoice_state_task_1)
        task_2 = len(trial_сhoice_state_task_2)
        task_3 = len(trial_сhoice_state_task_3)
        aligned_rates_task_1 = aligned_rates[:task_1]
        aligned_rates_task_2 = aligned_rates[task_1:task_1+task_2]
        aligned_rates_task_3 = aligned_rates[task_1+task_2:]
        unique_neurons  = np.unique(spikes[0])
        for i in range(len(unique_neurons)):
            mean_firing_rate_task_1  = np.mean(aligned_rates_task_1[:,i,:],0)
            mean_firing_rate_task_2  = np.mean(aligned_rates_task_2[:,i,:],0)
            cluster_list_task_1.append(mean_firing_rate_task_1) 
            cluster_list_task_2.append(mean_firing_rate_task_2)
        all_clusters_task_1.append(cluster_list_task_1[:])
        all_clusters_task_2.append(cluster_list_task_2[:])
    all_clusters_task_1 = np.array(all_clusters_task_1)
    all_clusters_task_2 = np.array(all_clusters_task_2)
    same_shape_task_1 = []
    same_shape_task_2 = []
    same_shape_task_3 = []
    for i in all_clusters_task_1:
        for ii in i:
            same_shape_task_1.append(ii)
    for i in all_clusters_task_2:
        for ii in i:
            same_shape_task_2.append(ii)
    same_shape_task_1 = np.array(same_shape_task_1)
    same_shape_task_2 = np.array(same_shape_task_2)
    peak_inds = np.argmax(same_shape_task_1,1)
    ordering = np.argsort(peak_inds)
    activity_sorted = same_shape_task_2[ordering,:]
    ten_percent = int( activity_sorted.shape[0]*0.15)
    not_normed = same_shape_task_1[ordering,:]
    not_normed += 1
    not_normed = np.log(not_normed)
    norm_activity_sorted = (activity_sorted - np.min(activity_sorted,1)[:, None]) / (np.max(activity_sorted,1)[:, None] - np.min(activity_sorted,1)[:, None])
    where_are_Nans = isnan(norm_activity_sorted)
    #norm_activity_sorted[where_are_Nans] = 0
    plt.imshow(not_normed[ten_percent:], aspect='auto')  
    plt.colorbar()
    



