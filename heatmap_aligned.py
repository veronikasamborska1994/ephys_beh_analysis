#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:03:19 2018

@author: behrenslab
"""
import copy 
import numpy as np
import pandas as pd
import data_import as di
import OpenEphys as op 
import Esync as es
import align_activity as aa
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import itertools
from scipy.spatial.distance import cdist
from scipy.spatial.distance import correlation
from scipy.spatial.distance import seuclidean
import pylab as pl

from scipy.stats import pearsonr
from itertools import combinations
import ephys_beh_import as ep

import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

#ephys_path = '/Users/veronikasamborska/Desktop/neurons'
#beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
  
#HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path)
#experiment_aligned_HP = all_sessions_aligment(HP)
#experiment_aligned_PFC = all_sessions_aligment(PFC)

#a1_a2_all_neurons_hp, a2_a3_all_neurons_hp, a1_a3_all_neurons_hp, b1_b2_all_neurons_hp, b2_b3_all_neurons_hp, b1_b3_all_neurons_hp =  correlation_trials(experiment_aligned_HP)
#a1_a2_all_neurons_pfc, a2_a3_all_neurons_pfc, a1_a3_all_neurons_pfc, b1_b2_all_neurons_pfc, b2_b3_all_neurons_pfc, b1_b3_all_neurons_pfc=  correlation_trials(experiment_aligned_PFC)



#flattened__hp_list_a2_a3 = np.array(flattened__hp_list_a2_a3)
#flattened__hp_list_a2_a3 = flattened__hp_list_a2_a3[~np.isnan(flattened__hp_list_a2_a3)]
#
#bins_a = int(len(flattened_list_a2_a3)/100)
#hist(flattened_list_a2_a3, bins = bins_a)
#
#
#flattened_hp_list_a1_a2 = []
#flattened__hp_list_a2_a3 = []
#for x in a1_a2_all_neurons_hp:
#    for y in x:
#        flattened_hp_list_a1_a2.append(y)
#
#for x in a2_a3_all_neurons_hp:
#    for y in x:
#        flattened__hp_list_a2_a3.append(y)    
#        
#        
#
#flattened_list_hp_b1_b2 = []
#flattened_list_hp_b2_b3 = []
#for x in b1_b2_all_neurons_hp:
#    for y in x:
#        flattened_list_hp_b1_b2.append(y)
#
#for x in b2_b3_all_neurons_hp:
#    for y in x:
#        flattened_list_hp_b2_b3.append(y)    
#        
#        
#flattened_pfc_list_a1_a2 = []
#flattened__pfc_list_a2_a3 = []
#for x in a1_a2_all_neurons_pfc:
#    for y in x:
#        flattened_pfc_list_a1_a2.append(y)
#
#for x in a2_a3_all_neurons_pfc:
#    for y in x:
#        flattened__pfc_list_a2_a3.append(y)    
#            
#flattened_pfc_list_b1_b2 = []
#flattened__pfc_list_b2_b3 = []
#for x in b1_b2_all_neurons_pfc:
#    for y in x:
#        flattened_pfc_list_b1_b2.append(y)
#
#for x in b2_b3_all_neurons_pfc:
#    for y in x:
#        flattened__pfc_list_b2_b3.append(y)    
#            
#flattened_pfc_list_a1_a2 = np.array(flattened_pfc_list_a1_a2)
#flattened_pfc_list_a1_a2 = flattened_pfc_list_a1_a2[~np.isnan(flattened_pfc_list_a1_a2)]
#flattened_pfc_list_a2_a3 = np.array(flattened__pfc_list_a2_a3)
#flattened_pfc_list_a2_a3 = flattened_pfc_list_a2_a3[~np.isnan(flattened__pfc_list_a2_a3)]
#flattened_pfc_list_b1_b2 = np.array(flattened_pfc_list_b1_b2)
#flattened_pfc_list_b1_b2 = flattened_pfc_list_b1_b2[~np.isnan(flattened_pfc_list_b1_b2)]
#
#bins_a = int(len(flattened_pfc_list_b1_b2)/100)
#hist(flattened_pfc_list_b1_b2, bins = 50)

def correlation_trials(experiment):
    a1_a2_all_neurons = []
    a2_a3_all_neurons = []
    a1_a3_all_neurons = []
    b1_b2_all_neurons = []
    b2_b3_all_neurons = []
    b1_b3_all_neurons = []
    for session in experiment:
        spikes_a = []
        spikes_b = []
        aligned_spikes= session.aligned_rates 
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape
        poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = ep.extract_choice_pokes(session)
        predictor_A_Task_1,  predictor_A_Task_2,  predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = predictors_f(session)
        spikes_B_task_1 =aligned_spikes[np.where(predictor_B_Task_1 ==1)]
        spikes_A_task_1 =aligned_spikes[np.where(predictor_A_Task_1 ==1)]
        spikes_B_task_2 =aligned_spikes[np.where(predictor_B_Task_2 ==1)]
        spikes_A_task_2 =aligned_spikes[np.where(predictor_A_Task_2 ==1)]
        spikes_B_task_3 =aligned_spikes[np.where(predictor_B_Task_3 ==1)]
        spikes_A_task_3 =aligned_spikes[np.where(predictor_A_Task_3 ==1)]
        
        a1 = np.arange(spikes_A_task_1.shape[0])
        a2 = np.arange(spikes_A_task_2.shape[0])
        a3 = np.arange(spikes_A_task_3.shape[0])
    
    
        spikes_a.append(spikes_A_task_1)
        spikes_a.append(spikes_A_task_2)
        spikes_a.append(spikes_A_task_3)
    
        combinations_a1_a2 = list(itertools.product(range(a1.shape[0]),range(a2.shape[0])))
        combinations_a2_a3 = list(itertools.product(range(a2.shape[0]),range(a3.shape[0])))
        combinations_a1_a3 = list(itertools.product(range(a1.shape[0]),range(a3.shape[0])))
        neurons_n = spikes_a[0].shape[1]
        a1_a2 = np.ones(shape=(neurons_n,len(combinations_a1_a2)))
        a1_a2[:] = np.NaN
        a2_a3 = np.ones(shape=(neurons_n,len(combinations_a2_a3)))
        a2_a3[:] = np.NaN
        a3_a1 = np.ones(shape=(neurons_n,len(combinations_a1_a3)))
        a3_a1[:]= np.NaN
        
        for i,combination in enumerate(combinations_a1_a2):
            for neuron in range(spikes_a[0].shape[1]):
                corr, p_value = pearsonr(spikes_a[0][combinations_a1_a2[i][0],neuron,:], spikes_a[1][combinations_a1_a2[i][1],neuron,:])
                a1_a2[neuron,i] = corr
                a1_a2_median = np.nanmedian(a1_a2, axis =1)
        for i,combination in enumerate(combinations_a2_a3):
            for neuron in range(spikes_a[0].shape[1]):
                corr, p_value = pearsonr(spikes_a[1][combinations_a2_a3[i][0],neuron,:], spikes_a[2][combinations_a2_a3[i][1],neuron,:])
                a2_a3[neuron,i] = corr
                a2_a3_median = np.nanmedian(a2_a3, axis =1)
        for i,combination in enumerate(combinations_a1_a3):
            for neuron in range(spikes_a[0].shape[1]):
                corr, p_value = pearsonr(spikes_a[0][combinations_a1_a3[i][0],neuron,:], spikes_a[2][combinations_a1_a3[i][1],neuron,:])
                a3_a1[neuron,i] = corr
                a1_a3_median = np.nanmedian(a3_a1, axis =1)
    
        b1 = np.arange(spikes_B_task_1.shape[0])
        b2 = np.arange(spikes_B_task_2.shape[0])
        b3 = np.arange(spikes_B_task_3.shape[0])
        spikes_b.append(spikes_B_task_1)
        spikes_b.append(spikes_B_task_2)
        spikes_b.append(spikes_B_task_3)
        
        combinations_b1_b2 = list(itertools.product(range(b1.shape[0]),range(b2.shape[0])))
        combinations_b2_b3 = list(itertools.product(range(b2.shape[0]),range(b3.shape[0])))
        combinations_b1_b3 = list(itertools.product(range(b1.shape[0]),range(b3.shape[0])))
        
        b1_b2 = np.ones(shape=(neurons_n,len(combinations_b1_b2)))
        b1_b2[:] = np.NaN
        b2_b3 = np.ones(shape=(neurons_n,len(combinations_b2_b3)))
        b1_b2[:] = np.NaN
        b3_b1 = np.ones(shape=(neurons_n,len(combinations_b1_b3)))
        b3_b1[:]= np.NaN
        
        for i,combination in enumerate(combinations_b1_b2):
            for neuron in range(spikes_b[0].shape[1]):
                corr, p_value = pearsonr(spikes_b[0][combinations_b1_b2[i][0],neuron,:], spikes_b[1][combinations_b1_b2[i][1],neuron,:])
                b1_b2[neuron,i] = corr
                b1_b2_median = np.nanmedian(b1_b2, axis = 1)
        for i,combination in enumerate(combinations_b2_b3):
            for neuron in range(spikes_b[0].shape[1]):
                corr, p_value = pearsonr(spikes_b[1][combinations_b2_b3[i][0],neuron,:], spikes_b[2][combinations_b2_b3[i][1],neuron,:])
                b2_b3[neuron,i] = corr
                b2_b3_median = np.nanmedian(b2_b3, axis = 1)
        for i,combination in enumerate(combinations_b1_b3):
            for neuron in range(spikes_b[0].shape[1]):
                corr, p_value = pearsonr(spikes_b[0][combinations_b1_b3[i][0],neuron,:], spikes_b[2][combinations_b1_b3[i][1],neuron,:])
                b3_b1[neuron,i] = corr
                b3_b1_median = np.nanmedian(b3_b1, axis = 1)
                
        a1_a2_all_neurons.append(a1_a2_median)
        a2_a3_all_neurons.append(a2_a3_median)
        a1_a3_all_neurons.append(a1_a3_median)
        b1_b2_all_neurons.append(b1_b2_median)
        b2_b3_all_neurons.append(b2_b3_median)
        b1_b3_all_neurons.append(b3_b1_median)
        
    return a1_a2_all_neurons, a2_a3_all_neurons, a1_a3_all_neurons, b1_b2_all_neurons, b2_b3_all_neurons, b1_b3_all_neurons
                
                  
            
            
def angle_similarity(experiment_aligned):
     predictors, C, X, y,cpd, C_choice_mean = regression(experiment_aligned)
     
     A1_A2 = angle(C_choice_mean[:,0], C_choice_mean[:,1])
     A2_A3 = angle(C_choice_mean[:,1], C_choice_mean[:,2])
     A3_A1 = angle(C_choice_mean[:,2], C_choice_mean[:,0])
     
     B1_B2 = angle(C_choice_mean[:,3], C_choice_mean[:,4])
     B2_B3 = angle(C_choice_mean[:,4], C_choice_mean[:,5])
     B3_B1 = angle(C_choice_mean[:,5], C_choice_mean[:,3])
     
     A1_B1 = angle(C_choice_mean[:,0], C_choice_mean[:,3])
     A1_B2 = angle(C_choice_mean[:,0], C_choice_mean[:,4])
     A1_B3 = angle(C_choice_mean[:,0], C_choice_mean[:,5])
     A2_B1 = angle(C_choice_mean[:,1], C_choice_mean[:,3])
     A2_B2 = angle(C_choice_mean[:,1], C_choice_mean[:,4])
     A2_B3 = angle(C_choice_mean[:,1], C_choice_mean[:,5])
     A3_B1 = angle(C_choice_mean[:,2], C_choice_mean[:,3])
     A3_B2 = angle(C_choice_mean[:,2], C_choice_mean[:,4])
     A3_B3 = angle(C_choice_mean[:,2], C_choice_mean[:,5])
     
     mean_a = np.mean([A1_A2,A2_A3,A3_A1])
     mean_b = np.mean([B1_B2,B2_B3,B3_B1])
     mean_a_b = np.mean([A1_B1,A1_B2,A1_B3,A2_B1,A2_B2,A2_B3,A3_B1,A3_B2,A3_B3 ])
     std_a = np.std([A1_A2,A2_A3,A3_A1])/np.sqrt(3)
     std_b = np.std([B1_B2,B2_B3,B3_B1])/np.sqrt(3)
     std_a_b = np.std([A1_B1,A1_B2,A1_B3,A2_B1,A2_B2,A2_B3,A3_B1,A3_B2,A3_B3 ])/np.sqrt(9)
     mean_a_b_ab = [mean_a, mean_b, mean_a_b]
     std_a_b_ab = [std_a,std_b,std_a_b]
     x_pos = [1,2,3]
     plt.errorbar(x = x_pos, y = mean_a_b_ab, yerr = std_a_b_ab, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    
     plt.xticks([1,2,3], ('A', 'B', 'AB'))
     plt.ylabel('cosine of the angle between two vectors')
     plt.title('PFC')
     
     
def _CPD(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression(copy_X = True,fit_intercept= False)
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
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = predictors_f(session)
        aligned_spikes= session.aligned_rates 
        n_neurons = aligned_spikes.shape[1]
        n_trials = aligned_spikes.shape[0]
        spikes_B_task_1 =aligned_spikes[np.where(predictor_B_Task_1 ==1)]
        spikes_A_task_1 =aligned_spikes[np.where(predictor_A_Task_1 ==1)]
        spikes_B_task_2 =aligned_spikes[np.where(predictor_B_Task_2 ==1)]
        spikes_A_task_2 =aligned_spikes[np.where(predictor_A_Task_2 ==1)]
        spikes_B_task_3 =aligned_spikes[np.where(predictor_B_Task_3 ==1)]
        spikes_A_task_3 =aligned_spikes[np.where(predictor_A_Task_3 ==1)]
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
    choices = session.trial_data['choices']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    outcomes_all = session.trial_data['outcomes'] 
    reward = outcomes_all[non_forced_array]
    
    
    choice_non_forced = choices[non_forced_array]
#    if n_trials != len(choice_non_forced):
#        n_trials = n_trials -1
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    poke_A = session.trial_data['poke_A']
    poke_B = session.trial_data['poke_B']
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = ep.extract_choice_pokes(session)
    
    #Task 1 
    choices_a = np.where(choice_non_forced == 1)
    choices_b = np.where(choice_non_forced == 0)
    
    predictor_a = np.zeros([1,n_trials])
    predictor_a[0][choices_a[0]] = 1
    predictor_b = np.zeros([1,n_trials])
    predictor_b[0][choices_b[0]] = 1
    if len(reward)!= len(predictor_a[0]):
        reward = np.append(reward,0)
        
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
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
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
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward
        
def regression(experiment):
    C = []    # To strore predictor loadings for each session.
    cpd = []  # To strore cpd for each session.
    for s,session in enumerate(experiment):
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
        forced_trials = session.trial_data['forced_trial']
        choices = session.trial_data['choices']
        non_forced_array = np.where(forced_trials == 0)[0]
        aligned_spikes= session.aligned_rates[:]
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
        predictor_A_Task_1,  predictor_A_Task_2,  predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = predictors_f(session)
        
        spikes_B_task_1 =aligned_spikes[np.where(predictor_B_Task_1 ==1)]
        spikes_A_task_1 =aligned_spikes[np.where(predictor_A_Task_1 ==1)]
        spikes_B_task_2 =aligned_spikes[np.where(predictor_B_Task_2 ==1)]
        spikes_A_task_2 =aligned_spikes[np.where(predictor_A_Task_2 ==1)]
        spikes_B_task_3 =aligned_spikes[np.where(predictor_B_Task_3 ==1)]
        spikes_A_task_3 =aligned_spikes[np.where(predictor_A_Task_3 ==1)]
        mean_spikes_B_task_1 = np.mean(spikes_B_task_1,axis = 0)
        mean_spikes_A_task_1 = np.mean(spikes_A_task_1,axis = 0)
        mean_spikes_B_task_2 = np.mean(spikes_B_task_2,axis = 0)
        mean_spikes_A_task_2 = np.mean(spikes_A_task_2,axis = 0)
        mean_spikes_B_task_3 = np.mean(spikes_B_task_3,axis = 0)
        mean_spikes_A_task_3 = np.mean(spikes_A_task_3,axis = 0)
        
        predictors = OrderedDict([
                                      ('a_task_1' , predictor_A_Task_1),
                                      ('a_task_2' , predictor_A_Task_2),
                                      ('a_task_3' , predictor_A_Task_3),
                                      ('b_task_1' , predictor_B_Task_1),
                                      ('b_task_2' , predictor_B_Task_2),
                                      ('b_task_3' , predictor_B_Task_3),
                                      ('reward', reward)])
            
        X = np.vstack(predictors.values()).T[:n_trials,:].astype(float)
        n_predictors = X.shape[1]
        y = aligned_spikes.reshape([n_trials,-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        ols = LinearRegression(copy_X = True,fit_intercept= False)
        ols.fit(X,y)
        C.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
    
    C = np.concatenate(C,0)
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0) # Population CPD is mean over neurons.
    ind_before_choice = ind_choice-7
    ind_after_choice = ind_choice+7
    C_choice = C[:,ind_before_choice:ind_after_choice,:]
    C_choice_mean  = np.mean(C_choice, axis =1) 
#    for i, predictor in enumerate(predictors):
#        if predictor == 'a_task_1':
#            plot(t_out,C_mean[:, i], label = '{}'.format(predictor), color = 'red')
#        elif predictor == 'a_task_2':
#            plot(t_out, C_mean[:, i], label = '{}'.format(predictor), color = 'pink')
#        elif predictor == 'a_task_3':
#            plot(t_out, C_mean[:, i], label = '{}'.format(predictor), color = 'purple')
#        elif predictor == 'b_task_1':
#            plot(t_out, C_mean[:, i], label = '{}'.format(predictor), color = 'black')
#        elif predictor == 'b_task_2':
#            plot(t_out, C_mean[:, i], label = '{}'.format(predictor), color = 'grey')
#        elif predictor == 'b_task_3':
#            plot(t_out, C_mean[:, i], label = '{}'.format(predictor), color = 'blue')
#        elif predictor == 'reward':
#            plot(t_out, C_mean[:, i], label = '{}'.format(predictor), color = 'yellow')

#    reward_time = initiate_choice_t[-2]+250
#    plt.axvline(reward_time, color='red', linestyle=':')
#    plt.legend()
#    
#    plt.figure()
#    for i, predictor in enumerate(predictors):
#        if predictor == 'a_task_1':
#            plt.plot(t_out, 100*cpd[:,i], label=predictor,  color = 'red')   
#        elif predictor == 'a_task_2':
#             plt.plot(t_out, 100*cpd[:,i], label=predictor,  color = 'pink')
#        elif predictor == 'a_task_3':
#             plt.plot(t_out, 100*cpd[:,i], label=predictor, color = 'purple')
#        elif predictor == 'b_task_1':
#             plt.plot(t_out, 100*cpd[:,i], label=predictor,  color = 'black')
#        elif predictor == 'b_task_2':
#             plt.plot(t_out, 100*cpd[:,i], label=predictor,  color = 'grey')
#        elif predictor == 'b_task_3':
#             plt.plot(t_out, 100*cpd[:,i], label=predictor, color = 'blue')
#        elif predictor == 'reward':
#             plt.plot(t_out, 100*cpd[:,i], label=predictor,  color = 'yellow')
#    for t in initiate_choice_t[1:-1]:
#        plt.axvline(t, color='k', linestyle=':')
        #figure, ax = plt.subplots(figsize = (15,5), ncols = n_neurons , nrows =2 )
#        for neuron in range(n_neurons):
#            for i,predictor in enumerate(predictors): 
#                if predictor == 'a_task_1':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'red')
#                elif predictor == 'a_task_2':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'pink')
#                elif predictor == 'a_task_3':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'purple')
#                elif predictor == 'b_task_1':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'black')
#                elif predictor == 'b_task_2':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'grey')
#                elif predictor == 'b_task_3':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'blue')
#                elif predictor == 'reward':
#                    ax[1][neuron].plot(C[s][neuron,:, i], label = '{}'.format(predictor), color = 'yellow')
#                    
#            ax[0][neuron].plot(mean_spikes_A_task_1[neuron], label = 'A Task 1', color = 'red')
#            ax[0][neuron].plot(mean_spikes_A_task_2[neuron], label = 'A Task 2', color = 'pink')
#            ax[0][neuron].plot(mean_spikes_A_task_3[neuron], label = 'A Task 3', color = 'purple')
#            ax[0][neuron].plot(mean_spikes_B_task_1[neuron], label = 'B Task 1', color = 'black')
#            ax[0][neuron].plot(mean_spikes_B_task_2[neuron], label = 'B Task 2', color = 'grey')
#            ax[0][neuron].plot(mean_spikes_B_task_3[neuron], label = 'B Task 3', color = 'blue')
#            
#            ax[1][0].legend(fontsize = 'xx-small')
#        plt.title('{}'.format(session.file_name))
    
    return predictors, C, X, y,cpd, C_choice_mean
    

def target_times_f(experiment):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
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
        
    return target_times

def all_sessions_aligment(experiment):
    target_times  = target_times_f(experiment)
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
        session.target_times = target_times
        experiment_aligned.append(session)
        
    return experiment_aligned 


def heatplot_aligned(experiment_aligned): 
    all_clusters_task_1 = []
    all_clusters_task_2 = []
    for session in experiment_aligned:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        reward = initiate_choice_t[-2] +250
        cluster_list_task_1 = []
        cluster_list_task_2 = []
        aligned_rates = session.aligned_rates
        poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = ep.extract_choice_pokes(session)
        trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3 = ep.initiation_and_trial_end_timestamps(session)
        task_1 = len(trial_сhoice_state_task_1)
        task_2 = len(trial_сhoice_state_task_2)
        if poke_I == poke_I_task_2: 
            aligned_rates_task_1 = aligned_rates[:task_1]
            aligned_rates_task_2 = aligned_rates[:task_1+task_2]
        elif poke_I == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
        elif poke_I_task_2 == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1+task_2]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
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
    #not_normed = same_shape_task_1[ordering,:]
    #not_normed += 1
    #not_normed = np.log(not_normed)
    norm_activity_sorted = (activity_sorted - np.min(activity_sorted,1)[:, None]) / (np.max(activity_sorted,1)[:, None] - np.min(activity_sorted,1)[:, None])
    where_are_Nans = np.isnan(norm_activity_sorted)
    norm_activity_sorted[where_are_Nans] = 0
    plt.imshow(norm_activity_sorted, aspect='auto')  
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward)).argmin()
    
    plt.xticks([ind_init, ind_choice, ind_reward], ('I', 'C', 'R'))
    plt.title('PFC')
    plt.colorbar()
    



