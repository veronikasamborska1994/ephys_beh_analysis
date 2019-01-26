#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:01:30 2018

@author: veronikasamborska
"""

import data_import as di 
import utility as ut
import plotting as pl
import heatmap_aligned as ha

import copy 
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import itertools
from scipy.stats import pearsonr
import ephys_beh_import as ep
import pylab as plt
import math 

# Function for finding the dot product of two vectors 
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

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
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
        predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
        predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3  = predictors_pokes(session)
        
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


def angle_between_vectors(experiment):
    C_task_1, C_task_2 = regression(experiment)
    angle_between = angle(C_task_1[:,0], C_task_2[:,0])
    return angle_between

def predictors_pokes(session):
    pyControl_choice = [event.time for event in session.events if event.name in ['choice_state']]

    n_trials = len(pyControl_choice)
    choices = session.trial_data['choices']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]   
    outcomes_all = session.trial_data['outcomes'] 
    reward = outcomes_all[non_forced_array]
    choice_non_forced = choices[non_forced_array]
    
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
    
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = ep.state_indices(session)

    if poke_A1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)

    elif poke_A1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_A1_B2_A3 == True: 
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)

    elif poke_A1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_a_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_B1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_B1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)
        
    elif poke_B1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_a_good)
        predictor_a_good_task_3 = copy.copy(state_t3_b_good)

    elif poke_B1_B2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(state_b_good)
        predictor_a_good_task_2 = copy.copy(state_t2_b_good)
        predictor_a_good_task_3 = copy.copy(state_t3_a_good)
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3
        
def regression(experiment):
    C_task_1= []     # To strore predictor loadings for each session in task 1.
    C_task_2 = []    # To strore predictor loadings for each session in task 2.
    
    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            t_out = session.t_out
            initiate_choice_t = session.target_times #Initiation and Choice Times
            ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
            ind_after_choice = ind_choice + 7 # 1 sec after choice
            spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
            mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =2)
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = predictors_pokes(session)     
            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
    
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]        
            firing_rate_task_1 =  mean_spikes_around_choice[:len(task_1)]
            firing_rate_task_2 =  mean_spikes_around_choice[len(task_1):len(task_1)+len(task_2)]
            
            # For regressions for each task independently 
            predictor_A_Task_1 = predictor_A_Task_1[:len(task_1)]
            reward_t1 = reward[:len(task_1)]
            reward_t2 = reward[len(task_1):len(task_1)+len(task_2)]
    
            
            predictors_task_1 = OrderedDict([
                                          ('A_task_1' , predictor_A_Task_1),
                                          ('reward_t1', reward_t1)])
        
           
            X_task_1 = np.vstack(predictors_task_1.values()).T[:len(predictor_A_Task_1),:].astype(float)
            n_predictors = X_task_1.shape[1]
            y_t1 = firing_rate_task_1.reshape([len(firing_rate_task_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X_task_1,y_t1)
            C_task_1.append(ols.coef_.reshape(n_neurons, n_predictors)) # Predictor loadings
    
    C_task_1 = np.concatenate(C_task_1,0)
    # Finding correlation coefficients from task two 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0:  # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            t_out = session.t_out
            initiate_choice_t = session.target_times #Initiation and Choice Times
            ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
            ind_after_choice = ind_choice + 7 # 1 sec after choice
            spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
            mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =2)
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = predictors_pokes(session)     
            
            # Getting out task indicies
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']

            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]        
            firing_rate_task_2 =  mean_spikes_around_choice[len(task_1):len(task_1)+len(task_2)]
            
            # For regressions for each task independently 
            predictor_A_Task_2 = predictor_A_Task_2[len(task_1):len(task_1)+len(task_2)]
            reward_t2 = reward[len(task_1):len(task_1)+len(task_2)]            
            predictors_task_2 = OrderedDict([
                                          ('A_task_2' , predictor_A_Task_2),
                                          ('reward_t2', reward_t2)])
        
           
            X_task_2 = np.vstack(predictors_task_2.values()).T[:len(predictor_A_Task_2),:].astype(float)
            n_predictors = X_task_2.shape[1]
            y_t2 = firing_rate_task_2.reshape([len(firing_rate_task_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X_task_2,y_t2)
            C_task_2.append(ols.coef_.reshape(n_neurons, n_predictors)) # Predictor loadings
                
    C_task_2 = np.concatenate(C_task_2,0)
    return C_task_1, C_task_2
