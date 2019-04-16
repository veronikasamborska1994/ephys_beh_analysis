#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:01:01 2019

@author: veronikasamborska
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:31:41 2019

@author: veronikasamborska
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import ephys_beh_import as ep
import matplotlib.pyplot as plt
import regressions as re
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict
from numpy.linalg import matrix_rank
from matplotlib import colors as mcolors



def extract_poke_times_include_a(session):       
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    configuration = session.trial_data['configuration_i']
    
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
     
    task = session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)

    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3_change[0]])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3_change[0]])
    outcomes = session.trial_data['outcomes']
    outcomes_non_forced = outcomes[non_forced_array]
    
    # Pokes A, B and I 
    a_pokes = np.unique(session.trial_data['poke_A'])
    #print('These are A pokes')
    #print(poke_A, poke_A_task_2, poke_A_task_3)
    b_pokes = np.unique(session.trial_data['poke_B'])
    #print('These are B pokes')
    #print(poke_B, poke_B_task_2, poke_B_task_3)
    i_pokes = np.unique(configuration)
    #print('These are I pokes')
    configuration = session.trial_data['configuration_i']
    i_poke_task_1 = configuration[0]
    i_poke_task_2 = configuration[task_2_change[0]]
    i_poke_task_3 = configuration[task_3_change[0]]
    #print(i_poke_task_1, i_poke_task_2, i_poke_task_3)
    all_pokes = np.concatenate([a_pokes, b_pokes, i_pokes])
    all_pokes = np.unique(all_pokes)
    
    poke_I = 'poke_'+str(i_poke_task_1)
    poke_I_task_2 = 'poke_'+str(i_poke_task_2)
    poke_I_task_3 = 'poke_'+str(i_poke_task_3)

    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)
    
    if poke_A1_A2_A3 == True:
        constant_poke_a = poke_A
    if poke_A1_B2_B3 == True:
        constant_poke_a = poke_A
    if poke_A1_B2_A3 == True:
        constant_poke_a = poke_A
    if poke_A1_A2_B3 == True:
        constant_poke_a = poke_A
    if poke_B1_B2_B3 == True:
        constant_poke_a = poke_B
    if poke_B1_A2_A3 == True:
        constant_poke_a = poke_B
    if poke_B1_A2_B3 == True:
        constant_poke_a = poke_B
    if poke_B1_B2_A3 == True:
        constant_poke_a = poke_B
        
    #Events for Pokes Irrespective of Meaning
    pokes = {}
    for i, poke in enumerate(all_pokes):
        pokes[poke] = [event.time for event in session.events if event.name in ['poke_'+str(all_pokes[i])]]
    
    events_and_times = [[event.name, event.time] for event in session.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]

    poke_list_B = []
    poke_list_A = []

    initation_choice = []
    choice_state = False 
    choice_state_count = 0
    poke_identity = []
    initiation_time_stamps = []
    
    for i,event in enumerate(events_and_times):
        if 'choice_state' in event:        
            choice_state_count +=1 
            choice_state = True  
            if choice_state_count <= len(task_1):
                poke_identity.append(poke_I)
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
                poke_identity.append(poke_I_task_2)
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                poke_identity.append(poke_I_task_3)
            initiation_time_stamps.append(event[1])
            initation_choice.append(0)
            
        if choice_state_count <= len(task_1):
            if poke_A in event: 
                if choice_state == True:
                    choice_state = False
                    poke_list_A.append(event[1])
                    poke_identity.append(poke_A)
                    initation_choice.append(0.5)
            if poke_B in event:
                if choice_state == True:
                    poke_list_B.append(event[1])
                    poke_identity.append(poke_B)
                    choice_state = False
                    initation_choice.append(1)
                    
        elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
            if poke_B_task_2 in event:
               if choice_state == True:
                   choice_state = False
                   poke_list_B.append(event[1])
                   poke_identity.append(poke_B_task_2)
                   initation_choice.append(1)
                   
            elif poke_A_task_2 in event:
                if choice_state == True:
                    poke_list_A.append(event[1])
                    poke_identity.append(poke_A_task_2)
                    initation_choice.append(1)
                    choice_state = False
                        
        elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
            
            if poke_B_task_3 in event:
                if choice_state == True:
                    poke_list_B.append(event[1])
                    poke_identity.append(poke_B_task_3)
                    initation_choice.append(1)
                    choice_state = False
                            
            if poke_A_task_3 in event:
                if choice_state == True:
                    poke_list_A.append(event[1])
                    poke_identity.append(poke_A_task_3)
                    initation_choice.append(1)
                    choice_state = False      
                    
    init_times = session.times['choice_state']
    inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['choice_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
    choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'choice_state'])
    if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
    trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T                
   
    all_events = np.concatenate([initiation_time_stamps,poke_list_A,poke_list_B])       
    all_events  =  sorted(all_events)
    choices =  np.concatenate([poke_list_A,poke_list_B])     
    choices =  sorted(choices)
    return poke_identity,outcomes_non_forced,initation_choice,initiation_time_stamps,poke_list_A, poke_list_B,all_events,constant_poke_a,choices,trial_times


def predictors_around_pokes_include_a(session):
    poke_identity,outcomes_non_forced,initation_choice,initiation_time_stamps,poke_list_A, poke_list_B,all_events,constant_poke_a,choices,trial_times = extract_poke_times_include_a(session)
    unique_pokes = np.unique(poke_identity)
    
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
    
    poke_A = predictor_A_Task_1+predictor_A_Task_2+predictor_A_Task_3
    
    i = np.where(unique_pokes == constant_poke_a)
    unique_pokes = np.delete(unique_pokes,i)

    poke_1_id = constant_poke_a
    poke_2_id = unique_pokes[0]
    poke_3_id = unique_pokes[1]
    poke_4_id = unique_pokes[2]
    poke_5_id = unique_pokes[3]
        
    poke_1 = np.zeros(len(poke_identity))
    poke_2 = np.zeros(len(poke_identity))
    poke_3 = np.zeros(len(poke_identity))
    poke_4 = np.zeros(len(poke_identity))
    poke_5 = np.zeros(len(poke_identity))
  
    # Make a predictor for outcome which codes Initiation as 0
    outcomes =[]
    for o,outcome in enumerate(outcomes_non_forced):
        outcomes.append(0)
        if outcome == 1:
            outcomes.append(1)
        elif outcome == 0:
            outcomes.append(0)
    outcomes = np.asarray(outcomes)  
    
    choices = []
    for c,choice in enumerate(poke_A):
        choices.append(0)
        if choice == 1:
            #choices.append(0)
            choices.append(0.5)
        elif choice == 0:
            #choices.append(0)
            choices.append(-0.5)
    
    choices = np.asarray(choices) 

    init_choices_a_b = []
    for c,choice in enumerate(poke_A):
        if choice == 1:
            init_choices_a_b.append(0)
            init_choices_a_b.append(0)
        elif choice == 0:
            init_choices_a_b.append(1)
            init_choices_a_b.append(1)
    init_choices_a_b   = np.asarray(init_choices_a_b)  

    choices_initiation = []
    for c,choice in enumerate(poke_A):
        choices_initiation.append(1)
        if choice == 1:
            choices_initiation.append(0)
        elif choice == 0:
            choices_initiation.append(0)

    choices_initiation = np.asarray(choices_initiation)
    
    for p,poke in enumerate(poke_identity):
        if poke == poke_1_id:
            poke_1[p] = 1
        if poke == poke_2_id:
            poke_2[p] = 1
        elif poke == poke_3_id:
            poke_3[p] = 1
        elif poke == poke_4_id:
            poke_4[p] = 1
        elif poke == poke_5_id:
            poke_5[p] = 1
            

#    for p,poke in enumerate(poke_identity):
#            if poke == poke_1_id:
#                poke_1[p] = 1
#            if poke == poke_2_id:
#                poke_2[p] = 1
#            elif poke == poke_3_id:
#                poke_3[p] = 1
#            elif poke == poke_4_id:
#                poke_4[p] = 1
#            elif poke == poke_5_id:
#                poke_1[p] = -1
#                poke_2[p] = -1
#                poke_3[p] = -1
#                poke_4[p] = -1
#                poke_5[p] = -1
    

    return poke_1,poke_2,poke_3,poke_4,poke_5,outcomes,initation_choice,unique_pokes,constant_poke_a,choices,choices_initiation,init_choices_a_b

def histogram_include_a(session):
    
    poke_identity,outcomes_non_forced,initation_choice,initiation_time_stamps,poke_list_A, poke_list_B,all_events,constant_poke_a,choices, trial_times  = extract_poke_times_include_a(session)
    
    neurons = np.unique(session.ephys[0])
    spikes = session.ephys[1]
    window_to_plot = 4000
    #all_neurons_all_spikes_raster_plot_task = []
    smooth_sd_ms = 1
    bin_width_ms = 50
   
    bin_edges_trial = np.arange(-4050,window_to_plot, bin_width_ms)
    trial_length = 60
    aligned_rates = np.zeros([len(all_events), len(neurons), trial_length]) # Array to store trial aligned firing rates. 

    for i,neuron in enumerate(neurons):  
        spikes_ind = np.where(session.ephys[0] == neuron)
        spikes_n = spikes[spikes_ind]

        for e,event in enumerate(all_events):
            period_min = event - window_to_plot
            period_max = event + window_to_plot
            spikes_ind = spikes_n[(spikes_n >= period_min) & (spikes_n<= period_max)]

            spikes_to_save = (spikes_ind - event)   
            hist_task,edges_task = np.histogram(spikes_to_save, bins= bin_edges_trial)
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms)
            normalised_task = normalised_task*20
            if len(normalised_task) > 0:
                normalised_task = normalised_task[50:110]
                
            aligned_rates[e,i,:]  = normalised_task

    return aligned_rates


        
def raster_plot_save(experiment):
    all_sessions = []
    for s,session in enumerate(experiment):
        
        aligned_rates = histogram_include_a(session)
        aligned_rates = np.asarray(aligned_rates)
        all_sessions.append(aligned_rates)
        
    return all_sessions


        
def _CPD(X,y):
    
    '''Evaluate coefficient of partial determination for each predictor in X'''   
    ols = LinearRegression(fit_intercept = False)
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd


def task_specific_regressors(session):
    task = session.trial_data['task']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]   
    task_non_forced = task[non_forced_array]
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0]        

    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
    
    predictor_B_Task_1[len(task_1)+len(task_2):] = -1 
    predictor_B_Task_1[len(task_1):len(task_1)+len(task_2)] = -2

    
    predictor_B_Task_2[len(task_1)+len(task_2):] = -1
    predictor_B_Task_2[:len(task_1)] = -2
    
    predictor_B_Task_1 = predictor_B_Task_1+predictor_B_Task_3
    predictor_B_Task_2 = predictor_B_Task_2+predictor_B_Task_3

    predictor_B_Task_1_choice = []
    predictor_B_Task_2_choice = []
    
    for c,choice in enumerate(predictor_B_Task_1):
        if choice == 1:
            predictor_B_Task_1_choice.append(0)
            predictor_B_Task_1_choice.append(1)
        elif choice == 0:
            predictor_B_Task_1_choice.append(0)
            predictor_B_Task_1_choice.append(0)
        elif choice == -2:
            predictor_B_Task_1_choice.append(0)
            predictor_B_Task_1_choice.append(0)
        elif choice == -1:
            predictor_B_Task_1_choice.append(0)
            predictor_B_Task_1_choice.append(-1)
            
    for c,choice in enumerate(predictor_B_Task_2):
        if choice == 1:
            predictor_B_Task_2_choice.append(0)
            predictor_B_Task_2_choice.append(1)
        elif choice == 0:
            predictor_B_Task_2_choice.append(0)
            predictor_B_Task_2_choice.append(0)
        elif choice == -2:
            predictor_B_Task_2_choice.append(0)
            predictor_B_Task_2_choice.append(0)
        elif choice == -1:
            predictor_B_Task_2_choice.append(0)
            predictor_B_Task_2_choice.append(-1)
            
    predictor_B_Task_1_initiation = []
    predictor_B_Task_2_initiation = []
    
    for c,choice in enumerate(predictor_B_Task_1):
        if choice == 1:
            predictor_B_Task_1_initiation.append(1)
            predictor_B_Task_1_initiation.append(0)
        elif choice == -2:
            predictor_B_Task_1_initiation.append(0)
            predictor_B_Task_1_initiation.append(0)
        elif choice == 0:
            predictor_B_Task_1_initiation.append(1)
            predictor_B_Task_1_initiation.append(0)
        elif choice == -1:
            predictor_B_Task_1_initiation.append(-1)
            predictor_B_Task_1_initiation.append(0)
    
    for c,choice in enumerate(predictor_B_Task_2):
        if choice == 1:
            predictor_B_Task_2_initiation.append(1)
            predictor_B_Task_2_initiation.append(0)
        elif choice == -2:
            predictor_B_Task_2_initiation.append(0)
            predictor_B_Task_2_initiation.append(0)
        elif choice == 0:
            predictor_B_Task_2_initiation.append(1)
            predictor_B_Task_2_initiation.append(0)
        elif choice == -1:
            predictor_B_Task_2_initiation.append(-1)
            predictor_B_Task_2_initiation.append(0)
            
    t_3 = predictor_B_Task_1_initiation[(len(task_1)+len(task_2))*2:]
    t_3 = np.asarray(t_3)
    
    ind = np.where(t_3 == 1)
    ind = np.asarray(ind)+(len(task_1)+len(task_2))*2
    
    predictor_B_Task_1_initiation = np.asarray(predictor_B_Task_1_initiation)
    predictor_B_Task_2_initiation = np.asarray(predictor_B_Task_2_initiation)
    predictor_B_Task_1_initiation[ind[0]] = 0 
    predictor_B_Task_2_initiation[ind[0]] = 0 

    
    return predictor_B_Task_1_initiation,predictor_B_Task_2_initiation,predictor_B_Task_1_choice,predictor_B_Task_2_choice

def regression_choices_c(experiment, all_sessions):
            
    C_task_1 = []     # To strore predictor loadings for each session in task 1.
    C_task_2 = []    # To strore predictor loadings for each session in task 2.
    C_task_3 = []    # To strore predictor loadings for each session in task 2.
     
    # Finding correlation coefficients for task 1  
    for s,session in enumerate(experiment):
        all_neurons_all_spikes_raster_plot_task = all_sessions[s]
    
        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            #Select  Choices only
            all_neurons_all_spikes_raster_plot_task  = all_neurons_all_spikes_raster_plot_task[1::2,:,:]
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]        
    
            n_trials,n_neurons, n_timepoints = all_neurons_all_spikes_raster_plot_task.shape 

            # For regressions for each task independently 
            predictor_A_Task_1 = predictor_A_Task_1[:len(task_1)]
            all_neurons_all_spikes_raster_plot_task_1 = all_neurons_all_spikes_raster_plot_task[:len(task_1),:,:]
            all_neurons_all_spikes_raster_plot_task_1 = np.mean(all_neurons_all_spikes_raster_plot_task_1, axis = 2)

            predictors_task_1 = OrderedDict([
                                              ('A_task_1' , predictor_A_Task_1)])
            
            X_task_1 = np.vstack(predictors_task_1.values()).T[:len(predictor_A_Task_1),:].astype(float)
            n_predictors = X_task_1.shape[1]
            y_t1 = all_neurons_all_spikes_raster_plot_task_1.reshape([all_neurons_all_spikes_raster_plot_task_1.shape[0],-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X_task_1,y_t1)
            C_task_1.append(ols.coef_.reshape(n_neurons, n_predictors)) # Predictor loadings
            
            # For regressions for each task independently 
            predictor_A_Task_2 = predictor_A_Task_2[len(task_1):len(task_1)+len(task_2)]
            all_neurons_all_spikes_raster_plot_task_2 = all_neurons_all_spikes_raster_plot_task[len(task_1):len(task_1)+len(task_2),:,:]
            all_neurons_all_spikes_raster_plot_task_2 = np.mean(all_neurons_all_spikes_raster_plot_task_2, axis = 2)

            predictors_task_2 = OrderedDict([
                                               ('A_task_2' , predictor_A_Task_2)])
            
            X_task_2 = np.vstack(predictors_task_2.values()).T[:len(predictor_A_Task_2),:].astype(float)
            n_predictors = X_task_2.shape[1]
            y_t2 = all_neurons_all_spikes_raster_plot_task_2.reshape([all_neurons_all_spikes_raster_plot_task_2.shape[0],-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X_task_2,y_t2)
            C_task_2.append(ols.coef_.reshape(n_neurons, n_predictors)) # Predictor loadings    
    
                
            # For regressions for each task independently 
            predictor_A_Task_3 = predictor_A_Task_3[len(task_1)+len(task_2):]
            all_neurons_all_spikes_raster_plot_task_3 =  all_neurons_all_spikes_raster_plot_task[len(task_1)+len(task_2):,:,:]
            all_neurons_all_spikes_raster_plot_task_3 = np.mean(all_neurons_all_spikes_raster_plot_task_3, axis = 2)
            
            predictors_task_3 = OrderedDict([
                                               ('A_task_3' , predictor_A_Task_3)])
        
            X_task_3 = np.vstack(predictors_task_3.values()).T[:len(predictor_A_Task_3),:].astype(float)
            n_predictors = X_task_3.shape[1]
            y_t3 = all_neurons_all_spikes_raster_plot_task_3.reshape([all_neurons_all_spikes_raster_plot_task_3.shape[0],-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X_task_3,y_t3)
            C_task_3.append(ols.coef_.reshape(n_neurons, n_predictors)) # Predictor loadings    

    C_task_1 = np.concatenate(C_task_1,0)
    C_task_2 = np.concatenate(C_task_2,0)              
    C_task_3 = np.concatenate(C_task_3,0)
    
    return C_task_1, C_task_2,C_task_3


def sorting_by_task_1(experiment,all_sessions):
    C_task_1, C_task_2, C_task_3 = regression_choices_c(experiment, all_sessions)

    task_1 = C_task_1[:,0].flatten()
    task_2 = C_task_2[:,0].flatten()
    task_3 = C_task_3[:,0].flatten()

    argmax_neuron = np.argsort(-task_1)
    task_2_by_1 = task_2[argmax_neuron]
    task_1 = task_1[argmax_neuron]
    task_3_by_1 = task_3[argmax_neuron]
    
    
    plt.figure(3)
    
    y = np.arange(len(task_1))
    
    plt.scatter(y,task_2_by_1,s = 0.5, color = 'red', label = 'Task 2 sorted by Task 1')

    plt.scatter(y,task_3_by_1,s = 0.5,color = 'slateblue', label = 'Task 3 sorted by Task 1')
    
    plt.scatter(y,task_1,s = 0.5,color = 'black', label = 'Task 1 sorted')
    plt.legend()
    


def plotting_coeff(C_task_1,C_task_2,C_task_3):
    task_1 = C_task_1[:,0].flatten()
    task_2 = C_task_2[:,0].flatten()
    task_3 = C_task_3[:,0].flatten()

    argmax_neuron = np.argsort(-task_1)
    task_2_by_1 = task_2[argmax_neuron]
    task_1 = task_1[argmax_neuron]
    task_3_by_1 = task_3[argmax_neuron]
    
    y = np.arange(len(task_1))
    plt.scatter(y,task_2_by_1,s = 2, color = 'red', label = 'Task 2 sorted by Task 1')

    plt.scatter(y,task_3_by_1,s = 2,color = 'slateblue', label = 'Task 3 sorted by Task 1')
    
    plt.scatter(y,task_1,s = 2,color = 'black', label = 'Task 1 sorted')

    #plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
    
    plt.legend()
    plt.title('PFC')


  
def average_firing_rates(experiment,all_sessions):
    all_sessions_Ar = []
    all_sessions_Br = []
    all_sessions_Anr = []
    all_sessions_Bnr = []
    for s,session in enumerate(experiment):
        all_neurons_all_spikes_raster_plot_task = all_sessions[s]

        if  all_neurons_all_spikes_raster_plot_task.shape[1]> 0: 
            all_neurons_all_spikes_raster_plot_task = np.asarray(all_neurons_all_spikes_raster_plot_task)
            all_neurons_all_spikes_raster_plot_task  = all_neurons_all_spikes_raster_plot_task[:,1::2,:]


            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
            reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
            same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1,\
            different_outcome_task_2, different_outcome_task_3 = re.predictors_include_previous_trial(session)     
            n_neurons, n_trials, n_timepoints = all_neurons_all_spikes_raster_plot_task.shape 

            all_neurons_all_spikes_raster_plot_task = all_neurons_all_spikes_raster_plot_task[:,:len(predictor_A_Task_1),:]
            predictor_A = predictor_A_Task_1+predictor_A_Task_2 + predictor_A_Task_3
            predictor_A = predictor_A[:all_neurons_all_spikes_raster_plot_task.shape[1]]
            reward = reward[:all_neurons_all_spikes_raster_plot_task.shape[1]]
        
            #Indexing A rewarded, B rewarded firing rates in 3 tasks
            aligned_rates_A_reward = all_neurons_all_spikes_raster_plot_task[:,np.where((predictor_A ==1) & (reward == 1 )),:]
            aligned_rates_A_Nreward = all_neurons_all_spikes_raster_plot_task[:,np.where((predictor_A ==1) & (reward == 0 )),:]
            aligned_rates_B_reward = all_neurons_all_spikes_raster_plot_task[:,np.where((predictor_A ==0) & (reward == 1 )),:]
            aligned_rates_B_Nreward = all_neurons_all_spikes_raster_plot_task[:,np.where((predictor_A ==0) & (reward == 0 )),:]
            
            mean_aligned_rates_A_reward = np.mean(aligned_rates_A_reward,axis = 2)
            mean_aligned_rates_A_Nreward = np.mean(aligned_rates_A_Nreward,axis = 2)
            mean_aligned_rates_B_reward = np.mean(aligned_rates_B_reward,axis = 2)
            mean_aligned_rates_B_Nreward = np.mean(aligned_rates_B_Nreward,axis = 2)
            
            mean_aligned_rates_A_reward = np.mean(mean_aligned_rates_A_reward,axis = 0)
            mean_aligned_rates_A_Nreward = np.mean(mean_aligned_rates_A_Nreward,axis = 0)
            mean_aligned_rates_B_reward = np.mean(mean_aligned_rates_B_reward,axis = 0)
            mean_aligned_rates_B_Nreward = np.mean(mean_aligned_rates_B_Nreward,axis = 0)

            all_sessions_Ar.append(mean_aligned_rates_A_reward[0,:])
            all_sessions_Br.append(mean_aligned_rates_B_reward[0,:])
            
            all_sessions_Anr.append(mean_aligned_rates_A_Nreward[0,:])
            all_sessions_Bnr.append(mean_aligned_rates_B_Nreward[0,:])
         
    all_sessions_Ar = np.asarray(all_sessions_Ar)
    all_sessions_Br = np.asarray(all_sessions_Br)
    all_sessions_Anr = np.asarray(all_sessions_Anr)
    all_sessions_Bnr = np.asarray(all_sessions_Bnr)
    
    average_Ar = np.mean(all_sessions_Ar, axis = 0 )
    average_Br = np.mean(all_sessions_Br, axis = 0 )

    average_Anr = np.mean(all_sessions_Anr, axis = 0 )
    average_Bnr = np.mean(all_sessions_Bnr, axis = 0 )
    
    plt.plot(average_Ar, label = 'A reward')
    plt.plot(average_Br, label = 'B reward')
    plt.plot(average_Anr, label = 'A No reward')
    plt.plot(average_Bnr, label = 'B No reward')
    plt.legend()

    
def regression_pokes_aligned_warped_simple(experiment,all_sessions):   
    
    C_warped = []
    cpd_warped =[]
    # Finding correlation coefficients for task 1  
    for s,session in enumerate(experiment):            
        all_neurons_all_spikes_raster_plot_task = all_sessions[s]      
        if all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 

            all_neurons_all_spikes_raster_plot_task  = all_neurons_all_spikes_raster_plot_task[1::2,:,:]
            
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
            reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
            same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1,\
            different_outcome_task_2, different_outcome_task_3 = re.predictors_include_previous_trial(session)     
            
            n_trials,n_neurons, n_timepoints = all_neurons_all_spikes_raster_plot_task.shape 
            
           
            predictor_A = predictor_A_Task_1+predictor_A_Task_2 + predictor_A_Task_3
            predictor_A = predictor_A[:all_neurons_all_spikes_raster_plot_task.shape[0]]
            reward = reward[:all_neurons_all_spikes_raster_plot_task.shape[0]]
#            interaction = reward*predictor_A
            ones = np.ones(len(predictor_A))

            predictors= OrderedDict([
                                               ('A' , predictor_A),
                                               ('reward',reward),
                                               ('ones',ones)])
                                               #('interaction',interaction)])
                                          
        
            X= np.vstack(predictors.values()).T[:len(predictor_A),:].astype(float)
            ones = np.ones(X.shape[0]).reshape(X.shape[0],1)
            
            X_check_rank = np.hstack([X,ones])
            X_check_rank = X
            print(X_check_rank.shape[1])
            rank = matrix_rank(X_check_rank)
            print(rank)
            n_predictors = X.shape[1]

            y = all_neurons_all_spikes_raster_plot_task.reshape([all_neurons_all_spikes_raster_plot_task.shape[0],-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X,y)
            
            C_warped.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
            cpd_warped.append(_CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
       
            
    C_warped = np.concatenate(C_warped,0)
    cpd_warped = np.nanmean(np.concatenate(cpd_warped,0), axis = 0) # Population CPD is mean over neurons.
     
        
        
def plotting(experiment,all_sessions):
    X,cpd, predictors = regression_pokes_aligned_warped_trials(experiment, all_sessions)
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    c = [*colors]
    c =  ['violet', 'black', 'red','chocolate', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink']
    p = [*predictors]
    plt.figure(2)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        #plt.title('PFC')
   # plt.vlines(32,ymin = 0, ymax = 0.15,linestyles= '--', color = 'grey', label = 'Poke')
    plt.legend()
    plt.ylabel('Coefficient of Partial Determination')
    plt.xlabel('Time (ms)')
    plt.xticks([0,10,20,30,40,50,60],[-1500,-1000,-500,0,500,1000,-1500])
    
 
def regression_pokes_aligned_warped_trials(experiment, all_sessions):   
    
    C = []
    cpd =[]
    
    # Finding correlation coefficients for task 1  
    for s,session in enumerate(experiment_aligned_HP):
        session =experiment_aligned_HP[1]
        all_neurons_all_spikes_raster_plot_task = all_sessions_HP[1]
        all_neurons_all_spikes_raster_plot_task = np.asarray(all_neurons_all_spikes_raster_plot_task)
        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            poke_1,poke_2,poke_3,poke_4,poke_5,outcomes,initation_choice,unique_pokes,constant_poke_a,choices,choices_initiation,init_choices_a_b = predictors_around_pokes_include_a(session)
            
            predictor_B_Task_1_initiation,predictor_B_Task_2_initiation,predictor_B_Task_1_choice,predictor_B_Task_2_choice   = task_specific_regressors(session)
          
            n_trials,n_neurons, n_timepoints = all_neurons_all_spikes_raster_plot_task.shape 
            initation_choice =initation_choice[:len(poke_1)]
            initation_choice = np.asarray(initation_choice)

            choices = choices[:len(poke_1)]
            choices_initiation = choices_initiation[:len(poke_1)]
            outcomes = outcomes[:len(poke_1)]
            outcomes_choices_interaction = outcomes*choices
            outcomes_choices_init_interaction = outcomes*init_choices_a_b
            all_neurons_all_spikes_raster_plot_task = all_neurons_all_spikes_raster_plot_task[:len(poke_1),:,:]
            
            #predictor_B_Task_1_initiation = predictor_B_Task_1_initiation - np.mean(predictor_B_Task_1_initiation)
            predictor_B_Task_2_initiation = predictor_B_Task_2_initiation - np.mean(predictor_B_Task_2_initiation)
            
            predictor_B_Task_1_choice = predictor_B_Task_1_choice - np.mean(predictor_B_Task_1_choice)
            predictor_B_Task_2_choice = predictor_B_Task_2_choice - np.mean(predictor_B_Task_2_choice)

            outcome_choice_task_1 = predictor_B_Task_1_choice*outcomes
            outcome_choice_task_2 = predictor_B_Task_2_choice*outcomes
            ones = np.ones(len(choices))

            predictors = OrderedDict([
                                          ('poke_1', poke_1),
                                          ('poke_2', poke_2), 
                                          ('poke_3', poke_3),
                                          ('poke_4', poke_4),
                                          ('poke_5', poke_5),
                                          ('choice_vs_initiation', choices_initiation),
                                          #('choice_a_b_at_choice',choices),
                                          ('choice_a_b_at_initiation',init_choices_a_b),
                                          ('outcomes', outcomes),
                                          ('ones', ones),
                                          #('outcomes_interaction_at_initiation',outcomes_choices_init_interaction),
                                          ('outcomes_interaction_at_choice',outcomes_choices_interaction)])
                                          #('Task_1_initiation_choice', predictor_B_Task_1_initiation),
                                          #('Task_2_initiation_choice',predictor_B_Task_2_initiation),
                                          #('Task_1_choice', predictor_B_Task_1_choice),
                                          #('Task_2_choice',predictor_B_Task_2_choice),
                                          #('outcome_choice_task_1',outcome_choice_task_1),
                                          #('outcome_choice_task_2',outcome_choice_task_2)])
        
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            
            #X_check_rank = np.hstack([X,ones])
            
    
            print(X.shape[1])  
            rank = matrix_rank(X) 
            print(rank)
            n_predictors = X.shape[1]
            
            y = all_neurons_all_spikes_raster_plot_task.reshape([all_neurons_all_spikes_raster_plot_task.shape[0],-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X,y)
  
            C.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
       
    C = np.concatenate(C,0)
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0) # Population CPD is mean over neurons.
   
    return X,cpd, predictors

scipy.io.savemat('/Users/veronikasamborska/Desktop/data.mat', mdict={'data': all_neurons_all_spikes_raster_plot_task})

scipy.io.savemat('/Users/veronikasamborska/Desktop/design_m.mat', mdict={'Design_matrix': X})
