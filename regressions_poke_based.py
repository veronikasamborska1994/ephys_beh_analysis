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


def extract_poke_times(session):       
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    configuration = session.trial_data['configuration_i']
    
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
     
    task = session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
  
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
    print('These are A pokes')
    print(poke_A, poke_A_task_2, poke_A_task_3)
    b_pokes = np.unique(session.trial_data['poke_B'])
    print('These are B pokes')
    print(poke_B, poke_B_task_2, poke_B_task_3)
    i_pokes = np.unique(configuration)
    print('These are I pokes')
    configuration = session.trial_data['configuration_i']
    i_poke_task_1 = configuration[0]
    i_poke_task_2 = configuration[task_2_change[0]]
    i_poke_task_3 = configuration[task_3_change[0]]
    print(i_poke_task_1, i_poke_task_2, i_poke_task_3)
    all_pokes = np.concatenate([a_pokes, b_pokes, i_pokes])
    all_pokes = np.unique(all_pokes)
    
    poke_I = 'poke_'+str(i_poke_task_1)
    poke_I_task_2 = 'poke_'+str(i_poke_task_2)
    poke_I_task_3 = 'poke_'+str(i_poke_task_3)

    #Events for Pokes Irrespective of Meaning
    pokes = {}
    for i, poke in enumerate(all_pokes):
        pokes[poke] = [event.time for event in session.events if event.name in ['poke_'+str(all_pokes[i])]]
    
    events_and_times = [[event.name, event.time] for event in session.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]

    poke_list = []
    initation_choice = []
    choice_state = False 
    choice_state_count = 0
    poke_identity = []
    initiation_time_stamps = []
    b_count = 0

    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
    
    predictors_B = predictor_B_Task_1+predictor_B_Task_2+predictor_B_Task_3
    ind_B = np.where(predictors_B == 1)[0]
    
    for i,event in enumerate(events_and_times):
        if 'choice_state' in event:
            if choice_state_count > 0:
                b_count +=1 
            choice_state_count +=1 
            choice_state = True  
            if choice_state_count <= len(task_1):
                poke_identity.append(poke_I)
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
                poke_identity.append(poke_I_task_2)
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                poke_identity.append(poke_I_task_3)
            initiation_time_stamps.append(event[1])
            
        if poke_A1_A2_A3 == True:
            if choice_state_count <= len(task_1):
                if poke_B in event: 
                    if choice_state == True:
                        if b_count in ind_B:
                            choice_state = False
                            poke_list.append(event[1])
                            poke_identity.append(poke_B)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_A in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

    
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
                if poke_B_task_2 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            choice_state = False
                            poke_list.append(event[1])
                            poke_identity.append(poke_B_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_A_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False
                        
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_B_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            choice_state = False
                            poke_list.append(event[1])
                            poke_identity.append(poke_B_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_A_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False
                        
        elif poke_A1_B2_B3 == True:
            if choice_state_count <= len(task_1):
                if poke_B in event: 
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_A in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False
                            
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):             
                if poke_A_task_2 in event:
                    if choice_state == True: 
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False     
                            poke_identity.append(poke_A_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_A_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_B_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
        elif poke_A1_B2_A3 == True: 
            if choice_state_count <= len(task_1):
                if poke_B in event: 
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_A in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):                                  
                if poke_A_task_2 in event:
                    if choice_state == True:  
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False     
                            poke_identity.append(poke_A_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_B_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_B_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_A_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False
                        
        elif poke_A1_A2_B3 == True:
            if choice_state_count <= len(task_1):
                if poke_B in event: 
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_A in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                        
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
                if poke_B_task_2 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_A_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False
                        

            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_A_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False
                        
                        
        elif poke_B1_A2_A3 == True:
            if choice_state_count <= len(task_1):
                if poke_A in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_B in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                            
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
                if poke_B_task_2 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_A_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                            
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_B_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_A_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                        
        elif poke_B1_A2_B3 == True:
            if choice_state_count <= len(task_1):             
                if poke_A in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                        
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
                if poke_B_task_2 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B_task_2)
                            initation_choice.append(1) 
                            initation_choice.append(1)

                elif poke_A_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                            
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):             
                if poke_A_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
                        
        elif poke_B1_B2_A3 == True:
            if choice_state_count <= len(task_1):
                if poke_A in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                            
            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):                                       
                if poke_A_task_2 in event:
                    if choice_state == True:  
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False     
                            poke_identity.append(poke_A_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_B_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_B_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_A_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                            
        elif poke_B1_B2_B3 == True:
            if choice_state_count <= len(task_1):
                if poke_A in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_B in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

            elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):                                       
                if poke_A_task_2 in event:
                    if choice_state == True:  
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False     
                            poke_identity.append(poke_A_task_2)
                            initation_choice.append(1)
                            initation_choice.append(1)

                elif poke_B_task_2 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

                        
            elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
                if poke_A_task_3 in event:
                    if choice_state == True:
                        if b_count in ind_B:
                            poke_list.append(event[1])
                            choice_state = False
                            poke_identity.append(poke_A_task_3)
                            initation_choice.append(1)
                            initation_choice.append(1)
                elif poke_B_task_3 in event:
                    if choice_state == True:
                        initation_choice.append(0)
                        choice_state = False

    #initiation_time_stamps = np.asarray(initiation_time_stamps)
    #initiation_time_stamps = initiation_time_stamps[ind_B]
    #outcomes_non_forced = outcomes_non_forced[ind_B]
    
    return poke_identity,poke_list,outcomes_non_forced,initation_choice,initiation_time_stamps

def predictors_around_pokes(session):
    poke_identity,poke_list,outcomes_non_forced,initation_choice,initiation_time_stamps = extract_poke_times(session)
    unique_pokes = np.unique(poke_identity)
    
    poke_1_id = unique_pokes[0]
    poke_2_id = unique_pokes[1]
    poke_3_id = unique_pokes[2]
    poke_4_id = unique_pokes[3]

        
    poke_1 = np.zeros(len(poke_identity))
    poke_2 = np.zeros(len(poke_identity))
    poke_3 = np.zeros(len(poke_identity))
    poke_4 = np.zeros(len(poke_identity))
    
    outcomes =[]
    for o,outcome in enumerate(outcomes_non_forced):
        outcomes.append(outcome)

    outcomes = np.asarray(outcomes)  

    for p,poke in enumerate(poke_identity):
            if poke == poke_1_id:
                poke_1[p] = 1
            if poke == poke_2_id:
                poke_2[p] = 1
            elif poke == poke_3_id:
                poke_3[p] = 1
            elif poke == poke_4_id:
                poke_4[p] = 1
              
    return poke_1,poke_2,poke_3,poke_4,outcomes,initation_choice,unique_pokes

def histogram(session):
    
    poke_identity,poke_list,outcomes_non_forced,initation_choice,initiation_time_stamps = extract_poke_times(session)
    
    all_events = np.concatenate([poke_list,initiation_time_stamps])
    all_events = sorted(all_events)
    
    neurons = np.unique(session.ephys[0])
    spikes = session.ephys[1]
    window_to_plot = 100
    all_neurons_all_spikes_raster_plot_task = []
    bin_width_ms = 5
    smooth_sd_ms = 10
    fr_convert = 1000
    trial_duration = 100
    bin_edges_trial = np.arange(-100,trial_duration, bin_width_ms)
   
    for i,neuron in enumerate(neurons):  
        spikes_ind = np.where(session.ephys[0] == neuron)
        spikes_n = spikes[spikes_ind]
        all_spikes_raster_plot_task = []
        for event in all_events:
            period_min = event - window_to_plot
            period_max = event + window_to_plot
            spikes_ind = spikes_n[(spikes_n >= period_min) & (spikes_n<= period_max)]
            spikes_to_save = (spikes_ind - event)   
            
            hist_task,edges_task = np.histogram(spikes_to_save, bins= bin_edges_trial)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            normalised_task = normalised_task*fr_convert
            all_spikes_raster_plot_task.append(normalised_task)
            #mean_time = np.mean(all_spikes_raster_plot_task)
        all_neurons_all_spikes_raster_plot_task.append(all_spikes_raster_plot_task)
    return all_neurons_all_spikes_raster_plot_task
     
   
def regression_pokes_aligned(session):     
    C = []
    
    # Finding correlation coefficients for task 1 
    #for s,session in enumerate(experiment):
        
    all_neurons_all_spikes_raster_plot_task = histogram(session)
    all_neurons_all_spikes_raster_plot_task = np.asarray(all_neurons_all_spikes_raster_plot_task)

    poke_1,poke_2,poke_3,poke_4,outcomes,initation_choice,unique_pokes = predictors_around_pokes(session)
    n_neurons = all_neurons_all_spikes_raster_plot_task.shape[0]
    n_timepoints = all_neurons_all_spikes_raster_plot_task.shape[2]
    initation_choice =initation_choice[:len(poke_1)]
    initation_choice = np.asarray(initation_choice)
    
    all_neurons_all_spikes_raster_plot_task = all_neurons_all_spikes_raster_plot_task[:,:len(poke_1),:]
    #initiation_choice_outcome = initation_choice*outcomes
   
    predictors = OrderedDict([
                                  ('poke_1', poke_1),
                                  ('poke_2', poke_2), 
                                  ('poke_3', poke_3),
                                  ('poke_4', poke_4)])
                                  #('outcomes', outcomes),
                                  #('initation_choice', initation_choice)])
                                  #('initiation_choice_outcome', initiation_choice_outcome)])

   
    
    X_task_1 = np.vstack(predictors.values()).T[:len(initation_choice),:].astype(float)
    
    u, s, v = np.linalg.svd(X_task_1, full_matrices = False)
    print(matrix_rank(X_task_1))
    print(max(s)/min(s))
    
    b = np.linalg.pinv(X_task_1)
    plt.figure(7)
    plt.imshow(b,aspect = 'auto')
    plt.yticks(ticks=[0,1,2,3,4], labels= [unique_pokes[0],unique_pokes[1],unique_pokes[2],unique_pokes[3], 'Choice'])
    plt.colorbar()
    
    n_predictors = X_task_1.shape[1]
    y_t1 = all_neurons_all_spikes_raster_plot_task.reshape([all_neurons_all_spikes_raster_plot_task.shape[1],-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
    ols = LinearRegression(copy_X = True,fit_intercept= True)
    ols.fit(X_task_1,y_t1)
    C.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings

    C = np.concatenate(C,0)
    plt.figure(6)
    plt.plot(C[1,:,:])
    
