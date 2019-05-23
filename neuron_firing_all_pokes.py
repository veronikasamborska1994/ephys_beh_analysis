#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:12:15 2018

@author: veronikasamborska
"""

import numpy as np
import ephys_beh_import as ep
import heatmap_aligned as ha
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import regressions as re
import matplotlib.pyplot as plt


def make_consistent_poke_names(session):
# Function to make A pokes to be in the same spatial position in three tasks 
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3, poke_B1_B2_A3  = ep.poke_A_B_make_consistent(session)        
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = ep.extract_choice_pokes(session)

    if poke_A1_A2_A3 == True:
        poke_a1 = poke_A
        poke_a2 = poke_A_task_2
        poke_a3 = poke_A_task_3
        poke_b1 = poke_B
        poke_b2 = poke_B_task_2
        poke_b3 = poke_B_task_3
    
    elif poke_A1_B2_B3 == True:
        poke_a1 = poke_A
        poke_a2 = poke_B_task_2
        poke_a3 = poke_B_task_3
        poke_b1 = poke_B
        poke_b2 = poke_A_task_2
        poke_b3 = poke_A_task_3
   
    elif poke_A1_B2_A3 == True: 
        poke_a1 = poke_A
        poke_a2 = poke_B_task_2
        poke_a3 = poke_A_task_3
        poke_b1 = poke_B
        poke_b2 = poke_A_task_2
        poke_b3 = poke_B_task_3
        
    elif poke_A1_A2_B3 == True:
        poke_a1 = poke_A
        poke_a2 = poke_A_task_2
        poke_a3 = poke_B_task_3
        poke_b1 = poke_B
        poke_b2 = poke_B_task_2
        poke_b3 = poke_A_task_3
       
    elif poke_B1_B2_B3 == True:
        poke_a1 = poke_B
        poke_a2 = poke_B_task_2
        poke_a3 = poke_B_task_3
        poke_b1 = poke_A
        poke_b2 = poke_A_task_2
        poke_b3 = poke_A_task_3
      
    elif poke_B1_A2_A3 == True:
        poke_a1 = poke_B
        poke_a2 = poke_A_task_2
        poke_a3 = poke_A_task_3
        poke_b1 = poke_A
        poke_b2 = poke_B_task_2
        poke_b3 = poke_B_task_3
       
    elif poke_B1_A2_B3 == True:
        poke_a1 = poke_B
        poke_a2 = poke_A_task_2
        poke_a3 = poke_B_task_3
        poke_b1 = poke_A
        poke_b2 = poke_B_task_2
        poke_b3 = poke_A_task_3    
       
    elif poke_B1_B2_A3 == True:
        poke_a1 = poke_B
        poke_a2 = poke_B_task_2
        poke_a3 = poke_A_task_3
        poke_b1 = poke_A
        poke_b2 = poke_A_task_2
        poke_b3 = poke_B_task_3
        
    return poke_I, poke_I_task_2, poke_I_task_3, poke_a1, poke_a2, poke_a3, poke_b1, poke_b2, poke_b3 

def task_indicies(session):
    forced_trials = session.trial_data['forced_trial']
    task = session.trial_data['task']
    non_forced_array = np.where(forced_trials == 0)[0]
    task_non_forced = task[non_forced_array]
    task_non_forced = task[non_forced_array]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_2_start = task_2[0]
    task_3_start = task_2[-1]+1
    return task_2_start, task_3_start, 

def latent_state_indices(session):
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = ep.state_indices(session)
    task_2_start, task_3_start  = task_indicies(session)
    state_t2_a_good += task_2_start
    state_t2_b_good += task_2_start
    state_t3_a_good += task_3_start
    state_t3_b_good += task_3_start
    return state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good


def means_for_heatplots(session):
  
    # Extract spikes and event times for the session 
    aligned_spikes= session.aligned_rates 
    n_neurons = aligned_spikes.shape[1]
    
    # Find Task Indicies
    task_2_start, task_3_start = task_indicies(session)
    
    # Indicies for A and B choices
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)    
    
    index_reward = np.where(reward ==1)
    index_no_reward = np.where(reward ==0)
    
    task_1_reward = reward[:task_2_start] 
    task_2_reward = reward[task_2_start:task_3_start] 
    task_3_reward = reward[task_3_start:] 
    
        
    a_1 = np.where(predictor_A_Task_1 == 1) #Poke A task 1 idicies
    a_2 = np.where(predictor_A_Task_2 == 1) #Poke A task 2 idicies
    a_3 = np.where(predictor_A_Task_3 == 1) #Poke A task 3 idicies
    b_1 = np.where(predictor_B_Task_1 == 1) #Poke A task 1 idicies
    b_2 = np.where(predictor_B_Task_2 == 1) #Poke A task 2 idicies
    b_3 = np.where(predictor_B_Task_3 == 1) #Poke A task 3 idicies
        
    a1_nR = [a for a in a_1[0] if a in index_no_reward[0]]
    a2_nR = [a for a in a_2[0] if a in index_no_reward[0]]
    a3_nR = [a for a in a_3[0] if a in index_no_reward[0]] 
    
    a1_R = [a for a in a_1[0] if a in index_reward[0]]
    a2_R= [a for a in a_2[0] if a in index_reward[0]]
    a3_R = [a for a in a_3[0] if a in index_reward[0]] 
    
    b1_nR = [a for a in b_1[0] if a in index_no_reward[0]]
    b2_nR = [a for a in b_2[0] if a in index_no_reward[0]]
    b3_nR = [a for a in b_3[0] if a in index_no_reward[0]] 
    
    b1_R = [a for a in b_1[0] if a in index_reward[0]]
    b2_R= [a for a in b_2[0] if a in index_reward[0]]
    b3_R = [a for a in b_3[0] if a in index_reward[0]] 
    
    spikes_B_task_1_R =aligned_spikes[b1_R]
    spikes_A_task_1_R =aligned_spikes[a1_R]
    spikes_B_task_2_R =aligned_spikes[b2_R]
    spikes_A_task_2_R =aligned_spikes[a2_R]
    spikes_B_task_3_R =aligned_spikes[b3_R]
    spikes_A_task_3_R =aligned_spikes[a3_R]
    
    spikes_B_task_1_nR =aligned_spikes[b1_nR]
    spikes_A_task_1_nR =aligned_spikes[a1_nR]
    spikes_B_task_2_nR =aligned_spikes[b2_nR]
    spikes_A_task_2_nR =aligned_spikes[a2_nR]
    spikes_B_task_3_nR =aligned_spikes[b3_nR]
    spikes_A_task_3_nR =aligned_spikes[a3_nR]
    
    mean_spikes_B_task_1_R = np.mean(spikes_B_task_1_R,axis = 0)
    mean_spikes_A_task_1_R = np.mean(spikes_A_task_1_R,axis = 0)
    mean_spikes_B_task_2_R = np.mean(spikes_B_task_2_R,axis = 0)
    mean_spikes_A_task_2_R = np.mean(spikes_A_task_2_R,axis = 0)
    mean_spikes_B_task_3_R = np.mean(spikes_B_task_3_R,axis = 0)
    mean_spikes_A_task_3_R = np.mean(spikes_A_task_3_R,axis = 0)
    
    mean_spikes_B_task_1_nR = np.mean(spikes_B_task_1_nR,axis = 0)
    mean_spikes_A_task_1_nR = np.mean(spikes_A_task_1_nR,axis = 0)
    mean_spikes_B_task_2_nR = np.mean(spikes_B_task_2_nR,axis = 0)
    mean_spikes_A_task_2_nR = np.mean(spikes_A_task_2_nR,axis = 0)
    mean_spikes_B_task_3_nR = np.mean(spikes_B_task_3_nR,axis = 0)
    mean_spikes_A_task_3_nR = np.mean(spikes_A_task_3_nR,axis = 0)
    
    # Spikes around different initiation ports
    initiation_port_task_1 = aligned_spikes[:task_2_start] 
    initiation_port_task_1_R = initiation_port_task_1[np.where(task_1_reward == 1)]
    initiation_port_task_1_nR = initiation_port_task_1[np.where(task_1_reward == 0)]

    initiation_port_task_2 = aligned_spikes[task_2_start: task_3_start]
    initiation_port_task_2_R = initiation_port_task_2[np.where(task_2_reward == 1)]
    initiation_port_task_2_nR = initiation_port_task_2[np.where(task_2_reward == 0)]


    initiation_port_task_3 = aligned_spikes[task_3_start:]
    initiation_port_task_3_R = initiation_port_task_3[np.where(task_3_reward == 1)]
    initiation_port_task_3_nR = initiation_port_task_3[np.where(task_3_reward == 0)]


    
    mean_spikes_I_task_1_R = np.mean(initiation_port_task_1_R,axis = 0)
    mean_spikes_I_task_2_R = np.mean(initiation_port_task_2_R,axis = 0)
    mean_spikes_I_task_3_R = np.mean(initiation_port_task_3_R,axis = 0)
    
    mean_spikes_I_task_1_nR = np.mean(initiation_port_task_1_nR,axis = 0)
    mean_spikes_I_task_2_nR = np.mean(initiation_port_task_2_nR,axis = 0)
    mean_spikes_I_task_3_nR = np.mean(initiation_port_task_3_nR,axis = 0)
    
    
    vector_for_normalising = np.concatenate([mean_spikes_A_task_1_R,mean_spikes_A_task_2_R, mean_spikes_A_task_3_R,\
                                             mean_spikes_B_task_1_R, mean_spikes_B_task_2_R, mean_spikes_B_task_3_R,\
                                             mean_spikes_I_task_1_R, mean_spikes_I_task_2_R,mean_spikes_I_task_3_R,\
                                             mean_spikes_A_task_1_nR,mean_spikes_A_task_2_nR, mean_spikes_A_task_3_nR,\
                                             mean_spikes_B_task_1_nR, mean_spikes_B_task_2_nR, mean_spikes_B_task_3_nR,\
                                             mean_spikes_I_task_1_nR, mean_spikes_I_task_2_nR, mean_spikes_I_task_3_nR
                                             ], axis = 1)
    
    
 
    normalised = (vector_for_normalising - np.min(vector_for_normalising,1)[:, None]) / (np.max(vector_for_normalising,1)[:, None]+1e-08 - np.min(vector_for_normalising,1)[:, None])
    

    A_task_1_norm_R = normalised[:, :mean_spikes_I_task_1_R.shape[1]]   
    A_task_2_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]:mean_spikes_I_task_1_R.shape[1]*2]
    A_task_3_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*2:mean_spikes_I_task_1_R.shape[1]*3]
    B_task_1_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*3:mean_spikes_I_task_1_R.shape[1]*4]
    B_task_2_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*4:mean_spikes_I_task_1_R.shape[1]*5]
    B_task_3_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*5:mean_spikes_I_task_1_R.shape[1]*6]
    I_task_1_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*6:mean_spikes_I_task_1_R.shape[1]*7]
    I_task_2_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*7:mean_spikes_I_task_1_R.shape[1]*8]
    I_task_3_norm_R = normalised[:, mean_spikes_I_task_1_R.shape[1]*8:mean_spikes_I_task_1_R.shape[1]*9]

    A_task_1_norm_nR = normalised[:,mean_spikes_I_task_1_R.shape[1]*9 :mean_spikes_I_task_1_nR.shape[1]*10]   
    A_task_2_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*10 :mean_spikes_I_task_1_nR.shape[1]*11]
    A_task_3_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*11 :mean_spikes_I_task_1_nR.shape[1]*12]
    B_task_1_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*12 :mean_spikes_I_task_1_nR.shape[1]*13]
    B_task_2_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*13 :mean_spikes_I_task_1_nR.shape[1]*14]
    B_task_3_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*14 :mean_spikes_I_task_1_nR.shape[1]*15]
    I_task_1_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*15 :mean_spikes_I_task_1_nR.shape[1]*16]
    I_task_2_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*16 :mean_spikes_I_task_1_nR.shape[1]*17]
    I_task_3_norm_nR = normalised[:, mean_spikes_I_task_1_R.shape[1]*17 :mean_spikes_I_task_1_nR.shape[1]*18]
    
    
    return  A_task_1_norm_R, A_task_2_norm_R, A_task_3_norm_R, B_task_1_norm_R, B_task_2_norm_R, B_task_3_norm_R, I_task_1_norm_R,\
    I_task_2_norm_R, I_task_3_norm_R, A_task_1_norm_nR, A_task_2_norm_nR, A_task_3_norm_nR, B_task_1_norm_nR, B_task_2_norm_nR,\
    B_task_3_norm_nR, I_task_1_norm_nR, I_task_2_norm_nR, I_task_3_norm_nR
    
    
    
def a_b_times(session):
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = ep.extract_choice_pokes(session)
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
       
    

    task = session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3_change[0]])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3_change[0]])
    
    events_and_times = [[event.name, event.time] for event in session.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    poke_list = []
    for i,event in enumerate(events_and_times):
        if 'choice_state' in event:
            if events_and_times[-1] != 'choice_state':
                poke_time = events_and_times[i+1]
                poke_list.append(poke_time[1])
    poke_list = np.array(poke_list)       
    time_A_task_1 = poke_list[np.where(predictor_A_Task_1 == 1)]
    time_A_task_2 = poke_list[np.where(predictor_A_Task_2 == 1)]
    time_A_task_3 = poke_list[np.where(predictor_A_Task_3 == 1)]
    
    time_B_task_1 = poke_list[np.where(predictor_B_Task_1 == 1)]
    time_B_task_2 = poke_list[np.where(predictor_B_Task_2 == 1)]
    time_B_task_3 = poke_list[np.where(predictor_B_Task_3 == 1)]
    
    return time_A_task_1, time_A_task_2,time_A_task_3, time_B_task_1, time_B_task_2,time_B_task_3
    
        
                    
def means_for_trial_aligned_FR(session):
    window_to_plot = 2000
    spikes_list = []
    pyControl_choice = [event.time for event in session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    raw_spikes = session.ephys
    neurons = np.unique(raw_spikes[0])
    for n in neurons:
        n_indexes = np.where(raw_spikes[0] == n)
        spikes = raw_spikes[1][n_indexes]
        spikes_list.append(spikes)
    spikes_list = np.array(spikes_list)
    
    time_A_task_1, time_A_task_2,time_A_task_3, time_B_task_1, time_B_task_2,time_B_task_3 = a_b_times(session)

    task_2_start, task_3_start = task_indicies(session)
    initiation_port_task_1 = pyControl_choice[:task_2_start]
    initiation_port_task_2 = pyControl_choice[task_2_start: task_3_start]
    initiation_port_task_3 = pyControl_choice[task_3_start:]

    a_1_list = []
    a_2_list = []
    a_3_list = []
    b_1_list = []
    b_2_list = []
    b_3_list = []
    
    i_1_list = []
    i_2_list = []
    i_3_list = []
    
    for neuron in range(len(neurons)):
        spikes_times = spikes_list[neuron]        
        spikes_to_plot_a_task_1 =  np.array([])
        spikes_to_plot_a_task_2 =  np.array([])
        spikes_to_plot_a_task_3 =  np.array([])
        
        spikes_to_plot_b_task_1 =  np.array([])
        spikes_to_plot_b_task_2 =  np.array([])
        spikes_to_plot_b_task_3 =  np.array([])
        
        spikes_to_plot_i_task_1 =  np.array([])
        spikes_to_plot_i_task_2 =  np.array([])
        spikes_to_plot_i_task_3 =  np.array([])
        
        for a_task_1 in time_A_task_1:
            period_min_a_1 = a_task_1 - window_to_plot
            period_max_a_1 = a_task_1 + window_to_plot
            spikes_ind_a_task_1 = spikes_times[(spikes_times >= period_min_a_1) & (spikes_times<= period_max_a_1)]
            spikes_to_save_a_task_1 = spikes_ind_a_task_1 - a_task_1   
            spikes_to_plot_a_task_1 = np.append(spikes_to_plot_a_task_1,spikes_to_save_a_task_1)
        a_1_list.append(spikes_to_plot_a_task_1)
        
        for a_task_2 in time_A_task_2:
            period_min_a_2 = a_task_2 - window_to_plot
            period_max_a_2 = a_task_2 + window_to_plot
            spikes_ind_a_task_2 = spikes_times[(spikes_times >= period_min_a_2) & (spikes_times<= period_max_a_2)]
            spikes_to_save_a_task_2 = spikes_ind_a_task_2 - a_task_2    
            spikes_to_plot_a_task_2 = np.append(spikes_to_plot_a_task_2, spikes_to_save_a_task_2)
        a_2_list.append(spikes_to_plot_a_task_2)
            
        for a_task_3 in time_A_task_3:
            period_min_a_3 = a_task_3 - window_to_plot
            period_max_a_3 = a_task_3 + window_to_plot
            spikes_ind_a_task_3 = spikes_times[(spikes_times >= period_min_a_3) & (spikes_times<= period_max_a_3)]
            spikes_to_save_a_task_3 = spikes_ind_a_task_3 - a_task_3   
            spikes_to_plot_a_task_3 = np.append(spikes_to_plot_a_task_3, spikes_to_save_a_task_3)
        a_3_list.append(spikes_to_plot_a_task_3)
        
        for b_task_1 in time_B_task_1:
            period_min_b_1 = b_task_1 - window_to_plot
            period_max_b_1 = b_task_1 + window_to_plot
            spikes_ind_b_task_1 = spikes_times[(spikes_times >= period_min_b_1) & (spikes_times<= period_max_b_1)]
            spikes_to_save_b_task_1 = spikes_ind_b_task_1 - b_task_1   
            spikes_to_plot_b_task_1 = np.append(spikes_to_plot_b_task_1, spikes_to_save_b_task_1)
        b_1_list.append(spikes_to_plot_b_task_1)
        
        for b_task_2 in time_B_task_2:
            period_min_b_2 = b_task_2 - window_to_plot
            period_max_b_2 = b_task_2 + window_to_plot
            spikes_ind_b_task_2 = spikes_times[(spikes_times >= period_min_b_2) & (spikes_times<= period_max_b_2)]
            spikes_to_save_b_task_2 = spikes_ind_b_task_2 - b_task_2    
            spikes_to_plot_b_task_2 = np.append(spikes_to_plot_b_task_2, spikes_to_save_b_task_2)
        b_2_list.append(spikes_to_plot_b_task_2) 
        
        for b_task_3 in time_B_task_3:
            period_min_b_3 = b_task_3 - window_to_plot
            period_max_b_3 = b_task_3 + window_to_plot
            spikes_ind_b_task_3 = spikes_times[(spikes_times >= period_min_b_3) & (spikes_times<= period_max_b_3)]
            spikes_to_save_b_task_3 = spikes_ind_b_task_3 - b_task_3   
            spikes_to_plot_b_task_3 = np.append(spikes_to_plot_b_task_3, spikes_to_save_b_task_3)
        b_3_list.append(spikes_to_plot_b_task_3)   
        
        
        for i_task_1 in initiation_port_task_1:
            period_min_i_1 = i_task_1 - window_to_plot
            period_max_i_1 = i_task_1 + window_to_plot
            spikes_ind_i_task_1 = spikes_times[(spikes_times >= period_min_i_1) & (spikes_times<= period_max_i_1)]
            spikes_to_save_i_task_1 = spikes_ind_i_task_1 - i_task_1   
            spikes_to_plot_i_task_1 = np.append(spikes_to_plot_i_task_1, spikes_to_save_i_task_1)
        i_1_list.append(spikes_to_plot_i_task_1)
        
        for i_task_2 in initiation_port_task_2:
            period_min_i_2 = i_task_2 - window_to_plot
            period_max_i_2 = i_task_2 + window_to_plot
            spikes_ind_i_task_2 = spikes_times[(spikes_times >= period_min_i_2) & (spikes_times<= period_max_i_2)]
            spikes_to_save_i_task_2 = spikes_ind_i_task_2 - i_task_2    
            spikes_to_plot_i_task_2 = np.append(spikes_to_plot_i_task_2, spikes_to_save_i_task_2)
        i_2_list.append(spikes_to_plot_i_task_2) 
        
        for i_task_3 in initiation_port_task_3:
            period_min_i_3 = i_task_3 - window_to_plot
            period_max_i_3 = i_task_3 + window_to_plot
            spikes_ind_i_task_3 = spikes_times[(spikes_times >= period_min_i_3) & (spikes_times<= period_max_i_3)]
            spikes_to_save_i_task_3 = spikes_ind_i_task_3 - i_task_3   
            spikes_to_plot_i_task_3 = np.append(spikes_to_plot_i_task_3, spikes_to_save_i_task_3)
        i_3_list.append(spikes_to_plot_i_task_3)   
    a_1_list = np.array(a_1_list)/1000
    a_2_list = np.array(a_2_list)/1000
    a_3_list = np.array(a_3_list)/1000
    b_1_list = np.array(b_1_list)/1000
    b_2_list = np.array(b_2_list)/1000
    b_3_list = np.array(b_3_list)/1000
    i_1_list = np.array(i_1_list)/1000
    i_2_list = np.array(i_2_list)/1000
    i_3_list = np.array(i_3_list)/1000
        
        
    return a_1_list, a_2_list, a_3_list, b_1_list, b_2_list, b_3_list, i_1_list, i_2_list, i_3_list
        
def histograms(session):
    a_1_list, a_2_list, a_3_list, b_1_list, b_2_list, b_3_list, i_1_list, i_2_list, i_3_list = means_for_trial_aligned_FR(session)
    neurons = range(len(a_1_list))
    bin_width_ms = 0.1
    smooth_sd_ms = 0.1
    fr_convert = 10
    trial_duration = 2
    bin_edges_trial = np.arange(-2,trial_duration, bin_width_ms)
    hist_a1_list = []
    hist_a2_list = []
    hist_a3_list = []
    
    hist_b1_list = []
    hist_b2_list = []
    hist_b3_list = []
    
    hist_i1_list = []
    hist_i2_list = []
    hist_i3_list = []
    
    for neuron in neurons:
        hist_task_a1,edges_task_a1 = np.histogram(a_1_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_a1 = hist_task_a1/len(a_1_list[neuron])
        hist_task_a1 = hist_task_a1/bin_width_ms
        normalised_a_1 = gaussian_filter1d(hist_task_a1.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_a_1= normalised_a_1*fr_convert
        hist_a1_list.append(normalised_a_1)
        
        hist_task_a2,edges_task_a2 = np.histogram(a_2_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_a2 = hist_task_a2/len(a_2_list[neuron])
        hist_task_a2 = hist_task_a2/bin_width_ms
        normalised_a_2 = gaussian_filter1d(hist_task_a2.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_a_2= normalised_a_2*fr_convert
        hist_a2_list.append(normalised_a_2)
                
        hist_task_a3,edges_task_a3 = np.histogram(a_3_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_a3 = hist_task_a3/len(a_3_list[neuron])
        hist_task_a3 = hist_task_a3/bin_width_ms
        normalised_a_3 = gaussian_filter1d(hist_task_a3.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_a_3= normalised_a_3*fr_convert
        hist_a3_list.append(normalised_a_3)
        
        
        hist_task_b1,edges_task_b1 = np.histogram(b_1_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_b1 = hist_task_b1/len(b_1_list[neuron])
        hist_task_b1 = hist_task_b1/bin_width_ms
        normalised_b_1 = gaussian_filter1d(hist_task_b1.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_b_1= normalised_b_1*fr_convert
        hist_b1_list.append(normalised_b_1)
        
        hist_task_b2,edges_task_b2 = np.histogram(b_2_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_b2 = hist_task_b2/len(b_2_list[neuron])
        hist_task_b2 = hist_task_b2/bin_width_ms
        normalised_b_2 = gaussian_filter1d(hist_task_b2.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_b_2= normalised_b_2*fr_convert
        hist_b2_list.append(normalised_b_2)
                
        hist_task_b3,edges_task_b3 = np.histogram(b_3_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_b3 = hist_task_b3/len(b_3_list[neuron])
        hist_task_b3 = hist_task_b3/bin_width_ms
        normalised_b_3 = gaussian_filter1d(hist_task_b3.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_b_3= normalised_b_3*fr_convert
        hist_b3_list.append(normalised_b_3)
        
        
        hist_task_i1,edges_task_i1 = np.histogram(i_1_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_i1 = hist_task_i1/len(b_1_list[neuron])
        hist_task_i1 = hist_task_i1/bin_width_ms
        normalised_i_1 = gaussian_filter1d(hist_task_i1.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_i_1= normalised_i_1*fr_convert
        hist_i1_list.append(normalised_i_1)
        
        hist_task_i2,edges_task_i2 = np.histogram(i_2_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_i2 = hist_task_i2/len(i_2_list[neuron])
        hist_task_i2 = hist_task_i2/bin_width_ms
        normalised_i_2 = gaussian_filter1d(hist_task_i2.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_i_2= normalised_i_2*fr_convert
        hist_i2_list.append(normalised_i_2)
                
        hist_task_i3,edges_task_i3 = np.histogram(i_3_list[neuron], bins= bin_edges_trial)# histogram per second
        hist_task_i3 = hist_task_i3/len(b_3_list[neuron])
        hist_task_i3 = hist_task_i3/bin_width_ms
        normalised_i_3 = gaussian_filter1d(hist_task_i3.astype(float), smooth_sd_ms/bin_width_ms)
        normalised_i_3= normalised_i_3*fr_convert
        hist_i3_list.append(normalised_i_3)
        
        
    return  hist_a1_list, hist_a2_list, hist_a3_list, hist_b1_list, hist_b2_list, hist_b3_list, hist_i1_list, hist_i2_list, hist_i3_list 
    

def normalise_hist(session):
    
     # Extract spikes and event times for the session 
    aligned_spikes= session.aligned_rates 
    n_neurons = aligned_spikes.shape[1]
    
    #Get histograms
    hist_a1_list, hist_a2_list, hist_a3_list, hist_b1_list,\
    hist_b2_list, hist_b3_list, hist_i1_list, hist_i2_list, hist_i3_list = histograms(session)

    vector_for_normalising = np.concatenate([hist_a1_list, hist_a2_list, hist_a3_list, hist_b1_list,\
    hist_b2_list, hist_b3_list, hist_i1_list, hist_i2_list, hist_i3_list],axis = 1)
    
    
    #normalised = (vector_for_normalising - np.min(vector_for_normalising,1)[:, None]) / ((np.max(vector_for_normalising,1)[:, None]+1e-08) - np.min(vector_for_normalising,1)[:, None])
    
    normalised = (vector_for_normalising - np.min(vector_for_normalising,1)[:, None]) / (np.max(vector_for_normalising,1)[:, None]+1e-08 - np.min(vector_for_normalising,1)[:, None])
    
    A_task_1_norm = normalised[:, :shape(hist_a1_list)[1]]   
    A_task_2_norm = normalised[:, shape(hist_a1_list)[1]:shape(hist_a1_list)[1]*2]
    A_task_3_norm = normalised[:, shape(hist_a1_list)[1]*2:shape(hist_a1_list)[1]*3]
    B_task_1_norm = normalised[:, shape(hist_a1_list)[1]*3:shape(hist_a1_list)[1]*4]
    B_task_2_norm = normalised[:, shape(hist_a1_list)[1]*4:shape(hist_a1_list)[1]*5]
    B_task_3_norm = normalised[:, shape(hist_a1_list)[1]*5:shape(hist_a1_list)[1]*6]
    I_task_1_norm = normalised[:, shape(hist_a1_list)[1]*6:shape(hist_a1_list)[1]*7]
    I_task_2_norm = normalised[:, shape(hist_a1_list)[1]*7:shape(hist_a1_list)[1]*8]
    I_task_3_norm = normalised[:, shape(hist_a1_list)[1]*8:shape(hist_a1_list)[1]*9]
    
    return A_task_1_norm, A_task_2_norm, A_task_3_norm, B_task_1_norm, B_task_2_norm, B_task_3_norm, I_task_1_norm,\
    I_task_2_norm, I_task_3_norm
    
def coordinates_for_plots(session):
    poke_I, poke_I_task_2, poke_I_task_3, poke_a1, poke_a2, poke_a3, poke_b1, poke_b2, poke_b3  = make_consistent_poke_names(session)
#    mean_spikes_A_task_1_norm, mean_spikes_A_task_2_norm, mean_spikes_A_task_3_norm, mean_spikes_B_task_1_norm,\
#    mean_spikes_B_task_2_norm, mean_spikes_B_task_3_norm, mean_spikes_I_task_1_norm, mean_spikes_I_task_2_norm,\
#    mean_spikes_I_task_3_norm  = means_for_heatplots(session)
   
    A_task_1_norm_R, A_task_2_norm_R, A_task_3_norm_R, B_task_1_norm_R, B_task_2_norm_R, B_task_3_norm_R, I_task_1_norm_R,\
    I_task_2_norm_R, I_task_3_norm_R, A_task_1_norm_nR, A_task_2_norm_nR, A_task_3_norm_nR, B_task_1_norm_nR, B_task_2_norm_nR,\
    B_task_3_norm_nR, I_task_1_norm_nR, I_task_2_norm_nR, I_task_3_norm_nR =  means_for_heatplots(session)
   
    #Set x coordinates of spike times 
    x_coordinates = x_coordinates = np.arange(A_task_1_norm_R.shape[1])
    # Adding x and y offsets to represent physical I pokes 
    # Task 1
    if int(poke_I[5]) == 1:
        x_coordinates_I_1_R = x_coordinates+300
        y_coordinates_I_1_R = I_task_1_norm_R + 5 
        x_coordinates_I_1_nR = x_coordinates+300
        y_coordinates_I_1_nR = I_task_1_norm_nR + 5 
    elif int(poke_I[5]) == 9:
        x_coordinates_I_1_R = x_coordinates+300
        y_coordinates_I_1_R = I_task_1_norm_R + 1
        x_coordinates_I_1_nR = x_coordinates+300
        y_coordinates_I_1_nR = I_task_1_norm_nR + 1
    # Task 2
    if int(poke_I_task_2[5]) == 1:
        x_coordinates_I_2_R = x_coordinates+300
        y_coordinates_I_2_R = I_task_2_norm_R + 5 
        x_coordinates_I_2_nR = x_coordinates+300
        y_coordinates_I_2_nR = I_task_2_norm_nR + 5 
    elif int(poke_I_task_2[5]) == 9:
        x_coordinates_I_2_R= x_coordinates+300
        y_coordinates_I_2_R = I_task_2_norm_R + 1
        x_coordinates_I_2_nR= x_coordinates+300
        y_coordinates_I_2_nR = I_task_2_norm_nR + 1
    # Task 3    
    if int(poke_I_task_3[5]) == 1:
        x_coordinates_I_3_R = x_coordinates+300
        y_coordinates_I_3_R = I_task_3_norm_R + 5 
        x_coordinates_I_3_nR = x_coordinates+300
        y_coordinates_I_3_nR = I_task_3_norm_nR + 5 
    elif int(poke_I_task_3[5]) == 9:
        x_coordinates_I_3_R = x_coordinates+300
        y_coordinates_I_3_R = I_task_3_norm_R + 1
        x_coordinates_I_3_nR = x_coordinates+300
        y_coordinates_I_3_nR = I_task_3_norm_nR + 1

    # Adding x and y offsets to represent physical A pokes 
    if int(poke_a1[5]) == 4 and int(poke_a2[5]) == 4 and int(poke_a3[5]) == 4:
        x_coordinates_A_1_R= x_coordinates+100
        y_coordinates_A_1_R = A_task_1_norm_R + 3 
        x_coordinates_A_2_R = x_coordinates+100
        y_coordinates_A_2_R = A_task_2_norm_R + 3
        x_coordinates_A_3_R = x_coordinates+100
        y_coordinates_A_3_R = A_task_3_norm_R + 3 
        x_coordinates_A_1_nR= x_coordinates+100
        y_coordinates_A_1_nR = A_task_1_norm_nR + 3 
        x_coordinates_A_2_nR = x_coordinates+100
        y_coordinates_A_2_nR = A_task_2_norm_nR + 3
        x_coordinates_A_3_nR = x_coordinates+100
        y_coordinates_A_3_nR = A_task_3_norm_nR + 3 
    elif int(poke_a1[5]) == 6 and int(poke_a2[5]) == 6 and int(poke_a3[5]) == 6:
        x_coordinates_A_1_R= x_coordinates+500
        y_coordinates_A_1_R = A_task_1_norm_R + 3 
        x_coordinates_A_2_R = x_coordinates+500
        y_coordinates_A_2_R = A_task_2_norm_R + 3
        x_coordinates_A_3_R = x_coordinates+500
        y_coordinates_A_3_R = A_task_3_norm_R + 3 
        x_coordinates_A_1_nR = x_coordinates+500
        y_coordinates_A_1_nR = A_task_1_norm_nR + 3 
        x_coordinates_A_2_nR = x_coordinates+500
        y_coordinates_A_2_nR = A_task_2_norm_nR + 3
        x_coordinates_A_3_nR = x_coordinates+500
        y_coordinates_A_3_nR = A_task_3_norm_nR + 3 
        
    # Adding x and y offsets to represent physical B task 1 pokes 
    if int(poke_b1[5]) == 2:
        x_coordinates_B_1_R= x_coordinates+200
        y_coordinates_B_1_R = B_task_1_norm_R + 4    
        x_coordinates_B_1_nR = x_coordinates+200
        y_coordinates_B_1_nR = B_task_1_norm_nR + 4    
    elif int(poke_b1[5]) == 3:
        x_coordinates_B_1_R= x_coordinates+400
        y_coordinates_B_1_R = B_task_1_norm_R + 4
        x_coordinates_B_1_nR = x_coordinates+400
        y_coordinates_B_1_nR = B_task_1_norm_nR + 4
    elif int(poke_b1[5]) == 7:
        x_coordinates_B_1_R= x_coordinates+200
        y_coordinates_B_1_R = B_task_1_norm_R + 2
        x_coordinates_B_1_nR = x_coordinates+200
        y_coordinates_B_1_nR = B_task_1_norm_nR + 2
    elif int(poke_b1[5]) == 8:
        x_coordinates_B_1_R = x_coordinates+400
        y_coordinates_B_1_R = B_task_1_norm_R + 2
        x_coordinates_B_1_nR = x_coordinates+400
        y_coordinates_B_1_nR = B_task_1_norm_nR + 2
    elif int(poke_b1[5]) == 1:
        x_coordinates_B_1_R = x_coordinates+300
        y_coordinates_B_1_R = B_task_1_norm_R + 5
        x_coordinates_B_1_nR = x_coordinates+300
        y_coordinates_B_1_nR = B_task_1_norm_nR + 5
    elif int(poke_b1[5]) == 9:
        x_coordinates_B_1_R = x_coordinates+300
        y_coordinates_B_1_R = B_task_1_norm_R + 1
        x_coordinates_B_1_nR = x_coordinates+300
        y_coordinates_B_1_nR = B_task_1_norm_nR + 1
    elif int(poke_b1[5]) == 4:
        x_coordinates_B_1_R = x_coordinates+100
        y_coordinates_B_1_R = B_task_1_norm_R + 3
        x_coordinates_B_1_nR= x_coordinates+100
        y_coordinates_B_1_nR = B_task_1_norm_nR + 3
    elif int(poke_b1[5]) == 6:
        x_coordinates_B_1_R = x_coordinates+500
        y_coordinates_B_1_R = B_task_1_norm_R + 3
        x_coordinates_B_1_nR = x_coordinates+500
        y_coordinates_B_1_nR = B_task_1_norm_nR + 3
        
    # Adding x and y offsets to represent physical B task 2 pokes 
    if int(poke_b2[5]) == 2:
        x_coordinates_B_2_R = x_coordinates+200
        y_coordinates_B_2_R = B_task_2_norm_R + 4 
        x_coordinates_B_2_nR = x_coordinates+200
        y_coordinates_B_2_nR = B_task_2_norm_nR + 4 
    elif int(poke_b2[5]) == 3:
        x_coordinates_B_2_R = x_coordinates+400
        y_coordinates_B_2_R = B_task_2_norm_R + 4
        x_coordinates_B_2_nR = x_coordinates+400
        y_coordinates_B_2_nR = B_task_2_norm_nR + 4
    elif int(poke_b2[5]) == 7:
        x_coordinates_B_2_R = x_coordinates+200
        y_coordinates_B_2_R = B_task_2_norm_R + 2
        x_coordinates_B_2_nR = x_coordinates+200
        y_coordinates_B_2_nR = B_task_2_norm_nR + 2
    elif int(poke_b2[5]) == 8:
        x_coordinates_B_2_R = x_coordinates+400
        y_coordinates_B_2_R = B_task_2_norm_R + 2
        x_coordinates_B_2_nR = x_coordinates+400
        y_coordinates_B_2_nR = B_task_2_norm_nR + 2
    elif int(poke_b2[5]) == 1:
        x_coordinates_B_2_R = x_coordinates+300
        y_coordinates_B_2_R = B_task_2_norm_R + 5
        x_coordinates_B_2_nR = x_coordinates+300
        y_coordinates_B_2_nR = B_task_2_norm_nR + 5
    elif int(poke_b2[5]) == 9:
        x_coordinates_B_2_R = x_coordinates+300
        y_coordinates_B_2_R = B_task_2_norm_R + 1
        x_coordinates_B_2_nR = x_coordinates+300
        y_coordinates_B_2_nR = B_task_2_norm_nR + 1
    elif int(poke_b2[5]) == 4:
        x_coordinates_B_2_R = x_coordinates+100
        y_coordinates_B_2_R = B_task_2_norm_R + 3
        x_coordinates_B_2_nR = x_coordinates+100
        y_coordinates_B_2_nR = B_task_2_norm_nR + 3
    elif int(poke_b2[5]) == 6:
        x_coordinates_B_2_R = x_coordinates+500
        y_coordinates_B_2_R = B_task_2_norm_R + 3
        x_coordinates_B_2_nR = x_coordinates+500
        y_coordinates_B_2_nR = B_task_2_norm_nR + 3
    
    # Adding x and y offsets to represent physical B task 3 pokes 
    if int(poke_b3[5]) == 2:
        x_coordinates_B_3_R = x_coordinates+200
        y_coordinates_B_3_R = B_task_3_norm_R + 4    
        x_coordinates_B_3_nR = x_coordinates+200
        y_coordinates_B_3_nR = B_task_3_norm_nR + 4    
    elif int(poke_b3[5]) == 3:
        x_coordinates_B_3_R = x_coordinates+400
        y_coordinates_B_3_R = B_task_3_norm_R + 4
        x_coordinates_B_3_nR = x_coordinates+400
        y_coordinates_B_3_nR = B_task_3_norm_nR + 4
    elif int(poke_b3[5]) == 7:
        x_coordinates_B_3_R = x_coordinates+200
        y_coordinates_B_3_R = B_task_3_norm_R + 2
        x_coordinates_B_3_nR = x_coordinates+200
        y_coordinates_B_3_nR = B_task_3_norm_nR + 2
    elif int(poke_b3[5]) == 8:
        x_coordinates_B_3_R = x_coordinates+400
        y_coordinates_B_3_R = B_task_3_norm_R + 2
        x_coordinates_B_3_nR = x_coordinates+400
        y_coordinates_B_3_nR = B_task_3_norm_nR + 2
    elif int(poke_b3[5]) == 1:
        x_coordinates_B_3_R = x_coordinates+300
        y_coordinates_B_3_R = B_task_3_norm_R + 5
        x_coordinates_B_3_nR = x_coordinates+300
        y_coordinates_B_3_nR = B_task_3_norm_nR + 5
    elif int(poke_b3[5]) == 9:
        x_coordinates_B_3_R = x_coordinates+300
        y_coordinates_B_3_R = B_task_3_norm_R + 1
        x_coordinates_B_3_nR = x_coordinates+300
        y_coordinates_B_3_nR = B_task_3_norm_nR + 1
    elif int(poke_b3[5]) == 4:
        x_coordinates_B_3_R = x_coordinates+100
        y_coordinates_B_3_R = B_task_3_norm_R + 3
        x_coordinates_B_3_nR = x_coordinates+100
        y_coordinates_B_3_nR = B_task_3_norm_nR + 3
    elif int(poke_b3[5]) == 6:
        x_coordinates_B_3_R = x_coordinates+500
        y_coordinates_B_3_R = B_task_3_norm_R + 3
        x_coordinates_B_3_nR = x_coordinates+500
        y_coordinates_B_3_nR = B_task_3_norm_nR + 3
        
    # Times of events      
     
    return x_coordinates_A_1_R, x_coordinates_A_2_R, x_coordinates_A_3_R,y_coordinates_A_1_R,\
    y_coordinates_A_2_R, y_coordinates_A_3_R, x_coordinates_B_1_R, x_coordinates_B_2_R,\
    x_coordinates_B_3_R, y_coordinates_B_1_R, y_coordinates_B_2_R, y_coordinates_B_3_R, x_coordinates_I_1_R,\
    x_coordinates_I_2_R, x_coordinates_I_3_R, y_coordinates_I_1_R, y_coordinates_I_2_R, y_coordinates_I_3_R,\
    x_coordinates_A_1_nR, x_coordinates_A_2_nR, x_coordinates_A_3_nR,y_coordinates_A_1_nR,\
    y_coordinates_A_2_nR, y_coordinates_A_3_nR, x_coordinates_B_1_nR, x_coordinates_B_2_nR,\
    x_coordinates_B_3_nR, y_coordinates_B_1_nR, y_coordinates_B_2_nR, y_coordinates_B_3_nR, x_coordinates_I_1_nR,\
    x_coordinates_I_2_nR, x_coordinates_I_3_nR, y_coordinates_I_1_nR, y_coordinates_I_2_nR, y_coordinates_I_3_nR
    


def session_spikes_vs_trials_plot(raw_spikes,pyControl_choice):
    spikes_list = []
    pyControl_choice = np.array(pyControl_choice)
    session_duration_ms = int(max(raw_spikes[1])) - int(min(raw_spikes[1]))
    neurons = np.unique(raw_spikes[0])
    for n in neurons:
        n_indexes = np.where(raw_spikes[0] == n)
        spikes = raw_spikes[1][n_indexes]
        spikes_list.append(spikes)
    return spikes_list, session_duration_ms
       



def plotting(experiment,experiment_aligned):
    
    bin_width_ms = 1000
    smooth_sd_ms = 4000
    pdf = PdfPages('/Users/veronikasamborska/Desktop/PFC.pdf')

    for s,session in zip(experiment, experiment_aligned):     
        # Get raw spike data across the task 
        raw_spikes = s.ephys
        neurons = np.unique(raw_spikes[0])
        # Get trial times
        pyControl_choice = [event.time for event in s.events if event.name in ['choice_state']]
        if len(raw_spikes[0]) > 0:
            spikes_list,session_duration_ms =  session_spikes_vs_trials_plot(raw_spikes,pyControl_choice)
            aligned_spikes= session.aligned_rates
            n_neurons = aligned_spikes.shape[1]
            n_trials = aligned_spikes.shape[0]
            t_out = session.t_out
            initiate_choice_t = session.target_times 
            initiate_t = initiate_choice_t[1]
            choice_t = initiate_choice_t[2]
            reward_t = initiate_choice_t[-2] +250
            ind_init = np.abs(t_out-initiate_t).argmin()
            ind_choice = np.abs(t_out-choice_t).argmin()
            ind_reward = np.abs(t_out-reward_t).argmin()
            
            ind_init_poke_1 = ind_init+300
            ind_choice_poke_1 = ind_choice+300
            ind_reward_poke_1 = ind_reward +300
            
            ind_init_poke_2 = ind_init+200
            ind_choice_poke_2 = ind_choice+200
            ind_reward_poke_2 = ind_reward +200
            
            ind_init_poke_3 = ind_init+400
            ind_choice_poke_3 = ind_choice+400
            ind_reward_poke_3 = ind_reward +400
            
                        
            ind_init_poke_4 = ind_init+100
            ind_choice_poke_4 = ind_choice+100
            ind_reward_poke_4 = ind_reward +100
            
            
            ind_init_poke_6 = ind_init+500
            ind_choice_poke_6 = ind_choice+500
            ind_reward_poke_6 = ind_reward +500
            
            ind_init_poke_7 = ind_init+200
            ind_choice_poke_7 = ind_choice+200
            ind_reward_poke_7 = ind_reward +200
            
            ind_init_poke_8 = ind_init+400
            ind_choice_poke_8 = ind_choice+400
            ind_reward_poke_8 = ind_reward +400
            
            
            ind_init_poke_9 = ind_init+300
            ind_choice_poke_9 = ind_choice+300
            ind_reward_poke_9 = ind_reward +300
            
            plt.ioff()
            
            x_coordinates_A_1_R, x_coordinates_A_2_R, x_coordinates_A_3_R,y_coordinates_A_1_R,\
            y_coordinates_A_2_R, y_coordinates_A_3_R, x_coordinates_B_1_R, x_coordinates_B_2_R,\
            x_coordinates_B_3_R, y_coordinates_B_1_R, y_coordinates_B_2_R, y_coordinates_B_3_R, x_coordinates_I_1_R,\
            x_coordinates_I_2_R, x_coordinates_I_3_R, y_coordinates_I_1_R, y_coordinates_I_2_R, y_coordinates_I_3_R,\
            x_coordinates_A_1_nR, x_coordinates_A_2_nR, x_coordinates_A_3_nR,y_coordinates_A_1_nR,\
            y_coordinates_A_2_nR, y_coordinates_A_3_nR, x_coordinates_B_1_nR, x_coordinates_B_2_nR,\
            x_coordinates_B_3_nR, y_coordinates_B_1_nR, y_coordinates_B_2_nR, y_coordinates_B_3_nR, x_coordinates_I_1_nR,\
            x_coordinates_I_2_nR, x_coordinates_I_3_nR, y_coordinates_I_1_nR, y_coordinates_I_2_nR, y_coordinates_I_3_nR = coordinates_for_plots(session)
            
            x_points = [132,232,232,332,332,432,432,532]
            y_points = [2.8,3.8,1.8,4.8,0.8,3.8,1.8,2.8]
            task_2_start, task_3_start = task_indicies(session) 
            task_1_end_time = pyControl_choice[task_2_start-1]/1000
            task_2_start_time = pyControl_choice[task_2_start]/1000
            task_2_end_time = pyControl_choice[task_3_start-1]/1000
            task_3_start_time = pyControl_choice[task_3_start]/1000
            task_3_end_time = pyControl_choice[-1]/1000
            
            task_arrays = np.zeros(shape=(n_trials,3))
            task_arrays[:task_2_start,0] = 1
            task_arrays[task_2_start:task_3_start,1] = 1
            task_arrays[task_3_start:,2] = 1
            for neuron,ID in enumerate(neurons):
                bin_edges = np.arange(0,session_duration_ms, bin_width_ms)
                hist,edges = np.histogram(spikes_list[neuron], bins= bin_edges)# histogram per second
                normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
                
                #Firing rate and trial rate 
                plt.figure()
                gridspec.GridSpec(2,2)
                plt.subplot2grid((2,1), (0,0))
                plt.axvspan(0, task_1_end_time, alpha=0.1, color='firebrick',zorder =0)
                plt.axvspan(task_2_start_time, task_2_end_time, alpha=0.1, color='cadetblue',zorder =0)
                plt.axvspan(task_3_start_time, task_3_end_time, alpha=0.1, color='olive',zorder =0)
                plt.plot(bin_edges[:-1]/bin_width_ms, normalised/max(normalised), label='Firing Rate', color ='cadetblue',zorder =1) 
                trial_rate,edges_py = np.histogram(pyControl_choice, bins=bin_edges)
                trial_rate = gaussian_filter1d(trial_rate.astype(float), smooth_sd_ms/bin_width_ms)
                plt.plot(bin_edges[:-1]/bin_width_ms, trial_rate/max(trial_rate), label='Rate', color = 'lightblue',zorder =1)
    
                 
                plt.xlabel('Time (ms)')
                plt.ylabel('Smoothed Firing Rate')
                plt.title('{}'.format(session.file_name))
                plt.legend()
                
                #Port Firing
                plt.subplot2grid((2,2), (1,0))
                plt.plot(x_coordinates_A_1_R, y_coordinates_A_1_R[neuron], color = 'firebrick')   
                plt.plot(x_coordinates_A_2_R, y_coordinates_A_2_R[neuron], color = 'cadetblue')      
                plt.plot(x_coordinates_A_3_R, y_coordinates_A_3_R[neuron], color = 'olive')      
                plt.plot(x_coordinates_B_1_R, y_coordinates_B_1_R[neuron], color = 'firebrick')    
                plt.plot(x_coordinates_B_2_R, y_coordinates_B_2_R[neuron], color = 'cadetblue' )       
                plt.plot(x_coordinates_B_3_R, y_coordinates_B_3_R[neuron], color = 'olive')   
                plt.plot(x_coordinates_I_1_R, y_coordinates_I_1_R[neuron], color = 'red', linestyle = ':')   
                plt.plot(x_coordinates_I_2_R, y_coordinates_I_2_R[neuron], color = 'blue',  linestyle = ':')      
                plt.plot(x_coordinates_I_3_R, y_coordinates_I_3_R[neuron], color = 'green',  linestyle = ':')  
                
                plt.plot(x_coordinates_A_1_nR, y_coordinates_A_1_nR[neuron], color = 'red', linestyle='dashed')   
                plt.plot(x_coordinates_A_2_nR, y_coordinates_A_2_nR[neuron], color = 'blue', linestyle='dashed')      
                plt.plot(x_coordinates_A_3_nR, y_coordinates_A_3_nR[neuron], color = 'green',linestyle='dashed')      
                plt.plot(x_coordinates_B_1_nR, y_coordinates_B_1_nR[neuron], color = 'red',linestyle='dashed')    
                plt.plot(x_coordinates_B_2_nR, y_coordinates_B_2_nR[neuron], color = 'blue',linestyle='dashed' )       
                plt.plot(x_coordinates_B_3_nR, y_coordinates_B_3_nR[neuron], color = 'green', linestyle='dashed')   
                plt.plot(x_coordinates_I_1_nR, y_coordinates_I_1_nR[neuron], color = 'red',linestyle='dashed')   
                plt.plot(x_coordinates_I_2_nR, y_coordinates_I_2_nR[neuron], color = 'blue',  linestyle='dashed')      
                plt.plot(x_coordinates_I_3_nR, y_coordinates_I_3_nR[neuron], color = 'green',  linestyle='dashed')    
                
                                             
                
                plt.axvline(ind_init_poke_1,ymin=0.85, ymax=0.9,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_1,ymin=0.85, ymax=0.9,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_1, ymin=0.85, ymax=0.9,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                plt.axvline(ind_init_poke_2,ymin=0.7, ymax=0.8,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_2,ymin=0.7, ymax=0.8,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_2, ymin=0.7, ymax=0.8,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                
                plt.axvline(ind_init_poke_3,ymin=0.7, ymax=0.8,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_3,ymin=0.7, ymax=0.8,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_3, ymin=0.7, ymax=0.8,linestyle = '--', c = 'Pink', linewidth=0.5)
               
                plt.axvline(ind_init_poke_4,ymin=0.5, ymax=0.6,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_4,ymin=0.5, ymax=0.6,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_4, ymin=0.5, ymax=0.6,linestyle = '--', c = 'Pink', linewidth=0.5)
               
               
                plt.axvline(ind_init_poke_6,ymin=0.5, ymax=0.6,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_6,ymin=0.5, ymax=0.6,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_6, ymin=0.5, ymax=0.6,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                plt.axvline(ind_init_poke_7,ymin=0.3, ymax=0.4,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_7,ymin=0.3, ymax=0.4,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_7, ymin=0.3, ymax=0.4,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                plt.axvline(ind_init_poke_8,ymin=0.3, ymax=0.4,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_8,ymin=0.3, ymax=0.4,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_8, ymin=0.3, ymax=0.4,linestyle = '--', c = 'Pink', linewidth=0.5)
              
                
                plt.axvline(ind_init_poke_9,ymin=0.1, ymax=0.2,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_9,ymin=0.1, ymax=0.2,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_9, ymin=0.1, ymax=0.2,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                
                # Pokes 
                plt.scatter(x_points,y_points,s =100, c = 'black')
                plt.axis('off')
                
                # Heatmap  
                plt.subplot2grid((2,2), (1,1))
                heatplot = aligned_spikes[:,neuron,:]
                normalised = (heatplot - np.min(heatplot,1)[:, None]) / (np.max(heatplot,1)[:, None]+1e-08 - np.min(heatplot,1)[:, None])
                heatplot_con = np.concatenate([normalised,task_arrays], axis = 1)
    
                plt.imshow(heatplot_con,aspect = 'auto')
                plt.xticks([ind_init, ind_choice, ind_reward], ('I', 'C', 'O'))
                plt.title('{}'.format(ID))
                
                pdf.savefig()
                plt.clf()
    pdf.close()
            
def plotting_no_hist(experiment,experiment_aligned):
    
    bin_width_ms = 1000
    smooth_sd_ms = 4000
    pdf = PdfPages('/Users/veronikasamborska/Desktop/HP_spikes.pdf')

    for s,session in zip(experiment, experiment_aligned):     
        # Get raw spike data across the task 
        raw_spikes = s.ephys
        neurons = np.unique(raw_spikes[0])
        # Get trial times
        pyControl_choice = [event.time for event in s.events if event.name in ['choice_state']]
        if len(raw_spikes[0]) > 0:
            spikes_list,session_duration_ms =  session_spikes_vs_trials_plot(raw_spikes,pyControl_choice)
            aligned_spikes= session.aligned_rates
            n_neurons = aligned_spikes.shape[1]
            n_trials = aligned_spikes.shape[0]
            t_out = session.t_out
            initiate_choice_t = session.target_times 
            initiate_t = initiate_choice_t[1]
            choice_t = initiate_choice_t[2]
            reward_t = initiate_choice_t[-2] +250
            ind_init = np.abs(t_out-initiate_t).argmin()
            ind_choice = np.abs(t_out-choice_t).argmin()
            ind_reward = np.abs(t_out-reward_t).argmin()
            
            ind_init_poke_1 = ind_init+300
            ind_choice_poke_1 = ind_choice+300
            ind_reward_poke_1 = ind_reward +300
            
            ind_init_poke_2 = ind_init+200
            ind_choice_poke_2 = ind_choice+200
            ind_reward_poke_2 = ind_reward +200
            
            ind_init_poke_3 = ind_init+400
            ind_choice_poke_3 = ind_choice+400
            ind_reward_poke_3 = ind_reward +400
            
                        
            ind_init_poke_4 = ind_init+100
            ind_choice_poke_4 = ind_choice+100
            ind_reward_poke_4 = ind_reward +100
            
            
            ind_init_poke_6 = ind_init+500
            ind_choice_poke_6 = ind_choice+500
            ind_reward_poke_6 = ind_reward +500
            
            ind_init_poke_7 = ind_init+200
            ind_choice_poke_7 = ind_choice+200
            ind_reward_poke_7 = ind_reward +200
            
            ind_init_poke_8 = ind_init+400
            ind_choice_poke_8 = ind_choice+400
            ind_reward_poke_8 = ind_reward +400
            
            
            ind_init_poke_9 = ind_init+300
            ind_choice_poke_9 = ind_choice+300
            ind_reward_poke_9 = ind_reward +300
            
            plt.ioff()
            
            x_coordinates_A_1_R, x_coordinates_A_2_R, x_coordinates_A_3_R,y_coordinates_A_1_R,\
            y_coordinates_A_2_R, y_coordinates_A_3_R, x_coordinates_B_1_R, x_coordinates_B_2_R,\
            x_coordinates_B_3_R, y_coordinates_B_1_R, y_coordinates_B_2_R, y_coordinates_B_3_R, x_coordinates_I_1_R,\
            x_coordinates_I_2_R, x_coordinates_I_3_R, y_coordinates_I_1_R, y_coordinates_I_2_R, y_coordinates_I_3_R,\
            x_coordinates_A_1_nR, x_coordinates_A_2_nR, x_coordinates_A_3_nR,y_coordinates_A_1_nR,\
            y_coordinates_A_2_nR, y_coordinates_A_3_nR, x_coordinates_B_1_nR, x_coordinates_B_2_nR,\
            x_coordinates_B_3_nR, y_coordinates_B_1_nR, y_coordinates_B_2_nR, y_coordinates_B_3_nR, x_coordinates_I_1_nR,\
            x_coordinates_I_2_nR, x_coordinates_I_3_nR, y_coordinates_I_1_nR, y_coordinates_I_2_nR, y_coordinates_I_3_nR = coordinates_for_plots(session)
            
            x_points = [132,232,232,332,332,432,432,532]
            y_points = [2.8,3.8,1.8,4.8,0.8,3.8,1.8,2.8]
            task_2_start, task_3_start = task_indicies(session) 
            
            task_arrays = np.zeros(shape=(n_trials,3))
            task_arrays[:task_2_start,0] = 1
            task_arrays[task_2_start:task_3_start,1] = 1
            task_arrays[task_3_start:,2] = 1
            for neuron,ID in enumerate(neurons):
                bin_edges = np.arange(0,session_duration_ms, bin_width_ms)
                hist,edges = np.histogram(spikes_list[neuron], bins= bin_edges)# histogram per second
                normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)

                #Port Firing
                fig = plt.figure(figsize=(8, 15))
                grid = plt.GridSpec(2, 1, hspace=0.7, wspace=0.4)
                fig.add_subplot(grid[0]) 

                plt.plot(x_coordinates_A_1_R, y_coordinates_A_1_R[neuron], color = 'firebrick',label =  'A task 1')   
                plt.plot(x_coordinates_A_2_R, y_coordinates_A_2_R[neuron], color = 'cadetblue',label =  'A task 2')   
                plt.plot(x_coordinates_A_3_R, y_coordinates_A_3_R[neuron], color = 'olive',label =  'A task 3')      
                plt.plot(x_coordinates_B_1_R, y_coordinates_B_1_R[neuron], color = 'firebrick',label =  'B task 1')    
                plt.plot(x_coordinates_B_2_R, y_coordinates_B_2_R[neuron], color = 'cadetblue', label =  'B task 2')       
                plt.plot(x_coordinates_B_3_R, y_coordinates_B_3_R[neuron], color = 'olive', label =  'B task 3')   
                plt.plot(x_coordinates_I_1_R, y_coordinates_I_1_R[neuron], color = 'red', linestyle = ':', label =  'I task 1')   
                plt.plot(x_coordinates_I_2_R, y_coordinates_I_2_R[neuron], color = 'blue',  linestyle = ':',label =  'I task 2')      
                plt.plot(x_coordinates_I_3_R, y_coordinates_I_3_R[neuron], color = 'green',  linestyle = ':',label =  'I task 3')  
                
                plt.legend()

                plt.plot(x_coordinates_A_1_nR, y_coordinates_A_1_nR[neuron], color = 'firebrick', linestyle='dashed', alpha=0.5)   
                plt.plot(x_coordinates_A_2_nR, y_coordinates_A_2_nR[neuron], color = 'cadetblue', linestyle='dashed', alpha=0.5)      
                plt.plot(x_coordinates_A_3_nR, y_coordinates_A_3_nR[neuron], color = 'olive',linestyle='dashed', alpha=0.5)      
                plt.plot(x_coordinates_B_1_nR, y_coordinates_B_1_nR[neuron], color = 'firebrick',linestyle='dashed', alpha=0.5)    
                plt.plot(x_coordinates_B_2_nR, y_coordinates_B_2_nR[neuron], color = 'cadetblue',linestyle='dashed', alpha=0.5 )       
                plt.plot(x_coordinates_B_3_nR, y_coordinates_B_3_nR[neuron], color = 'olive', linestyle='dashed', alpha=0.5)   
                plt.plot(x_coordinates_I_1_nR, y_coordinates_I_1_nR[neuron], color = 'red',linestyle='dashed', alpha=0.5)   
                plt.plot(x_coordinates_I_2_nR, y_coordinates_I_2_nR[neuron], color = 'blue',  linestyle='dashed', alpha=0.5)      
                plt.plot(x_coordinates_I_3_nR, y_coordinates_I_3_nR[neuron], color = 'green',  linestyle='dashed',alpha=0.5 )    
                plt.legend()
                                           
                plt.axvline(ind_init_poke_1,ymin=0.8, ymax=1,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_1,ymin=0.8, ymax=1,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_1, ymin=0.8, ymax=1,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                plt.axvline(ind_init_poke_2,ymin=0.6, ymax=0.8,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_2,ymin=0.6, ymax=0.8,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_2, ymin=0.6, ymax=0.8,linestyle = '--', c = 'Pink', linewidth=0.5)

                
                plt.axvline(ind_init_poke_3,ymin=0.6, ymax=0.8,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_3,ymin=0.6, ymax=0.8,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_3, ymin=0.6, ymax=0.8,linestyle = '--', c = 'Pink', linewidth=0.5)
               
                plt.axvline(ind_init_poke_4,ymin=0.45, ymax=0.65,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_4,ymin=0.45, ymax=0.65,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_4, ymin=0.45, ymax=0.65,linestyle = '--', c = 'Pink', linewidth=0.5)
               
               
                plt.axvline(ind_init_poke_6,ymin=0.45, ymax=0.65,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_6,ymin=0.45, ymax=0.65,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_6, ymin=0.45, ymax=0.65,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                plt.axvline(ind_init_poke_7,ymin=0.3, ymax=0.5,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_7,ymin=0.3, ymax=0.5,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_7, ymin=0.3, ymax=0.5,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                plt.axvline(ind_init_poke_8,ymin=0.3, ymax=0.5,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_8,ymin=0.3, ymax=0.5,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_8, ymin=0.3, ymax=0.5,linestyle = '--', c = 'Pink', linewidth=0.5)
              
                
                plt.axvline(ind_init_poke_9,ymin=0.1, ymax=0.3,linestyle = '--' ,c = 'Grey', linewidth=0.5)
                plt.axvline(ind_choice_poke_9,ymin=0.1, ymax=0.3,linestyle = '--', c = 'Black', linewidth=0.5)
                plt.axvline(ind_reward_poke_9, ymin=0.1, ymax=0.3,linestyle = '--', c = 'Pink', linewidth=0.5)
                
                
                # Pokes 
                plt.scatter(x_points,y_points,s =300, c = 'black')
                plt.axis('off')
                
                # Heatmap  
                fig.add_subplot(grid[1]) 
                heatplot = aligned_spikes[:,neuron,:]
                normalised = (heatplot - np.min(heatplot,1)[:, None]) / (np.max(heatplot,1)[:, None]+1e-08 - np.min(heatplot,1)[:, None])
                heatplot_con = np.concatenate([normalised,task_arrays], axis = 1)
    
                plt.imshow(heatplot_con,aspect = 'auto')
                plt.xticks([ind_init, ind_choice, ind_reward], ('I', 'C', 'O'))
                #plt.title('{}'.format(ID))
                plt.title('{}'.format(session.file_name))

                pdf.savefig()
                plt.clf()
    pdf.close()
            
            