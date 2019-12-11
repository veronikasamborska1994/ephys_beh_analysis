#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:14:22 2019

@author: veronikasamborska
"""

#SVDs script for finding SVDs based on A and B (rewarded and unrewarded) in trials split in halfs

import ephys_beh_import as ep
import regressions as re
import numpy as np
import matplotlib.pyplot as plt
import heatmap_aligned as ha
import forced_trials_extract_data as ft
import svds_u_only as svdu



def extract_session_predictors_rates(session, tasks_unchanged = True):
    #Extracts firing rates for A and B rewarded and unrewarded trials with each task split into first and second half
    spikes = session.ephys
    spikes = spikes[:,~np.isnan(spikes[1,:])] 
    aligned_rates = session.aligned_rates

    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = ep.extract_choice_pokes(session)
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3 = ep.initiation_and_trial_end_timestamps(session)
    task_1 = len(trial_сhoice_state_task_1)
    task_2 = len(trial_сhoice_state_task_2)
    
    # Getting choice indices 
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
    if aligned_rates.shape[0] != predictor_A_Task_1.shape[0]:
         predictor_A_Task_1 = predictor_A_Task_1[:aligned_rates.shape[0]] 
         predictor_A_Task_2 = predictor_A_Task_2[:aligned_rates.shape[0]] 
         predictor_A_Task_3 = predictor_A_Task_3[:aligned_rates.shape[0]] 
         predictor_B_Task_1 = predictor_B_Task_1[:aligned_rates.shape[0]] 
         predictor_B_Task_2 = predictor_B_Task_2[:aligned_rates.shape[0]] 
         predictor_B_Task_3 = predictor_B_Task_3[:aligned_rates.shape[0]] 
         reward = reward[:aligned_rates.shape[0]] 
         
   #Find the tasks that have a shared I poke
    if tasks_unchanged == False:
        if poke_I == poke_I_task_2: 
            aligned_rates_task_1 = aligned_rates[:task_1]
            predictor_A_Task_1 = predictor_A_Task_1[:task_1]
            reward_task_1 = reward[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1:task_1+task_2]
            predictor_A_Task_2 = predictor_A_Task_2[task_1:task_1+task_2]
            reward_task_2 = reward[task_1:task_1+task_2]
            
        elif poke_I == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1]
            predictor_A_Task_1 = predictor_A_Task_1[:task_1]
            reward_task_1 = reward[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
            predictor_A_Task_2 = predictor_A_Task_3[task_1+task_2:]
            reward_task_2 = reward[task_1+task_2:]
            
        elif poke_I_task_2 == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[task_1:task_1+task_2]
            predictor_A_Task_1 = predictor_A_Task_2[task_1:task_1+task_2]
            reward_task_1 = reward[task_1:task_1+task_2]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
            predictor_A_Task_2 = predictor_A_Task_3[task_1+task_2:]
            reward_task_2 = reward[task_1+task_2:]
           
        aligned_rates_task_1_first_half = aligned_rates_task_1[:int(len(aligned_rates_task_1)/2)]
        aligned_rates_task_1_second_half = aligned_rates_task_1[int(len(aligned_rates_task_1)/2):]
        aligned_rates_task_2_first_half = aligned_rates_task_2[:int(len(aligned_rates_task_2)/2)]
        aligned_rates_task_2_second_half = aligned_rates_task_2[int(len(aligned_rates_task_2)/2):] 
        
        predictor_A_Task_1_first_half = predictor_A_Task_1[:int(len(aligned_rates_task_1)/2)]
        predictor_A_Task_1_second_half = predictor_A_Task_1[int(len(aligned_rates_task_1)/2):]
        predictor_A_Task_2_first_half = predictor_A_Task_2[:int(len(predictor_A_Task_2)/2)]
        predictor_A_Task_2_second_half = predictor_A_Task_2[int(len(predictor_A_Task_2)/2):]
        
        
        reward_Task_1_first_half = reward_task_1[:int(len(reward_task_1)/2)]
        reward_Task_1_second_half = reward_task_1[int(len(reward_task_1)/2):]
        reward_Task_2_first_half = reward_task_2[:int(len(reward_task_2)/2)]
        reward_Task_2_second_half = reward_task_2[int(len(reward_task_2)/2):]
        
        #Indexing A rewarded, B rewarded firing rates in 3 tasks
        aligned_rates_task_1_first_half_A_reward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half ==1) & (reward_Task_1_first_half == 1 ))]
        aligned_rates_task_1_first_half_A_Nreward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half ==1) & (reward_Task_1_first_half == 0 ))]
        aligned_rates_task_1_second_half_A_reward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half ==1) & (reward_Task_1_second_half == 1 ))]
        aligned_rates_task_1_second_half_A_Nreward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half ==1) & (reward_Task_1_second_half == 0 ))]
        
        aligned_rates_task_1_first_half_B_reward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half == 0) & (reward_Task_1_first_half == 1 ))]
        aligned_rates_task_1_first_half_B_Nreward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half == 0) & (reward_Task_1_first_half == 0 ))]
        aligned_rates_task_1_second_half_B_reward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half == 0) & (reward_Task_1_second_half == 1 ))]
        aligned_rates_task_1_second_half_B_Nreward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half == 0 ) & (reward_Task_1_second_half == 0 ))]
        
        aligned_rates_task_2_first_half_A_reward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half ==1) & (reward_Task_2_first_half == 1 ))]
        aligned_rates_task_2_first_half_A_Nreward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half ==1) & (reward_Task_2_first_half == 0 ))]
        aligned_rates_task_2_second_half_A_reward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half ==1) & (reward_Task_2_second_half == 1 ))]
        aligned_rates_task_2_second_half_A_Nreward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half ==1) & (reward_Task_2_second_half == 0 ))]
        
        aligned_rates_task_2_first_half_B_reward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_first_half == 1 ))]
        aligned_rates_task_2_first_half_B_Nreward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_first_half == 0 ))]
        aligned_rates_task_2_second_half_B_reward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half == 0) & (reward_Task_2_second_half == 1 ))]
        aligned_rates_task_2_second_half_B_Nreward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half == 0 ) & (reward_Task_2_second_half == 0 ))]
        
        
    elif tasks_unchanged == True:
          aligned_rates_task_1 = aligned_rates[:task_1]
          aligned_rates_task_2 = aligned_rates[task_1:task_1+task_2]
          aligned_rates_task_3 = aligned_rates[task_1+task_2:]
          
          predictor_A_Task_1_cut = predictor_A_Task_1[:task_1]
          reward_task_1_cut = reward[:task_1]
          
          predictor_A_Task_2_cut = predictor_A_Task_2[task_1:task_1+task_2]
          reward_task_2_cut = reward[task_1:task_1+task_2]
          
          predictor_A_Task_3_cut = predictor_A_Task_3[task_1+task_2:]
          reward_task_3_cut = reward[task_1+task_2:]
          
          
          #Split in Halfs Task 1 
          aligned_rates_task_1_first_half = aligned_rates_task_1[:int(len(aligned_rates_task_1)/2)]
          aligned_rates_task_1_second_half = aligned_rates_task_1[int(len(aligned_rates_task_1)/2):]
          predictor_A_Task_1_first_half = predictor_A_Task_1_cut[:int(len(aligned_rates_task_1)/2)]
          predictor_A_Task_1_second_half = predictor_A_Task_1_cut[int(len(aligned_rates_task_1)/2):]
          reward_Task_1_first_half = reward_task_1_cut[:int(len(aligned_rates_task_1)/2)]
          reward_Task_1_second_half = reward_task_1_cut[int(len(aligned_rates_task_1)/2):]
        
          #Split in Halfs Task 2
          aligned_rates_task_2_first_half = aligned_rates_task_2[:int(len(aligned_rates_task_2)/2)]
          aligned_rates_task_2_second_half = aligned_rates_task_2[int(len(aligned_rates_task_2)/2):] 
          predictor_A_Task_2_first_half = predictor_A_Task_2_cut[:int(len(aligned_rates_task_2)/2)]
          predictor_A_Task_2_second_half = predictor_A_Task_2_cut[int(len(aligned_rates_task_2)/2):]
          reward_Task_2_first_half = reward_task_2_cut[:int(len(aligned_rates_task_2)/2)]
          reward_Task_2_second_half = reward_task_2_cut[int(len(aligned_rates_task_2)/2):]
         
          #Split in Halfs Task 3
          aligned_rates_task_3_first_half = aligned_rates_task_3[:int(len(aligned_rates_task_3)/2)]
          aligned_rates_task_3_second_half = aligned_rates_task_3[int(len(aligned_rates_task_3)/2):]  
          predictor_A_Task_3_first_half = predictor_A_Task_3_cut[:int(len(aligned_rates_task_3)/2)]
          predictor_A_Task_3_second_half = predictor_A_Task_3_cut[int(len(aligned_rates_task_3)/2):]
          reward_Task_3_first_half = reward_task_3_cut[:int(len(aligned_rates_task_3)/2)]
          reward_Task_3_second_half = reward_task_3_cut[int(len(aligned_rates_task_3)/2):]
          
         
          aligned_rates_task_1_first_half_A_reward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half ==1) & (reward_Task_1_first_half == 1 ))]
          aligned_rates_task_1_first_half_A_Nreward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half ==1) & (reward_Task_1_first_half == 0 ))]
          aligned_rates_task_1_second_half_A_reward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half ==1) & (reward_Task_1_second_half == 1 ))]
          aligned_rates_task_1_second_half_A_Nreward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half ==1) & (reward_Task_1_second_half == 0 ))]     

          aligned_rates_task_1_first_half_B_reward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half == 0) & (reward_Task_1_first_half == 1 ))]
          aligned_rates_task_1_first_half_B_Nreward = aligned_rates_task_1_first_half[np.where((predictor_A_Task_1_first_half == 0) & (reward_Task_1_first_half == 0 ))]
          aligned_rates_task_1_second_half_B_reward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half == 0) & (reward_Task_1_second_half == 1 ))]
          aligned_rates_task_1_second_half_B_Nreward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_second_half == 0 ) & (reward_Task_1_second_half == 0 ))]
    
          aligned_rates_task_2_first_half_A_reward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 1) & (reward_Task_2_first_half == 1 ))]
          aligned_rates_task_2_first_half_A_Nreward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half ==1) & (reward_Task_2_first_half == 0 ))]
          aligned_rates_task_2_second_half_A_reward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half ==1) & (reward_Task_2_second_half == 1 ))]
          aligned_rates_task_2_second_half_A_Nreward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half ==1) & (reward_Task_2_second_half == 0 ))]
          
          aligned_rates_task_2_first_half_B_reward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_first_half == 1 ))]
          aligned_rates_task_2_first_half_B_Nreward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_first_half == 0 ))]
          aligned_rates_task_2_second_half_B_reward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half == 0) & (reward_Task_2_second_half == 1 ))]
          aligned_rates_task_2_second_half_B_Nreward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half == 0 ) & (reward_Task_2_second_half == 0 ))]
         
    
          aligned_rates_task_3_first_half_A_reward = aligned_rates_task_3_first_half[np.where((predictor_A_Task_3_first_half ==1) & (reward_Task_3_first_half == 1 ))]
          aligned_rates_task_3_first_half_A_Nreward = aligned_rates_task_3_first_half[np.where((predictor_A_Task_3_first_half ==1) & (reward_Task_3_first_half == 0 ))]
          aligned_rates_task_3_second_half_A_reward = aligned_rates_task_3_second_half[np.where((predictor_A_Task_3_second_half ==1) & (reward_Task_3_second_half == 1 ))]
          aligned_rates_task_3_second_half_A_Nreward = aligned_rates_task_3_second_half[np.where((predictor_A_Task_3_second_half ==1) & (reward_Task_3_second_half == 0 ))]
         
          aligned_rates_task_3_first_half_B_reward = aligned_rates_task_3_first_half[np.where((predictor_A_Task_3_first_half == 0) & (reward_Task_3_first_half == 1 ))]
          aligned_rates_task_3_first_half_B_Nreward = aligned_rates_task_3_first_half[np.where((predictor_A_Task_3_first_half == 0) & (reward_Task_3_first_half == 0 ))]
          aligned_rates_task_3_second_half_B_reward = aligned_rates_task_3_second_half[np.where((predictor_A_Task_3_second_half == 0) & (reward_Task_3_second_half == 1 ))]
          aligned_rates_task_3_second_half_B_Nreward = aligned_rates_task_3_second_half[np.where((predictor_A_Task_3_second_half == 0 ) & (reward_Task_3_second_half == 0 ))]
    
    if tasks_unchanged == True:  
        return spikes, aligned_rates_task_1_first_half_A_reward, aligned_rates_task_1_first_half_A_Nreward,\
        aligned_rates_task_1_second_half_A_reward,aligned_rates_task_1_second_half_A_Nreward,\
        aligned_rates_task_1_first_half_B_reward,aligned_rates_task_1_first_half_B_Nreward,\
        aligned_rates_task_1_second_half_B_reward,aligned_rates_task_1_second_half_B_Nreward,\
        aligned_rates_task_2_first_half_A_reward,aligned_rates_task_2_first_half_A_Nreward,\
        aligned_rates_task_2_second_half_A_reward,aligned_rates_task_2_second_half_A_Nreward,\
        aligned_rates_task_2_first_half_B_reward,aligned_rates_task_2_first_half_B_Nreward,\
        aligned_rates_task_2_second_half_B_reward,aligned_rates_task_2_second_half_B_Nreward,\
        aligned_rates_task_3_first_half_A_reward,aligned_rates_task_3_first_half_A_Nreward,\
        aligned_rates_task_3_second_half_A_reward,aligned_rates_task_3_second_half_A_Nreward,\
        aligned_rates_task_3_first_half_B_reward,aligned_rates_task_3_first_half_B_Nreward,\
        aligned_rates_task_3_second_half_B_reward,aligned_rates_task_3_second_half_B_Nreward
    elif tasks_unchanged == False:  
        return spikes, aligned_rates_task_1_first_half_A_reward, aligned_rates_task_1_first_half_A_Nreward,\
        aligned_rates_task_1_second_half_A_reward,aligned_rates_task_1_second_half_A_Nreward,\
        aligned_rates_task_1_first_half_B_reward,aligned_rates_task_1_first_half_B_Nreward,\
        aligned_rates_task_1_second_half_B_reward,aligned_rates_task_1_second_half_B_Nreward,\
        aligned_rates_task_2_first_half_A_reward,aligned_rates_task_2_first_half_A_Nreward,\
        aligned_rates_task_2_second_half_A_reward,aligned_rates_task_2_second_half_A_Nreward,\
        aligned_rates_task_2_first_half_B_reward,aligned_rates_task_2_first_half_B_Nreward,\
        aligned_rates_task_2_second_half_B_reward,aligned_rates_task_2_second_half_B_Nreward
        
    
def svd_trial_selection(experiment, tasks_unchanged = True, just_a = False, just_b = False, average_reward = False):
    #Finds means of rates on trials of interest
    all_clusters_task_1_first_half = []
    all_clusters_task_1_second_half = []
    all_clusters_task_2_first_half = []
    all_clusters_task_2_second_half = []
    all_clusters_task_3_first_half = []
    all_clusters_task_3_second_half = []
    
  
    for s,session in enumerate(experiment):
        if s != 15 and s !=31 and s!= 29:
            
       #Empty lists to hold append mean firing rates to 
            cluster_list_task_1_first_half = []
            cluster_list_task_1_second_half = []
            cluster_list_task_2_first_half = []
            cluster_list_task_2_second_half = []
            cluster_list_task_3_first_half = []
            cluster_list_task_3_second_half = []

            if tasks_unchanged == True:  
                spikes, aligned_rates_task_1_first_half_A_reward, aligned_rates_task_1_first_half_A_Nreward,\
        aligned_rates_task_1_second_half_A_reward,aligned_rates_task_1_second_half_A_Nreward,\
        aligned_rates_task_1_first_half_B_reward,aligned_rates_task_1_first_half_B_Nreward,\
        aligned_rates_task_1_second_half_B_reward,aligned_rates_task_1_second_half_B_Nreward,\
        aligned_rates_task_2_first_half_A_reward,aligned_rates_task_2_first_half_A_Nreward,\
        aligned_rates_task_2_second_half_A_reward,aligned_rates_task_2_second_half_A_Nreward,\
        aligned_rates_task_2_first_half_B_reward,aligned_rates_task_2_first_half_B_Nreward,\
        aligned_rates_task_2_second_half_B_reward,aligned_rates_task_2_second_half_B_Nreward,\
        aligned_rates_task_3_first_half_A_reward,aligned_rates_task_3_first_half_A_Nreward,\
        aligned_rates_task_3_second_half_A_reward,aligned_rates_task_3_second_half_A_Nreward,\
        aligned_rates_task_3_first_half_B_reward,aligned_rates_task_3_first_half_B_Nreward,\
        aligned_rates_task_3_second_half_B_reward,aligned_rates_task_3_second_half_B_Nreward = extract_session_predictors_rates(session, tasks_unchanged = tasks_unchanged)
            
            elif tasks_unchanged == False:  
                spikes, aligned_rates_task_1_first_half_A_reward, aligned_rates_task_1_first_half_A_Nreward,\
        aligned_rates_task_1_second_half_A_reward,aligned_rates_task_1_second_half_A_Nreward,\
        aligned_rates_task_1_first_half_B_reward,aligned_rates_task_1_first_half_B_Nreward,\
        aligned_rates_task_1_second_half_B_reward,aligned_rates_task_1_second_half_B_Nreward,\
        aligned_rates_task_2_first_half_A_reward,aligned_rates_task_2_first_half_A_Nreward,\
        aligned_rates_task_2_second_half_A_reward,aligned_rates_task_2_second_half_A_Nreward,\
        aligned_rates_task_2_first_half_B_reward,aligned_rates_task_2_first_half_B_Nreward,\
        aligned_rates_task_2_second_half_B_reward,aligned_rates_task_2_second_half_B_Nreward = extract_session_predictors_rates(session, tasks_unchanged = tasks_unchanged)
        
            if  (len(aligned_rates_task_1_first_half_A_reward) > 0) & (len(aligned_rates_task_1_first_half_A_Nreward) > 0) & (len(aligned_rates_task_1_second_half_A_reward) > 0) &\
            (len(aligned_rates_task_1_second_half_A_Nreward) > 0) & (len(aligned_rates_task_1_first_half_B_reward) > 0) & (len(aligned_rates_task_1_first_half_B_Nreward) > 0) &\
            (len(aligned_rates_task_1_second_half_B_reward) > 0) & (len(aligned_rates_task_1_second_half_B_Nreward) > 0)&\
            (len(aligned_rates_task_2_first_half_A_reward) > 0) & (len(aligned_rates_task_2_first_half_A_Nreward) > 0) & (len(aligned_rates_task_2_second_half_A_reward) > 0) &\
            (len(aligned_rates_task_2_second_half_A_Nreward) > 0) & (len(aligned_rates_task_2_first_half_B_reward) > 0) & (len(aligned_rates_task_2_first_half_B_Nreward) > 0) &\
            (len(aligned_rates_task_2_second_half_B_reward) > 0) & (len(aligned_rates_task_2_second_half_B_Nreward) > 0):
  
  
        
                unique_neurons  = np.unique(spikes[0])   
                
                for i in range(len(unique_neurons)):
                    
                    mean_firing_rate_task_1_first_half_A_reward  = np.mean(aligned_rates_task_1_first_half_A_reward[:,i,:],0)
                    mean_firing_rate_task_1_first_half_A_Nreward  = np.mean(aligned_rates_task_1_first_half_A_Nreward[:,i,:],0)
                    mean_firing_rate_task_1_second_half_A_reward  = np.mean(aligned_rates_task_1_second_half_A_reward[:,i,:],0)
                    mean_firing_rate_task_1_second_half_A_Nreward  = np.mean(aligned_rates_task_1_second_half_A_Nreward[:,i,:],0)
                    
          
                    mean_firing_rate_task_2_first_half_A_reward  = np.mean(aligned_rates_task_2_first_half_A_reward[:,i,:],0)
                    mean_firing_rate_task_2_first_half_A_Nreward  = np.mean(aligned_rates_task_2_first_half_A_Nreward[:,i,:],0)
                    mean_firing_rate_task_2_second_half_A_reward  = np.mean(aligned_rates_task_2_second_half_A_reward[:,i,:],0)
                    mean_firing_rate_task_2_second_half_A_Nreward  = np.mean(aligned_rates_task_2_second_half_A_Nreward[:,i,:],0)
                    
                    mean_firing_rate_task_1_first_half_B_reward  = np.mean(aligned_rates_task_1_first_half_B_reward[:,i,:],0)
                    mean_firing_rate_task_1_first_half_B_Nreward  = np.mean(aligned_rates_task_1_first_half_B_Nreward[:,i,:],0)           
                    mean_firing_rate_task_1_second_half_B_reward  = np.mean(aligned_rates_task_1_second_half_B_reward[:,i,:],0)     
                    mean_firing_rate_task_1_second_half_B_Nreward  = np.mean(aligned_rates_task_1_second_half_B_Nreward[:,i,:],0)
            
                    mean_firing_rate_task_2_first_half_B_reward  = np.mean(aligned_rates_task_2_first_half_B_reward[:,i,:],0)
                    mean_firing_rate_task_2_first_half_B_Nreward  = np.mean(aligned_rates_task_2_first_half_B_Nreward[:,i,:],0)               
                    mean_firing_rate_task_2_second_half_B_reward  = np.mean(aligned_rates_task_2_second_half_B_reward[:,i,:],0)
                    mean_firing_rate_task_2_second_half_B_Nreward  = np.mean(aligned_rates_task_2_second_half_B_Nreward[:,i,:],0)
                    
                          
                    if average_reward == False and just_a == False and just_b == False: 
                        
                        mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward,\
                                                                             mean_firing_rate_task_1_first_half_B_reward,\
                                                                             mean_firing_rate_task_1_first_half_B_Nreward), axis = 0)
                        
                        mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward,\
                                                                             mean_firing_rate_task_1_second_half_B_reward,\
                                                                             mean_firing_rate_task_1_second_half_B_Nreward), axis = 0)
                        
                        mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward,\
                                                                             mean_firing_rate_task_2_first_half_B_reward,\
                                                                             mean_firing_rate_task_2_first_half_B_Nreward), axis = 0)
                        
                        mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward,\
                                                                             mean_firing_rate_task_2_second_half_B_reward,\
                                                                             mean_firing_rate_task_2_second_half_B_Nreward), axis = 0)
                        
                    elif just_a == True and just_b == False :
                            
                        mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward), axis = 0)
                        
                        mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward), axis = 0)
                        
                        mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward), axis = 0)
                        
                        mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward), axis = 0)
                      
                    elif just_a == False and just_b == True:
                       
                       mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward), axis = 0)
                       
                       mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward), axis = 0)
                        
                       mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward), axis = 0)
                        
                       mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward), axis = 0)
                    
                    elif average_reward == True:
                       
                       a_average_reward_task_1_first_half  =  np.mean([mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward],axis = 0)
                       
                       b_average_reward_task_1_first_half  =  np.mean([mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward],axis = 0)
    
                       a_average_reward_task_1_second_half  =  np.mean([mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward],axis = 0)
                       
                       b_average_reward_task_1_second_half  =  np.mean([mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward],axis = 0)
                       
                       a_average_reward_task_2_first_half  =  np.mean([mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward],axis = 0)
                       
                       b_average_reward_task_2_first_half  =  np.mean([mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward],axis = 0)
    
                       a_average_reward_task_2_second_half  =  np.mean([mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward],axis = 0)
                       
                       b_average_reward_task_2_second_half  =  np.mean([mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward],axis = 0)
    
    
    
                       mean_firing_rate_task_1_first_half = np.concatenate((a_average_reward_task_1_first_half, b_average_reward_task_1_first_half), axis = 0)
                       
                       mean_firing_rate_task_1_second_half = np.concatenate((a_average_reward_task_1_second_half, b_average_reward_task_1_second_half), axis = 0)
                        
                       mean_firing_rate_task_2_first_half = np.concatenate((a_average_reward_task_2_first_half, b_average_reward_task_2_first_half), axis = 0)
                        
                       mean_firing_rate_task_2_second_half = np.concatenate((a_average_reward_task_2_second_half, b_average_reward_task_2_second_half), axis = 0)
               
                    
                    cluster_list_task_1_first_half.append(mean_firing_rate_task_1_first_half) 
                    cluster_list_task_1_second_half.append(mean_firing_rate_task_1_second_half)
                    cluster_list_task_2_first_half.append(mean_firing_rate_task_2_first_half) 
                    cluster_list_task_2_second_half.append(mean_firing_rate_task_2_second_half)
                    
                
                    if tasks_unchanged == True: 
        
                        mean_firing_rate_task_3_first_half_A_reward  = np.mean(aligned_rates_task_3_first_half_A_reward[:,i,:],0)
                        mean_firing_rate_task_3_first_half_A_Nreward  = np.mean(aligned_rates_task_3_first_half_A_Nreward[:,i,:],0)
                    
                        mean_firing_rate_task_3_second_half_A_reward  = np.mean(aligned_rates_task_3_second_half_A_reward[:,i,:],0)
                        mean_firing_rate_task_3_second_half_A_Nreward  = np.mean(aligned_rates_task_3_second_half_A_Nreward[:,i,:],0)
        
                        mean_firing_rate_task_3_first_half_B_reward  = np.mean(aligned_rates_task_3_first_half_B_reward[:,i,:],0)
                        mean_firing_rate_task_3_first_half_B_Nreward  = np.mean(aligned_rates_task_3_first_half_B_Nreward[:,i,:],0)
                       
                        mean_firing_rate_task_3_second_half_B_reward  = np.mean(aligned_rates_task_3_second_half_B_reward[:,i,:],0)
                        mean_firing_rate_task_3_second_half_B_Nreward  = np.mean(aligned_rates_task_3_second_half_B_Nreward[:,i,:],0)
                  
                        if average_reward == False and just_a == False and just_b == False: 
                            
                            mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward,\
                                                                             mean_firing_rate_task_3_first_half_B_reward,\
                                                                             mean_firing_rate_task_3_first_half_B_Nreward), axis = 0)
                        
                            mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward,\
                                                                             mean_firing_rate_task_3_second_half_B_reward,\
                                                                             mean_firing_rate_task_3_second_half_B_Nreward), axis = 0)
                        elif just_a == True and just_b== False :
                            
                            mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward), axis = 0)                    
                            mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward), axis = 0)
                        
                        elif just_a == False and just_b == True:
                            
                            mean_firing_rate_task_3_first_half = np.concatenate((mean_firing_rate_task_3_first_half_B_reward, mean_firing_rate_task_3_first_half_B_Nreward), axis = 0)
                            mean_firing_rate_task_3_second_half = np.concatenate((mean_firing_rate_task_3_second_half_B_reward, mean_firing_rate_task_3_second_half_B_Nreward), axis = 0)
                            
                        elif average_reward == True:
                            
                            a_average_reward_task_3_first_half  =  np.mean([mean_firing_rate_task_3_first_half_A_reward, mean_firing_rate_task_3_first_half_A_Nreward],axis = 0)
                       
                            b_average_reward_task_3_first_half  =  np.mean([mean_firing_rate_task_3_first_half_B_reward, mean_firing_rate_task_3_first_half_B_Nreward],axis = 0)
    
                            a_average_reward_task_3_second_half  =  np.mean([mean_firing_rate_task_3_second_half_A_reward, mean_firing_rate_task_3_second_half_A_Nreward],axis = 0)
                           
                            b_average_reward_task_3_second_half  =  np.mean([mean_firing_rate_task_3_second_half_B_reward, mean_firing_rate_task_3_second_half_B_Nreward],axis = 0)
                            
                            mean_firing_rate_task_3_first_half = np.concatenate((a_average_reward_task_3_first_half, b_average_reward_task_3_first_half), axis = 0)
                        
                            mean_firing_rate_task_3_second_half = np.concatenate((a_average_reward_task_3_second_half, b_average_reward_task_3_second_half), axis = 0)
               
                        cluster_list_task_3_first_half.append(mean_firing_rate_task_3_first_half) 
                        cluster_list_task_3_second_half.append(mean_firing_rate_task_3_second_half)
                
            cluster_list_task_1_first_half = np.asarray(cluster_list_task_1_first_half)
            cluster_list_task_1_second_half = np.asarray(cluster_list_task_1_second_half)
            cluster_list_task_2_first_half = np.asarray(cluster_list_task_2_first_half)
            cluster_list_task_2_second_half = np.asarray(cluster_list_task_2_second_half)
            
            all_clusters_task_1_first_half.append(cluster_list_task_1_first_half[:])
            all_clusters_task_1_second_half.append(cluster_list_task_1_second_half[:])
            all_clusters_task_2_first_half.append(cluster_list_task_2_first_half[:])
            all_clusters_task_2_second_half.append(cluster_list_task_2_second_half[:])
            
            if tasks_unchanged == True: 
                
                cluster_list_task_3_first_half = np.asarray(cluster_list_task_3_first_half)
                cluster_list_task_3_second_half = np.asarray(cluster_list_task_3_second_half)
                all_clusters_task_3_first_half.append(cluster_list_task_3_first_half[:])
                all_clusters_task_3_second_half.append(cluster_list_task_3_second_half[:])
            
    return all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
        all_clusters_task_2_first_half, all_clusters_task_2_second_half,\
        all_clusters_task_3_first_half,all_clusters_task_3_second_half

def flatten(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False):
    
    all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
    all_clusters_task_2_first_half, all_clusters_task_2_second_half,\
    all_clusters_task_3_first_half,all_clusters_task_3_second_half = svd_trial_selection(experiment, tasks_unchanged = tasks_unchanged, just_a = plot_a, just_b = plot_b, average_reward = average_reward)
   
   
    all_clusters_task_1_first_half  = np.asarray(all_clusters_task_1_first_half)
    all_clusters_task_1_second_half = np.asarray(all_clusters_task_1_second_half)
    
    all_clusters_task_2_first_half = np.asarray(all_clusters_task_2_first_half)
    all_clusters_task_2_second_half = np.asarray(all_clusters_task_2_second_half)
    all_clusters_task_3_first_half = np.asarray(all_clusters_task_3_first_half)
    all_clusters_task_3_second_half = np.asarray(all_clusters_task_3_second_half)
    
    
    # Empty lists to get flattened arrays for SVD analysis 
    flattened_all_clusters_task_1_first_half = []
    flattened_all_clusters_task_1_second_half = []
    flattened_all_clusters_task_2_first_half = []
    flattened_all_clusters_task_2_second_half = []
    flattened_all_clusters_task_3_first_half = []
    flattened_all_clusters_task_3_second_half = []
    
    #Flatten lists 
    for x in all_clusters_task_1_first_half:
        for y in x:
            flattened_all_clusters_task_1_first_half.append(y)
                
    for x in all_clusters_task_1_second_half:
        for y in x:
            flattened_all_clusters_task_1_second_half.append(y)            
    
    for x in all_clusters_task_2_first_half:
        for y in x:
            flattened_all_clusters_task_2_first_half.append(y)          
    
    for x in all_clusters_task_2_second_half:
        for y in x:
            flattened_all_clusters_task_2_second_half.append(y)
            
    if tasks_unchanged == True:        
        for x in all_clusters_task_3_first_half:
            for y in x:
                flattened_all_clusters_task_3_first_half.append(y)          
        
        for x in all_clusters_task_3_second_half:
            for y in x:
                flattened_all_clusters_task_3_second_half.append(y)
            
    flattened_all_clusters_task_1_first_half = np.asarray(flattened_all_clusters_task_1_first_half)     
    flattened_all_clusters_task_1_second_half = np.asarray(flattened_all_clusters_task_1_second_half)  
    flattened_all_clusters_task_2_first_half = np.asarray(flattened_all_clusters_task_2_first_half)
    flattened_all_clusters_task_2_second_half = np.asarray(flattened_all_clusters_task_2_second_half)
    flattened_all_clusters_task_3_first_half = np.asarray(flattened_all_clusters_task_3_first_half)
    flattened_all_clusters_task_3_second_half = np.asarray(flattened_all_clusters_task_3_second_half)
    
    if tasks_unchanged == True: 
        return flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,\
        flattened_all_clusters_task_3_second_half     
    else:
        return flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half



def plotting_correlation_matrix(experiment):
    session = experiment_aligned_HP[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward = initiate_choice_t[-2] +250

    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward)).argmin()
    
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half, flattened_all_clusters_task_3_second_half\
    = svdu.demean_data(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)
    
 #   flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
 #   flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = svdu.demean_data(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)
     

    
    m_full = np.concatenate([flattened_all_clusters_task_1_first_half,flattened_all_clusters_task_1_second_half,\
                             flattened_all_clusters_task_2_first_half,flattened_all_clusters_task_2_second_half], axis =1)
        
    plt.figure(1)

    corrmf = np.corrcoef(np.transpose(m_full))

    plt.imshow(corrmf)
    ticks_n  = np.linspace(0, corrmf.shape[0],16)

    plt.yticks(ticks_n, ('A Reward T1 1', 'A No Reward T1 1','B Reward T1 1', 'B No Reward T1 1',\
                         'A Reward T1 2', 'A No Reward T1 2','B Reward T1 2', 'B No Reward T1 2',
                         'A Reward T2 1', 'A No Reward T2 1','B Reward T2 1', 'B No Reward T2 1',\
                         'A Reward T2 2', 'A No Reward T2 2','B Reward T2 2', 'B No Reward T2 2'))
    
    plt.xticks(ticks_n, ('A Reward T1 1', 'A No Reward T1 1','B Reward T1 1', 'B No Reward T1 1',\
                         'A Reward T1 2', 'A No Reward T1 2','B Reward T1 2', 'B No Reward T1 2'
                         'A Reward T2 1', 'A No Reward T2 1','B Reward T2 1', 'B No Reward T2 1',\
                         'A Reward T2 2', 'A No Reward T2 2','B Reward T2 2', 'B No Reward T2 2'), rotation = 'vertical')


    T1_T1 = corrmf[64*2:64*4,:64*2]
    T1_T2 = corrmf[64*10:64*14,:64]
    #a = corrmf[0:64,64:64*3]
    #plt.imshow(a)
    #plt.figure(2)
    #difference = T1_T1-T1_T2
    plt.imshow(T1_T2)
    plt.xticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ), rotation = 'vertical')  
    ticks_n  = np.linspace(0, T1_T2.shape[0],4)

    plt.yticks(ticks_n, ('B Reward T2', 'B No Reward T2',\
                         'A Reward T2', 'A No Reward T2'))
    
    #plt.yticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ))  

   
  
    fig = plt.figure(figsize=(8, 25))
    grid = plt.GridSpec(5, 4, hspace=0.7, wspace=0.4)

    example_reward = corrmf[:64,64*2:64*3]
    for i in range(example_reward.shape[1]):
        if i > ind_reward and i <60:
            fig.add_subplot(grid[i-ind_reward-1])
            plt.plot(example_reward[:,i], label = 'Reward --> Reward')
            plt.xticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ), rotation = 'vertical')  

    example_no_reward = corrmf[:64,64*3:64*4]
    for i in range(example_no_reward.shape[1]):
        if i > ind_reward and i <60:
            fig.add_subplot(grid[i-ind_reward-1])
            plt.plot(example_no_reward[:,i],label ='No Reward --> No Reward')
            plt.xticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ), rotation = 'vertical')  
            
    example_no_reward_to_reward = corrmf[64*3:64*4,64*2:64*3]
    for i in range(example_no_reward_to_reward.shape[1]):
        if i > ind_reward and i <60:
            fig.add_subplot(grid[i-ind_reward-1])
            plt.plot(example_no_reward_to_reward[:,i], label = 'No Reward --> Reward')
            plt.xticks([ind_init,ind_choice,ind_reward], label =('Initiation', 'Choice', 'Reward' ), rotation = 'vertical')  
    
    example_reward_no_reward = corrmf[:64,64:64*2]
    for i in range(example_reward_no_reward.shape[1]):
        if i > ind_reward and i <60:
            fig.add_subplot(grid[i-ind_reward-1])
            plt.plot(example_reward_no_reward[:,i], label = 'Reward --> No Reward')
            plt.xticks([ind_init,ind_choice,ind_reward], label =('Initiation', 'Choice', 'Reward' ), rotation = 'vertical')      
            plt.ylabel('Corr  Coef')
    
        
    plt.legend()
    fig.add_subplot(grid[16])
    plt.imshow(example_reward)
    plt.xlabel('Reward')
    plt.ylabel('Reward')
    
    fig.add_subplot(grid[17])
    plt.imshow(example_no_reward)
    plt.xlabel('No Reward')
    plt.ylabel('No Reward')
    
    fig.add_subplot(grid[18])
    plt.imshow(example_no_reward_to_reward)
    plt.ylabel('No Reward')
    plt.xlabel('Reward')
    
    fig.add_subplot(grid[19])
    plt.imshow(example_reward_no_reward)
    plt.ylabel('Reward')
    plt.xlabel('No Reward')
    

    plt.yticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ))  
    plt.xticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ))  



def svd_plotting(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = False, demean_all_tasks = True):
   
    #Calculating SVDs for trials split by A and B, reward/no reward (no block information) 
    if tasks_unchanged == True:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
    else:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = flatten(experiment, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward)
  
     #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = False)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)
    
    

    if tasks_unchanged == True:
        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = False)
    
        #SVDsu.shape, s.shape, vh.shape for task 3 first half
        u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = False)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)  
    t_v = np.transpose(vh_t1_1)  

    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_1 = np.transpose(u_t2_1)   
    t_v_t_2_1 = np.transpose(vh_t2_1)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  

    if tasks_unchanged == True:
        t_u_t_3_2 = np.transpose(u_t3_2)
        t_v_t_3_2 = np.transpose(vh_t3_2)  
    
    #Compare task 1 Second Half 
    s_task_1_2 = np.linalg.multi_dot([t_u, flattened_all_clusters_task_1_second_half, t_v])
    
    if diagonal == False:
        s_1_2 = s_task_1_2.diagonal()
    else:
        s_1_2 = np.sum(s_task_1_2**2, axis = 1)
     
    sum_c_task_1_2 = np.cumsum(abs(s_1_2))/flattened_all_clusters_task_1_second_half.shape[0]
    
   
    #Compare task 2 First Half from task 1 Last Half 
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half, t_v_t_1_2])
    if diagonal == False:
        s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    else:
        s_2_1_from_t_1_2 = np.sum(s_task_2_1_from_t_1_2**2, axis = 1)
    sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/flattened_all_clusters_task_2_first_half.shape[0]


    
    #Compare task 2 Second Half from first half
    s_task_2_2_from_t_2_1 = np.linalg.multi_dot([t_u_t_2_1, flattened_all_clusters_task_2_second_half, t_v_t_2_1])    
    if diagonal == False:
        s_2_2_from_t_2_1 = s_task_2_2_from_t_2_1.diagonal()
    else:
        s_2_2_from_t_2_1 = np.sum(s_task_2_2_from_t_2_1**2, axis = 1)
    sum_c_task_2_2_from_t_2_1 = np.cumsum(abs(s_2_2_from_t_2_1))/flattened_all_clusters_task_2_second_half.shape[0]
    
     #Compare task 2 Firs Half from second half
    s_task_2_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half, t_v_t_2_2])    
    if diagonal == False:
        s_2_1_from_t_2_2 = s_task_2_1_from_t_2_2.diagonal()
    else:
        s_2_1_from_t_2_2 = np.sum(s_task_2_1_from_t_2_2**2, axis = 1)
    sum_c_task_2_1_from_t_2_2 = np.cumsum(abs(s_2_1_from_t_2_2))/flattened_all_clusters_task_2_first_half.shape[0]


    
    if tasks_unchanged == True: 
        
        #Compare task 3 First Half from Task 2 Last Half 
        s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_3_first_half, t_v_t_2_2])
        if diagonal == False:
            s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
        else:
            s_3_1_from_t_2_2 = np.sum(s_task_3_1_from_t_2_2**2, axis = 1)
        sum_c_task_3_1_from_t_2_2 = np.cumsum(abs(s_3_1_from_t_2_2))/flattened_all_clusters_task_3_first_half.shape[0]


        s_task_3_1_from_t_3_2 = np.linalg.multi_dot([t_u_t_3_2, flattened_all_clusters_task_3_first_half, t_v_t_3_2])
        if diagonal == False:
            s_3_1_from_t_3_2 = s_task_3_1_from_t_3_2.diagonal()
        else:
            s_3_1_from_t_3_2 = np.sum(s_task_3_1_from_t_3_2**2, axis = 1)
        sum_c_task_3_1_from_t_3_2 = np.cumsum(abs(s_3_1_from_t_3_2))/flattened_all_clusters_task_3_first_half.shape[0]

        average_within = np.mean([sum_c_task_2_1_from_t_2_2, sum_c_task_3_1_from_t_3_2], axis = 0)
        average_between = np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_2_2], axis = 0)

       
    else:
        average_within = np.mean([sum_c_task_1_2,sum_c_task_2_2_from_t_2_1], axis = 0)
        average_between = np.mean([sum_c_task_2_1_from_t_1_2], axis = 0)
        
    if diagonal == True:
        average_within = average_within/average_within[-1]
        average_between = average_between/average_between[-1]
    if HP == True:
        plt.figure(10)
        if demean_all_tasks == True: #Demean Across all taks (False - Within Each Task)
            plt.plot(average_within, label = 'Within Tasks_HP', color='green')
            plt.plot(average_between, label = 'Between Tasks_HP',linestyle = '--', color = 'green')
        else:
            plt.plot(average_within, label = 'Within Tasks_HP', color='green')
            plt.plot(average_between, label = 'Between Tasks_HP',linestyle = '--', color = 'green')
        
    if HP == False:
        plt.figure(10)
        if demean_all_tasks == True: #Demean Across all taks (False - Within Each Task)
            plt.plot(average_within, label = 'Within Tasks_PFC', color='black')
            plt.plot(average_between, label = 'Between Tasks_PFC',linestyle = '--', color = 'black')
        else:
            plt.plot(average_within, label = 'Within Tasks_PFC', color='black')
            plt.plot(average_between, label = 'Between Tasks_PFC',linestyle = '--', color = 'black')
      
    plt.figure(10)
    plt.title('Full SVD')
    plt.legend()
    
    return average_within,average_between
    
