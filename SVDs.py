#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:14:22 2019

@author: veronikasamborska
"""

#SVDs script for finding SVDs based on A and B (rewarded and unrewarded) in trials split in halfs

import ephys_beh_import as ep
import heatmap_aligned as ha 
import regressions as re
import neuron_firing_all_pokes as ne
import copy 
import forced_trials_extract_data as ft 
import numpy as np

ephys_path = '/Users/veronikasamborska/Desktop/neurons'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

#HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
#experiment_aligned_HP = ha.all_sessions_aligment(HP)
#experiment_aligned_PFC = ha.all_sessions_aligment(PFC)

#HP_forced = ft.all_sessions_aligment_forced(HP)
#PFC_forced = ft.all_sessions_aligment_forced(PFC)


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
            predictor_B_Task_1 = predictor_B_Task_1[:task_1]
            reward_task_1 = reward[:task_1]
            aligned_rates_task_2 = aligned_rates[:task_1+task_2]
            predictor_A_Task_2 = predictor_A_Task_2[:task_1+task_2]
            predictor_B_Task_2 = predictor_B_Task_2[:task_1+task_2]
            reward_task_2 = reward[:task_1+task_2]
        elif poke_I == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1]
            predictor_A_Task_1 = predictor_A_Task_1[:task_1]
            predictor_B_Task_1 = predictor_B_Task_1[:task_1]
            reward_task_1 = reward[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
            predictor_A_Task_2 = predictor_A_Task_3[task_1+task_2:]
            predictor_B_Task_2 = predictor_B_Task_3[task_1+task_2:]
            reward_task_2 = reward[task_1+task_2:]
        elif poke_I_task_2 == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1+task_2]
            predictor_A_Task_1 = predictor_A_Task_2[:task_1+task_2]
            predictor_B_Task_1 = predictor_B_Task_2[:task_1+task_2]
            reward_task_1 = reward[:task_1+task_2]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
            predictor_A_Task_2 = predictor_A_Task_3[task_1+task_2:]
            predictor_B_Task_2 = predictor_B_Task_3[task_1+task_2:]
            reward_task_2 = reward[task_1+task_2:]
           
        aligned_rates_task_1_first_half = aligned_rates_task_1[:int(len(aligned_rates_task_1)/2)]
        aligned_rates_task_1_second_half = aligned_rates_task_1[int(len(aligned_rates_task_1)/2):]
        aligned_rates_task_2_first_half = aligned_rates_task_2[:int(len(aligned_rates_task_2)/2)]
        aligned_rates_task_2_second_half = aligned_rates_task_2[int(len(aligned_rates_task_2)/2):] 
        
        predictor_A_Task_1_first_half = predictor_A_Task_1[:int(len(aligned_rates_task_1)/2)]
        predictor_A_Task_1_second_half = predictor_A_Task_1[int(len(aligned_rates_task_1)/2):]

        predictor_A_Task_2_first_half = predictor_B_Task_2[:int(len(predictor_A_Task_2)/2)]
        predictor_A_Task_2_second_half = predictor_B_Task_2[int(len(predictor_A_Task_2)/2):]
        
        
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
        aligned_rates_task_1_second_half_B_reward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_first_half == 0) & (reward_Task_1_second_half == 1 ))]
        aligned_rates_task_1_second_half_B_Nreward = aligned_rates_task_1_second_half[np.where((predictor_A_Task_1_first_half == 0 ) & (reward_Task_1_second_half == 0 ))]
        
        aligned_rates_task_2_first_half_A_reward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half ==1) & (reward_Task_2_first_half == 1 ))]
        aligned_rates_task_2_first_half_A_Nreward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half ==1) & (reward_Task_2_first_half == 0 ))]
        aligned_rates_task_2_second_half_A_reward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half ==1) & (reward_Task_2_second_half == 1 ))]
        aligned_rates_task_2_second_half_A_Nreward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_second_half ==1) & (reward_Task_2_second_half == 0 ))]
        
        aligned_rates_task_2_first_half_B_reward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_first_half == 1 ))]
        aligned_rates_task_2_first_half_B_Nreward = aligned_rates_task_2_first_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_first_half == 0 ))]
        aligned_rates_task_2_second_half_B_reward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_first_half == 0) & (reward_Task_2_second_half == 1 ))]
        aligned_rates_task_2_second_half_B_Nreward = aligned_rates_task_2_second_half[np.where((predictor_A_Task_2_first_half == 0 ) & (reward_Task_2_second_half == 0 ))]

    else:
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
    
    
def svd_trial_selection(experiment, tasks_unchanged = True, just_a = False, just_b = False):
    #Finds means of rates on trials of interest
    all_clusters_task_1_first_half = []
    all_clusters_task_1_second_half = []
    all_clusters_task_2_first_half = []
    all_clusters_task_2_second_half = []
    all_clusters_task_3_first_half = []
    all_clusters_task_3_second_half = []
    
  
    for s,session in enumerate(experiment):
        if s != 15 and s !=31:
            
            #Empty lists to hold append mean firing rates to 
            cluster_list_task_1_first_half = []
            cluster_list_task_1_second_half = []
            cluster_list_task_2_first_half = []
            cluster_list_task_2_second_half = []
            cluster_list_task_3_first_half = []
            cluster_list_task_3_second_half = []
            
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
            aligned_rates_task_3_second_half_B_reward,aligned_rates_task_3_second_half_B_Nreward = extract_session_predictors_rates(session)
            
            
    
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
                
                
                if tasks_unchanged == True:
                    mean_firing_rate_task_3_first_half_A_reward  = np.mean(aligned_rates_task_3_first_half_A_reward[:,i,:],0)
                    mean_firing_rate_task_3_first_half_A_Nreward  = np.mean(aligned_rates_task_3_first_half_A_Nreward[:,i,:],0)
                
                    mean_firing_rate_task_3_second_half_A_reward  = np.mean(aligned_rates_task_3_second_half_A_reward[:,i,:],0)
                    mean_firing_rate_task_3_second_half_A_Nreward  = np.mean(aligned_rates_task_3_second_half_A_Nreward[:,i,:],0)
    
                    mean_firing_rate_task_3_first_half_B_reward  = np.mean(aligned_rates_task_3_first_half_B_reward[:,i,:],0)
                    mean_firing_rate_task_3_first_half_B_Nreward  = np.mean(aligned_rates_task_3_first_half_B_Nreward[:,i,:],0)
                   
                    mean_firing_rate_task_3_second_half_B_reward  = np.mean(aligned_rates_task_3_second_half_B_reward[:,i,:],0)
                    mean_firing_rate_task_3_second_half_B_Nreward  = np.mean(aligned_rates_task_3_second_half_B_Nreward[:,i,:],0)
                    
                if just_a == False and just_b == False: 
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
                    
                elif just_a == True and just_b== False :
                        
                    mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_A_reward, mean_firing_rate_task_1_first_half_A_Nreward), axis = 0)
                    
                    mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_A_reward, mean_firing_rate_task_1_second_half_A_Nreward), axis = 0)
                    
                    mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_A_reward, mean_firing_rate_task_2_first_half_A_Nreward), axis = 0)
                    
                    mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_A_reward, mean_firing_rate_task_2_second_half_A_Nreward), axis = 0)
                  
                elif just_a == False and just_b == True:
                   mean_firing_rate_task_1_first_half = np.concatenate((mean_firing_rate_task_1_first_half_B_reward, mean_firing_rate_task_1_first_half_B_Nreward), axis = 0)
                   mean_firing_rate_task_1_second_half = np.concatenate((mean_firing_rate_task_1_second_half_B_reward, mean_firing_rate_task_1_second_half_B_Nreward), axis = 0)
                    
                   mean_firing_rate_task_2_first_half = np.concatenate((mean_firing_rate_task_2_first_half_B_reward, mean_firing_rate_task_2_first_half_B_Nreward), axis = 0)
                    
                   mean_firing_rate_task_2_second_half = np.concatenate((mean_firing_rate_task_2_second_half_B_reward, mean_firing_rate_task_2_second_half_B_Nreward), axis = 0)
                  
                cluster_list_task_1_first_half.append(mean_firing_rate_task_1_first_half) 
                cluster_list_task_1_second_half.append(mean_firing_rate_task_1_second_half)
                cluster_list_task_2_first_half.append(mean_firing_rate_task_2_first_half) 
                cluster_list_task_2_second_half.append(mean_firing_rate_task_2_second_half)
                
            
                if tasks_unchanged == True: 
                    if just_a == False and just_b == False: 
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



def svd_plotting(experiment, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True):
    #Calculating SVDs for trials split by A and B, reward/no reward (no block information) 

    all_clusters_task_1_first_half, all_clusters_task_1_second_half,\
    all_clusters_task_2_first_half, all_clusters_task_2_second_half,\
    all_clusters_task_3_first_half,all_clusters_task_3_second_half = svd_trial_selection(experiment, just_a = plot_a, just_b = plot_b)
   
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
  
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = False)

    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = False)


    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)
    t_v = np.transpose(vh_t1_1)
    
    t_u_t_1_2 = np.transpose(u_t1_2)
    t_v_t_1_2 = np.transpose(vh_t1_2)
    
    t_u_t_2_1 = np.transpose(u_t2_1)
    t_v_t_2_1 = np.transpose(vh_t2_1)

    
    t_u_t_2_2 = np.transpose(u_t2_2)
    t_v_t_2_2 = np.transpose(vh_t2_2)
    
    t_u_t_3_1 = np.transpose(u_t3_1)
    t_v_t_3_1 = np.transpose(vh_t3_1)
    
    #Compare task 1 Second Half 
    s_task_1_2 = np.linalg.multi_dot([t_u, flattened_all_clusters_task_1_second_half, t_v])
    s_1_2 = s_task_1_2.diagonal()
    sum_c_task_1_2 = np.cumsum(s_1_2)/flattened_all_clusters_task_1_second_half.shape[0]
    
    #Compare task 2 First Half 
    s_task_2_1 = np.linalg.multi_dot([t_u, flattened_all_clusters_task_2_first_half, t_v])
    s_2_1 = s_task_2_1.diagonal()
    sum_c_task_2_1 = np.cumsum(s_2_1)/flattened_all_clusters_task_2_first_half.shape[0]
    
    #Compare task 2 Second Half
    s_task_2_2 = np.linalg.multi_dot([t_u, flattened_all_clusters_task_2_second_half, t_v])
    s_2_2 = s_task_2_2.diagonal()
    sum_c_task_2_2 = np.cumsum(s_2_2)/flattened_all_clusters_task_2_second_half.shape[0]
    
    #Compare task 2 First Half from task 1 Last Half 
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half, t_v_t_1_2])
    s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    sum_c_task_2_1_from_t_1_2 = np.cumsum(s_2_1_from_t_1_2)/flattened_all_clusters_task_2_first_half.shape[0]
    
    
    #Compare task 2 Second Half from first half
    s_task_2_2_from_t_2_1 = np.linalg.multi_dot([t_u_t_2_1, flattened_all_clusters_task_2_second_half, t_v_t_2_1])
    s_2_2_from_t_2_1 = s_task_2_2_from_t_2_1.diagonal()
    sum_c_task_2_2_from_t_2_1 = np.cumsum(s_2_2_from_t_2_1)/flattened_all_clusters_task_2_second_half.shape[0]
    
    
    if tasks_unchanged == True: 
        #Compare task 3 First Half from Task 1
        s_task_3_1 = np.linalg.multi_dot([t_u, flattened_all_clusters_task_3_first_half, t_v])
        s_3_1 = s_task_3_1.diagonal()
        sum_c_task_3_1 = np.cumsum(s_3_1)/flattened_all_clusters_task_3_first_half.shape[0]
        
        #Compare task 3 Second Half from Task 1
        s_task_3_2 = np.linalg.multi_dot([t_u, flattened_all_clusters_task_3_second_half, t_v])
        s_3_2 = s_task_3_2.diagonal()
        sum_c_task_3_2 = np.cumsum(s_3_2)/flattened_all_clusters_task_3_second_half.shape[0]
        
        #Compare task 3 First Half from Task 2 Last Half 
        s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_3_first_half, t_v_t_2_2])
        s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
        sum_c_task_3_1_from_t_2_2 = np.cumsum(s_3_1_from_t_2_2)/flattened_all_clusters_task_2_first_half.shape[0]
    
        #Compare task 3 First Half from Task 3 first Half 
        s_task_3_2_from_t_3_1 = np.linalg.multi_dot([t_u_t_3_1, flattened_all_clusters_task_3_second_half, t_v_t_3_1])
        s_3_2_from_t_3_1 = s_task_3_2_from_t_3_1.diagonal()
        sum_c_task_3_2_from_t_3_1 = np.cumsum(s_3_2_from_t_3_1)/flattened_all_clusters_task_3_second_half.shape[0]
    
        
        average_3_tasks = np.mean([sum_c_task_2_1, sum_c_task_2_2, sum_c_task_3_1, sum_c_task_3_2], axis = 0)
        
        average_within = np.mean([sum_c_task_1_2,sum_c_task_2_2_from_t_2_1 ,sum_c_task_3_2_from_t_3_1], axis = 0)
        std_within = np.std([sum_c_task_1_2,sum_c_task_2_2_from_t_2_1 ,sum_c_task_3_2_from_t_3_1], axis = 0)
        x_within = np.arange(len(average_within))
        average_between = np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_2_2], axis = 0)
        
        std_between = np.std([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_2_2], axis = 0)
        x_between = np.arange(len(average_between))

    if HP == True:
        plot(average_within, label = 'Within Tasks_HP', linestyle = '--', color='red')
        fill_between(x_within,average_within+std_within,average_within-std_within, color = 'red', alpha = 0.2)
        plot(average_between, label = 'Between Tasks_HP', color = 'red')
        fill_between(x_between,average_between+std_between, average_between-std_between, color = 'red', alpha = 0.2)

    if HP == False:
        plot(average_within, label = 'Within Tasks_PFC', linestyle = '--', color='blue')
        fill_between(x_within,average_within+std_within, average_within-std_within, color = 'blue', alpha = 0.2)
        plot(average_between, label = 'Between Tasks_PFC', color = 'blue')
        fill_between(x_between,average_between+std_between, average_between-std_between, color = 'blue', alpha = 0.2)

    legend()