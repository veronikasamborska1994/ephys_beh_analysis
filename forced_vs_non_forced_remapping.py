#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:08:19 2019

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt
import svd_block_analysis as svdall
import svd_forced_trials as svdf



def extract_trial_firing_rate_non_forced(experiment):
## Function for extracting mean firing rates for non forced A and B choices in A and B blocks
## Takes HP_forced or PFC_forced experiments and finds 
    all_clusters_task_1_forced = []
    all_clusters_task_2_forced = []
    all_clusters_task_3_forced = []
    
    all_clusters_task_1_non_forced = []
    all_clusters_task_2_non_forced = []
    all_clusters_task_3_non_forced = []
    
    for session in experiment:
        # Extract firing rates for choices based on the state for non-forced trials 
        state_A_choice_A_t1,state_A_choice_B_t1,state_B_choice_A_t1,state_B_choice_B_t1,\
        state_A_choice_A_t2, state_A_choice_B_t2,state_B_choice_A_t2,state_B_choice_B_t2,\
        state_A_choice_A_t3, state_A_choice_B_t3, state_B_choice_A_t3, state_B_choice_B_t3, spikes = svdall.extract_session_a_b_based_on_block(session, tasks_unchanged = True)
        
        # Extract firing rates for choices based on the state for non-forced trials 
        state_A_choice_A_t1_f,state_A_choice_B_t1_f,state_B_choice_A_t1_f,state_B_choice_B_t1_f,\
        state_A_choice_A_t2_f, state_A_choice_B_t2_f,state_B_choice_A_t2_f,state_B_choice_B_t2_f,\
        state_A_choice_A_t3_f, state_A_choice_B_t3_f, state_B_choice_A_t3_f, state_B_choice_B_t3_f, spikes_f = svdf.extract_session_a_b_based_on_block_forced_trials(session, tasks_unchanged = True)
        
        
        if (len(state_A_choice_A_t1) > 0) & (len(state_A_choice_B_t1) > 0) & (len(state_B_choice_A_t1) > 0) &\
                (len(state_B_choice_B_t1) > 0) & (len(state_A_choice_A_t2) > 0) & (len(state_A_choice_B_t2) > 0) &\
                (len(state_B_choice_A_t2) > 0) & (len(state_B_choice_B_t2) > 0) & (len(state_A_choice_A_t3) > 0) &\
                (len(state_A_choice_B_t3) > 0) & (len(state_B_choice_A_t3) > 0) & (len(state_B_choice_B_t3) > 0) &\
                (len(state_A_choice_A_t1_f) > 0) & (len(state_A_choice_B_t1_f) > 0) & (len(state_B_choice_A_t1_f) > 0) &\
                (len(state_B_choice_B_t1_f) > 0) & (len(state_A_choice_A_t2_f) > 0) & (len(state_A_choice_B_t2_f) > 0) &\
                (len(state_B_choice_A_t2_f) > 0) & (len(state_B_choice_B_t2_f) > 0) & (len(state_A_choice_A_t3_f) > 0) &\
                (len(state_A_choice_B_t3_f) > 0) & (len(state_B_choice_A_t3_f) > 0) & (len(state_B_choice_B_t3_f) > 0):
                
                unique_neurons  = np.unique(spikes[0])   
                for i in range(len(unique_neurons)):                
                    mean_firing_rate_task_1_a_good_A_ch  = np.mean(state_A_choice_A_t1[:,i,:],0)
                    mean_firing_rate_task_1_a_good_B_ch  = np.mean(state_A_choice_B_t1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_B_ch  = np.mean(state_B_choice_B_t1[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch  = np.mean(state_B_choice_A_t1[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch  = np.mean(state_A_choice_A_t2[:,i,:],0)
                    mean_firing_rate_task_2_a_good_B_ch  = np.mean(state_A_choice_B_t2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_B_ch  = np.mean(state_B_choice_B_t2[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch  = np.mean(state_B_choice_A_t2[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch  = np.mean(state_A_choice_A_t3[:,i,:],0)
                    mean_firing_rate_task_3_a_good_B_ch  = np.mean(state_A_choice_B_t3[:,i,:],0)
                    mean_firing_rate_task_3_b_good_B_ch  = np.mean(state_B_choice_B_t3[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch  = np.mean(state_B_choice_A_t3[:,i,:],0)
                    
                    mean_firing_rate_task_1_a_good_A_ch_f  = np.mean(state_A_choice_A_t1_f[:,i,:],0)
                    mean_firing_rate_task_1_a_good_B_ch_f  = np.mean(state_A_choice_B_t1_f[:,i,:],0)
                    mean_firing_rate_task_1_b_good_B_ch_f = np.mean(state_B_choice_B_t1_f[:,i,:],0)
                    mean_firing_rate_task_1_b_good_A_ch_f  = np.mean(state_B_choice_A_t1_f[:,i,:],0)
                    
                    mean_firing_rate_task_2_a_good_A_ch_f  = np.mean(state_A_choice_A_t2_f[:,i,:],0)
                    mean_firing_rate_task_2_a_good_B_ch_f  = np.mean(state_A_choice_B_t2_f[:,i,:],0)
                    mean_firing_rate_task_2_b_good_B_ch_f  = np.mean(state_B_choice_B_t2_f[:,i,:],0)
                    mean_firing_rate_task_2_b_good_A_ch_f  = np.mean(state_B_choice_A_t2_f[:,i,:],0)
                    
                    mean_firing_rate_task_3_a_good_A_ch_f  = np.mean(state_A_choice_A_t3_f[:,i,:],0)
                    mean_firing_rate_task_3_a_good_B_ch_f  = np.mean(state_A_choice_B_t3_f[:,i,:],0)
                    mean_firing_rate_task_3_b_good_B_ch_f  = np.mean(state_B_choice_B_t3_f[:,i,:],0)
                    mean_firing_rate_task_3_b_good_A_ch_f  = np.mean(state_B_choice_A_t3_f[:,i,:],0)
                    
                    
                    clusters_task_1_non_forced = np.concatenate([mean_firing_rate_task_1_a_good_A_ch,mean_firing_rate_task_1_b_good_A_ch,\
                                                                 mean_firing_rate_task_1_a_good_B_ch, mean_firing_rate_task_1_b_good_B_ch], axis = 0)
                    clusters_task_2_non_forced = np.concatenate([mean_firing_rate_task_2_a_good_A_ch,mean_firing_rate_task_2_b_good_A_ch,\
                                                                 mean_firing_rate_task_2_a_good_B_ch, mean_firing_rate_task_2_b_good_B_ch], axis = 0)
                    clusters_task_3_non_forced = np.concatenate([mean_firing_rate_task_3_a_good_A_ch,mean_firing_rate_task_3_b_good_A_ch,\
                                                                 mean_firing_rate_task_3_a_good_B_ch, mean_firing_rate_task_3_b_good_B_ch], axis = 0)
            
                    clusters_task_1_forced = np.concatenate([mean_firing_rate_task_1_a_good_A_ch_f,mean_firing_rate_task_1_b_good_A_ch_f,\
                                                                 mean_firing_rate_task_1_a_good_B_ch_f, mean_firing_rate_task_1_b_good_B_ch_f], axis = 0)
                    clusters_task_2_forced = np.concatenate([mean_firing_rate_task_2_a_good_A_ch_f,mean_firing_rate_task_2_b_good_A_ch_f,\
                                                                 mean_firing_rate_task_2_a_good_B_ch_f, mean_firing_rate_task_2_b_good_B_ch_f], axis = 0)
                    clusters_task_3_forced = np.concatenate([mean_firing_rate_task_3_a_good_A_ch_f,mean_firing_rate_task_3_b_good_A_ch_f,\
                                                                 mean_firing_rate_task_3_a_good_B_ch_f, mean_firing_rate_task_3_b_good_B_ch_f], axis = 0)
            
        
                    all_clusters_task_1_non_forced.append(clusters_task_1_non_forced)
                    all_clusters_task_2_non_forced.append(clusters_task_2_non_forced)
                    all_clusters_task_3_non_forced.append(clusters_task_3_non_forced)
                    
                    all_clusters_task_1_forced.append(clusters_task_1_forced)
                    all_clusters_task_2_forced.append(clusters_task_2_forced)
                    all_clusters_task_3_forced.append(clusters_task_3_forced)
    return all_clusters_task_1_forced, all_clusters_task_2_forced, all_clusters_task_3_forced, all_clusters_task_1_non_forced,\
           all_clusters_task_2_non_forced,all_clusters_task_3_non_forced
        