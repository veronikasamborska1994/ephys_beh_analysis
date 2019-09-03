#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:57:22 2019

@author: veronikasamborska
"""

import regressions as re
import numpy as np
import matplotlib.pyplot as plt


def a_b_previous_choice_for_svd(session): 
    
    # Ones are As
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)    
    
    a_after_a_task_1 = []
    a_after_a_task_2 = []
    a_after_a_task_3 = []
    
    a_after_b_task_1 = []
    a_after_b_task_2 = []
    a_after_b_task_3 = []
    
    b_after_b_task_1 = []
    b_after_b_task_2 = []
    b_after_b_task_3 = []
    
    b_after_a_task_1 = []
    b_after_a_task_2 = []
    b_after_a_task_3 = []
    
    
    task = session.trial_data['task']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    
    task_non_forced = task[non_forced_array]
    task_1 = np.where(task_non_forced == 1)[0]
    task_1_len   = len(task_1) 
                
    task_2 = np.where(task_non_forced == 2)[0]        
    task_2_len  = len(task_2)
    
    predictor_A_Task_1[task_1_len:] = -1
    predictor_A_Task_2[:task_1_len] = -1
    predictor_A_Task_2[(task_1_len+task_2_len):] = -1
    predictor_A_Task_3[:(task_1_len+task_2_len)] = -1

    for i,predictor in enumerate(predictor_A_Task_1):
        if i > 0:
            if predictor_A_Task_1[i-1] == 1 and predictor_A_Task_1[i] == 1:
                a_after_a_task_1.append(1)
                a_after_b_task_1.append(0)
                b_after_b_task_1.append(0)
                b_after_a_task_1.append(0)
                
            elif predictor_A_Task_1[i-1] == 0 and predictor_A_Task_1[i] == 1:
                a_after_b_task_1.append(1)
                a_after_a_task_1.append(0)
                b_after_b_task_1.append(0)
                b_after_a_task_1.append(0)

            elif predictor_A_Task_1[i-1] == 0 and predictor_A_Task_1[i] == 0:
                a_after_b_task_1.append(0)
                a_after_a_task_1.append(0)
                b_after_a_task_1.append(0)
                b_after_b_task_1.append(1)
                
            elif predictor_A_Task_1[i-1] == 1 and predictor_A_Task_1[i] == 0:
                a_after_b_task_1.append(0)
                a_after_a_task_1.append(0)
                b_after_b_task_1.append(0)
                b_after_a_task_1.append(1)
            else:
                a_after_b_task_1.append(-1)
                a_after_a_task_1.append(-1)
                b_after_b_task_1.append(-1)
                b_after_a_task_1.append(-1)

    for i,predictor in enumerate(predictor_A_Task_2):
        if i > 0:
            if predictor_A_Task_2[i-1] == 1 and predictor_A_Task_2[i] == 1:
                a_after_a_task_2.append(1)
                a_after_b_task_2.append(0)
                b_after_b_task_2.append(0)
                b_after_a_task_2.append(0)
                
            elif predictor_A_Task_2[i-1] == 0 and predictor_A_Task_2[i] == 1:
                a_after_b_task_2.append(1)
                a_after_a_task_2.append(0)
                b_after_b_task_2.append(0)
                b_after_a_task_2.append(0)

            elif predictor_A_Task_2[i-1] == 0 and predictor_A_Task_2[i] == 0:
                a_after_b_task_2.append(0)
                a_after_a_task_2.append(0)
                b_after_a_task_2.append(0)
                b_after_b_task_2.append(1)
                
            elif predictor_A_Task_2[i-1] == 1 and predictor_A_Task_2[i] == 0:
                a_after_b_task_2.append(0)
                a_after_a_task_2.append(0)
                b_after_b_task_2.append(0)
                b_after_a_task_2.append(1)
            else:
                a_after_b_task_2.append(-1)
                a_after_a_task_2.append(-1)
                b_after_b_task_2.append(-1)
                b_after_a_task_2.append(-1)

                
    for i,predictor in enumerate(predictor_A_Task_3):
        
        if i > 0:
            
            if predictor_A_Task_3[i-1] == 1 and predictor_A_Task_3[i] == 1:
                a_after_a_task_3.append(1)
                a_after_b_task_3.append(0)
                b_after_b_task_3.append(0)
                b_after_a_task_3.append(0)
                
            elif predictor_A_Task_3[i-1] == 0 and predictor_A_Task_3[i] == 1:
                a_after_b_task_3.append(1)
                a_after_a_task_3.append(0)
                b_after_b_task_3.append(0)
                b_after_a_task_3.append(0)

            elif predictor_A_Task_3[i-1] == 0 and predictor_A_Task_3[i] == 0:
                a_after_b_task_3.append(0)
                a_after_a_task_3.append(0)
                b_after_a_task_3.append(0)
                b_after_b_task_3.append(1)
                
            elif predictor_A_Task_3[i-1] == 1 and predictor_A_Task_3[i] == 0:
                a_after_b_task_3.append(0)
                a_after_a_task_3.append(0)
                b_after_b_task_3.append(0)
                b_after_a_task_3.append(1)
                
            else:
                a_after_b_task_3.append(-1)
                a_after_a_task_3.append(-1)
                b_after_b_task_3.append(-1)
                b_after_a_task_3.append(-1)
               
    reward_previous = []               
    for i,predictor in enumerate(reward):
        if i > 0:
            if reward[i-1] == 1 :
                reward_previous.append(1)
            else:
                reward_previous.append(0)  
                
    a_after_a_task_1_reward = np.where((np.asarray(a_after_a_task_1) == 1) & (np.asarray(reward[1:]) == 1))
    a_after_a_task_2_reward = np.where((np.asarray(a_after_a_task_2) == 1) & (np.asarray(reward[1:]) == 1))    
    a_after_a_task_3_reward = np.where((np.asarray(a_after_a_task_3) == 1) & (np.asarray(reward[1:]) == 1))
    
    a_after_b_task_1_reward =  np.where((np.asarray(a_after_b_task_1) == 1) & (np.asarray(reward[1:]) == 1))
    a_after_b_task_2_reward = np.where((np.asarray(a_after_b_task_2) == 1) & (np.asarray(reward[1:]) == 1))
    a_after_b_task_3_reward = np.where((np.asarray(a_after_b_task_3) == 1) & (np.asarray(reward[1:]) == 1))
    
    b_after_b_task_1_reward = np.where((np.asarray(b_after_b_task_1) == 1) & (np.asarray(reward[1:]) == 1))
    b_after_b_task_2_reward = np.where((np.asarray(b_after_b_task_2) == 1) & (np.asarray(reward[1:]) == 1))
    b_after_b_task_3_reward = np.where((np.asarray(b_after_b_task_3) == 1) & (np.asarray(reward[1:]) == 1))
    
    b_after_a_task_1_reward = np.where((np.asarray(b_after_a_task_1) == 1) & (np.asarray(reward[1:]) == 1))
    b_after_a_task_2_reward = np.where((np.asarray(b_after_a_task_2) == 1) & (np.asarray(reward[1:]) == 1))
    b_after_a_task_3_reward = np.where((np.asarray(b_after_a_task_3) == 1) & (np.asarray(reward[1:]) == 1))


    a_after_a_task_1_nreward = np.where((np.asarray(a_after_a_task_1) == 1) & (np.asarray(reward[1:]) == 0))
    a_after_a_task_2_nreward = np.where((np.asarray(a_after_a_task_2) == 1) & (np.asarray(reward[1:]) == 0))
    a_after_a_task_3_nreward = np.where((np.asarray(a_after_a_task_3) == 1) & (np.asarray(reward[1:]) == 0))
    
    a_after_b_task_1_nreward = np.where((np.asarray(a_after_b_task_1) == 1) & (np.asarray(reward[1:]) == 0))
    a_after_b_task_2_nreward = np.where((np.asarray(a_after_b_task_2) == 1) & (np.asarray(reward[1:]) == 0))
    a_after_b_task_3_nreward = np.where((np.asarray(a_after_b_task_3) == 1) & (np.asarray(reward[1:]) == 0))
    
    b_after_b_task_1_nreward = np.where((np.asarray(b_after_b_task_1) == 1) & (np.asarray(reward[1:]) == 0))
    b_after_b_task_2_nreward = np.where((np.asarray(b_after_b_task_2) == 1) & (np.asarray(reward[1:]) == 0))
    b_after_b_task_3_nreward = np.where((np.asarray(b_after_b_task_3) == 1) & (np.asarray(reward[1:]) == 0))
    
    b_after_a_task_1_nreward =  np.where((np.asarray(b_after_a_task_1)== 1) & (np.asarray(reward[1:]) == 0))
    b_after_a_task_2_nreward = np.where((np.asarray(b_after_a_task_2) == 1) & (np.asarray(reward[1:]) == 0))
    b_after_a_task_3_nreward = np.where((np.asarray(b_after_a_task_3) == 1) & (np.asarray(reward[1:]) == 0))
    
    return a_after_a_task_1_reward,a_after_a_task_2_reward,a_after_a_task_3_reward,a_after_b_task_1_reward,\
    a_after_b_task_2_reward,a_after_b_task_3_reward,b_after_b_task_1_reward,b_after_b_task_2_reward,\
    b_after_b_task_3_reward,b_after_a_task_1_reward,b_after_a_task_2_reward,b_after_a_task_3_reward,\
    a_after_a_task_1_nreward,a_after_a_task_2_nreward,a_after_a_task_3_nreward,\
    a_after_b_task_1_nreward,a_after_b_task_2_nreward,a_after_b_task_3_nreward,\
    b_after_b_task_1_nreward,b_after_b_task_2_nreward,b_after_b_task_3_nreward,\
    b_after_a_task_1_nreward,b_after_a_task_2_nreward,b_after_a_task_3_nreward


def extract_trials(session):
    
    a_after_a_task_1_reward,a_after_a_task_2_reward,a_after_a_task_3_reward,a_after_b_task_1_reward,\
    a_after_b_task_2_reward,a_after_b_task_3_reward,b_after_b_task_1_reward,b_after_b_task_2_reward,\
    b_after_b_task_3_reward,b_after_a_task_1_reward,b_after_a_task_2_reward,b_after_a_task_3_reward,\
    a_after_a_task_1_nreward,a_after_a_task_2_nreward,a_after_a_task_3_nreward,\
    a_after_b_task_1_nreward,a_after_b_task_2_nreward,a_after_b_task_3_nreward,\
    b_after_b_task_1_nreward,b_after_b_task_2_nreward,b_after_b_task_3_nreward,\
    b_after_a_task_1_nreward,b_after_a_task_2_nreward,b_after_a_task_3_nreward = a_b_previous_choice_for_svd(session)
    
    spikes = session.ephys
    spikes = spikes[:,~np.isnan(spikes[1,:])] 
    aligned_rates = session.aligned_rates
    aligned_rates = aligned_rates[1:,:,:]

    aligned_rates_task_1_a_a_r = aligned_rates[a_after_a_task_1_reward]
    aligned_rates_task_2_a_a_r = aligned_rates[a_after_a_task_2_reward]
    aligned_rates_task_3_a_a_r = aligned_rates[a_after_a_task_3_reward]

    aligned_rates_task_1_a_b_r =  aligned_rates[a_after_b_task_1_reward]
    aligned_rates_task_2_a_b_r =  aligned_rates[a_after_b_task_2_reward]
    aligned_rates_task_3_a_b_r =  aligned_rates[a_after_b_task_3_reward]
    
    aligned_rates_task_1_b_b_r = aligned_rates[b_after_b_task_1_reward]
    aligned_rates_task_2_b_b_r = aligned_rates[b_after_b_task_2_reward]
    aligned_rates_task_3_b_b_r = aligned_rates[b_after_b_task_3_reward]

    aligned_rates_task_1_b_a_r = aligned_rates[b_after_a_task_1_reward]
    aligned_rates_task_2_b_a_r = aligned_rates[b_after_a_task_2_reward]
    aligned_rates_task_3_b_a_r = aligned_rates[b_after_a_task_3_reward]

    aligned_rates_task_1_a_a_nr = aligned_rates[a_after_a_task_1_nreward]
    aligned_rates_task_2_a_a_nr = aligned_rates[a_after_a_task_2_nreward]
    aligned_rates_task_3_a_a_nr = aligned_rates[a_after_a_task_3_nreward]

    aligned_rates_task_1_a_b_nr =  aligned_rates[a_after_b_task_1_nreward]
    aligned_rates_task_2_a_b_nr =  aligned_rates[a_after_b_task_2_nreward]
    aligned_rates_task_3_a_b_nr =  aligned_rates[a_after_b_task_3_nreward]
    
    aligned_rates_task_1_b_b_nr = aligned_rates[b_after_b_task_1_nreward]
    aligned_rates_task_2_b_b_nr = aligned_rates[b_after_b_task_2_nreward]
    aligned_rates_task_3_b_b_nr = aligned_rates[b_after_b_task_3_nreward]

    aligned_rates_task_1_b_a_nr = aligned_rates[b_after_a_task_1_nreward]
    aligned_rates_task_2_b_a_nr = aligned_rates[b_after_a_task_2_nreward]
    aligned_rates_task_3_b_a_nr = aligned_rates[b_after_a_task_3_nreward]
    
    # Split into halves rewarded
    
    aligned_rates_task_1_a_a_r_1 = aligned_rates_task_1_a_a_r[:int(len(aligned_rates_task_1_a_a_r)/2)]
    aligned_rates_task_1_a_a_r_2 = aligned_rates_task_1_a_a_r[int(len(aligned_rates_task_1_a_a_r)/2):]

    aligned_rates_task_2_a_a_r_1 = aligned_rates_task_2_a_a_r[:int(len(aligned_rates_task_2_a_a_r)/2)]
    aligned_rates_task_2_a_a_r_2 = aligned_rates_task_2_a_a_r[int(len(aligned_rates_task_2_a_a_r)/2):]

    aligned_rates_task_3_a_a_r_1 = aligned_rates_task_3_a_a_r[:int(len(aligned_rates_task_3_a_a_r)/2)]
    aligned_rates_task_3_a_a_r_2 = aligned_rates_task_3_a_a_r[int(len(aligned_rates_task_3_a_a_r)/2):]
 
    aligned_rates_task_1_a_b_r_1 = aligned_rates_task_1_a_b_r[:int(len(aligned_rates_task_1_a_b_r)/2)]
    aligned_rates_task_1_a_b_r_2 = aligned_rates_task_1_a_b_r[int(len(aligned_rates_task_1_a_b_r)/2):]

    aligned_rates_task_2_a_b_r_1 = aligned_rates_task_2_a_b_r[:int(len(aligned_rates_task_2_a_b_r)/2)]
    aligned_rates_task_2_a_b_r_2 = aligned_rates_task_2_a_b_r[int(len(aligned_rates_task_2_a_b_r)/2):]

    aligned_rates_task_3_a_b_r_1 = aligned_rates_task_3_a_b_r[:int(len(aligned_rates_task_3_a_b_r)/2)]
    aligned_rates_task_3_a_b_r_2 = aligned_rates_task_3_a_b_r[int(len(aligned_rates_task_3_a_b_r)/2):]

    aligned_rates_task_1_b_b_r_1 = aligned_rates_task_1_b_b_r[:int(len(aligned_rates_task_1_b_b_r)/2)]
    aligned_rates_task_1_b_b_r_2 = aligned_rates_task_1_b_b_r[int(len(aligned_rates_task_1_b_b_r)/2):]

    aligned_rates_task_2_b_b_r_1 = aligned_rates_task_2_b_b_r[:int(len(aligned_rates_task_2_b_b_r)/2)]
    aligned_rates_task_2_b_b_r_2 = aligned_rates_task_2_b_b_r[int(len(aligned_rates_task_2_b_b_r)/2):]

    aligned_rates_task_3_b_b_r_1 = aligned_rates_task_3_b_b_r[:int(len(aligned_rates_task_3_b_b_r)/2)]
    aligned_rates_task_3_b_b_r_2 = aligned_rates_task_3_b_b_r[int(len(aligned_rates_task_3_b_b_r)/2):]

    aligned_rates_task_1_b_a_r_1 = aligned_rates_task_1_b_a_r[:int(len(aligned_rates_task_1_b_a_r)/2)]
    aligned_rates_task_1_b_a_r_2 = aligned_rates_task_1_b_a_r[int(len(aligned_rates_task_1_b_a_r)/2):]

    aligned_rates_task_2_b_a_r_1 = aligned_rates_task_2_b_a_r[:int(len(aligned_rates_task_2_b_a_r)/2)]
    aligned_rates_task_2_b_a_r_2 = aligned_rates_task_2_b_a_r[int(len(aligned_rates_task_2_b_a_r)/2):]

    aligned_rates_task_3_b_a_r_1 = aligned_rates_task_3_b_a_r[:int(len(aligned_rates_task_3_b_a_r)/2)]
    aligned_rates_task_3_b_a_r_2 = aligned_rates_task_3_b_a_r[int(len(aligned_rates_task_3_b_a_r)/2):]
    
    # Split into halves non - rewarded
    
    aligned_rates_task_1_a_a_nr_1 = aligned_rates_task_1_a_a_nr[:int(len(aligned_rates_task_1_a_a_nr)/2)]
    aligned_rates_task_1_a_a_nr_2 = aligned_rates_task_1_a_a_nr[int(len(aligned_rates_task_1_a_a_nr)/2):]

    aligned_rates_task_2_a_a_nr_1 = aligned_rates_task_2_a_a_nr[:int(len(aligned_rates_task_2_a_a_nr)/2)]
    aligned_rates_task_2_a_a_nr_2 = aligned_rates_task_2_a_a_nr[int(len(aligned_rates_task_2_a_a_nr)/2):]

    aligned_rates_task_3_a_a_nr_1 = aligned_rates_task_3_a_a_nr[:int(len(aligned_rates_task_3_a_a_nr)/2)]
    aligned_rates_task_3_a_a_nr_2 = aligned_rates_task_3_a_a_nr[int(len(aligned_rates_task_3_a_a_nr)/2):]
 
    aligned_rates_task_1_a_b_nr_1 = aligned_rates_task_1_a_b_nr[:int(len(aligned_rates_task_1_a_b_nr)/2)]
    aligned_rates_task_1_a_b_nr_2 = aligned_rates_task_1_a_b_nr[int(len(aligned_rates_task_1_a_b_nr)/2):]

    aligned_rates_task_2_a_b_nr_1 = aligned_rates_task_2_a_b_nr[:int(len(aligned_rates_task_2_a_b_nr)/2)]
    aligned_rates_task_2_a_b_nr_2 = aligned_rates_task_2_a_b_nr[int(len(aligned_rates_task_2_a_b_nr)/2):]

    aligned_rates_task_3_a_b_nr_1 = aligned_rates_task_3_a_b_nr[:int(len(aligned_rates_task_3_a_b_nr)/2)]
    aligned_rates_task_3_a_b_nr_2 = aligned_rates_task_3_a_b_nr[int(len(aligned_rates_task_3_a_b_nr)/2):]

    aligned_rates_task_1_b_b_nr_1 = aligned_rates_task_1_b_b_nr[:int(len(aligned_rates_task_1_b_b_nr)/2)]
    aligned_rates_task_1_b_b_nr_2 = aligned_rates_task_1_b_b_nr[int(len(aligned_rates_task_1_b_b_nr)/2):]

    aligned_rates_task_2_b_b_nr_1 = aligned_rates_task_2_b_b_nr[:int(len(aligned_rates_task_2_b_b_nr)/2)]
    aligned_rates_task_2_b_b_nr_2 = aligned_rates_task_2_b_b_nr[int(len(aligned_rates_task_2_b_b_nr)/2):]

    aligned_rates_task_3_b_b_nr_1 = aligned_rates_task_3_b_b_nr[:int(len(aligned_rates_task_3_b_b_nr)/2)]
    aligned_rates_task_3_b_b_nr_2 = aligned_rates_task_3_b_b_nr[int(len(aligned_rates_task_3_b_b_nr)/2):]

    aligned_rates_task_1_b_a_nr_1 = aligned_rates_task_1_b_a_nr[:int(len(aligned_rates_task_1_b_a_nr)/2)]
    aligned_rates_task_1_b_a_nr_2 = aligned_rates_task_1_b_a_nr[int(len(aligned_rates_task_1_b_a_nr)/2):]

    aligned_rates_task_2_b_a_nr_1 = aligned_rates_task_2_b_a_nr[:int(len(aligned_rates_task_2_b_a_nr)/2)]
    aligned_rates_task_2_b_a_nr_2 = aligned_rates_task_2_b_a_nr[int(len(aligned_rates_task_2_b_a_nr)/2):]

    aligned_rates_task_3_b_a_nr_1 = aligned_rates_task_3_b_a_nr[:int(len(aligned_rates_task_3_b_a_nr)/2)]
    aligned_rates_task_3_b_a_nr_2 = aligned_rates_task_3_b_a_nr[int(len(aligned_rates_task_3_b_a_nr)/2):]
    
    
    return aligned_rates_task_1_a_a_r_1, aligned_rates_task_1_a_a_r_2, aligned_rates_task_2_a_a_r_1,\
     aligned_rates_task_2_a_a_r_2, aligned_rates_task_3_a_a_r_1,\
     aligned_rates_task_3_a_a_r_2, aligned_rates_task_1_a_b_r_1, aligned_rates_task_1_a_b_r_2,\
     aligned_rates_task_2_a_b_r_1, aligned_rates_task_2_a_b_r_2, aligned_rates_task_3_a_b_r_1,\
     aligned_rates_task_3_a_b_r_2, aligned_rates_task_1_b_b_r_1, aligned_rates_task_1_b_b_r_2,\
     aligned_rates_task_2_b_b_r_1, aligned_rates_task_2_b_b_r_2, aligned_rates_task_3_b_b_r_1,\
     aligned_rates_task_3_b_b_r_2, aligned_rates_task_1_b_a_r_1, aligned_rates_task_1_b_a_r_2,\
     aligned_rates_task_2_b_a_r_1, aligned_rates_task_2_b_a_r_2, aligned_rates_task_3_b_a_r_1,\
     aligned_rates_task_3_b_a_r_2, aligned_rates_task_1_a_a_nr_1, aligned_rates_task_1_a_a_nr_2,\
     aligned_rates_task_2_a_a_nr_1, aligned_rates_task_2_a_a_nr_2,  aligned_rates_task_3_a_a_nr_1,\
     aligned_rates_task_3_a_a_nr_2, aligned_rates_task_1_a_b_nr_1, aligned_rates_task_1_a_b_nr_2,\
     aligned_rates_task_2_a_b_nr_1, aligned_rates_task_2_a_b_nr_2, aligned_rates_task_3_a_b_nr_1,\
     aligned_rates_task_3_a_b_nr_2, aligned_rates_task_1_b_b_nr_1, aligned_rates_task_1_b_b_nr_2,\
     aligned_rates_task_2_b_b_nr_1,  aligned_rates_task_2_b_b_nr_2, aligned_rates_task_3_b_b_nr_1,\
     aligned_rates_task_3_b_b_nr_2, aligned_rates_task_1_b_a_nr_1,  aligned_rates_task_1_b_a_nr_2,\
     aligned_rates_task_2_b_a_nr_1, aligned_rates_task_2_b_a_nr_2, aligned_rates_task_3_b_a_nr_1,\
     aligned_rates_task_3_b_a_nr_2 



def mean_across_trials(experiment):
    all_clusters_task_1_first_half = []
    all_clusters_task_1_second_half = []
    all_clusters_task_2_first_half = []
    all_clusters_task_2_second_half = []
    all_clusters_task_3_first_half = []
    all_clusters_task_3_second_half = []
    
    for s,session in enumerate(experiment):
       spikes = session.ephys
       spikes = spikes[:,~np.isnan(spikes[1,:])] 
       aligned_rates_task_1_a_a_r_1, aligned_rates_task_1_a_a_r_2, aligned_rates_task_2_a_a_r_1,\
       aligned_rates_task_2_a_a_r_2, aligned_rates_task_3_a_a_r_1,\
       aligned_rates_task_3_a_a_r_2, aligned_rates_task_1_a_b_r_1, aligned_rates_task_1_a_b_r_2,\
       aligned_rates_task_2_a_b_r_1, aligned_rates_task_2_a_b_r_2, aligned_rates_task_3_a_b_r_1,\
       aligned_rates_task_3_a_b_r_2, aligned_rates_task_1_b_b_r_1, aligned_rates_task_1_b_b_r_2,\
       aligned_rates_task_2_b_b_r_1, aligned_rates_task_2_b_b_r_2, aligned_rates_task_3_b_b_r_1,\
       aligned_rates_task_3_b_b_r_2, aligned_rates_task_1_b_a_r_1, aligned_rates_task_1_b_a_r_2,\
       aligned_rates_task_2_b_a_r_1, aligned_rates_task_2_b_a_r_2, aligned_rates_task_3_b_a_r_1,\
       aligned_rates_task_3_b_a_r_2, aligned_rates_task_1_a_a_nr_1, aligned_rates_task_1_a_a_nr_2,\
       aligned_rates_task_2_a_a_nr_1, aligned_rates_task_2_a_a_nr_2,  aligned_rates_task_3_a_a_nr_1,\
       aligned_rates_task_3_a_a_nr_2, aligned_rates_task_1_a_b_nr_1, aligned_rates_task_1_a_b_nr_2,\
       aligned_rates_task_2_a_b_nr_1, aligned_rates_task_2_a_b_nr_2, aligned_rates_task_3_a_b_nr_1,\
       aligned_rates_task_3_a_b_nr_2, aligned_rates_task_1_b_b_nr_1, aligned_rates_task_1_b_b_nr_2,\
       aligned_rates_task_2_b_b_nr_1,  aligned_rates_task_2_b_b_nr_2, aligned_rates_task_3_b_b_nr_1,\
       aligned_rates_task_3_b_b_nr_2, aligned_rates_task_1_b_a_nr_1,  aligned_rates_task_1_b_a_nr_2,\
       aligned_rates_task_2_b_a_nr_1, aligned_rates_task_2_b_a_nr_2, aligned_rates_task_3_b_a_nr_1,\
       aligned_rates_task_3_b_a_nr_2 = extract_trials(session)

       if  (len(aligned_rates_task_1_a_a_r_1) > 0) & (len(aligned_rates_task_1_a_a_r_2) > 0) & \
           (len(aligned_rates_task_2_a_a_r_1) > 0) & (len(aligned_rates_task_2_a_a_r_2) > 0) & \
           (len(aligned_rates_task_3_a_a_r_1) > 0) & (len(aligned_rates_task_3_a_a_r_2) > 0) & \
           (len(aligned_rates_task_1_a_b_r_1) > 0) & (len(aligned_rates_task_1_a_b_r_2) > 0) & \
           (len(aligned_rates_task_2_a_b_r_1) > 0) & (len(aligned_rates_task_2_a_b_r_2) > 0) & \
           (len(aligned_rates_task_3_a_b_r_1) > 0) & (len(aligned_rates_task_3_a_b_r_2) > 0) & \
           (len(aligned_rates_task_1_b_b_r_1) > 0) & (len(aligned_rates_task_1_b_b_r_2) > 0) & \
           (len(aligned_rates_task_2_b_b_r_1) > 0) & (len(aligned_rates_task_2_b_b_r_2) > 0) & \
           (len(aligned_rates_task_3_b_b_r_1) > 0) & (len(aligned_rates_task_3_b_b_r_2) > 0) & \
           (len(aligned_rates_task_1_b_a_r_1) > 0) & (len(aligned_rates_task_1_b_a_r_2) > 0) & \
           (len(aligned_rates_task_2_b_a_r_1) > 0) & (len(aligned_rates_task_2_b_a_r_2) > 0) & \
           (len(aligned_rates_task_3_b_a_r_1) > 0) & (len(aligned_rates_task_3_b_a_r_2) > 0) & \
           (len(aligned_rates_task_1_a_a_nr_1) > 0) & (len(aligned_rates_task_1_a_a_nr_2) > 0) & \
           (len(aligned_rates_task_2_a_a_nr_1) > 0) & (len(aligned_rates_task_2_a_a_nr_2) > 0) & \
           (len(aligned_rates_task_3_a_a_nr_1) > 0) & (len(aligned_rates_task_3_a_a_nr_2) > 0) & \
           (len(aligned_rates_task_1_a_b_nr_1) > 0) & (len(aligned_rates_task_1_a_b_nr_2) > 0) & \
           (len(aligned_rates_task_2_a_b_nr_1) > 0) & (len(aligned_rates_task_2_a_b_nr_2) > 0) & \
           (len(aligned_rates_task_3_a_b_nr_1) > 0) & (len(aligned_rates_task_3_a_b_nr_2) > 0) & \
           (len(aligned_rates_task_1_b_b_nr_1) > 0) & (len(aligned_rates_task_1_b_b_nr_2) > 0) & \
           (len(aligned_rates_task_2_b_b_nr_1) > 0) & (len(aligned_rates_task_2_b_b_nr_2) > 0) & \
           (len(aligned_rates_task_3_b_b_nr_1) > 0) & (len(aligned_rates_task_3_b_b_nr_2) > 0) & \
           (len(aligned_rates_task_1_b_a_nr_1) > 0) & (len(aligned_rates_task_1_b_a_nr_2) > 0) & \
           (len(aligned_rates_task_2_b_a_nr_1) > 0) & (len(aligned_rates_task_2_b_a_nr_2) > 0) & \
           (len(aligned_rates_task_3_b_a_nr_1) > 0) & (len(aligned_rates_task_3_b_a_nr_2) > 0):
               
               unique_neurons  = np.unique(spikes[0])   
               for i in range(len(unique_neurons)):
                    mean_aligned_rates_task_1_a_a_r_1  = np.mean(aligned_rates_task_1_a_a_r_1[:,i,:],0)
                    mean_aligned_rates_task_1_a_a_r_2  = np.mean(aligned_rates_task_1_a_a_r_2[:,i,:],0)

                    mean_aligned_rates_task_2_a_a_r_1  = np.mean(aligned_rates_task_2_a_a_r_1[:,i,:],0)
                    mean_aligned_rates_task_2_a_a_r_2  = np.mean(aligned_rates_task_2_a_a_r_2[:,i,:],0)

                    mean_aligned_rates_task_3_a_a_r_1  = np.mean(aligned_rates_task_3_a_a_r_1[:,i,:],0)
                    mean_aligned_rates_task_3_a_a_r_2  = np.mean(aligned_rates_task_3_a_a_r_2[:,i,:],0)
                    
                    mean_aligned_rates_task_1_a_b_r_1  = np.mean(aligned_rates_task_1_a_b_r_1[:,i,:],0)
                    mean_aligned_rates_task_1_a_b_r_2  = np.mean(aligned_rates_task_1_a_b_r_2[:,i,:],0)

                    mean_aligned_rates_task_2_a_b_r_1  = np.mean(aligned_rates_task_2_a_b_r_1[:,i,:],0)
                    mean_aligned_rates_task_2_a_b_r_2  = np.mean(aligned_rates_task_2_a_b_r_2[:,i,:],0)

                    mean_aligned_rates_task_3_a_b_r_1  = np.mean(aligned_rates_task_3_a_b_r_1[:,i,:],0)
                    mean_aligned_rates_task_3_a_b_r_2  = np.mean(aligned_rates_task_3_a_b_r_2[:,i,:],0)

                    mean_aligned_rates_task_1_b_b_r_1  = np.mean(aligned_rates_task_1_b_b_r_1[:,i,:],0)
                    mean_aligned_rates_task_1_b_b_r_2  = np.mean(aligned_rates_task_1_b_b_r_2[:,i,:],0)

                    mean_aligned_rates_task_2_b_b_r_1  = np.mean(aligned_rates_task_2_b_b_r_1[:,i,:],0)
                    mean_aligned_rates_task_2_b_b_r_2  = np.mean(aligned_rates_task_2_b_b_r_2[:,i,:],0)

                    mean_aligned_rates_task_3_b_b_r_1  = np.mean(aligned_rates_task_3_b_b_r_1[:,i,:],0)
                    mean_aligned_rates_task_3_b_b_r_2  = np.mean(aligned_rates_task_3_b_b_r_2[:,i,:],0)

                    mean_aligned_rates_task_1_b_a_r_1  = np.mean(aligned_rates_task_1_b_a_r_1[:,i,:],0)
                    mean_aligned_rates_task_1_b_a_r_2  = np.mean(aligned_rates_task_1_b_a_r_2[:,i,:],0)
                    
                    mean_aligned_rates_task_2_b_a_r_1  = np.mean(aligned_rates_task_2_b_a_r_1[:,i,:],0)
                    mean_aligned_rates_task_2_b_a_r_2  = np.mean(aligned_rates_task_2_b_a_r_2[:,i,:],0)

                    mean_aligned_rates_task_3_b_a_r_1  = np.mean(aligned_rates_task_3_b_a_r_1[:,i,:],0)
                    mean_aligned_rates_task_3_b_a_r_2  = np.mean(aligned_rates_task_3_b_a_r_2[:,i,:],0)

                    mean_aligned_rates_task_1_a_a_nr_1  = np.mean(aligned_rates_task_1_a_a_nr_1[:,i,:],0)
                    mean_aligned_rates_task_1_a_a_nr_2  = np.mean(aligned_rates_task_1_a_a_nr_2[:,i,:],0)

                    mean_aligned_rates_task_2_a_a_nr_1  = np.mean(aligned_rates_task_2_a_a_nr_1[:,i,:],0)
                    mean_aligned_rates_task_2_a_a_nr_2  = np.mean(aligned_rates_task_2_a_a_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_3_a_a_nr_1  = np.mean(aligned_rates_task_3_a_a_nr_1[:,i,:],0)
                    mean_aligned_rates_task_3_a_a_nr_2  = np.mean(aligned_rates_task_3_a_a_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_1_a_b_nr_1  = np.mean(aligned_rates_task_1_a_b_nr_1[:,i,:],0)
                    mean_aligned_rates_task_1_a_b_nr_2  = np.mean(aligned_rates_task_1_a_b_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_2_a_b_nr_1  = np.mean(aligned_rates_task_2_a_b_nr_1[:,i,:],0)
                    mean_aligned_rates_task_2_a_b_nr_2  = np.mean(aligned_rates_task_2_a_b_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_3_a_b_nr_1  = np.mean(aligned_rates_task_3_a_b_nr_1[:,i,:],0)
                    mean_aligned_rates_task_3_a_b_nr_2  = np.mean(aligned_rates_task_3_a_b_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_1_b_b_nr_1  = np.mean(aligned_rates_task_1_b_b_nr_1[:,i,:],0)
                    mean_aligned_rates_task_1_b_b_nr_2  = np.mean(aligned_rates_task_1_b_b_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_2_b_b_nr_1  = np.mean(aligned_rates_task_2_b_b_nr_1[:,i,:],0)
                    mean_aligned_rates_task_2_b_b_nr_2  = np.mean(aligned_rates_task_2_b_b_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_3_b_b_nr_1  = np.mean(aligned_rates_task_3_b_b_nr_1[:,i,:],0)
                    mean_aligned_rates_task_3_b_b_nr_2  = np.mean(aligned_rates_task_3_b_b_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_1_b_a_nr_1  = np.mean(aligned_rates_task_1_b_a_nr_1[:,i,:],0)
                    mean_aligned_rates_task_1_b_a_nr_2  = np.mean(aligned_rates_task_1_b_a_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_2_b_a_nr_1  = np.mean(aligned_rates_task_2_b_a_nr_1[:,i,:],0)
                    mean_aligned_rates_task_2_b_a_nr_2  = np.mean(aligned_rates_task_2_b_a_nr_2[:,i,:],0)
                    
                    mean_aligned_rates_task_3_b_a_nr_1  = np.mean(aligned_rates_task_3_b_a_nr_1[:,i,:],0)
                    mean_aligned_rates_task_3_b_a_nr_2  = np.mean(aligned_rates_task_3_b_a_nr_2[:,i,:],0)
                    
                    task_1_first_half = np.concatenate((mean_aligned_rates_task_1_a_a_r_1, mean_aligned_rates_task_1_a_b_r_1,\
                                                        mean_aligned_rates_task_1_b_b_r_1,mean_aligned_rates_task_1_b_a_r_1,\
                                                        mean_aligned_rates_task_1_a_a_nr_1,mean_aligned_rates_task_1_a_b_nr_1,\
                                                        mean_aligned_rates_task_1_b_b_nr_1,mean_aligned_rates_task_1_b_a_nr_1 ), axis = 0)
                       
                    task_1_second_half = np.concatenate((mean_aligned_rates_task_1_a_a_r_2, mean_aligned_rates_task_1_a_b_r_2,\
                                                        mean_aligned_rates_task_1_b_b_r_2,mean_aligned_rates_task_1_b_a_r_2,\
                                                        mean_aligned_rates_task_1_a_a_nr_2,mean_aligned_rates_task_1_a_b_nr_2,\
                                                        mean_aligned_rates_task_1_b_b_nr_2,mean_aligned_rates_task_1_b_a_nr_2 ), axis = 0)
                       
                    
                    task_2_first_half = np.concatenate((mean_aligned_rates_task_2_a_a_r_1, mean_aligned_rates_task_2_a_b_r_1,\
                                                        mean_aligned_rates_task_2_b_b_r_1,mean_aligned_rates_task_2_b_a_r_1,\
                                                        mean_aligned_rates_task_2_a_a_nr_1,mean_aligned_rates_task_2_a_b_nr_1,\
                                                        mean_aligned_rates_task_2_b_b_nr_1,mean_aligned_rates_task_2_b_a_nr_1 ), axis = 0)
                       
                    task_2_second_half = np.concatenate((mean_aligned_rates_task_2_a_a_r_2, mean_aligned_rates_task_2_a_b_r_2,\
                                                        mean_aligned_rates_task_2_b_b_r_2,mean_aligned_rates_task_2_b_a_r_2,\
                                                        mean_aligned_rates_task_2_a_a_nr_2,mean_aligned_rates_task_2_a_b_nr_2,\
                                                        mean_aligned_rates_task_2_b_b_nr_2,mean_aligned_rates_task_2_b_a_nr_2 ), axis = 0)
                       
                    
                    task_3_first_half = np.concatenate((mean_aligned_rates_task_3_a_a_r_1, mean_aligned_rates_task_3_a_b_r_1,\
                                                        mean_aligned_rates_task_3_b_b_r_1,mean_aligned_rates_task_3_b_a_r_1,\
                                                        mean_aligned_rates_task_3_a_a_nr_1,mean_aligned_rates_task_3_a_b_nr_1,\
                                                        mean_aligned_rates_task_3_b_b_nr_1,mean_aligned_rates_task_3_b_a_nr_1 ), axis = 0)
                       
                    task_3_second_half = np.concatenate((mean_aligned_rates_task_3_a_a_r_2, mean_aligned_rates_task_3_a_b_r_2,\
                                                        mean_aligned_rates_task_3_b_b_r_2,mean_aligned_rates_task_3_b_a_r_2,\
                                                        mean_aligned_rates_task_3_a_a_nr_2,mean_aligned_rates_task_3_a_b_nr_2,\
                                                        mean_aligned_rates_task_3_b_b_nr_2,mean_aligned_rates_task_3_b_a_nr_2 ), axis = 0)
                       
                    
                    all_clusters_task_1_first_half.append(task_1_first_half) 
                    all_clusters_task_1_second_half.append(task_1_second_half)
                    all_clusters_task_2_first_half.append(task_2_first_half) 
                    all_clusters_task_2_second_half.append(task_2_second_half)
                    all_clusters_task_3_first_half.append(task_3_first_half) 
                    all_clusters_task_3_second_half.append(task_3_second_half)
                    
    all_clusters_task_1_first_half = np.asarray(all_clusters_task_1_first_half)
    all_clusters_task_1_second_half = np.asarray(all_clusters_task_1_second_half)

    all_clusters_task_2_first_half = np.asarray(all_clusters_task_2_first_half)
    all_clusters_task_2_second_half = np.asarray(all_clusters_task_2_second_half)

    all_clusters_task_3_first_half = np.asarray(all_clusters_task_3_first_half)
    all_clusters_task_3_second_half = np.asarray(all_clusters_task_3_second_half)
    
    return all_clusters_task_1_first_half,all_clusters_task_1_second_half,\
    all_clusters_task_2_first_half,all_clusters_task_2_second_half,\
    all_clusters_task_3_first_half,all_clusters_task_3_second_half

def demean(experiment):
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = mean_across_trials(experiment)
    
    all_data = np.concatenate([flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_first_half,\
                                       flattened_all_clusters_task_2_second_half, flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half], axis = 1)
    all_data_mean = np.mean(all_data, axis = 1)
        
           
    demeaned = np.transpose(all_data)- all_data_mean
    demeaned = np.transpose(demeaned)            
            
    demean_all_clusters_task_1_first_half = demeaned[:,:flattened_all_clusters_task_1_first_half.shape[1]]
    demean_all_clusters_task_1_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]:flattened_all_clusters_task_1_first_half.shape[1]*2]
            
    demean_all_clusters_task_2_first_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*2:flattened_all_clusters_task_1_first_half.shape[1]*3]
    demean_all_clusters_task_2_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*3:flattened_all_clusters_task_1_first_half.shape[1]*4]
        
    demean_all_clusters_task_3_first_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*4:flattened_all_clusters_task_1_first_half.shape[1]*5]
    demean_all_clusters_task_3_second_half = demeaned[:,flattened_all_clusters_task_1_first_half.shape[1]*5:flattened_all_clusters_task_1_first_half.shape[1]*6]
                     
    return demean_all_clusters_task_1_first_half,demean_all_clusters_task_1_second_half, demean_all_clusters_task_2_first_half,\
        demean_all_clusters_task_2_second_half, demean_all_clusters_task_3_first_half, demean_all_clusters_task_3_second_half

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = demean(experiment_aligned_HP)
 
m_full = np.concatenate([flattened_all_clusters_task_1_first_half,\
                            flattened_all_clusters_task_1_second_half], axis =1)
        
plt.figure()
corrmf = np.corrcoef(np.transpose(m_full))
plt.imshow(corrmf)

ticks_n  = np.linspace(0, corrmf.shape[0],16)
                    
plt.yticks(ticks_n, ('A A Reward T1 1', 'A B Reward T1 1','B B Reward T1 1',\
                     'B A Reward T1 1','A A No Reward T1 1', 'A B No Reward T1 1','B B No Reward T1 1',\
                     'B A No Reward T1 1', 'A A Reward T1 2', 'A B Reward T1 2','B B Reward T1 2',\
                     'B A Reward T1 2','A A No Reward T1 2', 'A B No Reward T1 2','B B No Reward T1 2',\
                     'B A No Reward T1 2'))
                     
plt.xticks(ticks_n, ('A A Reward T1 1', 'A B Reward T1 1','B B Reward T1 1',\
                     'B A Reward T1 1','A A No Reward T1 1', 'A B No Reward T1 1','B B No Reward T1 1',\
                     'B A No Reward T1 1', 'A A Reward T1 2', 'A B Reward T1 2','B B Reward T1 2',\
                     'B A Reward T1 2','A A No Reward T1 2', 'A B No Reward T2','B B No Reward T1 2',\
                     'B A No Reward T1 2'), rotation  = 'vertical')
      


def svd_plot(experiment, diagonal = False,HP = True):
    #Calculating SVDs for trials split by A and B, reward/no reward (no block information) 
   
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = demean(experiment)
     #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = True)
    
   
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = True)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)  
    t_v = np.transpose(vh_t1_1)  

    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_1 = np.transpose(u_t2_1)   
    t_v_t_2_1 = np.transpose(vh_t2_1)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  

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

   
      
   # average_within = average_within/average_within[-1]
    #average_between = average_between/average_between[-1]
    if HP == True:
        plt.figure(10)
        plt.plot(average_within, label = 'Within Tasks_HP', color='green')
        plt.plot(average_between, label = 'Between Tasks_HP',linestyle = '--', color = 'green')
       
    if HP == False:
        plt.figure(10)
        plt.plot(average_within, label = 'Within Tasks_PFC', color='black')
        plt.plot(average_between, label = 'Between Tasks_PFC',linestyle = '--', color = 'black')
        
        
    plt.figure(10)
    plt.title('SVD')
    plt.legend()
    
