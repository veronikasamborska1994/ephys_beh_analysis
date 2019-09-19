#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:28:47 2019

@author: veronikasamborska
"""
# =============================================================================
# Script for decoding A/B rewarded/non-rewarded over time of the trial in the same vs different task 
# =============================================================================

import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
sns.set_style("white") 




def classifier_pseudo_simultaneous_a_b_reward_no_reward(data, session, color, title):    
    
     
    y = data['DM']
    X = data['Data']

    all_sessions_1_1 = []
    all_sessions_1_2 = []
    all_sessions_2_1 = []
    all_sessions_2_2 = []
    all_sessions_3_1 = []

    for s, sess in enumerate(X):
        length = []

        # Design matrix for the session
        DM = y[s]
      
                    
        firing_rates_all_time = X[s]
        
        # Select 26 trials in each session
        min_trials_in_task = 26
        
        choices =  DM[:,1]
        task = DM[:,4]
        reward = DM[:,2]


        task_1_b_NR = np.where((task == 1) & (choices == 0) & (reward == 0))[0]
        task_2_b_NR = np.where((task == 2) & (choices == 0) & (reward == 0))[0]
        task_3_b_NR = np.where((task == 3) & (choices == 0) & (reward == 0))[0]
        
        task_1_b_R = np.where((task == 1) & (choices == 0) & (reward == 1))[0]
        task_2_b_R = np.where((task == 2) & (choices == 0) & (reward == 1))[0]
        task_3_b_R = np.where((task == 3) & (choices == 0) & (reward == 1))[0]
        
        task_1_a_NR = np.where((task == 1) & (choices == 1) & (reward == 0))[0]
        task_2_a_NR = np.where((task == 2) & (choices == 1) & (reward == 0))[0]
        task_3_a_NR = np.where((task == 3) & (choices == 1) & (reward == 0))[0]
        
        task_1_a_R = np.where((task == 1) & (choices == 1) & (reward == 1))[0]
        task_2_a_R = np.where((task == 2) & (choices == 1) & (reward == 1))[0]
        task_3_a_R = np.where((task == 3) & (choices == 1) & (reward == 1))[0]
        
        length.append([len(task_1_b_NR),len(task_2_b_NR),len(task_3_b_NR),len(task_1_b_R),len(task_2_b_R), len(task_3_b_R),\
                       len(task_1_a_NR), len(task_2_a_NR), len(task_3_a_NR), len(task_1_a_R), len(task_2_a_R), len(task_3_a_R)])
    
        # Only use sessions with min 20 trials of each trial type per task 
        min_trials_in_task  = 10
        
        if np.min(length) >= 20:
    
            # Select  Non-Rewarded B trials in all tasks 
            firing_rates_mean_task_1_1_nR_B = firing_rates_all_time[task_1_b_NR]
            firing_rates_mean_task_1_1_nR_B = firing_rates_mean_task_1_1_nR_B[:min_trials_in_task,:]
            
            firing_rates_mean_task_2_1_nR_B = firing_rates_all_time[task_2_b_NR]
            firing_rates_mean_task_2_1_nR_B = firing_rates_mean_task_2_1_nR_B[:min_trials_in_task,:]
            
            firing_rates_mean_task_3_1_nR_B = firing_rates_all_time[task_3_b_NR]
            firing_rates_mean_task_3_1_nR_B = firing_rates_mean_task_3_1_nR_B[:min_trials_in_task,:]
            
            # Select  Rewarded B trials in all tasks 
            firing_rates_mean_task_1_1_R_B = firing_rates_all_time[task_1_b_R]
            firing_rates_mean_task_1_1_R_B = firing_rates_mean_task_1_1_R_B[:min_trials_in_task,:]
            
            firing_rates_mean_task_2_1_R_B = firing_rates_all_time[task_2_b_R]
            firing_rates_mean_task_2_1_R_B = firing_rates_mean_task_2_1_R_B[:min_trials_in_task,:]
            
            firing_rates_mean_task_3_1_R_B = firing_rates_all_time[task_3_b_R]
            firing_rates_mean_task_3_1_R_B = firing_rates_mean_task_3_1_R_B[:min_trials_in_task,:]
            
            # Select  Non-Rewarded B trials in all tasks 
            firing_rates_mean_task_1_1_nR_A = firing_rates_all_time[task_1_a_NR]
            firing_rates_mean_task_1_1_nR_A = firing_rates_mean_task_1_1_nR_A[:min_trials_in_task,:]
            
            firing_rates_mean_task_2_1_nR_A = firing_rates_all_time[task_2_a_NR]
            firing_rates_mean_task_2_1_nR_A = firing_rates_mean_task_2_1_nR_A[:min_trials_in_task,:]
            
            firing_rates_mean_task_3_1_nR_A = firing_rates_all_time[task_3_a_NR]
            firing_rates_mean_task_3_1_nR_A = firing_rates_mean_task_3_1_nR_A[:min_trials_in_task,:]
            
            # Select  Rewarded B trials in all tasks 
            firing_rates_mean_task_1_1_R_A = firing_rates_all_time[task_1_a_R]
            firing_rates_mean_task_1_1_R_A = firing_rates_mean_task_1_1_R_A[:min_trials_in_task,:]
            
            firing_rates_mean_task_2_1_R_A = firing_rates_all_time[task_2_a_R]
            firing_rates_mean_task_2_1_R_A = firing_rates_mean_task_2_1_R_A[:min_trials_in_task,:]
            
            firing_rates_mean_task_3_1_R_A = firing_rates_all_time[task_3_a_R]
            firing_rates_mean_task_3_1_R_A = firing_rates_mean_task_3_1_R_A[:min_trials_in_task,:]
            
            
            # Finding the indices of initiations, choices and rewards 
            n_time = firing_rates_mean_task_1_1_nR_B.shape[2] 
                    
            task_1 = np.concatenate((firing_rates_mean_task_1_1_nR_B, firing_rates_mean_task_1_1_R_B, firing_rates_mean_task_1_1_nR_A,\
                                firing_rates_mean_task_1_1_R_A), axis = 0)
            
            task_2 = np.concatenate((firing_rates_mean_task_2_1_nR_B, firing_rates_mean_task_2_1_R_B, firing_rates_mean_task_2_1_nR_A,\
                                firing_rates_mean_task_2_1_R_A), axis = 0)
            
            task_3 = np.concatenate((firing_rates_mean_task_3_1_nR_B, firing_rates_mean_task_3_1_R_B, firing_rates_mean_task_3_1_nR_A,\
                               firing_rates_mean_task_3_1_R_A), axis = 0)
            
            all_sessions_1_1.append(np.concatenate(task_1, axis = 1))
            all_sessions_2_1.append(np.concatenate(task_2, axis = 1))
            all_sessions_3_1.append(np.concatenate(task_3, axis = 1))
          
            
            # Creating a vector which identifies trial stage in the firing rate vector
            b_n_rewarded_i = np.zeros(n_time*firing_rates_mean_task_3_1_R_A.shape[0])
            b_rewarded_i = np.ones(n_time*firing_rates_mean_task_3_1_R_A.shape[0])
            a_n_rewarded_i = np.zeros(n_time*firing_rates_mean_task_3_1_R_A.shape[0])
            a_n_rewarded_i[:] = 2
            a_rewarded_i = np.zeros(n_time*firing_rates_mean_task_3_1_R_A.shape[0])
            a_rewarded_i[:] = 3           
            
            Y = np.hstack((b_n_rewarded_i,b_rewarded_i, a_n_rewarded_i,a_rewarded_i))
            
    all_sessions_1  = np.concatenate(all_sessions_1_1, axis = 0) 
    
    all_sessions_2  = np.concatenate(all_sessions_2_1, axis = 0)
    
    all_sessions_3  = np.concatenate(all_sessions_3_1, axis = 0)

    model_nb = svm.SVC(gamma='scale')
    #model_nb = LogisticRegression()
    
    model_nb.fit(np.transpose(all_sessions_1),Y)  
    y_pred_class_between_t_2_1 = model_nb.predict(np.transpose(all_sessions_2))
    
    model_nb.fit(np.transpose(all_sessions_2),Y)  
    y_pred_class_between_t_3_1 = model_nb.predict(np.transpose(all_sessions_3))
    
    
    # Task_1 
    y_reshape =  y_pred_class_between_t_2_1.reshape(task_1.shape[0],task_1.shape[2])
    Y_reshape = Y.reshape(task_1.shape[0],task_1.shape[2])
  
    corr = (y_reshape == Y_reshape)
    along_time = np.asarray(corr, dtype = int)
    
    sum_time  = np.sum(along_time, axis = 0)
    accuracy_across_time = sum_time/40
    
    # Task_2
    y_reshape_2 =  y_pred_class_between_t_3_1.reshape(task_1.shape[0],task_1.shape[2])
    Y_reshape_2 = Y.reshape(task_1.shape[0],task_1.shape[2])
  
    corr_2 = (y_reshape_2 == Y_reshape_2)
    along_time_2 = np.asarray(corr_2, dtype = int)
    
    sum_time_2 = np.sum(along_time_2, axis = 0)
    accuracy_across_time_2 = sum_time_2/40
    
    # Finding the indices of initiations, choices and rewards 
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()
    reward_time = initiate_choice_t[-2] +250           
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()
    
    mean_accuracy_across_time = np.mean([accuracy_across_time_2,accuracy_across_time],axis = 0)
    std_accuracy_across_time = np.std([accuracy_across_time_2,accuracy_across_time],axis = 0)/np.sqrt(2)
    x = np.arange(63)
    
    plt.plot(mean_accuracy_across_time, color = color, label = title)
    plt.fill_between(x, mean_accuracy_across_time-std_accuracy_across_time, mean_accuracy_across_time+std_accuracy_across_time,  color = color, alpha =  0.3)

    plt.vlines(initiation, ymin = 0, ymax = 1,linestyle ='--', color = 'grey')
    plt.vlines(ind_choice, ymin = 0, ymax = 1,linestyle = '--', color  = 'black')
    plt.vlines(ind_reward, ymin = 0, ymax = 1, linestyle = '--', color  = 'pink')
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    #cnf_matrix = metrics.confusion_matrix(y_pred_class_between_t_2_1, Y)

        
        