#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:18:33 2019

@author: veronikasamborska
"""
# =============================================================================
# SVM classifier of time points in the trial; decoding I when it becomes B (same location) --> classifying as location in space or location in task?
# =============================================================================

import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white") 



def search_for_tasks_where_init_becomes_choice(DM):
     b_pokes = DM[:,6]
     i_pokes = DM[:,7]
    
     unique_b = np.unique(b_pokes)
     unique_i = np.unique(i_pokes)
     poke = np.intersect1d(unique_b,unique_i) 
     for i in poke:
         if i > 0:
             init_choice_port = i
     ind_choice = np.where(b_pokes ==  init_choice_port)[0]
     ind_init = np.where(i_pokes ==  init_choice_port)[0]
    
     return ind_choice, ind_init

def classifier_pseudo_simultaneous(data, session):    
    
    y = data['DM']
    X = data['Data']

    all_sessions_1_1 = []
    all_sessions_1_2 = []
    all_sessions_2_1 = []
    all_sessions_2_2 = []
   
    
    correct_list_within = []
    correct_list_between = []
    for s, sess in enumerate(X):
        
        # Design matrix for the session
        DM = y[s]
      
        
        firing_rates_all_time = X[s]
        
        # Select 26 trials in each session
        min_trials_in_task = 26
    
        ind_choices, ind_init = search_for_tasks_where_init_becomes_choice(DM)
      
        ind_choices_1 = ind_choices[:min_trials_in_task]
        ind_choices_2 = ind_choices[-min_trials_in_task:]

        ind_init_1 = ind_init[:min_trials_in_task]
        ind_init_2 = ind_init[-min_trials_in_task:]

        # Select the first min_trials_in_task in task one
        firing_rates_mean_task_1_1 = firing_rates_all_time[ind_choices_1]
        
        # Select the last min_trials_in_task in task one
        firing_rates_mean_task_1_2 = firing_rates_all_time[ind_choices_2]
       
        # Select the first min_trials_in_task in task two
        firing_rates_mean_task_2_1 = firing_rates_all_time[ind_init_1]
        firing_rates_mean_task_2_2 = firing_rates_all_time[ind_init_2]

        
        # Finding the indices of initiations, choices and rewards 
        n_time = firing_rates_mean_task_1_1.shape[2]
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()
        reward_time = initiate_choice_t[-2] +250
                
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
        ind_reward = (np.abs(t_out-reward_time)).argmin()

        # Finding the indices of initiations, choices and rewards        
        bins_before_init = np.zeros(initiation)
        bins_between_init_choice = np.ones(ind_choice-initiation)
        bins_between_choice_reward = np.ones(ind_reward-ind_choice)
        bins_between_choice_reward[:]= 2
        bins_between_post_reward = np.ones(n_time-ind_reward)
        bins_between_post_reward[:] = 3
        
        bins = np.hstack((bins_before_init,bins_between_init_choice,bins_between_choice_reward,bins_between_post_reward))
        
        all_sessions_1_1.append(np.concatenate(firing_rates_mean_task_1_1, axis = 1))
        all_sessions_1_2.append(np.concatenate(firing_rates_mean_task_1_2, axis = 1))
        all_sessions_2_1.append(np.concatenate(firing_rates_mean_task_2_1, axis = 1))
        all_sessions_2_2.append(np.concatenate(firing_rates_mean_task_2_2, axis = 1))
      
        l = np.concatenate(firing_rates_mean_task_1_1, axis = 1).shape[1]
        
        # Creating a vector which identifies trial stage in the firing rate vector
        Y = np.tile(bins,int(l/len(bins)))
        
    all_sessions_1_1  = np.concatenate(all_sessions_1_1, axis = 0)[:392,:]      
    all_sessions_1_2  = np.concatenate(all_sessions_1_2, axis = 0)[:392,:]

    all_sessions_2_1  = np.concatenate(all_sessions_2_1, axis = 0)[:392,:]   
    all_sessions_2_2  = np.concatenate(all_sessions_2_2, axis = 0)[:392,:]
    
    model_nb = svm.SVC(gamma='scale', class_weight='balanced')
            
    model_nb.fit(np.transpose(all_sessions_1_1),Y)  
    y_pred_class_between_t_1 = model_nb.predict(np.transpose(all_sessions_2_1))
    
    model_nb.fit(np.transpose(all_sessions_1_1),Y)     
    y_pred_class_within_t_1 = model_nb.predict(np.transpose(all_sessions_1_2))
   
    model_nb.fit(np.transpose(all_sessions_2_1),Y)
    y_pred_class_within_t_2 = model_nb.predict(np.transpose(all_sessions_2_2))

  
    correct_within_t_1 = metrics.accuracy_score(Y, y_pred_class_within_t_1)    
    correct_within_t_2 = metrics.accuracy_score(Y, y_pred_class_within_t_2)
  
    correct_between_t_1_2 = metrics.accuracy_score(Y, y_pred_class_between_t_1)

    correct_list_within.append(correct_within_t_1)
    #correct_list_within.append(correct_within_t_2)
    
    correct_list_between.append(correct_between_t_1_2)
    
    return correct_list_within, correct_list_between,y_pred_class_between_t_1, Y


def confusion_mat(y_pred_class,Y,title):    
    
    # Init Errors

    init_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 0))[0])
    init_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 0))[0])
    init_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 0))[0])
    init_overall_Y = np.sum([init_hit_rate_choice,init_hit_rate_reward,init_hit_rate_post_reward])
    sns.set(palette="Blues_d")
    sns.set_style("white") 
    plt.figure(figsize=[4,12])
    plt.subplot(411)
    plt.bar([1,2,3], [init_hit_rate_choice/init_overall_Y,init_hit_rate_reward/init_overall_Y,init_hit_rate_post_reward/init_overall_Y], tick_label = ['Choice-Reward', 'Reward', 'Post Reward'])
    plt.title('Pre-Initiation Errors')
    
    # Choice Errors
    choice_hit_rate_init = len(np.where((y_pred_class == 0)& (Y == 1))[0])
    choice_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 1))[0])
    choice_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 1))[0])
    init_overall_Y_choice = np.sum([choice_hit_rate_init,choice_hit_rate_reward,choice_hit_rate_post_reward])
    plt.subplot(412)
    plt.bar([1,2,3], [choice_hit_rate_init/init_overall_Y_choice,choice_hit_rate_reward/init_overall_Y_choice,choice_hit_rate_post_reward/init_overall_Y_choice], tick_label = ['Pre_Initiation', 'Choice-Reward', 'Post Reward'])
    plt.title('Initiation - Choice Errors')

    # Reward Errors
    reward_hit_rate_init = len(np.where((y_pred_class == 0)& (Y == 2))[0])
    reward_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 2))[0])
    reward_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 2))[0])
    init_overall_Y_reward = np.sum([reward_hit_rate_init,reward_hit_rate_choice,reward_hit_rate_post_reward])

    plt.subplot(413)
    plt.bar([1,2,3], [reward_hit_rate_init/init_overall_Y_reward,reward_hit_rate_choice/init_overall_Y_reward,reward_hit_rate_post_reward/init_overall_Y_reward], tick_label = ['Pre_Initiation', 'Init-Choice', 'Post Reward'])
    plt.title('Choice - Reward Errors')

    # Post - Reward Errors
    post_reward_hit_rate_init = len(np.where((y_pred_class == 0)& (Y == 3))[0])
    post_reward_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 3))[0])
    post_reward_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 3))[0])
    init_overall_Y_post_reward = np.sum([post_reward_hit_rate_init,post_reward_hit_rate_choice,post_reward_hit_rate_reward])
    plt.subplot(414)
    plt.bar([1,2,3], [post_reward_hit_rate_init/init_overall_Y_post_reward,post_reward_hit_rate_choice/init_overall_Y_post_reward,post_reward_hit_rate_reward/init_overall_Y_post_reward], tick_label = ['Pre_Initiation', 'Init-Choice', 'Reward'])
    plt.title('Reward - Post Reward Errors')
    plt.tight_layout()

    
  
    

def plot():
     
    correct_list_within_PFC, correct_list_between_PFC, y_pred_class_between_t_1_2_PFC, Y_PFC = classifier_pseudo_simultaneous(data_PFC, session)
    correct_list_within_HP, correct_list_between_HP, y_pred_class_between_t_1_2_HP, Y_HP = classifier_pseudo_simultaneous(data_HP, session)

    plt.figure(2)
    sns.barplot(data=[correct_list_within_PFC,correct_list_between_PFC,correct_list_within_HP,correct_list_between_HP], capsize=.1, ci="sd",  palette="Blues_d")
    #sns.swarmplot(data=[correct_list_within_PFC,correct_list_between_PFC,correct_list_within_HP,correct_list_between_HP], color="0", alpha=.35)
    plt.xticks(np.arange(4),('PFC within task', 'PFC between tasks', 'HP within task', 'HP between tasks'))
    plt.ylabel('% correct')
    plt.title('SVM')
          