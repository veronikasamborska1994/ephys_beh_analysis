#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:04:28 2019

@author: veronikasamborska
"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import create_data_arrays_for_tim as cda
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import svm

import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt



#data_PFC = cda.tim_create_mat(experiment_aligned_PFC, experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC, 'PFC') 
#data_HP = cda.tim_create_mat(experiment_aligned_HP, experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP, experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP, 'HP')


def classifier(data):    
    
    y = data['DM']
    X = data['Data']
    length = []
    correct_list_within = []
    correct_list_between = []
    all_y_within_1 = []
    all_y_between_1 = []
    all_Ys = []
   

    for s, session in enumerate(X):
        
        # Design matrix for the session
        DM = y[s]
     
        firing_rates_all_time = X[s]
        
        # Tasks indicies 
        task = DM[:,4]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]
        task_3 = np.where(task == 3)[0]
        
        # Find the maximum length of any of the tasks in a session
        length.append(len(task_1))
        length.append(len(task_2))
        length.append(len(task_3))     
        min_trials_in_task = int(np.min(length)/2)
        
        # Select the first min_trials_in_task in task one
        firing_rates_mean_task_1_1 = firing_rates_all_time[:min_trials_in_task,:]
        # Select the last min_trials_in_task in task one
        firing_rates_mean_task_1_2 = firing_rates_all_time[task_2[0]-1-min_trials_in_task:task_2[0]-1,:]
       
        # Select the first min_trials_in_task in task two
        firing_rates_mean_task_2_1 = firing_rates_all_time[task_2[0]:task_2[0]+min_trials_in_task,:]
        firing_rates_mean_task_2_2 = firing_rates_all_time[task_3[0]-1-min_trials_in_task:task_3[0]-1,:]

        # Select the first min_trials_in_task in task three
        firing_rates_mean_task_3_1 = firing_rates_all_time[task_3[0]:task_3[0]+min_trials_in_task,:]
        firing_rates_mean_task_3_2 = firing_rates_all_time[task_3[-1]-min_trials_in_task:task_3[-1],:]

        
        # Finding the indices of initiations, choices and rewards 
        n_time = firing_rates_mean_task_1_1.shape[2]
        session = experiment_aligned_HP[0]
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()
        reward_time = initiate_choice_t[-2] +250
                
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
        ind_reward = (np.abs(t_out-reward_time)).argmin()

        # Finding the indices of initiations, choices and rewards 
#        
        bins_before_init = np.zeros(initiation)
        bins_between_init_choice = np.ones(ind_choice-initiation)
        bins_between_choice_reward = np.ones(ind_reward-ind_choice)
        bins_between_choice_reward[:]= 2
        bins_between_post_reward = np.ones(n_time-ind_reward)
        bins_between_post_reward[:] = 3
        
        bins = np.hstack((bins_before_init,bins_between_init_choice,bins_between_choice_reward,bins_between_post_reward))
    
#        
#        bins_before_init = np.zeros(initiation-5)
#        bins_between_around_init = np.ones(5+int((ind_choice-initiation)/2))
#        bins_between_around_choice = np.ones(int((ind_choice-initiation)/2+(ind_reward-ind_choice)/2))
#        bins_between_around_choice[:]= 2
#        bins_between_around_reward = np.ones(int((ind_reward-ind_choice)/2 + (n_time-ind_reward)/2))
#        bins_between_around_reward[:] = 3
#        trial_n = np.hstack((bins_before_init,bins_between_around_init,bins_between_around_choice,bins_between_around_reward))
#        bins_post_reward = np.ones(n_time-len(trial_n))
#        bins_post_reward[:] = 4
#        bins = np.hstack((bins_before_init,bins_between_around_init,bins_between_around_choice,bins_between_around_reward,bins_post_reward))
#        
       
       
        firing_rates_mean_1_1 = np.concatenate(firing_rates_mean_task_1_1, axis = 1)
        firing_rates_mean_1_2 = np.concatenate(firing_rates_mean_task_1_2, axis = 1)
        firing_rates_mean_2_1 = np.concatenate(firing_rates_mean_task_2_1, axis = 1)
        firing_rates_mean_2_2 = np.concatenate(firing_rates_mean_task_2_2, axis = 1)
        firing_rates_mean_3_1 = np.concatenate(firing_rates_mean_task_3_1, axis = 1)
        firing_rates_mean_3_2 = np.concatenate(firing_rates_mean_task_3_2, axis = 1)

        l = firing_rates_mean_1_1.shape[1]
        # Creating a vector which identifies trial stage in the firing rate vector
        Y = np.tile(bins,int(l/len(bins)))
        
        
        #model_nb = svm.SVC(gamma='scale')
        model_nb = LinearDiscriminantAnalysis()
        
        model_nb.fit(np.transpose(firing_rates_mean_1_2),Y)  
        y_pred_class_between_t_1_2 = model_nb.predict(np.transpose(firing_rates_mean_2_1))
        
        model_nb.fit(np.transpose(firing_rates_mean_1_1),Y)     
        y_pred_class_within_t_1_2 = model_nb.predict(np.transpose(firing_rates_mean_1_2))
       
        model_nb.fit(np.transpose(firing_rates_mean_2_2),Y)
        y_pred_class_between_t_2_3 = model_nb.predict(np.transpose(firing_rates_mean_3_1))
        
        model_nb.fit(np.transpose(firing_rates_mean_2_1),Y)
        y_pred_class_within_t_2_3 = model_nb.predict(np.transpose(firing_rates_mean_2_2))

        model_nb.fit(np.transpose(firing_rates_mean_3_1),Y)
        y_pred_class_within_t_3 = model_nb.predict(np.transpose(firing_rates_mean_3_2))

        correct_within_t_1 = metrics.accuracy_score(Y, y_pred_class_within_t_1_2)
        correct_between_t_1 = metrics.accuracy_score(Y, y_pred_class_between_t_1_2)
        
        correct_within_t_2 = metrics.accuracy_score(Y, y_pred_class_within_t_2_3)
        correct_between_t_2 = metrics.accuracy_score(Y, y_pred_class_between_t_2_3)

        correct_within_t_3 = metrics.accuracy_score(Y, y_pred_class_within_t_3)
        
        correct_list_within.append(correct_within_t_1)
        correct_list_within.append(correct_within_t_2)
        correct_list_within.append(correct_within_t_3)

        correct_list_between.append(correct_between_t_1)
        correct_list_between.append(correct_between_t_2)
        all_y_within_1.append(y_pred_class_within_t_1_2)
        all_y_between_1.append(y_pred_class_between_t_1_2)
        all_Ys.append(Y)

    return correct_list_within, correct_list_between,all_y_within_1,all_y_between_1,all_Ys


def confusion_m(y_pred_class,Y,title):
    
    init_overall_rate = len(np.where(Y == y_pred_class)[0])
    init_hit_rate = len(np.where((y_pred_class == 0)& (Y == 0))[0])/init_overall_rate
    choice_hit_rate = len(np.where((y_pred_class == 1)& (Y == 1))[0])/init_overall_rate    
    reward_hit_rate = len(np.where((y_pred_class == 2)& (Y == 2))[0])/init_overall_rate
    post_reward_hit_rate = len(np.where((y_pred_class == 3)& (Y == 3))[0])/init_overall_rate
    
    
    init_overall_non_rate = len(np.where(Y != y_pred_class)[0])
    init_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 0))[0])/init_overall_non_rate
    init_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 0))[0])/init_overall_non_rate
    init_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 0))[0])/init_overall_non_rate

    choice_hit_rate_init = len(np.where((y_pred_class == 0)& (Y == 1))[0])/init_overall_non_rate
    choice_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 1))[0])/init_overall_non_rate
    choice_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 1))[0])/init_overall_non_rate

    reward_hit_rate_init = len(np.where((y_pred_class == 0)& (Y == 2))[0])/init_overall_non_rate
    reward_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 2))[0])/init_overall_non_rate
    reward_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 2))[0])/init_overall_non_rate

    post_reward_hit_rate_init = len(np.where((y_pred_class == 0)& (Y == 3))[0])/init_overall_non_rate
    post_reward_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 3))[0])/init_overall_non_rate
    post_reward_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 3))[0])/init_overall_non_rate
    
    confusion_m = np.zeros((4,4))
    confusion_m[0,0] = init_hit_rate
    confusion_m[1,1] = choice_hit_rate
    confusion_m[2,2] = reward_hit_rate
    confusion_m[3,3] = post_reward_hit_rate
    
    confusion_m[0,1] = init_hit_rate_choice
    confusion_m[0,2] = init_hit_rate_reward
    confusion_m[0,3] = init_hit_rate_post_reward

    confusion_m[1,0] = choice_hit_rate_init
    confusion_m[1,2] = choice_hit_rate_reward
    confusion_m[1,3] = choice_hit_rate_post_reward

    confusion_m[2,0] = reward_hit_rate_init
    confusion_m[2,1] = reward_hit_rate_choice
    confusion_m[2,3] = reward_hit_rate_post_reward

    confusion_m[3,0] = post_reward_hit_rate_init 
    confusion_m[3,1] = post_reward_hit_rate_choice
    confusion_m[3,2] = post_reward_hit_rate_reward

    
    plt.imshow(confusion_m)
    plt.xticks((0,1,2,3),['Pre-Init', 'Init-Choice', 'Choice-Reward', 'Post-Reward'])
    plt.yticks((0,1,2,3),['Pre-Init', 'Init-Choice', 'Choice-Reward', 'Post-Reward'])
    plt.colorbar()
    plt.title(title)


def plot(data_PFC, data_HP):
    correct_list_within_PFC, correct_list_between_PFC,y_pred_class_within_PFC,y_pred_class_between_PFC,Y_PFC = classifier(data_PFC)
    correct_list_within_HP, correct_list_between_HP,y_pred_class_within_HP,y_pred_class_between_HP,Y_HP= classifier(data_HP)
    
    Y_PFC = np.concatenate(Y_PFC, 0)
    y_pred_class_within_PFC = np.concatenate(y_pred_class_within_PFC, 0)
    y_pred_class_between_PFC = np.concatenate(y_pred_class_between_PFC, 0)

    Y_HP = np.concatenate(Y_HP, 0)
    y_pred_class_within_HP = np.concatenate(y_pred_class_within_HP, 0)
    y_pred_class_between_HP = np.concatenate(y_pred_class_between_HP, 0)

    mean_within_PFC = np.mean(correct_list_within_PFC)
    std_err_mean_within_PFC = np.std(correct_list_within_PFC)/np.sqrt(len(correct_list_within_PFC))
    mean_between_PFC = np.mean(correct_list_between_PFC)
    std_err_mean_between_PFC =  np.std(correct_list_between_PFC)/np.sqrt(len(correct_list_between_PFC))
    mean_within_HP = np.mean(correct_list_within_HP)
    std_err_mean_within_HP = np.std(correct_list_within_HP)/np.sqrt(len(correct_list_within_HP))
    mean_between_HP = np.mean(correct_list_between_HP)
    std_err_mean_between_HP = np.std(correct_list_between_HP)/np.sqrt(len(correct_list_between_HP))
    
    
    sns.barplot(data=[correct_list_within_PFC,correct_list_between_PFC,correct_list_within_HP,correct_list_between_HP], capsize=.1, ci="sd",  palette="Blues_d")
    sns.swarmplot(data=[correct_list_within_PFC,correct_list_between_PFC,correct_list_within_HP,correct_list_between_HP], color="0", alpha=.35)
    plt.xticks(np.arange(4),('PFC within task', 'PFC between tasks', 'HP within task', 'HP between tasks'))
    plt.ylabel('% correct')
    plt.title('LDA')
          
    