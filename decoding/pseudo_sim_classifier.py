#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:18:33 2019

@author: veronikasamborska
"""
# =============================================================================
# Decoding I when it becomes B (same location) --> classifying as 
# location in space or location in task? Plotting confusion matrices to see which errors it's making
# =============================================================================

import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression



sns.set_style("white") 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

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
     
     int_init_same = np.where(i_pokes !=  init_choice_port)[0]
    
     return ind_choice, ind_init, int_init_same


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
    
        ind_choices, ind_init, int_init_same= search_for_tasks_where_init_becomes_choice(DM)
      
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
        #bins = np.arange(n_time)
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
    cnf_matrix = metrics.confusion_matrix(Y,y_pred_class_between_t_1)/all_sessions_1_1.shape[1]
    
    plt.figure()
    plt.imshow(cnf_matrix)
    plt.colorbar()
    
    return correct_list_within, correct_list_between,y_pred_class_between_t_1, Y

def permutation_decoder():
    activity = np.vstack((firing_t1,firing_t2))
    activity_diff = np.mean(firing_t1, axis = 0) - np.mean(firing_t2, axis = 0)
    n_trials  = int(activity.shape[0]/2)
    activity_perm = np.zeros((n_perm, firing_t1.shape[1]))
    activity_max = np.zeros(n_perm)
    
    for i in range(n_perm):
        np.random.shuffle(activity) # Shuffle A / B trials (axis 0 only).
        activity_perm[i,:] = (np.mean(activity[:n_trials,:],0) -
                                         np.mean(activity[n_trials:,:],0))
        activity_max[i] = np.max(activity_perm[i,:])
       
    p = np.percentile(activity_perm,99, axis = 0)
    p_max = np.percentile(activity_max, 99)
    x_max = np.zeros(len(p))
    x_max[:] = p_max
    
    
def confusion_mat(y_pred_class,Y,title):    
    
    # Init Errors

    init_hit_rate_choice = len(np.where((y_pred_class == 1)& (Y == 0))[0])
    init_hit_rate_reward = len(np.where((y_pred_class == 2)& (Y == 0))[0])
    init_hit_rate_post_reward = len(np.where((y_pred_class == 3)& (Y == 0))[0])
    init_overall_Y = np.sum([init_hit_rate_choice,init_hit_rate_reward,init_hit_rate_post_reward])
    sns.set(palette="RdGy")
    sns.set_style("white") 
    plt.figure(figsize=[4,12])
    plt.subplot(411)
    plt.bar([1,2,3], [init_hit_rate_choice/init_overall_Y,init_hit_rate_reward/init_overall_Y,init_hit_rate_post_reward/init_overall_Y], tick_label = ['Initiation-Choice', 'Choice-Reward', 'Post Reward'])
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
    plt.bar([1,2,3], [post_reward_hit_rate_init/init_overall_Y_post_reward,post_reward_hit_rate_choice/init_overall_Y_post_reward,post_reward_hit_rate_reward/init_overall_Y_post_reward], tick_label = ['Pre_Initiation', 'Init-Choice', 'Choice-Reward'])
    plt.title('Reward - Post Reward Errors')
    plt.tight_layout()


def classifier_pseudo_simultaneous_init_vs_choice(data, session, condition = 'Space', title = 'HP', i = 121):    
    
    y = data['DM']
    X = data['Data']

    all_sessions_task_1 = []
    all_sessions_task_2 = []
    pokes_init_b = []
    
    pokes_same_space = []   
    pokes_change_init = []    
    poke_change_b = []
    
    for s, sess in enumerate(X):
        
        firing_rates_all_time = X[s]

        # Design matrix for the session
        DM = y[s]
      
        choices =  DM[:,1]

        choices_b = np.where(choices == 0)[0]
        choices_a = np.where(choices == 1)[0]
        
        b_pokes = DM[:,6]
        i_pokes = DM[:,7]
        a_pokes = DM[:,5]
        
        ind_choices, ind_init,int_init_same = search_for_tasks_where_init_becomes_choice(DM)

       
        i_becomes_b = i_pokes[ind_init][0]
        b_when_i = b_pokes[ind_init][0]
        i_when_b = i_pokes[ind_choices][0]
        
        if [i_becomes_b,b_when_i,i_when_b,i_becomes_b] not in pokes_init_b:
            pokes_init_b.append([i_becomes_b,b_when_i,i_when_b,i_becomes_b])
    
        i_poke_space = i_pokes[int_init_same][0]
        a_poke_space = a_pokes[0]
        
        if [i_poke_space,a_poke_space] not in pokes_same_space:
            pokes_same_space.append([i_poke_space,a_poke_space])
        
        a_i = np.intersect1d(choices_a, int_init_same)
        a_i_change = np.intersect1d(choices_a, ind_init)
        
        init_t1 = i_pokes[a_i][0]
        init_t2 = i_pokes[a_i_change][0]
        
        if [init_t1,a_poke_space, init_t2,a_poke_space] not in pokes_change_init:
            pokes_change_init.append([init_t1,a_poke_space,init_t2,a_poke_space])
    
        b_i = np.intersect1d(choices_b, int_init_same)
        init_b = i_pokes[b_i][0]
        change_b_1 = np.unique(b_pokes[b_i])[0]
        change_b_2 = np.unique(b_pokes[b_i])[1]

        if [init_b,change_b_1, init_b,change_b_2] not in poke_change_b:
            poke_change_b.append([init_b,change_b_1,init_b,change_b_2 ])
       
        min_trials_in_task = 24
    
        # Finding the indices of initiations, choices and rewards 
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()        
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
        
        ind_around_init = np.arange(initiation-5, initiation+5)
        ind_around_choice = np.arange(ind_choice-5, ind_choice+5)
        
        reward_time = initiate_choice_t[-2] +250
        ind_reward = (np.abs(t_out-reward_time)).argmin()
        ind_around_reward =  np.arange(ind_reward, ind_reward+10)

        if condition == 'Initiation_B':
            
            # Indicies of B choices when Initiation is B in another task 
            i_vs_b_I = np.intersect1d(ind_init,choices_b)  
            # Indicies of B choices B is I in another task 
            b_vs_i_B = np.intersect1d(ind_choices, choices_b) 
            
            fr_i_vs_b_i = firing_rates_all_time[i_vs_b_I,:,:] 
            fr_i_vs_b_I = fr_i_vs_b_i[:,:,ind_around_init] # Initiation Times
            fr_i_vs_b_IR = fr_i_vs_b_i[:,:,ind_around_reward] # Initiation Times
            
            fr_i_vs_b_b = firing_rates_all_time[i_vs_b_I,:,:]
            fr_i_vs_b_B = fr_i_vs_b_b[:,:,ind_around_choice] # Choice Times
            fr_i_vs_b_BR = fr_i_vs_b_b[:,:,ind_around_reward] # Initiation Times

            fr_b_vs_i_i = firing_rates_all_time[b_vs_i_B,:,:]
            fr_b_vs_i_I = fr_b_vs_i_i[:,:,ind_around_init] # Initiation Times
            fr_i_vs_i_IR = fr_b_vs_i_i[:,:,ind_around_reward] # Initiation Times

            fr_b_vs_i_b = firing_rates_all_time[b_vs_i_B,:,:]
            fr_b_vs_i_B = fr_b_vs_i_b[:,:,ind_around_choice] # Choice Times
            fr_i_vs_i_BR = fr_b_vs_i_b[:,:,ind_around_reward] # Initiation Times
       
            
            fr_i_vs_b_I = fr_i_vs_b_I[:min_trials_in_task]
            fr_i_vs_b_IR = fr_i_vs_b_IR[:min_trials_in_task]
            fr_i_vs_b_B = fr_i_vs_b_B[:min_trials_in_task]
            fr_i_vs_b_BR = fr_i_vs_b_BR[:min_trials_in_task]
            fr_b_vs_i_I = fr_b_vs_i_I[:min_trials_in_task]
            fr_i_vs_i_IR = fr_i_vs_i_IR[:min_trials_in_task]
            fr_b_vs_i_B = fr_b_vs_i_B[:min_trials_in_task]
            fr_i_vs_i_BR = fr_i_vs_i_BR[:min_trials_in_task] # Initiation Times

           
            y_decode_i_a = np.zeros(fr_i_vs_b_I.shape[2])
            y_decode_a = np.ones(fr_i_vs_b_I.shape[2])
    
            
            fr_i_vs_b = np.concatenate((fr_i_vs_b_I,fr_i_vs_b_IR, fr_i_vs_b_B,fr_i_vs_b_BR), axis =  0)
            fr_b_vs_i = np.concatenate((fr_b_vs_i_I,fr_i_vs_i_IR, fr_b_vs_i_B,fr_i_vs_i_BR), axis = 0)
            
            all_sessions_task_1.append(np.concatenate(fr_i_vs_b, axis = 1))
            all_sessions_task_2.append(np.concatenate(fr_b_vs_i, axis = 1))
          
            y_decode_i_a = np.zeros(int(np.concatenate(fr_i_vs_b, axis = 1).shape[1]/2))
            y_decode_a = np.ones(int(np.concatenate(fr_i_vs_b, axis = 1).shape[1]/2))
    
            Y = np.hstack((y_decode_i_a,y_decode_a))   


        elif  condition == 'Space':
            
            task = DM[:,4]
            task_1 = np.where(task == 1)[0]
            task_2 = np.where(task == 2)[0]
            task_3 = np.where(task == 3)[0]

            a_i = np.intersect1d(choices_a, int_init_same)
            
            task_1_a = np.intersect1d(task_1, a_i)
            task_2_a = np.intersect1d(task_2, a_i)
            task_3_a = np.intersect1d(task_3, a_i)
            if task_3_a.shape[0] == 0:
                a_1 = task_1_a
                a_2 = task_2_a

            elif task_2_a.shape[0] == 0:
                a_1 = task_1_a
                a_2 = task_3_a
                
            elif task_1_a.shape[0] == 0:
                a_1 = task_2_a
                a_2 = task_3_a

            fr_i_vs_a_I = firing_rates_all_time[a_1,:,:] 
            fr_i_vs_a_I = fr_i_vs_a_I[:,:,ind_around_init] # Initiation Times
            
            fr_i_vs_a_A = firing_rates_all_time[a_1,:,:]
            fr_i_vs_a_A = fr_i_vs_a_A[:,:,ind_around_choice] # Choice Times
            
            fr_a_vs_i_I = firing_rates_all_time[a_2,:,:]
            fr_a_vs_i_I = fr_a_vs_i_I[:,:,ind_around_init] # Initiation Times
              
            fr_a_vs_i_A = firing_rates_all_time[a_2,:,:]
            fr_a_vs_i_A = fr_a_vs_i_A[:,:,ind_around_choice] # Choice Times
                    
            
            fr_i_vs_a_I = fr_i_vs_a_I[:min_trials_in_task]
            fr_i_vs_a_A = fr_i_vs_a_A[:min_trials_in_task]
            fr_a_vs_i_I = fr_a_vs_i_I[:min_trials_in_task]
            fr_a_vs_i_A = fr_a_vs_i_A[:min_trials_in_task]
           
            
            fr_i_vs_a = np.concatenate((fr_i_vs_a_I, fr_i_vs_a_A), axis =  0)
            fr_a_vs_i = np.concatenate((fr_a_vs_i_I, fr_a_vs_i_A), axis = 0)
            
            all_sessions_task_1.append(np.concatenate(fr_i_vs_a, axis = 1))
            all_sessions_task_2.append(np.concatenate(fr_a_vs_i, axis = 1))
            
            y_decode_i_a = np.zeros(int(np.concatenate(fr_i_vs_a, axis = 1).shape[1]/2))
            y_decode_a = np.ones(int(np.concatenate(fr_i_vs_a, axis = 1).shape[1]/2))
    
            Y = np.hstack((y_decode_i_a,y_decode_a))       
        
        elif  condition == 'Init_Change':
           
            a_i = np.intersect1d(choices_a, int_init_same)
            a_i_change = np.intersect1d(choices_a, ind_init)

            fr_i_vs_a_I = firing_rates_all_time[a_i,:,:] 
            fr_i_vs_a_I = fr_i_vs_a_I[:,:,ind_around_init] # Initiation Times
            
            fr_i_vs_a_A = firing_rates_all_time[a_i,:,:]
            fr_i_vs_a_A = fr_i_vs_a_A[:,:,ind_around_choice] # Choice Times
            
            fr_a_vs_i_I = firing_rates_all_time[a_i_change,:,:]
            fr_a_vs_i_I = fr_a_vs_i_I[:,:,ind_around_init] # Initiation Times
              
            fr_a_vs_i_A = firing_rates_all_time[a_i_change,:,:]
            fr_a_vs_i_A = fr_a_vs_i_A[:,:,ind_around_choice] # Choice Times
                    
            
            fr_i_vs_a_I = fr_i_vs_a_I[:min_trials_in_task]
            fr_i_vs_a_A = fr_i_vs_a_A[:min_trials_in_task]
            fr_a_vs_i_I = fr_a_vs_i_I[:min_trials_in_task]
            fr_a_vs_i_A = fr_a_vs_i_A[:min_trials_in_task]
           
            
            fr_i_vs_a = np.concatenate((fr_i_vs_a_I, fr_i_vs_a_A), axis =  0)
            fr_a_vs_i = np.concatenate((fr_a_vs_i_I, fr_a_vs_i_A), axis = 0)
            
            all_sessions_task_1.append(np.concatenate(fr_i_vs_a, axis = 1))
            all_sessions_task_2.append(np.concatenate(fr_a_vs_i, axis = 1))
            
            y_decode_i_a = np.zeros(int(np.concatenate(fr_i_vs_a, axis = 1).shape[1]/2))
            y_decode_a = np.ones(int(np.concatenate(fr_i_vs_a, axis = 1).shape[1]/2))
    
            Y = np.hstack((y_decode_i_a,y_decode_a))      
            
        elif  condition == 'B_Change':
            
            task = DM[:,4]
            task_1 = np.where(task == 1)[0]
            task_2 = np.where(task == 2)[0]
            task_3 = np.where(task == 3)[0]

            b_i = np.intersect1d(choices_b, int_init_same)
            
            task_1_b = np.intersect1d(task_1, b_i)
            task_2_b = np.intersect1d(task_2, b_i)
            task_3_b = np.intersect1d(task_3, b_i)
            
            if task_3_b.shape[0] == 0:
                b_1 = task_1_b
                b_2 = task_2_b

            elif task_2_b.shape[0] == 0:
                b_1 = task_1_b
                b_2 = task_3_b
                
            elif task_1_b.shape[0] == 0:
                b_1 = task_2_b
                b_2 = task_3_b

            fr_i_vs_b_I = firing_rates_all_time[b_1,:,:] 
            fr_i_vs_b_I = fr_i_vs_b_I[:,:,ind_around_init] # Initiation Times
            
            fr_i_vs_b_B = firing_rates_all_time[b_1,:,:]
            fr_i_vs_b_B = fr_i_vs_b_B[:,:,ind_around_choice] # Choice Times
            
            fr_b_vs_b_I = firing_rates_all_time[b_2,:,:]
            fr_b_vs_b_I = fr_b_vs_b_I[:,:,ind_around_init] # Initiation Times
              
            fr_b_vs_i_B = firing_rates_all_time[b_2,:,:]
            fr_b_vs_i_B = fr_b_vs_i_B[:,:,ind_around_choice] # Choice Times
            
            
            fr_i_vs_b_I = fr_i_vs_b_I[:min_trials_in_task]
            fr_i_vs_b_B = fr_i_vs_b_B[:min_trials_in_task]
            fr_b_vs_b_I = fr_b_vs_b_I[:min_trials_in_task]
            fr_b_vs_i_B = fr_b_vs_i_B[:min_trials_in_task]
           
            
            fr_i_vs_b = np.concatenate((fr_i_vs_b_I, fr_i_vs_b_B), axis =  0)
            fr_b_vs_i = np.concatenate((fr_b_vs_b_I, fr_b_vs_i_B), axis = 0)
            
            all_sessions_task_1.append(np.concatenate(fr_i_vs_b, axis = 1))
            all_sessions_task_2.append(np.concatenate(fr_b_vs_i, axis = 1))
            
            y_decode_i_a = np.zeros(int(np.concatenate(fr_i_vs_b, axis = 1).shape[1]/2))
            y_decode_a = np.ones(int(np.concatenate(fr_i_vs_b, axis = 1).shape[1]/2))
    
            Y = np.hstack((y_decode_i_a,y_decode_a))   
            
            
    model_nb = svm.SVC(gamma='scale')
    all_sessions_task_1 = np.concatenate(all_sessions_task_1, axis = 0)
    all_sessions_task_2 = np.concatenate(all_sessions_task_2, axis = 0)

    #model_nb = LogisticRegression()
    model_nb.fit(np.transpose(all_sessions_task_1),Y)  
    y_pred = model_nb.predict((np.transpose(all_sessions_task_2)))
  
    accuracy = metrics.accuracy_score(Y,y_pred)    
    conf_m = metrics.confusion_matrix(Y, y_pred)
    c_m = conf_m/all_sessions_task_1.shape[1]
    
    plt.subplot(i)
    plt.imshow(c_m)
    plt.colorbar()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks([0,1],['Init','Choice'])
    plt.yticks([0,1],['Init','Choice'])
    plt.title(title)
 
    return accuracy, c_m, y_pred, Y,all_sessions_task_1, all_sessions_task_2, pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b


def plot_init_choice():   
    accuracy_H_sp, c_m_H_sp,y_pred_H_sp,  Y_H_sp, all_sessions_task_1_H_sp,\
    all_sessions_task_2_H_sp, pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_HP, session, condition = 'Space', title = 'HP Same A and B', i = 241)
    accuracy_H_i, c_m_H_i,y_pred_H_i,  Y_H_i, all_sessions_task_1_H_i, all_sessions_task_2_H_i,\
    pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_HP, session, condition = 'Init_Change', title = 'HP Different Init', i = 242)
    accuracy_H_b, c_m_H_b,y_pred_H_b,  Y_H_b,all_sessions_task_1_H_b, all_sessions_task_2_H_b,\
    pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_HP, session, condition = 'B_Change', title = 'HP Different B', i = 243)
    accuracy_H_ib, c_m_H_ib,y_pred_H_ib,  Y_H_ib, all_sessions_task_1_H_ib, all_sessions_task_2_H_ib,\
    pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_HP, session, condition = 'Initiation_B', title = 'HP Init becomes B', i = 244)


    accuracy_PFC_sp, c_m_PFC_i,y_pred_PFC_i,  Y_PFC_i, all_sessions_task_1_PFC_sp, all_sessions_task_2_PFC_sp,\
    pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_PFC, session, condition = 'Space', title = 'PFC Same A and B', i = 245)
    accuracy_PFC_i, c_m_PFC_i,y_pred_PFC_i,  Y_PFC_i,  all_sessions_task_1_PFC_i, all_sessions_task_2_PFC_i,\
    pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_PFC, session, condition = 'Init_Change', title = 'PFC Different Init', i = 246)
    accuracy_PFC_b, c_m_PFC_b,y_pred_PFC_b,  Y_PFC_b, all_sessions_task_1_PFC_b, all_sessions_task_2_PFC_b,\
    pokes_init_b, pokes_same_space, pokes_change_init,poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_PFC, session, condition = 'B_Change', title = 'PFC Different B', i = 247)
    accuracy_PFC_ib, c_m_PFC_b,y_pred_PFC_ib,  Y_PFC_ib, all_sessions_task_1_PFC_ib, all_sessions_task_2_PFC_ib,\
    pokes_init_b, pokes_same_space, pokes_change_init, poke_change_b = classifier_pseudo_simultaneous_init_vs_choice(data_PFC, session, condition = 'Initiation_B', title = 'PFC Init becomes B', i = 248)

    plt.subplot(121)
    plt.bar([1,2,3,4],[accuracy_H_sp,accuracy_H_i,accuracy_H_b,accuracy_H_ib])
    plt.xticks([1,2,3,4],['Same Space','Different I','Different B', 'I becomes B'])
    plt.title('HP')
    plt.ylabel('Accuracy')

    plt.subplot(122)
    plt.bar([1,2,3,4],[accuracy_PFC_sp,accuracy_PFC_i,accuracy_PFC_b,accuracy_PFC_ib])
    plt.xticks([1,2,3,4],['Same Space','Different I','Different B', 'I becomes B'])
    plt.title('PFC')


def plot_decoding_conf_m():
    space_HP = np.concatenate((all_sessions_task_1_H_sp, all_sessions_task_2_H_sp), axis = 1) 
   
    different_i_HP = np.concatenate((all_sessions_task_1_H_i, all_sessions_task_2_H_i), axis = 1) 
    
    different_b_HP = np.concatenate((all_sessions_task_1_H_b, all_sessions_task_2_H_b), axis = 1) 
   
    different_ib_HP = np.concatenate((all_sessions_task_1_H_ib, all_sessions_task_2_H_ib), axis = 1) 

   

    space_PFC = np.concatenate((all_sessions_task_1_PFC_sp, all_sessions_task_2_PFC_sp), axis = 1) 
   
    different_i_PFC = np.concatenate((all_sessions_task_1_PFC_i, all_sessions_task_2_PFC_i), axis = 1) 
    
    different_b_PFC = np.concatenate((all_sessions_task_1_PFC_b, all_sessions_task_2_PFC_b), axis = 1) 
   
    different_ib_PFC = np.concatenate((all_sessions_task_1_PFC_ib, all_sessions_task_2_PFC_ib), axis = 1) 


    corr_space_HP = np.corrcoef(space_HP.T,space_HP.T)
    
    corr_ib_HP = np.corrcoef(different_ib_HP.T,different_ib_HP.T)
    
    
    ## Plots averaged across cells 
    
def plot_corr_cells(all_sessions_task_1,all_sessions_task_2, title, ind_c_i, ind_c_e,ind_r):
    fig = plt.figure(1, figsize=(6, 25))

    grid = plt.GridSpec(4, 12, hspace=0.5, wspace= 0.3)
    init_H_sp_1 = all_sessions_task_1[:,:240]
    init_H_sp_1 = np.mean(init_H_sp_1.reshape(all_sessions_task_1.shape[0],24,10), axis = 1)
    choice_H_sp_1 = all_sessions_task_1[:,240:]
    choice_H_sp_1 = np.mean(choice_H_sp_1.reshape(all_sessions_task_1.shape[0],24,10), axis = 1)

    s_HP_mean_1 = np.hstack((init_H_sp_1,choice_H_sp_1))
    
    init_H_sp_2 = all_sessions_task_2[:,:240]
    init_H_sp_2 = np.mean(init_H_sp_2.reshape(init_H_sp_2.shape[0],24,10), axis = 1)
    choice_H_sp_2 = all_sessions_task_2[:,240:]
    choice_H_sp_2 = np.mean(choice_H_sp_2.reshape(all_sessions_task_1.shape[0],24,10), axis = 1)
    s_HP_mean_2 = np.hstack((init_H_sp_2,choice_H_sp_2))

    corr_space_HP = np.corrcoef(s_HP_mean_1.T,s_HP_mean_2.T)

    fig.add_subplot(grid[ind_r,ind_c_i:ind_c_e])
    plt.imshow(corr_space_HP)
    plt.xticks([5,15,25,35],['I1','C1','I2', 'C2'])
    plt.yticks([5,15,25,35],['I1','C1','I2', 'C2'])
    #plt.axis('off')

    #plt.colorbar()
    plt.title(title)

def pl():
    
    plot_corr_cells(all_sessions_task_1_H_sp,all_sessions_task_2_H_sp, 'HP Same A and I', ind_c_i = 0, ind_c_e = 3, ind_r =0)
    plot_corr_cells(all_sessions_task_1_H_i,all_sessions_task_2_H_i, 'HP Different Init',ind_c_i = 3, ind_c_e = 6, ind_r =0)
    plot_corr_cells(all_sessions_task_1_H_b,all_sessions_task_2_H_b, 'HP Different B',ind_c_i = 6, ind_c_e = 9, ind_r =0)
    plot_corr_cells(all_sessions_task_1_H_ib,all_sessions_task_2_H_ib, 'HP Different Init becomes B', ind_c_i = 9, ind_c_e = 12, ind_r =0)

    plot_corr_cells(all_sessions_task_1_PFC_sp,all_sessions_task_2_PFC_sp, 'PFC Same A and I', ind_c_i = 0, ind_c_e = 3, ind_r =1)
    plot_corr_cells(all_sessions_task_1_PFC_i,all_sessions_task_2_PFC_i, 'PFC Different Init',ind_c_i = 3, ind_c_e = 6, ind_r =1)
    plot_corr_cells(all_sessions_task_1_PFC_b,all_sessions_task_2_PFC_b, 'PFC Different B', ind_c_i = 6, ind_c_e = 9, ind_r =1)
    plot_corr_cells(all_sessions_task_1_PFC_ib,all_sessions_task_2_PFC_ib, 'PFC Different Init becomes B',  ind_c_i = 9, ind_c_e = 12, ind_r =1)
    
    
    # Plotting configurations
    
    x_all = [132,232,232,332,332,432,432,532]
    y_all = [2.8,3.8,1.8,4.8,0.8,3.8,1.8,2.8]
    
    fig = plt.figure(1)
    a = 0
    for c,ind in enumerate(pokes_same_space):  
        if c == 0:
            a = 0
        elif c > 0:
            a += 1
        for i, poke in enumerate(ind):
            fig.add_subplot(12,4,a*4+25)
            plt.axis('off')
            plt.scatter(x_all,y_all, color = 'grey', s = 10, alpha = 0.2)

            if i == 0:

                if poke == 1:
                    x_coords = 332
                    y_coords = 4.8
                elif poke == 9:
                    x_coords = 332
                    y_coords = 0.8
                if i == 0:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5)
    
            elif i == 1:
                if poke == 4:
                    x_coords = 132
                    y_coords = 2.8
                elif poke == 6:
                    x_coords = 532
                    y_coords = 2.8
                if i == 1:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5, alpha = 0.5)
     
    plt.axis('off')
    a = 0
    for c,ind in enumerate(pokes_change_init):   
        if c == 0:
            a = 0
        elif c > 0:
            a += 1
        for i, poke in enumerate(ind):
            fig.add_subplot(12,4,a*4+26)
            plt.axis('off')

            plt.scatter(x_all,y_all, color = 'grey', s = 10, alpha = 0.2)

            if i == 0 or i == 2:
                if poke == 1:
                    x_coords = 332
                    y_coords = 4.8
                elif poke == 9:
                    x_coords = 332
                    y_coords = 0.8
                if i == 0:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5)
                elif i == 2:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5, alpha = 0.5)
            elif i == 1 or i == 3:
                if poke == 4:
                    x_coords = 132
                    y_coords = 2.8
                elif poke == 6:
                    x_coords = 532
                    y_coords = 2.8
                if i == 1:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5)
                elif i == 3:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5, alpha = 0.5)
    
    
    # Configs for change of B 
  
    a = 0
    for c,ind in enumerate(poke_change_b):   
        if c == 0:
            a = 0
        elif c > 0:
            a += 1
        for i, poke in enumerate(ind):
            fig.add_subplot(12,4,a*4+27)
            plt.axis('off')

            plt.scatter(x_all,y_all, color = 'grey', s = 10, alpha = 0.2)

            if i == 0 or i == 2:
                if poke == 1:
                    x_coords = 332
                    y_coords = 4.8
                elif poke == 9:
                    x_coords = 332
                    y_coords = 0.8
                if i == 0:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5)
                elif i == 2:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5)
           
            elif i == 1 or i == 3:
                if poke == 4:
                    x_coords = 132
                    y_coords = 2.8

                elif poke == 6:
                    x_coords = 532
                    y_coords = 2.8

                    
                if poke == 9:
                    x_coords = 332
                    y_coords = 0.8

                elif poke == 1:
                    x_coords = 332
                    y_coords = 4.8

                    
                if i == 1:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5)
                elif i == 3:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5, alpha = 0.5)
                
                
    # Change Init and B
    a = 0
    for c,ind in enumerate(pokes_init_b):
        
        if c == 0:
            a = 0
        elif c > 0:
            a += 1
        for i, poke in enumerate(ind):
            fig.add_subplot(12,4,a*4+28)
            plt.axis('off')

            plt.scatter(x_all,y_all, color = 'grey', s = 10, alpha = 0.2)

        
            if i == 0 or i == 2:
                if poke == 1:
                    x_coords = 332
                    y_coords = 4.8
                elif poke == 9:
                    x_coords = 332
                    y_coords = 0.8
                if i == 0:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5, alpha = 0.5)
                elif i == 2:
                    plt.scatter(x_coords,y_coords, color = 'purple', s = 5, alpha = 0.5)
            elif i == 1 or i == 3:
                if poke == 2:
                    x_coords = 232
                    y_coords = 3.8
                elif poke == 3:
                    x_coords = 432
                    y_coords = 3.8
                elif poke == 7:
                    x_coords = 232
                    y_coords = 1.8
                elif poke == 8:
                    x_coords = 432
                    y_coords = 1.8
                elif poke == 9:
                    x_coords = 332
                    y_coords = 0.8
                if i == 1:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5, alpha = 0.5)
                elif i == 3:
                    plt.scatter(x_coords,y_coords, color = 'yellow', s = 5, alpha = 0.5)
               
def plot(data_HP,data_PFC, session):
    

    correct_list_within_PFC, correct_list_between_PFC, y_pred_class_between_t_1_2_PFC, Y_PFC = classifier_pseudo_simultaneous(data_PFC, session)
    correct_list_within_HP, correct_list_between_HP, y_pred_class_between_t_1_2_HP, Y_HP = classifier_pseudo_simultaneous(data_HP, session)   
    plt.figure(2)
    sns.barplot(data=[correct_list_within_PFC,correct_list_between_PFC,correct_list_within_HP,correct_list_between_HP], capsize=.1, ci="sd",  palette="Blues_d")
    #sns.swarmplot(data=[correct_list_within_PFC,correct_list_between_PFC,correct_list_within_HP,correct_list_between_HP], color="0", alpha=.35)
    plt.xticks(np.arange(4),('PFC within task', 'PFC between tasks', 'HP within task', 'HP between tasks'))
    plt.ylabel('% correct')
    plt.title('SVM')
    

    
    
          