#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:28:46 2020

@author: veronikasamborska
"""


from sklearn.linear_model import LogisticRegression

import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
from sklearn.linear_model import LinearRegression
import seaborn as sns
from collections import OrderedDict
import regression_function as reg_f
from palettable import wesanderson as wes
from scipy import io
from itertools import combinations 
import regressions as re 
import scipy
import value_reg as vg
import palettable
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)
     
def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid

def reversals_during_recordings(m484, m479, m483, m478, m486, m480, m481):
    subj = [m478, m486, m480, m481]
    #subj = [m484, m479, m483, m478, m486, m480, m481]
    #subj = [m484, m479, m483]

    all_subjects = []
    for s in subj:
        s = np.asarray(s)
        date  = []
        for ses in s:
            date.append(ses.datetime)
            
        ind_sort = np.argsort(date)    
        s = s[ind_sort]
    
        all_sessions = []
        
        
        for session in s: 
            sessions_block = session.trial_data['block']
            n_trials = session.trial_data['n_trials']
            # Find trials where threshold crossed.
            prt = (session.trial_data['pre-reversal trials'] > 0).astype(int)
                
            forced_trials = session.trial_data['forced_trial']
            forced_trials_sum = sum(forced_trials)
            forced_array = np.where(forced_trials == 1)[0]
            sessions_block = np.delete(sessions_block, forced_array)
            prt = np.delete(prt,forced_array)
            #Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
           # threshold_crossing_trials = np.where(Block_transitions == 1)[0]
          
            n_trials = n_trials -  forced_trials_sum
            threshold_crossing_trials = np.where((prt[1:] - prt[:-1]) == 1)[0]
            block_length = np.append(np.diff(threshold_crossing_trials),threshold_crossing_trials[0])
            all_sessions.append(np.nanmean(block_length))
        
        all_subjects.append(np.nanmean(all_sessions))
        
    return all_subjects

def run_correlations(PFC):
    
    
    C_1_all_a,C_2_all_a,C_3_all_a = time_in_block_corr(PFC, area = 'PFC', n = 11, plot_a = False, plot_b = False)
    
    value_a = []
    c_1 = 3
    j = 2
    for i, ii in enumerate(C_1_all_a):
        C_1 = C_1_all_a[i]
        C_2 = C_2_all_a[i]
        C_3 = C_3_all_a[i]

        C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
        C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
        C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
        nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
        C_1 = np.transpose(C_1[:,nans[0],:],[2,0,1]); C_2 = np.transpose(C_2[:,nans[0],:],[2,0,1]);  C_3 = np.transpose(C_3[:,nans[0],:],[2,0,1])
        val_a =  generalisation_plot_betas_corr(C_1,C_2,C_3, c_1, reward_times_to_choose = np.asarray([20,25,35,42]))
        value_a.append(np.max(val_a[j],0))
        
    all_subjects = reversals_during_recordings(m484, m479, m483, m478, m486, m480, m481)
    coef, p = scipy.stats.spearmanr(value_a,all_subjects)
    plt.scatter(value_a,all_subjects, c = 'black')
    sns.despine()
    plt.ylabel('Mean # of Trials to Threshold')
    plt.xlabel('Maximum Beta')

def generalisation_plot_betas_corr(C_1,C_2,C_3, c_1, reward_times_to_choose = np.asarray([25,35,42])):
    
    c_1 = c_1
    C_1_rew = C_1[c_1]; C_2_rew = C_2[c_1]; C_3_rew = C_3[c_1]
    C_1_rew_count = C_1[c_1]; C_2_rew_count = C_2[c_1]; C_3_rew_count = C_3[c_1]
   
    reward_times_to_choose = reward_times_to_choose
    
    C_1_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_2_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_3_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
   
    j = 0
    for i in reward_times_to_choose:
        if i ==reward_times_to_choose[0]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-20:i],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-20:i],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-20:i],1)
        if i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-2:i+2],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-2:i+2],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-2:i+2],1)
        elif i == reward_times_to_choose[3]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i:],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i:],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i:],1)
         
        j +=1
    
   
    C_1_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_2_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_3_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    j = 0
    for i in reward_times_to_choose:
        if i ==reward_times_to_choose[0]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-20:i],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-20:i],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-20:i],1)
        if i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-2:i+2],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-2:i+2],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-2:i+2],1)
        elif i == reward_times_to_choose[3]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i:],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i:],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i:],1)
     
        j +=1
        
    
        
    cpd_1_2_rew, cpd_1_2_rew_var = vg.regression_code_session(C_2_rew_count, C_1_rew_proj);  
    cpd_1_3_rew, cpd_1_3_rew_var = vg.regression_code_session(C_3_rew_count, C_1_rew_proj); 
    cpd_2_3_rew, cpd_2_3_rew_var = vg.regression_code_session(C_3_rew_count, C_2_rew_proj)
    
   
    value_to_value = (cpd_1_2_rew + cpd_1_3_rew +cpd_2_3_rew)/np.sqrt((cpd_1_2_rew_var+cpd_1_3_rew_var+cpd_2_3_rew_var))
    
    return value_to_value

def time_in_block_corr(PFC, area = 'PFC', n = 11, plot_a = False, plot_b = False):
   
   
    all_subjects = [PFC['DM'][0][:9], PFC['DM'][0][9:25],PFC['DM'][0][25:39],PFC['DM'][0][39:]]
    all_firing = [PFC['Data'][0][:9], PFC['Data'][0][9:25],PFC['Data'][0][25:39],PFC['Data'][0][39:]]
   #  all_subjects = [HP['DM'][0][:16], HP['DM'][0][16:24],HP['DM'][0][24:],PFC['DM'][0][:9], PFC['DM'][0][9:25],PFC['DM'][0][25:39],PFC['DM'][0][39:]]
    # all_firing = [HP['Data'][0][:16], HP['Data'][0][16:24],HP['Data'][0][24:],PFC['Data'][0][:9], PFC['Data'][0][9:25],PFC['Data'][0][25:39],PFC['Data'][0][39:]]
    # all_subjects = [HP['DM'][0][:16], HP['DM'][0][16:24],HP['DM'][0][24:]]
    # all_firing = [HP['Data'][0][:16], HP['Data'][0][16:24],HP['Data'][0][24:]]
    
    C_1_all = []; C_2_all = []; C_3_all = []
    
    cpd_1 = []; cpd_2 = []; cpd_3 = []
    
    average = vg.rew_prev_behaviour(PFC, n = n, perm = False)
    # average = vg.rew_prev_behaviour(HP, n = n, perm = False)

    for d,dd in enumerate(all_subjects):
        C_1 = []; C_2 = []; C_3 = []

        dm = all_subjects[d]
        firing = all_firing[d]

        for  s, sess in enumerate(dm):
            # if d <3:
            #     average=average_2
            # else:
            #     average=average_1
 
            
           
            DM = dm[s]
            firing_rates = firing[s]
           
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            
            choices = DM[:,1]
            reward = DM[:,2]  
            task =  DM[:,5]
            a_pokes = DM[:,6]
            b_pokes = DM[:,7]
            
            taskid = vg.task_ind(task, a_pokes, b_pokes)
          
            
            task_1 = np.where(taskid == 1)[0]
            task_2 = np.where(taskid == 2)[0]
            task_3 = np.where(taskid == 3)[0]
    
            reward_current = reward
            choices_current = choices-0.5
    
           
            rewards_1 = reward_current[task_1]
            choices_1 = choices_current[task_1]
            
            previous_rewards_1 = scipy.linalg.toeplitz(rewards_1, np.zeros((1,n)))[n-1:-1]
             
            previous_choices_1 = scipy.linalg.toeplitz(0.5-choices_1, np.zeros((1,n)))[n-1:-1]
             
            interactions_1 = scipy.linalg.toeplitz((((0.5-choices_1)*(rewards_1-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_1)).reshape(len(interactions_1),1)
             
            X_1 = np.hstack([previous_rewards_1,previous_choices_1,interactions_1,ones])
            #average_val_ex_ch = np.concatenate([average[n].reshape(1),average[n*2:]])
            #X_exl_1 = np.concatenate([X_1[:,n].reshape(len(X_1),1),X_1[:,n*2:]],1)
            #value = np.matmul(X[:,n*2:], average[n*2:])
            value_1 =np.matmul(X_1, average)
            #value_1 =np.matmul(X_exl_1, average_val_ex_ch)
    
            rewards_1 = rewards_1[n:]
            choices_1 = choices_1[n:]
              
            
            ones_1 = np.ones(len(choices_1))
            trials_1 = len(choices_1)
            value_1_choice_1 = choices_1*value_1
           
          
            firing_rates_1 = firing_rates[task_1][n:]
            
            a_1 = np.where(choices_1 == 0.5)[0]
            b_1 = np.where(choices_1 == -0.5)[0]
            
            if plot_a == True:
                rewards_1 = rewards_1[a_1] 
                choices_1 = choices_1[a_1]    
                value_1 = value_1[a_1]
                ones_1  = ones_1[a_1]
                firing_rates_1 = firing_rates_1[a_1]
               # rewards_1 = scipy.stats.zscore(rewards_1)
              #  value_1 = scipy.stats.zscore(value_1)
    
              
            elif plot_b == True:
                
                rewards_1 = rewards_1[b_1] 
                choices_1 = choices_1[b_1]
                value_1 = value_1[b_1]
                ones_1  = ones_1[b_1]
                firing_rates_1 = firing_rates_1[b_1]
              #  rewards_1 = scipy.stats.zscore(rewards_1)
              #  value_1 = scipy.stats.zscore(value_1)
    
            predictors_all = OrderedDict([
                                        ('Choice', choices_1),
                                        ('Reward', rewards_1),
                                        ('Value',value_1), 
                                        ('Value Сhoice',value_1_choice_1), 
                                   #     ('Prev Rew Ch', prev_ch_reward_1),
    
                               #      ('Prev Rew', prev_reward_1),
                                     #  ('Prev Ch', prev_choice_1),
                                       ('ones', ones_1)
                                        ])
            
            X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
            
            n_predictors = X_1.shape[1]
            y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
           # tstats,x = regression_code_session(y_1, X_1)
            #tstats =  reg_f.regression_code(y_1, X_1)
            ols = LinearRegression()
            ols.fit(X_1,y_1)
            C_1.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
            #C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            cpd_1.append(re._CPD(X_1,y_1).reshape(n_neurons, n_timepoints, n_predictors))
            
            
            rewards_2 = reward_current[task_2]
            choices_2 = choices_current[task_2]
            
            previous_rewards_2 = scipy.linalg.toeplitz(rewards_2, np.zeros((1,n)))[n-1:-1]
             
            previous_choices_2 = scipy.linalg.toeplitz(0.5-choices_2, np.zeros((1,n)))[n-1:-1]
             
            interactions_2 = scipy.linalg.toeplitz((((0.5-choices_2)*(rewards_2-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_2)).reshape(len(interactions_2),1)
             
            X_2 = np.hstack([previous_rewards_2,previous_choices_2,interactions_2,ones])
            # average_val_ex_ch = np.concatenate([average[n].reshape(1),average[n*2:]])
            # X_exl_2 = np.concatenate([X_2[:,n].reshape(len(X_2),1),X_2[:,n*2:]],1)
            # value = np.matmul(X[:,n*2:], average[n*2:])
            value_2 =np.matmul(X_2, average)
            #value_2 =np.matmul(X_exl_2, average_val_ex_ch)
    
            rewards_2 = rewards_2[n:]
            choices_2 = choices_2[n:]
              
            
            ones_2 = np.ones(len(choices_2))
            trials_2 = len(choices_2)
    
            firing_rates_2 = firing_rates[task_2][n:]
            
            value_2_choice_2 = choices_2*value_2

            a_2 = np.where(choices_2 == 0.5)[0]
            b_2 = np.where(choices_2 == -0.5)[0]
            
            if plot_a == True:
                rewards_2 = rewards_2[a_2] 
                choices_2 = choices_2[a_2]
                value_2 = value_2[a_2]
                ones_2  = ones_2[a_2]
                firing_rates_2 = firing_rates_2[a_2]
               # rewards_2 = scipy.stats.zscore(rewards_2)
               # value_2 = scipy.stats.zscore(value_2)
    
            elif plot_b == True:
                
                rewards_2 = rewards_2[b_2] 
                choices_2 = choices_2[b_2]
                value_2 = value_2[b_2]
                ones_2  = ones_2[b_2]
                firing_rates_2 = firing_rates_2[b_2]
               # rewards_2 = scipy.stats.zscore(rewards_2)
               # value_2 = scipy.stats.zscore(value_2)
    
            predictors_all = OrderedDict([
                                         ('Choice', choices_2),
                                        ('Reward', rewards_2),
                                        ('Value',value_2), 
                                         ('Value Сhoice',value_2_choice_2), 
                                     #   ('Prev Rew Ch', prev_ch_reward_2),
    #
                                   #    ('Prev Rew', prev_reward_2),
                                      # ('Prev Ch', prev_choice_2),
                                        ('ones', ones_2)
                                        ])
            
            X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
            
            n_predictors = X_2.shape[1]
            y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
           # tstats,x = regression_code_session(y_2, X_2)
            ols = LinearRegression()
            ols.fit(X_2,y_2)
            C_2.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
            cpd_2.append(re._CPD(X_2,y_2).reshape(n_neurons, n_timepoints, n_predictors))
      
        
            
            rewards_3 = reward_current[task_3]
            choices_3 = choices_current[task_3]
            
            previous_rewards_3 = scipy.linalg.toeplitz(rewards_3, np.zeros((1,n)))[n-1:-1]
             
            previous_choices_3 = scipy.linalg.toeplitz(0.5-choices_3, np.zeros((1,n)))[n-1:-1]
             
            interactions_3 = scipy.linalg.toeplitz((((0.5-choices_3)*(rewards_3-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_3)).reshape(len(interactions_3),1)
             
            X_3 = np.hstack([previous_rewards_3,previous_choices_3,interactions_3,ones])
           #  average_val_ex_ch = np.concatenate([average[n].reshape(1),average[n*2:]])
           #  X_exl_3 = np.concatenate([X_3[:,n].reshape(len(X_3),1),X_3[:,n*2:]],1)
           # # value = np.matmul(X[:,n*2:], average[n*2:])
            value_3 =np.matmul(X_3, average)
            # value_3 =np.matmul(X_exl_3, average_val_ex_ch)
    
            rewards_3 = rewards_3[n:]
            choices_3 = choices_3[n:]
              
            
            ones_3 = np.ones(len(choices_3))
            trials_3 = len(choices_3)
    
            firing_rates_3 = firing_rates[task_3][n:]
            
            value_3_choice_3 = choices_3*value_3

            a_3 = np.where(choices_3 == 0.5)[0]
            b_3 = np.where(choices_3 == -0.5)[0]
            
            if plot_a == True:
                rewards_3 = rewards_3[a_3] 
                choices_3 = choices_3[a_3]
                value_3 = value_3[a_3]
                ones_3  = ones_3[a_3]
    
                firing_rates_3 = firing_rates_3[a_3]
              #  rewards_3 = scipy.stats.zscore(rewards_3)
              #  value_3 = scipy.stats.zscore(value_3)
    
               
            elif plot_b == True:
                rewards_3 = rewards_3[b_3] 
                choices_3 = choices_3[b_3]
              
                value_3 = value_3[b_3]
                ones_3  = ones_3[b_3]
    
                firing_rates_3 = firing_rates_3[b_3]
             #   rewards_3 = scipy.stats.zscore(rewards_3)
             #   value_3 = scipy.stats.zscore(value_3)
    
               
              
      
            predictors_all = OrderedDict([
                                        ('Ch', choices_3),
                                        ('Rew', rewards_3),
                                        ('Value',value_3), 
                                        ('Value Сhoice',value_3_choice_3), 
    #                                   
                                     #   ('Prev Rew Ch', prev_ch_reward_3),
                                  #    ('Prev Rew', prev_reward_3),
                                      # ('Prev Ch', prev_choice_3),
                                        ('ones', ones_3)
                                        ])
            
            X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
            rank = np.linalg.matrix_rank(X_1)
            print(rank)
            n_predictors = X_3.shape[1]
            print(n_predictors)
            y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            #tstats,x = regression_code_session(y_3, X_3)
            ols = LinearRegression()
            ols.fit(X_3,y_3)
            C_3.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings
    
           # C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            cpd_3.append(re._CPD(X_3,y_3).reshape(n_neurons, n_timepoints, n_predictors))
            
        C_1_all.append(np.concatenate(C_1,0)); C_2_all.append(np.concatenate(C_2,0)); C_3_all.append(np.concatenate(C_3,0))
     
    return C_1_all,C_2_all,C_3_all
    