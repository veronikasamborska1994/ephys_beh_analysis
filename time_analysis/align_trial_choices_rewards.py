#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:37:28 2020

@author: veronikasamborska
"""


## Warping code for interpolating firing rates on neighbouring trials (including rewards and choices + regressions)

import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')

import utility as ut 
from palettable import wesanderson as wes
from scipy.special import erf
from collections import OrderedDict
import regression_function as reg_f
import regressions as re
from sklearn.linear_model import LinearRegression
import seaborn as sns
from palettable import wesanderson as wes



def regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True, perm = False):
    
    C = []
    cpd = []
    
    if perm:
        C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
   
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
    
    for s, session in enumerate(a_a_matrix_t_1_list):
        
        firing_rates = np.concatenate([a_a_matrix_t_1_list[s],a_a_matrix_t_2_list[s],a_a_matrix_t_3_list[s]],1)
        n_neurons = firing_rates.shape[0]
        rewards_a_1 = a_a_matrix_t_1_list_rewards[s]
        rewards_a_2 = a_a_matrix_t_2_list_rewards[s]
        rewards_a_3 = a_a_matrix_t_3_list_rewards[s]
        rewards = np.hstack([rewards_a_1,rewards_a_2,rewards_a_3])

        choices_a_1 = a_a_matrix_t_1_list_choices[s]
        choices_a_2 = a_a_matrix_t_2_list_choices[s]
        choices_a_3 = a_a_matrix_t_3_list_choices[s]
        choices = np.hstack([choices_a_1,choices_a_2,choices_a_3])

        block_length = np.tile(np.arange(session.shape[1]/2),2)
        trial_number = np.tile(block_length,3)
        noise = np.random.normal(np.mean(choices), np.std(choices), len(trial_number))
        ones = np.ones(len(choices))
        trials = len(ones)
        
        predictors = OrderedDict([('Reward', rewards),
                                      ('Choice', choices),
                                      ('Trial Number',trial_number),
                                      #('Noise', noise),
                                      ('Constant', ones)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
        cpd.append(re._CPD(X,y.T).reshape(n_neurons, n_predictors))
        
        if perm:
           for i in range(perm):
               
               X_perm= np.roll(X,np.random.randint(trials), axis = 0)

               tstats = reg_f.regression_code(y.T, X_perm)
        
               C_perm[i].append(tstats.reshape(n_predictors,n_neurons))   # Predictor loadings
               cpd_perm[i].append(re._CPD(X_perm,y.T).reshape(n_neurons, n_predictors))
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        p_95 = np.percentile(cpd_perm,95, axis = 0)
        p_99 = np.percentile(cpd_perm,99, axis = 0)
        C_perm =np.stack([np.concatenate(C_i,1) for C_i in C_perm],1)
 
     
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    return cpd, C,cpd_perm,p_95,p_99,C_perm



def plot(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC):
    
    color = 'lightblue'
    plt.rcParams.update({'font.size': 8})
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors

    # No reward HP
    cpd_a_HP, C_a_HP, cpd_perm_a_HP, p_95_a_HP, p_99_a_HP, C_perm_a_HP = regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, perm = 1000)
    cpd_b_HP, C_b_HP, cpd_perm_b_HP, p_95_b_HP, p_99_b_HP,C_perm_b_HP = regression_time_choices_rewards_b_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, perm = 1000)
    
    cpd_a_PFC, C_a_PFC, cpd_perm_a_PFC, p_95_a_PFC, p_99_a_PFC,C_perm_a_PFC = regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, perm = 1000)
    cpd_b_PFC, C_b_PFC, cpd_perm_b_PFC, p_95_b_PFC, p_99_b_PFC, C_perm_b_PFC = regression_time_choices_rewards_b_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, perm = 1000)
    
    corr_perm = np.zeros((C_perm_a_HP.shape[0],C_perm_a_HP.shape[1]))
    for p,predictor in enumerate(C_perm_a_HP):
        for i,ii in enumerate(predictor):
            corr_perm[p,i] = np.corrcoef(C_perm_a_HP[p,i],C_perm_b_HP[p,i])[0,1]
            
    corr_perm_PFC = np.zeros((C_perm_a_PFC.shape[0],C_perm_a_PFC.shape[1]))
    for p,predictor in enumerate(C_perm_a_PFC):
        for i,ii in enumerate(predictor):
            corr_perm_PFC[p,i] = np.corrcoef(C_perm_a_PFC[p,i],C_perm_b_PFC[p,i])[0,1]
    
    percentile_99_HP = np.percentile(corr_perm,5, axis = 1)
    percentile_99_PFC = np.percentile(corr_perm_PFC,5, axis = 1)
    pred = 2
    plt.subplot(2,1,1)
    plt.hist(corr_perm[pred,:],100,color = 'lightblue')
    plt.vlines(percentile_99_HP[pred],ymin = 0, ymax = np.max(np.histogram(corr_perm[pred,:],100)[0]), color = 'grey',linestyle = '--',label = '0.01')
    plt.vlines((np.corrcoef(C_a_HP[pred,:],C_b_HP[pred,:])[0,1]),ymin = 0, ymax = np.max(np.histogram(corr_perm[pred,:],100)[0]),label = 'Data')
   # plt.legend()
    plt.title('CA1 # in Block Anti-Correlation')

    plt.subplot(2,1,2)
    plt.hist(corr_perm_PFC[pred,:], 100,color = 'pink')
    plt.vlines(percentile_99_PFC[pred],ymin = 0, ymax = np.max(np.histogram(corr_perm_PFC[pred,:],100)[0]),color = 'grey', linestyle = '--',label = '0.01')
    plt.vlines((np.corrcoef(C_a_PFC[pred,:],C_b_PFC[pred,:])[0,1]),ymin = 0, ymax = np.max(np.histogram(corr_perm_PFC[pred,:],100)[0]), label = 'Data')
    plt.legend()
    plt.title('PFC # in Block Anti-Correlation')
    plt.tight_layout()
    sns.despine()


    y  = [0,1,2]
    plt.subplot(2,2,1)
    plt.bar(y, cpd_a_HP[:-1], color = c[0], alpha = 0.8, tick_label = ['Reward', 'Choice', 'Block'])
    
    for i,p_95 in enumerate(p_95_a_HP):
        if p_99_a_HP[i] < cpd_a_HP[i]:
            plt.annotate('***',[i-0.1, (cpd_a_HP[i]+0.023)])
        elif p_95 < cpd_a_HP[i]:
            plt.annotate('*',[i, (cpd_a_HP[i]+0.02)])
        
    plt.ylim([0, np.max(cpd_a_HP[:-1])+0.03])
    sns.despine()
    plt.title('CA1')
    plt.ylabel('CPD')
    plt.tight_layout()

    plt.subplot(2,2,2)
    plt.bar(y, cpd_b_HP[:-1], color = c[1], alpha = 0.8, tick_label = ['Reward', 'Choice', 'Block'])
    for i,p_95 in enumerate(p_95_b_HP):
        if p_99_b_HP[i] < cpd_b_HP[i]:
            plt.annotate('***',[i-0.1, (cpd_a_HP[i]+0.023)])
        elif p_95 < cpd_b_HP[i]:
            plt.annotate('*',[i, (cpd_a_HP[i]+0.02)])
        
    plt.ylim([0, np.max(cpd_b_HP[:-1])+0.03])
    sns.despine()
    plt.title('CA1')
    plt.ylabel('CPD')
    plt.tight_layout()

    plt.subplot(2,2,3)
    plt.bar(y, cpd_a_PFC[:-1], color = c[0], alpha = 0.8, tick_label = ['Reward', 'Choice', 'Block'])
    for i,p_95 in enumerate(p_95_a_PFC):
        if p_99_a_PFC[i] < cpd_a_PFC[i]:
            plt.annotate('***',[i-0.1, (cpd_a_PFC[i]+0.023)])
        elif p_95 < cpd_a_PFC[i]:
            plt.annotate('*',[i, (cpd_a_PFC[i]+0.02)])
        
    plt.ylim([0, np.max(cpd_a_PFC[:-1])+0.04])
    sns.despine()
    plt.title('PFC')
    plt.ylabel('CPD')
    plt.tight_layout()

    plt.subplot(2,2,4)
    plt.bar(y, cpd_b_PFC[:-1], color = c[1], alpha = 0.8, tick_label = ['Reward', 'Choice', 'Block'])
    for i,p_95 in enumerate(p_95_b_PFC):
        if p_99_b_PFC[i] < cpd_b_PFC[i]:
            plt.annotate('***',[i-0.1, (cpd_b_PFC[i]+0.023)])
        elif p_95 < cpd_b_PFC[i]:
            plt.annotate('*',[i, (cpd_b_PFC[i]+0.02)])
        
    plt.ylim([0, np.max(cpd_b_PFC[:-1])+0.04])
    sns.despine()
    plt.title('PFC')
    plt.ylabel('CPD')
    plt.tight_layout()

    
    # x = [C_a[0],C_a[1],C_a[2]]
    # y = [C_b[0],C_b[1],C_b[2]]
    # p = 0
    # titles = ['Reward', 'Choice', 'Block']
    # plt.figure(figsize = (4,10))
    # plt.title('HP')
    # for i,ii in zip(x,y):
    #     p+=1
    #     plt.subplot(4,1,p)
    #     sns.regplot(i,ii, color = color)
    #     corr = np.corrcoef(i,ii)[0,1]
    #     plt.xlabel(titles[p-1] + ' ' + 'A')
    #     plt.ylabel(titles[p-1] + ' ' + 'B')
    #     plt.annotate('r = ' + str(np.around(corr,3)), [np.max(i),np.max(ii)])
    #     plt.tight_layout()
    #     sns.despine()

    # cpd_a, C_a, cpd_perm_a, p_95_a, p_99_a = regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 62, HP = HP, perm = 2, include_reward = False)
    # cpd_b, C_b, cpd_perm_b, p_95_b, p_99_b = regression_time_choices_rewards_b_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 62, HP = HP, perm = 2, include_reward = False)
    # i = C_a[0]
    # ii = C_b[0]
    # plt.subplot(4,1,4)

    # sns.regplot(i,ii, color =color)
    # corr = np.corrcoef(C_a[0],C_b[0])[0,1]
    # plt.xlabel('Block in Reg without Reward or Choice' + ' ' + 'A')
    # plt.ylabel('Block in Reg without Reward or Choice' + ' ' + 'B')
    # plt.annotate('r = ' + str(np.around(corr,3)), [np.max(C_a[0]),np.max(C_b[0])])
    # plt.tight_layout()
    # sns.despine()
  
       
def regression_time_choices_rewards_b_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True, perm = False):
    
    C = []
    cpd = []
    if perm:
       C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
       cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
   
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
    
    for s, session in enumerate(b_b_matrix_t_1_list):
        
        firing_rates = np.concatenate([b_b_matrix_t_1_list[s],b_b_matrix_t_2_list[s],b_b_matrix_t_3_list[s]],1)
        n_neurons = firing_rates.shape[0]
        rewards_b_1 = b_b_matrix_t_1_list_rewards[s]
        rewards_b_2 = b_b_matrix_t_2_list_rewards[s]
        rewards_b_3 = b_b_matrix_t_3_list_rewards[s]
        rewards = np.hstack([rewards_b_1,rewards_b_2,rewards_b_3])

        choices_b_1 = b_b_matrix_t_1_list_choices[s]
        choices_b_2 = b_b_matrix_t_2_list_choices[s]
        choices_b_3 = b_b_matrix_t_3_list_choices[s]
        choices = np.hstack([choices_b_1,choices_b_2,choices_b_3])

        block_length = np.tile(np.arange(session.shape[1]/2),2)
        trial_number = np.tile(block_length,3)
        ones = np.ones(len(choices))
        trials = len(ones)
        noise = np.random.normal(np.mean(choices), np.std(choices), len(trial_number))

        predictors = OrderedDict([('Reward', rewards),
                                      ('Choice', choices),
                                      ('Trial Number',trial_number),
                                     # ('Noise', noise),
                                      ('Constant', ones)])
        
           
        X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
        cpd.append(re._CPD(X,y.T).reshape(n_neurons, n_predictors))
        
        if perm:
          for i in range(perm):
              X_perm= np.roll(X,np.random.randint(trials), axis = 0)
              tstats = reg_f.regression_code(y.T, X_perm)
        
              C_perm[i].append(tstats.reshape(n_predictors,n_neurons))   # Predictor loadings
              cpd_perm[i].append(re._CPD(X_perm,y.T).reshape(n_neurons, n_predictors))
    
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        p_95 = np.percentile(cpd_perm,95, axis = 0)
        p_99 = np.percentile(cpd_perm,99, axis = 0)
        C_perm =np.stack([np.concatenate(C_i,1) for C_i in C_perm],1)
 
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    return cpd, C, cpd_perm, p_95, p_99, C_perm
         

def run():
    
    HP_aligned_time, HP_aligned_choices, HP_aligned_rewards, PFC_aligned_time, PFC_aligned_choices, PFC_aligned_rewards = all_sessions_align_beh_raw_data_choices_rewards(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end)    



def hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True):
   
    HP_aligned_time, HP_aligned_choices, HP_aligned_rewards, PFC_aligned_time,\
        PFC_aligned_choices, PFC_aligned_rewards,state_list_HP,\
            state_list_PFC,task_list_HP, task_list_PFC = all_sessions_align_beh_raw_data_choices_rewards(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end)    
        
    if HP == True:
        exp = HP_aligned_time
        state_exp = state_list_HP
        task_exp = task_list_HP
        choices_exp = HP_aligned_choices
        reward_exp = HP_aligned_rewards

    else:
        exp = PFC_aligned_time
        state_exp = state_list_PFC
        task_exp = task_list_PFC
        choices_exp = PFC_aligned_choices
        reward_exp = PFC_aligned_rewards

    
    a_a_matrix_t_1_list = []
    b_b_matrix_t_1_list = []
    a_a_matrix_t_2_list = []
    b_b_matrix_t_2_list = []
    a_a_matrix_t_3_list = []
    b_b_matrix_t_3_list = []
    
    
    a_a_matrix_t_1_list_rewards = []
    b_b_matrix_t_1_list_rewards = []
    a_a_matrix_t_2_list_rewards = []
    b_b_matrix_t_2_list_rewards = []
    a_a_matrix_t_3_list_rewards = []
    b_b_matrix_t_3_list_rewards = []
     
    
    a_a_matrix_t_1_list_choices = []
    b_b_matrix_t_1_list_choices = []
    a_a_matrix_t_2_list_choices = []
    b_b_matrix_t_2_list_choices = []
    a_a_matrix_t_3_list_choices = []
    b_b_matrix_t_3_list_choices = []
   
    for s, session in enumerate(exp):
        
        state = state_exp[s]
        task = task_exp[s]
        reward = reward_exp[s]
        choice = choices_exp[s]
        
        a_s_t_1 = np.where((state == 1) & (task == 1))[0][0]
        a_s_t_2 = np.where((state == 1) & (task == 2))[0][0] - np.where(task == 2)[0][0]
        a_s_t_3 = np.where((state == 1) & (task == 3))[0][0] - np.where(task == 3)[0][0]
        
        # Task 1 
        if a_s_t_1 == 0:
            a_s_t1_1 = session[0]
            a_s_t1_2 = session[2]
            b_s_t1_1 = session[1]
            b_s_t1_2 = session[3]
            
            a_a_matrix_t_1 = np.hstack((a_s_t1_1,a_s_t1_2)) # At 13 change
            b_b_matrix_t_1 = np.hstack((b_s_t1_1, b_s_t1_2))# At 13 change 
           
            a_s_t1_1_rew = reward[0]
            a_s_t1_2_rew = reward[2]
            b_s_t1_1_rew = reward[1]
            b_s_t1_2_rew = reward[3]
            
            a_a_matrix_t_1_rew = np.hstack((a_s_t1_1_rew,a_s_t1_2_rew)) # At 13 change
            b_b_matrix_t_1_rew = np.hstack((b_s_t1_1_rew, b_s_t1_2_rew))# At 13 change 
           
            
            a_s_t1_1_ch = choice[0]
            a_s_t1_2_ch = choice[2]
            b_s_t1_1_ch = choice[1]
            b_s_t1_2_ch = choice[3]
            
            a_a_matrix_t_1_ch = np.hstack((a_s_t1_1_ch,a_s_t1_2_ch)) # At 13 change
            b_b_matrix_t_1_ch = np.hstack((b_s_t1_1_ch, b_s_t1_2_ch))# At 13 change 
           
        elif a_s_t_1 !=0 :
            a_s_t1_1 = session[1]
            a_s_t1_2 = session[3]
            b_s_t1_1 = session[0]
            b_s_t1_2 = session[2]
            
            a_a_matrix_t_1 = np.hstack((a_s_t1_1,a_s_t1_2)) # At 13 change
            b_b_matrix_t_1 = np.hstack((b_s_t1_1, b_s_t1_2))# At 13 change
          
            a_s_t1_1_rew = reward[1]
            a_s_t1_2_rew = reward[3]
            b_s_t1_1_rew = reward[0]
            b_s_t1_2_rew = reward[2]
            
            a_a_matrix_t_1_rew = np.hstack((a_s_t1_1_rew,a_s_t1_2_rew)) # At 13 change
            b_b_matrix_t_1_rew = np.hstack((b_s_t1_1_rew, b_s_t1_2_rew))# At 13 change 
           
            
            a_s_t1_1_ch = choice[1]
            a_s_t1_2_ch = choice[3]
            b_s_t1_1_ch = choice[0]
            b_s_t1_2_ch = choice[2]
            
            a_a_matrix_t_1_ch = np.hstack((a_s_t1_1_ch,a_s_t1_2_ch)) # At 13 change
            b_b_matrix_t_1_ch = np.hstack((b_s_t1_1_ch, b_s_t1_2_ch))# At 13 change 
           
        if a_s_t_2 == 0:
            
            a_s_t2_1 = session[4]
            a_s_t2_2 = session[6]
            b_s_t2_1 = session[5]
            b_s_t2_2 = session[7]
            a_a_matrix_t_2 = np.hstack((a_s_t2_1,a_s_t2_2)) # At 13 change
            b_b_matrix_t_2 = np.hstack((b_s_t2_1, b_s_t2_2))# At 13 change
           
            a_s_t2_1_rew = reward[4]
            a_s_t2_2_rew = reward[6]
            b_s_t2_1_rew = reward[5]
            b_s_t2_2_rew = reward[7]
            
            a_a_matrix_t_2_rew = np.hstack((a_s_t2_1_rew,a_s_t2_2_rew)) # At 13 change
            b_b_matrix_t_2_rew = np.hstack((b_s_t2_1_rew, b_s_t2_2_rew))# At 13 change 
           
            
            a_s_t2_1_ch = choice[4]
            a_s_t2_2_ch = choice[6]
            b_s_t2_1_ch = choice[5]
            b_s_t2_2_ch = choice[7]
            
            a_a_matrix_t_2_ch = np.hstack((a_s_t2_1_ch,a_s_t2_2_ch)) # At 13 change
            b_b_matrix_t_2_ch = np.hstack((b_s_t2_1_ch, b_s_t2_2_ch))# At 13 change 
           
        elif a_s_t_2 !=0:
            a_s_t2_1 = session[5]
            a_s_t2_2 = session[7]
            b_s_t2_1 = session[4]
            b_s_t2_2 = session[6]
            a_a_matrix_t_2 = np.hstack((a_s_t2_1,a_s_t2_2)) # At 13 change
            b_b_matrix_t_2 = np.hstack((b_s_t2_1, b_s_t2_2))# At 13 change
            
            a_s_t2_1_rew = reward[5]
            a_s_t2_2_rew = reward[7]
            b_s_t2_1_rew = reward[4]
            b_s_t2_2_rew = reward[6]
            
            a_a_matrix_t_2_rew = np.hstack((a_s_t2_1_rew,a_s_t2_2_rew)) # At 13 change
            b_b_matrix_t_2_rew = np.hstack((b_s_t2_1_rew, b_s_t2_2_rew))# At 13 change 
           
            
            a_s_t2_1_ch = choice[5]
            a_s_t2_2_ch = choice[7]
            b_s_t2_1_ch = choice[4]
            b_s_t2_2_ch = choice[6]
            
            a_a_matrix_t_2_ch = np.hstack((a_s_t2_1_ch,a_s_t2_2_ch)) # At 13 change
            b_b_matrix_t_2_ch = np.hstack((b_s_t2_1_ch, b_s_t2_2_ch))# At 13 change 
           
        if a_s_t_3 == 0:
            a_s_t3_1 = session[8]
            a_s_t3_2 = session[10]
            b_s_t3_1 = session[9]
            b_s_t3_2 = session[11]
            a_a_matrix_t_3 = np.hstack((a_s_t3_1,a_s_t3_2)) # At 13 change
            b_b_matrix_t_3 = np.hstack((b_s_t3_1, b_s_t3_2))# At 13 change
           
            a_s_t3_1_rew = reward[8]
            a_s_t3_2_rew = reward[10]
            b_s_t3_1_rew = reward[9]
            b_s_t3_2_rew = reward[11]
            
            a_a_matrix_t_3_rew = np.hstack((a_s_t3_1_rew,a_s_t3_2_rew)) # At 13 change
            b_b_matrix_t_3_rew = np.hstack((b_s_t3_1_rew, b_s_t3_2_rew))# At 13 change 
           
            
            a_s_t3_1_ch = choice[8]
            a_s_t3_2_ch = choice[10]
            b_s_t3_1_ch = choice[9]
            b_s_t3_2_ch = choice[11]
            
            a_a_matrix_t_3_ch = np.hstack((a_s_t3_1_ch,a_s_t3_2_ch)) # At 13 change
            b_b_matrix_t_3_ch = np.hstack((b_s_t3_1_ch, b_s_t3_2_ch))# At 13 change 
           
            
        elif a_s_t_3 != 0:
            a_s_t3_1 = session[9]
            a_s_t3_2 = session[11]
            b_s_t3_1 = session[8]
            b_s_t3_2 = session[10]
            a_a_matrix_t_3 = np.hstack((a_s_t3_1,a_s_t3_2)) # At 13 change
            b_b_matrix_t_3 = np.hstack((b_s_t3_1, b_s_t3_2))# At 13 change
      

            a_s_t3_1_rew = reward[9]
            a_s_t3_2_rew = reward[11]
            b_s_t3_1_rew = reward[8]
            b_s_t3_2_rew = reward[10]
            
            a_a_matrix_t_3_rew = np.hstack((a_s_t3_1_rew,a_s_t3_2_rew)) # At 13 change
            b_b_matrix_t_3_rew = np.hstack((b_s_t3_1_rew, b_s_t3_2_rew))# At 13 change 
           
            
            a_s_t3_1_ch = choice[9]
            a_s_t3_2_ch = choice[11]
            b_s_t3_1_ch = choice[8]
            b_s_t3_2_ch = choice[10]
            
            a_a_matrix_t_3_ch = np.hstack((a_s_t3_1_ch,a_s_t3_2_ch)) # At 13 change
            b_b_matrix_t_3_ch = np.hstack((b_s_t3_1_ch, b_s_t3_2_ch))# At 13 change 
           
        a_a_matrix_t_1_list.append(a_a_matrix_t_1)
        b_b_matrix_t_1_list.append(b_b_matrix_t_1)
        a_a_matrix_t_2_list.append(a_a_matrix_t_2) 
        b_b_matrix_t_2_list.append(b_b_matrix_t_2)
        a_a_matrix_t_3_list.append(a_a_matrix_t_3)
        b_b_matrix_t_3_list.append(b_b_matrix_t_3)
        
        
        a_a_matrix_t_1_list_rewards.append(a_a_matrix_t_1_rew)
        b_b_matrix_t_1_list_rewards.append(b_b_matrix_t_1_rew)
        a_a_matrix_t_2_list_rewards.append(a_a_matrix_t_2_rew)
        b_b_matrix_t_2_list_rewards.append(b_b_matrix_t_2_rew)
        a_a_matrix_t_3_list_rewards.append(a_a_matrix_t_3_rew)
        b_b_matrix_t_3_list_rewards.append(b_b_matrix_t_3_rew)
         
        
        a_a_matrix_t_1_list_choices.append(a_a_matrix_t_1_ch)
        b_b_matrix_t_1_list_choices.append(b_b_matrix_t_1_ch)
        a_a_matrix_t_2_list_choices.append(a_a_matrix_t_2_ch)
        b_b_matrix_t_2_list_choices.append(b_b_matrix_t_2_ch)
        a_a_matrix_t_3_list_choices.append(a_a_matrix_t_3_ch)
        b_b_matrix_t_3_list_choices.append(b_b_matrix_t_3_ch)
         
        
    # A and B in each task  
    a_a_matrix_t_1_list= np.asarray(a_a_matrix_t_1_list)
    b_b_matrix_t_1_list= np.asarray(b_b_matrix_t_1_list)
    a_a_matrix_t_2_list= np.asarray(a_a_matrix_t_2_list)
    b_b_matrix_t_2_list= np.asarray(b_b_matrix_t_2_list)
    a_a_matrix_t_3_list= np.asarray(a_a_matrix_t_3_list)
    b_b_matrix_t_3_list= np.asarray(b_b_matrix_t_3_list)
    
    
    a_a_matrix_t_1_list_rewards = np.asarray(a_a_matrix_t_1_list_rewards)
    b_b_matrix_t_1_list_rewards = np.asarray(b_b_matrix_t_1_list_rewards)
    a_a_matrix_t_2_list_rewards = np.asarray(a_a_matrix_t_2_list_rewards)
    b_b_matrix_t_2_list_rewards = np.asarray(b_b_matrix_t_2_list_rewards)
    a_a_matrix_t_3_list_rewards = np.asarray(a_a_matrix_t_3_list_rewards)
    b_b_matrix_t_3_list_rewards = np.asarray(b_b_matrix_t_3_list_rewards)
     
    
    a_a_matrix_t_1_list_choices = np.asarray(a_a_matrix_t_1_list_choices)
    b_b_matrix_t_1_list_choices = np.asarray(b_b_matrix_t_1_list_choices)
    a_a_matrix_t_2_list_choices = np.asarray(a_a_matrix_t_2_list_choices)
    b_b_matrix_t_2_list_choices = np.asarray(b_b_matrix_t_2_list_choices)
    a_a_matrix_t_3_list_choices = np.asarray(a_a_matrix_t_3_list_choices)
    b_b_matrix_t_3_list_choices = np.asarray(b_b_matrix_t_3_list_choices)

    
    return  a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
            a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
            a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
            a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
            a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
            a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
            a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
            a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
            a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices

def raw_data_time_warp_beh_choices_rewards(data, experiment_aligned_data):
    
    ''' Extract raw data aligned on behaviour switch including choices, rewards and firing rate'''

    dm = data['DM'][0]
    firing = data['Data'][0]

    res_list = []
    list_block_changes = []
    trials_since_block_list = []
    state_list = []
    task_list = []
    choice_list = []
    reward_list = []
    for  s, sess in enumerate(dm):
        
        
        DM = dm[s]
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
     
        task = DM[:,5]
        task_ind = np.where(np.diff(task)!=0)[0]
        
        firing_rates = firing[s] 
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]

        if len(ind_block) >= 12:
         
            #Because moving average resets --> calucate corrects for all tasks
            
            task_1_state = state[:task_ind[0]]
            task_2_state=  state[task_ind[0]:task_ind[1]]
            task_3_state = state[task_ind[1]:]
            task_1_choice = choices[:task_ind[0]]
            task_2_choice=  choices[task_ind[0]:task_ind[1]]
            task_3_choice = choices[task_ind[1]:]
            correct_ind_task_1 = np.where(task_1_state == task_1_choice)
            correct_ind_task_2 = np.where(task_2_state == task_2_choice)
            correct_ind_task_3 = np.where(task_3_state == task_3_choice)

            correct_task_1 = np.zeros(len(task_1_state))
            correct_task_1[correct_ind_task_1] = 1
            correct_task_2 = np.zeros(len(task_2_state))
            correct_task_2[correct_ind_task_2] = 1
            correct_task_3 = np.zeros(len(task_3_state))
            correct_task_3[correct_ind_task_3] = 1

            # Calculate movign average to determine behavioural switches
            mov_av_task_1 = ut.exp_mov_ave(correct_task_1,initValue = 0.5,tau = 8)
            mov_av_task_2 = ut.exp_mov_ave(correct_task_2,initValue = 0.5,tau = 8)
            mov_av_task_3 = ut.exp_mov_ave(correct_task_3,initValue = 0.5,tau = 8)
            mov_av = np.concatenate((mov_av_task_1,mov_av_task_2,mov_av_task_3))
            moving_av_0_6 = np.where(mov_av > 0.63)[0]
            
 
            b_1 = [m for m  in moving_av_0_6 if m in np.where(block == 0)[0]]
            b_2 = [m for m  in moving_av_0_6 if m in np.where(block == 1)[0]]
            b_3 = [m for m  in moving_av_0_6 if m in np.where(block == 2)[0]]
            b_4 = [m for m  in moving_av_0_6 if m in np.where(block == 3)[0]]
            b_5 = [m for m  in moving_av_0_6 if m in np.where(block == 4)[0]]
            b_6 = [m for m  in moving_av_0_6 if m in np.where(block == 5)[0]]
            b_7 = [m for m  in moving_av_0_6 if m in np.where(block == 6)[0]]
            b_8 = [m for m  in moving_av_0_6 if m in np.where(block == 7)[0]]
            b_9 = [m for m  in moving_av_0_6 if m in np.where(block == 8)[0]]
            b_10 = [m for m  in moving_av_0_6 if m in np.where(block == 9)[0]]
            b_11 = [m for m  in moving_av_0_6 if m in np.where(block == 10)[0]]
            b_12 = [m for m  in moving_av_0_6 if m in np.where(block == 11)[0]]

           
            all_ind_triggered_on_beh = np.concatenate((b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_10,b_11,b_12))
            ind_blocks = np.median(np.hstack((len(b_1), len(b_2), len(b_3),  len(b_4),  len(b_5),\
                len(b_6),  (len(b_7), len(b_8),  len(b_9),  len(b_10),\
                      len(b_11),len(b_12)))))
                
            trials_since_block = np.hstack((np.arange(len(b_1)), np.arange(len(b_2)), np.arange(len(b_3)),np.arange(len(b_4)),\
                                            np.arange(len(b_5)), np.arange(len(b_6)), np.arange(len(b_7)),  np.arange(len(b_8)),\
                                            np.arange(len(b_9)), np.arange(len(b_10)), np.arange(len(b_11)),\
                                                                                               np.arange(len(b_12))))
            firing_rates  = firing_rates[all_ind_triggered_on_beh]
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            
            state = state[all_ind_triggered_on_beh]
            task = task[all_ind_triggered_on_beh]
            
            choices = choices[all_ind_triggered_on_beh]
            reward = reward[all_ind_triggered_on_beh]
  
    
            res_list.append(firing_rates)
            trials_since_block_list.append(trials_since_block)
            list_block_changes.append(ind_blocks)
            state_list.append(state)
            task_list.append(task)
            choice_list.append(choices)
            reward_list.append(reward)
  
    return res_list, list_block_changes, trials_since_block_list, state_list,task_list,reward_list,choice_list



def raw_data_align_choices_rewards(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC):
    
    ''' Get the number of trials from alinged to behaviour data in PFC and HP'''

    res_list_HP, list_block_changes_HP, trials_since_block_list_HP, state_list_HP,task_list_HP,reward_list_HP,choice_list_HP = raw_data_time_warp_beh_choices_rewards(data_HP,experiment_aligned_HP)
    res_list_PFC, list_block_changes_PFC, trials_since_block_list_PFC, state_list_PFC,task_list_PFC,reward_list_PFC,choice_list_PFC = raw_data_time_warp_beh_choices_rewards(data_PFC,experiment_aligned_PFC)
    target_trials = np.median([np.median(list_block_changes_HP),np.median(list_block_changes_PFC)])
    
    
    return res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,trials_since_block_list_PFC, state_list_HP,state_list_PFC,task_list_HP, task_list_PFC, \
        reward_list_HP,reward_list_PFC, choice_list_HP,choice_list_PFC


def all_sessions_align_beh_raw_data_choices_rewards(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end):
    
    ''' Trial warp data aligned to behaviour '''

    res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,trials_since_block_list_PFC, state_list_HP,state_list_PFC,task_list_HP, task_list_PFC, \
        reward_list_HP,reward_list_PFC, choice_list_HP,choice_list_PFC  = raw_data_align_choices_rewards(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC)
    
    HP_aligned_time  = []
    HP_aligned_choices  = []
    HP_aligned_rewards  = []

    PFC_aligned_time  = []
    PFC_aligned_choices  = []
    PFC_aligned_rewards  = []
    
    smooth_SD = 1
    edge  = 2
    
    # HP
    for res,residuals in enumerate(res_list_HP):
        trials_since_block = trials_since_block_list_HP[res]
        ends = np.where(np.diff(trials_since_block)!=1)[0]
        activity = np.mean(res_list_HP[res][:,:,start:end],axis = 2).T # Find mean acrosss the trial 
        starts = np.sort(np.append(ends[:-1:1],0))    
        starts = np.append(starts,ends[-1])
        ends = np.append(ends,len(trials_since_block))
        frame_times = np.arange(len(trials_since_block))
        trial_times = np.vstack((starts,ends)).T
        target_times = np.vstack((starts,starts+int(target_trials))).T
        aligned_activity = align_activity_firing(activity, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)
        
        choices_align = choice_list_HP[res]
        reward_align = reward_list_HP[res]
        choices_aligned = align_activity_choices_rewards(choices_align, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)
        reward_aligned = align_activity_choices_rewards(reward_align, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)

        HP_aligned_time.append(aligned_activity)
        HP_aligned_choices.append(choices_aligned)
        HP_aligned_rewards.append(reward_aligned)

        
    for res,residuals in enumerate(res_list_PFC):
        trials_since_block = trials_since_block_list_PFC[res]
        ends = np.where(np.diff(trials_since_block)!=1)[0]
        activity = np.mean(res_list_PFC[res][:,:,start:end],axis = 2).T # Find mean acrosss the trial 
        starts = np.sort(np.append(ends[:-1:1],0))    
        starts = np.append(starts,ends[-1])
        ends = np.append(ends,len(trials_since_block))
        frame_times = np.arange(len(trials_since_block))
        trial_times = np.vstack((starts,ends)).T
        target_times = np.vstack((starts,starts+int(target_trials))).T

        aligned_activity = align_activity_firing(activity, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)
        
        choices_align = choice_list_PFC[res]
        reward_align = reward_list_PFC[res]
        choices_aligned = align_activity_choices_rewards(choices_align, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)
        reward_aligned = align_activity_choices_rewards(reward_align, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)

        PFC_aligned_time.append(aligned_activity)
        PFC_aligned_choices.append(choices_aligned)
        PFC_aligned_rewards.append(reward_aligned)

    return HP_aligned_time, HP_aligned_choices, HP_aligned_rewards, PFC_aligned_time,\
        PFC_aligned_choices, PFC_aligned_rewards,state_list_HP,state_list_PFC,task_list_HP, task_list_PFC




def align_activity_firing(activity, frame_times, trial_times, target_times, smooth_SD = 1, plot_warp=False,  edge = 2):
    '''
    Timewarp neuronal activity to align event times on each trial to specified target
    event times. For each trial, input frame times are linearly time warped to align
    that trial's event times with the target times.  Activity is then evaluated at a set of
    regularly spaced timepoints relative to the target event times by linearly interpolating
    activity between input frames followed by Gaussian smoothing around output timepoints.
    This allows a single mathematical operation to handle both time streching (where 
    interpolation in needed) and time compression (where averaging is needed).
    It is recomended to use a 2X higher sampling rate for the output activity than that 
    of the input activity as due to random jitter across trials between frame times
    and event times, there is information on a finer temporal resolution than the raw input
    frame rate.
    Optionally, pre_win and post_win arguments can be used to specify time windows before 
    the first and after the last alignment event on each trial to be inlcuded in the output 
    activity.
    Arguments: 
    activity     : Neuronal activity [n_neurons, n_frames]
    frame_times  : Times when the scope frames occured (ms) [n_frames]
    trial_times  : Times of events used for alignment for each trial (ms) [n_trials, n_events]
    target_times : Times of events used for alignment in output aligned trial (ms) [n_events].
    smooth_SD    : Standard deviation (ms) of Gaussian smoothing applied to output activity.
                   If set to 'auto', smooth_SD is set to 1000/fs_out.
    plot_warp    : If True the input and output activity are plotted for the most active 
                   neurons for each trial.
   '''
    assert not np.any(np.diff(trial_times,1)<0), 'trial_times give negative interval duration'
    assert not np.any(np.diff(target_times)<0) , 'target_times give negative interval duration'
    
    target_times_1 = target_times[:,0] - edge
    target_times_2 = target_times[:,1] + edge
    target_times = np.vstack([target_times_1,target_times_2]).T
    trial_time_pre = trial_times[:,0]
    trial_time_post = trial_times[:,1]
    trial_times = np.vstack((trial_time_pre,trial_time_post)).T
    t_out = np.arange(target_times[0,0], target_times[0,1])
   
    align_IEI = (target_times[0][1] - target_times[0][0]) # Duration of inter event intervals for aligned activity (ms).
    trial_IEI = np.diff(trial_times,1) # Duration of inter event intervals for each trial (ms).

    n_trials, n_neurons, n_timepoints = (trial_times.shape[0], activity.shape[0], len(t_out))
    aligned_activity = np.full([n_trials, n_neurons, n_timepoints], np.nan)
    
    for i in np.arange(n_trials):
      
            # Linearly warp frame times to align inter event intervals to target.
            trial_frames = ((trial_times[i,0] <= frame_times) & (frame_times < trial_times[i,-1]))
            trial_activity = activity[:,np.where(trial_frames)[0]]
            
            # ## Add same firing rates 3 times at the end and the beginning 
            trial_activity = np.insert(trial_activity,0, trial_activity[:,0],1)
            trial_activity = np.insert(trial_activity,0, trial_activity[:,0],1)
            
            trial_activity = np.insert(trial_activity,-1, trial_activity[:,-1],1)
            trial_activity = np.insert(trial_activity,-1, trial_activity[:,-1],1)
          
            t0 = frame_times[trial_frames]
            
            #t1 = np.zeros(trial_activity.shape[1])          # Trial frame times after warping
            t1 = np.zeros(len(t0))          # Trial frame times after warping
            mask = (trial_times[i,0] <= t0) & (t0 < trial_times[i,1])

            t1[mask]= (t0[mask]-trial_times[i][0])*align_IEI/trial_IEI[i] + target_times[i][1]
            t1 = t1-t1[0]
      
            # # Trick to get the time right to get the edge effects out
            t1 = np.insert(t1,0,-1)
            t1 = np.insert(t1,0,-2)
            t1 = np.append(t1,(t1[-1]+1))
            t1 = np.append(t1,(t1[-1]+1))
           
            # # Calculate aligned activity using analytical solution to overlap integral between
            # linearly interpolated activity and gaussian around output timepoints.

            aligned_activity[i,:,:] = (np.sum(_int_norm_lin_prod(trial_activity[:,:-1],
                trial_activity[:,1:],t1[:-1],t1[1:],t_out[:,None,None],s=smooth_SD),2).T)
            
            if plot_warp: # Plot input and output activity for the most active neurons.
                isl  = wes.Royal3_5.mpl_colors

                most_active = np.argsort(np.mean(trial_activity,1))[-1:]
                plt.figure(2, figsize=[10,3.2]).clf()
                plt.subplot2grid((1,3),(0,0))
                plt.plot(t0,t1[edge:-edge],'.-')

                plt.ylabel('Aligned trial time (ms)')
                plt.xlabel('True trial time (ms)')
                plt.subplot2grid((2,3),(0,1), colspan=2)
                plt.plot(t1[edge:-edge],trial_activity[most_active,edge:-edge].T,color = isl[0])
                plt.plot(t1,trial_activity[most_active,:].T,'--', color = isl[0])

                plt.ylabel('Activity')
                plt.xlabel('True trial time (ms)')
                plt.subplot2grid((2,3),(1,1), colspan=2)
                plt.plot(t_out[edge:-edge],aligned_activity[i,most_active,edge:-edge].T,'.-',color = isl[0])
                plt.plot(t_out,aligned_activity[i,most_active,:].T,'--',color = isl[0])

                #plt.xlim(t_out[0], t_out[-1])
                plt.ylabel('Activity')
                plt.xlabel('Aligned trial time (ms)')
                plt.tight_layout()
                plt.pause(0.05)
                if input("Press enter for next trial, 'x' to stop plotting:") == 'x':
                    plot_warp  = False
    aligned_activity = aligned_activity[:,:,edge:-edge] 

    return aligned_activity


def align_activity_choices_rewards(activity, frame_times, trial_times, target_times, smooth_SD = 1, plot_warp=False,  edge = 2):
    '''
    Timewarp neuronal activity to align event times on each trial to specified target
    event times. For each trial, input frame times are linearly time warped to align
    that trial's event times with the target times.  Activity is then evaluated at a set of
    regularly spaced timepoints relative to the target event times by linearly interpolating
    activity between input frames followed by Gaussian smoothing around output timepoints.
    This allows a single mathematical operation to handle both time streching (where 
    interpolation in needed) and time compression (where averaging is needed).
    It is recomended to use a 2X higher sampling rate for the output activity than that 
    of the input activity as due to random jitter across trials between frame times
    and event times, there is information on a finer temporal resolution than the raw input
    frame rate.
    Optionally, pre_win and post_win arguments can be used to specify time windows before 
    the first and after the last alignment event on each trial to be inlcuded in the output 
    activity.
    Arguments: 
    activity     : Choices [n_frames]
    frame_times  : Times when the scope frames occured (ms) [n_frames]
    trial_times  : Times of events used for alignment for each trial (ms) [n_trials, n_events]
    target_times : Times of events used for alignment in output aligned trial (ms) [n_events].
    smooth_SD    : Standard deviation (ms) of Gaussian smoothing applied to output activity.
                   If set to 'auto', smooth_SD is set to 1000/fs_out.
    plot_warp    : If True the input and output activity are plotted for the most active 
                   neurons for each trial.
   '''
    assert not np.any(np.diff(trial_times,1)<0), 'trial_times give negative interval duration'
    assert not np.any(np.diff(target_times)<0) , 'target_times give negative interval duration'
    target_times_1 = target_times[:,0] - edge
    target_times_2 = target_times[:,1] + edge
    target_times = np.vstack([target_times_1,target_times_2]).T
    trial_time_pre = trial_times[:,0]
    trial_time_post = trial_times[:,1]
    trial_times = np.vstack((trial_time_pre,trial_time_post)).T
    t_out = np.arange(target_times[0,0], target_times[0,1])
   
    align_IEI = (target_times[0][1] - target_times[0][0]) # Duration of inter event intervals for aligned activity (ms).
    trial_IEI = np.diff(trial_times,1) # Duration of inter event intervals for each trial (ms).

    n_trials, n_timepoints = (trial_times.shape[0], len(t_out))
    aligned_activity = np.full([n_trials, n_timepoints], np.nan)
    for i in np.arange(n_trials):
      
            # Linearly warp frame times to align inter event intervals to target.
            trial_frames = ((trial_times[i,0] <= frame_times) & (frame_times < trial_times[i,-1]))
            trial_activity = activity[np.where(trial_frames)[0]]
            
            # ## Add same firing rates 3 times at the end and the beginning 
            trial_activity = np.insert(trial_activity,0, trial_activity[0],0)
            trial_activity = np.insert(trial_activity,0, trial_activity[0],0)
            
            trial_activity = np.insert(trial_activity,-1, trial_activity[-1],0)
            trial_activity = np.insert(trial_activity,-1, trial_activity[-1],0)
          
            t0 = frame_times[trial_frames]
            
            #t1 = np.zeros(trial_activity.shape[1])          # Trial frame times after warping
            t1 = np.zeros(len(t0))          # Trial frame times after warping
            mask = (trial_times[i,0] <= t0) & (t0 < trial_times[i,1])

            t1[mask]= (t0[mask]-trial_times[i][0])*align_IEI/trial_IEI[i] + target_times[i][1]
            t1 = t1-t1[0]
      
            # # Trick to get the time right to get the edge effects out
            t1 = np.insert(t1,0,-1)
            t1 = np.insert(t1,0,-2)
            t1 = np.append(t1,(t1[-1]+1))
            t1 = np.append(t1,(t1[-1]+1))
           
            # # Calculate aligned rewards/choices using analytical solution to overlap integral between
            # linearly interpolated activity and gaussian around output timepoints.
            
              
            
            aligned_activity[i,:] = (np.sum(_int_norm_lin_prod(trial_activity[:-1],
                trial_activity[1:],t1[:-1],t1[1:],t_out[:,None,None],s=smooth_SD),2).T)
            
            if plot_warp: # Plot input and output activity for the most active neurons.
                isl  = wes.Royal3_5.mpl_colors

                plt.figure(2, figsize=[10,3.2]).clf()
                plt.subplot2grid((1,3),(0,0))
                plt.plot(t0,t1[edge:-edge],'.-')

                plt.ylabel('Aligned trial time (ms)')
                plt.xlabel('True trial time (ms)')
                plt.subplot2grid((2,3),(0,1), colspan=2)
                plt.plot(t1[edge:-edge],trial_activity[edge:-edge].T,color = isl[0])
                plt.plot(t1,trial_activity.T,'--', color = isl[0])

                plt.ylabel('Activity')
                plt.xlabel('True trial time (ms)')
                plt.subplot2grid((2,3),(1,1), colspan=2)
                plt.plot(t_out[edge:-edge],aligned_activity[i,edge:-edge].T,'.-',color = isl[0])
                plt.plot(t_out,aligned_activity[i,:].T,'--',color = isl[0])

                #plt.xlim(t_out[0], t_out[-1])
                plt.ylabel('Activity')
                plt.xlabel('Aligned trial time (ms)')
                plt.tight_layout()
                plt.pause(0.05)
                if input("Press enter for next trial, 'x' to stop plotting:") == 'x':
                    plot_warp  = False
    aligned_activity = aligned_activity[:,edge:-edge] 

    return aligned_activity


# ----------------------------------------------------------------------------------

r2pi = np.sqrt(2*np.pi)
r2   = np.sqrt(2) 


def _int_norm_lin_prod(a,b,v,t,u,s):

    '''Evaluate the integral w.r.t. x of (a+(b-a)*(x-v)/(t-v))*Npdf(u,s) from v to t where 
    Npdf is the Normal distribution probability density function. Wolfram Integrator: 
    integrate ((a+(b-a)*(x-v)/(t-v))/(s*sqrt(2*pi)))*exp(-((x-u)^2)/(2*s^2)) dx from v to t'''

    return (1/(2*r2pi*(t-v)))*(r2pi*(a*(t-u)+b*(u-v))*(erf((t-u)/(r2*s))-erf((v-u)/(r2*s)))+
            2*s*(a-b)*(np.exp(-((t-u)**2)/(2*s**2))-np.exp(-((v-u)**2)/(2*s**2))))


def _int_norm_lin_prod_no_s(a,b,v,t):

    '''Evaluate the integral w.r.t. x of (a+(b-a)*(x-v)/(t-v))*Npdf(u,s) from v to t where 
    Npdf is the Normal distribution probability density function. Wolfram Integrator: 
    integrate ((a+(b-a)*(x-v)/(t-v))/(s*sqrt(2*pi)))*exp(-((x-u)^2)/(2*s^2)) dx from v to t'''

    return (1/(2*r2pi*(t-v))+ 2*(a-b))



