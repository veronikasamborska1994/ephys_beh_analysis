#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:45:43 2020

@author: veronikasamborska
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:37:07 2020

@author: veronikasamborska
"""

from sklearn.linear_model import LogisticRegression
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
# import sys 
# import statsmodels.api as sm
# from statsmodels.stats.anova import AnovaRM
# import pandas as pd
# from palettable import wesanderson as wes
from scipy import stats 
import random
import seaborn as sns


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}
import scipy
     
def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid


def rew_prev_behaviour(data,n, perm = True):
   
    dm = data['DM'][0]
    results_array = []
    std_err = []
    
    for  s, sess in enumerate(dm):
            
         DM = dm[s] 
           
         choices = DM[:,1]
         
         reward = DM[:,2]
         
         previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
         
         previous_choices = scipy.linalg.toeplitz(choices-0.5, np.zeros((1,n)))[n-1:-1]
         
         interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
         

         choices_current = choices[n:]
         ones = np.ones(len(interactions)).reshape(len(interactions),1)
         
         X = np.hstack([previous_rewards,previous_choices,interactions,ones])
         
         model = LogisticRegression()
         results = model.fit(X,choices_current)
         results_array.append(results.coef_[0])
         #cov = results.cov_params()
         #std_err.append(np.sqrt(np.diag(cov)))

    average = np.mean(results_array,0)
    #std = np.std(results_array,0)/len(dm)

    # average = np.mean(results_array,0)
    # c = 'green'
    # plt.plot(np.arange(len(average))[n*2:-1], average[n*2:-1], color = c, label = 'PFC')
    # plt.fill_between(np.arange(len(average))[n*2:-1], average[n*2:-1]+std[n*2:-1], average[n*2:-1]- std[n*2:-1],alpha = 0.2, color =c)
    # plt.hlines(0, xmin = np.arange(len(average))[n*2:-1][0],xmax = np.arange(len(average))[n*2:-1][-1])
    # length = len(np.arange(len(average))[n*2:-1])
    # plt.xticks(np.arange(len(average))[n*2:-1],np.arange(length))
    # sns.despine()
    # plt.legend()
    
    return average

def regression_code_session(data, design_matrix): 
    
    tc = np.identity(design_matrix.shape[1])
    
    pdes = np.linalg.pinv(design_matrix)
    tc_pdes = np.matmul(tc,pdes)
    pdes_tc = np.matmul(np.transpose(pdes),np.transpose(tc))
    
    prevar = np.diag(np.matmul(tc_pdes, pdes_tc))
    
    R = np.identity(design_matrix.shape[0]) - np.matmul(design_matrix, pdes)
    tR = np.trace(R)
    
    pe = np.matmul(pdes,data)
    cope = np.matmul(tc,pe)
    
    res = data - np.matmul(design_matrix,pe)
    sigsq = np.sum((res*res)/tR, axis = 0)
    sigsq = np.reshape(sigsq,(1,res.shape[1]))
    prevar = np.reshape(prevar,(tc.shape[0],1))
    varcope = prevar*sigsq
    
    #tstats = cope/np.sqrt(varcope)
    
    return cope,varcope


def value_reg_svd(data, n = 10, first_half = 1, a ='PFC'):
  
   # dm = data['DM'][0]
   # firing = data['Data'][0]

    average = rew_prev_behaviour(data, n = n)
    if a == 'PFC':
        all_subjects = [data['DM'][0][:9], data['DM'][0][9:25],data['DM'][0][25:39],data['DM'][0][39:]]
        all_firing = [data['Data'][0][:9], data['Data'][0][9:25],data['Data'][0][25:39],data['Data'][0][39:]]
    else:   
        all_subjects = [data['DM'][0][:16], data['DM'][0][16:24],data['DM'][0][24:]]
        all_firing = [data['Data'][0][:16], data['Data'][0][16:24],data['Data'][0][24:]]
        
    C_1_all = []; C_2_all = []; C_3_all = []
  
    for d,dd in enumerate(all_subjects):
        C_1 = []; C_2 = []; C_3 = []

        dm = all_subjects[d]
        firing = all_firing[d]

    
        for  s, sess in enumerate(dm):
            
           
            DM = dm[s]
            block = DM[:,4]
            block_df = np.diff(block)
            ind_block = np.where(block_df != 0)[0]
        
            if len(ind_block) >11  :
                firing_rates = firing[s]#[:ind_block[11]]
               # firing_rates = scipy.stats.zscore(firing_rates,0)

                n_trials, n_neurons, n_timepoints = firing_rates.shape
                choices = DM[:,1]#[:ind_block[11]]
                reward = DM[:,2]#[:ind_block[11]]  
                state = DM[:,0]
                task =  DM[:,5]#[:ind_block[11]]
               
                a_pokes = DM[:,6]#[:ind_block[11]]
                b_pokes = DM[:,7]#[:ind_block[11]]
                
                taskid = task_ind(task, a_pokes, b_pokes)
              
                
                task_1 = np.where(taskid == 1)[0]
                task_2 = np.where(taskid == 2)[0]
                task_3 = np.where(taskid == 3)[0]
                # plt.figure()
                # plt.imshow(np.mean(firing_rates,2).T, aspect ='auto')
        
        
                reward_current = reward
                choices_current = choices-0.5
        
               
                rewards_1 = reward_current[task_1]
                choices_1 = choices_current[task_1]
                
                previous_rewards_1 = scipy.linalg.toeplitz(rewards_1, np.zeros((1,n)))[n-1:-1]         
                previous_choices_1 = scipy.linalg.toeplitz(0.5-choices_1, np.zeros((1,n)))[n-1:-1]       
                interactions_1 = scipy.linalg.toeplitz((((0.5-choices_1)*(rewards_1-0.5))*2),np.zeros((1,n)))[n-1:-1]
                 
        
                ones = np.ones(len(interactions_1)).reshape(len(interactions_1),1)
                 
                X_1 = np.hstack([previous_rewards_1,previous_choices_1,interactions_1,ones])
                value_1 =np.matmul(X_1, average)
        
                rewards_1 = rewards_1[n:]
                choices_1 = choices_1[n:]
                state_1 = state[task_1]
                state_1 = state_1[n:]
 
                value_1_choice_1 = choices_1 * value_1

                ones_1 = np.ones(len(choices_1))
                trials_1 = len(choices_1)
                
               
              
                firing_rates_1 = firing_rates[task_1][n:]
                
                # state_g_a_1 = np.where(value_1_choice_1 > np.median(value_1_choice_1))[0]
                # state_g_b_1 = np.where(value_1_choice_1 < np.median(value_1_choice_1))[0]

                # ind_a =int(len(state_g_a_1)/2)
                # ind_b = int(len(state_g_b_1)/2)
                # ind_1 = np.concatenate([state_g_a_1[:ind_a],state_g_b_1[:ind_b]])
                # ind_2 = np.concatenate([state_g_b_1[ind_a:],state_g_a_1[ind_b:]])
                ind_1 = np.arange(len(ones_1))[::2]
                ind_2 = np.arange(len(ones_1))[1::2]

                if first_half == 1:
                    rewards_1 = rewards_1[ind_1] 
                    choices_1 = choices_1[ind_1]    
                    value_1 = value_1[ind_1]
                    ones_1  = ones_1[ind_1]
                    value_1_choice_1 =value_1_choice_1[ind_1]
                    firing_rates_1 = firing_rates_1[ind_1]
                   
                elif first_half == 2:
              
                    rewards_1 = rewards_1[ind_2] 
                    choices_1 = choices_1[ind_2]    
                    value_1 = value_1[ind_2]
                    ones_1  = ones_1[ind_2]
                    value_1_choice_1 =value_1_choice_1[ind_2]
                    firing_rates_1 = firing_rates_1[ind_2]
                       
                predictors_all = OrderedDict([
                                            ('Choice', choices_1),
                                            ('Rew', rewards_1),
                                            ('Value',value_1), 
                                            ('Value Сhoice',value_1_choice_1), 
                                           # ])

                                            ('ones', ones_1)])
                
                X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
                
                n_predictors = X_1.shape[1]
                y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
                tstats,x = regression_code_session(y_1, X_1)
                #tstats =  reg_f.regression_code(y_1, X_1)
        
                C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
                
                
                rewards_2 = reward_current[task_2]
                choices_2 = choices_current[task_2]
                
                previous_rewards_2 = scipy.linalg.toeplitz(rewards_2, np.zeros((1,n)))[n-1:-1]      
                previous_choices_2 = scipy.linalg.toeplitz(0.5-choices_2, np.zeros((1,n)))[n-1:-1]        
                interactions_2 = scipy.linalg.toeplitz((((0.5-choices_2)*(rewards_2-0.5))*2),np.zeros((1,n)))[n-1:-1]
                 
        
                ones = np.ones(len(interactions_2)).reshape(len(interactions_2),1)
                 
                X_2 = np.hstack([previous_rewards_2,previous_choices_2,interactions_2,ones])
                value_2 =np.matmul(X_2, average)
        
                rewards_2 = rewards_2[n:]
                choices_2 = choices_2[n:]
                state_2 = state[task_2]
                state_2 = state_2[n:]
    
                
                ones_2 = np.ones(len(choices_2))
                trials_2 = len(choices_2)
        
                firing_rates_2 = firing_rates[task_2][n:]
                
                value_2_choice_2 = choices_2 * value_2

                state_g_a_1 = np.where(value_2_choice_2 > np.median(value_2_choice_2))[0]
                state_g_b_1 = np.where(value_2_choice_2 < np.median(value_2_choice_2))[0]

                # ind_a =int(len(state_g_a_1)/2)
                # ind_b = int(len(state_g_b_1)/2)
                # ind_1 = np.concatenate([state_g_a_1[:ind_a],state_g_b_1[:ind_b]])
                # ind_2 = np.concatenate([state_g_b_1[ind_a:],state_g_a_1[ind_b:]])
                ind_1 = np.arange(len(ones_2))[::2]
                ind_2 = np.arange(len(ones_2))[1::2]

                if first_half == 1:
                    rewards_2 = rewards_2[ind_1] 
                    choices_2 = choices_2[ind_1]    
                    value_2 = value_2[ind_1]
                    ones_2  = ones_2[ind_1]
                    value_2_choice_2 =value_2_choice_2[ind_1]
                    firing_rates_2 = firing_rates_2[ind_1]
                   
                elif first_half == 2:
              
                    rewards_2 = rewards_2[ind_2] 
                    choices_2 = choices_2[ind_2]    
                    value_2 = value_2[ind_2]
                    ones_2  = ones_2[ind_2]
                    firing_rates_2 = firing_rates_2[ind_2]
                    value_2_choice_2 =value_2_choice_2[ind_2]
  
                 
                predictors_all = OrderedDict([
                                             ('Choice', choices_2),
                                            ('Rew', rewards_2),
                                            ('Value',value_2), 
                                            ('Value Сhoice',value_2_choice_2), 
                                           # ])

                                           ('ones', ones_2)])
                
                X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
                
                n_predictors = X_2.shape[1]
                y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
                tstats,x = regression_code_session(y_2, X_2)
        
                C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
          
            
                
                rewards_3 = reward_current[task_3]
                choices_3 = choices_current[task_3]
                
                previous_rewards_3 = scipy.linalg.toeplitz(rewards_3, np.zeros((1,n)))[n-1:-1]
                 
                previous_choices_3 = scipy.linalg.toeplitz(0.5-choices_3, np.zeros((1,n)))[n-1:-1]
                 
                interactions_3 = scipy.linalg.toeplitz((((0.5-choices_3)*(rewards_3-0.5))*2),np.zeros((1,n)))[n-1:-1]
                 
        
                ones = np.ones(len(interactions_3)).reshape(len(interactions_3),1)
                 
                X_3 = np.hstack([previous_rewards_3,previous_choices_3,interactions_3,ones])
                value_3 =np.matmul(X_3, average)
        
                rewards_3 = rewards_3[n:]
                choices_3 = choices_3[n:]
                state_3 = state[task_3]
                state_3 = state_3[n:]
                value_3_choice_3 = choices_3 * value_3

                
                ones_3 = np.ones(len(choices_3))
                trials_3 = len(choices_3)
        
                firing_rates_3 = firing_rates[task_3][n:]
                
                # state_g_a_1 = np.where(value_3_choice_3 > np.median(value_3_choice_3))[0]
                # state_g_b_1 = np.where(value_3_choice_3 < np.median(value_3_choice_3))[0]

                # ind_a =int(len(state_g_a_1)/2)
                # ind_b = int(len(state_g_b_1)/2)
                # ind_1 = np.concatenate([state_g_a_1[:ind_a],state_g_b_1[:ind_b]])
                # ind_2 = np.concatenate([state_g_b_1[ind_a:],state_g_a_1[ind_b:]])
              
                ind_1 = np.arange(len(ones_3))[::2]
                ind_2 = np.arange(len(ones_3))[1::2]

                if first_half == 1:
                    rewards_3 = rewards_3[ind_1] 
                    choices_3 = choices_3[ind_1]    
                    value_3 = value_3[ind_1]
                    ones_3  = ones_3[ind_1]
                    firing_rates_3 = firing_rates_3[ind_1]
                    value_3_choice_3 = value_3_choice_3[ind_1]
                elif first_half == 2:
              
                    rewards_3 = rewards_3[ind_2] 
                    choices_3 = choices_3[ind_2]    
                    value_3 = value_3[ind_2]
                    ones_3  = ones_3[ind_2]
                    firing_rates_3 = firing_rates_3[ind_2]
                    value_3_choice_3 = value_3_choice_3[ind_2]

  
                predictors_all = OrderedDict([
                                            ('Choice', choices_3),
                                            ('Rew', rewards_3),
                                            ('Value',value_3), 
                                            ('Value Сhoice',value_3_choice_3), 
                                           # ])
                                            ('ones', ones_3)])
                
                X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
                rank = np.linalg.matrix_rank(X_1)
                #print(rank)
                n_predictors = X_3.shape[1]
               # print(n_predictors)
                y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
                tstats,x = regression_code_session(y_3, X_3)
                #tstats =  reg_f.regression_code(y_3, X_3)
        
                C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_1_all.append(np.concatenate(C_1,1)); C_2_all.append(np.concatenate(C_2,1)); C_3_all.append(np.concatenate(C_3,1))

        
    # C_1 = np.concatenate(C_1,1)
    
    # C_2 = np.concatenate(C_2,1)
    
    # C_3 = np.concatenate(C_3,1)
   
    # C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    # C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    # C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    # nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
    # C_1 = C_1[:,nans[0],:]; C_2 = C_2[:,nans[0],:];  C_3 = C_3[:,nans[0],:]
    

      
    return C_1_all,C_2_all,C_3_all

def stats_run():
    
    u_v_area_HP, v_area_HP, u_area_HP = svd_on_coefs_chosen_v(HP, n = 11,  c = 'green', area='HP')
    u_v_area_PFC, v_area_PFC, u_area_PFC = svd_on_coefs_chosen_v(PFC, n = 11,  c = 'grey', area='PFC')

    s = stats.ttest_ind(u_v_area_HP,u_v_area_PFC)
    s = stats.ttest_ind(v_area_HP,v_area_PFC)
    s = stats.ttest_ind(u_area_HP,u_area_PFC)

def svd_on_coefs_chosen_v(data, n = 11,  c = 'grey', area='HP'):
    
    
    # n = 11
    # data = PFC
    # area = 'PFC'
    # c = 'green'
     
   
    C_1_all_1, C_2_all_1, C_3_all_1 = value_reg_svd(data, n = n,  first_half = 1, a =area)    
    C_1_all_2, C_2_all_2, C_3_all_2 = value_reg_svd(data, n = n,   first_half = 2, a =area) 
    

    u_v_area = []
    v_area = []
    u_area = []
    j = 3
    for i, ii in enumerate(C_1_all_1):
        plt.figure()
        value_1_1 = C_1_all_1[i][j]
        value_2_1 = C_2_all_1[i][j]
        value_3_1 = C_3_all_1[i][j]

        value_1_2 = C_1_all_2[i][j]
        value_2_2 = C_2_all_2[i][j]
        value_3_2 = C_3_all_2[i][j]
        
        # value_1_1 = np.concatenate(C_1_all_1,1)[j]
        # value_2_1 = np.concatenate(C_2_all_1,1)[j]
        # value_3_1 = np.concatenate(C_3_all_1,1)[j]

        # value_1_2 = np.concatenate(C_1_all_2,1)[j]
        # value_2_2 = np.concatenate(C_2_all_2,1)[j]
        # value_3_2 = np.concatenate(C_3_all_2,1)[j]
        
        #Task 1 2
        n_neurons = value_1_1.shape[0]
        u_1_2,s_1_2,v_1_2 = np.linalg.svd(value_1_2,full_matrices = False)
        tu_1_2 = np.transpose(u_1_2)  
        t_v_1_2 = np.transpose(v_1_2)
      
        within_1 = np.linalg.multi_dot([tu_1_2, value_1_1, t_v_1_2])
        within_1_d = within_1.diagonal()
        sum_within_1 = np.cumsum(abs(within_1_d))/n_neurons
       
        u_only_1_within = np.linalg.multi_dot([tu_1_2, value_1_1])
        u_only_1_within_sq = np.sum(u_only_1_within**2, axis = 1)
        u_only_1_within_sq_sum = np.cumsum(u_only_1_within_sq)/n_neurons
        u_only_1_within_sq_sum = u_only_1_within_sq_sum/u_only_1_within_sq_sum[-1]
       
        # Using V
        v_only_1_within = np.linalg.multi_dot([value_1_1,t_v_1_2])
        v_only_1_within_sq = np.sum(v_only_1_within**2, axis = 0)
        v_only_1_within_sq_sum = np.cumsum(v_only_1_within_sq)/n_neurons
        v_only_1_within_sq_sum = v_only_1_within_sq_sum/v_only_1_within_sq_sum[-1]
       
        
        between_1_2 = np.linalg.multi_dot([tu_1_2, value_2_1, t_v_1_2])
        between_1_2_d = between_1_2.diagonal()
        sum_between_1_2_d = np.cumsum(abs(between_1_2_d))/n_neurons
      
        u_only_1_2_between = np.linalg.multi_dot([tu_1_2, value_2_1])
        u_only_1_2_between_sq = np.sum(u_only_1_2_between**2, axis = 1)
        u_only_1_2_between_sq_sum = np.cumsum(u_only_1_2_between_sq)/n_neurons
        u_only_1_2_between_sq_sum = u_only_1_2_between_sq_sum/u_only_1_2_between_sq_sum[-1]
       
        # Using V
        v_only_1_2_between = np.linalg.multi_dot([value_2_1,t_v_1_2])
        v_only_1_2_between_sq = np.sum(v_only_1_2_between**2, axis = 0)
        v_only_1_2_between_sq_sum = np.cumsum(v_only_1_2_between_sq)/n_neurons
        v_only_1_2_between_sq_sum = v_only_1_2_between_sq_sum/v_only_1_2_between_sq_sum[-1]
      
         
        # Task 2 1
    
        u_2_1,s_2_1,v_2_1 = np.linalg.svd(value_2_1,full_matrices = False)
        tu_2_1 = np.transpose(u_2_1)  
        t_v_2_1 = np.transpose(v_2_1)
      
        within_2_2 = np.linalg.multi_dot([tu_2_1, value_2_2, t_v_2_1])
        within_2_2_d = within_2_2.diagonal()
        sum_within_2_2 = np.cumsum(abs(within_2_2_d))/n_neurons
       
        u_only_2_2_within = np.linalg.multi_dot([tu_2_1, value_2_2])
        u_only_2_2_within_sq = np.sum(u_only_2_2_within**2, axis = 1)
        u_only_2_2_within_sq_sum = np.cumsum(u_only_2_2_within_sq)/n_neurons
        u_only_2_2_within_sq_sum = u_only_2_2_within_sq_sum/u_only_2_2_within_sq_sum[-1]
       
        # Using V
        v_only_2_2_within = np.linalg.multi_dot([value_2_2,t_v_2_1])
        v_only_2_2_within_sq = np.sum(v_only_2_2_within**2, axis = 0)
        v_only_2_2_within_sq_sum = np.cumsum(v_only_2_2_within_sq)/n_neurons
        v_only_2_2_within_sq_sum = v_only_2_2_within_sq_sum/v_only_2_2_within_sq_sum[-1]
       
        
        between_2_1 = np.linalg.multi_dot([tu_2_1, value_1_2, t_v_2_1])
        between_2_1_d = between_2_1.diagonal()
        sum_between_2_1_d = np.cumsum(abs(between_2_1_d))/n_neurons
      
        u_only_2_2_between = np.linalg.multi_dot([tu_2_1, value_1_2])
        u_only_2_2_between_sq = np.sum(u_only_2_2_between**2, axis = 1)
        u_only_2_2_between_sq_sum = np.cumsum(u_only_2_2_between_sq)/n_neurons
        u_only_2_2_between_sq_sum = u_only_2_2_between_sq_sum/u_only_2_2_between_sq_sum[-1]
       
        # Using V
        v_only_2_2_between = np.linalg.multi_dot([value_2_1,t_v_1_2])
        v_only_2_2_between_sq = np.sum(v_only_2_2_between**2, axis = 0)
        v_only_2_2_between_sq_sum = np.cumsum(v_only_2_2_between_sq)/n_neurons
        v_only_2_2_between_sq_sum = v_only_2_2_between_sq_sum/v_only_2_2_between_sq_sum[-1]
       
        # Task 2 3
    
        u_2_2,s_2_2,v_2_2 = np.linalg.svd(value_2_2,full_matrices = False)
        tu_2_2 = np.transpose(u_2_2)  
        t_v_2_2 = np.transpose(v_2_2)
        
        within_2 = np.linalg.multi_dot([tu_2_2, value_2_1, t_v_2_2])
        within_2_d = within_2.diagonal()
        sum_within_2 = np.cumsum(abs(within_2_d))/n_neurons
        
        u_only_2_within = np.linalg.multi_dot([tu_2_2, value_2_1]) 
        u_only_2_within_sq = np.sum(u_only_2_within**2, axis = 1)
        u_only_2_within_sq_sum = np.cumsum(u_only_2_within_sq)/n_neurons
        u_only_2_within_sq_sum = u_only_2_within_sq_sum/u_only_2_within_sq_sum[-1]
        
       
        # Using V
        v_only_2_within = np.linalg.multi_dot([value_2_1,t_v_2_2])
        v_only_2_within_sq = np.sum(v_only_2_within**2, axis = 0)
        v_only_2_within_sq_sum = np.cumsum(v_only_2_within_sq)/n_neurons
        v_only_2_within_sq_sum = v_only_2_within_sq_sum/v_only_2_within_sq_sum[-1]
       
        
        between_2_3 = np.linalg.multi_dot([tu_2_2, value_3_1, t_v_2_2])
        between_2_3_d = between_2_3.diagonal()
        sum_between_2_3_d = np.cumsum(abs(between_2_3_d))/n_neurons
      
        u_only_2_3_between = np.linalg.multi_dot([tu_2_2, value_3_1])
        u_only_2_3_between_sq = np.sum(u_only_2_3_between**2, axis = 1)
        u_only_2_3_between_sq_sum = np.cumsum(u_only_2_3_between_sq)/n_neurons
        u_only_2_3_between_sq_sum = u_only_2_3_between_sq_sum/u_only_2_3_between_sq_sum[-1]
      
        # Using V
        v_only_2_3_between = np.linalg.multi_dot([value_3_1,t_v_2_2])
        v_only_2_3_between_sq = np.sum(v_only_2_3_between**2, axis = 0)
        v_only_2_3_between_sq_sum = np.cumsum(v_only_2_3_between_sq)/n_neurons
        v_only_2_3_between_sq_sum = v_only_2_3_between_sq_sum/v_only_2_3_between_sq_sum[-1]
      

        # Task 3 2
    
        u_3_2,s_3_2,v_3_2 = np.linalg.svd(value_3_1,full_matrices = False)
        tu_3_2 = np.transpose(u_3_2)  
        t_v_3_2 = np.transpose(v_3_2)
        
        within_3 = np.linalg.multi_dot([tu_3_2, value_3_2, t_v_3_2])
        within_3_d = within_3.diagonal()
        sum_within_3 = np.cumsum(abs(within_3_d))/n_neurons
        
        u_only_3_within = np.linalg.multi_dot([tu_3_2, value_3_2]) 
        u_only_3_within_sq = np.sum(u_only_3_within**2, axis = 1)
        u_only_3_within_sq_sum = np.cumsum(u_only_3_within_sq)/n_neurons
        u_only_3_within_sq_sum = u_only_3_within_sq_sum/u_only_3_within_sq_sum[-1]
        
       
        # Using V
        v_only_3_within = np.linalg.multi_dot([value_3_2,t_v_3_2])
        v_only_3_within_sq = np.sum(v_only_3_within**2, axis = 0)
        v_only_3_within_sq_sum = np.cumsum(v_only_3_within_sq)/n_neurons
        v_only_3_within_sq_sum = v_only_3_within_sq_sum/v_only_3_within_sq_sum[-1]
       
        
        between_3_2 = np.linalg.multi_dot([tu_3_2, value_2_2, t_v_3_2])
        between_3_2_d = between_3_2.diagonal()
        sum_between_3_2_d = np.cumsum(abs(between_3_2_d))/n_neurons
      
        u_only_3_2_between = np.linalg.multi_dot([tu_3_2, value_2_2])
        u_only_3_2_between_sq = np.sum(u_only_3_2_between**2, axis = 1)
        u_only_3_2_between_sq_sum = np.cumsum(u_only_3_2_between_sq)/n_neurons
        u_only_3_2_between_sq_sum = u_only_3_2_between_sq_sum/u_only_3_2_between_sq_sum[-1]
      
        # Using V
        v_only_3_2_between = np.linalg.multi_dot([value_2_2,t_v_3_2])
        v_only_3_2_between_sq = np.sum(v_only_3_2_between**2, axis = 0)
        v_only_3_2_between_sq_sum = np.cumsum(v_only_3_2_between_sq)/n_neurons
        v_only_3_2_between_sq_sum = v_only_3_2_between_sq_sum/v_only_3_2_between_sq_sum[-1]
      
        
        
        # Task 1 3     
        between_1_3 = np.linalg.multi_dot([tu_1_2, value_3_1, t_v_1_2])
        between_1_3_d = between_1_3.diagonal()
        sum_between_1_3_d = np.cumsum(abs(between_1_3_d))/n_neurons
        
        u_only_1_3_between = np.linalg.multi_dot([tu_1_2, value_3_1])
        u_only_1_3_between_sq = np.sum(u_only_1_3_between**2, axis = 1)
        u_only_1_3_between_sq_sum = np.cumsum(u_only_1_3_between_sq)/n_neurons
        u_only_1_3_between_sq_sum = u_only_1_3_between_sq_sum/u_only_1_3_between_sq_sum[-1]
      
        # Using V
        v_only_1_3_between = np.linalg.multi_dot([value_3_1,t_v_1_2])
        v_only_1_3_between_sq = np.sum(v_only_1_3_between**2, axis = 0)
        v_only_1_3_between_sq_sum = np.cumsum(v_only_1_3_between_sq)/n_neurons
        v_only_1_3_between_sq_sum = v_only_1_3_between_sq_sum/v_only_1_3_between_sq_sum[-1]
         
        # Task 3 1     
        between_3_1 = np.linalg.multi_dot([tu_3_2, value_1_2, t_v_3_2])
        between_3_1_d = between_3_1.diagonal()
        sum_between_3_1_d = np.cumsum(abs(between_3_1_d))/n_neurons
        
        u_only_3_1_between = np.linalg.multi_dot([tu_3_2, value_1_2])
        u_only_3_1_between_sq = np.sum(u_only_3_1_between**2, axis = 1)
        u_only_3_1_between_sq_sum = np.cumsum(u_only_3_1_between_sq)/n_neurons
        u_only_3_1__between_sq_sum = u_only_3_1_between_sq_sum/u_only_3_1_between_sq_sum[-1]
      
        # Using V
        v_only_3_1_between = np.linalg.multi_dot([value_1_2,t_v_3_2])
        v_only_3_1_between_sq = np.sum(v_only_3_1_between**2, axis = 0)
        v_only_3_1_between_sq_sum = np.cumsum(v_only_3_1_between_sq)/n_neurons
        v_only_3_1_between_sq_sum = v_only_3_1_between_sq_sum/v_only_3_1_between_sq_sum[-1]
      
        # within  = np.mean([sum_within_1,sum_within_2,sum_within_3],0)
        # between  = np.mean([sum_between_1_3_d,sum_between_2_3_d,sum_between_1_2_d ],0)

        within  = np.mean([sum_within_1,sum_within_2,sum_within_3,sum_within_2_2],0)
        between  = np.mean([sum_between_1_3_d,sum_between_2_3_d,sum_between_1_2_d, sum_between_3_2_d,sum_between_3_1_d,sum_between_2_1_d ],0)
        u_v_area.append((np.trapz(sum_within_1) - np.trapz(sum_between_1_2_d))/sum_within_1.shape[0])
       
        plt.subplot(3,1,1)
        plt.plot(within, label ='within'+ ' '+ area, color = c)
        plt.plot(between, label = 'between'+ ' '+ area, linestyle = '--', color = c)
        sns.despine()
       # plt.legend()
        within_u  = np.mean([u_only_2_within_sq_sum ,u_only_1_within_sq_sum, u_only_3_within_sq_sum ],0)
        between_u  = np.mean([u_only_3_2_between_sq_sum,u_only_2_3_between_sq_sum ,u_only_1_3_between_sq_sum ,u_only_3_1__between_sq_sum ,\
                              u_only_1_2_between_sq_sum ,u_only_2_2_between_sq_sum],0)
        u_area.append((np.trapz(within_u) - np.trapz(between_u))/within_u.shape[0])
 
        plt.subplot(3,1,2)
        plt.plot(within_u, label ='within'+ ' '+ area, color = c)
        plt.plot(between_u, label = 'between'+ ' '+ area, linestyle = '--', color = c)
        sns.despine()
    # plt.legend()
 
        within_v  = np.mean([v_only_2_within_sq_sum ,v_only_1_within_sq_sum, v_only_3_within_sq_sum ],0)
        between_v  = np.mean([v_only_3_2_between_sq_sum,v_only_2_3_between_sq_sum ,v_only_1_3_between_sq_sum ,v_only_3_1_between_sq_sum ,\
                              v_only_1_2_between_sq_sum ,v_only_2_2_between_sq_sum],0)
        v_area.append((np.trapz(within_v) - np.trapz(between_v))/within_v.shape[0])
 
     
        plt.subplot(3,1,3)
        plt.plot(within_v, label ='within'+ ' '+ area, color = c)
        plt.plot(between_v, label = 'between'+ ' '+ area, linestyle = '--', color = c)
        sns.despine()
      #plt.legend()

    return  u_v_area, v_area, u_area 

      
