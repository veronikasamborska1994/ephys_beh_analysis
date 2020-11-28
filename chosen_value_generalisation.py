#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:42:59 2020

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
import palettable
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)
import value_reg as vg

def load():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap

    
def chosen_value_reg(data, area = 'PFC', n = 10,  perm = True):
   
    if perm:
        dm = data[0]
        firing = data[1]

    else:
        dm = data['DM'][0]
        firing = data['Data'][0]

    C_1 = []; C_2 = []; C_3 = []
    cpd_1 = []; cpd_2 = []; cpd_3 = []
    average = vg.rew_prev_behaviour(data, n = n, perm = perm)

    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s]
       # firing_rates = scipy.stats.zscore(firing_rates,0)
        #firing_rates = firing_rates - np.mean(firing_rates,0)

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
          
        value_1_choice_1 = value_1*choices_1
        ones_1 = np.ones(len(choices_1))
        trials_1 = len(choices_1)
        prev_ch_reward_1 = choices_1*rewards_1
 
       
      
        firing_rates_1 = firing_rates[task_1][n:]
        
     
        predictors_all = OrderedDict([
                                    ('Choice', choices_1),
                                    ('Reward', rewards_1),
                                    ('Value',value_1), 
                                    ('Value Сhoice',value_1_choice_1), 
                                    ('Prev Rew Ch', prev_ch_reward_1),

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
          
        value_2_choice_2 = choices_2*value_2
 
        ones_2 = np.ones(len(choices_2))
        trials_2 = len(choices_2)

        firing_rates_2 = firing_rates[task_2][n:]
        
        prev_ch_reward_2 = choices_2*rewards_2


        predictors_all = OrderedDict([
                                     ('Choice', choices_2),
                                    ('Reward', rewards_2),
                                    ('Value',value_2), 
                                    ('Value Сhoice',value_2_choice_2), 
                                    ('Prev Rew Ch', prev_ch_reward_2),
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
        value_3_choice_3 = choices_3*value_3
        
        prev_ch_reward_3 =choices_3*rewards_3
        ones_3 = np.ones(len(choices_3))
        trials_3 = len(choices_3)

        firing_rates_3 = firing_rates[task_3][n:]
      
          
  
        predictors_all = OrderedDict([
                                     ('Ch', choices_3),
                                    ('Rew', rewards_3),
                                    ('Value',value_3), 
                                   ('Value Сhoice',value_3_choice_3), 
#                                   
                                    ('Prev Rew Ch', prev_ch_reward_3),
                              #    ('Prev Rew', prev_reward_3),
                                  # ('Prev Ch', prev_choice_3),
                                    ('ones', ones_3)
                                    ])
        
        X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
        n_predictors = X_3.shape[1]
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        #tstats,x = regression_code_session(y_3, X_3)
        ols = LinearRegression()
        ols.fit(X_3,y_3)
        C_3.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # Predictor loadings

       # C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_3.append(re._CPD(X_3,y_3).reshape(n_neurons, n_timepoints, n_predictors))
        
    
    C_1 = np.concatenate(C_1,0)
    
    C_2 = np.concatenate(C_2,0)
    
    C_3 = np.concatenate(C_3,0)
   
    C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
    C_1 = np.transpose(C_1[:,nans[0],:],[2,0,1]); C_2 = np.transpose(C_2[:,nans[0],:],[2,0,1]);  C_3 = np.transpose(C_3[:,nans[0],:],[2,0,1])
   
      
    return C_1,C_2,C_3


def perm_animals_chosen_value(HP, PFC, c_1 = 1, n = 6 , reward_times_to_choose = [1,2,3,4], task_check = 0):
    
 
    C_1_HP, C_2_HP, C_3_HP = chosen_value_reg(HP, area = 'HP', n = n, perm = False)
    C_1_PFC ,C_2_PFC, C_3_PFC = chosen_value_reg(PFC, area = 'PFC', perm = False)
    
   
   
    value_to_value_PFC = vg.generalisation_plot(C_1_PFC,C_2_PFC,C_3_PFC, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC = value_to_value_PFC[:-1]

    value_to_value_HP = vg.generalisation_plot(C_1_HP,C_2_HP,C_3_HP, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP = value_to_value_HP[:-1]

    difference = []


    all_subjects = [PFC['DM'][0][:9], PFC['DM'][0][9:25],PFC['DM'][0][25:39],PFC['DM'][0][39:],HP['DM'][0][:16], HP['DM'][0][16:24],HP['DM'][0][24:]]
    all_subjects_firing = [PFC['Data'][0][:9], PFC['Data'][0][9:25],PFC['Data'][0][25:39],PFC['Data'][0][39:],HP['Data'][0][:16], HP['Data'][0][16:24],HP['Data'][0][24:]]

    animals_PFC = [0,1,2,3]
    animals_HP = [4,5,6]
    m, n = len(animals_PFC), len(animals_HP)
  
    for indices_PFC in combinations(range(m + n), m):
        indices_HP = [i for i in range(m + n) if i not in indices_PFC]
       
        PFC_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_PFC)])
        HP_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_HP)])
        
        PFC_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_PFC)])
        HP_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_HP)])
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

         
        C_1_HP, C_2_HP, C_3_HP = chosen_value_reg(HP_shuffle, area = 'HP', n = n, perm = True)
        C_1_PFC ,C_2_PFC, C_3_PFC = chosen_value_reg(PFC_shuffle, area = 'PFC', perm = True)
    
     
        value_to_value_PFC_perm = vg.generalisation_plot(C_1_PFC,C_2_PFC,C_3_PFC, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_PFC_perm  = value_to_value_PFC_perm[:-1]

        value_to_value_HP_perm  = vg.generalisation_plot(C_1_HP,C_2_HP,C_3_HP, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_HP_perm  = value_to_value_HP_perm[:-1]
       
      

        difference.append((value_to_value_PFC_perm-value_to_value_HP_perm))
        
        
         
    perm = np.max(np.percentile(difference,95,0),1)
    
    diff_real = (value_to_value_PFC - value_to_value_HP)
   
    perms_pval = np.where(diff_real.T > perm)
    
    return perms_pval

def perumute_sessions_ch_value(HP, PFC, c_1 = 1, n = 6 , reward_times_to_choose = [1,2,3,4], task_check = 0, perm_n = 500):
    
 
    C_1_HP, C_2_HP, C_3_HP = chosen_value_reg(HP, area = 'HP', n = n, perm = False)
    C_1_PFC ,C_2_PFC, C_3_PFC = chosen_value_reg(PFC, area = 'PFC', perm = False)
    
   
   
    value_to_value_PFC = vg.generalisation_plot(C_1_PFC,C_2_PFC,C_3_PFC, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_PFC = value_to_value_PFC[:-1]

    value_to_value_HP = vg.generalisation_plot(C_1_HP,C_2_HP,C_3_HP, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
    value_to_value_HP = value_to_value_HP[:-1]

    difference = []


    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    
    sessions_n = np.arange(len(all_subjects))
  
    for i in range(perm_n):
        np.random.shuffle(sessions_n) # Shuffle PFC/HP sessions
        indices_HP = sessions_n[46:]
        indices_PFC = sessions_n[:46]

        PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)]
        HP_shuffle_dm = all_subjects[np.asarray(indices_HP)]
        
        PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)]
        HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

         
        C_1_HP, C_2_HP, C_3_HP = chosen_value_reg(HP_shuffle, area = 'HP', n = n, perm = True)
        C_1_PFC ,C_2_PFC, C_3_PFC = chosen_value_reg(PFC_shuffle, area = 'PFC', perm = True)
    
     
        value_to_value_PFC_perm = vg.generalisation_plot(C_1_PFC,C_2_PFC,C_3_PFC, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_PFC_perm  = value_to_value_PFC_perm[:-1]

        value_to_value_HP_perm  = vg.generalisation_plot(C_1_HP,C_2_HP,C_3_HP, c_1, reward_times_to_choose = reward_times_to_choose, task_check = task_check)
        value_to_value_HP_perm  = value_to_value_HP_perm[:-1]
       
      

        difference.append((value_to_value_PFC_perm-value_to_value_HP_perm))
        
        
         
    perm = np.max(np.percentile(difference,95,0),1)
    
    diff_real = (value_to_value_PFC - value_to_value_HP)
   
    perms_pval = np.where(diff_real.T > perm)
    
    return perms_pval

def find_pvals():
    
     perms_all = perm_animals_chosen_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 0)
     perms_1_2 = perm_animals_chosen_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 1)
     perms_1_3 = perm_animals_chosen_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 2)
     perms_2_3 = perm_animals_chosen_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 3)


     perms_all_s = perumute_sessions_ch_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 0,  perm_n = 1000)
     perms_1_2_s = perumute_sessions_ch_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 1, perm_n = 1000)
     perms_1_3_s = perumute_sessions_ch_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 2, perm_n = 1000)
     perms_2_3_s = perumute_sessions_ch_value(HP, PFC, c_1 = 3, n = 11, reward_times_to_choose = np.asarray([20,25,36,42]),task_check = 3, perm_n = 1000)


