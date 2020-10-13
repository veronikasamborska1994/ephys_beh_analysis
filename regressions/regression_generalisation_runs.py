#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:39:22 2020

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
import regression_function as reg_f
from collections import OrderedDict
import regressions as re
import palettable.wesanderson as wes
from matplotlib.backends.backend_pdf import PdfPages


def find_coefficients(C_1,C_2,C_3,interactions = True, a = True):
    
    pre_init = 20
    init = 25
    choice = 35
    reward = 42 
    
    
    if a == True:
        rew_ind = 4
        err_ind = 2
    else:
        rew_ind = 5
        err_ind = 3
    rew_int_ind = 5
    err_int_ind = 4

   
  
    X_Pre_init_rew = np.nanmean(C_1[rew_ind][:,pre_init-10:pre_init],1)
    X_init_rew = np.nanmean(C_1[rew_ind][:,init-5:init+5],1)
    X_ch_rew = np.nanmean(C_1[rew_ind][:,choice-5:choice+5],1)
    X_rew_rew = np.nanmean(C_1[rew_ind][:,reward-2:reward+5],1)

    X_Pre_init_err = np.nanmean(C_1[err_ind][:,pre_init-10:pre_init:10],1)
    X_init_err = np.nanmean(C_1[err_ind][:,init-5:init+5],1)
    X_ch_err = np.nanmean(C_1[err_ind][:,choice-5:choice+5],1)
    X_rew_err = np.nanmean(C_1[err_ind][:,reward-2:reward+5],1)
    
    
    
    X_Pre_init_rew_2 = np.nanmean(C_2[rew_ind][:,pre_init-10:pre_init],1)
    X_init_rew_2 = np.nanmean(C_2[rew_ind][:,init-5:init+5],1)
    X_ch_rew_2 = np.nanmean(C_2[rew_ind][:,choice-5:choice+5],1)
    X_rew_rew_2 = np.nanmean(C_2[rew_ind][:,reward-2:reward+5],1)
    
    X_Pre_init_err_2 = np.nanmean(C_2[err_ind][:,pre_init-10:pre_init],1)
    X_init_err_2 = np.nanmean(C_2[err_ind][:,init-5:init+5],1)
    X_ch_err_2 = np.nanmean(C_2[err_ind][:,choice-5:choice+5],1)
    X_rew_err_2 = np.nanmean(C_2[err_ind][:,reward-2:reward+5],1)
    
    
   
    X_Pre_init_rew_3 = np.nanmean(C_3[rew_ind][:,pre_init-10:pre_init],1)
    X_init_rew_3 = np.nanmean(C_3[rew_ind][:,init-5:init+5],1)
    X_ch_rew_3 = np.nanmean(C_3[rew_ind][:,choice-5:choice+5],1)
    X_rew_rew_3 = np.nanmean(C_3[rew_ind][:,reward-2:reward+5],1)
    
    X_Pre_init_err_3 = np.nanmean(C_3[err_ind][:,pre_init-10:pre_init],1)
    X_init_err_3 = np.nanmean(C_3[err_ind][:,init-5:init+5],1)
    X_ch_err_3 = np.nanmean(C_3[err_ind][:,choice-5:choice+5],1)
    X_rew_err_3 = np.nanmean(C_3[err_ind][:,reward-2:reward+5],1)
  
   
    ones = np.ones(len(X_rew_err))
    ones_2 = np.ones(len(X_rew_err_2))
    ones_3 = np.ones(len(X_rew_err_3))
    
    if interactions == True:
     
        X_Pre_init_rew_int = np.nanmean(C_1[rew_int_ind][:,pre_init-10:pre_init],1)
        X_init_rew_int = np.nanmean(C_1[rew_int_ind][:,init-5:init+5],1)
        X_ch_rew_int = np.nanmean(C_1[rew_int_ind][:,choice-5:choice+5],1)
        X_rew_rew_int = np.nanmean(C_1[rew_int_ind][:,reward-2:reward+5],1)
    
        X_Pre_init_err_int = np.nanmean(C_1[err_int_ind][:,pre_init-10:pre_init],1)
        X_init_err_int = np.nanmean(C_1[err_int_ind][:,init-5:init+5],1)
        X_ch_err_int = np.nanmean(C_1[err_int_ind][:,choice-5:choice+5],1)
        X_rew_err_int = np.nanmean(C_1[err_int_ind][:,reward-2:reward+5],1)
        
        
        X_Pre_init_rew_2_int = np.nanmean(C_2[rew_int_ind][:,pre_init-10:pre_init],1)
        X_init_rew_2_int = np.nanmean(C_2[rew_int_ind][:,init-5:init+init],1)
        X_ch_rew_2_int = np.nanmean(C_2[rew_int_ind][:,choice-5:choice+5],1)
        X_rew_rew_2_int = np.nanmean(C_2[rew_int_ind][:,reward-2:reward+5],1)
        
        X_Pre_init_err_2_int = np.nanmean(C_2[err_int_ind][:,pre_init-10:pre_init],1)
        X_init_err_2_int = np.nanmean(C_2[err_int_ind][:,init-5:init+5],1)
        X_ch_err_2_int = np.nanmean(C_2[err_int_ind][:,choice-5:choice+5],1)
        X_rew_err_2_int = np.nanmean(C_2[err_int_ind][:,reward-2:reward+5],1)
      
        
        X_Pre_init_rew_3_int = np.nanmean(C_3[rew_int_ind][:,pre_init-10:pre_init],1)
        X_init_rew_3_int = np.nanmean(C_3[rew_int_ind][:,init-5:init+5],1)
        X_ch_rew_3_int = np.nanmean(C_3[rew_int_ind][:,choice-5:choice+5],1)
        X_rew_rew_3_int = np.nanmean(C_3[rew_int_ind][:,reward-2:reward+5],1)
        
        X_Pre_init_err_3_int = np.nanmean(C_3[err_int_ind][:,pre_init-10:pre_init],1)
        X_init_err_3_int = np.nanmean(C_3[err_int_ind][:,init-5:init+5],1)
        X_ch_err_3_int = np.nanmean(C_3[err_int_ind][:,choice-5:choice+5],1)
        X_rew_err_3_int = np.nanmean(C_3[err_int_ind][:,reward-2:reward+5],1)
    
       
       # ones = np.ones(len(X_rew))
        X = np.vstack([X_Pre_init_rew,X_init_rew,X_ch_rew,X_rew_rew,ones,X_Pre_init_rew_int,X_init_rew_int,X_ch_rew_int,X_rew_rew_int,ones,
                    X_Pre_init_err, X_init_err, X_ch_err,X_rew_err, ones,X_Pre_init_err_int, X_init_err_int, X_ch_err_int, X_rew_err_int, ones,\
                    X_Pre_init_rew_2,X_init_rew_2,X_ch_rew_2,X_rew_rew_2, ones,X_Pre_init_rew_2_int,X_init_rew_2_int,X_ch_rew_2_int,X_rew_rew_2_int,ones_2,\
                    X_Pre_init_err_2, X_init_err_2, X_ch_err_2,X_rew_err_2,ones,X_Pre_init_err_2_int, X_init_err_2_int, X_ch_err_2_int,X_rew_err_2_int,  ones_2,\
                    X_Pre_init_rew_3,X_init_rew_3,X_ch_rew_3,X_rew_rew_3,ones,X_Pre_init_rew_3_int,X_init_rew_3_int,X_ch_rew_3_int,X_rew_rew_3_int, ones_3,\
                    X_Pre_init_err_3, X_init_err_3, X_ch_err_3,X_rew_err_3,ones, X_Pre_init_err_3_int, X_init_err_3_int, X_ch_err_3_int,X_rew_err_3_int, ones_3]).T
            
    else: 
        
         # ones = np.ones(len(X_rew))
        X = np.vstack([X_Pre_init_rew,X_init_rew,X_ch_rew,X_rew_rew,ones,\
                    X_Pre_init_err, X_init_err, X_ch_err,X_rew_err, ones,\
                    X_Pre_init_rew_2,X_init_rew_2,X_ch_rew_2,X_rew_rew_2, ones_2,\
                    X_Pre_init_err_2, X_init_err_2, X_ch_err_2,X_rew_err_2, ones_2,\
                    X_Pre_init_rew_3,X_init_rew_3,X_ch_rew_3,X_rew_rew_3, ones_3,\
                    X_Pre_init_err_3, X_init_err_3, X_ch_err_3,X_rew_err_3,ones_3]).T
  
    X_inf = X[~np.isinf(X).any(axis=1)]
    X_nan = X_inf[~np.isnan(X_inf).any(axis =1)]

    firing_rates_1_nan_rew = C_1[rew_ind][~np.isinf(X).any(axis=1)]
    firing_rates_1_inf_rew = firing_rates_1_nan_rew[~np.isnan(X_inf).any(axis=1)]

    firing_rates_2_nan_rew = C_2[rew_ind][~np.isinf(X).any(axis=1)]
    firing_rates_2_inf_rew = firing_rates_2_nan_rew[~np.isnan(X_inf).any(axis=1)]
    
    firing_rates_3_nan_rew = C_3[rew_ind][~np.isinf(X).any(axis=1)]
    firing_rates_3_inf_rew = firing_rates_3_nan_rew[~np.isnan(X_inf).any(axis=1)]
    
    firing_rates_1_nan_error = C_1[err_ind][~np.isinf(X).any(axis=1)]
    firing_rates_1_inf_error = firing_rates_1_nan_error[~np.isnan(X_inf).any(axis=1)]

    firing_rates_2_nan_error = C_2[err_ind][~np.isinf(X).any(axis=1)]
    firing_rates_2_inf_error = firing_rates_2_nan_error[~np.isnan(X_inf).any(axis=1)]
    
    firing_rates_3_nan_error = C_3[err_ind][~np.isinf(X).any(axis=1)]
    firing_rates_3_inf_error = firing_rates_3_nan_error[~np.isnan(X_inf).any(axis=1)]
    
    ind_to_delete = []
    for i in [firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,firing_rates_3_inf_error]:
        
        infs = np.where(np.isinf(i).any(axis=1) == True)[0]
        if len(infs)>0:
            for i in infs:
                ind_to_delete.append(i)
        
    to_delete = np.unique(ind_to_delete)
    indexes = np.arange(len(firing_rates_1_inf_rew))
    delete = np.delete(indexes,to_delete)
    firing_rates_1_inf_rew = firing_rates_1_inf_rew[delete]
    firing_rates_2_inf_rew = firing_rates_2_inf_rew[delete]
    firing_rates_3_inf_rew = firing_rates_3_inf_rew[delete]
    firing_rates_1_inf_error = firing_rates_1_inf_error[delete]
    firing_rates_2_inf_error = firing_rates_2_inf_error[delete]
    firing_rates_3_inf_error = firing_rates_3_inf_error[delete]
       
    X_nan = X_nan[delete]
    
    if interactions == True:
        firing_rates_1_nan_rew_int = C_1[rew_int_ind][~np.isinf(X).any(axis=1)]
        firing_rates_1_inf_rew_int = firing_rates_1_nan_rew_int[~np.isnan(X_inf).any(axis=1)]
    
        firing_rates_2_nan_rew_int = C_2[rew_int_ind][~np.isinf(X).any(axis=1)]
        firing_rates_2_inf_rew_int = firing_rates_2_nan_rew_int[~np.isnan(X_inf).any(axis=1)]
        
        firing_rates_3_nan_rew_int = C_3[rew_int_ind][~np.isinf(X).any(axis=1)]
        firing_rates_3_inf_rew_int = firing_rates_3_nan_rew_int[~np.isnan(X_inf).any(axis=1)]
        
        firing_rates_1_nan_error_int = C_1[err_int_ind][~np.isinf(X).any(axis=1)]
        firing_rates_1_inf_error_int = firing_rates_1_nan_error_int[~np.isnan(X_inf).any(axis=1)]
    
        firing_rates_2_nan_error_int = C_2[err_int_ind][~np.isinf(X).any(axis=1)]
        firing_rates_2_inf_error_int = firing_rates_2_nan_error_int[~np.isnan(X_inf).any(axis=1)]
        
        firing_rates_3_nan_error_int = C_3[err_int_ind][~np.isinf(X).any(axis=1)]
        firing_rates_3_inf_error_int = firing_rates_3_nan_error_int[~np.isnan(X_inf).any(axis=1)]

        firing_rates_1_inf_rew_int = firing_rates_1_inf_rew_int[delete]
        firing_rates_2_inf_rew_int = firing_rates_2_inf_rew_int[delete]
        firing_rates_3_inf_rew_int = firing_rates_3_inf_rew_int[delete]
        firing_rates_1_inf_error_int = firing_rates_1_inf_error_int[delete]
        firing_rates_2_inf_error_int = firing_rates_2_inf_error_int[delete]
        firing_rates_3_inf_error_int = firing_rates_3_inf_error_int[delete]
          
        
        return X_nan, firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,\
            firing_rates_3_inf_error,firing_rates_1_inf_rew_int,firing_rates_2_inf_rew_int,firing_rates_3_inf_rew_int,firing_rates_1_inf_error_int,\
           firing_rates_2_inf_error_int,firing_rates_3_inf_error_int
  
    else:
       return X_nan, firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,\
        firing_rates_3_inf_error
       

def sequence_rewards_errors_regression_generalisation(data, perm = True, area = 'HP_', interactions = True, a = True):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    C_1 = []
    cpd_1 = []
    
    C_2 = []
    cpd_2 = []
   
    C_3 = []
    cpd_3 = []
    if perm:
        
      C_1_perm = [[] for i in range(perm)]
      
      C_2_perm = [[] for i in range(perm)]
   
      C_3_perm = [[] for i in range(perm)]
      
      cpd_perm_all = [[] for i in range(perm)]
      
      
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape

        choices = DM[:,1]
       # choices[np.where(choices ==0)[0]] = -1

        reward = DM[:,2]    

        task =  DM[:,5]
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]
        task_3 = np.where(task == 3)[0]
        state = DM[:,0]

        correct = np.where(state == choices)[0]
        incorrect = np.where(state != choices)[0]
      
        cum_error = []
        runs_list_corr = []
        runs_list_incorr =[]
        err = 0
        for r in reward:
            if r == 0:   
                err+=1
            else:
                err = 0
            cum_error.append(err)
        
        cum_reward = []
        for r in reward:
            if r == 1:
                err+=1
            else:
                err = 0
            cum_reward.append(err)
       
        run = 0
        for c, ch in enumerate(choices):
            if c > 0:
                if choices[c] == choices[c-1]:
                    run += 1
                elif choices[c] != choices[c-1]:
                    run = 0
                runs_list.append(run)
       
        corr_run = 0
        run_ind_c =[]
        for c, ch in enumerate(choices):
            if c > 0  and c in correct:
                if choices[c] == choices[c-1]:
                    if corr_run == 0:
                        run_ind_c.append(c)
                    corr_run +=1
                elif choices[c] != choices[c-1]:
                    corr_run = 0
            else:
                corr_run = 0
            runs_list_corr.append(corr_run)
         
        incorr_run = 0
        run_ind_inc = []
        for c, ch in enumerate(choices):
            if c > 0  and c in incorrect:
                if choices[c] == choices[c-1]:
                    if incorr_run ==0:
                        run_ind_inc.append(c)
                    incorr_run +=1
                elif choices[c] != choices[c-1]:
                    incorr_run = 0
            else:
                incorr_run = 0
                
            runs_list_incorr.append(incorr_run)
        
        choices_a = np.where(choices==1)[0]
        choices_b = np.where(choices==0)[0]

        a_cum_rew = np.copy(np.asarray(np.asarray(cum_reward)))
        b_cum_rew = np.copy(np.asarray(np.asarray(cum_reward)))

        a_cum_rew[choices_b] = 0
        b_cum_rew[choices_a] = 0

        a_cum_error = np.copy(np.asarray(np.asarray(cum_error)))
        b_cum_error= np.copy(np.asarray(np.asarray(cum_error)))

        a_cum_error[choices_b] = 0
        b_cum_error[choices_a] = 0

        ones = np.ones(len(reward))
        reward_1 = reward[task_1]
        choices_1 = choices[task_1]
        cum_error_1= np.asarray(cum_error)[task_1]
        cum_reward_1 = np.asarray(cum_reward)[task_1]
        
        cum_error_1_a = np.asarray(a_cum_error)[task_1]
        cum_error_1_b = np.asarray(b_cum_error)[task_1]

        cum_reward_1_a = np.asarray(a_cum_rew)[task_1]
        cum_reward_1_b = np.asarray(b_cum_rew)[task_1]

        ones_1 = ones[task_1]
        cum_error_1_ch = cum_error_1*choices_1
        cum_rew_1_ch = cum_reward_1*choices_1
        firing_rates_1 = firing_rates[task_1]
        
        int_rew_ch_1 = reward_1*choices_1
        if interactions == True:
            predictors_all = OrderedDict([('Reward', reward_1),
                                      ('Choice', choices_1),
                                      ('Errors', cum_error_1),
                                      ('Rewards',cum_reward_1),
                                      ('Choice x Cum Error', cum_error_1_ch),
                                      ('Choice x Cum Reward', cum_rew_1_ch),
                                      ('Choice x Reward',int_rew_ch_1),
                                      ('ones', ones_1)])
        else:
            predictors_all = OrderedDict([('Reward', reward_1),
                                      ('Choice', choices_1),
                                     
                                      ('Errors A', cum_error_1_a),
                                      ('Errors B', cum_error_1_b),

                                      ('Rewards A',cum_reward_1_a), 
                                      ('Rewards B',cum_reward_1_b),  

                                      ('Choice x Reward',int_rew_ch_1),

                                      ('ones', ones_1)])   
            
        X_1 = np.vstack(predictors_all.values()).T[:len(choices_1),:].astype(float)
        rank = np.linalg.matrix_rank(X_1)
        # print(rank)
        # print(X_1.shape[1])
        n_predictors = X_1.shape[1]
        y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y_1, X_1)
        
        C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(re._CPD(X_1,y_1).reshape(n_neurons,n_timepoints, n_predictors))
        
        
        
        reward_2 = reward[task_2]
        choices_2 = choices[task_2]
        cum_error_2 = np.asarray(cum_error)[task_2]
        cum_reward_2 = np.asarray(cum_reward)[task_2]
        ones_2 = ones[task_2]
        cum_error_2_ch = cum_error_2*choices_2
        cum_rew_2_ch = cum_reward_2*choices_2
        int_rew_ch_2 = reward_2*choices_2
        cum_error_2_a = np.asarray(a_cum_error)[task_2]
        cum_error_2_b = np.asarray(b_cum_error)[task_2]

        cum_reward_2_a = np.asarray(a_cum_rew)[task_2]
        cum_reward_2_b = np.asarray(b_cum_rew)[task_2]

        firing_rates_2 = firing_rates[task_2]
        if interactions == True:

            predictors_all = OrderedDict([('Reward', reward_2),
                                      ('Choice', choices_2),
                                      ('Errors', cum_error_2),
                                      ('Rewards',cum_reward_2),
                                      ('Choice x Cum Error', cum_error_2_ch),
                                      ('Choice x Cum Reward', cum_rew_2_ch),
                                      ('Choice x Reward',int_rew_ch_2),

                                      ('ones', ones_2)])
        else:
            predictors_all = OrderedDict([('Reward', reward_2),
                                      ('Choice', choices_2),
                                      ('Errors A', cum_error_2_a),
                                      ('Errors B', cum_error_2_b),

                                      ('Rewards A',cum_reward_2_a), 
                                      ('Rewards B',cum_reward_2_b),  

                                      ('Choice x Reward',int_rew_ch_2),
                                      ('ones', ones_2)])
         
               
        X_2 = np.vstack(predictors_all.values()).T[:len(choices_2),:].astype(float)
        y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y_2, X_2)
        
        C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_2.append(re._CPD(X_2,y_2).reshape(n_neurons,n_timepoints, n_predictors))
        
        reward_3 = reward[task_3]
        choices_3 = choices[task_3]
        cum_error_3 = np.asarray(cum_error)[task_3]
        cum_reward_3 = np.asarray(cum_reward)[task_3]
        ones_3 = ones[task_3]
        cum_error_3_ch = cum_error_3*choices_3
        cum_rew_3_ch = cum_reward_3*choices_3
        int_rew_ch_3 = reward_3*choices_3
        cum_error_3_a = np.asarray(a_cum_error)[task_3]
        cum_error_3_b = np.asarray(b_cum_error)[task_3]

        cum_reward_3_a = np.asarray(a_cum_rew)[task_3]
        cum_reward_3_b = np.asarray(b_cum_rew)[task_3]

        firing_rates_3 = firing_rates[task_3]
        if interactions == True:

            predictors_all = OrderedDict([('Reward', reward_3),
                                      ('Choice', choices_3),
                                      ('Errors', cum_error_3),
                                      ('Rewards',cum_reward_3),
                                      ('Choice x Cum Error', cum_error_3_ch),
                                      ('Choice x Cum Reward', cum_rew_3_ch),
                                      ('Choice x Reward',int_rew_ch_3),

                                      ('ones', ones_3)])
            
        else:
            predictors_all = OrderedDict([('Reward', reward_3),
                                      ('Choice', choices_3),
                                     
                                      ('Errors A', cum_error_3_a),
                                      ('Errors B', cum_error_3_b),

                                      ('Rewards A',cum_reward_3_a), 
                                      ('Rewards B',cum_reward_3_b),  

                                      ('Choice x Reward',int_rew_ch_3),

                                    
                                      ('ones', ones_3)])
            
        X_3 = np.vstack(predictors_all.values()).T[:len(choices_3),:].astype(float)
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y_3, X_3)
        
        C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_3.append(re._CPD(X_3,y_3).reshape(n_neurons,n_timepoints, n_predictors))
        
        if perm:
           for i in range(perm):
               X_perm_1 = np.roll(X_1,np.random.randint(len(X_1)), axis = 0)
               tstats = reg_f.regression_code(y_1, X_perm_1)
               C_1_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
               cpd_perm_1 = re._CPD(X_perm_1,y_1).reshape(n_neurons,n_timepoints, n_predictors)

               X_perm_2 = np.roll(X_2,np.random.randint(len(X_2)), axis = 0)
               
               tstats = reg_f.regression_code(y_2, X_perm_2)
               C_2_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
               cpd_perm_2 = re._CPD(X_perm_2,y_2).reshape(n_neurons,n_timepoints, n_predictors)

               X_perm_3 = np.roll(X_3,np.random.randint(len(X_3)), axis = 0)
               
               tstats = reg_f.regression_code(y_3, X_perm_3)
               C_3_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
               cpd_perm_3 = re._CPD(X_perm_3,y_3).reshape(n_neurons,n_timepoints, n_predictors)
               
               cpd_perm_all[i].append(np.nanmean([cpd_perm_1,cpd_perm_2,cpd_perm_3],0))

    cpd_perm_all = np.stack([np.mean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm_all],0)

    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    C_1 = np.concatenate(C_1,1)
    
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)
    C_2 = np.concatenate(C_2,1)
    
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    C_3 = np.concatenate(C_3,1)  
    
    cpds_true = np.mean([cpd_1,cpd_2,cpd_3],0)
    
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
    
    cpd_perm_all =  np.max(np.percentile(cpd_perm_all,95, axis = 0),0)
    j = 0
    plt.figure()
    for i in cpds_true.T[:-1]:
        plt.plot(i, color = c[j],label = list(predictors_all.keys())[j])
        plt.hlines(cpd_perm_all[j], xmin = 0, xmax = 63, color = c[j],linestyle = ':')

        j+=1
    plt.legend()
    sns.despine()
        
    plt.title( area +'CPDs')
    
    if interactions == True:
        X_nan, firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,\
        firing_rates_3_inf_error,firing_rates_1_inf_rew_int,firing_rates_2_inf_rew_int,firing_rates_3_inf_rew_int,firing_rates_1_inf_error_int,\
           firing_rates_2_inf_error_int,firing_rates_3_inf_error_int = find_coefficients(C_1,C_2,C_3,interactions = interactions, a = a)

        X_1_rew = X_nan[:,:5]
        X_1_rew_int = X_nan[:,5:10]
    
        X_1_error = X_nan[:,10:15]
        X_1_error_int = X_nan[:,15:20]
    
        X_2_rew = X_nan[:,20:25]
        X_2_rew_int = X_nan[:,25:30]
    
        X_2_error = X_nan[:,30:35]
        X_2_error_int = X_nan[:,35:40]
    
        X_3_rew = X_nan[:,40:45]
        X_3_rew_int = X_nan[:,45:50]
    
        X_3_error = X_nan[:,50:55]
        X_3_error_int = X_nan[:,55:60]
        
        # plt.figure()
        
        # for i in range(4):
        #     plt.subplot(2,2,i+1)
        #     sns.regplot(X_1_rew[:,i],X_2_rew[:,i], color = 'red',label = 'Reward')
        #     #sns.regplot(X_1_rew[:,i],X_3_rew[:,i], color = 'black')
        #     #sns.regplot(X_3_rew[:,i],X_2_rew[:,i], color = 'grey')
        #     corr = np.mean([np.corrcoef(X_1_rew[:,i],X_2_rew[:,i])[0,1],np.corrcoef(X_1_rew[:,i],X_3_rew[:,i])[0,1],np.corrcoef(X_2_rew[:,i],X_3_rew[:,i])[0,1]])
        #     plt.annotate(np.round(corr,2),(1,1.5))
        # plt.legend()

        # sns.despine()
         
        # plt.figure()
        
        # for i in range(4):
        #     plt.subplot(2,2,i+1)
        #     sns.regplot(X_1_rew_int[:,i],X_2_rew_int[:,i], color = 'grey', label = 'Reward Interaction')

        #     corr = np.mean([np.corrcoef(X_1_rew_int[:,i],X_2_rew_int[:,i])[0,1],np.corrcoef(X_1_rew_int[:,i],X_3_rew_int[:,i])[0,1],np.corrcoef(X_2_rew_int[:,i],X_3_rew_int[:,i])[0,1]])
        #     plt.annotate(np.round(corr,2),(1,1.5))
        # plt.legend()
        # sns.despine()
      
          
        # plt.figure()
        
        # for i in range(4):
        #     plt.subplot(2,2,i+1)
        #     sns.regplot(X_1_error[:,i],X_2_error[:,i], color = 'purple', label = 'Error')

        #     corr = np.mean([np.corrcoef(X_1_error[:,i],X_2_error[:,i])[0,1],np.corrcoef(X_1_error[:,i],X_3_error[:,i])[0,1],np.corrcoef(X_2_error[:,i],X_3_error[:,i])[0,1]])
        #     plt.annotate(np.round(corr,2),(1,1.5))
            
        
        # plt.figure()
        
        # for i in range(4):
        #     plt.subplot(2,2,i+1)
        #     sns.regplot(X_1_error_int[:,i],X_2_error_int[:,i], color = 'purple', label = 'Error')

        #     corr = np.mean([np.corrcoef(X_1_error_int[:,i],X_2_error_int[:,i])[0,1],np.corrcoef(X_1_error_int[:,i],X_3_error_int[:,i])[0,1],np.corrcoef(X_2_error_int[:,i],X_3_error_int[:,i])[0,1]])
        #     plt.annotate(np.round(corr,2),(1,1.5))
     
        # plt.legend()
        # sns.despine()
      
        cpd_1_2_rew = re._CPD(X_1_rew,firing_rates_2_inf_rew); cpd_1_3_rew = re._CPD(X_1_rew,firing_rates_3_inf_rew)
       
        cpd_2_1_rew = re._CPD(X_2_rew,firing_rates_1_inf_rew); cpd_2_3_rew = re._CPD(X_2_rew,firing_rates_3_inf_rew)
        
        cpd_3_1_rew = re._CPD(X_3_rew,firing_rates_1_inf_rew); cpd_3_2_rew = re._CPD(X_3_rew,firing_rates_2_inf_rew)
         
        cpd_1_2_error = re._CPD(X_1_error,firing_rates_2_inf_error); cpd_1_3_error = re._CPD(X_1_error,firing_rates_3_inf_error)
       
        cpd_2_1_error = re._CPD(X_2_error,firing_rates_1_inf_error); cpd_2_3_error = re._CPD(X_2_error,firing_rates_3_inf_error)
        
        cpd_3_1_error = re._CPD(X_3_error,firing_rates_1_inf_error); cpd_3_2_error = re._CPD(X_3_error,firing_rates_2_inf_error)
        
       
    
        cpd_1_2_rew_int = re._CPD(X_1_rew_int,firing_rates_2_inf_rew_int); cpd_1_3_rew_int = re._CPD(X_1_rew_int,firing_rates_3_inf_rew_int)
        
   
        cpd_2_1_rew_int = re._CPD(X_2_rew_int,firing_rates_1_inf_rew_int); cpd_2_3_rew_int = re._CPD(X_2_rew_int,firing_rates_3_inf_rew_int)
        
        cpd_3_1_rew_int = re._CPD(X_3_rew_int,firing_rates_1_inf_rew_int); cpd_3_2_rew_int = re._CPD(X_3_rew_int,firing_rates_2_inf_rew_int)
        
        cpd_1_2_error_int = re._CPD(X_1_error_int,firing_rates_2_inf_error_int); cpd_1_3_error_int = re._CPD(X_1_error_int,firing_rates_3_inf_error_int)
       
        cpd_2_1_error_int = re._CPD(X_2_error_int,firing_rates_1_inf_error_int); cpd_2_3_error_int = re._CPD(X_2_error_int,firing_rates_3_inf_error_int)
        
        cpd_3_1_error_int = re._CPD(X_3_error_int,firing_rates_1_inf_error_int); cpd_3_2_error_int = re._CPD(X_3_error_int,firing_rates_2_inf_error_int)
        

        
        cpd_rew_int = np.nanmean([cpd_1_2_rew_int,cpd_1_3_rew_int,cpd_2_1_rew_int,cpd_2_3_rew_int,cpd_3_1_rew_int,cpd_3_2_rew_int],0)
        cpd_error_int = np.nanmean([cpd_1_2_error_int,cpd_1_3_error_int,cpd_2_1_error_int,cpd_2_3_error_int,cpd_3_1_error_int,cpd_3_2_error_int],0)


    else:
        
        X_nan, firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,\
        firing_rates_3_inf_error  = find_coefficients(C_1,C_2,C_3,interactions = interactions, a = a)
        
        X_1_rew = X_nan[:,:5]
    
        X_1_error = X_nan[:,5:10]
    
        X_2_rew = X_nan[:,10:15]
    
        X_2_error = X_nan[:,15:20]
    
        X_3_rew = X_nan[:,20:25]
    
        X_3_error = X_nan[:,25:30]
    
        # plt.figure()
        
        # for i in range(4):
        #     plt.subplot(2,2,i+1)
        #     sns.regplot(X_1_rew[:,i],X_2_rew[:,i], color = 'red',label = 'Reward')
        #     #sns.regplot(X_1_rew[:,i],X_3_rew[:,i], color = 'black')
        #     #sns.regplot(X_3_rew[:,i],X_2_rew[:,i], color = 'grey')
        #     corr = np.mean([np.corrcoef(X_1_rew[:,i],X_2_rew[:,i])[0,1],np.corrcoef(X_1_rew[:,i],X_3_rew[:,i])[0,1],np.corrcoef(X_2_rew[:,i],X_3_rew[:,i])[0,1]])
        #     plt.annotate(np.round(corr,2),(1,1.5))
        # plt.legend()
        
        # plt.figure()
        
        # for i in range(4):
        #     plt.subplot(2,2,i+1)
        #     sns.regplot(X_1_error[:,i],X_2_error[:,i], color = 'purple', label = 'Error')

        #     corr = np.mean([np.corrcoef(X_1_error[:,i],X_2_error[:,i])[0,1],np.corrcoef(X_1_error[:,i],X_3_error[:,i])[0,1],np.corrcoef(X_2_error[:,i],X_3_error[:,i])[0,1]])
        #     plt.annotate(np.round(corr,2),(1,1.5))
            
     

        cpd_1_2_rew = re._CPD(X_1_rew,firing_rates_2_inf_rew); cpd_1_3_rew = re._CPD(X_1_rew,firing_rates_3_inf_rew)
       
        cpd_2_1_rew = re._CPD(X_2_rew,firing_rates_1_inf_rew); cpd_2_3_rew = re._CPD(X_2_rew,firing_rates_3_inf_rew)
        
        cpd_3_1_rew = re._CPD(X_3_rew,firing_rates_1_inf_rew); cpd_3_2_rew = re._CPD(X_3_rew,firing_rates_2_inf_rew)
        
        cpd_1_2_error = re._CPD(X_1_error,firing_rates_2_inf_error); cpd_1_3_error = re._CPD(X_1_error,firing_rates_3_inf_error)
       
        cpd_2_1_error = re._CPD(X_2_error,firing_rates_1_inf_error); cpd_2_3_error = re._CPD(X_2_error,firing_rates_3_inf_error)
        
        cpd_3_1_error = re._CPD(X_3_error,firing_rates_1_inf_error); cpd_3_2_error = re._CPD(X_3_error,firing_rates_2_inf_error)
        
        
      
    
    cpd_rew = np.nanmean([cpd_1_2_rew,cpd_1_3_rew,cpd_2_1_rew,cpd_2_3_rew,cpd_3_1_rew,cpd_3_2_rew],0)
    cpd_error = np.nanmean([cpd_1_2_error,cpd_1_3_error,cpd_2_1_error,cpd_2_3_error,cpd_3_1_error,cpd_3_2_error],0)
     
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    if interactions == True:
        sub = 2
    else:
        sub = 1
    plt.figure()
    plt.subplot(2,sub,1)
    plt.plot(cpd_rew[:,0], color = pal[0], label = 'Pre Init Period')
    plt.plot(cpd_rew[:,1], color = pal[2], label = 'Init Period')
    plt.plot(cpd_rew[:,2], color = pal[4], label = 'Choice Period')
    plt.plot(cpd_rew[:,3], color = pal[6], label = 'Reward Period')
    plt.title( area +'Between Tasks Reward Runs')
    plt.legend() 
    sns.despine()
    # plt.ylim(0,0.18)
   
    plt.subplot(2,sub,2)

    plt.plot(cpd_error[:,0], color = pal_c[0], label = 'Pre Init Period')
    plt.plot(cpd_error[:,1], color = pal_c[2], label = 'Init Period')
    plt.plot(cpd_error[:,2], color = pal_c[4], label = 'Choice Period')
    plt.plot(cpd_error[:,3], color = pal_c[6], label = 'Reward Period')
    plt.title( area +'Between Tasks Error Runs')
    # plt.ylim(0,0.06)
    if interactions == True:
        plt.subplot(2,sub,3)
        plt.plot(cpd_rew_int[:,0], color = pal[0], label = 'Pre Init Period')
        plt.plot(cpd_rew_int[:,1], color = pal[2], label = 'Init Period')
        plt.plot(cpd_rew_int[:,2], color = pal[4], label = 'Choice Period')
        plt.plot(cpd_rew_int[:,3], color = pal[6], label = 'Reward Period')
        plt.title( area +'Between Tasks Reward Runs Interactions with Choice')
        plt.legend() 
        sns.despine()
        # plt.ylim(0,0.18)
       
        plt.subplot(2,sub,4)
    
        plt.plot(cpd_error_int[:,0], color = pal_c[0], label = 'Pre Init Period')
        plt.plot(cpd_error_int[:,1], color = pal_c[2], label = 'Init Period')
        plt.plot(cpd_error_int[:,2], color = pal_c[4], label = 'Choice Period')
        plt.plot(cpd_error_int[:,3], color = pal_c[6], label = 'Reward Period')
        plt.title( area +'Between Tasks Error Runs Interactions with Choice')
        # plt.ylim(0,0.06)

    plt.legend() 
    sns.despine()
    
        
   
    cpd_rew_perm = []
    cpd_error_perm = []
    cpd_rew_int_perm = []
    cpd_error_int_perm = []

    for i, ii, in enumerate(C_3_perm):
        
        C_1 = np.concatenate(C_1_perm[i],1)

        C_2 = np.concatenate(C_3_perm[i],1)
    
        C_3 = np.concatenate(C_2_perm[i],1)
        
        if interactions == True:
            X_nan, firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,\
            firing_rates_3_inf_error,firing_rates_1_inf_rew_int,firing_rates_2_inf_rew_int,firing_rates_3_inf_rew_int,firing_rates_1_inf_error_int,\
               firing_rates_2_inf_error_int,firing_rates_3_inf_error_int = find_coefficients(C_1,C_2,C_3,interactions = interactions,  a = a)
    
            X_1_rew = X_nan[:,:5]
            X_1_rew_int = X_nan[:,5:10]
        
            X_1_error = X_nan[:,10:15]
            X_1_error_int = X_nan[:,15:20]
        
            X_2_rew = X_nan[:,20:25]
            X_2_rew_int = X_nan[:,25:30]
        
            X_2_error = X_nan[:,30:35]
            X_2_error_int = X_nan[:,35:40]
        
            X_3_rew = X_nan[:,40:45]
            X_3_rew_int = X_nan[:,45:50]
        
            X_3_error = X_nan[:,50:55]
            X_3_error_int = X_nan[:,55:60]
            
            cpd_1_2_rew = re._CPD(X_1_rew,firing_rates_2_inf_rew); cpd_1_3_rew = re._CPD(X_1_rew,firing_rates_3_inf_rew)
           
            cpd_2_1_rew = re._CPD(X_2_rew,firing_rates_1_inf_rew); cpd_2_3_rew = re._CPD(X_2_rew,firing_rates_3_inf_rew)
            
            cpd_3_1_rew = re._CPD(X_3_rew,firing_rates_1_inf_rew); cpd_3_2_rew = re._CPD(X_3_rew,firing_rates_2_inf_rew)
        
            
            cpd_1_2_error = re._CPD(X_1_error,firing_rates_2_inf_error); cpd_1_3_error = re._CPD(X_1_error,firing_rates_3_inf_error)
           
            cpd_2_1_error = re._CPD(X_2_error,firing_rates_1_inf_error); cpd_2_3_error = re._CPD(X_2_error,firing_rates_3_inf_error)
            
            cpd_3_1_error = re._CPD(X_3_error,firing_rates_1_inf_error); cpd_3_2_error = re._CPD(X_3_error,firing_rates_2_inf_error)
            
           
          
            
            cpd_1_2_rew_int = re._CPD(X_1_rew_int,firing_rates_2_inf_rew_int); cpd_1_3_rew_int = re._CPD(X_1_rew_int,firing_rates_3_inf_rew_int)
       
            cpd_2_1_rew_int = re._CPD(X_2_rew_int,firing_rates_1_inf_rew_int); cpd_2_3_rew_int = re._CPD(X_2_rew_int,firing_rates_3_inf_rew_int)
            
            cpd_3_1_rew_int = re._CPD(X_3_rew_int,firing_rates_1_inf_rew_int); cpd_3_2_rew_int = re._CPD(X_3_rew_int,firing_rates_2_inf_rew_int)
            
            cpd_1_2_error_int = re._CPD(X_1_error_int,firing_rates_2_inf_error_int); cpd_1_3_error_int = re._CPD(X_1_error_int,firing_rates_3_inf_error_int)
           
            cpd_2_1_error_int = re._CPD(X_2_error_int,firing_rates_1_inf_error_int); cpd_2_3_error_int = re._CPD(X_2_error_int,firing_rates_3_inf_error_int)
            
            cpd_3_1_error_int = re._CPD(X_3_error_int,firing_rates_1_inf_error_int); cpd_3_2_error_int = re._CPD(X_3_error_int,firing_rates_2_inf_error_int)
            
            
            cpd_rew_int_perm.append(np.nanmean([cpd_1_2_rew_int,cpd_1_3_rew_int,cpd_2_1_rew_int,cpd_2_3_rew_int,cpd_3_1_rew_int,cpd_3_2_rew_int],0))
            cpd_error_int_perm.append(np.nanmean([cpd_1_2_error_int,cpd_1_3_error_int,cpd_2_1_error_int,cpd_2_3_error_int,cpd_3_1_error_int,cpd_3_2_error_int],0))


        else:
        
            X_nan, firing_rates_1_inf_rew,firing_rates_2_inf_rew,firing_rates_3_inf_rew,firing_rates_1_inf_error,firing_rates_2_inf_error,\
            firing_rates_3_inf_error  = find_coefficients(C_1,C_2,C_3,interactions = interactions,  a = a)
            
            X_1_rew = X_nan[:,:5]
        
            X_1_error = X_nan[:,5:10]
        
            X_2_rew = X_nan[:,10:15]
        
            X_2_error = X_nan[:,15:20]
        
            X_3_rew = X_nan[:,20:25]
        
            X_3_error = X_nan[:,25:30]
        
    
    
            cpd_1_2_rew = re._CPD(X_1_rew,firing_rates_2_inf_rew); cpd_1_3_rew = re._CPD(X_1_rew,firing_rates_3_inf_rew)
           
            cpd_2_1_rew = re._CPD(X_2_rew,firing_rates_1_inf_rew); cpd_2_3_rew = re._CPD(X_2_rew,firing_rates_3_inf_rew)
            
            cpd_3_1_rew = re._CPD(X_3_rew,firing_rates_1_inf_rew); cpd_3_2_rew = re._CPD(X_3_rew,firing_rates_2_inf_rew)
            
            cpd_1_2_error = re._CPD(X_1_error,firing_rates_2_inf_error); cpd_1_3_error = re._CPD(X_1_error,firing_rates_3_inf_error)
           
            cpd_2_1_error = re._CPD(X_2_error,firing_rates_1_inf_error); cpd_2_3_error = re._CPD(X_2_error,firing_rates_3_inf_error)
            
            cpd_3_1_error = re._CPD(X_3_error,firing_rates_1_inf_error); cpd_3_2_error = re._CPD(X_3_error,firing_rates_2_inf_error)
            
           
            
      
    
        
        cpd_rew_perm.append(np.nanmean([cpd_1_2_rew,cpd_1_3_rew,cpd_2_1_rew,cpd_2_3_rew,cpd_3_1_rew,cpd_3_2_rew],0))
        cpd_error_perm.append(np.nanmean([cpd_1_2_error,cpd_1_3_error,cpd_2_1_error,cpd_2_3_error,cpd_3_1_error,cpd_3_2_error],0))
    
        
    cpd_rew_perm =  np.max(np.percentile(cpd_rew_perm,95, axis = 0),0)
    cpd_error_perm =  np.max(np.percentile(cpd_error_perm,95, axis = 0),0)
  
    
    plt.subplot(2,sub,1)
    plt.hlines(cpd_rew_perm[0], xmin = 0, xmax = 63,color = pal[0], label = 'Pre Init Period', linestyle = ':')
    plt.hlines(cpd_rew_perm[1], xmin = 0, xmax = 63,color = pal[2], label = 'Init Period',linestyle = ':')
    plt.hlines(cpd_rew_perm[2], xmin = 0, xmax = 63,color = pal[4], label = 'Choice Period',linestyle = ':')
    plt.hlines(cpd_rew_perm[3], xmin = 0, xmax = 63, color = pal[6], label = 'Reward Period',linestyle = ':')
    plt.legend() 
    sns.despine()
   
    plt.subplot(2,sub,2)

    plt.hlines(cpd_error_perm[0], xmin = 0, xmax = 63, color = pal_c[0], label = 'Pre Init Period',linestyle = ':')
    plt.hlines(cpd_error_perm[1], xmin = 0, xmax = 63, color = pal_c[2], label = 'Init Period',linestyle = ':')
    plt.hlines(cpd_error_perm[2], xmin = 0, xmax = 63,color = pal_c[4], label = 'Choice Period',linestyle = ':')
    plt.hlines(cpd_error_perm[3], xmin = 0, xmax = 63, color = pal_c[6], label = 'Reward Period',linestyle = ':')
    plt.legend() 
    sns.despine()
    
    if interactions ==True:
        
        cpd_rew_int_perm =  np.max(np.percentile(cpd_rew_int_perm,95, axis = 0),0)
        cpd_error_int_perm =  np.max(np.percentile(cpd_error_int_perm,95, axis = 0),0)

        plt.subplot(2,sub,3)
        plt.hlines(cpd_rew_int_perm[0],  xmin = 0, xmax = 63, color = pal[0], label = 'Pre Init Period',linestyle = ':')
        plt.hlines(cpd_rew_int_perm[1],  xmin = 0, xmax = 63, color = pal[2], label = 'Init Period',linestyle = ':')
        plt.hlines(cpd_rew_int_perm[2], xmin = 0, xmax = 63,  color = pal[4], label = 'Choice Period',linestyle = ':')
        plt.hlines(cpd_rew_int_perm[3],  xmin = 0, xmax = 63, color = pal[6], label = 'Reward Period',linestyle = ':')
        plt.legend() 
        sns.despine()
        plt.subplot(2,sub,4)
    
        plt.hlines(cpd_error_int_perm[0], xmin = 0, xmax = 63, color = pal_c[0], label = 'Pre Init Period',linestyle = ':')
        plt.hlines(cpd_error_int_perm[1], xmin = 0, xmax = 63, color = pal_c[2], label = 'Init Period',linestyle = ':')
        plt.hlines(cpd_error_int_perm[2], xmin = 0, xmax = 63, color = pal_c[4], label = 'Choice Period',linestyle = ':')
        plt.hlines(cpd_error_int_perm[3], xmin = 0, xmax = 63, color = pal_c[6], label = 'Reward Period',linestyle = ':')
        plt.legend() 
        sns.despine()

            
            
def run():
    
    sequence_rewards_errors_regression_generalisation(PFC, perm = 1000, area = 'PFC A' + ' ', interactions = False, a = True)
    sequence_rewards_errors_regression_generalisation(PFC, perm = 1000, area = 'PFC B' + ' ', interactions = False, a = False)

    sequence_rewards_errors_regression_generalisation(HP, perm = 1000, area = 'HP A' + ' ', interactions = False, a = True)
    sequence_rewards_errors_regression_generalisation(HP, perm = 1000, area = 'HP B' + ' ', interactions = False,  a = False)
   
    sequence_rewards_errors_regression_generalisation(PFC, perm = 1000, area = 'PFC' + ' ', interactions = True)
    sequence_rewards_errors_regression_generalisation(HP, perm = 1000,area = 'HP' + ' ',interactions = True)
  
    
 