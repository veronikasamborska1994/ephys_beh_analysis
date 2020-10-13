#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:33:20 2020

@author: veronikasamborska
"""

import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
import align_trial_choices_rewards as ch_rew_align
import seaborn as sns
from collections import OrderedDict
import regression_function as reg_f
from palettable import wesanderson as wes
from scipy import io
import statsmodels.api as sm
import seaborn as sns
from itertools import combinations 
import regressions as re
from scipy.fftpack import rfft, irfft
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

def phasescrable(ts):
    
    '''Returns a time series where power is preserved but phase is shuffled.'''
    
    fs = rfft(ts)
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    
    return tsr


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

 #des=[choice (choice-.5).*reward  reward stay stay.*(reward-0.5) stay.*(choice-0.5)  stay.*(choice-0.5).*(lastreward-.5)  const ];
def run_between_tasks():
    
    time_in_block(PFC, area = 'PFC')
    time_in_block(HP, area = 'HP')
    
    
#                                    0 ('Choice', choices_1),
#                                    1 ('Reward', rewards_1),
#                                    2 ('Reward Repeat/Switch', reward_PE_1),
#                                    3 ('Choice Repeat/Switch', choice_PE_1),
#                                    4 ('Repeat/Switch Current Reward', choice_PE_1_reward_current),
#                                    5 ('Repeat/Switch Prev Reward', choice_PE_1_reward_prev),
#                                    6 ('Choice x Reward ',rew_ch_1),
#                                    7 (' Ch x Stay x Last Reward',prev_choice_1_lr),

#                                    8 ('Prev Rew', prev_reward_1),
                                     
#                                    9  ('Prev Ch', prev_choice_1),
#                                    10  ('ones', ones_1)])
        
def time_in_block(data, area = 'PFC'):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    C_1 = []; C_2 = []; C_3 = []
    cpd_1 = []; cpd_2 = []; cpd_3 = []
   # cpd_perm_p_1 = []; cpd_perm_p_2 = []; cpd_perm_p_3 = []


    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s][1:]
       
        # firing_rates = firing_rates[:,:,:63]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]-0.5
        reward = DM[:,2] -0.5   

        task =  DM[:,5][1:]
        a_pokes = DM[:,6][1:]
        b_pokes = DM[:,7][1:]
        
      
        reward_prev = reward[:-1]
        reward = reward[1:]
        
       
        choices_prev = choices[:-1]
        choices = choices[1:]
     
   
        taskid = task_ind(task, a_pokes, b_pokes)
        
        
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]
        reward_PE = np.zeros(len(task))
        for r,rr in enumerate(reward):
            if reward[r] != reward[r-1]:
                reward_PE[r] = 0.5
            elif reward[r] == reward[r-1]:
                reward_PE[r] = -0.5
                
        choice_PE = np.zeros(len(task))
        for r,rr in enumerate(choices):
            if choices[r] != choices[r-1]:
                choice_PE[r] = 0.5
            elif choices[r] == choices[r-1]:
                choice_PE[r] = -0.5

        
        reward_PE_1 = reward_PE[task_1]
        choice_PE_1 = choice_PE[task_1]

       
        rewards_1 = reward[task_1]
        choices_1 = choices[task_1]
        rewards_1 = reward[task_1]
        ones_1 = np.ones(len(choices_1))
        trials_1 = len(choices_1)
        prev_reward_1 = reward_prev[task_1]
        prev_choice_1 = choices_prev[task_1]
        #prev_choice_1 = choices_1*choice_PE_1
        choice_PE_1_reward_current = choice_PE_1*rewards_1
        choice_PE_1_reward_prev = choice_PE_1*prev_reward_1
        
        rew_ch_1 = choices_1*rewards_1
        prev_choice_1_lr = prev_choice_1*prev_reward_1

        firing_rates_1= firing_rates[task_1]
        predictors_all = OrderedDict([
                                    ('Choice', choices_1),
                                    ('Reward', rewards_1),
                                    ('Reward Repeat/Switch', reward_PE_1),
                                    ('Choice Repeat/Switch', choice_PE_1),
                                    ('Repeat/Switch Current Reward', choice_PE_1_reward_current),
                                    ('Repeat/Switch Prev Reward', choice_PE_1_reward_prev),
                                    ('Choice x Reward ',rew_ch_1),
                                    ('Prev Ch x Last Reward',prev_choice_1_lr),

                                    ('Prev Rew', prev_reward_1),
                                     
                                     ('Prev Ch', prev_choice_1),
                                    ('ones', ones_1)])
        
        X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
        
        n_predictors = X_1.shape[1]
        y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y_1, X_1)
        C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(re._CPD(X_1,y_1).reshape(n_neurons, n_timepoints, n_predictors))
        
        choices_2 = choices[task_2]
        rewards_2 = reward[task_2]
        ones_2 = np.ones(len(choices_2))
        reward_PE_2 = reward_PE[task_2]
        choice_PE_2 = choice_PE[task_2]

        prev_reward_2 = reward_prev[task_2]
        prev_choice_2 = choices_prev[task_2]
        #prev_choice_2 = choices_2*choice_PE_2

        choice_PE_2_reward_current = choice_PE_2*rewards_2
        choice_PE_2_reward_prev = choice_PE_2*prev_reward_2

        trials_2 = len(choices_2)
        rew_ch_2 = choices_2*rewards_2
        prev_choice_2_lr = prev_choice_2*prev_reward_2

        ones_2 = np.ones(len(choices_2))
        firing_rates_2 = firing_rates[task_2]

        predictors_all = OrderedDict([
                                    ('Choice', choices_2),
                                    ('Reward', rewards_2),
                                    ('Reward Repeat/Switch', reward_PE_2),
                                    ('Choice Repeat/Switch', choice_PE_2),
                                    ('Repeat/Switch Current Reward', choice_PE_2_reward_current),
                                    ('Repeat/Switch Prev Reward', choice_PE_2_reward_prev),
                                    ('Choice x Reward ',rew_ch_2),
                                    (' Prev Ch x Last Reward',prev_choice_2_lr),

                                     ('Prev Rew', prev_reward_2),
                                     ('Prev Ch', prev_choice_2),
                                    ('ones', ones_2)])
        
        X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
        
        n_predictors = X_2.shape[1]
        y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y_2, X_2)
        C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_2.append(re._CPD(X_2,y_2).reshape(n_neurons, n_timepoints, n_predictors))
  
    
        choices_3 = choices[task_3]
        rewards_3 = reward[task_3]
        ones_3 = np.ones(len(choices_3))
        trials_3 = len(choices_3)
        ones_3 = np.ones(len(choices_3))
        prev_reward_3 = reward_prev[task_3]
        choice_PE_3 = choice_PE[task_3]
        reward_PE_3 = reward_PE[task_3]
        choice_PE_3_reward_current = choice_PE_3*rewards_3
        choice_PE_3_reward_prev = choice_PE_3*prev_reward_3
        prev_choice_3 = choices_prev[task_3]

        #prev_choice_3 = choices_3*choice_PE_3

        rew_ch_3 = choices_3*rewards_3
        prev_choice_3_lr = prev_choice_3*prev_reward_3

        firing_rates_3 = firing_rates[task_3]

        predictors_all = OrderedDict([
                                    ('Ch', choices_3),
                                    ('Rew', rewards_3),
                                    ('Rew Stay', reward_PE_3),
                                    ('Ch Stay', choice_PE_3),
                                    ('Stay Cur Rew', choice_PE_3_reward_current),
                                    ('Stay Prev Rew', choice_PE_3_reward_prev),
                                    ('Ch x Rew ',rew_ch_3),
                                    (' Prev Ch x Prev Rew',prev_choice_3_lr),

                                    ('Prev Rew', prev_reward_3),
                                     ('Prev Ch', prev_choice_3),
                                    ('ones', ones_3)])
        
        X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
        rank = np.linalg.matrix_rank(X_3)
        print(rank)
        n_predictors = X_3.shape[1]
        print(n_predictors)
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y_3, X_3)
        C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_3.append(re._CPD(X_3,y_3).reshape(n_neurons, n_timepoints, n_predictors))
        
    
    C_1 = np.concatenate(C_1,1)
    
    C_2 = np.concatenate(C_2,1)
    
    C_3 = np.concatenate(C_3,1)
    
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    cpd = np.mean([cpd_1,cpd_2,cpd_3],0)

    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
        
    j = 0
    plt.figure()
    pred = list(predictors_all.keys())
    pred = pred[2:-1]
    for ii,i in enumerate(cpd.T[2:-1]):
        plt.plot(i, color = c[j],label = pred[j])
      
        j+=1
    plt.legend()
    sns.despine()
    
    C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)

    C_1 = C_1[:,nans[0],:]; C_2 = C_2[:,nans[0],:];  C_3 = C_3[:,nans[0],:]
    
    preds = np.arange(len(list(predictors_all.keys())))
    coef_1 = np.tile(preds,11)
    coef_2 = np.concatenate((preds, np.roll(preds,1), np.roll(preds,2),np.roll(preds,3), np.roll(preds,4),np.roll(preds,5),\
                            np.roll(preds,6), np.roll(preds,7), np.roll(preds,8),np.roll(preds,9), np.roll(preds,10)))
        
    m = 0
    l = 0
    plt.figure(figsize = (10,10))
    for c_1, c_2 in zip(coef_1,coef_2):
        title = list(predictors_all.keys())[c_1] +  ' '  +'on' + ' '+list(predictors_all.keys())[c_2] 

        m+=1
        l+=1
        if m == 10:
            plt.savefig('/Users/veronikasamborska/Desktop/runs/within'+area +str(l)+'.png')
            plt.figure(figsize = (10,10))
            m -=9
        
    
        C_1_rew = C_1[c_1]; C_2_rew = C_2[c_1]; C_3_rew = C_3[c_1]
        C_1_rew_count = C_1[c_2]; C_2_rew_count = C_2[c_2]; C_3_rew_count = C_3[c_2]
       
        reward_times_to_choose = np.asarray([20,24,35,41])
        
        C_1_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
        C_2_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
        C_3_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
       
        j = 0
        for i in reward_times_to_choose:
            if i ==reward_times_to_choose[0]:
                C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-20:i],1)
                C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-20:i],1)
                C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-20:i],1)
            elif i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
                C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-5:i+5],1)
                C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-5:i+5],1)
                C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-5:i+5],1)
            elif i == reward_times_to_choose[3]:
                C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i:i+5],1)
                C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i:i+5],1)
                C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i:i+5],1)
             
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
            elif i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
                C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-5:i+5],1)
                C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-5:i+5],1)
                C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-5:i+5],1)
            elif i == reward_times_to_choose[3]:
                C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i:i+5],1)
                C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i:i+5],1)
                C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i:i+5],1)
          
            j +=1
            
        cpd_1_2_rew, cpd_1_2_rew_var = regression_code_session(C_1_rew_count, C_1_rew_proj);  
        cpd_1_3_rew, cpd_1_3_rew_var = regression_code_session(C_2_rew_count, C_2_rew_proj); 
        cpd_2_3_rew, cpd_2_3_rew_var = regression_code_session(C_3_rew_count, C_3_rew_proj)
        
        rew_to_count_cpd = (cpd_1_2_rew + cpd_1_3_rew +cpd_2_3_rew)/np.sqrt((cpd_1_2_rew_var+cpd_1_3_rew_var+cpd_2_3_rew_var))
        
        j = 0
        plt.subplot(5,2,m)
        for i in rew_to_count_cpd[:-1]:
            plt.plot(i, color = c[j], label = str(j))
            j+=1
       # plt.legend()  
        plt.title(area+ ' ' + str(title))
        sns.despine()
        
        plt.tight_layout()
        
    
     
      
    
    
          
