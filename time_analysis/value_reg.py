#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:25:08 2020

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
from statsmodels.regression.linear_model import OLS
import scipy
from scipy.fftpack import rfft, irfft
from math import factorial
import itertools

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)


def load():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    ## Longer trial
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP_RPE.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC_RPE.mat')
  

def run_between_tasks(PFC, HP):
    
    C_PFC = time_in_block(PFC, area = 'PFC_A_prev', n = 5, plot_a = True, plot_b = False, to_plot = 1)
    C_HP = time_in_block(HP, area = 'HP_A_prev', n = 5, plot_a = True, plot_b = False, to_plot = 1)
    
    C_PFC = time_in_block(PFC, area = 'PFC_B_prev', n = 5, plot_a = False, plot_b = True,to_plot = 1)
    C_HP = time_in_block(HP, area = 'HP_B_prev', n = 5, plot_a = False, plot_b = True, to_plot = 1)
    
     # predictors_all = OrderedDict([
     #                                ('Choice', choices_1),
     #                                ('Reward', rewards_1),
     #                                ('Choice x Reward ',rew_ch_1),

     #                                ('Reward Repeat/Switch', reward_PE_1),
     #                                ('Choice Repeat/Switch', choice_PE_1),
     #                                ('Repeat/Switch Current Reward', choice_PE_1_reward_current),
     #                                ('Repeat/Switch Prev Reward', choice_PE_1_reward_prev),
     #                                ('Prev Ch x Last Reward',prev_choice_1_lr),
     #                                ('Value',value_1), 
     #                                ('Value Сhoice',value_1_choice_1), 

     #                                ('Prev Rew', prev_reward_1),
                                     
     #                                  ('Prev Ch', prev_choice_1),
     #                                ('ones', ones_1)])
     
    
  
     
def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid


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


def rew_prev_behaviour(data,n, perm = True):
    if perm:
        dm = data[0]
    else:
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
         
        # model = sm.Logit(1-choices_current,X)
         model = OLS(choices_current-0.5,X)
         results = model.fit()
         results_array.append(results.params)
         cov = results.cov_params()
         std_err.append(np.sqrt(np.diag(cov)))

    average = np.mean(results_array,0)
    std = np.std(results_array,0)/len(dm)

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



def generalisation_plot(C_1,C_2,C_3, c_1):
    c_1 = c_1
    C_1_rew = C_1[c_1]; C_2_rew = C_2[c_1]; C_3_rew = C_3[c_1]
    C_1_rew_count = C_1[c_1]; C_2_rew_count = C_2[c_1]; C_3_rew_count = C_3[c_1]
   
    reward_times_to_choose = np.asarray([20,25,35,42])
    
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
        
    
        
    cpd_1_2_rew, cpd_1_2_rew_var = regression_code_session(C_2_rew_count, C_1_rew_proj);  
    cpd_1_3_rew, cpd_1_3_rew_var = regression_code_session(C_3_rew_count, C_1_rew_proj); 
    cpd_2_3_rew, cpd_2_3_rew_var = regression_code_session(C_3_rew_count, C_2_rew_proj)
    
    value_to_value = (cpd_1_2_rew + cpd_1_3_rew +cpd_2_3_rew)/np.sqrt((cpd_1_2_rew_var+cpd_1_3_rew_var+cpd_2_3_rew_var))
    
    return value_to_value

    
     
def plot(HP, PFC, perm = False, c_1 = 1):
    
    c_1 = 1
    C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP, area = 'HP', n = 12, plot_a = False, plot_b = True, perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC, area = 'PFC', n = 12, plot_a = False, plot_b = True, perm = False)
    
    C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP, area = 'HP', n = 12, plot_a = True, plot_b = False, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC, area = 'PFC', n = 12, plot_a = True, plot_b = False, perm = False)
  
    value_to_value_PFC_a = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1)
    value_to_value_PFC_b = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1)


    value_to_value_HP_a = generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1)
    value_to_value_HP_b = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1)

    # c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
    # plt.figure()
    # j = 0
    # for i in value_to_value_HP_a[:-1]:
    #     plt.plot(i, color = c[j], label = str(j))
    #     j+=1
    # sns.despine()
    
    # plt.tight_layout()
   
    if perm:
        difference_a = []
        difference_b = []
        all_subjects = [HP['DM'][0][:16], HP['DM'][0][16:24],HP['DM'][0][24:],PFC['DM'][0][:9], PFC['DM'][0][9:26],PFC['DM'][0][26:40],PFC['DM'][0][40:]]
        all_subjects_firing = [HP['Data'][0][:16], HP['Data'][0][16:24],HP['Data'][0][24:],PFC['Data'][0][:9], PFC['Data'][0][9:26],PFC['Data'][0][26:40],PFC['Data'][0][40:]]

        animals_PFC = [1,2,3,4]
        animals_HP = [5,6,7]
        m, n = len(animals_PFC), len(animals_HP)
      
        for indices_PFC in combinations(range(m + n), m):
            indices_HP = [i for i in range(m + n) if i not in indices_PFC]
            PFC_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_PFC)])
            HP_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_HP)])
            
            PFC_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_PFC)])
            HP_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_HP)])
            HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
            PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]

            C_1_HP_b, C_2_HP_b, C_3_HP_b = time_in_block(HP_shuffle, area = 'HP', n = 12, plot_a = False, plot_b = True,  perm = True)
            C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = time_in_block(PFC_shuffle, area = 'PFC', n = 12, plot_a = False, plot_b = True,  perm = True)
            
            C_1_HP_a, C_2_HP_a, C_3_HP_a = time_in_block(HP_shuffle, area = 'HP', n = 12, plot_a = True, plot_b = False,  perm = True)
            C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = time_in_block(PFC_shuffle, area = 'PFC', n = 12, plot_a = True, plot_b = False ,  perm = True)
            
            value_to_value_PFC_a_perm = generalisation_plot(C_1_PFC_a,C_2_PFC_a,C_3_PFC_a, c_1)
            value_to_value_PFC_b_perm = generalisation_plot(C_1_PFC_b,C_2_PFC_b,C_3_PFC_b, c_1)
        
        
            value_to_value_HP_a_perm =  generalisation_plot(C_1_HP_a,C_2_HP_a,C_3_HP_a, c_1)
            value_to_value_HP_b_perm = generalisation_plot(C_1_HP_b,C_2_HP_b,C_3_HP_b, c_1)
            
            difference_a.append(np.abs(value_to_value_PFC_a_perm-value_to_value_HP_a_perm))
            
            difference_b.append(np.abs(value_to_value_PFC_b_perm-value_to_value_HP_b_perm))
            
    perm_a = np.max(np.percentile(difference_a,95,0),1)
    
    perm_b = np.max(np.percentile(difference_b,95,0),1)
    
    a = np.abs(value_to_value_PFC_a - value_to_value_HP_a)
    b = np.abs(value_to_value_PFC_b - value_to_value_HP_b)
    
    # c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
    # plt.figure()
    # j = 0
    # for i in value_to_value_HP_a[:-1]:
    #     plt.plot(i, color = c[j], label = str(j))
    #     j+=1
   
    # sns.despine() 
    # plt.tight_layout()
   
    
def time_in_block(data, area = 'PFC', n = 10, plot_a = False, plot_b = False, perm = True):
    if perm:
        dm = data[0]
        firing = data[1]

    else:
        dm = data['DM'][0]
        firing = data['Data'][0]

    C_1 = []; C_2 = []; C_3 = []
    cpd_1 = []; cpd_2 = []; cpd_3 = []
    average = rew_prev_behaviour(data, n = n, perm = perm)

    for  s, sess in enumerate(dm):
        
       
        DM = dm[s]
        firing_rates = firing[s][n:]
       
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        choices = DM[:,1]
        reward = DM[:,2]  

        task =  DM[:,5][n:]
        
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]
        task_3 = np.where(task == 3)[0]
         
        previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
         
        previous_choices = scipy.linalg.toeplitz(0.5-choices, np.zeros((1,n)))[n-1:-1]
         
        interactions = scipy.linalg.toeplitz((((0.5-choices)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
         

        choices_current = choices[n:]
        ones = np.ones(len(interactions)).reshape(len(interactions),1)
         
        X = np.hstack([previous_rewards,previous_choices,interactions,ones])
        value = np.matmul(X, average)
        
        reward_current = reward[n:]
        choices_current = choices[n:]-0.5
        reward_prev = reward[n-1:]
        choices_prev = choices[n-1:]-0.5
        
        reward_PE = np.zeros(len(task))
        for r,rr in enumerate(reward_current):
            if reward_current[r] != reward_current[r-1]:
                reward_PE[r] = 0.5
            elif reward_current[r] == reward_current[r-1]:
                reward_PE[r] = -0.5
                
        choice_PE = np.zeros(len(task))
        for r,rr in enumerate(choices_current):
            if choices_current[r] != choices_current[r-1]:
                choice_PE[r] = 0.5
            elif choices_current[r] == choices_current[r-1]:
                choice_PE[r] = -0.5

        
        reward_PE_1 = reward_PE[task_1]
        choice_PE_1 = choice_PE[task_1]

       
        rewards_1 = reward_current[task_1]
        choices_1 = choices_current[task_1]
        ones_1 = np.ones(len(choices_1))
        trials_1 = len(choices_1)
        prev_reward_1 = reward_prev[task_1]
        prev_choice_1 = choices_prev[task_1]
        #prev_choice_1 = choices_1*choice_PE_1
        choice_PE_1_reward_current = choice_PE_1*rewards_1
        choice_PE_1_reward_prev = choice_PE_1*prev_reward_1
        value_1 = value[task_1]
        value_1_choice_1 = value_1*choices_1
        rew_ch_1 = choices_1*rewards_1
        prev_choice_1_lr = prev_choice_1*prev_reward_1

        firing_rates_1 = firing_rates[task_1]
        
        a_1 = np.where(choices_1 == 0.5)[0]
        b_1 = np.where(choices_1 == -0.5)[0]
        
        if plot_a == True:
            rewards_1 = rewards_1[a_1] 
            choices_1 = choices_1[a_1]
            rew_ch_1 = rew_ch_1[a_1]
            value_1 = value_1[a_1]
            value_1_choice_1 = value_1_choice_1[a_1]
            ones_1  = ones_1[a_1]

            firing_rates_1 = firing_rates_1[a_1]
            
            reward_PE_1 = reward_PE_1[a_1]
            choice_PE_1 = choice_PE_1[a_1]
            choice_PE_1_reward_current = choice_PE_1_reward_current[a_1]
            choice_PE_1_reward_prev = choice_PE_1_reward_prev[a_1]
            prev_choice_1_lr = prev_choice_1_lr[a_1]
            prev_reward_1 = prev_reward_1[a_1]
            prev_choice_1 = prev_choice_1[a_1]
          
        elif plot_b == True:
            
            rewards_1 = rewards_1[b_1] 
            choices_1 = choices_1[b_1]
            rew_ch_1 = rew_ch_1[b_1]
            value_1 = value_1[b_1]
            value_1_choice_1 = value_1_choice_1[b_1]
            ones_1  = ones_1[b_1]

            firing_rates_1 = firing_rates_1[b_1]
            
            reward_PE_1 = reward_PE_1[b_1]
            choice_PE_1 = choice_PE_1[b_1]
            choice_PE_1_reward_current = choice_PE_1_reward_current[b_1]
            choice_PE_1_reward_prev = choice_PE_1_reward_prev[b_1]
            prev_choice_1_lr = prev_choice_1_lr[b_1]
            prev_reward_1 = prev_reward_1[b_1]
            prev_choice_1 = prev_choice_1[b_1]
          
        predictors_all = OrderedDict([
                                    #('Choice', choices_1),
                                    ('Reward', rewards_1),
                                    ('Value',value_1), 
                                    # ('Value Сhoice',value_1_choice_1), 
                                    ('Prev Rew', prev_reward_1),
                                    ('Prev Ch', prev_choice_1),
                                    ('ones', ones_1)])
        
        X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
        
        n_predictors = X_1.shape[1]
        y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats,x = regression_code_session(y_1, X_1)
        C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(re._CPD(X_1,y_1).reshape(n_neurons, n_timepoints, n_predictors))
        
        choices_2 = choices_current[task_2]
        rewards_2 = reward_current[task_2]
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
        value_2= value[task_2]
        value_2_choice_2 = value_2*choices_2

        ones_2 = np.ones(len(choices_2))
        firing_rates_2 = firing_rates[task_2]
   
        a_2 = np.where(choices_2 == 0.5)[0]
        b_2 = np.where(choices_2 == -0.5)[0]
        
        if plot_a == True:
            rewards_2 = rewards_2[a_2] 
            choices_2 = choices_2[a_2]
            value_2 = value_2[a_2]
            ones_2  = ones_2[a_2]
            firing_rates_2 = firing_rates_2[a_2]
            reward_PE_2 = reward_PE_2[a_2]
            choice_PE_2 = choice_PE_2[a_2]
            choice_PE_2_reward_current = choice_PE_2_reward_current[a_2]
            choice_PE_2_reward_prev = choice_PE_2_reward_prev[a_2]
            prev_choice_2_lr = prev_choice_2_lr[a_2]
            prev_reward_2 = prev_reward_2[a_2]
            prev_choice_2 = prev_choice_2[a_2]
            
        elif plot_b == True:
            
            rewards_2 = rewards_2[b_2] 
            choices_2 = choices_2[b_2]
            value_2 = value_2[b_2]
            ones_2  = ones_2[b_2]
            firing_rates_2 = firing_rates_2[b_2]
            
            reward_PE_2 = reward_PE_2[b_2]
            choice_PE_2 = choice_PE_2[b_2]
            choice_PE_2_reward_current = choice_PE_2_reward_current[b_2]
            choice_PE_2_reward_prev = choice_PE_2_reward_prev[b_2]
            prev_choice_2_lr = prev_choice_2_lr[b_2]
            prev_reward_2 = prev_reward_2[b_2]
            prev_choice_2 = prev_choice_2[b_2]
    
        predictors_all = OrderedDict([
                                    # ('Choice', choices_2),
                                    ('Reward', rewards_2),
                                    ('Value',value_2), 
                                    # ('Value Сhoice',value_2_choice_2), 

                                    ('Prev Rew', prev_reward_2),
                                    ('Prev Ch', prev_choice_2),
                                    ('ones', ones_2)])
        
        X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
        
        n_predictors = X_2.shape[1]
        y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats,x = regression_code_session(y_2, X_2)
        C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_2.append(re._CPD(X_2,y_2).reshape(n_neurons, n_timepoints, n_predictors))
  
    
        choices_3 = choices_current[task_3]
        rewards_3 = reward_current[task_3]
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
        value_3= value[task_3]
        value_3_choice_3 = value_3*choices_3

        rew_ch_3 = choices_3*rewards_3
        prev_choice_3_lr = prev_choice_3*prev_reward_3

        firing_rates_3 = firing_rates[task_3]
        
        a_3 = np.where(choices_3 == 0.5)[0]
        b_3 = np.where(choices_3 == -0.5)[0]
        
        if plot_a == True:
            rewards_3 = rewards_3[a_3] 
            choices_3 = choices_3[a_3]
            rew_ch_3 = rew_ch_3[a_3]
            value_3 = value_3[a_3]
            value_3_choice_3 = value_3_choice_3[a_3]
            ones_3  = ones_3[a_3]

            firing_rates_3 = firing_rates_3[a_3]
            reward_PE_3 = reward_PE_3[a_3]
            choice_PE_3 = choice_PE_3[a_3]
            choice_PE_3_reward_current = choice_PE_3_reward_current[a_3]
            choice_PE_3_reward_prev = choice_PE_3_reward_prev[a_3]
            prev_choice_3_lr = prev_choice_3_lr[a_3]
            prev_reward_3 = prev_reward_3[a_3]
            prev_choice_3 = prev_choice_3[a_3]
           
           
        elif plot_b == True:
            rewards_3 = rewards_3[b_3] 
            choices_3 = choices_3[b_3]
            rew_ch_3 = rew_ch_3[b_3]
            value_3 = value_3[b_3]
            value_3_choice_3 = value_3_choice_3[b_3]
            ones_3  = ones_3[b_3]

            firing_rates_3 = firing_rates_3[b_3]
            reward_PE_3 = reward_PE_3[b_3]
            choice_PE_3 = choice_PE_3[b_3]
            choice_PE_3_reward_current = choice_PE_3_reward_current[b_3]
            choice_PE_3_reward_prev = choice_PE_3_reward_prev[b_3]
            prev_choice_3_lr = prev_choice_3_lr[b_3]
            prev_reward_3 = prev_reward_3[b_3]
            prev_choice_3 = prev_choice_3[b_3]
           
  
        predictors_all = OrderedDict([
                                    # ('Ch', choices_3),
                                    ('Rew', rewards_3),
                                    ('Value',value_3), 
                                    # ('Value Сhoice',value_3_choice_3), 
#
                                    ('Prev Rew', prev_reward_3),
                                    ('Prev Ch', prev_choice_3),
                                    ('ones', ones_3)])
        
        X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
        rank = np.linalg.matrix_rank(X_1)
        print(rank)
        n_predictors = X_3.shape[1]
        print(n_predictors)
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats,x = regression_code_session(y_3, X_3)
        C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_3.append(re._CPD(X_3,y_3).reshape(n_neurons, n_timepoints, n_predictors))
        
    
    C_1 = np.concatenate(C_1,1)
    
    C_2 = np.concatenate(C_2,1)
    
    C_3 = np.concatenate(C_3,1)
   

    
    # cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    # cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)
    # cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    # cpd = np.mean([cpd_1,cpd_2,cpd_3],0)

    # c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
        
    # j = 0
    # plt.figure()
    # pred = list(predictors_all.keys())
    # pred = pred[:-1]
    # for ii,i in enumerate(cpd.T[:-1]):
    #     plt.plot(i, color = c[j],label = pred[j])
      
    #     j+=1
    # plt.legend()
    # plt.title(area)

    # sns.despine()

  
    return C_1,C_2,C_3
    
     
      
  
    
    
          

    