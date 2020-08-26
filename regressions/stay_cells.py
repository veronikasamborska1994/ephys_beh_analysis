#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:20:03 2020

@author: veronikasamborska
"""


import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import regression_function as reg_f
import regressions as re
from collections import OrderedDict
import palettable.wesanderson as wes
from tqdm import tqdm
from scipy import io

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)

def run():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')

def regression_prev_choice(data, perm = True):
    
    C = []
    cpd = []

    dm = data['DM'][0]
    firing = data['Data'][0]
    
    if perm:
        C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
    
    
    for  s, sess in tqdm(enumerate(dm)):
        DM = dm[s]
        firing_rates = firing[s][1:,:,:]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        reward = DM[:,2]    
        task = DM[:,5]
        task_id = np.where(np.diff(task))[0]
        state = DM[:,0]

        stay=choices[0:-1]==choices[1:]
        stay = stay*1
        stay = np.insert(stay,0,0)
        lastreward = reward[0:-1]
        lastreward = np.insert(lastreward,0,0)


        rl = np.zeros(len(stay))
        rl[0]=1
      
        rl_right = np.zeros(len(stay))
        rl_right[0]=choices[0]==state[0]
        choice_rr_start=-100
         
         
        rl_wrong=np.zeros(len(stay));
        rl_wrong[0]=choices[0]!=state[0];
        choice_rw_start=-100;
        
        for tr in range(len(stay)):
            if tr > 0: 
                if stay[tr] == 1:
                    rl[tr] = rl[tr-1]+1
                else:
                    rl[tr]=1
                
                
                if ((choices[tr] == choice_rr_start) & (choices[tr]==state[tr])):
                    rl_right[tr]=rl_right[tr-1]+1
                    
                elif (choices[tr]==state[tr]):
                    
                    rl_right[tr]=1;
                    choice_rr_start=choices[tr]
                else:
                    rl_right[tr]=0;
                    choice_rr_start =-100; #If he made the wrong choice it can't be part of a correct run. 
                
                
                if ((choices[tr]==choice_rw_start) & (choices[tr]!=state[tr])):
                    rl_wrong[tr]=rl_wrong[tr-1]+1
                    
                elif choices[tr]!=state[tr]:
                    rl_wrong[tr]=1
                    choice_rw_start=choices[tr]
                else:
                    rl_wrong[tr] = 0;
                    choice_rw_start=-100 #If he made the right choice it can't be part of a wrong run. 
      
        trials = len(reward)
        rl_wrong = rl_wrong[1:]
        rl_right = rl_right[1:]
        rl = rl[1:]
        prev_choice = DM[:-1,1]
        choices = choices[1:]
        reward = reward[1:]   
        task = task[1:]
        state = state[1:]
        ones = np.ones(len(reward))
        int_repeat = choices*rl
        int_repeat_corr = state*rl_right
        int_repeat_incorr = state*rl_wrong
        error_count = []
        err_count = 0
        for r,rew in enumerate(reward): 
            if  rew == 0:
                if reward[r] == reward[r-1]:
                    err_count +=1
            else:
                err_count = 0 
            error_count.append(err_count)

                    

        predictors_all = OrderedDict([('Reward', reward),
                                      ('Choice', choices),
                                      ('State', state),
                                      ('Previous Choice 1', prev_choice),
                                      ('Repeat', rl),
                                      #('Error Count',error_count),
                                      #('Repeat Incorrect', rl_wrong),
                                      #('Repeat Correct', rl_right),
                                     # ('Repeat Int', int_repeat),
                                      #('Repeat Corr Int', int_repeat_corr),
                                      #('Repeat Incorr Int', int_repeat_incorr),
                                      ('ones', ones)])
            
               
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        
        C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
        
        if perm:
           for i in range(perm):
               X_perm= np.roll(X,np.random.randint(trials), axis = 0)
               tstats = reg_f.regression_code(y, X_perm)
        
               C_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
               cpd_perm[i].append(re._CPD(X_perm,y).reshape(n_neurons, n_timepoints, n_predictors))
    
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        p = np.percentile(cpd_perm,95, axis = 0)
           
     
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    return cpd_perm, p, cpd, C, predictors_all 

def regression_past_choice(data, perm = True):
    
    C = []
    cpd = []

    dm = data['DM'][0]
    firing = data['Data'][0]
    
    if perm:
        C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
    
    
    for  s, sess in tqdm(enumerate(dm)):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        reward = DM[:,2]    
        block = DM[:,4]
        block_df = np.diff(block)
        task = DM[:,5]
        task_id = np.where(np.diff(task))[0]

        ones = np.ones(len(block))
        trials = len(ones)
        stay = []
        for c,ch in enumerate(choices):
            if c > 0:
                if choices[c-1] == choices[c]:
                    stay.append(1)
                else:
                    stay.append(0)
            else:
                stay.append(0)

        a_side = 0
        a_side_l = []
        for r,rew in enumerate(reward):
            if r in task_id:
                a_side = 0
            elif reward[r] == 1 and  choices[r] == 1:
                a_side += 1
            a_side_l.append(a_side)
         
        b_side = 0
        b_side_l = []
        for r,rew in enumerate(reward):
            if r in task_id:
                b_side = 0
            elif reward[r] == 1 and  choices[r] == 0:
                b_side += 1
            b_side_l.append(b_side)
     
        stay_ch = np.asarray(stay)*choices
        predictors_all = OrderedDict([('Reward', reward),
                                      ('Choice', choices),
                                      ('Stay', stay),
                                      ('Stay x Choice',stay_ch),
                                      ('Reward Cum A', a_side_l),
                                      ('Reward Cum B', b_side_l),
                                      ('ones', ones)])
            
               
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        
        C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
        
        if perm:
           for i in range(perm):
               X_perm= np.roll(X,np.random.randint(trials), axis = 0)
               tstats = reg_f.regression_code(y, X_perm)
        
               C_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
               cpd_perm[i].append(re._CPD(X_perm,y).reshape(n_neurons, n_timepoints, n_predictors))
    
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        p = np.percentile(cpd_perm,95, axis = 0)
           
     
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    return cpd_perm, p, cpd, C, predictors_all 


def run_perms():
    
    plot_stay(HP, title = 'HP',perm = 1000)
    plot_stay(PFC, title = 'PFC',perm = 1000)


def plot_stay(data, title = 'HP',perm = 2):
    
    cpd_perm, p, cpd, C, predictors = regression_prev_choice(data, perm = perm)

    cells = np.where(abs(np.mean(C[3,:,:35],1)) > 3)[0]
    # cells_rew = np.where(abs(np.mean(C[0,:,42:],1)) > 3)[0]

    # cells_stay_rew = [cells_st for cells_st in cells_stay if cells_st in cells_rew]
    # cells_stay_rew_not = [cells_st for cells_st in cells_stay if cells_st not in cells_rew]



    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors


    # HP
    plt.figure()
    t = np.arange(0,63)
    cpd = cpd[:,:-1]
    cpd_perm = cpd_perm[:,:-1]
    p = [*predictors][:-1]
    values_95 = np.max(np.percentile(cpd_perm,95,axis = 0),axis=0)
    values_99 = np.max(np.percentile(cpd_perm,99,axis = 0),axis=0)
    array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
   
    for i in range(cpd.shape[1]):
        array_pvals[(np.where(cpd[:,i] > values_95[i])[0]),i] = 0.05
        array_pvals[(np.where(cpd[:,i] > values_99[i])[0]),i] = 0.001
 
    ymax = np.max(cpd)
    
    for i in np.arange(cpd.shape[1]):
      #  perm_plot = np.max(np.percentile(cpd_perm[:,:,i],95,axis = 0),axis=0)
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        y = ymax*(1+0.04*i)
        p_vals = array_pvals[:,i]
        t05 = t[p_vals == 0.05]
        t00 = t[p_vals == 0.001]
        plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color=c[i])
        plt.plot(t00, np.ones(t00.shape)*y, '.', markersize=9, color=c[i])     
        
       

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    plt.title(title)
    sns.despine()

def plot_cells(data,cells):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    all_neurons = 0
    cells_len = len(cells)
    i =0
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors

    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        #choices = DM[:,1]
        reward = DM[:,2]  
        n_rew = np.where(reward == 0 )[0]
        rew_ind = np.where(reward == 1 )[0]

        for n in range(n_neurons):
            all_neurons +=1
            if all_neurons in cells:
                i+=1
                r = np.mean(firing_rates[rew_ind,n,:],0)
                n_r = np.mean(firing_rates[n_rew,n,:],0)
                plt.subplot(5,int(cells_len/5)+1,i) 
                plt.plot(r, color=c[5], label = 'Rew')     
                plt.plot(n_r, color=c[5], label = 'No Rew', linestyle = '--')     
    plt.legend()
    sns.despine()
    
    
    
    
   
    
   