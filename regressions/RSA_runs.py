#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:42:26 2020

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


def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid
 

def RSA_make(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
  
  
    tr_runs = 3
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
  
    a_ind_rew_task_1 = np.zeros((neurons,63,tr_runs));  b_ind_rew_task_1 = np.zeros((neurons,63,tr_runs))
    a_ind_err_task_1 = np.zeros((neurons,63,tr_runs));  b_ind_err_task_1 = np.zeros((neurons,63,tr_runs))
   
    a_ind_rew_task_2 = np.zeros((neurons,63,tr_runs));  b_ind_rew_task_2 = np.zeros((neurons,63,tr_runs))
    a_ind_err_task_2 = np.zeros((neurons,63,tr_runs));  b_ind_err_task_2 = np.zeros((neurons,63,tr_runs))
   
    a_ind_rew_task_3 = np.zeros((neurons,63,tr_runs));  b_ind_rew_task_3 = np.zeros((neurons,63,tr_runs))
    a_ind_err_task_3 = np.zeros((neurons,63,tr_runs));  b_ind_err_task_3 = np.zeros((neurons,63,tr_runs))
    n_neurons_cum =0

    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons

        choices = DM[:,1]
        reward = DM[:,2]    

        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]
        
        taskid = task_ind(task, a_pokes, b_pokes)
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]

        cum_error = []
        
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
      
                
        choices_1 = choices[task_1]
        cum_error_1 = np.asarray(cum_error)[task_1]
        cum_reward_1 = np.asarray(cum_reward)[task_1]
       
        choices_2 = choices[task_2]
        cum_error_2 = np.asarray(cum_error)[task_2]
        cum_reward_2 = np.asarray(cum_reward)[task_2]
       
        choices_3 = choices[task_3]
        cum_error_3 = np.asarray(cum_error)[task_3]
        cum_reward_3 = np.asarray(cum_reward)[task_3]
      
        
        for i in range(tr_runs):
            
            

            a_ind_rew_task_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_reward_1) == i+1) & (choices_1 == 1)))[0]],0)
            b_ind_rew_task_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_reward_1) == i+1) & (choices_1 == 0)))[0]],0)

            a_ind_err_task_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_error_1) == i+1) & (choices_1 == 1)))[0]],0)
            b_ind_err_task_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_error_1) == i+1) & (choices_1 == 0)))[0]],0)
            
            a_ind_rew_task_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_error_2) == i+1) & (choices_2 == 1)))[0]],0)
            b_ind_rew_task_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_error_2) == i+1) & (choices_2 == 0)))[0]],0)
                  
            a_ind_err_task_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_reward_2) == i+1) & (choices_2 == 1)))[0]],0)
            b_ind_err_task_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_reward_2) == i+1) & (choices_2 == 0)))[0]],0)

            a_ind_rew_task_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_reward_3) == i+1) & (choices_3 == 1)))[0]],0)
            b_ind_rew_task_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_reward_3) == i+1) & (choices_3 == 0)))[0]],0)
          
            a_ind_err_task_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_error_3) == i+1) & (choices_3 == 1)))[0]],0)
            b_ind_err_task_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.nanmean(firing_rates[(np.where((np.asarray(cum_error_3) == i+1) & (choices_3 == 0)))[0]],0)
    
    all_tasks_reward_a = np.concatenate([a_ind_rew_task_1,a_ind_rew_task_2,a_ind_rew_task_3],2)
    all_tasks_error_a = np.concatenate([a_ind_err_task_1,a_ind_err_task_2,a_ind_err_task_3],2)
    
    all_tasks_reward_b = np.concatenate([b_ind_rew_task_1,b_ind_rew_task_2,b_ind_rew_task_3],2)
    all_tasks_error_b = np.concatenate([b_ind_err_task_1,b_ind_err_task_2,b_ind_err_task_3],2)

    return all_tasks_reward_a, all_tasks_error_a, all_tasks_reward_b, all_tasks_error_b
           

def plot_correlations():
    all_tasks_reward_a, all_tasks_error_a, all_tasks_reward_b, all_tasks_error_b = RSA_make(PFC)   
    all_tasks_reward_a = np.nanmean(all_tasks_reward_a[:,42:63,:],1)
 
    all_tasks_reward_a = all_tasks_reward_a[~np.isnan(all_tasks_reward_a).any(axis=1)]
    plt.imshow(np.corrcoef(all_tasks_reward_a.T)-0.96)
    plt.clim(-0.5, 0.5)

    plt.colorbar()
    