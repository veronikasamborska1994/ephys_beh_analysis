#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:39:22 2019

@author: veronikasamborska
"""

## Warping code for interpolating firing rates on neighbouring trials


import numpy as np
import pylab as plt
from scipy.special import erf
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
import utility as ut 
from collections import OrderedDict
from palettable import wesanderson as wes


def raw_data_time_warp(data, experiment_aligned_data):
        
    dm = data['DM'][0]
    firing = data['Data'][0]

    res_list = []
    list_block_changes = []
    trials_since_block_list = []
    state_list = []
    task_list = []
    for  s, sess in enumerate(dm):
        
        
        DM = dm[s]
        state = DM[:,0]
        task = DM[:,5]

        firing_rates = firing[s] 
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df ==1)[0]
      
        if len(ind_block) >= 12:
         
           trials_since_block = []
           t = 0
                
             #Bug in the state? 
           for st,s in enumerate(block):
                if block[st-1] != block[st]:
                    t = 0
                else:
                     t+=1
                trials_since_block.append(t)
           ind_12_blocks = ind_block[11]
           ind_blocks = np.median(np.append(np.diff(ind_block),ind_block[0]))
           firing_rates  = firing_rates[:ind_12_blocks]
           trials_since_block = np.asarray(trials_since_block)[:ind_12_blocks]
           state = state[:ind_12_blocks]
    
           task = task[:ind_12_blocks]
           
           res_list.append(firing_rates)
           list_block_changes.append(ind_blocks)
           trials_since_block_list.append(trials_since_block)
           state_list.append(state)
           task_list.append(task)
           
           
    return res_list, list_block_changes, trials_since_block_list, state_list,task_list

def raw_data_time_warp_beh(data, experiment_aligned_data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]

    res_list = []
    list_block_changes = []
    trials_since_block_list = []
    state_list = []
    task_list = []
   
    for  s, sess in enumerate(dm):
        
        
        DM = dm[s]
        state = DM[:,0]
        choices = DM[:,1]
     
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
            
           
    
            res_list.append(firing_rates)
            trials_since_block_list.append(trials_since_block)
            list_block_changes.append(ind_blocks)
            state_list.append(state)
            task_list.append(task)
           
    return res_list, list_block_changes, trials_since_block_list, state_list,task_list


def raw_data_align(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC):
    res_list_HP, list_block_changes_HP, trials_since_block_list_HP, state_list_HP,task_list_HP = raw_data_time_warp_beh(data_HP,experiment_aligned_HP)
    res_list_PFC, list_block_changes_PFC, trials_since_block_list_PFC, state_list_PFC,task_list_PFC = raw_data_time_warp_beh(data_PFC,experiment_aligned_PFC)
    target_trials = np.median([np.median(list_block_changes_HP),np.median(list_block_changes_PFC)])
    
    
    return res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,trials_since_block_list_PFC, state_list_HP,state_list_PFC,task_list_HP, task_list_PFC


def all_sessions_align_beh_raw_data(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end):
    res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,\
        trials_since_block_list_PFC, state_list_HP,state_list_PFC, task_list_HP, task_list_PFC = raw_data_align(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC)
    
    HP_aligned_time  = []
    PFC_aligned_time  = []
    smooth_SD = 1
    edge  = 2
    # HP
    ends_sessions = []
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
        ends_sessions.append(np.diff(trial_times).flatten())

        aligned_activity = align_activity(activity, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)


        HP_aligned_time.append(aligned_activity)
        
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
        
        ends_sessions.append(np.diff(trial_times).flatten())

        aligned_activity = align_activity(activity, frame_times, trial_times, target_times,  smooth_SD = smooth_SD, plot_warp = False, edge = edge)
        PFC_aligned_time.append(aligned_activity)
        
        
    return HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC
            
def regression_residuals_blocks_aligned_on_block(data,experiment_aligned_data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]

    res_list = []
    list_block_changes = []
    trials_since_block_list = []
    state_list = []
    task_list = []
    for  s, sess in enumerate(dm):
        
        
        DM = dm[s]
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
        task = DM[:,5]

        firing_rates = firing[s] 
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df ==1)[0]
      
        if len(ind_block) >= 12:
         
           trials_since_block = []
           t = 0
                
             #Bug in the state? 
           for st,s in enumerate(block):
                if block[st-1] != block[st]:
                    t = 0
                else:
                     t+=1
                trials_since_block.append(t)
           ind_12_blocks = ind_block[11]
           ind_blocks = np.median(np.append(np.diff(ind_block),ind_block[0]))
           firing_rates  = firing_rates[:ind_12_blocks]
           trials_since_block = np.asarray(trials_since_block)[:ind_12_blocks]
           state = state[:ind_12_blocks]
           choices = choices[:ind_12_blocks]            
           reward = reward[:ind_12_blocks]
           task = task[:ind_12_blocks]
           ones = np.ones(len(state))
             
           task_1 = np.where(task == 1)[0]
           task_2 = np.where(task == 2)[0]
           task_3 = np.where(task == 3)[0]

           
           ## Task 1 
           firing_rates_1  = firing_rates[task_1]
           choices_1 = choices[task_1]        
           reward_1 = reward[task_1]        
           reward_choice_1= choices_1*reward_1
           task_1 = task[task_1]
           ones_1 = np.ones(len(reward_choice_1))
           
           n_trials, n_neurons, n_timepoints = firing_rates_1.shape
           predictors_all = OrderedDict([#('Time', trials_since_block),
                                             ('Reward', reward_1),
                                             ('Choice', choices_1),
                                             ('Reward x Choice',reward_choice_1),
                                             ('ones', ones_1)])
           
                     
           X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
           y = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
             
           pdes = np.linalg.pinv(X)
           pe = np.matmul(pdes,y)
           res = y - np.matmul(X,pe)
           res_list_task_1 = res.reshape(n_trials, n_neurons, n_timepoints)
           
           
           ## Task 2 
           firing_rates_2  = firing_rates[task_2]
           choices_2 = choices[task_2]        
           reward_2 = reward[task_2]        
           reward_choice_2 = choices_2*reward_2  
           task_2 = task[task_2]
           ones_2 = np.ones(len(reward_choice_2))
           
           n_trials, n_neurons, n_timepoints = firing_rates_2.shape
           predictors_all = OrderedDict([#('Time', trials_since_block),
                                             ('Reward', reward_2),
                                             ('Choice', choices_2),
                                             ('Reward x Choice',reward_choice_2),
                                             ('ones', ones_2)])
           
                     
           X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
           y = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
             
           pdes = np.linalg.pinv(X)
           pe = np.matmul(pdes,y)
           res = y - np.matmul(X,pe)
           res_list_task_2 = res.reshape(n_trials, n_neurons, n_timepoints)
           
           ## Task 3 
           firing_rates_3  = firing_rates[task_3]
           choices_3 = choices[task_3]        
           reward_3 = reward[task_3]        
           reward_choice_3 = choices_3*reward_3    
           task_3 = task[task_3]
           ones_3 = np.ones(len(reward_choice_3))
           
           n_trials, n_neurons, n_timepoints = firing_rates_3.shape
           predictors_all = OrderedDict([#('Time', trials_since_block),
                                             ('Reward', reward_3),
                                             ('Choice', choices_3),
                                             ('Reward x Choice',reward_choice_3),
                                             ('ones', ones_3)])
           
                     
           X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
           y = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
             
           pdes = np.linalg.pinv(X)
           pe = np.matmul(pdes,y)
           res = y - np.matmul(X,pe)
           res_list_task_3 = res.reshape(n_trials, n_neurons, n_timepoints)
    
    
    
           res_list.append(np.concatenate((res_list_task_1,res_list_task_2,res_list_task_3),0))
           trials_since_block_list.append(trials_since_block)
           list_block_changes.append(ind_blocks)
           state_list.append(state)
           task_list.append(task)

    return res_list, list_block_changes, trials_since_block_list, state_list,task_list

def residuals_block_aligned(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC):
    
    res_list_HP, list_block_changes_HP, trials_since_block_list_HP, state_list_HP, task_list_HP = regression_residuals_blocks_aligned_on_block(data_HP,experiment_aligned_HP)
    res_list_PFC, list_block_changes_PFC, trials_since_block_list_PFC, state_list_PFC,task_list_PFC = regression_residuals_blocks_aligned_on_block(data_PFC,experiment_aligned_PFC)
    target_trials = np.median([np.median(list_block_changes_HP),np.median(list_block_changes_PFC)])
    
    
    return res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,trials_since_block_list_PFC, state_list_HP,state_list_PFC,task_list_HP, task_list_PFC

def all_sessions_aligned_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, start,end):
    res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,\
        trials_since_block_list_PFC, state_list_HP,state_list_PFC, task_list_HP, task_list_PFC = residuals_block_aligned(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC)
    
    HP_aligned_time  = []
    PFC_aligned_time  = []
    smooth_SD = 0.1
    edge = 2

    #HP
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
        aligned_activity = align_activity(activity, frame_times, trial_times, target_times, smooth_SD = smooth_SD, plot_warp=False, edge = edge)
        HP_aligned_time.append(aligned_activity)
        
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
        
        
        aligned_activity = align_activity(activity, frame_times, trial_times, target_times, smooth_SD = smooth_SD, plot_warp=False, edge = edge)
        PFC_aligned_time.append(aligned_activity)
        
    return HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC


def regression_residuals_blocks_aligned_on_beh(data,experiment_aligned_data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]

    res_list = []
    list_block_changes = []
    trials_since_block_list = []
    state_list = []
    task_list = []
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
            choices = choices[all_ind_triggered_on_beh]            
            reward = reward[all_ind_triggered_on_beh]
            task = task[all_ind_triggered_on_beh]
            ones = np.ones(len(state))
            task_1 = np.where(task == 1)[0]
            task_2 = np.where(task == 2)[0]
            task_3 = np.where(task == 3)[0]

           
           ## Task 1 
            firing_rates_1  = firing_rates[task_1]
            choices_1 = choices[task_1]        
            reward_1 = reward[task_1]        
            reward_choice_1= choices_1*reward_1
            task_1 = task[task_1]
            ones_1 = np.ones(len(reward_choice_1))
           
            n_trials, n_neurons, n_timepoints = firing_rates_1.shape
            predictors_all = OrderedDict([#('Time', trials_since_block),
                                             ('Reward', reward_1),
                                             ('Choice', choices_1),
                                            # ('Reward x Choice',reward_choice_1),
                                             ('ones', ones_1)])
           
                     
            X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
            y = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
              
            pdes = np.linalg.pinv(X)
            pe = np.matmul(pdes,y)
            res = y - np.matmul(X,pe)
            res_list_task_1 = res.reshape(n_trials, n_neurons, n_timepoints)
           
           
            ## Task 2 
            firing_rates_2  = firing_rates[task_2]
            choices_2 = choices[task_2]        
            reward_2 = reward[task_2]        
            reward_choice_2 = choices_2*reward_2  
            task_2 = task[task_2]
            ones_2 = np.ones(len(reward_choice_2))
            
            n_trials, n_neurons, n_timepoints = firing_rates_2.shape
            predictors_all = OrderedDict([#('Time', trials_since_block),
                                             ('Reward', reward_2),
                                             ('Choice', choices_2),
                                           #  ('Reward x Choice',reward_choice_2),
                                             ('ones', ones_2)])
           
                     
            X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
            y = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
             
            pdes = np.linalg.pinv(X)
            pe = np.matmul(pdes,y)
            res = y - np.matmul(X,pe)
            res_list_task_2 = res.reshape(n_trials, n_neurons, n_timepoints)
           
            ## Task 3 
            firing_rates_3  = firing_rates[task_3]
            choices_3 = choices[task_3]        
            reward_3 = reward[task_3]        
            reward_choice_3 = choices_3*reward_3    
            task_3 = task[task_3]
            ones_3 = np.ones(len(reward_choice_3))
           
            n_trials, n_neurons, n_timepoints = firing_rates_3.shape
            predictors_all = OrderedDict([#('Time', trials_since_block),
                                             ('Reward', reward_3),
                                             ('Choice', choices_3),
                                           #  ('Reward x Choice',reward_choice_3),
                                             ('ones', ones_3)])
           
                     
            X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
            y = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
             
            pdes = np.linalg.pinv(X)
            pe = np.matmul(pdes,y)
            res = y - np.matmul(X,pe)
            res_list_task_3 = res.reshape(n_trials, n_neurons, n_timepoints)
    
    
    
            res_list.append(np.concatenate((res_list_task_1,res_list_task_2,res_list_task_3),0))
            trials_since_block_list.append(trials_since_block)
            list_block_changes.append(ind_blocks)
            state_list.append(state)
            task_list.append(task)

    return res_list, list_block_changes, trials_since_block_list, state_list,task_list

def residuals_function(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC):
    res_list_HP, list_block_changes_HP, trials_since_block_list_HP, state_list_HP, task_list_HP = regression_residuals_blocks_aligned_on_beh(data_HP,experiment_aligned_HP)
    res_list_PFC, list_block_changes_PFC, trials_since_block_list_PFC, state_list_PFC,task_list_PFC = regression_residuals_blocks_aligned_on_beh(data_PFC,experiment_aligned_PFC)
    target_trials = np.median([np.median(list_block_changes_HP),np.median(list_block_changes_PFC)])
    
    
    return res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,trials_since_block_list_PFC, state_list_HP,state_list_PFC,task_list_HP, task_list_PFC


def all_sessions_align_beh(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end):
    res_list_HP,res_list_PFC,target_trials,trials_since_block_list_HP,\
        trials_since_block_list_PFC, state_list_HP,state_list_PFC, task_list_HP, task_list_PFC = residuals_function(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC)
    
    HP_aligned_time  = []
    PFC_aligned_time  = []
    smooth_SD = 0.1
    edge = 2
    
    #HP
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
        aligned_activity = align_activity(activity, frame_times, trial_times, target_times, smooth_SD = smooth_SD, plot_warp = False,edge = edge)
        HP_aligned_time.append(aligned_activity)
        
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
        
        
        aligned_activity = align_activity(activity, frame_times, trial_times, target_times, smooth_SD = smooth_SD, plot_warp = False,edge = edge)
        
        PFC_aligned_time.append(aligned_activity)
        
    return HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC


def align_activity(activity, frame_times, trial_times, target_times, smooth_SD = 1, plot_warp=False,  edge = 2):
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



