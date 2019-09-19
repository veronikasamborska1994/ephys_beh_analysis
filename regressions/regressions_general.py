#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:22:26 2019

@author: veronikasamborska
"""
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

import remapping_count as rc 
import numpy as np
import matplotlib.pyplot as plt
import regression_function as reg_f
import regressions as re
from collections import OrderedDict
from matplotlib import colors as mcolors
import utility as ut 

#predictors_all = OrderedDict([
#                          ('latent_state',state),
#                          ('choice',choices_forced_unforced ),
#                          ('reward', outcomes),
#                          ('forced_trials',forced_trials),
#                          ('task',task),
#                          ('A', a_pokes),
#                          ('B', b_pokes),
#                          ('Initiation', i_pokes),
#                          ('Chosen_Simple_RW',chosen_Q1),
#                          ('Chosen_Cross_learning_RW', chosen_Q4),
#                          ('Value_A_RW', Q1_value_a),
#                          ('Value_B_RW', Q1_value_b),
#                          ('Value_A_Cross_learning', Q4_value_a),
#                          ('ones', ones)])
            

def regression_general(data):
    C_1 = []
    cpd_1 = []
    
    
    dm = data['DM']
    dm = dm[:-1]
    firing = data['Data']
    firing = firing[:-1]
    
    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
        forced = DM[:,3]
        b_pokes = DM[:,6]
        a_pokes = DM[:,5]
        task = DM[:,4]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        taskid_1 = np.where(taskid==1)[0]
        choices_1 = np.zeros((len(choices)))
        reward_1 =  np.zeros((len(choices)))
        reward_exp = ut.exp_mov_ave(reward, tau = 15, initValue = 0.5)

        trials_since_block = []
        t = 0
        for st,s in enumerate(state):
            if state[st-1] != state[st]:
                t = 0
            else:
                t+=1
            trials_since_block.append(t)
        #task_1_a = np.where((taskid == 1) & (choices == 1))[0] # Find indicies for task 1 A
        #task_1_r = np.where((taskid == 1) & (reward == 1))[0] # Find indicies for task 1 A

        #choices_1[task_1_a] = 1
        #reward_1[task_1_r] = 1
        #choices_1 = choices_1[:len(taskid_1)]
        #reward_1 = reward_1[:len(taskid_1)]
        cum_reward = []
        c_r = 0
        for r in reward:
            if r == 1:
                c_r += 1
            elif r == 0:
                 c_r +=1
            cum_reward.append(c_r)
       
        choices_int = np.ones(len(reward))

        choices_int[np.where(choices == 0)] = -1
        reward_choice_int = choices_int*reward
        reward_exp_int = reward_exp*choices_int
        ones = np.ones(len(reward))
        #firing_rates = firing_rates[:len(reward_1),:,:]
       
        # Task 1 
        predictors = OrderedDict([('Reward', reward),
                                  ('Choice', choices),
                                 # ('Reward Rate', reward_exp),
                                  ('Trials_since_block', trials_since_block),
                                  #('Reward Rate x Choice', reward_exp_int),
                                  ('Reward x Choice', reward_choice_int),
                                  ('ones', ones)])
        
           
        X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
    
        C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
       
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    C_1 = np.concatenate(C_1,1)
#    b = C_1[0,2,:]
    
    return cpd_1, predictors 


def plot_cpd_gen(data,fig_n,title):
    
    cpd_1, predictors = regression_general(data)
    cpd_1 = cpd_1[:,:-1]
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    c = [*colors]
    c =  ['violet', 'black', 'red','chocolate', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink']
    p = [*predictors]
    plt.figure(fig_n)
    for i in np.arange(cpd_1.shape[1]):
        plt.plot(cpd_1[:,i], label =p[i], color = c[i])
        plt.title(title)
   # plt.vlines(32,ymin = 0, ymax = 0.15,linestyles= '--', color = 'grey', label = 'Poke')
    plt.legend()
    plt.ylabel('Coefficient of Partial Determination')
    plt.xlabel('Time (ms)')
    
    
    
    
    
    