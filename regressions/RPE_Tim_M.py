#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:15:29 2020

@author: veronikasamborska
"""
# =============================================================================
# Code to check if there are value and RPE signals in the data
# =============================================================================

import seaborn as sns
import numpy as np
from collections import OrderedDict
import regression_function as reg_f
import regressions as re
from palettable import wesanderson as wes
import matplotlib.pyplot as plt


def run(data_HP,data_PFC):
    
    plot_RPEs(data_HP, title = 'HP', perm = 1000)
    plot_RPEs(data_PFC, title = 'PFC', perm = 1000)


def regression_RPE(data, perm = True):
    
    C = []
    cpd = []

    dm = data['DM']
    firing = data['Data']
    
    if perm:
        C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
    
    
    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        reward = DM[:,2]    
       # state = DM[:,0]
        rpe = DM[:,14]
        q1 = DM[:,9]
        q4 = DM[:,10]
        rand = np.random.normal(np.mean(q4), np.std(q4), len(q4))
        ones = np.ones(len(rpe))
        trials = len(ones)
        
        predictors_all = OrderedDict([('Reward', reward),
                                      ('Choice', choices),
                                      #('State', state),
                                      ('RPE', rpe),  
                                      #('Q4', q4),
                                      #('Q1', q1),
                                      #('Noise', rand),
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


def plot_RPEs(data, title = 'HP',perm = 2):
    
    cpd_perm, p, cpd, C, predictors = regression_RPE(data, perm = perm)


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
        #t05 = t[p_vals == 0.05]
        t00 = t[p_vals == 0.001]
        #plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color=c[i])
        plt.plot(t00, np.ones(t00.shape)*y, '.', markersize=9, color=c[i])     
        
       

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    plt.title(title)
    sns.despine()
   
    
   