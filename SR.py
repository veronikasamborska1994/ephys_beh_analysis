#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:38:34 2020

@author: veronikasamborska
"""
from collections import OrderedDict
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import sys 
from matplotlib.cbook import flatten
#import utility as ut
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
import regression_function as reg_f

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)


def run():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
  

def SR_reg(data):
    
    C = []

    dm = data['DM'][0]
    firing = data['Data'][0]
        
    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        reward = DM[:,2]    
        ones = np.ones(len(reward))

        predictors_all = OrderedDict([('Reward', reward),
                                      ('Choice', choices),
                                      ('ones', ones)])
            
               
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]

        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
    C = np.concatenate(C,1)
    
    return C

def SR_test(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    a_ind_rew = []
    for  s, sess in enumerate(dm):
        
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        reward = DM[:,2]    
        rew_a = firing_rates[np.where((reward == 1) & (choices == 1))[0]]
        
        for neuron in range(n_neurons):
            a_ind_rew.append(rew_a[:,neuron,:])
    i_max = []
    for  i in a_ind_rew:
        i_max.append(np.argmax(i,1))
    subplot = 0
    corr = []
    C = SR_reg(data)
    choice_selective = np.where(np.mean(C[1,:, 36:40],1) > 3)[0]
    for i,max_i in enumerate(i_max):
        if i in choice_selective:
            #plt.subplot(4,5,subplot)
            #sns.regplot(np.arange(len(max_i)),max_i, color = 'black', fit_reg = 'True')
            corr.append(np.corrcoef(np.arange(len(max_i)),max_i)[0,1])
            if np.corrcoef(np.arange(len(max_i)),max_i)[0,1] < -0.2:
                plt.figure()
                sns.regplot(np.arange(len(max_i)),max_i, color = 'black', fit_reg = 'True')
                sns.despine()


                
        
    i_max_shuffle = []
    for  i in a_ind_rew:
        i_max_shuffle_run = []
        for ii in range(5):
            np.random.shuffle(i) # Shuffle axis 0 only (trials)
            i_max_shuffle_run.append(np.argmax(i,1))
        i_max_shuffle.append(np.mean(i_max_shuffle_run,0))
  
    corr_shuffle = []
    for i,max_i in enumerate(i_max_shuffle):
        if i in choice_selective:
        # subplot += 1 
        # if subplot == 20:
        #     plt.figure()
        #     subplot -= 19
            
        # plt.subplot(4,5,subplot)
        # sns.regplot(np.arange(len(i)),i, color = 'black', fit_reg = 'True')
            corr_shuffle.append(np.corrcoef(np.arange(len(max_i)),max_i)[0,1])
            
            
    plt.hist(corr, color = 'grey', alpha = 0.5)
    plt.hist(corr_shuffle, color = 'pink', alpha = 0.5)
    plt.vlines(np.mean(corr),0,20)
    plt.vlines(np.mean(corr_shuffle),0,20, linestyle = '--')
    
    #     plt.imshow(i, aspect = 'auto')
 # subplot = 0
    # for i in a_ind_rew:
    #     subplot += 1 
    #     if subplot == 20:
    #         plt.figure()
    #         subplot -= 19
            
    #     plt.subplot(4,5,subplot)
        
    #     plt.imshow(i, aspect = 'auto')
