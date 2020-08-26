#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:26:19 2020

@author: veronikasamborska
"""


import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import sys 
from matplotlib.cbook import flatten
#import utility as ut
import seaborn as sns

def run():
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')


def runs_recordings(data):
    dm = data['DM'][0]
    firing = data['Data'][0]
    run_length_list_task = []
          
    run_length_list_task_subj = []
    run_length_list_correct_task_subj = []
    run_length_list_incorrect_task_subj = []
    
    
   
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        runs_list_corr = []
        runs_list_incorr = []
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        state = DM[:,0]
        
   

        
        correct = np.where(state == choices)[0]
        incorrect = np.where(state != choices)[0]
       
        block = DM[:,4]
        state_ch = np.where(np.diff(block)!=0)[0]+1

       
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
            
        inc = []
        co =[]
        for st in state_ch:
            inc_ind = [i for i in run_ind_inc if i > st]
            if len(inc_ind) > 0:
                inc.append(min(inc_ind))
            co_ind = [i for i in run_ind_c if i > st]
            if len(co_ind) > 0:
                co.append(min(co_ind))
                
        if len(np.asarray(co)) == len(np.asarray(inc)):
            index_which_incorr_ignore = np.asarray(co) > np.asarray(inc)    
        elif len(np.asarray(co)) > len(np.asarray(inc)):
            index_which_incorr_ignore = np.asarray(co)[:len(np.asarray(inc)  )] > np.asarray(inc)    
          
        if len(inc)> len(index_which_incorr_ignore):
            inc = inc[:len(index_which_incorr_ignore)]
        elif len(index_which_incorr_ignore)> len(inc):
            index_which_incorr_ignore = index_which_incorr_ignore[:len(inc)]
   
        starts_to_ignore = np.asarray(inc)[index_which_incorr_ignore]
        all_ends = np.where(np.diff(runs_list_incorr) < 0)[0]
        ends_to_ignore =[]
        runs_list_incorr = np.asarray(runs_list_incorr)
        for st in starts_to_ignore:
            ends = [i for i in all_ends if i > st]
            if len(ends)>0:
                ends_to_ignore.append(min(ends))
         
        for i, ii in enumerate(starts_to_ignore):
            if len(starts_to_ignore) == len(ends_to_ignore):
                runs_list_incorr[starts_to_ignore[i]: ends_to_ignore[i]] = 0
            else:
                runs_list_incorr[starts_to_ignore[i]:] = 0
               
                
        run_length_list_task_subj.append(runs_list)
        run_length_list_correct_task_subj.append(runs_list_corr)
        run_length_list_incorrect_task_subj.append(runs_list_incorr)
    
    plt.figure(figsize = (10,5))
  
    plt.subplot(1,3,1)
    all_hist = np.asarray(list(flatten(run_length_list_task_subj)))
    all_hist = all_hist[np.nonzero(all_hist)]
    plt.hist(all_hist,10, color = 'grey', label = 'all')
    #plt.xlim(0,50)

    plt.subplot(1,3,2)
    inc_hist = np.asarray(list(flatten(run_length_list_incorrect_task_subj)))
    incorr_all_hist = inc_hist[np.nonzero(inc_hist)]
    plt.hist(incorr_all_hist,10, color = 'green', label = 'incorrect')

    plt.subplot(1,3,3)
    c_hist = np.asarray(list(flatten(run_length_list_correct_task_subj)))
    corr_all_hist = c_hist[np.nonzero(c_hist)]

    plt.hist(corr_all_hist,10,  color = 'black', label = 'correct')
    #plt.xlim(0,50)
    #plt.tight_layout()
    plt.legend()
    sns.despine()
