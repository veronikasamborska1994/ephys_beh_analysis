#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:57:44 2020

@author: veronikasamborska
"""

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import seaborn as sns
from palettable import wesanderson as wes

def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid


def plot_prev_trial(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    
    prev_a_rew_1 = np.zeros((2,neurons,63)); 
    prev_a_nrew_1 = np.zeros((2,neurons,63)); 

    prev_b_rew_1 = np.zeros((2,neurons,63)); 
    prev_b_nrew_1 = np.zeros((2,neurons,63)); 

    # b_prev_a_rew_1 = np.zeros((2,neurons,63)); 
    # b_prev_a_nrew_1 = np.zeros((2,neurons,63)); 

    # b_prev_b_rew_1 = np.zeros((2,neurons,63)); 
    # b_prev_b_nrew_1 = np.zeros((2,neurons,63)); 
    
    prev_a_rew_2 = np.zeros((2,neurons,63)); 
    prev_a_nrew_2 = np.zeros((2,neurons,63)); 

    prev_b_rew_2 = np.zeros((2,neurons,63)); 
    prev_b_nrew_2 = np.zeros((2,neurons,63)); 

    # b_prev_a_rew_2 = np.zeros((2,neurons,63)); 
    # b_prev_a_nrew_2 = np.zeros((2,neurons,63)); 

    # b_prev_b_rew_2 = np.zeros((2,neurons,63)); 
    # b_prev_b_nrew_2 = np.zeros((2,neurons,63));
    
    prev_a_rew_3 = np.zeros((2,neurons,63)); 
    prev_a_nrew_3 = np.zeros((2,neurons,63)); 

    prev_b_rew_3 = np.zeros((2,neurons,63)); 
    prev_b_nrew_3 = np.zeros((2,neurons,63)); 

    # b_prev_a_rew_3 = np.zeros((2,neurons,63)); 
    # b_prev_a_nrew_3 = np.zeros((2,neurons,63)); 

    # b_prev_b_rew_3 = np.zeros((2,neurons,63)); 
    # b_prev_b_nrew_3 = np.zeros((2,neurons,63));
   
    n_neurons_cum = 0
    
   
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]

        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons
        reward = DM[:,2]  
        choices = DM[:,1]
        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]
        
        taskid = task_ind(task, a_pokes, b_pokes)
      
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]

       
        choices_1 = choices[task_1]
        choices_2 = choices[task_2]
        choices_3 = choices[task_3]

        reward_1 = reward[task_1]
        reward_2 = reward[task_2]
        reward_3 = reward[task_3]
        firing_rates_1 = firing_rates[task_1]
        firing_rates_2 = firing_rates[task_2]
        firing_rates_3 = firing_rates[task_3]

        # Task 1 
        prev_a_rew_1_ind = []; prev_a_nrew_1_ind = []; prev_b_rew_1_ind = []; prev_b_nrew_1_ind = []
    
        #prev_a_rew_1_ind = []; b_prev_a_nrew_1_ind = []; prev_b_rew_1_ind = [];b_prev_b_nrew_1_ind = []
     
        prev_a_rew_2_ind = []; prev_a_nrew_2_ind = []; prev_b_rew_2_ind = []; prev_b_nrew_2_ind = []
        
       # b_prev_a_rew_2_ind = []; b_prev_a_nrew_2_ind = []; b_prev_b_rew_2_ind = [];b_prev_b_nrew_2_ind = []
     
        
        prev_a_rew_3_ind = []; prev_a_nrew_3_ind = []; prev_b_rew_3_ind = []; prev_b_nrew_3_ind = []
        
        #b_prev_a_rew_3_ind = []; b_prev_a_nrew_3_ind = []; b_prev_b_rew_3_ind = [];b_prev_b_nrew_3_ind = []
     
        for c, ch in enumerate(choices_1):
            #if ch == 1: # A
            if choices_1[c-1] == 1:
                if reward_1[c-1] ==1:
                    prev_a_rew_1_ind.append(c)
                elif reward_1[c-1] == 0:
                    prev_a_nrew_1_ind.append(c)

            if choices_1[c-1]  == 0:
                if reward_1[c-1] ==1: 
                    prev_b_rew_1_ind.append(c)
                elif reward_1[c-1] == 0:
                    prev_b_nrew_1_ind.append(c)
                        
           # # elif ch == 0: # B
           #  if choices_1[c-1] != choices_1[c]:
           #     if reward_1[c-1] ==1:
           #         b_prev_a_rew_1_ind.append(c)
           #      elif reward_1[c-1] == 0:
           #          b_prev_a_nrew_1_ind.append(c)

           #      if choices_1[c-1] == choices_1[c]:
           #          if reward_1[c-1] ==1: 
           #              b_prev_b_rew_1_ind.append(c)
           #          elif reward_1[c-1] == 0:
           #              b_prev_b_nrew_1_ind.append(c)

    
        for c, ch in enumerate(choices_2):
            # if ch == 1: # A
            if choices_2[c-1] == 1:
                if reward_2[c-1] ==1:
                    prev_a_rew_2_ind.append(c)
                elif reward_2[c-1] == 0:
                    prev_a_nrew_2_ind.append(c)

            if choices_2[c-1] == 0:
                if reward_2[c-1] ==1: 
                    prev_b_rew_2_ind.append(c)
                elif reward_2[c-1] == 0:
                    prev_b_nrew_2_ind.append(c)
                    
            # elif ch == 0: # B
            #     if choices_2[c-1] != choices_2[c]:
            #         if reward_2[c-1] ==1:
            #             b_prev_a_rew_2_ind.append(c)
            #         elif reward_2[c-1] == 0:
            #             b_prev_a_nrew_2_ind.append(c)

            #     if choices_2[c-1] == choices_2[c]:
            #         if reward_2[c-1] ==1: 
            #             b_prev_b_rew_2_ind.append(c)
            #         elif reward_2[c-1] == 0:
            #             b_prev_b_nrew_2_ind.append(c)

        
        for c, ch in enumerate(choices_3):
            # if ch == 1: # A
            if choices_3[c-1] == 1:
                if reward_3[c-1] ==1:
                    prev_a_rew_3_ind.append(c)
                elif reward_3[c-1] == 0:
                    prev_a_nrew_3_ind.append(c)

            if choices_3[c-1] == 0:
                if reward_3[c-1] ==1: 
                    prev_b_rew_3_ind.append(c)
                elif reward_3[c-1] == 0:
                    prev_b_nrew_3_ind.append(c)
                    
            # elif ch == 0: # B
            #     if choices_3[c-1] != choices_3[c]:
            #         if reward_3[c-1] ==1:
            #             b_prev_a_rew_3_ind.append(c)
            #         elif reward_3[c-1] == 0:
            #             b_prev_a_nrew_3_ind.append(c)

            #     if choices_3[c-1] == choices_3[c]:
            #         if reward_3[c-1] ==1: 
            #             b_prev_b_rew_3_ind.append(c)
            #         elif reward_3[c-1] == 0:
            #             b_prev_b_nrew_3_ind.append(c)

         ## Task 1
        #print(n_neurons_cum-n_neurons)
        prev_a_rew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[prev_a_rew_1_ind],0)
        prev_a_rew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[prev_a_rew_1_ind],0)/np.sqrt(len(prev_a_rew_1_ind))
    
        prev_b_rew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[prev_b_rew_1_ind],0)
        prev_b_rew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[prev_b_rew_1_ind],0)/np.sqrt(len(prev_b_rew_1_ind))
    
    
        # b_prev_a_rew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[b_prev_a_rew_1_ind],0)
        # b_prev_a_rew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[b_prev_a_rew_1_ind],0)/np.sqrt(len(b_prev_a_rew_1_ind))
    
        # b_prev_b_rew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[b_prev_b_rew_1_ind],0)
        # b_prev_b_rew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[b_prev_b_rew_1_ind],0)/np.sqrt(len(b_prev_b_rew_1_ind))
    
        # no reward
        prev_a_nrew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[prev_a_nrew_1_ind],0)
        prev_a_nrew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[prev_a_nrew_1_ind],0)/np.sqrt(len(prev_a_nrew_1_ind))
    
        prev_b_nrew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[prev_b_nrew_1_ind],0)
        prev_b_nrew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[prev_b_nrew_1_ind],0)/np.sqrt(len(prev_b_nrew_1_ind))
    
    
        # b_prev_a_nrew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[b_prev_a_nrew_1_ind],0)
        # b_prev_a_nrew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[b_prev_a_nrew_1_ind],0)/np.sqrt(len(b_prev_a_nrew_1_ind))
    
        # b_prev_b_nrew_1[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_1[b_prev_b_nrew_1_ind],0)
        # b_prev_b_nrew_1[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_1[b_prev_b_nrew_1_ind],0)/np.sqrt(len(b_prev_b_nrew_1_ind))
    
    
        ## Task 2
        prev_a_rew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[prev_a_rew_2_ind],0)
        prev_a_rew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[prev_a_rew_2_ind],0)/np.sqrt(len(prev_a_rew_2_ind))
    
        prev_b_rew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[prev_b_rew_2_ind],0)
        prev_b_rew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[prev_b_rew_2_ind],0)/np.sqrt(len(prev_b_rew_2_ind))
    
    
        # b_prev_a_rew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[b_prev_a_rew_2_ind],0)
        # b_prev_a_rew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[b_prev_a_rew_2_ind],0)/np.sqrt(len(b_prev_a_rew_2_ind))
    
        # b_prev_b_rew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[b_prev_b_rew_2_ind],0)
        # b_prev_b_rew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[b_prev_b_rew_2_ind],0)/np.sqrt(len(b_prev_b_rew_2_ind))
    
        # no reward
        prev_a_nrew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[prev_a_nrew_2_ind],0)
        prev_a_nrew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[prev_a_nrew_2_ind],0)/np.sqrt(len(prev_a_nrew_2_ind))
    
        prev_b_nrew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[prev_b_nrew_2_ind],0)
        prev_b_nrew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[prev_b_nrew_2_ind],0)/np.sqrt(len(prev_b_nrew_2_ind))
    
    
        # b_prev_a_nrew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[b_prev_a_nrew_2_ind],0)
        # b_prev_a_nrew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[b_prev_a_nrew_2_ind],0)/np.sqrt(len(b_prev_a_nrew_2_ind))
    
        # b_prev_b_nrew_2[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_2[b_prev_b_nrew_2_ind],0)
        # b_prev_b_nrew_2[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_2[b_prev_b_nrew_2_ind],0)/np.sqrt(len(b_prev_b_nrew_2_ind))
    
        
        ## Task 3
        prev_a_rew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[prev_a_rew_3_ind],0)
        prev_a_rew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[prev_a_rew_3_ind],0)/np.sqrt(len(prev_a_rew_3_ind))
    
        prev_b_rew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[prev_b_rew_3_ind],0)
        prev_b_rew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[prev_b_rew_3_ind],0)/np.sqrt(len(prev_b_rew_3_ind))
    
    
        # b_prev_a_rew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[b_prev_a_rew_3_ind],0)
        # b_prev_a_rew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[b_prev_a_rew_3_ind],0)/np.sqrt(len(b_prev_a_rew_3_ind))
    
        # b_prev_b_rew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[b_prev_b_rew_3_ind],0)
        # b_prev_b_rew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[b_prev_b_rew_3_ind],0)/np.sqrt(len(b_prev_b_rew_3_ind))
    
        # no reward
        prev_a_nrew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[prev_a_nrew_3_ind],0)
        prev_a_nrew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[prev_a_nrew_3_ind],0)/np.sqrt(len(prev_a_nrew_3_ind))
    
        prev_b_nrew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[prev_b_nrew_3_ind],0)
        prev_b_nrew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[prev_b_nrew_3_ind],0)/np.sqrt(len(prev_b_nrew_3_ind))
    
    
        # b_prev_a_nrew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[b_prev_a_nrew_3_ind],0)
        # b_prev_a_nrew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[b_prev_a_nrew_3_ind],0)/np.sqrt(len(b_prev_a_nrew_3_ind))
    
        # b_prev_b_nrew_3[0,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates_3[b_prev_b_nrew_3_ind],0)
        # b_prev_b_nrew_3[1,n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.std(firing_rates_3[b_prev_b_nrew_3_ind],0)/np.sqrt(len(b_prev_b_nrew_3_ind))
    
    # return  a_prev_a_rew_1, a_prev_a_nrew_1, a_prev_b_rew_1, a_prev_b_nrew_1,\
    #         b_prev_a_rew_1, b_prev_a_nrew_1, b_prev_b_rew_1, b_prev_b_nrew_1,\
    #         a_prev_a_rew_2, a_prev_a_nrew_2, a_prev_b_rew_2, a_prev_b_nrew_2,\
    #         b_prev_a_rew_2, b_prev_a_nrew_2, b_prev_b_rew_2, b_prev_b_nrew_2,\
    #         a_prev_a_rew_3, a_prev_a_nrew_3, a_prev_b_rew_3, a_prev_b_nrew_3,\
    #         b_prev_a_rew_3, b_prev_a_nrew_3, b_prev_b_rew_3, b_prev_b_nrew_3
    return  prev_a_rew_1, prev_a_nrew_1, prev_b_rew_1, prev_b_nrew_1,\
            prev_a_rew_2, prev_a_nrew_2, prev_b_rew_2, prev_b_nrew_2,\
            prev_a_rew_3, prev_a_nrew_3, prev_b_rew_3, prev_b_nrew_3
   

               
        
def plot_single_units_trials(data):
    # a_prev_a_rew_1, a_prev_a_nrew_1, a_prev_b_rew_1, a_prev_b_nrew_1,\
    #         b_prev_a_rew_1, b_prev_a_nrew_1, b_prev_b_rew_1, b_prev_b_nrew_1,\
    #         a_prev_a_rew_2, a_prev_a_nrew_2, a_prev_b_rew_2, a_prev_b_nrew_2,\
    #         b_prev_a_rew_2, b_prev_a_nrew_2, b_prev_b_rew_2, b_prev_b_nrew_2,\
    #         a_prev_a_rew_3, a_prev_a_nrew_3, a_prev_b_rew_3, a_prev_b_nrew_3,\
    #         b_prev_a_rew_3, b_prev_a_nrew_3, b_prev_b_rew_3, b_prev_b_nrew_3 = plot_prev_trial(data)
            
            
    prev_a_rew_1, prev_a_nrew_1, prev_b_rew_1, prev_b_nrew_1,\
            prev_a_rew_2, prev_a_nrew_2, prev_b_rew_2, prev_b_nrew_2,\
            prev_a_rew_3, prev_a_nrew_3, prev_b_rew_3, prev_b_nrew_3 = plot_prev_trial(data)
   
          
    pdf = PdfPages('/Users/veronikasamborska/Desktop/PFC_prev_trials.pdf')
    plt.ioff()
    neurons = prev_a_rew_1.shape[1]
    isl = wes.IsleOfDogs3_4.mpl_colors

    for n in range(neurons):
        plt.figure()
        plt.subplot(2,3,1)
        plt.plot(prev_a_rew_1[0,n,:], color = isl[0])
        plt.fill_between(np.arange(prev_a_rew_1[0,n,:].shape[0]),prev_a_rew_1[0,n,:]-prev_a_rew_1[1,n,:],\
                         prev_a_rew_1[0,n,:]+prev_a_rew_1[1,n,:], alpha = 0.5, color = isl[0])
        
        plt.plot(prev_b_rew_1[0,n,:], color = isl[1])
        plt.fill_between(np.arange(prev_b_rew_1[0,n,:].shape[0]),prev_b_rew_1[0,n,:]-prev_b_rew_1[1,n,:],\
                         prev_b_rew_1[0,n,:]+prev_b_rew_1[1,n,:], alpha = 0.5, color = isl[1])
        
        plt.plot(prev_a_nrew_1[0,n,:], color = isl[0], linestyle = '--')
        plt.fill_between(np.arange(prev_a_nrew_1[0,n,:].shape[0]),prev_a_nrew_1[0,n,:]-prev_a_nrew_1[1,n,:],\
                          prev_a_nrew_1[0,n,:]+prev_a_nrew_1[1,n,:], alpha = 0.5, color = isl[0])
        
        plt.plot(prev_b_nrew_1[0,n,:], color = isl[1],linestyle = '--')
        plt.fill_between(np.arange(prev_b_nrew_1[0,n,:].shape[0]),prev_b_nrew_1[0,n,:]-prev_b_nrew_1[1,n,:],\
                          prev_b_nrew_1[0,n,:]+prev_b_nrew_1[1,n,:], alpha = 0.5, color = isl[1])
        
            
        plt.subplot(2,3,2)
        plt.plot(prev_a_rew_2[0,n,:], color = isl[0])
        plt.fill_between(np.arange(prev_a_rew_2[0,n,:].shape[0]),prev_a_rew_2[0,n,:]-prev_a_rew_2[1,n,:],\
                         prev_a_rew_2[0,n,:]+prev_a_rew_2[1,n,:], alpha = 0.5, color = isl[0])
        
        plt.plot(prev_b_rew_2[0,n,:], color = isl[1])
        plt.fill_between(np.arange(prev_b_rew_2[0,n,:].shape[0]),prev_b_rew_2[0,n,:]-prev_b_rew_2[1,n,:],\
                         prev_b_rew_2[0,n,:]+prev_b_rew_2[1,n,:], alpha = 0.5, color = isl[1])
        
        plt.plot(prev_a_nrew_2[0,n,:], color = isl[0], linestyle = '--')
        plt.fill_between(np.arange(prev_a_nrew_2[0,n,:].shape[0]),prev_a_nrew_2[0,n,:]-prev_a_nrew_2[1,n,:],\
                         prev_a_nrew_2[0,n,:]+prev_a_nrew_2[1,n,:], alpha = 0.5, color = isl[0])
        
        plt.plot(prev_b_nrew_2[0,n,:], color = isl[1],linestyle = '--')
        plt.fill_between(np.arange(prev_b_nrew_2[0,n,:].shape[0]),prev_b_nrew_2[0,n,:]-prev_b_nrew_2[1,n,:],\
                         prev_b_nrew_2[0,n,:]+prev_b_nrew_2[1,n,:], alpha = 0.5, color = isl[1])


        plt.subplot(2,3,3)
        plt.plot(prev_a_rew_3[0,n,:], color = isl[0], label = 'prev A R')
        plt.fill_between(np.arange(prev_a_rew_3[0,n,:].shape[0]),prev_a_rew_3[0,n,:]-prev_a_rew_3[1,n,:],\
                         prev_a_rew_3[0,n,:]+prev_a_rew_3[1,n,:], alpha = 0.5, color = isl[0])
        
        plt.plot(prev_b_rew_3[0,n,:], color = isl[1], label = 'prev B R')
        plt.fill_between(np.arange(prev_b_rew_3[0,n,:].shape[0]),prev_b_rew_3[0,n,:]-prev_b_rew_3[1,n,:],\
                         prev_b_rew_3[0,n,:]+prev_b_rew_3[1,n,:], alpha = 0.5, color = isl[1])
        
        plt.plot(prev_a_nrew_3[0,n,:], color = isl[0], linestyle = '--',label = 'prev A NR')
        plt.fill_between(np.arange(prev_a_nrew_3[0,n,:].shape[0]),prev_a_nrew_3[0,n,:]-prev_a_nrew_3[1,n,:],\
                         prev_a_nrew_3[0,n,:]+prev_a_nrew_3[1,n,:], alpha = 0.5, color = isl[0])
        
        plt.plot(prev_b_nrew_3[0,n,:], color = isl[1],linestyle = '--', label = 'prev B NR')
        plt.fill_between(np.arange(prev_b_nrew_3[0,n,:].shape[0]),prev_b_nrew_3[0,n,:]-prev_b_nrew_3[1,n,:],\
                         prev_b_nrew_3[0,n,:]+prev_b_nrew_3[1,n,:], alpha = 0.5, color = isl[1])

        # ### B
        # plt.subplot(2,3,4)
        # plt.plot(b_prev_a_rew_1[0,n,:], color = isl[0])
        # plt.fill_between(np.arange(b_prev_a_rew_1[0,n,:].shape[0]),b_prev_a_rew_1[0,n,:]-b_prev_a_rew_1[1,n,:],\
        #                  b_prev_a_rew_1[0,n,:]+b_prev_a_rew_1[1,n,:], alpha = 0.5, color = isl[0])
        
        # plt.plot(b_prev_b_rew_1[0,n,:], color = isl[1])
        # plt.fill_between(np.arange(b_prev_b_rew_1[0,n,:].shape[0]),b_prev_b_rew_1[0,n,:]-b_prev_b_rew_1[1,n,:],\
        #                  b_prev_b_rew_1[0,n,:]+b_prev_b_rew_1[1,n,:], alpha = 0.5, color = isl[1])
        
        # plt.plot(b_prev_a_nrew_1[0,n,:], color = isl[0], linestyle = '--')
        # plt.fill_between(np.arange(b_prev_a_nrew_1[0,n,:].shape[0]),b_prev_a_nrew_1[0,n,:]-b_prev_a_nrew_1[1,n,:],\
        #                  b_prev_a_nrew_1[0,n,:]+b_prev_a_nrew_1[1,n,:], alpha = 0.5, color = isl[0])
        
        # plt.plot(b_prev_b_nrew_1[0,n,:], color = isl[1], linestyle = '--')
        # plt.fill_between(np.arange(b_prev_b_nrew_1[0,n,:].shape[0]),b_prev_b_nrew_1[0,n,:]-b_prev_b_nrew_1[1,n,:],\
        #                  b_prev_b_nrew_1[0,n,:]+b_prev_b_nrew_1[1,n,:], alpha = 0.5, color = isl[1])
        
            
        # plt.subplot(2,3,5)
        # plt.plot(b_prev_a_rew_2[0,n,:], color = isl[0])
        # plt.fill_between(np.arange(b_prev_a_rew_2[0,n,:].shape[0]),b_prev_a_rew_2[0,n,:]-b_prev_a_rew_2[1,n,:],\
        #                  b_prev_a_rew_2[0,n,:]+b_prev_a_rew_2[1,n,:], alpha = 0.5, color = isl[0])
        
        # plt.plot(b_prev_b_rew_2[0,n,:], color = isl[1])
        # plt.fill_between(np.arange(b_prev_b_rew_2[0,n,:].shape[0]),b_prev_b_rew_2[0,n,:]-b_prev_b_rew_2[1,n,:],\
        #                  b_prev_b_rew_2[0,n,:]+b_prev_b_rew_2[1,n,:], alpha = 0.5, color = isl[1])
        
        # plt.plot(b_prev_a_nrew_2[0,n,:], color = isl[0], linestyle = '--')
        # plt.fill_between(np.arange(b_prev_a_nrew_2[0,n,:].shape[0]),b_prev_a_nrew_2[0,n,:]-b_prev_a_nrew_2[1,n,:],\
        #                  b_prev_a_nrew_2[0,n,:]+b_prev_a_nrew_2[1,n,:], alpha = 0.5, color = isl[0])
        
        # plt.plot(b_prev_b_nrew_2[0,n,:], color = isl[1],linestyle = '--')
        # plt.fill_between(np.arange(b_prev_b_nrew_2[0,n,:].shape[0]),b_prev_b_nrew_2[0,n,:]-b_prev_b_nrew_2[1,n,:],\
        #                  b_prev_b_nrew_2[0,n,:]+b_prev_b_nrew_2[1,n,:], alpha = 0.5, color = isl[1])


        # plt.subplot(2,3,6)
        # plt.plot(b_prev_a_rew_3[0,n,:], color = isl[0], label = ' prev A R')
        # plt.fill_between(np.arange(b_prev_a_rew_3[0,n,:].shape[0]),b_prev_a_rew_3[0,n,:]-b_prev_a_rew_3[1,n,:],\
        #                  b_prev_a_rew_3[0,n,:]+b_prev_a_rew_3[1,n,:], alpha = 0.5, color = isl[0])
        
        # plt.plot(b_prev_b_rew_3[0,n,:], color = isl[1], label = 'prev B R')
        # plt.fill_between(np.arange(b_prev_b_rew_3[0,n,:].shape[0]),b_prev_b_rew_3[0,n,:]-b_prev_b_rew_3[1,n,:],\
        #                  b_prev_b_rew_3[0,n,:]+b_prev_b_rew_3[1,n,:], alpha = 0.5, color = isl[1])
        
        # plt.plot(b_prev_a_nrew_3[0,n,:], color = isl[0], linestyle = '--', label = ' prev A NR')
        # plt.fill_between(np.arange(b_prev_a_nrew_3[0,n,:].shape[0]),b_prev_a_nrew_3[0,n,:]-b_prev_a_nrew_3[1,n,:],\
        #                  b_prev_a_nrew_3[0,n,:]+b_prev_a_nrew_3[1,n,:], alpha = 0.5, color = isl[0])
        
        # plt.plot(b_prev_b_nrew_3[0,n,:], color = isl[1], linestyle = '--', label = ' prev B NR')
        # plt.fill_between(np.arange(b_prev_b_nrew_3[0,n,:].shape[0]),b_prev_b_nrew_3[0,n,:]-b_prev_b_nrew_3[1,n,:],\
        #                  b_prev_b_nrew_3[0,n,:]+b_prev_b_nrew_3[1,n,:], alpha = 0.5, color = isl[1])
        plt.legend()

        plt.tight_layout()
        sns.despine()
        pdf.savefig()
        plt.clf()
        
    pdf.close()
        