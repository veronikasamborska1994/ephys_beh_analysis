#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:07:39 2019

@author: veronikasamborska
"""


# Script for running remapping analysis based on a permutation test

import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm
import numpy.matlib as mat

font = {'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

def perm_test(firing_t1,firing_t2,n_perm):
    
    activity = np.vstack((firing_t1,firing_t2))
    activity_diff = np.mean(firing_t1, axis = 0) - np.mean(firing_t2, axis = 0)
    n_trials  = int(activity.shape[0]/2)
    activity_perm = np.zeros((n_perm, firing_t1.shape[1]))
    activity_max = np.zeros(n_perm)
    
    for i in range(n_perm):
        np.random.shuffle(activity) # Shuffle A / B trials (axis 0 only).
        activity_perm[i,:] = (np.mean(activity[:n_trials,:],0) -
                                         np.mean(activity[n_trials:,:],0))
        activity_max[i] = np.max(activity_perm[i,:])
       
    p = np.percentile(activity_perm,99, axis = 0)
    p_max = np.percentile(activity_max, 99)
    x_max = np.zeros(len(p))
    x_max[:] = p_max
    
    return p, p_max, x_max, activity_diff
  
    
def remap(data,n_perm):
    
    y = data['DM']
    x = data['Data']
    
    fg_n = 1
    fg_nr = 30
    all_ns = 0  
    all_ns_non_remapped = 0
    remapped_between = 0
    remapped_within = 0
    remapped_between_not_within = 0
    ns = 0

  
    for  s, sess in enumerate(x):
        DM = y[s]
        #state =  DM[:,0]
       
        choices = DM[:,1]
        b_pokes = DM[:,6]
        a_pokes = DM[:,5]
        task = DM[:,4]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        
        task_1_a = np.where((taskid == 1) & (choices == 0))[0]
        task_1_a_1 = task_1_a[:int(len(task_1_a)/2)]
        task_1_a_2 = task_1_a[int(len(task_1_a)/2):]

        task_2_a = np.where((taskid == 2) & (choices == 0))[0]
        task_3_a = np.where((taskid == 3) & (choices == 0))[0]
        task_2_a_1 = task_2_a[:int(len(task_2_a)/2)]
        task_2_a_2 = task_2_a[int(len(task_2_a)/2):]
        
        task_3_a = task_3_a[:int(len(task_3_a)/2)]
        task_3_a_1 = task_3_a[int(len(task_3_a)/2):]

        It = np.arange(25 ,30) #Init
        Ct = np.arange(36, 41) #Choice

        firing_rates_mean_time = x[s]
        firing_rates_all_time = x[s][:,:,:]
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        # Numpy arrays to fill the firing rates of each neuron where the A choice was made
       
        
        for neuron in range(n_neurons):
            n_firing = firing_rates_all_time[:,neuron]  # Firing rate of each neuron
            ns +=1 
            ## Task 1 --> 2 
            a1_fr_between = n_firing[task_1_a_2]
            a1_fr_between = a1_fr_between[:,It]  
            a2_fr_between = n_firing[task_2_a_1]
            a2_fr_between = a2_fr_between[:,It]  

            a1_fr_within = n_firing[task_1_a_1]
            a1_fr_within = a1_fr_within[:,It]  

            a2_fr_within = n_firing[task_1_a_2]

            a2_fr_within = a2_fr_within[:,It]  

            p_within, p_max_within, x_max_within, activity_diff_within = perm_test(a1_fr_within,a2_fr_within,n_perm = n_perm)
            p_between, p_max_between, x_max_between, activity_diff_between = perm_test(a1_fr_between,a2_fr_between,n_perm = n_perm)

            if (np.max(abs(activity_diff_between)) > p_max_between):
                remapped_between += 1 
                
                all_ns += 1 
                fig =  plt.figure(fg_n)
                
                if all_ns > 20:
                    fg_n += 1
                    fig =  plt.figure(fg_n)
                    all_ns = 1 
                        
                fig.add_subplot(4,5,all_ns)
                plt.plot(np.mean(firing_rates_all_time[task_2_a_2,neuron, :], axis = 0), color = 'coral', label = 'task 1 A')
                plt.plot(np.mean(firing_rates_all_time[task_3_a_1,neuron, :], axis =0),color = 'red', label = 'task 2 A')
                #plt.plot(np.mean(firing_rates_all_time[task_1_a_1,neuron, :], axis = 0), color = 'lightblue', label = 'task 1 A')
                
                ym = np.max([(np.mean(firing_rates_all_time[task_2_a_2,neuron, :], axis = 0)),\
                             (np.mean(firing_rates_all_time[task_3_a_1,neuron, :],axis =0))])
                             
                #print(ym)
               # plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
               # plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
               # plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')
                plt.plot(activity_diff_between, 'black')

                plt.plot(x_max_between, 'red', linestyle = '--', alpha = 0.5, linewidth = 0.5)
                plt.plot(-x_max_between, 'red', linestyle = '--', alpha = 0.5, linewidth = 0.5)

            
            if (np.max(abs(activity_diff_within)) > p_max_within):
                remapped_within += 1 
             
                all_ns_non_remapped += 1 
                fig =  plt.figure(fg_nr)
                if all_ns_non_remapped > 20:
                    fg_nr += 1
                    fig =  plt.figure(fg_nr)
                    all_ns_non_remapped = 1 
                        
                fig.add_subplot(4,5,all_ns_non_remapped)
                plt.plot(np.mean(firing_rates_all_time[task_2_a_2,neuron, :], axis = 0), color = 'blue', label = 'task 1 A')
                #plt.plot(np.mean(firing_rates_all_time[task_3_a_1,neuron, :], axis =0), color = 'red', label = 'task 2 A')
                plt.plot(np.mean(firing_rates_all_time[task_2_a_1,neuron, :], axis = 0), color = 'lightblue', label = 'task 1 A')
                
                ym = np.max([(np.mean(firing_rates_all_time[task_2_a_2,neuron, :], axis = 0)),\
                             (np.mean(firing_rates_all_time[task_2_a_1,neuron, :],axis = 0))])
                             
                #print(ym)
               # plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
              #  plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
              #  plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')
                
                plt.plot(activity_diff_within, 'grey')

                plt.plot(x_max_within, 'red', linestyle = '--', alpha = 0.5, linewidth = 0.5)
                plt.plot(-x_max_within, 'red', linestyle = '--', alpha = 0.5, linewidth = 0.5)

            if (np.max(abs(activity_diff_between)) > p_max_between) and (np.max(abs(activity_diff_within)) < p_max_within):
                
                remapped_between_not_within += 1
                
    return remapped_between_not_within, remapped_between, remapped_within, ns



def plot_pie(data_HP, data_PFC):
    
    remapped_between_not_within, remapped_between, remapped_within, ns = remap(data_HP, n_perm = 100)
    labels = 'Remapped Between HP', 'Remapped Within HP', 'No Remapping'
    sizes = [remapped_between, remapped_within, (ns-remapped_between-remapped_within)]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow = False, startangle=90)
    ax1.axis('equal')  
    
    plt.show()    
    
    remapped_between_not_within_PFC, remapped_between_PFC, remapped_within_PFC, ns_PFC = remap(data_PFC, n_perm = 100)         
    labels = 'Remapped Between PFC', 'Remapped Within PFC', 'No Remapping PFC'
    sizes = [remapped_between_PFC, remapped_within_PFC, (ns_PFC-remapped_within_PFC-remapped_between_PFC)]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow = False, startangle=90)
    ax1.axis('equal') 
    plt.show()
          
    
    