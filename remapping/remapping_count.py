#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:20:39 2019

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt


## Script to find  cells that are selective to A or I (that's the same across two tasks)
## in one task --> look at what happens at a different task

def task_ind(task, a_pokes, b_pokes):
    taskid = np.zeros(len(task));

    #TASK1 -> A and B at leftright extremes; b_pokes = 10-apokes;
    taskid[b_pokes == 10 - a_pokes] = 1
        
    #TASK2 -> b is one of the diagonal ones [2,3,7,8]
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2
        
    #TASK3 -> b either top or bottom [2,3,7,8] (1 or 9). 
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid

def autocorrelogram_fr(data, color, fig_n = 3):
    
    x = data['Data']
    mean_n = []
    for  s, sess in enumerate(x):
        fr = x[s]       
        n_trials, n_neurons, n_time = fr.shape  
        for neuron in range(n_neurons):        
            n_firing = fr[:,neuron,:]  # Firing rate of each neuron
            corr_list =[]

            for n_t in range(n_trials):
                trial_spikes = n_firing[n_t,:]  # Firing rate of each neuron
                corr = np.correlate(trial_spikes, trial_spikes,mode='full')
                corr_list.append(corr)
            m = np.mean(corr_list,axis = 0)    
            mean_n.append(m)
    all_ave = np.nanmean(mean_n, axis = 0 )
    all_err = np.nanstd(mean_n, axis = 0)/np.sqrt(len(mean_n))
    #all_err_demean = all_err/np.max(all_err)
    plt.figure(fig_n)
    plt.plot(all_ave, color = color)
    x = np.arange(all_ave.shape[0])
    plt.fill_between(x, all_ave+all_err, all_ave-all_err, color=color, alpha='0.3')
 

def Is_remapping(data):
    #Lists for appending the last 20 trials of task 1 and the first 20 trials of task 2 for neurons 
    # that either decreased or increased their firing rates between two tasks around choice time
    y = data['DM']
    x = data['Data']
    
    fg_n = 1
    fg_n_i = 7
    all_ns = 0
    fg_n_i_i = 12
    
    i1_i2_on = 0
    i2_i1_on = 0
    i2_i1_remap = 0
    i1_i2_remap = 0
    
    n = 0
    N_a = 0
    for  s, sess in enumerate(x):
        DM = y[s]

        choices = DM[:,1]

        b_pokes = DM[:,6]
        a_pokes = DM[:,5]
        task = DM[:,4]
            
        taskid = task_ind(task,a_pokes,b_pokes)
        It = np.arange(23 ,25) #Init
        Allt = np.arange(30, 63) #Init
        task_1_i = np.where(taskid == 1)[0]
        task_2_i = np.where(taskid == 3)[0]
        
        #task_1_i = np.where((taskid == 1) & (choices == 0))[0]
        #task_2_i = np.where((taskid == 3) & (choices == 0))[0]
        
        not_i =  np.where(taskid == 2)[0]
        
        firing_rates_mean_time = np.mean(x[s][:,:,It], axis = 2)
        firing_rates_all_time = x[s][:,:,:]
        firing_rates_mean_time_after_init = np.mean(x[s][:,:,Allt], axis = 2)
        n_trials, n_neurons = firing_rates_mean_time.shape
        
        for neuron in range(n_neurons):
            
            n+=1
            n_firing = firing_rates_mean_time[:,neuron]  # Firing rate of each neuron
            mean_firing_all_trials = np.mean(n_firing[not_i],axis = 0)
            std_firing_all_trials = np.std(n_firing[not_i],axis = 0)

            rest_firing = firing_rates_mean_time_after_init[:,neuron] 
            
            i1_fr = n_firing[task_1_i]  # Firing rates on poke A choices in Task 1 
            i1_mean = np.mean(i1_fr, axis = 0)  # Mean rate on poke A choices in Task 1 
            rest_firing_t1 = rest_firing[task_1_i]
            mean_rest_firing_t1 = np.mean(rest_firing_t1, axis = 0)

            i2_fr = n_firing[task_2_i]
            i2_mean = np.mean(i2_fr, axis = 0)
            rest_firing_t2 = rest_firing[task_2_i]
            mean_rest_firing_t2 = np.mean(rest_firing_t2, axis = 0)


            if i1_mean > 0.5 or i2_mean > 0.5:
                
                # Find neurons mean firing rate on A in any of the tasks is 2 standard deviations higher than baseline on all trials
                if i1_mean > mean_firing_all_trials+ (std_firing_all_trials) or i2_mean > mean_firing_all_trials+ (std_firing_all_trials):
                   
                    if i1_mean > mean_rest_firing_t1 or i2_mean > mean_rest_firing_t2:
                        N_a +=1
                        all_ns += 1 
                        fig =  plt.figure(fg_n)
                        if all_ns > 25:
                            fg_n += 1
                            fig =  plt.figure(fg_n)
                            all_ns = 1 
                        fig.add_subplot(5,5,all_ns)
                        plt.plot(np.mean(firing_rates_all_time[task_1_i,neuron, :], axis = 0), color = 'darkred', label = 'task 1 A')
                        plt.plot(np.mean(firing_rates_all_time[task_2_i,neuron, :],axis =0),color = 'salmon', label = 'task 2 A')
                        
                        ym = np.max([(np.mean(firing_rates_all_time[task_1_i,neuron, :], axis = 0)),\
                                     (np.mean(firing_rates_all_time[task_2_i,neuron, :],axis =0))])
                        plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                        plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                        plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')
                        
                        if i1_mean > (mean_firing_all_trials + std_firing_all_trials) \
                        and i2_mean < (mean_firing_all_trials + std_firing_all_trials):     
                            i1_i2_remap +=1
                            i1_i2_on +=1
                            fig =  plt.figure(fg_n_i_i)
                            if i1_i2_on > 25:
                                fg_n_i += 1
                                fig =  plt.figure(fg_n_i_i)
                                i1_i2_on = 1
                            fig.add_subplot(5,5,i1_i2_on)
                            plt.plot(np.mean(firing_rates_all_time[task_1_i,neuron, :], axis = 0), color = 'darkred')
                            plt.plot(np.mean(firing_rates_all_time[task_2_i,neuron, :],axis =0),color = 'salmon', label = 'task 2 A')
                            plt.title('I1 but not I2')
                            ym = np.max([(np.mean(firing_rates_all_time[task_1_i,neuron, :], axis = 0)),\
                                     (np.mean(firing_rates_all_time[task_2_i,neuron, :],axis =0))])
                            plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                            plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                            plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')
    
                        if i2_mean > (mean_firing_all_trials + std_firing_all_trials) \
                        and i1_mean < (mean_firing_all_trials + std_firing_all_trials):
                            i2_i1_remap +=1
                            i2_i1_on += 1
                            fig =  plt.figure(fg_n_i)
                            if i2_i1_on > 25:
                                fg_n_i += 1
                                fig =  plt.figure(fg_n_i)
                                i2_i1_on = 1
                            fig.add_subplot(5,5,i2_i1_on)
                            plt.plot(np.mean(firing_rates_all_time[task_1_i,neuron, :], axis = 0), color = 'darkred')
                            plt.plot(np.mean(firing_rates_all_time[task_2_i,neuron, :],axis =0),color = 'salmon', label = 'task 2 A')
                            plt.title('I2 but not I1')
                            ym = np.max([(np.mean(firing_rates_all_time[task_1_i,neuron, :], axis = 0)),\
                                     (np.mean(firing_rates_all_time[task_2_i,neuron, :],axis =0))])
                                     
        
                            plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                            plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                            plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')
    return N_a,i2_i1_remap,i1_i2_remap

        
    
def As_remapping(data_HP):
    #Lists for appending the last 20 trials of task 1 and the first 20 trials of task 2 for neurons 
    # that either decreased or increased their firing rates between two tasks around choice time
    y = data_HP['DM']
    x = data_HP['Data']
    
    fg_n = 1
    all_ns = 0
    a1_a2_on= 0
    a2_on = 0
    a1_on = 0
    a3_on = 0
    a2_a3_on = 0
    a1_a3_on = 0
    
    n = 0
    N_a = 0
    for  s, sess in enumerate(x):
        DM = y[s]
        #state =  DM[:,0]
       
        choices = DM[:,1]
        b_pokes = DM[:,6]
        a_pokes = DM[:,5]
        task = DM[:,4]
        taskid = task_ind(task,a_pokes,b_pokes)
        
        task_1_a = np.where((taskid == 1) & (choices == 1))[0]
        task_2_a = np.where((taskid == 2) & (choices == 1))[0]
        task_3_a = np.where((taskid == 3) & (choices == 1))[0]
        
        not_a = np.where(choices == 0)[0]
        
        Ct = np.arange(36, 63) #Choice

        firing_rates_mean_time = np.mean(x[s][:,:,Ct], axis = 2)
        firing_rates_all_time = x[s][:,:,:]
        n_trials, n_neurons = firing_rates_mean_time.shape
        
        # Numpy arrays to fill the firing rates of each neuron where the A choice was made
       
        
        for neuron in range(n_neurons):
            n+=1
            n_firing = firing_rates_mean_time[:,neuron]  # Firing rate of each neuron
            mean_firing_all_trials = np.mean(n_firing[not_a],axis = 0)
            std_firing_all_trials = np.std(n_firing[not_a],axis = 0)
           
            a1_fr = n_firing[task_1_a]  # Firing rates on poke A choices in Task 1 
            a1_mean = np.mean(a1_fr, axis = 0)  # Mean rate on poke A choices in Task 1 

            a2_fr = n_firing[task_2_a]
            a2_mean = np.mean(a2_fr, axis = 0)

            a3_fr = n_firing[task_3_a]
            a3_mean = np.mean(a3_fr, axis = 0)

            
            if a1_mean > 0.5 or a2_mean > 0.5 or a3_mean > 0.5:
                
                # Find neurons mean firing rate on A in any of the tasks is 2 standard deviations higher than baseline on all trials
                if a1_mean > mean_firing_all_trials+ (std_firing_all_trials) or a2_mean > mean_firing_all_trials+ (std_firing_all_trials) or  a3_mean > mean_firing_all_trials+(std_firing_all_trials) :
                    N_a +=1
                    all_ns += 1 
                    fig =  plt.figure(fg_n)
                    if all_ns > 25:
                        fg_n += 1
                        fig =  plt.figure(fg_n)
                        all_ns = 1 
                        
                    fig.add_subplot(5,5,all_ns)
                    plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue', label = 'task 1 A')
                    #plt.plot(np.mean(firing_rates_all_time[task_1_b,neuron, :], axis = 0),color = 'red')
                    plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                    #plt.plot(np.mean(firing_rates_all_time[task_2_b,neuron, :], axis = 0),color = 'orange')
                    plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                    #plt.plot(np.mean(firing_rates_all_time[task_3_b,neuron, :], axis = 0), color = 'yellow')
                    
                    ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
                    #print(ym)
                    plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                    plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                    plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')

                    if a1_mean > (mean_firing_all_trials + std_firing_all_trials) \
                    and a2_mean > (mean_firing_all_trials + std_firing_all_trials) \
                    and a3_mean < (mean_firing_all_trials + std_firing_all_trials):
                       a1_a2_on += 1 
                       fig =  plt.figure(6)
                       fig.add_subplot(3,4,a1_a2_on)
                       plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue')
                       plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                       plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                       plt.title('A1 and A2 not A3')
                       ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
    
                       plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                       plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                       plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')

                    
                    if a2_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a1_mean < (mean_firing_all_trials + std_firing_all_trials)\
                    and a3_mean <  (mean_firing_all_trials + std_firing_all_trials):
                        a2_on += 1  
                        fig =  plt.figure(7)
                        fig.add_subplot(3,4,a2_on)
                        plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue')
                        plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                        plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                        plt.title('A2 not A1 or A3')
                        ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
    
                        plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                        plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                        plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')

                    
                    if a1_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a2_mean < (mean_firing_all_trials + std_firing_all_trials)\
                    and a3_mean < (mean_firing_all_trials + std_firing_all_trials):
                        a1_on += 1
                        fig =  plt.figure(8)
                        fig.add_subplot(3,6,a1_on)
                        cell_t1 = firing_rates_all_time[task_1_a,neuron, :]
                        cell_t2 = firing_rates_all_time[task_2_a,neuron, :]

                        plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue')
                        plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                        plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                        plt.title('A1 not A2 or A3')
                        ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
    
                        plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                        plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                        plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')

                    
#               
#            
                    if a3_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a1_mean < (mean_firing_all_trials + std_firing_all_trials)\
                    and a2_mean <  (mean_firing_all_trials + std_firing_all_trials):
                        a3_on += 1
                        fig =  plt.figure(9)
                        fig.add_subplot(3,6,a3_on)
                        plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue')
                        plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                        plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                        plt.title('A3 not A1 or A2')
                        ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
    
                        plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                        plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                        plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')

                    
#                        
                    if a2_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a3_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a1_mean < (mean_firing_all_trials + std_firing_all_trials):
                        a2_a3_on += 1
                        fig =  plt.figure(10)
                        fig.add_subplot(2,5,a2_a3_on)
                        
                        plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue')
                        plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                        plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                        ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
    
                        plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                        plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                        plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')
 
                        plt.title('A2 and A3 not A1')
#                 
                        
                    if a1_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a3_mean > (mean_firing_all_trials + std_firing_all_trials)\
                    and a2_mean <  (mean_firing_all_trials +std_firing_all_trials):
                        a1_a3_on += 1
                        fig =  plt.figure(11)
                        fig.add_subplot(3,4,a1_a3_on)
                        
                        plt.plot(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0), color = 'darkblue')
                        plt.plot(np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0),color = 'blue', label = 'task 2 A')
                        plt.plot(np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0),color = 'lightblue', label = 'task 3 A')
                        plt.title('A1 and A3 not A2')
#                 
                        ym = np.max([(np.mean(firing_rates_all_time[task_1_a,neuron, :], axis = 0)),\
                                 (np.mean(firing_rates_all_time[task_2_a,neuron, :],axis =0)),\
                                 (np.mean(firing_rates_all_time[task_3_a,neuron, :],axis = 0))])
    
                        plt.vlines(25, ymin =  0, ymax = ym, linestyle = '--',color = 'grey')
                        plt.vlines(36, ymin =  0, ymax = ym,linestyle = '--', color = 'black')
                        plt.vlines(42, ymin =  0, ymax = ym,linestyle = '--', color = 'pink')

                    
                    remapped = a1_a3_on+a2_a3_on+a3_on+a1_on+a2_on+a1_a2_on
                    
    return  N_a, remapped,cell_t1,cell_t2