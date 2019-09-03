#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:07:39 2019

@author: veronikasamborska
"""


# Script for running remapping analysis based on a permutation test -- > good; surprise measure analysis ---> crap
 
import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm

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
          
    
    
def remap_surprise(data,fig_n,title):
    
    y = data['DM']
    x = data['Data']

    surprise_list_neurons_1_2 = []
    surprise_list_neurons_2_3 = []
    surprise_list_neurons_b1_b2 = []
    surprise_list_neurons_b_i = []
    surprise_list_neurons_rev_a = []
    
    for  s, sess in enumerate(x):
        DM = y[s]
        #state =  DM[:,0]
       
        choices = DM[:,1]
        b_pokes = DM[:,6]
        a_pokes = DM[:,5]
        task = DM[:,4]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        
        It = np.arange(20 ,25) #Init
        Ct = np.arange(30, 35) #Choice
        
        task_1_a = np.where((taskid == 1) & (choices == 1))[0] # Find indicies for task 1 A
        task_1_a_last = task_1_a[-10:] # Find indicies for task 1 A last 10 

        task_1_b = np.where((taskid == 1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_last = task_1_b[-10:] # Find indicies for task 1 B last 10 

        task_2_b = np.where((taskid == 2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_first = task_2_b[:10] # First indicies for task 2 B
        task_2_b_last = task_2_b[-10:]  # Last 10 indicies for task 2 B
        
        task_2_a = np.where((taskid == 2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_first = task_2_a[:10] # First 10 indicies for task 2 A 
        
        task_3_i = np.where((taskid == 3) & (choices == 1))[0]  # Find indicies for Initiations Task 3 As
        task_3_i_first = task_3_i[-10:] # First 10 indicies for Initiations Task 3 As
        
        firing_rates_mean_time = x[s]
        firing_rates_mean_time_ch = np.mean(firing_rates_mean_time[:,:,Ct], axis = 2)
        firing_rates_mean_time_init = np.mean(firing_rates_mean_time[:,:,It], axis = 2)

        n_trials, n_neurons = firing_rates_mean_time_ch.shape
        
        for neuron in range(n_neurons):
            
            n_firing_ch = firing_rates_mean_time_ch[:,neuron]  # Firing rate of each neuron
            n_firing_init =  firing_rates_mean_time_init[:,neuron]
                     
            # Task 2 --> 1 Mean rates on the last 20 trials of task 2 A for reversal
            choice_2_a_mean = np.mean(n_firing_ch[task_2_a[-10:]])
            choice_2_a_std = np.std(n_firing_ch[task_2_a[-10:]])
            
            # Task 1 --> 2 Mean rates on the first 20 trials of task 1 A
            choice_1_mean = np.mean(n_firing_ch[task_1_a[:10]])
            choice_1_std = np.std(n_firing_ch[task_1_a[:10]])
            
            # Task 2 --> 1 Mean rates on the first 20 trials of task 1 Bs 
            choice_1_b_mean = np.mean(n_firing_ch[task_1_b[:10]])
            choice_1_b_std = np.std(n_firing_ch[task_1_b[:10]])
           
            # Task 2 --> 3 Mean rates on the first 20 trials of task 2 around Initiations 
            init_2_mean = np.mean(n_firing_init[task_2_b[:10]]) 
            init_2_std = np.std(n_firing_init[task_2_b[:10]])
                
            # A last 10 trials around Choice task 1
            a1_fr_last = n_firing_ch[task_1_a_last]
            
            # A first 10 trials around Choice task 2
            a2_fr_first = n_firing_ch[task_2_a_first]
            
            # B last 10 trials around Choice task 1
            b1_fr_last = n_firing_ch[task_1_b_last]
            
            # B first 10 trials around Choice task 2
            b2_fr_first = n_firing_ch[task_2_b_first]
            
            # I first 10 trials around Init task 2
            i2_fr_first = n_firing_init[task_2_b_first]

            # I around Init in task 2 
            i2_fr_last = n_firing_init[task_2_b_last]
            
            # I around Ch in task 3
            i3_fr_first = n_firing_ch[task_3_i_first]
            
            if choice_1_mean > 3 and init_2_mean > 3 and choice_1_b_mean > 3 and choice_2_a_mean > 3:
                
                # B to I diff Pokes 
                surprise_b_i_1 = -norm.logpdf(b1_fr_last,choice_1_b_mean, choice_1_b_std)    
                surprise_b_i_2 = -norm.logpdf(i2_fr_first,choice_1_b_mean, choice_1_b_std)
                
                # B to B
                surprise_b1 = -norm.logpdf(b1_fr_last,choice_1_b_mean, choice_1_b_std)  
                surprise_b2 = -norm.logpdf(b2_fr_first,choice_1_b_mean, choice_1_b_std)
                
                # A to B
                surprise_a1 = -norm.logpdf(a1_fr_last,choice_1_mean, choice_1_std)                        
                surprise_a2 = -norm.logpdf(a2_fr_first,choice_1_mean, choice_1_std)
                
                # Check if A correct
                surprise_a1_rev = -norm.logpdf(a1_fr_last,choice_2_a_mean, choice_2_a_std)                       
                surprise_a2_rev = -norm.logpdf(a2_fr_first,choice_2_a_mean, choice_2_a_std)
                
                # B to I
                surprise_b2_last = -norm.logpdf(i2_fr_last,init_2_mean, init_2_std)                        
                surprise_i3_first= -norm.logpdf(i3_fr_first,init_2_mean, init_2_std)

                
                surprise_array_t1_2 = np.concatenate([surprise_a1, surprise_a2])                   
                surprise_array_t2_3 = np.concatenate([surprise_b2_last,surprise_i3_first])         
                surprise_array_b1_b2 = np.concatenate([surprise_b1,surprise_b2])
                surprise_array_b_i =  np.concatenate([surprise_b_i_1,surprise_b_i_2])
                surprise_array_rev_a =  np.concatenate([surprise_a1_rev,surprise_a2_rev])
                
                surprise_list_neurons_1_2.append(surprise_array_t1_2)
                surprise_list_neurons_2_3.append(surprise_array_t2_3)
                surprise_list_neurons_b1_b2.append(surprise_array_b1_b2)
                surprise_list_neurons_b_i.append(surprise_array_b_i)
                surprise_list_neurons_rev_a.append(surprise_array_rev_a)
    
    mean_1_2 =np.mean(surprise_list_neurons_1_2, axis = 0)
    std_1_2 =np.nanstd(surprise_list_neurons_1_2, axis = 0)
    serr_1_2 = std_1_2/np.sqrt(len(surprise_list_neurons_1_2))
    mean_2_3 =np.nanmean(surprise_list_neurons_2_3, axis = 0)
    std_2_3 =np.nanstd(surprise_list_neurons_2_3, axis = 0)
    serr_2_3 = std_2_3/np.sqrt(len(surprise_list_neurons_2_3))

    x_pos = np.arange(len(mean_2_3))
    task_change = 10
    
    plt.figure(fig_n)
    plt.plot(x_pos,mean_2_3,label = 'B Init Same Space', color = 'grey')
    plt.fill_between(x_pos, mean_2_3 - serr_2_3, mean_2_3 + serr_2_3, alpha=0.2,color = 'grey')
    
    plt.axvline(task_change, color='k', linestyle=':')
#    plt.title('Init B becomes Init')
    plt.ylabel('-log(p(X))')
    plt.xlabel('Trial #')

    plt.plot(x_pos,mean_1_2, label =  'A becomes A', color = 'darkblue')
    plt.fill_between(x_pos, mean_1_2 - serr_1_2, mean_1_2 + serr_1_2, alpha=0.2,  color = 'darkblue')

    plt.axvline(task_change, color='k', linestyle=':')
#    plt.title('A becomes A')
    plt.ylabel('-log(p(X))')
    plt.xlabel('Trial #')
    
    mean_b1_b2 =np.nanmean(surprise_list_neurons_b1_b2, axis = 0)
    std_b1_b2 =np.nanstd(surprise_list_neurons_b1_b2, axis = 0)
    serr_b1_b2 = std_b1_b2/np.sqrt(len(surprise_list_neurons_b1_b2))
    
    plt.plot(x_pos,mean_b1_b2,label = ' B becomes diff B', color = 'green')
    plt.fill_between(x_pos, mean_b1_b2 - serr_b1_b2, mean_b1_b2 + serr_b1_b2, alpha=0.2, color = 'green')

    
    mean_b_i =np.nanmean(surprise_list_neurons_b_i, axis = 0)
    std_b_i =np.nanstd(surprise_list_neurons_b_i, axis = 0)
    serr_b_i = std_b_i/np.sqrt(len(surprise_list_neurons_b_i))
    
    plt.plot(x_pos,mean_b_i,label = ' B I diff Space', color = 'darkred')
    plt.fill_between(x_pos, mean_b_i - serr_b_i, mean_b_i + serr_b_i, alpha=0.2, color = 'darkred')

    plt.legend()
    plt.title(title)


    mean_a_rev =np.nanmean(surprise_list_neurons_rev_a, axis = 0)
    std_a_rev =np.nanstd(surprise_list_neurons_rev_a, axis = 0)
    serr_a_rev = std_a_rev/np.sqrt(len(surprise_list_neurons_rev_a))
    
    plt.plot(x_pos,mean_a_rev,label = ' A to A rev', color = 'lightblue')
    plt.fill_between(x_pos, mean_a_rev - serr_a_rev, mean_a_rev + serr_a_rev, alpha=0.2, color = 'lightblue')

    plt.legend()
    plt.title(title)
    
    return surprise_list_neurons_1_2, surprise_list_neurons_2_3, surprise_list_neurons_b1_b2, surprise_list_neurons_b_i,\
            surprise_list_neurons_rev_a
            
            
            
            
    
    
def plot_surprise_n(a_a, b_i_same, b_b, b_i_diff, a_a_rev):        
    fig_n = 1
    n = 0 
    task_change  = 10
    
    for i in range(len(a_a)): 
        n+=1
        if n > 20:
            fig_n += 1
            fig =  plt.figure(fig_n)
            n = 1 
        fig =  plt.figure(fig_n)
        fig.add_subplot(4,5,n)   
        plt.plot(a_a[i], color = 'darkblue', label = 'A to A')
        plt.plot(b_i_same[i], color = 'grey', label = 'B to I Same Space')
        plt.plot(b_b[i], color = 'green', label = 'B to B')
        plt.plot(b_i_diff[i], color = 'darkred', label = 'B to I Diff Space')
        plt.plot(a_a_rev[i], color = 'lightblue', label = 'A to A Rev' )
        plt.axvline(task_change, color='k', linestyle=':')
    plt.legend(loc='upper center')

            
def run(data_HP):
    surprise_list_neurons_1_2, surprise_list_neurons_2_3,\
    surprise_list_neurons_b1_b2, surprise_list_neurons_b_i,\
    surprise_list_neurons_rev_a = remap_surprise(data_HP,1,'HP')
                 
    plot_surprise_n(surprise_list_neurons_1_2, surprise_list_neurons_2_3, surprise_list_neurons_b1_b2,\
                surprise_list_neurons_b_i, surprise_list_neurons_rev_a)  
    
    
    
    
    