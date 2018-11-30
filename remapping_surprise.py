#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:09:55 2018

@author: veronikasamborska
"""

import numpy as np
import ephys_beh_import as ep
import heatmap_aligned as ha
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.distributions import poisson

ephys_path = '/Users/veronikasamborska/Desktop/neurons'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
#ephys_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons'
##beh_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys'
#HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path)
#experiment_aligned_HP = ha.all_sessions_aligment(HP)
#experiment_aligned_PFC = ha.all_sessions_aligment(PFC)


def remapping_surprise(experiment, distribution):
    
    surprise_list =[]
    surprise_list_2 = []
    surprise_list_neurons_1_2 =[]
    surprise_list_neurons_2_3 = []    
    for i,session in enumerate(experiment):
        
        
        aligned_spikes= session.aligned_rates 
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape   
        
        # Trial indices  of choices 
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = ha.predictors_f(session)
        index_no_reward = np.where(reward ==0)
        t_out = session.t_out
      
        initiate_choice_t = session.target_times #Initiation and Choice Times
        
        ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
        ind_after_choice = ind_choice + 7 # 1 sec after choice
        spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
        mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =2)

        a_1 = np.where(predictor_A_Task_1 == 1)
        a_2 = np.where(predictor_A_Task_2 == 1)
        a_3 = np.where(predictor_A_Task_3 == 1)
        
        #a1_nR = [a for a in a_1[0] if a in index_no_reward[0]]
        #a2_nR = [a for a in a_2[0] if a in index_no_reward[0]]
        #a3_nR = [a for a in a_3[0] if a in index_no_reward[0]]

        baseline_mean_trial = np.mean(aligned_spikes, axis =2)
        baseline_mean_all_trials = np.mean(baseline_mean_trial, axis =0)
        std_all_trials = np.std(baseline_mean_trial, axis =1)
        
        choice_a1 = mean_spikes_around_choice[a_1]
        choice_a2 = mean_spikes_around_choice[a_2]
        choice_a3 = mean_spikes_around_choice[a_3]  
        
        choice_a1_mean  = np.mean(choice_a1, axis = 0)
        choice_a2_mean  = np.mean(choice_a2, axis = 0)
        choice_a3_mean  = np.mean(choice_a3, axis = 0)
         
        for neuron in range(n_neurons):
            trials_firing = mean_spikes_around_choice[:,neuron] 
            if choice_a1_mean[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or choice_a2_mean[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] \
            or choice_a3_mean[neuron] > baseline_mean_all_trials[neuron] + 3*std_all_trials[neuron] :     
       
                a1_fr = trials_firing[a_1] # Firing rates on poke A choices in Task 1
                a1_poisson = a1_fr.astype(int)
                a2_fr = trials_firing[a_2]
                a2_poisson = a2_fr.astype(int)
                a3_fr = trials_firing[a_3]
                a3_poisson = a3_fr.astype(int)
        
                a1_mean = np.nanmean(a1_fr )
                a1_poisson = np.nanmean(a1_poisson)
                a1_std = np.nanstd(a1_fr)
                a2_mean = np.nanmean(a2_fr)
                a2_poisson = np.nanmean(a2_poisson)
                a2_std = np.nanstd(a2_fr)
                a3_poisson = np.nanmean(a3_poisson)
                a3_mean = np.nanmean(a3_fr)
                
        
                a1_fr_last = a1_fr[-15:]
                a1_fr_last_poisson = a1_fr_last.astype(int)
                a2_fr_first = a2_fr[:15]
                a2_fr_first_poisson = a2_fr_first.astype(int)
                a2_fr_last = a2_fr[-15:]
                a2_fr_last_poisson = a2_fr_last.astype(int)
                a3_fr_first = a3_fr[:15]
                a3_fr_first_poisson = a3_fr_first.astype(int)
                
                if a1_mean > 0.1 and a2_mean > 0.1 and a3_mean > 0.1:
                    if distribution == 'Normal':
                        
                        surprise_a1 = -norm.logpdf(a1_fr_last,a1_mean, a1_std)
                        
                        surprise_a2 = -norm.logpdf(a2_fr_first,a1_mean, a1_std)
                        
                        surprise_a2_last = -norm.logpdf(a2_fr_last,a2_mean, a2_std)
                        
                        surprise_a3_first= -norm.logpdf(a3_fr_first,a2_mean, a2_std)
                            
                    elif distribution == 'Poisson':
                        
                        surprise_a1 = -poisson.logpmf(a1_fr_last_poisson, mu = a1_poisson)
                        
                        surprise_a2 = -poisson.logpmf(a2_fr_first_poisson, mu = a1_poisson)
                        
                        surprise_a2_last = -poisson.logpmf(a2_fr_last_poisson, mu = a2_poisson)
                        
                        surprise_a3_first= -poisson.logpmf(a3_fr_first_poisson, mu = a2_poisson)
        
                    surprise_array_t1_2 = np.concatenate([surprise_a1, surprise_a2])
                    
                    surprise_array_t2_3 = np.concatenate([surprise_a2_last,surprise_a3_first])
         
                if len(surprise_array_t1_2)>0 and len(surprise_array_t2_3)>0:
                    surprise_list_neurons_1_2.append(surprise_array_t1_2)
                    surprise_list_neurons_2_3.append(surprise_array_t2_3)

        surprise_list.append(surprise_list_neurons_1_2)
        surprise_list_2.append(surprise_list_neurons_2_3)
    
    surprise_list_neurons_1_2 = np.array(surprise_list_neurons_1_2)
    surprise_list_neurons_2_3 = np.array(surprise_list_neurons_2_3)
    

    mean_1_2 =np.nanmean(surprise_list_neurons_1_2, axis = 0)
    std_1_2 =np.nanstd(surprise_list_neurons_1_2, axis = 0)
    serr_1_2 = std_1_2/np.sqrt(len(surprise_list_neurons_1_2))
    mean_2_3 =np.nanmean(surprise_list_neurons_2_3, axis = 0)
    std_2_3 =np.nanstd(surprise_list_neurons_2_3, axis = 0)
    serr_2_3 = std_2_3/np.sqrt(len(surprise_list_neurons_2_3))

    x_pos = np.arange(len(mean_2_3))
    task_change = 15
    
    allmeans = mean_1_2 + mean_2_3/2
    allserr = serr_1_2 + serr_2_3/2

    figure()
    plot(x_pos,mean_1_2)
    plt.fill_between(x_pos, mean_1_2 -serr_1_2, mean_1_2 + serr_1_2, alpha=0.2)
    plt.axvline(task_change, color='k', linestyle=':')
    plt.title('Task 1 and 2')
    plt.ylabel('-log(p(X))')
    plt.xlabel('Trial #')

    figure()
    plot(x_pos,mean_2_3)
    plt.fill_between(x_pos, mean_2_3 -serr_2_3, mean_2_3 + serr_2_3, alpha=0.2)
    plt.axvline(task_change, color='k', linestyle=':')
    plt.title('Task 2 and 3')
    plt.ylabel('-log(p(X))')
    plt.xlabel('Trial #')
               
    figure()
    plot(x_pos,allmeans)
    plt.fill_between(x_pos, allmeans -allserr, allmeans + allserr, alpha=0.2)
    plt.axvline(task_change, color='k', linestyle=':')
    plt.title('Combined')
    plt.ylabel('-log(p(X))')
    plt.xlabel('Trial #')          
    

def plot_firing_rate_time_course(experiment):
    for session in experiment:
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward = ha.predictors_f(session)
        aligned_spikes= session.aligned_rates 
        n_neurons = aligned_spikes.shape[1]
        n_trials = aligned_spikes.shape[0]
        spikes_B_task_1 =aligned_spikes[np.where(predictor_B_Task_1 ==1)]
        spikes_A_task_1 =aligned_spikes[np.where(predictor_A_Task_1 ==1)]
        spikes_B_task_2 =aligned_spikes[np.where(predictor_B_Task_2 ==1)]
        spikes_A_task_2 =aligned_spikes[np.where(predictor_A_Task_2 ==1)]
        spikes_B_task_3 =aligned_spikes[np.where(predictor_B_Task_3 ==1)]
        spikes_A_task_3 =aligned_spikes[np.where(predictor_A_Task_3 ==1)]
        fig, axes = plt.subplots(figsize = (15,5), ncols = n_neurons , sharex=True, sharey = 'col')
        mean_spikes_B_task_1 = np.mean(spikes_B_task_1,axis = 0)
        mean_spikes_A_task_1 = np.mean(spikes_A_task_1,axis = 0)
        mean_spikes_B_task_2 = np.mean(spikes_B_task_2,axis = 0)
        mean_spikes_A_task_2 = np.mean(spikes_A_task_2,axis = 0)
        mean_spikes_B_task_3 = np.mean(spikes_B_task_3,axis = 0)
        mean_spikes_A_task_3 = np.mean(spikes_A_task_3,axis = 0)
        for neuron in range(n_neurons):
            axes[neuron].plot(mean_spikes_B_task_1[neuron], label = 'B Task 1')
            axes[neuron].plot(mean_spikes_A_task_1[neuron], label = 'A Task 1')
            axes[neuron].plot(mean_spikes_B_task_2[neuron], label = 'B Task 2')
            axes[neuron].plot(mean_spikes_A_task_2[neuron], label = 'A Task 2')
            axes[neuron].plot(mean_spikes_B_task_3[neuron], label = 'B Task 3')
            axes[neuron].plot(mean_spikes_A_task_3[neuron], label = 'A Task 3')
        axes[0].legend()
        plt.title('{}'.format(session.file_name))

    
