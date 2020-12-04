#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:03:19 2018

@author: behrenslab
"""

# =============================================================================
# Script for trial time alignmentt and plotting a v simple heatplot to order 
# neurons by another task
# =============================================================================

import numpy as np
import align_activity as aa
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import ephys_beh_import as ep
import pylab as plt

## Target times for aligned rates of non-forced trials 
def target_times_f(all_experiments):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
    trial_times_all_trials  = []
    j = 0

    for session in all_experiments:
        #j+=1

        #print(j)
         
        trials = session.trial_data['n_trials']
        
      
        itis_and_choices = [ev for ev in session.events if ev.name in 
                         [ 'free_reward_trial','a_forced_state', 'b_forced_state','choice_state','sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward','inter_trial_interval','init_trial']]
      
       
        init_cue_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if i < len(itis_and_choices)-2 and
                         itis_and_choices[i].name == 'init_trial' and
                         itis_and_choices[i+1].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i+2].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']])
        
        init_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if i < len(itis_and_choices)-1 and
                         itis_and_choices[i-1].name == 'init_trial' and
                         itis_and_choices[i].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i+1].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']])
       
        ch_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if i < len(itis_and_choices) and
                         itis_and_choices[i-2].name == 'init_trial' and
                         itis_and_choices[i-1].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']])
     
        inter_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if 
                        i < len(itis_and_choices) and
                         itis_and_choices[i-3].name == 'init_trial' and
                         itis_and_choices[i-2].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i-1].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward'] and itis_and_choices[i].name =='inter_trial_interval'])
     
        if len(inter_t)-1 == trials:
            ch_t = ch_t[:trials]
            inter_t = inter_t[:trials]
            init_t = init_t[:trials]
        if len(ch_t) !=trials:
            ch_t = inter_t[:trials]
            init_t = init_t[:trials]
        if len(init_cue_t) != trials:
            init_cue_t  = init_cue_t[:trials]
            
        
        trial_times = np.array([init_cue_t, init_t, ch_t, inter_t, inter_t+1750]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  =np.asarray(trial_times_all_trials)
    target_times = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times_all_trials[0],1),0))))    
        
    return target_times

      
## Target times for aligned rates of non-forced trials 
def all_sessions_aligment(experiment, all_experiments,  fs = 50):
   
    target_times  = target_times_f(all_experiments)
    experiment_aligned = []
    for session in experiment:
       
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        trials = session.trial_data['n_trials']
        
        itis_and_choices = [ev for ev in session.events if ev.name in 
                         [ 'free_reward_trial','a_forced_state', 'b_forced_state','choice_state','sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward','inter_trial_interval','init_trial']]
      
       
        init_cue_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if i < len(itis_and_choices)-2 and
                         itis_and_choices[i].name == 'init_trial' and
                         itis_and_choices[i+1].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i+2].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']])
        
        init_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if i < len(itis_and_choices)-1 and
                         itis_and_choices[i-1].name == 'init_trial' and
                         itis_and_choices[i].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i+1].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']])
       
        ch_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if i < len(itis_and_choices) and
                         itis_and_choices[i-2].name == 'init_trial' and
                         itis_and_choices[i-1].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']])
     
        inter_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if 
                        i < len(itis_and_choices) and
                         itis_and_choices[i-3].name == 'init_trial' and
                         itis_and_choices[i-2].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i-1].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward'] and itis_and_choices[i].name =='inter_trial_interval'])
     
        if len(inter_t)-1 == trials:
            ch_t = ch_t[:trials]
            inter_t = inter_t[:trials]
            init_t = init_t[:trials]
        if len(ch_t) !=trials:
            ch_t = ch_t[:trials]
            init_t = init_t[:trials]
        if len(init_cue_t) != trials:
            init_cue_t  = init_cue_t[:trials]
        
        trial_times = np.array([init_cue_t, init_t, ch_t,inter_t, inter_t+1750]).T
    
        if len(np.where((trial_times[:,2]- trial_times[:,3]) > 0)[0]) >1 :
         #   print(j)
            inter_t = np.array([ev.time for i, ev in enumerate(itis_and_choices) if 
                        i < len(itis_and_choices) and
                         itis_and_choices[i-3].name == 'init_trial' and
                         itis_and_choices[i-2].name in ['choice_state' ,'a_forced_state', 'b_forced_state'] and itis_and_choices[i-1].name in ['sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward'] and itis_and_choices[i].name =='inter_trial_interval'])
     
            if len(inter_t)-1 == trials:
                inter_t = inter_t[1:]

       
        trial_times = np.array([init_cue_t, init_t, ch_t,inter_t, inter_t+1750]).T
         
        aligned_rates, t_out, min_max_stretch = aa.align_activity(trial_times, target_times, spikes, fs = fs)
        session.aligned_rates = aligned_rates
        session.t_out = t_out
        session.target_times = target_times

        init_cue_t = init_cue_t[1:]
        init_t = init_t[1:]
        inter_t = inter_t[:-1]
        events_iti_to_cue =  []
        events_cue_to_init = []
        for i,ii in enumerate(init_t):
            events_iti_to_cue.append([ev for ev in session.events if ev.time in np.arange(inter_t[i],init_cue_t[i])])
        
        for i,ii in enumerate(init_t):
            events_cue_to_init.append([ev for ev in session.events if ev.time in np.arange(init_cue_t[i],init_t[i])])
            
        task = np.where(np.diff(session.trial_data['task']) ==1)[0]+1
        task = np.insert(task,0,0)
        a_pokes = session.trial_data['poke_A'][task]
        b_pokes = session.trial_data['poke_B'][task]
        
        
        ## Find when they make pokes into the prev choice port
        choice_count_iti_to_cue = []
        for i,ii in enumerate(events_iti_to_cue):
            if i < task[1]:
                choice_a = 'poke_'+str(a_pokes[0])
                choice_b = 'poke_'+str(b_pokes[0])
           
                choice_a_ex = 'poke_'+str(a_pokes[0])+'_out'
                choice_b_ex = 'poke_'+str(b_pokes[0])+'_out'
                ch = []
                for j in ii:
                    if j.name in [choice_a,choice_b, choice_a_ex,choice_b_ex]:
                        ch.append(j.name)

            if i > task[1] and i < task[2]:
                choice_a = 'poke_'+str(a_pokes[1])
                choice_b = 'poke_'+str(b_pokes[1])
               
                choice_a_ex = 'poke_'+str(a_pokes[1])+'_out'
                choice_b_ex = 'poke_'+str(b_pokes[1])+'_out'
                ch = []
                for j in ii:
                    if j.name in [choice_a,choice_b, choice_a_ex,choice_b_ex]:
                        ch.append(j.name)
            if i > task[2]:
                choice_a = 'poke_'+str(a_pokes[2])
                choice_b = 'poke_'+str(b_pokes[2])
               
                choice_a_ex = 'poke_'+str(a_pokes[2])+'_out'
                choice_b_ex = 'poke_'+str(b_pokes[2])+'_out'
                ch = []
                for j in ii:
                    if j.name in [choice_a,choice_b, choice_a_ex,choice_b_ex]:
                        ch.append(j.name)
            choice_count_iti_to_cue.append(ch)
            
        choice_re_entry_iti_to_cue = []
        for ii,i in enumerate(choice_count_iti_to_cue):
            if len(i) > 0:
                choice_re_entry_iti_to_cue.append(ii)
       
        choice_count_cue_to_init = []
        for i,ii in enumerate(events_cue_to_init):
            if i < task[1]:
                choice_a = 'poke_'+str(a_pokes[0])
                choice_b = 'poke_'+str(b_pokes[0])
           
                choice_a_ex = 'poke_'+str(a_pokes[0])+'_out'
                choice_b_ex = 'poke_'+str(b_pokes[0])+'_out'
                ch = []
                for j in ii:
                    if j.name in [choice_a,choice_b, choice_a_ex,choice_b_ex]:
                        ch.append(j.name)

            if i > task[1] and i < task[2]:
                choice_a = 'poke_'+str(a_pokes[1])
                choice_b = 'poke_'+str(b_pokes[1])
               
                choice_a_ex = 'poke_'+str(a_pokes[1])+'_out'
                choice_b_ex = 'poke_'+str(b_pokes[1])+'_out'
                ch = []
                for j in ii:
                    if j.name in [choice_a,choice_b, choice_a_ex,choice_b_ex]:
                        ch.append(j.name)
            if i > task[2]:
                choice_a = 'poke_'+str(a_pokes[2])
                choice_b = 'poke_'+str(b_pokes[2])
               
                choice_a_ex = 'poke_'+str(a_pokes[2])+'_out'
                choice_b_ex = 'poke_'+str(b_pokes[2])+'_out'
                ch = []
                for j in ii:
                    if j.name in [choice_a,choice_b, choice_a_ex,choice_b_ex]:
                        ch.append(j.name)
            choice_count_cue_to_init.append(ch)
            
        choice_re_entry_cue_to_init = []
        
        for ii,i in enumerate(choice_count_cue_to_init):
            if len(i) > 0:
                choice_re_entry_cue_to_init.append(ii)
                
        session.cue_init = np.asarray(choice_re_entry_cue_to_init)+1
        session.iti_cue = np.asarray(choice_re_entry_iti_to_cue)+1

        experiment_aligned.append(session)
       
        
    return experiment_aligned 


def heatplot_aligned(experiment_aligned): 
    all_clusters_task_1 = []
    all_clusters_task_2 = []
    for session in experiment_aligned:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        t_out = session.t_out
        initiate_choice_t = session.target_times 
        reward = initiate_choice_t[-2] +250
        cluster_list_task_1 = []
        cluster_list_task_2 = []
        aligned_rates = session.aligned_rates
        poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = ep.extract_choice_pokes(session)
        trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3 = ep.initiation_and_trial_end_timestamps(session)
        task_1 = len(trial_сhoice_state_task_1)
        task_2 = len(trial_сhoice_state_task_2)
        if poke_I == poke_I_task_2: 
            aligned_rates_task_1 = aligned_rates[:task_1]
            aligned_rates_task_2 = aligned_rates[:task_1+task_2]
        elif poke_I == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
        elif poke_I_task_2 == poke_I_task_3:
            aligned_rates_task_1 = aligned_rates[:task_1+task_2]
            aligned_rates_task_2 = aligned_rates[task_1+task_2:]
        unique_neurons  = np.unique(spikes[0])
        for i in range(len(unique_neurons)):
            mean_firing_rate_task_1  = np.mean(aligned_rates_task_1[:,i,:],0)
            mean_firing_rate_task_2  = np.mean(aligned_rates_task_2[:,i,:],0)
            cluster_list_task_1.append(mean_firing_rate_task_1) 
            cluster_list_task_2.append(mean_firing_rate_task_2)
        all_clusters_task_1.append(cluster_list_task_1[:])
        all_clusters_task_2.append(cluster_list_task_2[:])
    all_clusters_task_1 = np.array(all_clusters_task_1)
    all_clusters_task_2 = np.array(all_clusters_task_2)
    same_shape_task_1 = []
    same_shape_task_2 = []
    for i in all_clusters_task_1:
        for ii in i:
            same_shape_task_1.append(ii)
    for i in all_clusters_task_2:
        for ii in i:
            same_shape_task_2.append(ii)
    same_shape_task_1 = np.array(same_shape_task_1)
    same_shape_task_2 = np.array(same_shape_task_2)
    peak_inds = np.argmax(same_shape_task_1,1)
    ordering = np.argsort(peak_inds)
    activity_sorted = same_shape_task_1[ordering,:]
    normed = np.log(not_normed)
    
    #not_normed = same_shape_task_1[ordering,:]

    norm_activity_sorted = (activity_sorted - np.min(activity_sorted,1)[:, None]) / (np.max(activity_sorted,1)[:, None] - np.min(activity_sorted,1)[:, None])
    where_are_Nans = np.isnan(norm_activity_sorted)
    norm_activity_sorted[where_are_Nans] = 0
    plt.imshow(norm_activity_sorted, aspect='auto')  
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward)).argmin()
    
    plt.xticks([ind_init, ind_choice, ind_reward], ('I', 'C', 'R'))
    plt.title('HP')
    plt.colorbar()
    



