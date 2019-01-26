#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:47:36 2019

@author: veronikasamborska
"""

import ephys_beh_import as ep
import heatmap_aligned as ha 
import regressions as re
import neuron_firing_all_pokes as ne
import copy 
import numpy as np
import align_activity as aa


## Target times for aligned rates of forced trials 
def target_times_forced(experiment_forced):
    # Trial times is array of reference point times for each trial. Shape: [n_trials, n_ref_points]
    # Here we are using [init-1000, init, choice, choice+1000]    
    # target_times is the reference times to warp all trials to. Shape: [n_ref_points]
    # Here we are finding the median timings for a whole experiment 
    trial_times_all_trials  = []
    for session in experiment_forced:
        init_times = np.concatenate((session.times['b_forced_state'], session.times['a_forced_state']), axis = 0)
        init_times = sorted(init_times)
        init_times = np.asarray(init_times)
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['a_forced_state','b_forced_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'a_forced_state' or inits_and_choices[i-1].name == 'b_forced_state'])
        
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        trial_times_all_trials.append(trial_times)

    trial_times_all_trials  =np.asarray(trial_times_all_trials)
    target_times_forced = np.hstack(([0], np.cumsum(np.median(np.diff(trial_times_all_trials[0],1),0))))    
        
    return target_times_forced

## Target times for aligned rates of forced trials 
def all_sessions_aligment_forced(experiment_forced):
    target_times_forced_trials  = target_times_forced(experiment_forced)
    experiment_aligned_forced = []
    for session in experiment_forced:
        spikes = session.ephys
        spikes = spikes[:,~np.isnan(spikes[1,:])] 
        init_times = np.concatenate((session.times['b_forced_state'], session.times['a_forced_state']), axis = 0)
        init_times = sorted(init_times)
        init_times = np.asarray(init_times)
        inits_and_choices = [ev for ev in session.events if ev.name in 
                        ['a_forced_state','b_forced_state', 'sound_a_reward', 'sound_b_reward',
                         'sound_a_no_reward','sound_b_no_reward']]
        
        choice_times = np.array([ev.time for i, ev in enumerate(inits_and_choices) if 
                             i>0 and inits_and_choices[i-1].name == 'a_forced_state' or inits_and_choices[i-1].name == 'b_forced_state'])
            
        if len(choice_times) != len(init_times):
            init_times  =(init_times[:len(choice_times)])
            
        trial_times = np.array([init_times-1000, init_times, choice_times, choice_times+1000]).T
        aligned_rates_forced, t_out, min_max_stretch = aa.align_activity(trial_times, target_times_forced_trials, spikes)
        session.aligned_rates_forced = aligned_rates_forced
        session.t_out = t_out
        session.target_times_forced_trials = target_times_forced_trials
        experiment_aligned_forced.append(session)
        
    return experiment_aligned_forced 

def state_indices_forced(session):
    forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]
    state = session.trial_data['state']
    state_forced = state[forced_array]
    task = session.trial_data['task']
    task_forced = task[forced_array]
    task_1 = np.where(task_forced == 1)[0]
    task_2 = np.where(task_forced == 2)[0] 
    
    
    #Task 1 
    state_1 = state_forced[:len(task_1)]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    
    # Task 2
    state_2 = state_forced[len(task_1): (len(task_1) +len(task_2))]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_a_good = state_t2_a_good+len(task_1)
    state_t2_b_good = np.where(state_2 == 0)[0]
    state_t2_b_good = state_t2_b_good+len(task_1)


    #Task 3 
    state_3 = state_forced[len(task_1) + len(task_2):]
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]
    state_t3_a_good =state_t3_a_good + (len(task_1) + len(task_2))
    state_t3_b_good = state_t3_b_good + (len(task_1) + len(task_2))
    
    return state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good


def extract_correct_forced_states(session):
    events = session.events
    forced_trials = []
    for event in events:
        if 'a_forced_state' in event:
            forced_trials.append(1)
        elif 'b_forced_state' in event:
            forced_trials.append(0)
    forced_trials = np.asarray(forced_trials)
    return forced_trials
                        
                        
def predictors_forced(session):
    forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]
    
    task = session.trial_data['task']
    task_forced = task[forced_array]   
    outcomes_all = session.trial_data['outcomes'] 
    reward = outcomes_all[forced_array]
    choice_forced = extract_correct_forced_states(session)
    n_trials = len(choice_forced)
    task_1 = np.where(task_forced == 1)[0]
    task_2 = np.where(task_forced == 2)[0] 
    poke_A = session.trial_data['poke_A']
    poke_B = session.trial_data['poke_B']
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = ep.extract_choice_pokes(session)
    
    #Task 1 
    choices_a = np.where(choice_forced == 1)
    choices_b = np.where(choice_forced == 0)
    
    predictor_a = np.zeros([1,n_trials])
    predictor_a[0][choices_a[0]] = 1
    predictor_b = np.zeros([1,n_trials])
    predictor_b[0][choices_b[0]] = 1
    if len(reward)!= len(predictor_a[0]):
        reward = np.append(reward,0)
        
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)
    
    predictor_a_1 = copy.copy(predictor_a)
    predictor_a_1[0][len(task_1):] = 0
    predictor_a_2 =  copy.copy(predictor_a)
    predictor_a_2[0][:len(task_1)] = 0
    predictor_a_2[0][len(task_1)+len(task_2):] = 0 
    predictor_a_3 =  copy.copy(predictor_a)
    predictor_a_3[0][:len(task_1)+len(task_2)] = 0 
    
    predictor_b_1 =  copy.copy(predictor_b)
    predictor_b_1[0][len(task_1):] = 0
    predictor_b_2 = copy.copy(predictor_b)
    predictor_b_2[0][:len(task_1)] = 0
    predictor_b_2[0][len(task_1)+len(task_2):] = 0 
    predictor_b_3 = copy.copy(predictor_b)
    predictor_b_3[0][:len(task_1)+len(task_2)] = 0
    
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = state_indices_forced(session)
    
    predictor_state_a = np.zeros([n_trials])
    predictor_state_b = np.zeros([n_trials])
    predictor_state_a_1_good = copy.copy(predictor_state_a)
    predictor_state_a_2_good = copy.copy(predictor_state_a)
    predictor_state_a_3_good = copy.copy(predictor_state_a)
    
    predictor_state_b_1_good = copy.copy(predictor_state_b)
    predictor_state_b_2_good = copy.copy(predictor_state_b)
    predictor_state_b_3_good = copy.copy(predictor_state_b)

    predictor_state_a_1_good[state_a_good] = 1
    predictor_state_b_1_good[state_b_good] = 1
    predictor_state_a_2_good[state_t2_a_good] = 1
    predictor_state_b_2_good[state_t2_b_good] = 1
    predictor_state_a_3_good[state_t3_a_good] = 1
    predictor_state_b_3_good[state_t3_b_good] = 1
    
    if poke_A1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)

    elif poke_A1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_A1_B2_A3 == True: 
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)


    elif poke_A1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_a_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_b_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_a_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_B1_B2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_B1_A2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)
        
    elif poke_B1_A2_B3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_a_2[0])
        predictor_A_Task_3  = copy.copy(predictor_b_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_b_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_a_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_a_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_b_3_good)

    elif poke_B1_B2_A3 == True:
        predictor_A_Task_1 = copy.copy(predictor_b_1[0])
        predictor_A_Task_2 = copy.copy(predictor_b_2[0])
        predictor_A_Task_3  = copy.copy(predictor_a_3[0])
        predictor_B_Task_1 =  copy.copy(predictor_a_1[0])
        predictor_B_Task_2 =  copy.copy(predictor_a_2[0])
        predictor_B_Task_3 =  copy.copy(predictor_b_3[0])
        predictor_a_good_task_1 = copy.copy(predictor_state_b_1_good)
        predictor_a_good_task_2 = copy.copy(predictor_state_b_2_good)
        predictor_a_good_task_3 = copy.copy(predictor_state_a_3_good)
    
    return predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3
    
    
