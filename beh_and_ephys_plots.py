#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:13:18 2018

@author: behrenslab
"""
import funcs as fu
import data_import as di
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from os import path

#Import Ephys and PyControl Data
#ephys_session = fu.load_data('m481_2018-06-20_19-09-08','/Users/veronikasamborska/Desktop/Ephys 3 Tasks Processed Spikes/m481/','/',True)
#beh_session = di.Session('/Users/veronikasamborska/Desktop/data_3_tasks_ephys/m481-2018-06-20-190858.txt')


def change_block_firing_rates(ephys_session, beh_session):
    forced_trials = beh_session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0] 
    state = beh_session.trial_data['state']
    task = beh_session.trial_data['task']
    task_non_forced = task[non_forced_array]
    # Task 1 
    task_1 = np.where(task_non_forced == 1)[0]
    state_1 = state[task_1]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    
    # Task 2
    task_2 = np.where(task_non_forced == 2)[0]
    state_2 = state[task_2]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_b_good = np.where(state_2 == 0)[0]
    print(len(state_t2_a_good))
    print(len(state_t2_b_good))
    #task_2_non_forced_length = len(task_2)
    
    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    task_1_choice_list = pyControl_choice[:len(task_1)]
    task_2_choice_list = pyControl_choice[len(task_1): (len(task_1) + len(task_2))]
    clusters = ephys_session['spike_cluster'].unique()
    #session_length = np.nanmax(ephys_session['time']) - np.nanmin(ephys_session['time'])
   
    clusters = clusters[3:5]
   # print(clusters)
    fig, axes = plt.subplots(figsize=(50,5), ncols=1, nrows=2, sharex=True)
    
    for i,cluster in enumerate(clusters): 
        spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
        spikes_times = np.array(spikes['time'])
        spikes_count = 0
        firing_rate_trial_list = np.array([])

        for choice in task_1_choice_list:
            trial_start = choice - 3000
            trial_end = choice + 3000
            spikes_ind = spikes_times[(spikes_times >= trial_start) & (spikes_times<= trial_end)]
            spikes_count = np.count_nonzero(~np.isnan(spikes_ind))
            firing_rate_trial = spikes_count/6000
            firing_rate_trial_list = np.append(firing_rate_trial_list, firing_rate_trial)
            
        mean_firing_rate =np.mean(firing_rate_trial_list)
        std_firing_rate = np.std(firing_rate_trial_list)
        zscore= (firing_rate_trial_list- mean_firing_rate)/std_firing_rate
        axes[i].bar(np.arange(len(zscore)),zscore, color = 'black')
        axes[i].bar(state_t2_b_good,np.ones(len(state_t2_b_good)),alpha = 0.3, color = 'olive', label = 'State B Good, A Bad' )
        
        axes[i].set_title('{}'.format(cluster))
        axes[i].set(xlabel ='Trial #')
        axes[i].set(ylabel ='Z-score Firing Rate #')
        axes[i].legend()

    
    
def histogram_raster_plot_poke_aligned(ephys_session, beh_session,outpath, plot):
    #plot = 'Choice' #Specify which plot  #'Choice if want a plot to align at initiation, 'Poke' if want a plot aligned at A/B poke
    
    outpath = '/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Processed Spikes/m481'
    sns.set()
    forced_trials = beh_session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    configuration = beh_session.trial_data['configuration_i']


    task = beh_session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    
    task = beh_session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    #v.state True A is good, False B is good 
    state = beh_session.trial_data['state']
    state_non_forced = state[non_forced_array]
    poke_A = 'poke_'+str(beh_session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(beh_session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(beh_session.trial_data['poke_A'][task_3_change[0]])
    poke_A_exit = 'poke_'+str(beh_session.trial_data['poke_A'][0]) + '_out'
    poke_B = 'poke_'+str(beh_session.trial_data['poke_B'][0])
    poke_B_exit = 'poke_'+str(beh_session.trial_data['poke_B'][0]) + '_out'
    poke_B_task_2  = 'poke_'+str(beh_session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(beh_session.trial_data['poke_B'][task_3_change[0]])
    #Trial Initiation Timestamps

    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)


    pyControl_forced_trials = [event.time for event in beh_session.events if event.name in ['a_forced_state','b_forced_state']]

    #Rewarded vs Unrewarded Timestamps 
    pyControl_b_poke_rewarded_cue = [event.time for event in beh_session.events if event.name in ['sound_b_reward']]
    pyControl_b_poke_non_rewarded_cue = [event.time for event in beh_session.events if event.name in ['sound_b_no_reward']]
    pyControl_b_poke_rewarded_cue =np.array(pyControl_b_poke_rewarded_cue)
    pyControl_b_poke_non_rewarded_cue = np.array(pyControl_b_poke_non_rewarded_cue)
    pyControl_a_poke_rewarded_cue= [event.time for event in beh_session.events if event.name in ['sound_a_reward']]
    pyControl_a_poke_rewarded_cue = np.array(pyControl_a_poke_rewarded_cue)
    pyControl_a_poke_non_rewarded_cue= [event.time for event in beh_session.events if event.name in ['sound_a_no_reward']]
    pyControl_a_poke_non_rewarded_cue =np.array(pyControl_a_poke_rewarded_cue)

    #Poke A and Poke B Timestamps 
    pyControl_a_poke_entry = [event.time for event in beh_session.events if event.name in [poke_A,poke_A_task_2,poke_A_task_3]]
    pyControl_a_poke_exit = [event.time for event in beh_session.events if event.name in [poke_A_exit]]
    pyControl_b_poke_entry = [event.time for event in beh_session.events if event.name in [poke_B,poke_B_task_2,poke_B_task_3 ]]

    pyControl_b_poke_exit = [event.time for event in beh_session.events if event.name in [poke_B_exit]]
    #ITI Timestamps 
    pyControl_end_trial = [event.time for event in beh_session.events if event.name in ['inter_trial_interval']][2:] #first two ITIs are free rewards
    pyControl_end_trial = np.array(pyControl_end_trial)

    # Pokes A, B and I 
    a_pokes = np.unique(beh_session.trial_data['poke_A'])
    print('These are A pokes')
    print(a_pokes)
    b_pokes = np.unique(beh_session.trial_data['poke_B'])
    print('These are B pokes')
    print(b_pokes)
    i_pokes = np.unique(configuration)
    all_pokes = np.concatenate([a_pokes, b_pokes, i_pokes])
    all_pokes = np.unique(all_pokes)


    #Events for Pokes Irrespective of Meaning
    pokes = {}
    for i, poke in enumerate(all_pokes):
        pokes[poke] = [event.time for event in beh_session.events if event.name in ['poke_'+str(all_pokes[i])]]
    

    events_and_times = [[event.name, event.time] for event in beh_session.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]


    poke_B_list = []
    poke_A_list = []
    choice_state = False 
    choice_state_count = 0
    
    for event in events_and_times:
        if 'choice_state' in event:
            choice_state_count +=1 
            choice_state = True   
        if choice_state_count <= len(task_1):
            if poke_B in event: 
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False
        elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
            if poke_B_task_2 in event:
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A_task_2 in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False     
        elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
            if poke_B_task_3 in event:
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A_task_3 in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False
    #Task 1 

    state_1 = state_non_forced[:len(task_1)]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    
    # Task 2
   
    state_2 = state_non_forced[len(task_1): (len(task_1) +len(task_2))]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_b_good = np.where(state_2 == 0)[0]

    #Task 3 Time Events
    
    state_3 = state_non_forced[len(task_1) + len(task_2):]
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]

    #For Choice State Calculations
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]

    trial_сhoice_state_task_1 = pyControl_choice[:len(task_1)]
    trial_сhoice_state_task_2 = pyControl_choice[len(task_1):(len(task_1) +len(task_2))]
    trial_сhoice_state_task_3 = pyControl_choice[len(task_1) + len(task_2):]

    trial_сhoice_state_task_1_a_good = trial_сhoice_state_task_1[state_a_good]
    trial_сhoice_state_task_2_a_good = trial_сhoice_state_task_2[state_t2_a_good]
    trial_сhoice_state_task_3_a_good = trial_сhoice_state_task_3[state_t3_a_good]

    trial_сhoice_state_task_1_b_good = trial_сhoice_state_task_1[state_b_good]
    trial_сhoice_state_task_2_b_good = trial_сhoice_state_task_2[state_t2_b_good]
    trial_сhoice_state_task_3_b_good = trial_сhoice_state_task_3[state_t3_b_good]
    
    task_1_end_trial = np.where(task == 1)[0]
    task_2_end_trial = np.where(task == 2)[0]
    pyControl_end_trial_1 = pyControl_end_trial[:len(task_1_end_trial)]
    pyControl_end_trial_2 =pyControl_end_trial[len(task_1_end_trial)+2:(len(task_1_end_trial)+len(task_2_end_trial)+2)]
    pyControl_end_trial_3 = pyControl_end_trial[len(task_1_end_trial)+len(task_2_end_trial)+4:]
    pyControl_end_trial =   np.concatenate([pyControl_end_trial_1, pyControl_end_trial_2,pyControl_end_trial_3])
    #pyControl_end_trial =   np.concatenate(pyControl_end_trial,pyControl_end_trial_3)

    
    #For ITI Calculations
    ITI_non_forced = pyControl_end_trial[non_forced_array]   
    ITI_task_1 = ITI_non_forced[:len(task_1)]#[2:]
    ITI_task_1_a_good = ITI_task_1[state_a_good]
    ITI_task_1_b_good = ITI_task_1[state_b_good]
    
    ITI_task_2 = ITI_non_forced[(len(task_1)):(len(task_1)+len(task_2))]
    ITI_task_2_a_good  = ITI_task_2[state_t2_a_good]
    ITI_task_2_b_good =ITI_task_2[state_t2_b_good]

    ITI_task_3 = ITI_non_forced[len(task_1) + len(task_2):]
    ITI_task_3_a_good  = ITI_task_3[state_t3_a_good]
    ITI_task_3_b_good  = ITI_task_3[state_t3_b_good]

    print(ITI_task_3_b_good)
    print(trial_сhoice_state_task_3_b_good)
  
    # Task one
    entry_a_good_list = []
    a_good_choice_time_task_1 = []
    entry_b_bad_list = []
    b_bad_choice_time_task_1 = []
    entry_a_bad_list = []
    b_good_choice_time_task_1 = []
    entry_b_good_list = []
    a_bad_choice_time_task_1 = []

    #np.save('/Users/veronikasamborska/Desktop/poke_A_list.npy',poke_A_list)
    for start_trial,end_trial in zip(trial_сhoice_state_task_1_b_good, ITI_task_1_b_good):
        for entry_a in poke_A_list:
            if (entry_a >= start_trial and entry_a <= end_trial):
                entry_a_bad_list.append(entry_a)
                a_bad_choice_time_task_1.append(start_trial)
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial and entry_b <= end_trial):
                entry_b_good_list.append(entry_b)
                b_good_choice_time_task_1.append(start_trial)


    for start_trial_a_good,end_trial_a_good in zip(trial_сhoice_state_task_1_a_good, ITI_task_1_a_good):
        for entry in poke_A_list:
      
            if (entry >= start_trial_a_good and entry <= end_trial_a_good):
                entry_a_good_list.append(entry)
                a_good_choice_time_task_1.append(start_trial_a_good)
                
        for entry_b_bad in poke_B_list: 
            if (entry_b_bad >= start_trial_a_good and entry_b_bad <= end_trial_a_good):
                entry_b_bad_list.append(entry_b_bad)
                b_bad_choice_time_task_1.append(start_trial_a_good)
        

    entry_a_bad_list = np.array(entry_a_bad_list)
    entry_b_good_list = np.array(entry_b_good_list)
    entry_a_good_list = np.array(entry_a_good_list)
    entry_b_bad_list = np.array(entry_b_bad_list)
    

    #Task two  
    entry_a_good_task_2_list = []
    a_good_choice_time_task_2 = []
    entry_b_bad_list_task_2 = []
    b_bad_choice_time_task_2 = []
    entry_a_bad_task_2_list = []
    b_good_choice_time_task_2 = []
    entry_b_good_list_task_2 = []
    a_bad_choice_time_task_2= []

   
    for start_trial_task_2,end_trial_task_2 in zip(trial_сhoice_state_task_2_b_good, ITI_task_2_b_good):
        for entry in poke_A_list:
            if (entry >= start_trial_task_2 and entry <= end_trial_task_2):
                entry_a_bad_task_2_list.append(entry)           
                a_bad_choice_time_task_2.append(start_trial_task_2)
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial_task_2 and entry_b <= end_trial_task_2):
                entry_b_good_list_task_2.append(entry_b)
                b_good_choice_time_task_2.append(start_trial_task_2)
            
  
            
    for start_trial_task_2,end_trial_task_2 in zip(trial_сhoice_state_task_2_a_good, ITI_task_2_a_good):
        for entry in poke_A_list:
            if (entry >= start_trial_task_2 and entry <= end_trial_task_2):
                entry_a_good_task_2_list.append(entry)
                a_good_choice_time_task_2.append(start_trial_task_2)
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial_task_2 and entry_b <= end_trial_task_2):
                entry_b_bad_list_task_2.append(entry_b)
                b_bad_choice_time_task_2.append(start_trial_task_2)
    
    entry_b_good_list_task_2 = np.array(entry_b_good_list_task_2)
    entry_a_bad_task_2_list = np.array(entry_a_bad_task_2_list)
    entry_a_good_task_2_list = np.array(entry_a_good_task_2_list)
    entry_b_bad_list_task_2 = np.array(entry_b_bad_list_task_2)     
    
    #Task three         
    entry_a_good_task_3_list = []
    out_a_good_list_task_3 = []
    a_good_choice_time_task_3 = []
    entry_b_bad_list_task_3 = []
    out_b_bad_list_task_3 = []
    b_bad_choice_time_task_3 = []
    
    entry_b_good_list_task_3 = []
    out_b_good_list_task_3 = []
    b_good_choice_time_task_3 = []
    entry_a_bad_task_3_list = []
    out_a_bad_list_task_3 = []
    a_bad_choice_time_task_3 = []
              
    for start_trial_task_3,end_trial_task_3 in zip(trial_сhoice_state_task_3_b_good, ITI_task_3_b_good):
        for entry in poke_A_list:
            if (entry >= start_trial_task_3 and entry <= end_trial_task_3):
                entry_a_bad_task_3_list.append(entry)
                a_bad_choice_time_task_3.append(start_trial_task_3)
        for out in pyControl_a_poke_exit:
            if (out >= start_trial_task_3 and out <= end_trial_task_3):
                out_a_bad_list_task_3.append(out)
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial_task_3 and entry_b <= end_trial_task_3):
                entry_b_good_list_task_3.append(entry_b)
                b_good_choice_time_task_3.append(start_trial_task_3)
        for out_b in pyControl_b_poke_exit:
            if (out_b >= start_trial_task_3 and out_b <= end_trial_task_3):
                out_b_good_list_task_3.append(out_b)
               
    for start_trial_task_3,end_trial_task_3 in zip(trial_сhoice_state_task_3_a_good, ITI_task_3_a_good):
        for entry in poke_A_list:
            if (entry >= start_trial_task_3 and entry <= end_trial_task_3):
                entry_a_good_task_3_list.append(entry)
                a_good_choice_time_task_3.append(start_trial_task_3)
        for out in pyControl_a_poke_exit:
            if (out >= start_trial_task_3 and out <= end_trial_task_3):
                out_a_good_list_task_3.append(out)
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial_task_3 and entry_b <= end_trial_task_3):
                entry_b_bad_list_task_3.append(entry_b)
                b_bad_choice_time_task_3.append(start_trial_task_3)
        for out_b in pyControl_b_poke_exit:
            if (out_b >= start_trial_task_3 and out_b <= end_trial_task_3):
                out_b_bad_list_task_3.append(out_b)
    
    entry_b_good_list_task_3 = np.array(entry_b_good_list_task_3)
    entry_a_bad_task_3_list = np.array(entry_a_bad_task_3_list)
    entry_a_good_task_3_list =  np.array(entry_a_good_task_3_list)
    entry_b_bad_list_task_3 = np.array(entry_b_bad_list_task_3)     

    
    a_good_choice_time_task_3 = np.array(a_good_choice_time_task_3)
    a_good_choice_time_task_3 = np.unique(a_good_choice_time_task_3)
    a_good_choice_time_task_2 = np.array(a_good_choice_time_task_2)
    a_good_choice_time_task_2 = np.unique(a_good_choice_time_task_2)
    a_good_choice_time_task_1 = np.array(a_good_choice_time_task_1)
    a_good_choice_time_task_1 = np.unique(a_good_choice_time_task_1)
    a_bad_choice_time_task_3 = np.array(a_bad_choice_time_task_3)
    a_bad_choice_time_task_3 = np.unique(a_bad_choice_time_task_3)
    a_bad_choice_time_task_2 = np.array(a_bad_choice_time_task_2)
    a_bad_choice_time_task_2 = np.unique(a_bad_choice_time_task_2)
    a_bad_choice_time_task_1 = np.array(a_bad_choice_time_task_1)
    a_bad_choice_time_task_1 = np.unique(a_bad_choice_time_task_1)
    
    b_bad_choice_time_task_3 = np.array(b_bad_choice_time_task_3)
    b_bad_choice_time_task_3 = np.unique(b_bad_choice_time_task_3)
    b_bad_choice_time_task_2 = np.array(b_bad_choice_time_task_2)
    b_bad_choice_time_task_2 = np.unique(b_bad_choice_time_task_2)
    b_bad_choice_time_task_1 = np.array(b_bad_choice_time_task_1)
    b_bad_choice_time_task_1 = np.unique(b_bad_choice_time_task_1)
    b_good_choice_time_task_3 = np.array(b_good_choice_time_task_3)
    b_good_choice_time_task_3 = np.unique(b_good_choice_time_task_3)
    b_good_choice_time_task_2 = np.array(b_good_choice_time_task_2)
    b_good_choice_time_task_2 = np.unique(b_good_choice_time_task_2)
    b_good_choice_time_task_1 = np.array(b_good_choice_time_task_1)
    b_good_choice_time_task_1 = np.unique(b_good_choice_time_task_1)
    
    if plot == 'Choice':
        aligned_a_good_task_1 = a_good_choice_time_task_1
        aligned_a_bad_task_1 = a_bad_choice_time_task_1
        aligned_b_good_task_1 = b_good_choice_time_task_1
        aligned_b_bad_task_1 = b_bad_choice_time_task_1
        
        aligned_a_good_task_2 = a_good_choice_time_task_2
        aligned_a_bad_task_2 = a_bad_choice_time_task_2
        aligned_b_good_task_2 = b_good_choice_time_task_2
        aligned_b_bad_task_2 =b_bad_choice_time_task_2
        
        aligned_a_good_task_3 = a_good_choice_time_task_3
        aligned_a_bad_task_3 = a_bad_choice_time_task_3
        aligned_b_good_task_3 = b_good_choice_time_task_3
        aligned_b_bad_task_3 = b_bad_choice_time_task_3
        
    elif plot == 'Poke':
        aligned_a_good_task_1 = entry_a_good_list
        aligned_a_bad_task_1 = entry_a_bad_list
        aligned_b_good_task_1 = entry_b_good_list
        aligned_b_bad_task_1 = entry_b_bad_list
        
        aligned_a_good_task_2 = entry_a_good_task_2_list
        aligned_a_bad_task_2 = entry_a_bad_task_2_list
        aligned_b_good_task_2 = entry_b_good_list_task_2
        aligned_b_bad_task_2 = entry_b_bad_list_task_2
        
        aligned_a_good_task_3 = entry_a_good_task_3_list
        aligned_a_bad_task_3 = entry_a_bad_task_3_list
        aligned_b_good_task_3 = entry_b_good_list_task_3
        aligned_b_bad_task_3 = entry_b_bad_list_task_3
        

    
    
    clusters = ephys_session['spike_cluster'].unique()

    
    # Different figures for subplotting neurons in groups of 8 
    
    fig_clusters_8, ax_8 = plt.subplots(figsize = (50,5), ncols = 9, nrows = 3, sharex=True)
    fig_raster_8, axes_8 = plt.subplots(figsize=(50,5), ncols = 9, nrows = 3, sharex=True)
    
    fig_clusters_16, ax_16 = plt.subplots(figsize = (50,5), ncols = 9, nrows = 3, sharex=True)
    fig_raster_16, axes_16= plt.subplots(figsize=(50,5), ncols = 9, nrows = 3, sharex=True)
    
    fig_clusters_24, ax_24 = plt.subplots(figsize = (50,5), ncols = 9, nrows = 3, sharex=True)
    fig_raster_24, axes_24 = plt.subplots(figsize = (50,5), ncols = 9, nrows = 3, sharex=True)
    
    fig_clusters_36, ax_36 = plt.subplots(figsize = (50,5), ncols = 9, nrows = 3, sharex=True)
    fig_raster_36, axes_36 = plt.subplots(figsize = (50,5), ncols = 9, nrows = 3, sharex=True)
    
    
    group_1 = 9
    group_2 = 18
    group_3 = 27
    group_4 = 36
   
    for i,cluster in enumerate(clusters): 
        fig_clusters_8 = pl.figure(1)
        fig_raster_8 = pl.figure(2)
        fig_clusters_16 = pl.figure(3)
        fig_raster_16 = pl.figure(4)
        fig_clusters_24 = pl.figure(5)
        fig_raster_24 = pl.figure(6)
        fig_clusters_36 = pl.figure(7)
        fig_raster_36 =pl.figure(8)

        spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
        spikes_times = np.array(spikes['time'])
        
        all_spikes_raster_plot_task_1 = []
        all_spikes_raster_plot_task_2 = []
        
        spikes_to_save_a_bad_task_1 = 0
        spikes_to_plot_a_bad_task_1 = np.array([])
        spikes_to_save_a_good_task_1 = 0
        spikes_to_plot_a_good_task_1 = np.array([])
        spikes_to_save_b_good_task_1 = 0 
        spikes_to_plot_b_good_task_1 = np.array([])
        spikes_to_save_b_bad_task_1 = 0
        spikes_to_plot_b_bad_task_1 = np.array([])
        spikes_to_save_a_bad_task_2 = 0
        spikes_to_plot_a_bad_task_2 = np.array([])
        spikes_to_save_a_good_task_2 = 0
        spikes_to_plot_a_good_task_2 = np.array([])
        spikes_to_save_b_bad_task_2 = 0
        spikes_to_plot_b_bad_task_2 = np.array([])
        spikes_to_save_b_good_task_2 = 0
        spikes_to_plot_b_good_task_2 = np.array([])
        
        spikes_to_save_a_bad_task_3 = 0
        spikes_to_plot_a_bad_task_3 = np.array([])
        spikes_to_save_a_good_task_3 = 0
        spikes_to_plot_a_good_task_3 = np.array([])
        spikes_to_save_b_bad_task_3 = 0
        spikes_to_plot_b_bad_task_3 = np.array([])
        spikes_to_save_b_good_task_3 = 0
        spikes_to_plot_b_good_task_3 = np.array([])
     
        spikes_to_plot_a_bad_task_1_raster = []
        spikes_to_plot_a_good_task_1_raster = []
        spikes_to_plot_b_good_task_1_raster = []
        spikes_to_plot_b_bad_task_1_raster = []
        spikes_to_plot_b_good_task_2_raster = []
        spikes_to_plot_a_bad_task_2_raster = []
        spikes_to_plot_a_good_task_2_raster = []
        spikes_to_plot_b_bad_task_2_raster = []
        spikes_to_plot_b_good_task_3_raster = []
        spikes_to_plot_a_bad_task_3_raster = []
        spikes_to_plot_a_good_task_3_raster = []
        spikes_to_plot_b_bad_task_3_raster = []
    

        
    ############### Task 1 
        for a_bad_task_1 in aligned_a_bad_task_1:
            period_min_a_bad_task_1 = a_bad_task_1 - 3000
            period_max_a_bad_task_1 = a_bad_task_1 + 3000
            spikes_ind_a_bad_task_1 = spikes_times[(spikes_times >= period_min_a_bad_task_1) & (spikes_times<= period_max_a_bad_task_1)]

            spikes_to_save_a_bad_task_1 = (spikes_ind_a_bad_task_1 - a_bad_task_1)
            
            spikes_to_plot_a_bad_task_1= np.append(spikes_to_plot_a_bad_task_1,spikes_to_save_a_bad_task_1)
            spikes_to_plot_a_bad_task_1_raster.append(spikes_to_save_a_bad_task_1)
            
    
        for a_good_task_1 in aligned_a_good_task_1:
            period_min_a_bad = a_good_task_1 - 3000
            period_max_a_bad = a_good_task_1 + 3000
            spikes_ind_a_good = spikes_times[(spikes_times >= period_min_a_bad) & (spikes_times<=period_max_a_bad)]
  
            spikes_to_save_a_good_task_1 = (spikes_ind_a_good - a_good_task_1)
            spikes_to_plot_a_good_task_1= np.append(spikes_to_plot_a_good_task_1,spikes_to_save_a_good_task_1) 
            spikes_to_plot_a_good_task_1_raster.append(spikes_to_save_a_good_task_1)
            
        for b_good_task_1 in aligned_b_good_task_1:
       
            period_min_b_good = b_good_task_1 - 3000
            period_max_b_good = b_good_task_1 + 3000
            spikes_ind_b_good = spikes_times[(spikes_times >= period_min_b_good) & (spikes_times<=period_max_b_good)]

            spikes_to_save_b_good_task_1 = (spikes_ind_b_good - b_good_task_1)
            spikes_to_plot_b_good_task_1 = np.append(spikes_to_plot_b_good_task_1,spikes_to_save_b_good_task_1)  
            spikes_to_plot_b_good_task_1_raster.append(spikes_to_save_b_good_task_1)
            
        for b_bad_task_1 in aligned_b_bad_task_1:
            period_min_b_bad = b_bad_task_1 - 3000
            period_max_b_bad = b_bad_task_1 + 3000
            spikes_ind_b_bad = spikes_times[(spikes_times >= period_min_b_bad) & (spikes_times<=period_max_b_bad)]

            spikes_to_save_b_bad_task_1 = (spikes_ind_b_bad - b_bad_task_1)
            spikes_to_plot_b_bad_task_1= np.append(spikes_to_plot_b_bad_task_1,spikes_to_save_b_bad_task_1)  
            spikes_to_plot_b_bad_task_1_raster.append(spikes_to_save_b_bad_task_1)
            
    ################### Task 2
            
        for a_bad_task_2 in aligned_a_bad_task_2:
            period_min = a_bad_task_2 - 3000
            period_max = a_bad_task_2 + 3000
            spikes_ind_a_bad_task_2 = spikes_times[(spikes_times >= period_min) & (spikes_times<= period_max)]
 
            spikes_to_save_a_bad_task_2 = (spikes_ind_a_bad_task_2 - a_bad_task_2)
            spikes_to_plot_a_bad_task_2= np.append(spikes_to_plot_a_bad_task_2,spikes_to_save_a_bad_task_2)
            spikes_to_plot_a_bad_task_2_raster.append(spikes_to_save_a_bad_task_2)
            
        for a_good_task_2 in aligned_a_good_task_2:
            period_min_a_bad = a_good_task_2 - 3000
            period_max_a_bad = a_good_task_2 + 3000
            spikes_ind_a_good = spikes_times[(spikes_times >= period_min_a_bad) & (spikes_times<=period_max_a_bad)]

            spikes_to_save_a_good_task_2= (spikes_ind_a_good - a_good_task_2)
            spikes_to_plot_a_good_task_2 = np.append(spikes_to_plot_a_good_task_2,spikes_to_save_a_good_task_2) 
            spikes_to_plot_a_good_task_2_raster.append(spikes_to_save_a_good_task_2)
            
            
        for b_bad_task_2 in aligned_b_bad_task_2:
            period_min_b_bad = b_bad_task_2 - 3000
            period_max_b_bad = b_bad_task_2 + 3000
            spikes_ind_b_bad = spikes_times[(spikes_times >= (b_bad_task_2-6000)) & (spikes_times<=(b_bad_task_2+ 6000))]
 
            spikes_to_save_b_bad_task_2 = (spikes_ind_b_bad - b_bad_task_2)
            spikes_to_plot_b_bad_task_2= np.append(spikes_to_plot_b_bad_task_2,spikes_to_save_b_bad_task_2) 
            spikes_to_plot_b_bad_task_2_raster.append(spikes_to_save_b_bad_task_2)
    
    
        for b_good in aligned_b_good_task_2:
            period_min_b_good = b_good - 3000
            period_max_b_good = b_good + 3000
            spikes_ind_b_good = spikes_times[(spikes_times >= period_min_b_good) & (spikes_times<=period_max_b_good)]

            spikes_to_save_b_good_task_2 = (spikes_ind_b_good - b_good)
            spikes_to_plot_b_good_task_2= np.append(spikes_to_plot_b_good_task_2,spikes_to_save_b_good_task_2) 
            spikes_to_plot_b_good_task_2_raster.append(spikes_to_save_b_good_task_2)
        
################### Task 3
            
        for a_bad_task_3 in aligned_a_bad_task_3:
     
            period_min = a_bad_task_3 - 3000
            period_max = a_bad_task_3 + 3000
            spikes_ind_a_bad_task_2 = spikes_times[(spikes_times >= period_min) & (spikes_times<= period_max)]

            spikes_to_save_a_bad_task_3 = (spikes_ind_a_bad_task_2 - a_bad_task_3)
            spikes_to_plot_a_bad_task_3 = np.append(spikes_to_plot_a_bad_task_3,spikes_to_save_a_bad_task_3)
            spikes_to_plot_a_bad_task_3_raster.append(spikes_to_save_a_bad_task_3)
            
        for a_good_task_3 in aligned_a_good_task_3:
            period_min_a_bad = a_good_task_3 - 3000
            period_max_a_bad = a_good_task_3 + 3000
            spikes_ind_a_good = spikes_times[(spikes_times >= period_min_a_bad) & (spikes_times<=period_max_a_bad)]
            spikes_to_save_a_good_task_3 = (spikes_ind_a_good - a_good_task_3)
            spikes_to_plot_a_good_task_3 = np.append(spikes_to_plot_a_good_task_3,spikes_to_save_a_good_task_3) 
            spikes_to_plot_a_good_task_3_raster.append(spikes_to_save_a_good_task_3)
            
            
        for b_bad_task_3 in aligned_b_bad_task_3:
            period_min_b_bad = b_bad_task_3 - 3000
            period_max_b_bad = b_bad_task_3 + 3000
            spikes_ind_b_bad = spikes_times[(spikes_times >= (b_bad_task_3-6000)) & (spikes_times<=(b_bad_task_2+ 6000))]
            spikes_to_save_b_bad_task_3 = (spikes_ind_b_bad - b_bad_task_3)
            spikes_to_plot_b_bad_task_3= np.append(spikes_to_plot_b_bad_task_3,spikes_to_save_b_bad_task_3) 
            spikes_to_plot_b_bad_task_3_raster.append(spikes_to_save_b_bad_task_3)
    
    
        for b_good in aligned_b_good_task_3:
            period_min_b_good = b_good - 3000
            period_max_b_good = b_good + 3000
            spikes_ind_b_good = spikes_times[(spikes_times >= period_min_b_good) & (spikes_times<=period_max_b_good)]

            spikes_to_save_b_good_task_3 = (spikes_ind_b_good - b_good)
            spikes_to_plot_b_good_task_3= np.append(spikes_to_plot_b_good_task_3,spikes_to_save_b_good_task_3) 
            spikes_to_plot_b_good_task_3_raster.append(spikes_to_save_b_good_task_3)

        
        #Raster Plot  Task_1
        all_spikes_raster_plot_task_1 = spikes_to_plot_a_bad_task_1_raster + spikes_to_plot_a_good_task_1_raster  +spikes_to_plot_b_bad_task_1_raster +spikes_to_plot_b_good_task_1_raster
        for ith, trial in enumerate(all_spikes_raster_plot_task_1):           
            if i < group_1:
                pl.figure(2)
                axes_8[0][i].vlines(trial, ith + .5, ith + 1.5)
            elif i >= group_1 and i < group_2:
                pl.figure(4)
                axes_16[0][i-group_1].vlines(trial, ith + .5, ith + 1.5)
            elif i >= group_2 and i < group_3: 
                pl.figure(6)
                axes_24[0][i-group_2].vlines(trial, ith + .5, ith + 1.5)
            elif i >= group_3 and i < group_4: 
                pl.figure(8)
                axes_36[0][i-group_3].vlines(trial, ith + .5, ith + 1.5)
           
            pl.ylim(.5, len(all_spikes_raster_plot_task_1) + .5)
            pl.xlim(-1500, +1500)
            
        x = [-1500, 1500]
        length_block_1 = len(spikes_to_plot_a_bad_task_1_raster)
        length_block_2 = length_block_1 + len(spikes_to_plot_a_good_task_1_raster)
        length_block_3 = length_block_2 + len(spikes_to_plot_b_bad_task_1_raster)
        length_block_4 = length_block_3 + len(spikes_to_plot_b_good_task_1_raster)
        t0 = [0, 0]
        t1 = [length_block_1, length_block_1]
        t2 = [length_block_2, length_block_2]
        t3 = [length_block_3,length_block_3]
        t4 = [length_block_4,length_block_4 ]
        
        if i < group_1:
            pl.figure(2)
            axes_8[0][i].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_8[0][i].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_8[0][i].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_8[0][i].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_8[0][0].set(ylabel ='Trial #')
            axes_8[0][i].set_title('{}'.format(cluster))
          
        elif i >= group_1 and i < group_2:
            pl.figure(4)
            axes_16[0][i-group_1].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_16[0][i-group_1].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_16[0][i-group_1].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_16[0][i-group_1].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_16[0][0].set(ylabel ='Trial #')
            axes_16[0][i-group_1].set_title('{}'.format(cluster))
            
        elif i >= group_2 and i < group_3:
            pl.figure(6) 
            axes_24[0][i-group_2].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_24[0][i-group_2].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_24[0][i-group_2].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_24[0][i-group_2].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_24[0][0].set(ylabel ='Trial #')
            axes_24[0][i-group_2].set_title('{}'.format(cluster))
        elif i >= group_3 and i < group_4:
            pl.figure(8) 
            axes_36[0][i-group_3].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_36[0][i-group_3].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_36[0][i-group_3].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_36[0][i-group_3].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_36[0][0].set(ylabel ='Trial #')
            axes_36[0][i-group_3].set_title('{}'.format(cluster))
            
            
        
            
        #Raster Plot  Task_2  
        all_spikes_raster_plot_task_2 = spikes_to_plot_a_bad_task_2_raster + spikes_to_plot_a_good_task_2_raster  +spikes_to_plot_b_bad_task_2_raster +spikes_to_plot_b_good_task_2_raster
        for ith, trial in enumerate(all_spikes_raster_plot_task_2):
            if i < group_1:
                pl.figure(2)
                axes_8[1][i].vlines(trial, ith + .5, ith + 1.5, label = 'Task 2')
            elif i >= group_1 and i < group_2:
                pl.figure(4)
                axes_16[1][i-group_1].vlines(trial, ith + .5, ith + 1.5, label = 'Task 2')
            elif i >= group_2 and i < group_3: 
                pl.figure(6)
                axes_24[1][i-group_2].vlines(trial, ith + .5, ith + 1.5, label = 'Task 2')
            elif i >= group_3 and i < group_4: 
                pl.figure(8)
                axes_36[1][i-group_3].vlines(trial, ith + .5, ith + 1.5, label = 'Task 2')

            pl.ylim(.5, len(all_spikes_raster_plot_task_2) + .5)
            pl.xlim(-1500, +1500)

        length_block_1_task2 = len(spikes_to_plot_a_bad_task_2_raster)
        length_block_2_task_2 = length_block_1_task2 + len(spikes_to_plot_a_good_task_2_raster)
        length_block_3_task_2 = length_block_2_task_2 + len(spikes_to_plot_b_bad_task_2_raster)
        length_block_4_task_2 = length_block_3_task_2 + len(spikes_to_plot_b_good_task_2_raster)
        t0 = [0, 0]
        t1_task_2 = [length_block_1_task2, length_block_1_task2]
        t2_task_2 = [length_block_2_task_2, length_block_2_task_2]
        t3_task_2 = [length_block_3_task_2,length_block_3_task_2]
        t4_task_2 = [length_block_4_task_2,length_block_4_task_2]
        
        if i < group_1:
            pl.figure(2)
            axes_8[1][i].fill_between(x,t0, t1_task_2 ,color = 'gray', alpha = 0.5)
            axes_8[1][i].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            axes_8[1][i].fill_between(x,t2_task_2,t3_task_2,color = 'skyblue', alpha = 0.5)
            axes_8[1][i].fill_between(x, t3_task_2,t4_task_2,color = 'navy', alpha = 0.5)
            axes_8[1][0].set(ylabel ='Trial #')   
        elif i >= group_1 and i < group_2:
            pl.figure(4)
            axes_16[1][i-group_1].fill_between(x,t0, t1_task_2 ,color = 'gray', alpha = 0.5)
            axes_16[1][i-group_1].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            axes_16[1][i-group_1].fill_between(x,t2_task_2,t3_task_2,color = 'skyblue', alpha = 0.5)
            axes_16[1][i-group_1].fill_between(x, t3_task_2,t4_task_2,color = 'navy', alpha = 0.5)
            axes_16[1][0].set(ylabel ='Trial #')   
            
        elif i >= group_2 and i < group_3: 
            pl.figure(6) 
            axes_24[1][i-group_2].fill_between(x,t0, t1_task_2 ,color = 'gray', alpha = 0.5)
            axes_24[1][i-group_2].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            axes_24[1][i-group_2].fill_between(x,t2_task_2,t3_task_2,color = 'skyblue', alpha = 0.5)
            axes_24[1][i-group_2].fill_between(x, t3_task_2,t4_task_2,color = 'navy', alpha = 0.5)
            axes_24[1][0].set(ylabel ='Trial #')
            
        elif i >= group_3 and i < group_4: 
            pl.figure(8) 
            axes_36[1][i-group_3].fill_between(x,t0, t1_task_2 ,color = 'gray', alpha = 0.5)
            axes_36[1][i-group_3].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            axes_36[1][i-group_3].fill_between(x,t2_task_2,t3_task_2,color = 'skyblue', alpha = 0.5)
            axes_36[1][i-group_3].fill_between(x, t3_task_2,t4_task_2,color = 'navy', alpha = 0.5)
            axes_36[1][0].set(ylabel ='Trial #')
        
        all_spikes_raster_plot_task_3 = spikes_to_plot_a_bad_task_3_raster + spikes_to_plot_a_good_task_3_raster  +spikes_to_plot_b_bad_task_3_raster +spikes_to_plot_b_good_task_3_raster
        for ith, trial in enumerate(all_spikes_raster_plot_task_3):
            if i < group_1:
                pl.figure(2)
                axes_8[2][i].vlines(trial, ith + .5, ith + 1.5)
            elif i >= group_1 and i < group_2:
                pl.figure(4)
                axes_16[2][i-group_1].vlines(trial, ith + .5, ith + 1.5)
            elif i >= group_2 and i < group_3: 
                pl.figure(6)
                axes_24[2][i-group_2].vlines(trial, ith + .5, ith + 1.5)
            elif i >= group_3 and i < group_4: 
                pl.figure(8)
                axes_36[2][i-group_3].vlines(trial, ith + .5, ith + 1.5)
                
            pl.ylim(.5, len(all_spikes_raster_plot_task_3) + .5)
            pl.xlim(-1500, +1500)
        x = [-1500, 1500]
        length_block_1 = len(spikes_to_plot_a_bad_task_3_raster)
        length_block_2 = length_block_1 + len(spikes_to_plot_a_good_task_3_raster)
        length_block_3 = length_block_2 + len(spikes_to_plot_b_bad_task_3_raster)
        length_block_4 = length_block_3 + len(spikes_to_plot_b_good_task_3_raster)
        t0 = [0, 0]
        t1 = [length_block_1, length_block_1]
        t2 = [length_block_2, length_block_2]
        t3 = [length_block_3,length_block_3]
        t4 = [length_block_4,length_block_4 ]
        
        if i < group_1:
            pl.figure(2)
            axes_8[2][i].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_8[2][i].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_8[2][i].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_8[2][i].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_8[2][i].set(xlabel ='Time (ms)')
            axes_8[2][0].set(ylabel ='Trial #')
        elif i >= group_1 and i < group_2:
            pl.figure(4)
            axes_16[2][i-group_1].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_16[2][i-group_1].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_16[2][i-group_1].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_16[2][i-group_1].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_16[2][i-group_1].set(xlabel ='Time (ms)')
            axes_16[2][0].set(ylabel ='Trial #')
            
        elif i >= group_2 and i < group_3: 
            pl.figure(6) 
            axes_24[2][i-group_2].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_24[2][i-group_2].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_24[2][i-group_2].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_24[2][i-group_2].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_24[2][i-group_2].set(xlabel ='Time (ms)')
            axes_24[2][0].set(ylabel ='Trial #')
                   
        elif i >= group_3 and i < group_4: 
            pl.figure(8) 
            axes_36[2][i-group_3].fill_between(x, t1, t0 ,color = 'gray', alpha = 0.5)
            axes_36[2][i-group_3].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_36[2][i-group_3].fill_between(x,t3,t2,color = 'skyblue', alpha = 0.5)
            axes_36[2][i-group_3].fill_between(x,t4,t3,color = 'navy', alpha = 0.5)
            axes_36[2][i-group_3].set(xlabel ='Time (s)')
            axes_36[2][0].set(ylabel ='Trial #')
          
        
        bin_width_ms = 1
        smooth_sd_ms = 100
        sns.set(style="white", palette="muted", color_codes=True)
        trial_duration = 12000
        bin_edges_trial = np.arange(-6000,trial_duration, bin_width_ms)

        #Plotting A Good Task 1
        
        spikes_to_plot_a_good_task_1 = spikes_to_plot_a_good_task_1[~np.isnan(spikes_to_plot_a_good_task_1)]
        hist_task,edges_task = np.histogram(spikes_to_plot_a_good_task_1, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_a_good_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task, color = 'black', label='Poke 1 Task 1')
        elif i >= group_1 and i < group_2:
            ax_16[0][i- group_1].plot(bin_edges_trial[:-1], normalised_task, color = 'black', label='Poke 1 Task 1') 
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task, color = 'black', label='Poke 1 Good Task 1')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task, color = 'black', label='Poke 1 Good Task 1')
        
    
        #Plotting A Bad Task 1 

        spikes_to_plot_a_bad_task_1 = spikes_to_plot_a_bad_task_1[~np.isnan(spikes_to_plot_a_bad_task_1)]
        hist_task,edges_task = np.histogram(spikes_to_plot_a_bad_task_1, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_a_bad_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'gray', label='Poke 1 Bad Task 1')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'gray', label='Poke 1 Bad Task 1')
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'gray', label='Poke 1 Bad Task 1')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'gray', label='Poke 1 Bad Task 1')


        #Plotting B Good Task 1

        spikes_to_plot_b_good_task_1 = spikes_to_plot_b_good_task_1[~np.isnan(spikes_to_plot_b_good_task_1)]
        hist_task,edges_task = np.histogram(spikes_to_plot_b_good_task_1, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_b_good_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task, color = 'navy', label='Poke 2 Good Task 1')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task, color = 'navy', label='Poke 2 Good Task 1')
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task, color = 'navy', label='Poke 2 Good Task 1')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task, color = 'navy', label='Poke 2 Good Task 1')

        
        #Plotting B Good Task 1
        spikes_to_plot_b_bad_task_1 = spikes_to_plot_b_bad_task_1[~np.isnan(spikes_to_plot_b_bad_task_1)]
        hist_task,edges_task = np.histogram(spikes_to_plot_b_bad_task_1, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_b_bad_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 1')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 1')
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 1')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 1')

    
        #Plotting A Good Task 2 
        spikes_to_plot_a_good_task_2 = spikes_to_plot_a_good_task_2[~np.isnan(spikes_to_plot_a_good_task_2)]
        hist_task,edges_task = np.histogram(spikes_to_plot_a_good_task_2, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_a_good_task_2_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_A_task_2 == poke_A:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_A_task_2 == poke_B:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')  
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')  
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_A_task_2 == poke_A:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_A_task_2 == poke_B:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')
        elif i >= group_2 and i < group_3: 
            pl.figure(5)
            if poke_A_task_2 == poke_A:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_A_task_2 == poke_B:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')
        elif i >= group_3 and i < group_4: 
            pl.figure(7)
            if poke_A_task_2 == poke_A:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_A_task_2 == poke_B:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')

                
        #Plotting A Bad Task 2 
        spikes_to_plot_a_bad_task_2 = spikes_to_plot_a_bad_task_2[~np.isnan(spikes_to_plot_a_bad_task_2)]
        hist_task,edges_task = np.histogram(spikes_to_plot_a_bad_task_2, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_a_bad_task_2_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_A_task_2 == poke_A:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 2')
            elif poke_A_task_2 == poke_B:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'pink', label='Poke 3 Bad Task 2')
                

        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_A_task_2 == poke_A:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 2')
            elif poke_A_task_2 == poke_B:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'pink', label='Poke 3 Bad Task 2')

        elif  i >= group_2 and i < group_3: 
            pl.figure(5)
            if poke_A_task_2 == poke_A:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 2')
            elif poke_A_task_2 == poke_B:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'pink', label='Poke 3 Bad Task 2')

        elif  i >= group_3 and i < group_4: 
            pl.figure(7)
            if poke_A_task_2 == poke_A:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 2')
            elif poke_A_task_2 == poke_B:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 2')
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'pink', label='Poke 3 Bad Task 2')



    
        #Plotting B Good Task 2 
        spikes_to_plot_b_good_task_2 = spikes_to_plot_b_good_task_2[~np.isnan(spikes_to_plot_b_good_task_2)]
        hist_task,edges_task = np.histogram(spikes_to_plot_b_good_task_2, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_b_good_list_task_2)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_B_task_2 == poke_A:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_B_task_2 == poke_B:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_B_task_2 == poke_A:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_B_task_2 == poke_B:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')
        elif i >= group_2 and i < group_3: 
            pl.figure(5)
            if poke_B_task_2 == poke_A:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_B_task_2 == poke_B:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')
        elif i >= group_3 and i < group_4: 
            pl.figure(7)
            if poke_B_task_2 == poke_A:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 2')
            elif poke_B_task_2 == poke_B:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "red", label='Poke 3 Good Task 2')
        
        #Plotting B Bad Task 2 
        spikes_to_plot_b_bad_task_2 = spikes_to_plot_b_bad_task_2[~np.isnan(spikes_to_plot_b_bad_task_2)]
        hist_task,edges_task = np.histogram(spikes_to_plot_b_bad_task_2, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_b_bad_list_task_2)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_B_task_2 == poke_A:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 2')
            elif poke_B_task_2 == poke_B:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task,'--',color = "pink", label='Poke 3 Bad Task 2')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_B_task_2 == poke_A:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 2')
            elif poke_B_task_2 == poke_B:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--',color = "pink", label='Poke 3 Bad Task 2')
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            if poke_B_task_2 == poke_A:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 2')
            elif poke_B_task_2 == poke_B:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--',color = "pink", label='Poke 3 Bad Task 2')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            if poke_B_task_2 == poke_A:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 2')
            elif poke_B_task_2 == poke_B:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 2')
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--',color = "pink", label='Poke 3 Bad Task 2')
                
        #Plotting A Good Task 3
        spikes_to_plot_a_good_task_3 = spikes_to_plot_a_good_task_3[~np.isnan(spikes_to_plot_a_good_task_3)]
        hist_task,edges_task = np.histogram(spikes_to_plot_a_good_task_3, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_a_good_task_3_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_B:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_B:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
        elif i >= group_2 and i < group_3:
            pl.figure(5)        
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_B:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_A_task_3 == poke_B:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
                
        #Plotting A Bad Task 3
        spikes_to_plot_a_bad_task_3 = spikes_to_plot_a_bad_task_3[~np.isnan(spikes_to_plot_a_bad_task_3)]
        hist_task,edges_task = np.histogram(spikes_to_plot_a_bad_task_3, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_a_bad_task_3_list)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_B:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'lightyellow', label='Poke 4 Bad Task 3')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3') 
            elif poke_A_task_3 == poke_B:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 3') 
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'lightyellow', label='Poke 4 Bad Task 3') 
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_B:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'lightyellow', label='Poke 4 Bad Task 3')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'grey', label='Poke 1 Bad Task 3')
            elif poke_A_task_3 == poke_B:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'skyblue', label='Poke 2 Bad Task 3')
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--', color = 'lightyellow', label='Poke 4 Bad Task 3')
    
        #Plotting B Good Task 3 
        spikes_to_plot_b_good_task_3 = spikes_to_plot_b_good_task_3[~np.isnan(spikes_to_plot_b_good_task_3)]
        hist_task,edges_task = np.histogram(spikes_to_plot_b_good_task_3, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_b_good_list_task_3)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_B_task_3 == poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_B_task_3 == poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
        elif i >= group_2 and i < group_3:
            pl.figure(5)
            if poke_B_task_3 == poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
        elif i >= group_3 and i < group_4:
            pl.figure(7)
            if poke_B_task_3 == poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "black", label='Poke 1 Good Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "navy", label='Poke 2 Good Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,color = "yellow", label='Poke 4 Good Task 3')
                
        #Plotting B Bad Task 3 
        spikes_to_plot_b_bad_task_3 = spikes_to_plot_b_bad_task_3[~np.isnan(spikes_to_plot_b_bad_task_3)]
        hist_task,edges_task = np.histogram(spikes_to_plot_b_bad_task_3, bins= bin_edges_trial)# histogram per second
        hist_task = hist_task/len(entry_b_bad_list_task_3)
        normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
        if i < group_1:
            pl.figure(1)
            if poke_B_task_3 == poke_A:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task,'--',color = "lightyellow", label='Poke 4 Bad Task 3')
        elif i >= group_1 and i < group_2:
            pl.figure(3)
            if poke_B_task_3 == poke_A:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task,'--',color = "lightyellow", label='Poke 4 Bad Task 3')
        elif i >= group_2 and i < group_3: 
            pl.figure(5)
            if poke_B_task_3 == poke_A:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task,'--',color = "lightyellow", label='Poke 4 Bad Task 3') 
        elif i >= group_3 and i < group_4: 
            pl.figure(7)
            if poke_B_task_3 == poke_A:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--',color = "grey", label='Poke 1 Bad Task 3')
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--',color = "skyblue", label='Poke 2 Bad Task 3')
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task,'--',color = "lightyellow", label='Poke 4 Bad Task 3')
        
                
        pl.xlim(-1500, +1500)
        
        if i < group_1:

            ax_8[1][0].legend(fontsize = 'xx-small')
            ax_8[0][0].legend(fontsize = 'xx-small')
            ax_8[2][0].legend(fontsize = 'xx-small')
            ax_8[0][i].set_title('{}'.format(cluster))
            ax_8[0][0].set(ylabel='Firing Rate')
            ax_8[1][0].set(ylabel='Firing Rate')
            ax_8[2][0].set(ylabel='Firing Rate')
            ax_8[2][i].set(xlabel= 'Time (ms)')
        elif i >= group_1 and i < group_2:

            ax_16[1][0].legend(fontsize = 'xx-small')
            ax_16[0][0].legend(fontsize = 'xx-small')
            ax_16[2][0].legend(fontsize = 'xx-small')
            ax_16[0][i-group_1].set_title('{}'.format(cluster))            
            ax_16[0][0].set(ylabel='Firing Rate')
            ax_16[1][0].set(ylabel='Firing Rate')
            ax_16[2][0].set(ylabel='Firing Rate')
            ax_16[2][i-group_1].set(xlabel= 'Time (ms)')
        elif i >= group_2 and i < group_3:     
 
            ax_24[1][0].legend(fontsize = 'xx-small')
            ax_24[0][0].legend(fontsize = 'xx-small')
            ax_24[2][0].legend(fontsize = 'xx-small')
            ax_24[0][i-group_2].set_title('{}'.format(cluster))
            ax_24[0][0].set(ylabel='Firing Rate')
            ax_24[1][0].set(ylabel='Firing Rate')
            ax_24[2][0].set(ylabel='Firing Rate')
            ax_24[2][i-group_2].set(xlabel= 'Time (ms)')
            
            
#    if plot == 'Choice':        
#        fig_clusters_8.savefig(path.join(outpath, '%s Choice Histograms_G1.png' %beh_session.datetime))
#        fig_raster_8.savefig(path.join(outpath,'%s Choice Raster_G1.png' %beh_session.datetime ))
#        fig_clusters_16.savefig(path.join(outpath,'%s Choice Histograms_G2.png' %beh_session.datetime))
#        fig_raster_16.savefig(path.join(outpath,'%s Choice Raster_G2.png' %beh_session.datetime))
#        fig_clusters_24.savefig(path.join(outpath,'%s Choice Histograms_G3.png' %beh_session.datetime))
#        fig_raster_24.savefig(path.join(outpath, '%s Choice Raster_G3.png'%beh_session.datetime))
#        fig_clusters_36.savefig(path.join(outpath,'%s Choice Histograms_G4.png' %beh_session.datetime))
#        fig_raster_36.savefig(path.join(outpath, '%s Choice Raster_G4.png' %beh_session.datetime ))
#    elif plot == 'Poke':
#        fig_clusters_8.savefig(path.join(outpath, '%s Poke Histograms_G1.png' %beh_session.datetime))
#        fig_raster_8.savefig(path.join(outpath,'%s Poke Raster_G1.png' %beh_session.datetime ))
#        fig_clusters_16.savefig(path.join(outpath,'%s Poke Histograms_G2.png' %beh_session.datetime))
#        fig_raster_16.savefig(path.join(outpath,'%s Poke Raster_G2.png' %beh_session.datetime))
#        fig_clusters_24.savefig(path.join(outpath,'%s Poke Histograms_G3.png' %beh_session.datetime))
#        fig_raster_24.savefig(path.join(outpath, '%s Poke Raster_G3.png'%beh_session.datetime))
#        fig_clusters_36.savefig(path.join(outpath,'%s Poke Histograms_G4.png' %beh_session.datetime))
#        fig_raster_36.savefig(path.join(outpath, '%s Poke Raster_G4.png' %beh_session.datetime ))

    

# Session Plot
def session_spikes_vs_trials_plot(ephys_session,beh_session, cluster):
    bin_width_ms = 1000
    smooth_sd_ms = 20000
    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    spikes_cluster_8 = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
    session_duration_ms = int(np.nanmax(ephys_session.time))- int(np.nanmin(ephys_session.time))
    bin_edges = np.arange(0,session_duration_ms, bin_width_ms)
    spikes_cluster_8 = ephys_session.loc[ephys_session['spike_cluster'] == 45]
    spikes_times_8 = np.array(spikes_cluster_8['time'])
    spikes_times_8 = spikes_times_8[~np.isnan(spikes_times_8)]
    hist,edges = np.histogram(spikes_times_8, bins= bin_edges)# histogram per second
    normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
    pl.figure()
    pl.plot(bin_edges[:-1], normalised/max(normalised), label='Firing Rate', color ='navy') 

    trial_rate,edges_py = np.histogram(pyControl_choice, bins=bin_edges)
    trial_rate = gaussian_filter1d(trial_rate.astype(float), smooth_sd_ms/bin_width_ms)
    pl.plot(bin_edges[:-1], trial_rate/max(trial_rate), label='Rate', color = 'lightblue')
    pl.xlabel('Time (ms)')
    pl.ylabel('Smoothed Firing Rate')
    pl.legend()



