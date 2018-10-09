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
import matplotlib.patches as mpatches
#Import Ephys and PyControl Data
#ephys_session = fu.load_data('m481_2018-06-20_19-09-08','/Users/veronikasamborska/Desktop/Ephys 3 Tasks Processed Spikes/m481/','/',True)
#beh_session = di.Session('/Users/veronikasamborska/Desktop/data_3_tasks_ephys/m481-2018-06-20-190858.txt')


def change_block_firing_rates(ephys_session, beh_session):
    
    forced_trials = beh_session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0] 
    state = beh_session.trial_data['state']
    state_non_forced = state[non_forced_array]
    task = beh_session.trial_data['task']
    task_non_forced = task[non_forced_array]
    forced_array = np.where(forced_trials == 1)[0]
    
    #Task 1 
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]
    
    
    state_1 = state_non_forced[:len(task_1)]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    import pylab as pl

    state_2 = state_non_forced[len(task_1): (len(task_1) +len(task_2))]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_b_good = np.where(state_2 == 0)[0]
    
    state_3 = state_non_forced[len(task_1) + len(task_2):]
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]
    
    choices = beh_session.trial_data['choices']
    choices_non_forced = choices[non_forced_array]
    
    choice_task_1 = choices_non_forced[:len(task_1)]
    choice_task_2 = choices_non_forced[len(task_1): (len(task_1) +len(task_2))]
    choice_task_3 = choices_non_forced[len(task_1) + len(task_2):]
    
    choices_a_task_1 = np.where(choice_task_1 == 1)[0]
    choices_b_task_1 = np.where(choice_task_1 == 0)[0]
    choices_a_task_2 = np.where(choice_task_2 == 1)[0]
    choices_b_task_2 = np.where(choice_task_2 == 0)[0]
    choices_a_task_3 = np.where(choice_task_3 == 1)[0]
    choices_b_task_3 = np.where(choice_task_3 == 0)[0]

   
    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    
    trial_сhoice_state_task_1 = pyControl_choice[:len(task_1)]
    trial_сhoice_state_task_2 = pyControl_choice[len(task_1):(len(task_1) +len(task_2))]
    print(len(trial_сhoice_state_task_2))
    trial_сhoice_state_task_3 = pyControl_choice[len(task_1) + len(task_2):]
    
    
    clusters = ephys_session['spike_cluster'].unique()
   

    figure_bloc, ax_block = plt.subplots(figsize = (50,5), ncols = 2, nrows = 3, sharey = 'col')
    
    for i,cluster in enumerate(clusters): 
        figure_bloc = pl.figure(10)
        spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
        spikes_times = np.array(spikes['time'])
        spikes_count = 0
        firing_rate_trial_list = np.array([])
        pl.figure(10)
        for choice in trial_сhoice_state_task_1:
            trial_start = choice - 1000
            trial_end = choice + 1000
            spikes_ind = spikes_times[(spikes_times >= trial_start) & (spikes_times<= trial_end)]
            spikes_count = np.count_nonzero(~np.isnan(spikes_ind))
            firing_rate_trial = spikes_count/2000
            firing_rate_trial_list = np.append(firing_rate_trial_list, firing_rate_trial)
            
        mean_firing_rate =np.mean(firing_rate_trial_list)
        std_firing_rate = np.std(firing_rate_trial_list)
        zscore= (firing_rate_trial_list- mean_firing_rate)/std_firing_rate
        barlist = ax_block[0][i].bar(np.arange(len(zscore)),zscore)
        ax_block[0][i].set_title('{}'.format(cluster))
        
        for ind,bar in enumerate(barlist):
            if ind in state_a_good and ind in choices_a_task_1 :
                barlist[ind].set_color('r')
                barlist[ind].set_label('Poke 1 good, Correct Choice')
            elif ind in state_a_good and ind in choices_b_task_1:
                barlist[ind].set_color('olive')
                barlist[ind].set_label('Poke 1 good, Incorrect Choice')
                
            elif ind in state_b_good and ind in choices_b_task_1:
                barlist[ind].set_color('green')
                barlist[ind].set_label('Poke 2 good, Correct Choice')
            elif ind in state_b_good and ind in choices_a_task_1:
                barlist[ind].set_color('lightgreen')
                barlist[ind].set_label('Poke 2 good, Inorrect Choice')
                   
        firing_rate_trial_list_task_2 = np.array([])
        for choice in trial_сhoice_state_task_2:
            trial_start = choice - 1000
            trial_end = choice + 1000
            spikes_ind = spikes_times[(spikes_times >= trial_start) & (spikes_times<= trial_end)]
            spikes_count = np.count_nonzero(~np.isnan(spikes_ind))
            firing_rate_trial = spikes_count/2000
            firing_rate_trial_list_task_2 = np.append(firing_rate_trial_list_task_2, firing_rate_trial)
            
        mean_firing_rate_task_2 =np.mean(firing_rate_trial_list_task_2)
        std_firing_rate_task_2 = np.std(firing_rate_trial_list_task_2)
        zscore= (firing_rate_trial_list_task_2- mean_firing_rate_task_2)/std_firing_rate_task_2
        barlist_task_2 = ax_block[1][i].bar(np.arange(len(zscore)),zscore)
        ax_block[1][i].set_title('{}'.format(cluster))
        
        for ind,bar in enumerate(barlist_task_2):
            if ind in state_a_good and ind in choices_a_task_2 :
                barlist_task_2[ind].set_color('r')
                barlist_task_2[ind].set_label('Poke 1 good, Correct Choice')
            elif ind in state_a_good and ind in choices_b_task_2:
                barlist_task_2[ind].set_color('olive')
                barlist_task_2[ind].set_label('Poke 1 good, Incorrect Choice')
                
            elif ind in state_b_good and ind in choices_b_task_2:
                barlist_task_2[ind].set_color('green')
                barlist_task_2[ind].set_label('Poke 2 good, Correct Choice')
            elif ind in state_b_good and ind in choices_a_task_2:
                barlist_task_2[ind].set_color('lightgreen')
                barlist_task_2[ind].set_label('Poke 2 good, Inorrect Choice')
                
        firing_rate_trial_list_task_3 = np.array([])
        for choice in trial_сhoice_state_task_3:
            trial_start = choice - 1000
            trial_end = choice + 1000
            spikes_ind = spikes_times[(spikes_times >= trial_start) & (spikes_times<= trial_end)]
            spikes_count = np.count_nonzero(~np.isnan(spikes_ind))
            firing_rate_trial = spikes_count/2000
            firing_rate_trial_list_task_3 = np.append(firing_rate_trial_list_task_3, firing_rate_trial)
            
        mean_firing_rate_task_3 =np.mean(firing_rate_trial_list_task_3)
        std_firing_rate_task_3 = np.std(firing_rate_trial_list_task_3)
        zscore= (firing_rate_trial_list_task_3- mean_firing_rate_task_3)/std_firing_rate_task_3
        barlist_task_3 = ax_block[2][i].bar(np.arange(len(zscore)),zscore)
        ax_block[2][i].set_title('{}'.format(cluster))
        
        for ind,bar in enumerate(barlist_task_3):
            if ind in state_t2_a_good and ind in choices_a_task_1 :
                barlist_task_3[ind].set_color('r')
                barlist_task_3[ind].set_label('Poke 1 good, Correct Choice')
            elif ind in state_t2_a_good and ind in choices_b_task_1:
                barlist_task_3[ind].set_color('olive')
                barlist_task_3[ind].set_label('Poke 1 good, Incorrect Choice')
                
            elif ind in state_t2_b_good and ind in choices_b_task_1:
                barlist_task_3[ind].set_color('green')
                barlist_task_3[ind].set_label('Poke 2 good, Correct Choice')
            elif ind in state_t2_b_good and ind in choices_a_task_1:
                barlist_task_3[ind].set_color('lightgreen')
                barlist_task_3[ind].set_label('Poke 2 good, Inorrect Choice')

        
        ax_block[2][i].set(xlabel ='Trial #')
        ax_block[1][i].set(ylabel ='Z-score Firing Rate #')
        olive_patch = mpatches.Patch(color='olive', label='Poke 1 good, Correct Choice')
        olive_patch = mpatches.Patch(color='olive', label='Poke 1 good, Incorrect Choice')
        green_patch = mpatches.Patch(color='green', label='Poke 2 good, Correct Choice')
        lightgreen_patch = mpatches.Patch(color='lightolive', label='Poke 2 good, Inorrect Choice')
        ax_block[0][0].legend(handles=[olive_patch, olive_patch, green_patch, lightgreen_patch])

    
    
def histogram_raster_plot_poke_aligned(ephys_session, beh_session,outpath, plot, correct):
    #plot = 'Choice' #Specify which plot  #'Choice if want a plot to align at initiation, 'Poke' if want a plot aligned at A/B poke
    
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
    outcomes = beh_session.trial_data['outcomes']
    outcomes_non_forced = outcomes[non_forced_array]
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
    print(poke_A, poke_A_task_2, poke_A_task_3)
    b_pokes = np.unique(beh_session.trial_data['poke_B'])
    print('These are B pokes')
    print(poke_B, poke_B_task_2, poke_B_task_3)
    i_pokes = np.unique(configuration)
    print('These are I pokes')
    configuration = beh_session.trial_data['configuration_i']
    i_poke_task_1 = configuration[0]
    i_poke_task_2 = configuration[task_2_change[0]]
    i_poke_task_3 = configuration[task_3_change[0]]
    print(i_poke_task_1, i_poke_task_2, i_poke_task_3)
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
    
    for event in events_and_times:
        choice_state_count = 0
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

    all_events = poke_A_list + poke_B_list
    print(len(all_events))
    print(len(pyControl_choice))
    
    #Task 1 

    state_1 = state_non_forced[:len(task_1)]
    outcomes_1 = outcomes_non_forced[:len(task_1)]
    outcomes_1_NR = np.where(outcomes_1 == 0)[0]
    state_a_good = np.where(state_1 == 1)[0]
    state_a_good_NR = np.where(state_1[outcomes_1_NR] == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    state_b_good_NR = np.where(state_1[outcomes_1_NR] == 0)[0]
    # Task 2
   
    state_2 = state_non_forced[len(task_1): (len(task_1) +len(task_2))]
    outcomes_2 = outcomes_non_forced[len(task_1): (len(task_1) +len(task_2))]
    outcomes_2_NR = np.where(outcomes_2 == 0)[0]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_a_good_NR = np.where(state_2[outcomes_2_NR] == 1)[0]
    state_t2_b_good = np.where(state_2 == 0)[0]
    state_t2_b_good_NR = np.where(state_2[outcomes_2_NR] == 0)[0]

    #Task 3 Time Events
    
    state_3 = state_non_forced[len(task_1) + len(task_2):]
    outcomes_3 = outcomes_non_forced[len(task_1) + len(task_2):]
    outcomes_3_NR = np.where(outcomes_3 == 0)[0] 
    
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_a_good_NR = np.where(state_3[outcomes_3_NR] == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]
    state_t3_b_good_NR = np.where(state_3[outcomes_3_NR] == 0)[0]

    #For Choice State Calculations

    trial_сhoice_state_task_1 = pyControl_choice[:len(task_1)]
    trial_сhoice_state_task_2 = pyControl_choice[len(task_1):(len(task_1) +len(task_2))]
    trial_сhoice_state_task_3 = pyControl_choice[len(task_1) + len(task_2):]

    trial_сhoice_state_task_1_a_good = trial_сhoice_state_task_1[state_a_good]
    trial_сhoice_state_task_1_a_good_NR = trial_сhoice_state_task_1[state_a_good_NR]
    
    trial_сhoice_state_task_2_a_good = trial_сhoice_state_task_2[state_t2_a_good]
    trial_сhoice_state_task_2_a_good_NR = trial_сhoice_state_task_2[state_t2_a_good_NR]
    trial_сhoice_state_task_3_a_good = trial_сhoice_state_task_3[state_t3_a_good]
    trial_сhoice_state_task_3_a_good_NR = trial_сhoice_state_task_3[state_t3_a_good_NR]
    
    trial_сhoice_state_task_1_b_good = trial_сhoice_state_task_1[state_b_good]
    trial_сhoice_state_task_1_b_good_NR = trial_сhoice_state_task_1[state_b_good_NR]
    trial_сhoice_state_task_2_b_good = trial_сhoice_state_task_2[state_t2_b_good]
    trial_сhoice_state_task_2_b_good_NR = trial_сhoice_state_task_2[state_t2_b_good_NR]
    trial_сhoice_state_task_3_b_good = trial_сhoice_state_task_3[state_t3_b_good]
    trial_сhoice_state_task_3_b_good_NR = trial_сhoice_state_task_3[state_t3_b_good_NR]
    
    task_1_end_trial = np.where(task == 1)[0]
    task_2_end_trial = np.where(task == 2)[0]
    pyControl_end_trial_1 = pyControl_end_trial[:len(task_1_end_trial)]
    pyControl_end_trial_2 =pyControl_end_trial[len(task_1_end_trial)+2:(len(task_1_end_trial)+len(task_2_end_trial)+2)]
    pyControl_end_trial_3 = pyControl_end_trial[len(task_1_end_trial)+len(task_2_end_trial)+4:]
    pyControl_end_trial =   np.concatenate([pyControl_end_trial_1, pyControl_end_trial_2,pyControl_end_trial_3])
    #pyControl_end_trial =   np.concatenate(pyControl_end_trial,pyControl_end_trial_3)

    
    #For ITI Calculations
    
    print(len(pyControl_end_trial))
    ITI_non_forced = pyControl_end_trial[non_forced_array]   
    ITI_task_1 = ITI_non_forced[:len(task_1)]#[2:]
    ITI_task_1_a_good = ITI_task_1[state_a_good]
    ITI_task_1_a_good_NR = ITI_task_1[state_a_good_NR]
    ITI_task_1_b_good = ITI_task_1[state_b_good]
    ITI_task_1_b_good_NR = ITI_task_1[state_b_good_NR]
    
    
    ITI_task_2 = ITI_non_forced[(len(task_1)):(len(task_1)+len(task_2))]
    ITI_task_2_a_good  = ITI_task_2[state_t2_a_good]
    ITI_task_2_a_good_NR  = ITI_task_2[state_t2_a_good_NR]
    ITI_task_2_b_good =ITI_task_2[state_t2_b_good]
    ITI_task_2_b_good_NR =ITI_task_2[state_t2_b_good_NR]
    
    ITI_task_3 = ITI_non_forced[len(task_1) + len(task_2):]
    ITI_task_3_a_good  = ITI_task_3[state_t3_a_good]
    ITI_task_3_a_good_NR  = ITI_task_3[state_t3_a_good_NR]
    ITI_task_3_b_good  = ITI_task_3[state_t3_b_good]
    ITI_task_3_b_good_NR  = ITI_task_3[state_t3_b_good_NR]


    if correct == 'Meaning':
        trial_сhoice_state_task_1_a_good = trial_сhoice_state_task_1_a_good_NR
        trial_сhoice_state_task_2_a_good = trial_сhoice_state_task_2_a_good_NR
        trial_сhoice_state_task_3_a_good = trial_сhoice_state_task_3_a_good_NR 
        trial_сhoice_state_task_1_b_good = trial_сhoice_state_task_1_b_good_NR
        trial_сhoice_state_task_2_b_good = trial_сhoice_state_task_2_b_good_NR
        trial_сhoice_state_task_3_b_good = trial_сhoice_state_task_3_b_good_NR
        ITI_task_1_a_good = ITI_task_1_a_good_NR
        ITI_task_1_b_good = ITI_task_1_b_good_NR
        ITI_task_2_a_good = ITI_task_2_a_good_NR
        ITI_task_2_b_good = ITI_task_2_b_good_NR
        ITI_task_3_a_good = ITI_task_3_a_good_NR
        ITI_task_3_b_good = ITI_task_3_b_good_NR
        
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
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial_task_3 and entry_b <= end_trial_task_3):
                entry_b_good_list_task_3.append(entry_b)
                b_good_choice_time_task_3.append(start_trial_task_3)

               
    for start_trial_task_3,end_trial_task_3 in zip(trial_сhoice_state_task_3_a_good, ITI_task_3_a_good):
        for entry in poke_A_list:
            if (entry >= start_trial_task_3 and entry <= end_trial_task_3):
                entry_a_good_task_3_list.append(entry)
                a_good_choice_time_task_3.append(start_trial_task_3)
        for entry_b in poke_B_list: 
            if (entry_b >= start_trial_task_3 and entry_b <= end_trial_task_3):
                entry_b_bad_list_task_3.append(entry_b)
                b_bad_choice_time_task_3.append(start_trial_task_3)

    
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
    
    fig_clusters_8, ax_8 = plt.subplots(figsize = (15,5), ncols = 9, nrows = 3, sharex=True, sharey = 'col')
    fig_raster_8, axes_8 = plt.subplots(figsize=(15,5), ncols = 9, nrows = 3, sharex=True)
    
    fig_clusters_16, ax_16 = plt.subplots(figsize = (15,5), ncols = 9, nrows = 3, sharex=True, sharey = 'col')
    fig_raster_16, axes_16= plt.subplots(figsize=(15,5), ncols = 9, nrows = 3, sharex=True)
    
    fig_clusters_24, ax_24 = plt.subplots(figsize = (15,5), ncols = 9, nrows = 3, sharex=True, sharey = 'col')
    fig_raster_24, axes_24 = plt.subplots(figsize = (15,5), ncols = 9, nrows = 3, sharex=True)
    
    fig_clusters_36, ax_36 = plt.subplots(figsize = (15,5), ncols = 9, nrows = 3, sharex=True, sharey = 'col')
    fig_raster_36, axes_36 = plt.subplots(figsize = (15,5), ncols = 9, nrows = 3, sharex=True)
    
    
    
    group_1 = 9
    group_2 = 18
    group_3 = 27
    group_4 = 36
    window_to_plot = 5000
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
        bad_task_1 =np.array([])
        bad_task_2 = np.array([])
        bad_task_3 = np.array([])
        good_task_1 = np.array([])
        good_task_2 = np.array([])
        good_task_3 = np.array([])
        spikes_to_plot_b_task_1 = np.array([])
        spikes_to_plot_a_task_1 = np.array([])
        spikes_to_plot_b_task_2 = np.array([])
        spikes_to_plot_a_task_2 = np.array([])
        spikes_to_plot_b_task_3 = np.array([])
        spikes_to_plot_a_task_3 = np.array([])

        
    ############### Task 1 
        for a_bad_task_1 in aligned_a_bad_task_1:
            period_min_a_bad_task_1 = a_bad_task_1 - window_to_plot
            period_max_a_bad_task_1 = a_bad_task_1 + window_to_plot
            spikes_ind_a_bad_task_1 = spikes_times[(spikes_times >= period_min_a_bad_task_1) & (spikes_times<= period_max_a_bad_task_1)]
            spikes_to_save_a_bad_task_1 = (spikes_ind_a_bad_task_1 - a_bad_task_1)           
            spikes_to_plot_a_bad_task_1= np.append(spikes_to_plot_a_bad_task_1,spikes_to_save_a_bad_task_1)
            spikes_to_plot_a_task_1 = np.append(spikes_to_plot_a_task_1,spikes_to_save_a_bad_task_1)
            bad_task_1 = np.append(bad_task_1,spikes_to_save_a_bad_task_1)
            spikes_to_plot_a_bad_task_1_raster.append(spikes_to_save_a_bad_task_1)
            
    
        for a_good_task_1 in aligned_a_good_task_1:
            period_min_a_bad = a_good_task_1 - window_to_plot
            period_max_a_bad = a_good_task_1 + window_to_plot
            spikes_ind_a_good = spikes_times[(spikes_times >= period_min_a_bad) & (spikes_times<=period_max_a_bad)]
            spikes_to_save_a_good_task_1 = (spikes_ind_a_good - a_good_task_1)
            spikes_to_plot_a_good_task_1= np.append(spikes_to_plot_a_good_task_1,spikes_to_save_a_good_task_1) 
            spikes_to_plot_a_task_1 = np.append(spikes_to_plot_a_task_1,spikes_to_save_a_good_task_1)
            good_task_1 = np.append(good_task_1,spikes_to_save_a_good_task_1)
            spikes_to_plot_a_good_task_1_raster.append(spikes_to_save_a_good_task_1)
            
        for b_good_task_1 in aligned_b_good_task_1: 
            period_min_b_good = b_good_task_1 - window_to_plot
            period_max_b_good = b_good_task_1 + window_to_plot
            spikes_ind_b_good = spikes_times[(spikes_times >= period_min_b_good) & (spikes_times<=period_max_b_good)]
            spikes_to_save_b_good_task_1 = (spikes_ind_b_good - b_good_task_1)
            spikes_to_plot_b_good_task_1 = np.append(spikes_to_plot_b_good_task_1,spikes_to_save_b_good_task_1)  
            spikes_to_plot_b_task_1 = np.append(spikes_to_plot_b_task_1, spikes_to_save_b_good_task_1)
            good_task_1 = np.append(good_task_1,spikes_to_save_b_good_task_1)
            spikes_to_plot_b_good_task_1_raster.append(spikes_to_save_b_good_task_1)
          
            
        for b_bad_task_1 in aligned_b_bad_task_1:
            period_min_b_bad = b_bad_task_1 - window_to_plot
            period_max_b_bad = b_bad_task_1 + window_to_plot
            spikes_ind_b_bad = spikes_times[(spikes_times >= period_min_b_bad) & (spikes_times<=period_max_b_bad)]
            spikes_to_save_b_bad_task_1 = (spikes_ind_b_bad - b_bad_task_1)
            spikes_to_plot_b_bad_task_1= np.append(spikes_to_plot_b_bad_task_1,spikes_to_save_b_bad_task_1) 
            spikes_to_plot_b_task_1 = np.append(spikes_to_plot_b_task_1, spikes_to_save_b_bad_task_1)
            bad_task_1 = np.append(bad_task_1,spikes_to_save_b_bad_task_1)
            spikes_to_plot_b_bad_task_1_raster.append(spikes_to_save_b_bad_task_1)
            
    ################### Task 2
            
        for a_bad_task_2 in aligned_a_bad_task_2:
            period_min = a_bad_task_2 - window_to_plot
            period_max = a_bad_task_2 + window_to_plot
            spikes_ind_a_bad_task_2 = spikes_times[(spikes_times >= period_min) & (spikes_times<= period_max)]
            spikes_to_save_a_bad_task_2 = (spikes_ind_a_bad_task_2 - a_bad_task_2)
            spikes_to_plot_a_bad_task_2= np.append(spikes_to_plot_a_bad_task_2,spikes_to_save_a_bad_task_2)
            spikes_to_plot_a_task_2 = np.append(spikes_to_plot_a_task_2, spikes_to_save_a_bad_task_2)
            spikes_to_plot_a_bad_task_2_raster.append(spikes_to_save_a_bad_task_2)
            bad_task_2 = np.append(bad_task_2,spikes_to_save_a_bad_task_2)
            
        for a_good_task_2 in aligned_a_good_task_2:
            period_min_a_bad = a_good_task_2 - window_to_plot
            period_max_a_bad = a_good_task_2 + window_to_plot
            spikes_ind_a_good = spikes_times[(spikes_times >= period_min_a_bad) & (spikes_times<=period_max_a_bad)]

            spikes_to_save_a_good_task_2= (spikes_ind_a_good - a_good_task_2)
            spikes_to_plot_a_good_task_2 = np.append(spikes_to_plot_a_good_task_2,spikes_to_save_a_good_task_2) 
            spikes_to_plot_a_task_2 = np.append(spikes_to_plot_a_task_2, spikes_to_save_a_good_task_2)
            spikes_to_plot_a_good_task_2_raster.append(spikes_to_save_a_good_task_2)
            good_task_2 = np.append(good_task_2,spikes_to_save_a_good_task_2)
            
            
        for b_bad_task_2 in aligned_b_bad_task_2:
            period_min_b_bad = b_bad_task_2 - window_to_plot
            period_max_b_bad = b_bad_task_2 + window_to_plot
            spikes_ind_b_bad = spikes_times[(spikes_times >= period_min_b_bad) & (spikes_times<= period_max_b_bad)]
            spikes_to_save_b_bad_task_2 = (spikes_ind_b_bad - b_bad_task_2)
            spikes_to_plot_b_bad_task_2= np.append(spikes_to_plot_b_bad_task_2,spikes_to_save_b_bad_task_2) 
            spikes_to_plot_b_task_2 = np.append(spikes_to_plot_b_task_2, spikes_to_save_b_bad_task_2)
            spikes_to_plot_b_bad_task_2_raster.append(spikes_to_save_b_bad_task_2)
            bad_task_2 = np.append(bad_task_2,spikes_to_save_b_bad_task_2)
    
        for b_good in aligned_b_good_task_2:
            period_min_b_good = b_good - window_to_plot
            period_max_b_good = b_good + window_to_plot
            spikes_ind_b_good = spikes_times[(spikes_times >= period_min_b_good) & (spikes_times<=period_max_b_good)]

            spikes_to_save_b_good_task_2 = (spikes_ind_b_good - b_good)
            spikes_to_plot_b_good_task_2= np.append(spikes_to_plot_b_good_task_2,spikes_to_save_b_good_task_2) 
            spikes_to_plot_b_task_2 = np.append(spikes_to_plot_b_task_2, spikes_to_save_b_good_task_2)
            spikes_to_plot_b_good_task_2_raster.append(spikes_to_save_b_good_task_2)
            good_task_2 = np.append(good_task_2,spikes_to_save_b_good_task_2)
        
################### Task 3
            
        for a_bad_task_3 in aligned_a_bad_task_3:
            period_min = a_bad_task_3 - window_to_plot
            period_max = a_bad_task_3 + window_to_plot
            spikes_ind_a_bad_task_2 = spikes_times[(spikes_times >= period_min) & (spikes_times<= period_max)]

            spikes_to_save_a_bad_task_3 = (spikes_ind_a_bad_task_2 - a_bad_task_3)
            spikes_to_plot_a_bad_task_3 = np.append(spikes_to_plot_a_bad_task_3,spikes_to_save_a_bad_task_3)
            spikes_to_plot_a_task_3 = np.append(spikes_to_plot_a_task_3, spikes_to_save_a_bad_task_3)
            spikes_to_plot_a_bad_task_3_raster.append(spikes_to_save_a_bad_task_3)
            bad_task_3 = np.append(bad_task_3,spikes_to_save_a_bad_task_3)
            
        for a_good_task_3 in aligned_a_good_task_3:
            period_min_a_bad = a_good_task_3 - window_to_plot
            period_max_a_bad = a_good_task_3 + window_to_plot
            spikes_ind_a_good = spikes_times[(spikes_times >= period_min_a_bad) & (spikes_times<=period_max_a_bad)]
            spikes_to_save_a_good_task_3 = (spikes_ind_a_good - a_good_task_3)
            spikes_to_plot_a_good_task_3 = np.append(spikes_to_plot_a_good_task_3,spikes_to_save_a_good_task_3) 
            spikes_to_plot_a_task_3 = np.append(spikes_to_plot_a_task_3, spikes_to_save_a_good_task_3)
            spikes_to_plot_a_good_task_3_raster.append(spikes_to_save_a_good_task_3)
            good_task_3 = np.append(good_task_3,spikes_to_save_a_good_task_3)
            
        for b_bad_task_3 in aligned_b_bad_task_3:
            period_min_b_bad = b_bad_task_3 - window_to_plot
            period_max_b_bad = b_bad_task_3 + window_to_plot
            spikes_ind_b_bad = spikes_times[(spikes_times >= period_min_b_bad) & (spikes_times<=period_max_b_bad)]

            spikes_to_save_b_bad_task_3 = (spikes_ind_b_bad - b_bad_task_3)
            spikes_to_plot_b_bad_task_3= np.append(spikes_to_plot_b_bad_task_3,spikes_to_save_b_bad_task_3) 
            spikes_to_plot_b_task_3 = np.append(spikes_to_plot_b_task_3, spikes_to_save_b_bad_task_3)
            spikes_to_plot_b_bad_task_3_raster.append(spikes_to_save_b_bad_task_3)
            bad_task_3 = np.append(bad_task_3,spikes_to_save_b_bad_task_3)
    
        for b_good in aligned_b_good_task_3:
            period_min_b_good = b_good - window_to_plot
            period_max_b_good = b_good + window_to_plot
            spikes_ind_b_good = spikes_times[(spikes_times >= period_min_b_good) & (spikes_times<=period_max_b_good)]
            spikes_to_save_b_good_task_3 = (spikes_ind_b_good - b_good)
            spikes_to_plot_b_good_task_3= np.append(spikes_to_plot_b_good_task_3,spikes_to_save_b_good_task_3) 
            spikes_to_plot_b_task_3 = np.append(spikes_to_plot_b_task_3, spikes_to_save_b_good_task_3)
            spikes_to_plot_b_good_task_3_raster.append(spikes_to_save_b_good_task_3)
            good_task_3 = np.append(good_task_3,spikes_to_save_b_good_task_3)
        
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
            pl.xlim(-window_to_plot, +window_to_plot)
            
        x = [-window_to_plot, window_to_plot]
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
            axes_8[0][i].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
            axes_8[0][i].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_8[0][i].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
            axes_8[0][i].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            axes_8[0][0].set(ylabel ='Trial #')
            axes_8[0][i].set_title('{}'.format(cluster))
          
        elif i >= group_1 and i < group_2:
            pl.figure(4)
            axes_16[0][i-group_1].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
            axes_16[0][i-group_1].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_16[0][i-group_1].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
            axes_16[0][i-group_1].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            axes_16[0][0].set(ylabel ='Trial #')
            axes_16[0][i-group_1].set_title('{}'.format(cluster))
            
        elif i >= group_2 and i < group_3:
            pl.figure(6) 
            axes_24[0][i-group_2].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
            axes_24[0][i-group_2].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_24[0][i-group_2].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
            axes_24[0][i-group_2].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            axes_24[0][0].set(ylabel ='Trial #')
            axes_24[0][i-group_2].set_title('{}'.format(cluster))
        elif i >= group_3 and i < group_4:
            pl.figure(8) 
            axes_36[0][i-group_3].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
            axes_36[0][i-group_3].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            axes_36[0][i-group_3].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
            axes_36[0][i-group_3].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
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
            pl.xlim(-window_to_plot, +window_to_plot)

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
            if poke_A_task_2 == poke_A:
                axes_8[1][i].fill_between(x,t0, t1_task_2 ,color = 'black', alpha = 0.5)
                axes_8[1][i].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            elif poke_A_task_2 == poke_B:
                axes_8[1][i].fill_between(x,t0, t1_task_2 ,color = 'cadetblue', alpha = 0.5)
                axes_8[1][i].fill_between(x,t1_task_2,t2_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                 axes_8[1][i].fill_between(x,t0, t1_task_2 ,color = 'olive', alpha = 0.5)
                 axes_8[1][i].fill_between(x,t1_task_2,t2_task_2,color = 'olive', alpha = 0.5)  
            if poke_B_task_2 == poke_A:
                axes_8[1][i].fill_between(x,t2_task_2,t3_task_2,color = 'black', alpha = 0.5)
                axes_8[1][i].fill_between(x, t3_task_2,t4_task_2,color = 'black', alpha = 0.5)
            elif poke_B_task_2 == poke_B:
                axes_8[1][i].fill_between(x,t2_task_2,t3_task_2,color = 'cadetblue', alpha = 0.5)
                axes_8[1][i].fill_between(x, t3_task_2,t4_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                axes_8[1][i].fill_between(x,t2_task_2,t3_task_2,color = 'olive', alpha = 0.5)
                axes_8[1][i].fill_between(x, t3_task_2,t4_task_2,color = 'olive', alpha = 0.5)
            axes_8[1][0].set(ylabel ='Trial #')   
        elif i >= group_1 and i < group_2:
            pl.figure(4)
            if poke_A_task_2 == poke_A:
                axes_16[1][i-group_1].fill_between(x,t0, t1_task_2 ,color = 'black', alpha = 0.5)
                axes_16[1][i-group_1].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            elif poke_A_task_2 == poke_B:
                axes_16[1][i-group_1].fill_between(x,t0, t1_task_2 ,color = 'cadetblue', alpha = 0.5)
                axes_16[1][i-group_1].fill_between(x,t1_task_2,t2_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                 axes_16[1][i-group_1].fill_between(x,t0, t1_task_2 ,color = 'olive', alpha = 0.5)
                 axes_16[1][i-group_1].fill_between(x,t1_task_2,t2_task_2,color = 'olive', alpha = 0.5)  
            if poke_B_task_2 == poke_A:
                axes_16[1][i-group_1].fill_between(x,t2_task_2,t3_task_2,color = 'black', alpha = 0.5)
                axes_16[1][i-group_1].fill_between(x, t3_task_2,t4_task_2,color = 'black', alpha = 0.5)
            elif poke_B_task_2 == poke_B:
                axes_16[1][i-group_1].fill_between(x,t2_task_2,t3_task_2,color = 'cadetblue', alpha = 0.5)
                axes_16[1][i-group_1].fill_between(x, t3_task_2,t4_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                axes_16[1][i-group_1].fill_between(x,t2_task_2,t3_task_2,color = 'olive', alpha = 0.5)
                axes_16[1][i-group_1].fill_between(x, t3_task_2,t4_task_2,color = 'olive', alpha = 0.5)
                
            axes_16[1][0].set(ylabel ='Trial #')   
            
        elif i >= group_2 and i < group_3: 
            pl.figure(6) 
            if poke_A_task_2 == poke_A:
                axes_24[1][i-group_2].fill_between(x,t0, t1_task_2 ,color = 'black', alpha = 0.5)
                axes_24[1][i-group_2].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            elif poke_A_task_2 == poke_B:
                axes_24[1][i-group_2].fill_between(x,t0, t1_task_2 ,color = 'cadetblue', alpha = 0.5)
                axes_24[1][i-group_2].fill_between(x,t1_task_2,t2_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                axes_24[1][i-group_2].fill_between(x,t0, t1_task_2 ,color = 'olive', alpha = 0.5)
                axes_24[1][i-group_2].fill_between(x,t1_task_2,t2_task_2,color = 'olive', alpha = 0.5)  
            if poke_B_task_2 == poke_A:
                axes_24[1][i-group_2].fill_between(x,t2_task_2,t3_task_2,color = 'black', alpha = 0.5)
                axes_24[1][i-group_2].fill_between(x, t3_task_2,t4_task_2,color = 'black', alpha = 0.5)
            elif poke_B_task_2 == poke_B:
                axes_24[1][i-group_2].fill_between(x,t2_task_2,t3_task_2,color = 'cadetblue', alpha = 0.5)
                axes_24[1][i-group_2].fill_between(x, t3_task_2,t4_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                axes_24[1][i-group_2].fill_between(x,t2_task_2,t3_task_2,color = 'olive', alpha = 0.5)
                axes_24[1][i-group_2].fill_between(x, t3_task_2,t4_task_2,color = 'olive', alpha = 0.5)
           
            axes_24[1][0].set(ylabel ='Trial #')
            
        elif i >= group_3 and i < group_4: 
            pl.figure(8) 
            if poke_A_task_2 == poke_A:
                axes_36[1][i-group_3].fill_between(x,t0, t1_task_2 ,color = 'black', alpha = 0.5)
                axes_36[1][i-group_3].fill_between(x,t1_task_2,t2_task_2,color = 'black', alpha = 0.5)
            elif poke_A_task_2 == poke_B:
                axes_36[1][i-group_3].fill_between(x,t0, t1_task_2 ,color = 'cadetblue', alpha = 0.5)
                axes_36[1][i-group_3].fill_between(x,t1_task_2,t2_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                axes_36[1][i-group_3].fill_between(x,t0, t1_task_2 ,color = 'olive', alpha = 0.5)
                axes_36[1][i-group_3].fill_between(x,t1_task_2,t2_task_2,color = 'olive', alpha = 0.5)  
            if poke_B_task_2 == poke_A:
                axes_36[1][i-group_3].fill_between(x,t2_task_2,t3_task_2,color = 'black', alpha = 0.5)
                axes_36[1][i-group_3].fill_between(x, t3_task_2,t4_task_2,color = 'black', alpha = 0.5)
            elif poke_B_task_2 == poke_B:
                axes_36[1][i-group_3].fill_between(x,t2_task_2,t3_task_2,color = 'cadetblue', alpha = 0.5)
                axes_36[1][i-group_3].fill_between(x, t3_task_2,t4_task_2,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                axes_36[1][i-group_3].fill_between(x,t2_task_2,t3_task_2,color = 'olive', alpha = 0.5)
                axes_36[1][i-group_3].fill_between(x, t3_task_2,t4_task_2,color = 'olive', alpha = 0.5)
                

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
            pl.xlim(-5000, +5000)
        
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
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                axes_8[2][i].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_8[2][i].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_A:
                axes_8[2][i].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_8[2][i].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_B:
                axes_8[2][i].fill_between(x, t1, t0 ,color = 'cadetblue', alpha = 0.5)
                axes_8[2][i].fill_between(x,t2,t1,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                 axes_8[2][i].fill_between(x, t1, t0 ,color = 'palevioletred', alpha = 0.5)
                 axes_8[2][i].fill_between(x,t2,t1,color = 'palevioletred', alpha = 0.5)
                 
            if poke_B_task_3 == poke_A:
                axes_8[2][i].fill_between(x,t3,t2,color = 'black', alpha = 0.5)
                axes_8[2][i].fill_between(x,t4,t3,color = 'black', alpha = 0.5)
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                axes_8[2][i].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
                axes_8[2][i].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                axes_8[2][i].fill_between(x,t3,t2,color = 'palevioletred', alpha = 0.5)
                axes_8[2][i].fill_between(x,t4,t3,color = 'palevioletred', alpha = 0.5)
            
            axes_8[2][i].set(xlabel ='Time (ms)')
            axes_8[2][0].set(ylabel ='Trial #')
        elif i >= group_1 and i < group_2:
            pl.figure(4)
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                axes_16[2][i-group_1].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_A:
                axes_16[2][i-group_1].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_B:
                axes_16[2][i-group_1].fill_between(x, t1, t0 ,color = 'cadetblue', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t2,t1,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                axes_16[2][i-group_1].fill_between(x, t1, t0 ,color = 'palevioletred', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t2,t1,color = 'palevioletred', alpha = 0.5)
                 
            if poke_B_task_3 == poke_A:
                axes_16[2][i-group_1].fill_between(x,t3,t2,color = 'black', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t4,t3,color = 'black', alpha = 0.5)
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                axes_16[2][i-group_1].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                axes_16[2][i-group_1].fill_between(x,t3,t2,color = 'palevioletred', alpha = 0.5)
                axes_16[2][i-group_1].fill_between(x,t4,t3,color = 'palevioletred', alpha = 0.5)

            axes_16[2][i-group_1].set(xlabel ='Time (ms)')
            axes_16[2][0].set(ylabel ='Trial #')
            
        elif i >= group_2 and i < group_3: 
            pl.figure(6) 
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                axes_24[2][i-group_2].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_A:
                axes_24[2][i-group_2].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_B:
                axes_24[2][i-group_2].fill_between(x, t1, t0 ,color = 'cadetblue', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t2,t1,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                axes_24[2][i-group_2].fill_between(x, t1, t0 ,color = 'palevioletred', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t2,t1,color = 'palevioletred', alpha = 0.5)
                 
            if poke_B_task_3 == poke_A:
                axes_24[2][i-group_2].fill_between(x,t3,t2,color = 'black', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t4,t3,color = 'black', alpha = 0.5)
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                axes_24[2][i-group_2].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                axes_24[2][i-group_2].fill_between(x,t3,t2,color = 'palevioletred', alpha = 0.5)
                axes_24[2][i-group_2].fill_between(x,t4,t3,color = 'palevioletred', alpha = 0.5)

            axes_24[2][i-group_2].set(xlabel ='Time (ms)')
            axes_24[2][0].set(ylabel ='Trial #')
                   
        elif i >= group_3 and i < group_4: 
            pl.figure(8) 
            if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                axes_36[2][i-group_3].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_A:
                axes_36[2][i-group_3].fill_between(x, t1, t0 ,color = 'black', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t2,t1,color = 'black', alpha = 0.5)
            elif poke_A_task_3 == poke_B:
                axes_36[2][i-group_3].fill_between(x, t1, t0 ,color = 'cadetblue', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t2,t1,color = 'cadetblue', alpha = 0.5)
            elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                axes_36[2][i-group_3].fill_between(x, t1, t0 ,color = 'palevioletred', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t2,t1,color = 'palevioletred', alpha = 0.5)
                 
            if poke_B_task_3 == poke_A:
                axes_36[2][i-group_3].fill_between(x,t3,t2,color = 'black', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t4,t3,color = 'black', alpha = 0.5)
            elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                axes_36[2][i-group_3].fill_between(x,t3,t2,color = 'cadetblue', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t4,t3,color = 'cadetblue', alpha = 0.5)
            elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                axes_36[2][i-group_3].fill_between(x,t3,t2,color = 'palevioletred', alpha = 0.5)
                axes_36[2][i-group_3].fill_between(x,t4,t3,color = 'palevioletred', alpha = 0.5)
                
    
            axes_36[2][i-group_3].set(xlabel ='Time (s)')
            axes_36[2][0].set(ylabel ='Trial #')
          
        
        bin_width_ms = 1
        smooth_sd_ms = 100
        fr_convert = 1000
        sns.set(style="white", palette="muted", color_codes = True)
        trial_duration = 10000
        bin_edges_trial = np.arange(-5000,trial_duration, bin_width_ms)


        if correct == 'Standard':
            #Plotting A good Task 1
            spikes_to_plot_a_good_task_1 = spikes_to_plot_a_good_task_1[~np.isnan(spikes_to_plot_a_good_task_1)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_good_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_a_good_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='Poke {} Task 1'.format(poke_A))
            elif i >= group_1 and i < group_2:
                ax_16[0][i- group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='Poke {} Task 1'.format(poke_A)) 
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='Poke {} Task 1'.format(poke_A))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='Poke {} Task 1'.format(poke_A))
            
        
            #Plotting A Bad Task 1 
    
            spikes_to_plot_a_bad_task_1 = spikes_to_plot_a_bad_task_1[~np.isnan(spikes_to_plot_a_bad_task_1)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_bad_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_a_bad_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='Poke {} Bad Task 1'.format(poke_A))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='Poke {} Bad Task 1'.format(poke_A))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='Poke {} Bad Task 1'.format(poke_A))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='Poke {} Bad Task 1'.format(poke_A))
    
    
            #Plotting B good Task 1
    
            spikes_to_plot_b_good_task_1 = spikes_to_plot_b_good_task_1[~np.isnan(spikes_to_plot_b_good_task_1)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_good_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_b_good_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} good Task 1'.format(poke_B))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue',label='{} good Task 1'.format(poke_B))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} good Task 1'.format(poke_B))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} good Task 1'.format(poke_B))
    
            
            #Plotting B good Task 1
            spikes_to_plot_b_bad_task_1 = spikes_to_plot_b_bad_task_1[~np.isnan(spikes_to_plot_b_bad_task_1)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_bad_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_b_bad_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 1'.format(poke_B))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 1'.format(poke_B))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 1'.format(poke_B))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 1'.format(poke_B))
    
            #Plotting A good Task 2 
            spikes_to_plot_a_good_task_2 = spikes_to_plot_a_good_task_2[~np.isnan(spikes_to_plot_a_good_task_2)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_good_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_a_good_task_2_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_A_task_2 == poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_A_task_2))  
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_A_task_2))  
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_A_task_2 == poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_A_task_2))
            elif i >= group_2 and i < group_3: 
                pl.figure(5)
                if poke_A_task_2 == poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_A_task_2))
            elif i >= group_3 and i < group_4: 
                pl.figure(7)
                if poke_A_task_2 == poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_A_task_2))
    
                    
            #Plotting A Bad Task 2 
            spikes_to_plot_a_bad_task_2 = spikes_to_plot_a_bad_task_2[~np.isnan(spikes_to_plot_a_bad_task_2)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_bad_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_a_bad_task_2_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_A_task_2 == poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'olive', label='{} bad Task 2'.format(poke_A_task_2))
                    
    
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_A_task_2 == poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'olive', label='{} bad Task 2'.format(poke_A_task_2))
    
            elif  i >= group_2 and i < group_3: 
                pl.figure(5)
                if poke_A_task_2 == poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'olive', label='{} bad Task 2'.format(poke_A_task_2))
    
            elif  i >= group_3 and i < group_4: 
                pl.figure(7)
                if poke_A_task_2 == poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue', label='{} bad Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'olive', label='{} bad Task 2'.format(poke_A_task_2))
    
    
    
        
            #Plotting B good Task 2 
            spikes_to_plot_b_good_task_2 = spikes_to_plot_b_good_task_2[~np.isnan(spikes_to_plot_b_good_task_2)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_good_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_b_good_list_task_2)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_B_task_2 == poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_B_task_2))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_B_task_2 == poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_B_task_2))
            elif i >= group_2 and i < group_3: 
                pl.figure(5)
                if poke_B_task_2 == poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_B_task_2))
            elif i >= group_3 and i < group_4: 
                pl.figure(7)
                if poke_B_task_2 == poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} good Task 2'.format(poke_B_task_2))
            
            #Plotting B Bad Task 2 
            spikes_to_plot_b_bad_task_2 = spikes_to_plot_b_bad_task_2[~np.isnan(spikes_to_plot_b_bad_task_2)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_bad_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_b_bad_list_task_2)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_B_task_2 == poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "olive", label='{} bad Task 2'.format(poke_B_task_2))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_B_task_2 == poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "olive", label='{} bad Task 2'.format(poke_B_task_2))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                if poke_B_task_2 == poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "olive", label='{} bad Task 2'.format(poke_B_task_2))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_B_task_2 == poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue", label='{} bad Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "olive", label='{} bad Task 2'.format(poke_B_task_2))
                    
            #Plotting A good Task 3
            spikes_to_plot_a_good_task_3 = spikes_to_plot_a_good_task_3[~np.isnan(spikes_to_plot_a_good_task_3)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_good_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_a_good_task_3_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_A_task_3))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_A_task_3))
            elif i >= group_2 and i < group_3:
                pl.figure(5)        
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_A_task_3))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue",label='{} good Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_A_task_3))
                    
            #Plotting A Bad Task 3
            spikes_to_plot_a_bad_task_3 = spikes_to_plot_a_bad_task_3[~np.isnan(spikes_to_plot_a_bad_task_3)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_bad_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_a_bad_task_3_list)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black', label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'palevioletred',  label='{} bad Task 3'.format(poke_A_task_3))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3)) 
                elif poke_A_task_3 == poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue',  label='{} bad Task 3'.format(poke_A_task_3)) 
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'palevioletred',  label='{} bad Task 3'.format(poke_A_task_3)) 
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'palevioletred',  label='{} bad Task 3'.format(poke_A_task_3))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'black',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'cadetblue',  label='{} bad Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--', color = 'palevioletred',  label='{} bad Task 3'.format(poke_A_task_3))
        
            #Plotting B good Task 3 
            spikes_to_plot_b_good_task_3 = spikes_to_plot_b_good_task_3[~np.isnan(spikes_to_plot_b_good_task_3)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_good_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_b_good_list_task_3)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_B_task_3 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black",  label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_B_task_3))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_B_task_3 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_B_task_3))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                if poke_B_task_3 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} good Task 3'.format(poke_B_task_3))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_B_task_3 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} good Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred",label='{} good Task 3'.format(poke_B_task_3))
                    
            #Plotting B Bad Task 3 
            spikes_to_plot_b_bad_task_3 = spikes_to_plot_b_bad_task_3[~np.isnan(spikes_to_plot_b_bad_task_3)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_bad_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/len(entry_b_bad_list_task_3)
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_B_task_3 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black", label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue",  label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "palevioletred",  label='{} bad Task 3'.format(poke_B_task_3))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_B_task_3 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black",  label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue",  label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "palevioletred", label='{} bad Task 3'.format(poke_B_task_3))
            elif i >= group_2 and i < group_3: 
                pl.figure(5)
                if poke_B_task_3 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black",  label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue",  label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "palevioletred",  label='{} bad Task 3'.format(poke_B_task_3)) 
            elif i >= group_3 and i < group_4: 
                pl.figure(7)
                if poke_B_task_3 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "black", label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "cadetblue",  label='{} bad Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,'--',color = "palevioletred", label='{} bad Task 3'.format(poke_B_task_3))
            
        elif correct == 'Correct':
            #Plotting A good Task 1
            spikes_to_plot_a_task_1 = spikes_to_plot_a_task_1[~np.isnan(spikes_to_plot_a_task_1)]
            good_task_1 = good_task_1[~np.isnan(good_task_1)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_a_good_list) + len(entry_a_bad_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 1'.format(poke_A))
            elif i >= group_1 and i < group_2:
                ax_16[0][i- group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 1'.format(poke_A)) 
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 1'.format(poke_A))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 1'.format(poke_A))
           
            # Plotting B Task 1 
            spikes_to_plot_b_task_1 = spikes_to_plot_b_task_1[~np.isnan(spikes_to_plot_b_task_1)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_b_good_list) + len(entry_b_bad_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 1'.format(poke_B))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 1'.format(poke_B))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 1'.format(poke_B))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 1'.format(poke_B))
            
            # Plotting A task 2
            spikes_to_plot_a_task_2 = spikes_to_plot_a_task_2[~np.isnan(spikes_to_plot_a_task_2)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_a_bad_task_2_list) + len(entry_a_good_task_2_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_A_task_2 == poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'olive', label='{} Task 2'.format(poke_A_task_2))
                    
    
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_A_task_2 == poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue',label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'olive', label='{} Task 2'.format(poke_A_task_2))
    
            elif  i >= group_2 and i < group_3: 
                pl.figure(5)
                if poke_A_task_2 == poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'olive', label='{} Task 2'.format(poke_A_task_2))
    
            elif  i >= group_3 and i < group_4: 
                pl.figure(7)
                if poke_A_task_2 == poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'black', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 == poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'cadetblue', label='{} Task 2'.format(poke_A_task_2))
                elif poke_A_task_2 != poke_B and poke_A_task_2 != poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'olive', label='{} Task 2'.format(poke_A_task_2))
            
            #Plotting B Task 2 
            spikes_to_plot_b_task_2 = spikes_to_plot_b_task_2[~np.isnan(spikes_to_plot_b_task_2)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_b_bad_list_task_2) + len(entry_b_good_list_task_2))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_B_task_2 == poke_A:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} Task 2'.format(poke_B_task_2))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_B_task_2 == poke_A:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} Task 2'.format(poke_B_task_2))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                if poke_B_task_2 == poke_A:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} Task 2'.format(poke_B_task_2))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_B_task_2 == poke_A:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 == poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 2'.format(poke_B_task_2))
                elif poke_B_task_2 != poke_A and poke_B_task_2 != poke_B:
                    ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "olive", label='{} Task 2'.format(poke_B_task_2))
                    
            #Plotting A good Task 3
            spikes_to_plot_a_task_3 = spikes_to_plot_a_task_3[~np.isnan(spikes_to_plot_a_task_3)]
            hist_task,edges_task = np.histogram(spikes_to_plot_a_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_a_good_task_3_list) + len(entry_a_bad_task_3_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_A_task_3))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_A_task_3))
            elif i >= group_2 and i < group_3:
                pl.figure(5)        
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_A_task_3))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 == poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_A_task_3))
                elif poke_A_task_3 != poke_A_task_2 and poke_A_task_3 != poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_A_task_3))
            
            #Plotting B Task 3 
            spikes_to_plot_b_task_3 = spikes_to_plot_b_task_3[~np.isnan(spikes_to_plot_b_task_3)]
            hist_task,edges_task = np.histogram(spikes_to_plot_b_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_b_good_list_task_3) + len(entry_b_bad_list_task_3 ))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                if poke_B_task_3 == poke_A:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue",label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred",label='{} Task 3'.format(poke_B_task_3))
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                if poke_B_task_3 == poke_A:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_B_task_3))
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                if poke_B_task_3 == poke_A:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_B_task_3))
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                if poke_B_task_3 == poke_A:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "black", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "cadetblue", label='{} Task 3'.format(poke_B_task_3))
                elif poke_B_task_3 != poke_B_task_2 and poke_B_task_3 != poke_B:
                    ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "palevioletred", label='{} Task 3'.format(poke_B_task_3))
        
        elif correct == 'Meaning':
            #Plotting A good Task 1
            good_task_1 = good_task_1[~np.isnan(good_task_1)]
            hist_task,edges_task = np.histogram(good_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_a_good_list) + len(entry_b_good_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 1')
            elif i >= group_1 and i < group_2:
                ax_16[0][i- group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 1') 
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 1')
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 1')
           
            # Plotting Bad Task 1
            bad_task_1 = bad_task_1[~np.isnan(bad_task_1)]
            hist_task,edges_task = np.histogram(bad_task_1, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_a_bad_list) + len(entry_b_bad_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[0][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'grey', label='Bad Task 1')
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[0][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'grey', label='Bad Task 1')
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[0][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'grey', label='Bad Task 1')
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[0][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'grey', label='Bad Task 1')
            
            # Plotting Good task 2
            good_task_2 = good_task_2[~np.isnan(good_task_2)]
            hist_task,edges_task = np.histogram(good_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_b_good_list_task_2) + len(entry_a_good_task_2_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 2')
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 2')
            elif  i >= group_2 and i < group_3: 
                pl.figure(5)
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 2')    
            elif  i >= group_3 and i < group_4: 
                pl.figure(7)
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = 'navy', label='Good Task 2')
            
            #Plotting Bad Task 2 
            bad_task_2 = bad_task_2[~np.isnan(bad_task_2)]
            hist_task,edges_task = np.histogram(bad_task_2, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_b_bad_list_task_2) + len(entry_a_bad_task_2_list))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[1][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 2')
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[1][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 2')
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[1][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 2')
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[1][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 2')
                    
            #Plotting Good Task 3
            good_task_3 = good_task_3[~np.isnan(good_task_3)]
            hist_task,edges_task = np.histogram(good_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_a_good_task_3_list) + len(entry_b_good_list_task_3))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "navy", label='Good Task 3')
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "navy", label='Good Task 3')
            elif i >= group_2 and i < group_3:
                pl.figure(5)        
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "navy", label='Good Task 3')
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "navy", label='Good Task 3')
            
            #Plotting Bad Task 3 
            bad_task_3 = bad_task_3[~np.isnan(bad_task_3)]
            hist_task,edges_task = np.histogram(bad_task_3, bins= bin_edges_trial)# histogram per second
            hist_task = hist_task/(len(entry_b_bad_list_task_3) + len(entry_a_bad_task_3_list ))
            hist_task = hist_task/bin_width_ms
            normalised_task = gaussian_filter1d(hist_task.astype(float), smooth_sd_ms/bin_width_ms)
            if i < group_1:
                pl.figure(1)
                ax_8[2][i].plot(bin_edges_trial[:-1], normalised_task*fr_convert, color = "grey", label='Bad Task 3')
            elif i >= group_1 and i < group_2:
                pl.figure(3)
                ax_16[2][i-group_1].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 3')
            elif i >= group_2 and i < group_3:
                pl.figure(5)
                ax_24[2][i-group_2].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 3')
            elif i >= group_3 and i < group_4:
                pl.figure(7)
                ax_36[2][i-group_3].plot(bin_edges_trial[:-1], normalised_task*fr_convert,color = "grey", label='Bad Task 3')
                                            
        pl.xlim(-window_to_plot, +window_to_plot)
        
        if i < group_1:

            ax_8[1][0].legend(fontsize = 'xx-small')
            ax_8[0][0].legend(fontsize = 'xx-small')
            ax_8[2][0].legend(fontsize = 'xx-small')
            ax_8[0][i].set_title('{}'.format(cluster))
            ax_8[0][0].set(ylabel='Firing Rate (Hz)')
            ax_8[1][0].set(ylabel='Firing Rate (Hz)')
            ax_8[2][0].set(ylabel='Firing Rate (Hz)')
            ax_8[2][i].set(xlabel= 'Time (ms)')
        elif i >= group_1 and i < group_2:

            ax_16[1][0].legend(fontsize = 'xx-small')
            ax_16[0][0].legend(fontsize = 'xx-small')
            ax_16[2][0].legend(fontsize = 'xx-small')
            ax_16[0][i-group_1].set_title('{}'.format(cluster))            
            ax_16[0][0].set(ylabel='Firing Rate (Hz)')
            ax_16[1][0].set(ylabel='Firing Rate (Hz)')
            ax_16[2][0].set(ylabel='Firing Rate (Hz)')
            ax_16[2][i-group_1].set(xlabel= 'Time (ms)')
        elif i >= group_2 and i < group_3:     
 
            ax_24[1][0].legend(fontsize = 'xx-small')
            ax_24[0][0].legend(fontsize = 'xx-small')
            ax_24[2][0].legend(fontsize = 'xx-small')
            ax_24[0][i-group_2].set_title('{}'.format(cluster))
            ax_24[0][0].set(ylabel='Firing Rate (Hz)')
            ax_24[1][0].set(ylabel='Firing Rate (Hz)')
            ax_24[2][0].set(ylabel='Firing Rate (Hz)')
            ax_24[2][i-group_2].set(xlabel= 'Time (ms)')
            
            
    if correct == 'Standard':        
        fig_clusters_8.savefig(path.join(outpath, '%s  Histograms_G1.pdf' %beh_session.datetime), dpi = 1)
        #fig_raster_8.savefig(path.join(outpath,'%s  Raster_G1.png' %beh_session.datetime ))
        fig_clusters_16.savefig(path.join(outpath,'%s  Histograms_G2.pdf' %beh_session.datetime))
        #fig_raster_16.savefig(path.join(outpath,'%s  Raster_G2.png' %beh_session.datetime))
        fig_clusters_24.savefig(path.join(outpath,'%s  Histograms_G3.pdf' %beh_session.datetime))
        #fig_raster_24.savefig(path.join(outpath, '%s  Raster_G3.png'%beh_session.datetime))
        fig_clusters_36.savefig(path.join(outpath,'%s  Histograms_G4.pdf' %beh_session.datetime))
        fig_raster_36.savefig(path.join(outpath, '%s  Raster_G4.pdf' %beh_session.datetime ))
    elif correct == 'Correct':
        fig_clusters_8.savefig(path.join(outpath, '%s Spatial Histograms_G1.pdf' %beh_session.datetime))
        #fig_raster_8.savefig(path.join(outpath,'%s Spatial Raster_G1.png' %beh_session.datetime ))
        fig_clusters_16.savefig(path.join(outpath,'%s Spatial Histograms_G2.pdf' %beh_session.datetime))
        #fig_raster_16.savefig(path.join(outpath,'%s Spatial Raster_G2.png' %beh_session.datetime))
        fig_clusters_24.savefig(path.join(outpath,'%s Spatial Histograms_G3.pdf' %beh_session.datetime))
        #fig_raster_24.savefig(path.join(outpath, '%s Spatial Raster_G3.png'%beh_session.datetime))
        fig_clusters_36.savefig(path.join(outpath,'%s Spatial Histograms_G4.pdf' %beh_session.datetime))
        #fig_raster_36.savefig(path.join(outpath, '%s Spatial Raster_G4.pdf' %beh_session.datetime ))

    

# Session Plot
def session_spikes_vs_trials_plot(ephys_session,beh_session, cluster):
    bin_width_ms = 1000
    smooth_sd_ms = 2000
    pyControl_choice = [event.time for event in beh_session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
    spikes_cluster_8 = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
    session_duration_ms = int(np.nanmax(ephys_session.time))- int(np.nanmin(ephys_session.time))
    bin_edges = np.arange(0,session_duration_ms, bin_width_ms)
    spikes_times_8 = np.array(spikes_cluster_8['time'])
    spikes_times_8 = spikes_times_8[~np.isnan(spikes_times_8)]
    hist,edges = np.histogram(spikes_times_8, bins= bin_edges)# histogram per second
    normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
    pl.figure()
    pl.plot(bin_edges[:-1]/1000, normalised/max(normalised), label='Firing Rate', color ='cadetblue') 

    trial_rate,edges_py = np.histogram(pyControl_choice, bins=bin_edges)
    trial_rate = gaussian_filter1d(trial_rate.astype(float), smooth_sd_ms/bin_width_ms)
    pl.plot(bin_edges[:-1]/1000, trial_rate/max(trial_rate), label='Rate', color = 'lightblue')
    pl.xlabel('Time (ms)')
    pl.ylabel('Smoothed Firing Rate')
    pl.legend()



