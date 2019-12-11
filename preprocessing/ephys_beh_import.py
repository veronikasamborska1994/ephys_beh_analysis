#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:37:21 2018

@author: behrenslab
"""
# =============================================================================
# Create data objects with ephys and behaviour together, some utility funcs 
# =============================================================================

import os
import numpy as np
import data_import as di
import re
import datetime
import copy 
from datetime import datetime

#ephys_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/neurons'
#beh_path = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Reversal_learning Behaviour Data and Code/data_3_tasks_ephys'

def import_code(ephys_path,beh_path, lfp_analyse = 'True'):
    subjects_ephys = os.listdir(ephys_path)
    subjects_ephys = [subject for subject in subjects_ephys if not subject.startswith('.')] #Exclude .DS store from subject list
    subjects_beh = os.listdir(beh_path)
    
    m480 = []
    m483 = []
    m479 = []
    m486 = []
    m478 = []
    m481 = []
    m484 = []
    
    for subject_ephys in subjects_ephys: 
        # Go through each animal
        subject_subfolder = ephys_path + '/' + subject_ephys
        subject_sessions = os.listdir(subject_subfolder)
        # List all ephys_sessions
        subject_sessions = [session for session in subject_sessions if not session.startswith('.')] #Exclude .DS store from subject list
        subject_sessions = [session for session in subject_sessions if not session.startswith('LFP')] #Exclude LFP from subject list
        subject_sessions = [session for session in subject_sessions if not session.startswith('MUA')] #Exclude MUA from subject list

        # List all ephys sessions in LFP folder
        if lfp_analyse == 'True':
            lfp_folder = subject_subfolder + '/'  + 'LFP'
            lfp_sessions = os.listdir(lfp_folder)
            lfp_sessions = [s for s in lfp_sessions if not s.startswith('.')] #Exclude .DS store from subject list
    

        for session in subject_sessions:
            match_ephys = re.search(r'\d{4}-\d{2}-\d{2}', session)
            date_ephys = datetime.strptime(match_ephys.group(), '%Y-%m-%d').date()
            date_ephys = match_ephys.group()
            
            for subject in subjects_beh:
                if subject == subject_ephys:
                    subject_beh_subfolder = beh_path + '/' + subject
                    subject_beh_sessions = os.listdir(subject_beh_subfolder)
                    subject_beh_sessions = [session for session in subject_beh_sessions if not session.startswith('.')] #Exclude .DS store from subject list
                    for beh_session in subject_beh_sessions:
                        match_behaviour = re.search(r'\d{4}-\d{2}-\d{2}', beh_session)
                        date_behaviour = datetime.strptime(match_behaviour.group(), '%Y-%m-%d').date()
                        date_behaviour = match_behaviour.group()
                        if date_ephys == date_behaviour:
                            behaviour_path = subject_beh_subfolder +'/'+beh_session
                            behaviour_session = di.Session(behaviour_path)
                            neurons_path = subject_subfolder+'/'+session 
                            neurons = np.load(neurons_path)
                            neurons = neurons[:,~np.isnan(neurons[1,:])]
                            behaviour_session.ephys = neurons
                            
                            # Exclude sessions where ephys software stopped working in the middle of a session or no neurons for some reason 
                            
                            if lfp_analyse == 'True':
                                if (subject_ephys == 'm484'): #or (subject_ephys == 'm484') :#r (subject_ephys == 'm484'):
                                    for s in lfp_sessions: 
                                         match_lfp = re.search(r'\d{4}-\d{2}-\d{2}', s)
                                         date_lfp = datetime.strptime(match_lfp.group(), '%Y-%m-%d').date()
                                         date_lfp = match_lfp.group()
                                         if date_lfp == date_behaviour: 
                                            lfp_path = subject_subfolder+'/'+'LFP''/'+subject_ephys+'_'+ date_lfp+ '.npy'
                                            lfp = np.load(lfp_path)
                                            lfp_time = lfp[0,:]
                                            lfp_signal = lfp[1:,:]
                                            lfp_nan = lfp_signal[:,~np.isnan(lfp_time)]
                                            lfp_time_ex_nan = lfp_time[~np.isnan(lfp_time)]
                                            behaviour_session.lfp = lfp_nan
                                            behaviour_session.lfp_time = lfp_time_ex_nan     
                                            
                            if behaviour_session.file_name != 'm479-2018-08-12-150904.txt' and behaviour_session.file_name != 'm484-2018-08-12-150904.txt'\
                            and behaviour_session.file_name !='m483-2018-07-27-164242.txt' and behaviour_session.file_name != 'm480-2018-08-22-111012.txt':
                                if subject_ephys == 'm480':
                                    m480.append(behaviour_session)
                                elif subject_ephys == 'm483':
                                    m483.append(behaviour_session)
                                elif subject_ephys == 'm479':
                                    m479.append(behaviour_session)
                                elif subject_ephys == 'm486':
                                    m486.append(behaviour_session)
                                elif subject_ephys == 'm478':
                                    m478.append(behaviour_session)
                                elif subject_ephys == 'm481':
                                    m481.append(behaviour_session)
                                elif subject_ephys == 'm484':
                                    m484.append(behaviour_session)
                           
                                    
    HP = m484 + m479 + m483
    PFC = m478 + m486 + m480 + m481
    all_sessions = m484  + m479 + m483 + m478 + m486 + m480 + m481
    return HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions

# Extracts poke identities of poke A and B (1-9) for each task
def extract_choice_pokes(session):
    task = session.trial_data['task']
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]
    poke_I = 'poke_'+ str(session.trial_data['configuration_i'][0])
    poke_I_task_2 = 'poke_'+ str(session.trial_data['configuration_i'][task_2_change[0]])
    poke_I_task_3 = 'poke_'+ str(session.trial_data['configuration_i'][task_3_change[0]])  
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3_change[0]])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3_change[0]])    
    
    return poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3

# Extracts trial initiation timestamps and ITI timestamps
def extract_times_of_initiation_and_ITIs(session):
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = extract_choice_pokes(session)

    pyControl_choice = [event.time for event in session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
     
    #Poke A and Poke B Timestamps 
    pyControl_a_poke_entry = [event.time for event in session.events if event.name in [poke_A,poke_A_task_2,poke_A_task_3]]
    pyControl_b_poke_entry = [event.time for event in session.events if event.name in [poke_B,poke_B_task_2,poke_B_task_3 ]]

    #ITI Timestamps 
    pyControl_end_trial = [event.time for event in session.events if event.name in ['inter_trial_interval']][2:] #first two ITIs are free rewards
    pyControl_end_trial = np.array(pyControl_end_trial)
    
    return pyControl_choice, pyControl_a_poke_entry, pyControl_b_poke_entry, pyControl_end_trial


# Extracts Choices of A and B
# Looks for the first events after initiation trial so only adds A and B pokes that are choices that lead to outcomes and ITIs
def only_meaningful_A_and_B_pokes(session): 
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = extract_choice_pokes(session)
    events_and_times = [[event.name, event.time] for event in session.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    poke_B_list = []       
    poke_A_list = []
    
    task = session.trial_data['task']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task_non_forced = task[non_forced_array]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]
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
                    
    return poke_A_list, poke_B_list

#Timestamps for initiation and ITI states for each task (Task 1, 2 and 3)
def initiation_and_trial_end_timestamps(session):
    pyControl_choice, pyControl_a_poke_entry, pyControl_b_poke_entry, pyControl_end_trial = extract_times_of_initiation_and_ITIs(session)
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 

    #For Choice State Calculations
    trial_сhoice_state_task_1 = pyControl_choice[:len(task_1)]
    trial_сhoice_state_task_2 = pyControl_choice[len(task_1):(len(task_1) +len(task_2))]
    trial_сhoice_state_task_3 = pyControl_choice[len(task_1) + len(task_2):]


    task_1_end_trial = np.where(task == 1)[0]
    task_2_end_trial = np.where(task == 2)[0]
    pyControl_end_trial_1 = pyControl_end_trial[:len(task_1_end_trial)]
    pyControl_end_trial_2 =pyControl_end_trial[len(task_1_end_trial)+2:(len(task_1_end_trial)+len(task_2_end_trial)+2)]
    pyControl_end_trial_3 = pyControl_end_trial[len(task_1_end_trial)+len(task_2_end_trial)+4:]
    pyControl_end_trial =  np.concatenate([pyControl_end_trial_1, pyControl_end_trial_2,pyControl_end_trial_3])

    #For ITI Calculations
    ITI_non_forced = pyControl_end_trial[non_forced_array]  
    ITI_task_1 = ITI_non_forced[:len(task_1)]#[2:]
    ITI_task_2 = ITI_non_forced[(len(task_1)):(len(task_1)+len(task_2))]
    ITI_task_3 = ITI_non_forced[len(task_1) + len(task_2):]
    
    return trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3, ITI_task_1, ITI_task_2,ITI_task_3


# State indicies to index in the entire task
#Creates indices of when the state was A good or B good in each of the three tasks
def state_indices(session):
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3,ITI_task_1, ITI_task_2,ITI_task_3 = initiation_and_trial_end_timestamps(session)
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    state = session.trial_data['state']
    state_non_forced = state[non_forced_array]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
    
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    
    #Task 1 
    state_1 = state_non_forced[:len(task_1)]
    state_a_good = np.where(state_1 == 1)[0]
    state_b_good = np.where(state_1 == 0)[0]
    
    # Task 2
    state_2 = state_non_forced[len(task_1): (len(task_1) +len(task_2))]
    state_t2_a_good = np.where(state_2 == 1)[0]
    state_t2_b_good = np.where(state_2 == 0)[0]

    #Task 3 
    state_3 = state_non_forced[len(task_1) + len(task_2):]
    state_t3_a_good = np.where(state_3 == 1)[0]
    state_t3_b_good = np.where(state_3 == 0)[0]
    return state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good

# Timestamps of Initiation state on A good trialsin all tasks 
def initiation_a_good(session):   
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3,ITI_task_1, ITI_task_2,ITI_task_3 = initiation_and_trial_end_timestamps(session)
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = state_indices(session)   
    
    trial_сhoice_state_task_1_a_good = trial_сhoice_state_task_1[state_a_good]
    trial_сhoice_state_task_2_a_good = trial_сhoice_state_task_2[state_t2_a_good]
    trial_сhoice_state_task_3_a_good = trial_сhoice_state_task_3[state_t3_a_good]

    return  trial_сhoice_state_task_1_a_good, trial_сhoice_state_task_2_a_good, trial_сhoice_state_task_3_a_good
    
# Timestamps of Initiation state on B good trials in all tasks
def initiation_b_good(session):   
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3,ITI_task_1, ITI_task_2,ITI_task_3 = initiation_and_trial_end_timestamps(session)
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = state_indices(session)
    
    trial_сhoice_state_task_1_b_good = trial_сhoice_state_task_1[state_b_good]
    trial_сhoice_state_task_2_b_good = trial_сhoice_state_task_2[state_t2_b_good]
    trial_сhoice_state_task_3_b_good = trial_сhoice_state_task_3[state_t3_b_good]
    
    return trial_сhoice_state_task_1_b_good, trial_сhoice_state_task_2_b_good, trial_сhoice_state_task_3_b_good

# Timestamps of the ITIs for different A and B states for all three tasks
def ITIs_split_by_good_bad(session):
    state_a_good, state_b_good, state_t2_a_good, state_t2_b_good, state_t3_a_good, state_t3_b_good = state_indices(session)
    trial_сhoice_state_task_1, trial_сhoice_state_task_2, trial_сhoice_state_task_3,ITI_task_1, ITI_task_2,ITI_task_3 = initiation_and_trial_end_timestamps(session)
    #ITI Calculations

    ITI_task_1_a_good = ITI_task_1[state_a_good]
    ITI_task_1_b_good = ITI_task_1[state_b_good]
    
    ITI_task_2_a_good  = ITI_task_2[state_t2_a_good]
    ITI_task_2_b_good =ITI_task_2[state_t2_b_good]
    
    ITI_task_3_a_good  = ITI_task_3[state_t3_a_good]
    ITI_task_3_b_good  = ITI_task_3[state_t3_b_good]
    
    return ITI_task_1_a_good, ITI_task_1_b_good, ITI_task_2_a_good, ITI_task_2_b_good, ITI_task_3_a_good, ITI_task_3_b_good
   
#  Timestamps of choice of pokes when A is good, B is good, A is bad and B is bad on Task 1
def task_1_choice_time_good_bad(session):
    ITI_task_1_a_good, ITI_task_1_b_good, ITI_task_2_a_good, ITI_task_2_b_good, ITI_task_3_a_good, ITI_task_3_b_good = ITIs_split_by_good_bad(session)
    poke_A_list, poke_B_list  = only_meaningful_A_and_B_pokes(session)
    
    trial_сhoice_state_task_1_b_good, trial_сhoice_state_task_2_b_good, trial_сhoice_state_task_3_b_good = initiation_b_good(session)
    trial_сhoice_state_task_1_a_good, trial_сhoice_state_task_2_a_good, trial_сhoice_state_task_3_a_good = initiation_a_good(session)
    # Task one
    entry_a_good_list = []
    a_good_choice_time_task_1 = []
    entry_b_bad_list = []
    b_bad_choice_time_task_1 = []
    entry_a_bad_list = []
    b_good_choice_time_task_1 = []
    entry_b_good_list = []
    a_bad_choice_time_task_1 = []
    

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
                
    return entry_a_good_list,a_good_choice_time_task_1, entry_b_bad_list, b_bad_choice_time_task_1, entry_a_bad_list,b_good_choice_time_task_1,entry_b_good_list, a_bad_choice_time_task_1 


#  Timestamps of choice of pokes when A is good, B is good, A is bad and B is bad on Task 2
def task_2_choice_time_good_bad(session):
    ITI_task_1_a_good, ITI_task_1_b_good, ITI_task_2_a_good, ITI_task_2_b_good, ITI_task_3_a_good, ITI_task_3_b_good = ITIs_split_by_good_bad(session)
    poke_A_list, poke_B_list  = only_meaningful_A_and_B_pokes(session)
    trial_сhoice_state_task_1_b_good, trial_сhoice_state_task_2_b_good, trial_сhoice_state_task_3_b_good = initiation_b_good(session)
    trial_сhoice_state_task_1_a_good, trial_сhoice_state_task_2_a_good, trial_сhoice_state_task_3_a_good = initiation_a_good(session)

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
                
    return entry_a_good_task_2_list, a_good_choice_time_task_2, entry_b_bad_list_task_2,b_bad_choice_time_task_2, entry_a_bad_task_2_list, b_good_choice_time_task_2, entry_b_good_list_task_2, a_bad_choice_time_task_2
    

#  Timestamps of choice of pokes when A is good, B is good, A is bad and B is bad on Task 3
def task_3_choice_time_good_bad(session):  
    ITI_task_1_a_good, ITI_task_1_b_good, ITI_task_2_a_good, ITI_task_2_b_good, ITI_task_3_a_good, ITI_task_3_b_good = ITIs_split_by_good_bad(session)
    poke_A_list, poke_B_list =only_meaningful_A_and_B_pokes(session)
    trial_сhoice_state_task_1_b_good, trial_сhoice_state_task_2_b_good, trial_сhoice_state_task_3_b_good = initiation_b_good(session)
    trial_сhoice_state_task_1_a_good, trial_сhoice_state_task_2_a_good, trial_сhoice_state_task_3_a_good = initiation_a_good(session)

    #Task three         
    entry_a_good_task_3_list = []
    a_good_choice_time_task_3 = []
    entry_b_bad_list_task_3 = []
    b_bad_choice_time_task_3 = []
    entry_b_good_list_task_3 = []
    b_good_choice_time_task_3 = []
    entry_a_bad_task_3_list = []
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
                
    return entry_a_good_task_3_list, a_good_choice_time_task_3, entry_b_bad_list_task_3, b_bad_choice_time_task_3, entry_b_good_list_task_3, b_good_choice_time_task_3, entry_a_bad_task_3_list, a_bad_choice_time_task_3
    
    
def poke_state_task_1(session):
    entry_a_good_list,a_good_choice_time_task_1, entry_b_bad_list, b_bad_choice_time_task_1, entry_a_bad_list, b_good_choice_time_task_1,entry_b_good_list, a_bad_choice_time_task_1  = task_1_choice_time_good_bad(session)
    entry_a_bad_list = np.array(entry_a_bad_list)
    entry_b_good_list = np.array(entry_b_good_list)
    entry_a_good_list = np.array(entry_a_good_list)
    entry_b_bad_list = np.array(entry_b_bad_list)
    
    return entry_a_bad_list, entry_b_good_list, entry_a_good_list, entry_b_bad_list
    
def initiation_state_task_1(session):
    entry_a_good_list,a_good_choice_time_task_1, entry_b_bad_list, b_bad_choice_time_task_1, entry_a_bad_list, b_good_choice_time_task_1,entry_b_good_list, a_bad_choice_time_task_1  = task_1_choice_time_good_bad(session)
    a_good_choice_time_task_1 = np.array(a_good_choice_time_task_1)
    a_good_choice_time_task_1 = np.unique(a_good_choice_time_task_1)
    
    a_bad_choice_time_task_1 = np.array(a_bad_choice_time_task_1)
    a_bad_choice_time_task_1 = np.unique(a_bad_choice_time_task_1)
    
    b_bad_choice_time_task_1 = np.array(b_bad_choice_time_task_1)
    b_bad_choice_time_task_1 = np.unique(b_bad_choice_time_task_1)
    
    b_good_choice_time_task_1 = np.array(b_good_choice_time_task_1)
    b_good_choice_time_task_1 = np.unique(b_good_choice_time_task_1)
    
    return a_good_choice_time_task_1, a_bad_choice_time_task_1, b_bad_choice_time_task_1, b_good_choice_time_task_1
    
def poke_state_task_2(session):
    entry_a_good_task_2_list, a_good_choice_time_task_2, entry_b_bad_list_task_2,b_bad_choice_time_task_2, entry_a_bad_task_2_list, b_good_choice_time_task_2, entry_b_good_list_task_2, a_bad_choice_time_task_2 = task_2_choice_time_good_bad(session)
    
    entry_b_good_list_task_2 = np.array(entry_b_good_list_task_2)
    entry_a_bad_task_2_list = np.array(entry_a_bad_task_2_list)
    entry_a_good_task_2_list = np.array(entry_a_good_task_2_list)
    entry_b_bad_list_task_2 = np.array(entry_b_bad_list_task_2) 
    
    return entry_b_good_list_task_2, entry_a_bad_task_2_list, entry_a_good_task_2_list, entry_b_bad_list_task_2
    
def initiation_state_task_2(session):
    entry_a_good_task_2_list, a_good_choice_time_task_2, entry_b_bad_list_task_2,b_bad_choice_time_task_2, entry_a_bad_task_2_list, b_good_choice_time_task_2, entry_b_good_list_task_2, a_bad_choice_time_task_2 = task_2_choice_time_good_bad(session)
    
    a_good_choice_time_task_2 = np.array(a_good_choice_time_task_2)
    a_good_choice_time_task_2 = np.unique(a_good_choice_time_task_2)
   
    a_bad_choice_time_task_2 = np.array(a_bad_choice_time_task_2)
    a_bad_choice_time_task_2 = np.unique(a_bad_choice_time_task_2)
   
    b_bad_choice_time_task_2 = np.array(b_bad_choice_time_task_2)
    b_bad_choice_time_task_2 = np.unique(b_bad_choice_time_task_2)
   
    b_good_choice_time_task_2 = np.array(b_good_choice_time_task_2)
    b_good_choice_time_task_2 = np.unique(b_good_choice_time_task_2)
    
    return a_good_choice_time_task_2, a_bad_choice_time_task_2, b_bad_choice_time_task_2, b_good_choice_time_task_2

    
def poke_state_task_3(session):
    entry_a_good_task_3_list, a_good_choice_time_task_3, entry_b_bad_list_task_3, b_bad_choice_time_task_3, entry_b_good_list_task_3, b_good_choice_time_task_3, entry_a_bad_task_3_list, a_bad_choice_time_task_3 = task_3_choice_time_good_bad(session)
    
    entry_b_good_list_task_3 = np.array(entry_b_good_list_task_3)
    entry_a_bad_task_3_list = np.array(entry_a_bad_task_3_list)
    entry_a_good_task_3_list =  np.array(entry_a_good_task_3_list)
    entry_b_bad_list_task_3 = np.array(entry_b_bad_list_task_3)     
    
    return entry_b_good_list_task_3, entry_a_bad_task_3_list, entry_a_good_task_3_list, entry_b_bad_list_task_3

def initiation_state_task_3(session):
    entry_a_good_task_3_list, a_good_choice_time_task_3, entry_b_bad_list_task_3, b_bad_choice_time_task_3, entry_b_good_list_task_3, b_good_choice_time_task_3, entry_a_bad_task_3_list, a_bad_choice_time_task_3 = task_3_choice_time_good_bad(session)
    
    a_good_choice_time_task_3 = np.array(a_good_choice_time_task_3)
    a_good_choice_time_task_3 = np.unique(a_good_choice_time_task_3)
    
    a_bad_choice_time_task_3 = np.array(a_bad_choice_time_task_3)
    a_bad_choice_time_task_3 = np.unique(a_bad_choice_time_task_3)   
    
    b_bad_choice_time_task_3 = np.array(b_bad_choice_time_task_3)
    b_bad_choice_time_task_3 = np.unique(b_bad_choice_time_task_3)
    
    b_good_choice_time_task_3 = np.array(b_good_choice_time_task_3)
    b_good_choice_time_task_3 = np.unique(b_good_choice_time_task_3)
    
    return a_good_choice_time_task_3, a_bad_choice_time_task_3, b_bad_choice_time_task_3, b_good_choice_time_task_3


def poke_A_B_make_consistent(session):

    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = extract_choice_pokes(session)

    poke_A1_A2_A3 = False
    poke_A1_B2_B3 = False
    poke_A1_B2_A3 = False
    poke_A1_A2_B3 = False 
    poke_B1_B2_B3 = False
    poke_B1_A2_A3 = False
    poke_B1_A2_B3 = False
    poke_B1_B2_A3 = False
    
    if poke_A_task_3 == poke_A_task_2 and poke_A_task_2 == poke_A:
        poke_A1_A2_A3 = True 
    elif poke_B_task_3 == poke_B_task_2 and poke_B_task_2 == poke_A:
        poke_A1_B2_B3 = True
    elif poke_A == poke_A_task_3 and poke_B_task_2 == poke_A:
        poke_A1_B2_A3 = True
    elif poke_A == poke_A_task_2 and poke_A == poke_B_task_3:
        poke_A1_A2_B3 = True 
    elif poke_B == poke_B_task_2 and poke_B == poke_B_task_3:
        poke_B1_B2_B3 = True 
    elif poke_B == poke_A_task_2 and poke_B == poke_A_task_3:
        poke_B1_A2_A3 = True 
    elif poke_B == poke_A_task_2 and poke_B == poke_B_task_3:
        poke_B1_A2_B3 = True
    elif poke_B == poke_B_task_2 and poke_B == poke_A_task_3:
        poke_B1_B2_A3 = True
    return  poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3, poke_B1_B2_A3 


def poke_Is_make_consistent(session):
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = extract_choice_pokes(session)
    
    # Make Is consistent for future RSA analysis

    poke_I1_I2 = False
    poke_I2_I3 = False
    poke_I1_I3 = False
   
    if poke_I == poke_I_task_2:
        poke_I1_I2 = True
    elif poke_I == poke_I_task_3:
        poke_I1_I3 = True
    elif poke_I_task_2 == poke_I_task_3:
        poke_I2_I3 = True
        
    return poke_I1_I2,poke_I1_I3, poke_I2_I3
        
        
