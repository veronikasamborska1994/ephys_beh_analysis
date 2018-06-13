#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:36:22 2018

@author: veronikasamborska
"""
import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import sys 
sys.path.append('/Users/veronikasamborska/Desktop/2018-12-12-Reversal_learning/code/reversal_learning/')
import utility as ut
import seaborn as sns
import pandas as pd

# Reversals Plot -------------------------------------------------------------------------

def trials_till_reversal_plot(experiment, subject_IDs ='all' , fig_no=1):
    
#   Define variables
    FT = True # True if the experiment included Forced Choice Trials 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
    sessions_block = []
    trials_from_prev_session = [] #List to hold data from all subjects
    tasks = 11 # Maximum number of tasks
    reversals = 10
    reversal_to_threshold = np.ones(shape=(9,tasks,reversals))
    reversal_to_threshold[:] = np.NaN 
    task_1_2 = np.ones(shape=(9,2,reversals))
    task_1_2 = np.NaN 
    task_3_4 = np.ones(shape=(9,2,reversals))
    task_3_4 = np.NaN 
    task_5_6 = np.ones(shape=(9,2,reversals))
    task_5_6 = np.NaN 
    task_7_8 = np.ones(shape=(9,2,reversals))
    task_7_8 = np.NaN
    task_9_10 = np.ones(shape=(9,2,reversals))
    task_9_10 = np.NaN
    
# Looping through subjects and sessions    
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
        task_number = 0 # Current task number
        reversal_number = 0 #
        previous_session_config = 0 
        subject_sessions = experiment.get_sessions(subject_ID)
        trials_from_prev_session = 0
        configuration = subject_sessions[0].trial_data['configuration_i']
        if FT == True: 
            forced_trials = subject_sessions[0].trial_data['forced_trial']
        for j, session in enumerate(subject_sessions):
            sessions_block = session.trial_data['block']
            n_trials = session.trial_data['n_trials']
            # Find trials where threshold crossed.
            prt = (session.trial_data['pre-reversal trials'] > 0).astype(int)
            Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            
            configuration = session.trial_data['configuration_i'] 
            if FT == True:
                forced_trials = session.trial_data['forced_trial']
                forced_trials_sum = sum(forced_trials)
                forced_array = np.where(forced_trials == 1)[0]
                sessions_block = np.delete(sessions_block, forced_array)
                n_trials = n_trials -  forced_trials_sum
                prt = np.delete(prt, forced_array)
                Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            threshold_crossing_trials = np.where((prt[1:] - prt[:-1]) == 1)[0]
            n_reversals = len(reversal_trials)
            
            if configuration[0]!= previous_session_config:
                reversal_number = 0
                task_number += 1
                trials_from_prev_session = 0
                previous_session_config = configuration[0]  
            if len(reversal_trials) == 0:
                    trials_from_prev_session += n_trials
            else: 
                    for i, crossing_trial in enumerate(threshold_crossing_trials): 
                        if i< n_reversals:
                            if reversal_number <= 9:
                                if i == 0: # First element in the threshold_crossing_trials_list
                                    reversal_to_threshold[n_subj, task_number, reversal_number] = crossing_trial+trials_from_prev_session
                                    trials_from_prev_session = 0
                                elif (i > 0) and (i < n_reversals): # reversal occured.
                                    reversal_to_threshold[n_subj, task_number, reversal_number] = (crossing_trial-reversal_trials[i-1])
                                reversal_number += 1                            
                        else: # Revesal did not occur before end of session.
                            trials_from_prev_session = n_trials - reversal_trials[i-1]
                        
    # Exporting data as excel files using pandas to reversal_learning folder
    for task in range(tasks):
        pd.DataFrame(data=reversal_to_threshold[:,task,:]).to_csv('task{}_pd_reversals.csv'.format(task))
 
    # Mean and Median Calculations
    mean_threshold = np.nanmean(reversal_to_threshold,axis = 0)
    median_threshold= np.nanmedian(reversal_to_threshold,axis = 2)
    median_threshold_per_task = np.nanmean(median_threshold, axis = 0)
    std_threshold_per_task = np.nanstd(median_threshold, axis = 0)
    sample_size=np.sqrt(9)
    std_err_median= std_threshold_per_task/sample_size
    x_pos = np.arange(len(median_threshold_per_task))
    plt.figure()
    sns.set()
    plt.errorbar(x = x_pos, y = median_threshold_per_task, yerr = std_err_median, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    
    
    # Delete NaNs for the Trend Line calculation
    x_pos_no_NaNs = np.delete(x_pos, 0)
    median_threshold_per_task_no_NaNs = np.delete(median_threshold_per_task, 0)
    
    # Trend line 
    #z = np.polyfit(x_pos_no_NaNs, median_threshold_per_task_no_NaNs,1)
   # p = np.poly1d(z)
    #plt.plot(x_pos,p(x_pos),"--", color = 'grey', label = 'Trend Line')
    plt.title("Median Number of Trials till Threshold per Task ")
    plt.xlabel("Task Number")
    plt.ylabel("Number of Trials Till Threshold")
    plt.show()
    x=np.arange(reversals)
    std_proportion=np.nanstd(reversal_to_threshold, axis = 0)
    std_err= std_proportion/sample_size
    plt.figure()
    sns.set()
    for i in range(tasks - 1): 
         plt.plot(i * reversals + x, mean_threshold[i + 1])
         plt.fill_between(i * reversals + x, mean_threshold[i + 1]-std_err[i + 1], mean_threshold[i + 1]+std_err[i + 1], alpha=0.2)
    plt.ylabel('Number of Trials Till Threshold')
    plt.xlabel('Reversal Number')
    
    # Plot of 2 tasks combined 
    task_1_2 = reversal_to_threshold[:,1:3,:]
    task_1_2 = np.nanmean(task_1_2,axis = 1)
    task_1_2_mean = np.nanmean(task_1_2, axis = 0)
    task_3_4 = reversal_to_threshold[:,3:5,:]
    task_3_4 = np.nanmean(task_3_4,axis = 1)
    task_3_4_mean = np.nanmean(task_3_4, axis = 0)
    task_5_6 = reversal_to_threshold[:,5:7,:]
    task_5_6 = np.nanmean(reversal_to_threshold,axis = 1)
    task_5_6_mean = np.nanmean(task_5_6, axis = 0)
    task_7_8 = reversal_to_threshold[:,7:9,:]
    task_7_8 = np.nanmean(task_7_8,axis = 1)
    task_7_8_mean = np.nanmean(task_7_8, axis = 0)
    task_9_10 = reversal_to_threshold[:,9:11,:]
    task_9_10 = np.nanmean(task_9_10,axis = 1)
    task_9_10_mean = np.nanmean(task_9_10, axis = 0)
    plt.figure()
    plt.plot(task_1_2_mean, label='Tasks 1 & 2', color = 'pink' )
    plt.plot(task_3_4_mean, label='Tasks 3 & 4', color = 'indigo')
    plt.plot(task_5_6_mean, label='Tasks 5 & 6', color = 'silver')
    plt.plot(task_7_8_mean, label='Tasks 7 & 8', color = 'olive')
    plt.plot(task_9_10_mean, label='Tasks 9 & 10', color = 'teal')
    plt.ylabel('Number of Trials till Threshold')
    plt.xlabel('Reversal Number')
    plt.legend()

def session_reversals_trials(experiment,subject_IDs ='all'): 
    trials_reversals = np.zeros([9,50])
    trials_reversals[:] = np.NaN 
    trials_sum =  np.zeros([9,50])
    reversals_sum = np.zeros([9,50])
    trials_sum[:] = np.NaN 
    reversals_sum[:] = np.NaN
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        for n_subj, subject_ID in enumerate(subject_IDs):
            subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
            for j, session in enumerate(subject_sessions): 
                n_trials = session.trial_data['n_trials']   
                sessions_block = session.trial_data['block']
                Block_transitions = sessions_block[1:] - sessions_block[:-1] 
                sum_blocks = sum(Block_transitions)
                if sum_blocks > 0: 
                    trials_sum[n_subj,j] = n_trials
                    reversals_sum[n_subj,j] = sum_blocks
                    trials_reversals[n_subj,j]= (n_trials/sum_blocks)
                else: 
                    trials_reversals[n_subj,j] = n_trials
                    reversals_sum[n_subj,j] = sum_blocks
            #sns.set()
           # plt.figure()
            #plt.plot(trials_reversals[n_subj][:], label = [subject_ID])
            #plt.legend()
    trials_reversals_mean = np.nanmean(trials_reversals,axis = 1)
    reversals_mean = np.nanmean(reversals_sum,axis = 1)
    trials_mean = np.nanmean(trials_sum,axis = 1)
    plt.figure()
    plt.scatter(subject_IDs,trials_reversals_mean, color = 'blue')
    plt.xlabel('Subject')
    plt.ylabel('Number of Trials / Number of Reversals per Session')
    plt.title('Number of Trials / Number of Reversals per Session Plot')
    plt.figure()
    plt.scatter(subject_IDs,reversals_mean, color = 'salmon')
    plt.xlabel('Subject')
    plt.ylabel( 'Number of Reversals per Session')
    plt.title('Reversals Plot')
    plt.figure()
    plt.scatter(subject_IDs,trials_mean, color = 'green')
    plt.xlabel('Subject')
    plt.ylabel('Number of Trials per Session')
    plt.title('Trials Plot')
    
             
    
# Experiment plot of poke A or B following the choice of A or B per reversal -------------------------------------------------------------------------
def A_B_poke_reversal_plot(experiment,subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        n_subjects = len(subject_IDs)  
    else:
        n_subjects = len(subject_IDs)
# =============================================================================
#   Define what you want to plot (Probability of A/B Poke, Number of A/B pokes on Rewared Trials,
#   Number of A/B pokes on Non-Rewared Trials or Number of A/B pokes on All Trials
#   Could also do Probability Plots for Rewarded vs Non-Rewarded Trials 
# =============================================================================
    probability_plot = False # True for probability and True for all_trials/rewareded_trials/non_rewarded_trials 
    all_trials = True # True for all_trials; If need a plot for number of pokes probability_plot = False 
    rewarded_trials = False # True for rewarded
    non_rewarded_trials = False # True for non-rewarded
    FT = True # True if the Experiment Included Forced Choice Trials 
    Plot_FT = False # Plot Only Forced Choice Trials
    
    # Find number of subjects 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        n_subjects = len(subject_IDs)  
    else:
        n_subjects = len(subject_IDs)
    tasks = 11 # Maximum number of tasks completed 
    trial_count = 0 # Initiate trial count (used it for debugging)
    bad_pokes = np.zeros([n_subjects,tasks,10])  # Subject, task number, reversal number
    bad_pokes[:] = np.NaN # Fill it with NaNs 
    
    # Loop through all subjects and all sessions and put all trials into one list for each subject
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        previous_session_config = 0 # Initiate previous configuration
        task_number = 0 # Initiate task number
        rev = 0 # Initiate reversal number
        wrong_count = 0 # Initiate count of wrong pokes
        outcomes_count = 0 # Initiate count of outcomes
        task = [] # Empty list to later append to task number from all sessions
        reversals = [] # Empty list to later append to reversal number from all sessions
        outcome = [] # Empty list to later append to outcomes on each trial from all sessions (rewarded = 1, non_rewarded = 0)
        f_trials =[]
        ft_count = 0 # Forced trial 1, 0 normal trial 
        all_sessions_wrong_ch = [] # Get the list with only trials that were treated as trials in task programme
        rewarded_trial = False # State for whether the trial was rewarded (initate at False)
        non_rewarded_trial = False # State for whether the trial was non-rewarded(initate at False)
        choice_state = False # State for whether the event happens in the choice state (initate at False)
        forced_a = False
        forced_b = False
        
        for j, session in enumerate(subject_sessions): 
            forced_a = False
            forced_b = False
            trials = session.trial_data['trials']     
            original_trials = session.trial_data['trials'] 
            configuration = session.trial_data['configuration_i']
            sessions_block = session.trial_data['block']
            Block_transitions = sessions_block[1:] - sessions_block[:-1] 
            #reversal_trials = np.where(Block_transitions == 1)[0]
            outcomes= session.trial_data['outcomes']
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            
            if FT == True and Plot_FT == False: # If the Experiment has Forced Trials that we want to exclude
                forced_trials = session.trial_data['forced_trial']
                forced_array = np.where(forced_trials == 1)[0]
                trials = np.delete(trials, forced_array) # delete forced choice trials 
                sessions_block = np.delete(sessions_block, forced_array)
                Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            elif Plot_FT == True: # If the Experiment has Forced Trials we want to look at
                forced_trials = session.trial_data['forced_trial']
                forced_array = np.where(forced_trials == 0)[0]# Where are forced trials            
                to_delete = np.where(forced_trials == 0)[0]
                trials= np.delete(trials,to_delete) # delete non-forced choice trials  
                
            reversal_trials = np.where(Block_transitions == 1)[0]              
            trial_l = len(trials)             
            session_wrong_choice =[]
            prev_choice = []
            
            if  Plot_FT == False:
                events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial','sound_b_no_reward', 'sound_b_reward','sound_a_no_reward','sound_a_reward',poke_A, poke_B]]
            elif Plot_FT == True:
                events = [event.name for event in session.events if event.name in ['a_forced_state','init_trial','inter_trial_interval', 'b_forced_state' ,poke_A, poke_B]]
        
            # Check if the configuration has changed = set reversal number to 0 
            if configuration[0]!= previous_session_config:
                task_number += 1
                rev = 0
                previous_session_config = configuration[0]
                
            # Go through trials and make lists that have task, reversal and outcome information for the entire experiment 
            if Plot_FT == False:
                for trial in trials:
                    trial_count +=1
                    task.append(task_number)
                    reversals.append(rev)
                    outcome.append(outcomes_count)
                    for reversal in reversal_trials:
                        if reversal == trial:
                            rev += 1
            else: 
                for t in original_trials:
                    task.append(task_number)
                    reversals.append(rev)
                    f_trials.append(ft_count)
                    ft_count = 0
                    for reversal in reversal_trials:
                        if reversal == t:
                            rev += 1
                    for ft in forced_array: #go through forced trials 
                        if ft == t: 
                            ft_count = 1 # if forced = 0, if non-forced = 1  

                    
            # Go through events list and find the events of interest 
            for event in events:
                if Plot_FT == False:
                    if event == 'choice_state':
                        session_wrong_choice.append(wrong_count)
                        wrong_count = 0
                        choice_state = True
                        rewarded_trial = False                
                    elif event == 'sound_b_reward':
                        if rewarded_trials == True: 
                            rewarded_trial = True                        
                    elif event == 'sound_a_reward':
                        if rewarded_trials == True:
                            rewarded_trial = True
                            trial_count+=1
                    elif event == 'sound_a_no_reward':
                        if non_rewarded_trials == True:
                            non_rewarded_trial = True
                    elif event == 'sound_b_no_reward':
                        if non_rewarded_trials == True:
                            non_rewarded_trial = True
                    elif event == poke_A : 
                        if choice_state == True:
                            prev_choice = 'Poke_A'
                            choice_state = False
                        elif choice_state == False:
                            if all_trials == True: 
                                if prev_choice == 'Poke_B': 
                                    wrong_count += 1   
                            elif rewarded_trials == True:
                                if rewarded_trial == True and prev_choice == 'Poke_B': 
                                    wrong_count += 1
                            elif non_rewarded_trials == True:
                                if non_rewarded_trial == True and prev_choice == 'Poke_B':
                                    wrong_count += 1 
                                    
                    elif event == poke_B :
                        if choice_state == True:
                            prev_choice = 'Poke_B'
                            choice_state = False
                        elif choice_state == False: 
                            if all_trials == True: 
                                if prev_choice == 'Poke_A': 
                                    wrong_count += 1     
                            elif rewarded_trials == True:
                                if rewarded_trial == True and prev_choice == 'Poke_A': 
                                    wrong_count += 1  
                            elif non_rewarded_trials ==True: 
                                if non_rewarded_trial == True and prev_choice == 'Poke_A':
                                    wrong_count += 1 
                    elif event == 'init_trial':   
                        choice_state = False 
                        
               # Plotting Forced Choice trials          
                elif Plot_FT == True:
                    if event == 'a_forced_state':                  
                        forced_a = True 
                        session_wrong_choice.append(wrong_count)
                        trial_count+=1
                        wrong_count = 0
                    elif event == 'b_forced_state':
                        forced_b = True  
                        session_wrong_choice.append(wrong_count)
                        trial_count+=1         
                        wrong_count = 0 
                    elif event == poke_A: 
                        if forced_b == True:
                            wrong_count += 1   
                    elif event == poke_B:
                        if forced_a == True:
                            wrong_count += 1
                    elif event == 'init_trial':
                        forced_b = False 
                        forced_a = False   
                                                 
            if j == 0: 
                all_sessions_wrong_ch = session_wrong_choice[0:trial_l]
                
            elif j > 0: 
                all_sessions_wrong_ch += session_wrong_choice[0:trial_l]
                
        # Change lists to numpy arrays  
        np_task = np.asarray(task)
        np_reversals = np.asarray(reversals)
        np_outcomes = np.asarray(outcome) # 1s on the trials that were rewarded 
        np_pokes = np.asarray(all_sessions_wrong_ch)
        np.f_trials = np.asarray(f_trials) #contains 0 for trials, 1s for non - forced
        if Plot_FT == True:
            forced = np.where(np.f_trials == 1)[0] # 1s are where all non-forced trials are           
            np_task = np.delete(np_task, forced) # Delete non-forced choice trials
            np_task = np_task[0:(len(np_pokes))]       
            np_reversals = np.delete(np_reversals, forced) 
            np_reversals = np_reversals[0:(len(np_pokes))]
            
        
        # Finding A/B pokes for each task (tn), and each reversal (rn) within the task
        for tn in range(tasks):
            if tn > 0:
                for rn in range(10):
                   a = np_pokes[(np_task==tn) & (np_reversals==rn)] # For plots from all trials 
                   a_pokes = np.asarray(a)
                   
                   if Plot_FT == False:
                       outcomes = np_outcomes[(np_task == tn) & (np_reversals == rn)] #For rewarded or non-rewarded trials trials plots 
                       outcomes_sum_np=np.asarray(outcomes) #For rewarded and non-rewareded trial plots 
                   if probability_plot == True:  #For probability calculations  
                       np_probability_index = np.where(a_pokes != 0) #For probability calculations
                       a_pokes[np_probability_index] = 1
                       mean_pokes = np.nanmean(a_pokes)
                   elif all_trials == True and probability_plot == False: 
                       a_pokes = np.asarray(a) #For both rewarded and non-rewarded trials
                       mean_pokes=np.nanmean(a_pokes) #For all trials
                   elif rewarded_trials == True:
                       outcomes_sum_np = np.asarray(outcomes)
                       outcomes_sum = sum(outcomes_sum_np)
                       mean_pokes = (sum(a_pokes))/outcomes_sum #outcomes_sum #num_zeros #outcomes_sum 
                   elif non_rewarded_trials == True:
                        num_zeros = (outcomes_sum_np == 0).sum()
                        mean_pokes = (sum(a_pokes))/num_zeros     
                   elif np.isinf(mean_pokes):
                       mean_pokes = 0
                   bad_pokes[n_subj,tn,rn] = mean_pokes      
                   
    # Exporting data as excel files using pandas to reversal_learning folder  
             
    for task in range(tasks):
        pd.DataFrame(data=bad_pokes[:,task,:]).to_csv('task{}_ab_reversals.csv'.format(task))

    
    #Mean Pokes 
    mean_bad_pokes = np.nanmean(bad_pokes,axis = 0)
    median_bad_pokes=np.nanmean(bad_pokes,axis = 2)
    median_bad_pokes = np.nanmedian(median_bad_pokes, axis = 0)
    std_bad_pokes_med= np.nanstd(median_bad_pokes, axis = 0)
    sample_size=np.sqrt(9)
    std_err_median= std_bad_pokes_med/sample_size
    x_pos = np.arange(len(median_bad_pokes))
    plt.figure()
    sns.set()
    plt.errorbar(x = x_pos, y = median_bad_pokes, yerr = std_err_median, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    
    x_pos_no_NaNs = np.delete(x_pos, 0)
    median_threshold_per_task_no_NaNs = np.delete(median_bad_pokes, 0)
    z = np.polyfit(x_pos_no_NaNs, median_threshold_per_task_no_NaNs,1)
    p = np.poly1d(z)
    plt.plot(x_pos,p(x_pos),"--", color = 'grey', label = 'Trend Line')
    if probability_plot == True: 
        plt.title("Median Probability of Out of Sequence Pokes per Trial ")
        plt.ylabel("Probability of Out of Sequence Pokes per Trial")
    else:
        plt.title("Number of Out of Sequence Pokes per Trial ")
        plt.ylabel("Number of Out of Sequence Pokes per Trial")
    plt.xlabel("Task Number")
    
    plt.show()
    x=np.arange(10)
    # Standard Errors 
    std_proportion=np.nanstd(bad_pokes, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    #Plotting
    plt.figure()
    for i in range(tasks - 1): 
        plt.plot(i * 10 + x, mean_bad_pokes[i + 1])
        plt.fill_between(i * 10 + x, mean_bad_pokes[i + 1]-std_err[i + 1], mean_bad_pokes[i + 1]+std_err[i + 1], alpha=0.2)
    if probability_plot == True: 
        plt.title("Probability of Out of Sequence Pokes per Trial ")
        plt.ylabel('Probability of Out of Sequence Pokes per Trial')
    else:
        plt.title("Number of Out of Sequence Pokes per Trial ")
        plt.ylabel('Number of Out of Sequence Pokes per Trial')
    #plt.title('Rewarded Trials')
    plt.legend(loc='upper right')
    plt.xlabel('Reversal Number')
                   
            
# Experiment plot of I pokes during the choice state when A or B should be chosen per reversal -------------------------------------------------------------------------
def I_poke_reversal_plot(experiment,subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        n_subjects = len(subject_IDs)
    else:
        n_subjects = len(subject_IDs)
    reversals = []
    task = []
    task_number =0
    tasks=9
    bad_pokes = np.zeros([n_subjects,tasks,21])# subject, task number, reversal number
    bad_pokes[:] = np.NaN
    #Put all trials into one list for each subject
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        previous_session_config = 0
        task_number = 0
        rev=0
        trial = 0
        task=[]
        reversals =[]
        all_sessions_wrong_ch=[]
        prev_event_choice = False 
        period_before_ITI = False
        for j, session in enumerate(subject_sessions):
            trials=session.trial_data['trials']
            configuration = session.trial_data['configuration_i']
            sessions_block = session.trial_data['block']
            Block_transitions = sessions_block[1:] - sessions_block[:-1] #block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            prev_event_choice = False
            poke_I = 'poke_'+str(session.trial_data['configuration_i'][0])
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            events = [event.name for event in session.events if event.name in ['choice_state', 'period_before_iti', poke_I, poke_A, poke_B]]
            trials = [event.name for event in session.events if event.name in ['choice_state']]
            sessions_block = session.trial_data['block']
            trials=session.trial_data['trials']
            trial_l = len(trials)
            wrong_poke=0
            session_wrong_choice =[]    
            if configuration[0]!= previous_session_config:
                task_number += 1
                rev = 0
                previous_session_config = configuration[0]
            for trial in trials:
                task.append(task_number)
                reversals.append(rev)
                for reversal in reversal_trials:
                    if reversal == trial:
                        rev+=1  
            for event in events:
                if event == 'choice_state':
                    session_wrong_choice.append(wrong_poke)
                    prev_event_choice = True
                    period_before_ITI = True
                    wrong_poke = 0
                elif event == poke_I: 
                    if prev_event_choice == True and period_before_ITI== True:   
                        wrong_poke += 1
                elif event == 'period_before_iti':
                    period_before_ITI = False
            if j == 0: 
                all_sessions_wrong_ch = session_wrong_choice[0:trial_l]
            if j > 0: 
                all_sessions_wrong_ch +=session_wrong_choice[0:trial_l]
            np_task = np.asarray(task)
            np_reversals = np.asarray(reversals)
            np_pokes = np.asarray(all_sessions_wrong_ch)
        for tn in range(tasks):
            if tn > 0:
                for rn in range(21):
                    a=np_pokes[(np_task==tn) & (np_reversals==rn)]
                    a_pokes = np.asarray(a)
                    mean_pokes= np.nanmean(a_pokes)
                    bad_pokes[n_subj,tn,rn] = mean_pokes        
    mean_bad_pokes=np.nanmean(bad_pokes,axis = 0)
    std_proportion=np.nanstd(bad_pokes, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    x=np.arange(21)
    for i in range(tasks - 1): 
        plt.plot(i * 20 + x, mean_bad_pokes[i + 1])
        plt.fill_between(i * 20 + x, mean_bad_pokes[i + 1]-std_err[i + 1], mean_bad_pokes[i + 1]+std_err[i + 1], alpha=0.2)     
    plt.ylabel('Number of I pokes during Choice State')
    plt.xlabel('Reversal')
            
#Plot of pokes in order -------------------------------------------------------------------------
    
# Experiment plot of poke I in the period following trial initiation -------------------------------------------------------------------------
def I_poke_session_plot(experiment, subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        #n_subjects = len(subject_IDs)
    print(subject_IDs)
    wrong_poke=0
    correct_poke=0
    wrong=[]
    number_A_B = np.ones(shape=(14,50))
    number_A_B[:] = np.NaN 
    correct=[]
    number_I=np.ones(shape=(14,50))
    number_I[:] = np.NaN
    sessions=np.arange(1,51)
    prev_event_choice = False
      
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        wrong=[]
        correct=[]
        correct_poke = 0
        prev_event_choice = False
        period_before_ITI= False
        for j, session in enumerate(subject_sessions):
            wrong=[]
            correct=[]
            prev_event_choice = False
            poke_I = 'poke_'+str(session.trial_data['configuration_i'][0])
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            events = [event.name for event in session.events if event.name in ['choice_state', 'period_before_iti', poke_I, poke_A, poke_B]]
            trials = [event.name for event in session.events if event.name in ['choice_state']]
            l_trials=len(trials)
            if l_trials == 0:
                continue
            for event in events:
                if event == 'choice_state':
                    prev_event_choice = True
                    period_before_ITI = True
                    correct.append(correct_poke) 
                    wrong.append(wrong_poke)
                    wrong_poke = 0
                    correct_poke=0 
                elif event == poke_I: 
                    if prev_event_choice == True and period_before_ITI== True:   
                        wrong_poke += 1                     
                elif event == poke_A:  
                    if prev_event_choice == True and period_before_ITI== True:
                        correct_poke += 1
                elif event == poke_B:
                    if prev_event_choice == True and period_before_ITI== True:
                        correct_poke += 1
                elif event == 'period_before_iti':
                    period_before_ITI = False
            number_I[n_subj,j] = (sum(wrong)/l_trials)
            number_A_B[n_subj,j] = (sum(correct)/l_trials)            
    number_I_mean=np.nanmean(number_I,axis = 0)
    sns.set()
    std_proportion=np.nanstd(number_I_mean, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    plt.figure()
    plt.fill_between(sessions, number_I_mean-std_err, number_I_mean+std_err, alpha=0.2, facecolor='r')
    plt.plot(sessions,number_I_mean,'r')
    plt.ylabel('Proportion of I to A/B pokes during the choice period')
    plt.xlabel('Session') 
    
# Experiment plot of poke A or B following the choice of A or B -------------------------------------------------------------------------
def session_A_B_poke_exp(experiment,subject_IDs ='all', fig_no = 1): 
    two_task_plot = True
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        print(subject_IDs)
        #n_subjects = len(subject_IDs)
    wrong_choice = []
    wrong_choice_task_2 = []
    wrong_count_task_2 = 0
    prev_choice = []
    wrong_count = 0
    choice_state = False
    wrong_ch = np.ones(shape =(14,40))
    wrong_ch[:] = np.NaN 
    sessions=np.arange(1,41)

    for n_subj, subject_ID in enumerate(subject_IDs):
        #if subject_ID < 478:
            #subject_ID += 5
        subject_sessions = experiment.get_sessions(subject_ID)
        wrong_choice =[]
        for j, session in enumerate(subject_sessions):
            choice_state_count = 0 
            wrong_choice =[]
            wrong_choice_task_2 = []
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            trials = [event.name for event in session.events if event.name in ['choice_state']]
            l_trials = len(trials)
            if two_task_plot == True: 
                tasks = session.trial_data['task']
                task_transitions = tasks[1:] - tasks[:-1]
                task_number = np.where(task_transitions == 1)[0] 
                if task_number > 0:
                    poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_number[0]])
                    poke_B_task_2 = 'poke_'+str(session.trial_data['poke_B'][task_number[0]])
                    events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial', poke_A, poke_B, poke_A_task_2, poke_B_task_2]]
                else:
                     events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial', poke_A, poke_B]]                   
            else:
                events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial', poke_A, poke_B]]   
            for i, event in enumerate(events):
                if task_number > 0:
                    if choice_state_count < task_number[0]:
                        if event == 'choice_state':
                            choice_state_count +=1
                            wrong_choice.append(wrong_count)
                            wrong_count = 0
                            choice_state = True
                        elif event == poke_A : 
                            if choice_state == True:
                                prev_choice = 'Poke_A'
                                choice_state = False
                            elif choice_state == False:
                                if prev_choice == 'Poke_B': 
                                    wrong_count += 1  
                        elif event == poke_A_task_2 or event == poke_B_task_2:
                            if choice_state == False:
                                if prev_choice == 'Poke_B' or prev_choice == 'Poke_A': 
                                    wrong_count += 1            
                        elif event == poke_B :
                            if choice_state == True:
                                prev_choice = 'Poke_B'
                                choice_state = False
                            elif choice_state == False: 
                                if prev_choice == 'Poke_A': 
                                    wrong_count += 1
                        elif event == 'init_trial':   
                            choice_state = False
                            wrong_count = 0 
                    elif choice_state_count >= task_number[0]:
                        if event == 'choice_state':
                            wrong_choice_task_2.append(wrong_count_task_2)
                            wrong_count_task_2 = 0
                            choice_state = True
                        elif event == poke_A_task_2 : 
                            if choice_state == True:
                                prev_choice = 'Poke_A'
                                choice_state = False
                            elif choice_state == False:
                                if prev_choice == 'Poke_B': 
                                    wrong_count_task_2 += 1  
                        elif event == poke_A or event == poke_B:
                            if choice_state == False:
                                if prev_choice == 'Poke_B' or prev_choice == 'Poke_A': 
                                    wrong_count_task_2 += 1  
                        elif event == poke_B_task_2 :
                            if choice_state == True:
                                prev_choice = 'Poke_B'
                                choice_state = False
                            elif choice_state == False: 
                                if prev_choice == 'Poke_A': 
                                    wrong_count_task_2 += 1
                        elif event == 'init_trial':   
                            choice_state = False
                            wrong_count_task_2 = 0     
                if task_number > 0:
                    wrong_ch[n_subj,j] = ((sum(wrong_choice)/task_number[0]) + (sum(wrong_choice_task_2)/(l_trials- task_number[0])))
                else:
                    wrong_ch[n_subj,j] = (sum(wrong_choice)/l_trials)
    wrong_ch_mean = np.nanmean(wrong_ch, axis = 0)
    std_dev=np.nanstd(wrong_ch, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_dev/sample_size
    #plt.errorbar(sessions, wrong_ch_mean, yerr = std_err)
    plt.fill_between(sessions, wrong_ch_mean-std_err, wrong_ch_mean+std_err, alpha=0.2, facecolor='b')
    plt.plot(sessions,wrong_ch_mean,'b')
    plt.ylabel('Number of A/B poke following A/B choice ')
    plt.xlabel('Session') 
    

 # Session plot of reversals-------------------------------------------------------------------------  
def session_plot_moving_average(session, fig_no = 1, is_subplot = False):
    FT = True
    block=session.trial_data['block']
    'Plot reward probabilities and moving average of choices for a single session.'
    if not is_subplot: plt.figure(fig_no, figsize = [7.5, 1.8]).clf()
    Block_transitions = block[1:]-block[:-1]
    choices = session.trial_data['choices']
    #threshold = block_transsitions(session.trial_data['pre-reversal trials']) # If you want threshold crossed on the plot 
    index_block = []
    n_trials = session.trial_data['n_trials']
    print(n_trials)
    if FT == True:
        forced_trials = session.trial_data['forced_trial']
        forced_array = np.where(forced_trials == 1)[0]
        forced_trials_sum = sum(forced_trials)
        choices = np.delete(choices, forced_array)
        block = np.delete(block, forced_array)
        n_trials = n_trials- forced_trials_sum
    Block_transitions = block[1:]-block[:-1]
    for i in Block_transitions:
        index_block = np.where(Block_transitions == 1)[0]
    for i in index_block:
        plot.axvline(x = i,color = 'g',linestyle = '-', lw = '1')
    #for i in threshold: # If you want threshold crossed on the plot 
    #    plot.axvline(x=i,color='k',linestyle='--', lw='0.6')
    plot.axhline(y = 0.25, color = 'r', lw = 0.8)
    plot.axhline(y = 0.75, color = 'r', lw = 0.8)
        
    exp_average= ut.exp_mov_ave(choices,initValue = 0.5,tau = 8)
    plt.plot(exp_average,'--')
    plt.ylim(-0,1)
    plt.xlim(1,n_trials)

    if not is_subplot: 
        plt.ylabel('Exp moving average ')
        plt.xlabel('Trials') 
   

# Experiment plot -------------------------------------------------------------------------
def experiment_plot(experiment, subject_IDs = 'all', when = 'all', fig_no = 1,
                   ignore_bc_trials = False):
    'Plot specified set of sessions from an experiment in a single figure'
    if subject_IDs == 'all': 
        subject_IDs = experiment.subject_IDs
        plt.figure(fig_no, figsize = [11.7,8.3]).clf()
    else:
        plt.figure(fig_no,figsize = [11,1])
    n_subjects = len(subject_IDs)
    for i,subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID, when)
        n_sessions = len(subject_sessions)
        for j, session in enumerate(subject_sessions):
            plt.subplot(n_subjects, n_sessions, i*n_sessions+j+1)
            session_plot_moving_average(session, is_subplot=True)
            if j == 0: plt.ylabel(session.subject_ID)
            if i == (n_subjects -1): plt.xlabel(str(session.datetime.date()))
    plt.tight_layout(pad=0.3)