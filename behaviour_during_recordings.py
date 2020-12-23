#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:59:13 2019

@author: veronikasamborska
"""
import sys
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

import remapping_count as rc 
import ephys_beh_import as ep
from scipy import io
import pingouin as pg      
# Script for plotting behavior during recordings

from statsmodels.stats.anova import AnovaRM


def import_data():
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

    #HP_LFP,PFC_LFP, m484_LFP, m479_LFP, m483_LFP, m478_LFP, m486_LFP, m480_LFP, m481_LFP, all_sessions_LFP = ep.import_code(ephys_path,beh_path, lfp_analyse = 'True') 
    HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')

    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    Data_HP = HP['Data'][0]
    DM_HP = HP['DM'][0]
    Data_PFC = PFC['Data'][0]
    DM_PFC = PFC['DM'][0]

def reversals_during_recordings(m484, m479, m483, m478, m486, m480, m481):
    subj = [m484, m479, m483, m478, m486, m480, m481]
    
    all_subjects = []
    for s in subj:
        s = np.asarray(s)
        date  = []
        for ses in s:
            date.append(ses.datetime)
            
        ind_sort = np.argsort(date)    
        s = s[ind_sort]
    
        all_sessions = []
        
        
        for session in s: 
            sessions_block = session.trial_data['block']
            n_trials = session.trial_data['n_trials']
            # Find trials where threshold crossed.
            prt = (session.trial_data['pre-reversal trials'] > 0).astype(int)
                
            forced_trials = session.trial_data['forced_trial']
            forced_trials_sum = sum(forced_trials)
            forced_array = np.where(forced_trials == 1)[0]
            sessions_block = np.delete(sessions_block, forced_array)
            prt = np.delete(prt,forced_array)
    
            n_trials = n_trials -  forced_trials_sum
            Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            threshold_crossing_trials = np.where((prt[1:] - prt[:-1]) == 1)[0]
            
            if len(threshold_crossing_trials) > 11:
                reversal_number = 0
                reversal_to_threshold = np.zeros((12))
                reversal_to_threshold[:] = np.NaN
         
                for i, crossing_trial in enumerate(threshold_crossing_trials): 
                 
                    if reversal_number <  12:
                        if i == 0:
                            reversal_to_threshold[reversal_number] = crossing_trial
                        else:
                            reversal_to_threshold[reversal_number] = (crossing_trial-reversal_trials[i-1])
                      
                        reversal_number += 1  
                all_sessions.append([reversal_to_threshold[:4], reversal_to_threshold[4:8],\
                                        reversal_to_threshold[8:12]])
            
        all_sessions_sub = np.concatenate(all_sessions) 
        all_sessions_sub = all_sessions_sub[:18,:]
        all_subjects.append(all_sessions_sub)
        
    all_subjects = np.asarray(all_subjects)
    tasks = all_subjects.shape[1]
    data = np.concatenate(all_subjects,0)
    data = np.concatenate(data,0)
    rev = np.tile(np.arange(4),126)
    task_n = np.tile(np.repeat(np.arange(18),4),7)
    n_subj =np.repeat(np.arange(7),72)
    

    # for task in range(tasks):
    #     pd.DataFrame(data=all_subjects[:,task,:]).to_csv('task{}_reversals_recording.csv'.format(task))

    anova = {'Data':data,'Sub_id': n_subj,'cond1': task_n, 'cond2':rev}
    
    anova_pd = pd.DataFrame.from_dict(data = anova)
                                      
    aovrm_es = pg.rm_anova(anova_pd, dv = 'Data', within=['cond1','cond2'],subject = 'Sub_id' )
    posthoc = pg.pairwise_ttests(data=anova_pd, dv='Data',within=['cond2'],subject = 'Sub_id',\
                             parametric=True, padjust='fdr_bh', effsize='hedges')
 
    
    all_rev = np.mean(all_subjects,axis = 0)
    std_err = (np.std(all_subjects, axis = 0))/7
    reversals = 4
    x=np.arange(reversals)
    plt.figure(figsize=(10,5))

  
    for i in range(tasks): 
         plt.plot(i * reversals + x, all_rev[i])
         plt.fill_between(i * reversals + x, all_rev[i]-std_err[i], all_rev[i]+std_err[i], alpha=0.2)
    
    rev_1 = np.nanmean(all_rev[:,0])
    rev_2 = np.nanmean(all_rev[:,1])
    rev_3 = np.nanmean(all_rev[:,2])
    rev_4 = np.nanmean(all_rev[:,3])
    
    st_1 = np.nanmean(std_err[:,0])
    st_2 = np.nanmean(std_err[:,1])
    st_3 = np.nanmean(std_err[:,2])
    st_4 = np.nanmean(std_err[:,3])
    
    xs = [1,2,3,4]
    rev = [rev_1,rev_2,rev_3,rev_4]
    st = [st_1,st_2,st_3,st_4]
    z = np.polyfit(xs, rev,1)
    p = np.poly1d(z)
    plt.figure()
    plt.plot(xs,p(xs),"--", color = 'grey', label = 'Trend Line')
  
    plt.errorbar(x = xs, y = rev, yerr = st, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    


    plt.ylabel('Number of Trials Till Threshold')
    plt.xlabel('Reversal Number')
    # Mean and Median Calculations
    mean_sub = np.mean(all_subjects, axis = 2)
    mean_sub = np.mean(mean_sub,axis = 0)
    std_threshold_per_task = np.std(all_subjects, axis = 2)
    std_threshold_per_task = np.std(std_threshold_per_task,axis = 0)
    sample_size=np.sqrt(7)
    std_err_median= std_threshold_per_task/sample_size
    x_pos = np.arange(len(mean_sub))
    plt.figure(figsize=(10,10))
    sns.set(style="white", palette="muted", color_codes=True)
    plt.errorbar(x = x_pos, y = mean_sub, yerr = std_err_median, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    
    
    z = np.polyfit(x_pos, mean_sub,1)
    p = np.poly1d(z)
    plt.plot(x_pos,p(x_pos),"--", color = 'grey', label = 'Trend Line')
    plt.ylim(14,30)
    


    
def out_of_sequence(m484, m479, m483, m478, m486, m480, m481,data_HP, data_PFC):
    #HP = m484 + m479 + m483
    #PFC = m478 + m486 + m480 + m481
    all_subjects = [data_HP['DM'][0][:16], data_HP['DM'][0][16:24],data_HP['DM'][0][24:],data_PFC['DM'][0][:9], data_PFC['DM'][0][9:25],data_PFC['DM'][0][25:39],data_PFC['DM'][0][39:]]
    subj = [m484, m479, m483, m478, m486, m480, m481[:-1]]
    all_subj_mean = []
    all_subj_std = []
    

    for s,subject in zip(subj, all_subjects):
        reversal_number = 0

        s = np.asarray(s)
        date  = []
        for ses in s:
            date.append(ses.datetime)
            
        ind_sort = np.argsort(date)    
        subject = np.asarray(subject)
        sesssions_dm = subject[ind_sort]
        sessions_beh_event = s[ind_sort]
        sess_count = 0
        all_sessions_wrong_ch = [] # Get the list with only trials that were treated as trials in task programme
        reversal_number = 0
        task_number = 0 

                    
        all_reversals = []
        all_tasks = []
         
        for session,session_event in zip(sesssions_dm,sessions_beh_event):
            sess_count += 1
            reversal_number = 0
            sessions_block = session[:,4]
            forced_trials = session[:,3]
            forced_array = np.where(forced_trials == 1)[0]
            sessions_block = np.delete(sessions_block, forced_array)
            Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            task = session[:,5]
            task = np.delete(task, forced_array)
           
            poke_I = np.delete(session[:,8],forced_array)
            poke_A = np.delete(session[:,6],forced_array)
            poke_B = np.delete(session[:,7],forced_array)
            
            taskid = rc.task_ind(task,poke_A,poke_B)
            task_1 = np.where(taskid  == 1)[0]
            task_2 = np.where(taskid  == 2)[0]
            task_3 = np.where(taskid  == 3)[0]

            
            poke_A_1 = 'poke_'+str(int(poke_A[task_1[0]]))
            poke_B_1 = 'poke_'+str(int(poke_B[task_1[0]]))
            poke_A_2 = 'poke_'+str(int(poke_A[task_2[0]]))
            poke_B_2 = 'poke_'+str(int(poke_B[task_2[0]]))
            poke_A_3 = 'poke_'+str(int(poke_A[task_3[0]]))
            poke_B_3 = 'poke_'+str(int(poke_B[task_3[0]]))
            
            poke_I_1 = 'poke_'+str(int(poke_I[task_1[0]]))
            poke_I_2 = 'poke_'+str(int(poke_I[task_2[0]]))
            poke_I_3 = 'poke_'+str(int(poke_I[task_3[0]]))
            

            reversal_trials = np.where(Block_transitions == 1)[0]    
            
            if len(reversal_trials) >= 12:
                task_number += 1

    
                events = [event.name for event in session_event.events if event.name in ['choice_state', 'init_trial','a_forced_state', 'b_forced_state',poke_A_1, poke_B_1,\
                                                                                   poke_A_2, poke_B_2,poke_A_3, poke_B_3,poke_I_1 ,\
                                                                                   poke_I_2,poke_I_3]]
    
    
                session_wrong_choice = []
                # Go through events list and find the events of interest 
                wrong_count = 0
                choice_state_count = 0
                wrong_count_state = []
                prev_choice  = 'forced_state'
                prev_choice_arr = []
                choice_state = False
                for event in events:
                    prev_choice_arr.append(prev_choice)
                    if event == 'choice_state':
                        session_wrong_choice.append(wrong_count)
                        wrong_count = 0
                        choice_state = True
                        choice_state_count += 1
                        wrong_count_state.append('choice')
                    
                    elif event == 'a_forced_state' or event == 'b_forced_state':
                        prev_choice  = 'forced_state'
                        
                    # In task 1 B is different to every other B, init  in 1 is the same as init in 3 --> so exclude init 3 
                    if choice_state_count in task_1:# Task 1 
                        if choice_state_count == task_1[0]:
                            prev_choice  = 'forced_state'
    
                        if event == poke_A_1: 
                            if choice_state == True:
                                prev_choice = 'Poke_A_1'
                                choice_state = False
                            elif choice_state == False and  prev_choice == 'Poke_B_1':
                                if event == poke_B_1:
                                    wrong_count += 1 
                                    wrong_count_state.append(poke_A_1)
                                        
                        elif event == poke_B_1 : 
                            if choice_state == True:
                                prev_choice = 'Poke_B_1'
                                choice_state = False
                            elif choice_state == False and prev_choice == 'Poke_A_1':
                                if event == poke_B_1:
                                    wrong_count += 1 
                                    wrong_count_state.append(poke_B_1)
#    
                        # elif event == poke_I_2: 
                        #     if choice_state == False and  prev_choice == 'Poke_B_1' or prev_choice == 'Poke_A_1' :
                        #         wrong_count += 1 
                        #         wrong_count_state.append(poke_I_2)
    
                                
                        # elif event == poke_B_2: 
                        #     if choice_state == False and  prev_choice == 'Poke_B_1' or prev_choice == 'Poke_A_1' :
                        #         wrong_count += 1 
                        #         wrong_count_state.append(poke_B_2)
    
                        # elif event == poke_B_3: 
                        #     if choice_state == False and  prev_choice == 'Poke_B_1' or prev_choice == 'Poke_A_1' :
                        #         wrong_count += 1 
                        #         wrong_count_state.append(poke_B_3)
    
                    
                    # In task 2 B is different to every other B, init in 2 becomes B in 3 --> so exclude B 3 but include other Inits
                    
                    elif choice_state_count in task_2:# Task 2
                         if choice_state_count == task_2[0]:
                            prev_choice  = 'forced_state'
    
                         if event == poke_A_2 : 
                            if choice_state == True:
                                prev_choice = 'Poke_A_2'
                                choice_state = False
                            elif choice_state == False and  prev_choice == 'Poke_B_2':
                                wrong_count += 1 
                                wrong_count_state.append(poke_B_2)
                            
                         elif event == poke_B_2: 
                            if choice_state == True:
                                prev_choice = 'Poke_B_2'
                                choice_state = False
                            elif choice_state == False and prev_choice == 'Poke_A_2':
                                wrong_count += 1 
                                wrong_count_state.append(poke_A_2)
                                    
                         # elif event == poke_I_1: 
                         #    if choice_state == False and  prev_choice == 'Poke_B_2' or prev_choice == 'Poke_A_2' :
                         #        wrong_count += 1 
                         #        wrong_count_state.append(poke_I_1)
                                
                         # elif event == poke_B_1: 
                         #    if choice_state == False and  prev_choice == 'Poke_B_2' or prev_choice == 'Poke_A_2' :
                         #        wrong_count += 1 
                         #        wrong_count_state.append(poke_B_1)
                         
                    # In task 3 B is the same as Init in task 2, init in 2 becomes B in 3 --> so exclude I2 but include other Bs
    
                    elif choice_state_count in task_3:# Task 2
                         if choice_state_count == task_3[0]:
                            prev_choice  = 'forced_state'
    
                         if event == poke_A_3 : 
                            if choice_state == True:
                                prev_choice = 'Poke_A_3'
                                choice_state = False
                            elif choice_state == False and  prev_choice == 'Poke_B_3':
                                wrong_count += 1 
                            
                         elif event == poke_B_3: 
                             
                            if choice_state == True:
                                prev_choice = 'Poke_B_3'
                                choice_state = False
                            elif choice_state == False and prev_choice == 'Poke_A_3':
                                wrong_count += 1 
#                                    
                         # elif event == poke_I_2: 
                         #    if choice_state == False and  prev_choice == 'Poke_B_3' or prev_choice == 'Poke_A_3' :
                         #        wrong_count += 1 
                                
                         # elif event == poke_B_1: 
                         #    if choice_state == False and  prev_choice == 'Poke_B_3' or prev_choice == 'Poke_A_3' :
                         #        wrong_count += 1 
                         
                         # elif event == poke_B_2: 
                         #    if choice_state == False and  prev_choice == 'Poke_B_3' or prev_choice == 'Poke_A_3' :
                         #        wrong_count += 1 
                        
                if sess_count == 1: 
                    all_sessions_wrong_ch = session_wrong_choice[:len(task)]
                    
                elif sess_count > 1: 
                    all_sessions_wrong_ch += session_wrong_choice[:len(task)]
                    
                 
                task_change = np.where(np.diff(task)!= 0)[0]
                for i in range(len(task)):
                    if i in reversal_trials:
                        reversal_number += 1
                            
                    if i in task_change:
                        task_number += 1
                        reversal_number = 0
    
                    all_reversals.append(reversal_number)
                    all_tasks.append(task_number)
        
     
        rev_over_4 = np.where(np.asarray(all_reversals) > 3)[0]
        all_reversals_np = np.delete(np.asarray(all_reversals),rev_over_4)
                
        all_tasks_np  = np.delete(np.asarray(all_tasks),rev_over_4)
        pokes_np = np.delete(np.asarray(all_sessions_wrong_ch),rev_over_4)
        
        task_pl = np.zeros((18,4))
        task_pl[:] = np.NaN
        std_plt = np.zeros((18,4))
        std_plt[:] = np.NaN
        
        where_21_t = np.where(all_tasks_np > 18)[0]
        all_reversals_np = np.delete(np.asarray(all_reversals_np),where_21_t)
        all_tasks_np  = np.delete(np.asarray(all_tasks_np),where_21_t)
        pokes_np = np.delete(np.asarray(pokes_np),where_21_t)
        for t in np.unique(all_tasks_np):
            for r in np.unique(all_reversals_np):
                plot_task = pokes_np[(all_tasks_np==t) & (all_reversals_np==r)]# For plots from all trials 
                #print(len(plot_task))
                mean_plot  = np.mean(plot_task)
                std_plot  = np.std(plot_task)

                task_pl[t-1,r] = mean_plot
                std_plt[t-1,r] = std_plot
 
        all_subj_mean.append(task_pl)
        all_subj_std.append(std_plt)
        
    all_subj_mean_np = np.asarray(all_subj_mean)
    tasks = all_subj_mean_np.shape[1]
    for task in range(tasks):
        pd.DataFrame(data=all_subj_mean_np[:,task,:]).to_csv('task{}_rab_recording.csv'.format(task))
  
    
    tasks = all_subj_mean_np.shape[1]
    data = np.concatenate(all_subj_mean_np,0)
    data = np.concatenate(data,0)
    rev = np.tile(np.arange(4),126)
    task_n = np.tile(np.repeat(np.arange(18),4),7)
    n_subj =np.repeat(np.arange(7),72)
    

    # for task in range(tasks):
    #     pd.DataFrame(data=all_subjects[:,task,:]).to_csv('task{}_reversals_recording.csv'.format(task))

    anova = {'Data':data,'Sub_id': n_subj,'cond1': task_n, 'cond2':rev}
    
    anova_pd = pd.DataFrame.from_dict(data = anova)
                                      
    aovrm_es = pg.rm_anova(anova_pd, dv = 'Data', within=['cond1','cond2'],subject = 'Sub_id' )
    posthoc = pg.pairwise_ttests(data=anova_pd, dv='Data',within=['cond2'],subject = 'Sub_id',\
                             parametric=True, padjust='fdr_bh', effsize='hedges')
 
    

    all_rev = np.mean(all_subj_mean,axis = 0)
    std_err = (np.std(all_subj_mean, axis = 0))/7
    reversals = 4
    x=np.arange(reversals)
    plt.figure(figsize=(10,5))

    for i in range(tasks): 
         plt.plot(i * reversals + x, all_rev[i])
         plt.fill_between(i * reversals + x, all_rev[i]-std_err[i], all_rev[i]+std_err[i], alpha=0.2)
    
    rev_1 = np.nanmean(all_rev[:,0])
    rev_2 = np.nanmean(all_rev[:,1])
    rev_3 = np.nanmean(all_rev[:,2])
    rev_4 = np.nanmean(all_rev[:,3])
    
    st_1 = np.nanmean(std_err[:,0])
    st_2 = np.nanmean(std_err[:,1])
    st_3 = np.nanmean(std_err[:,2])
    st_4 = np.nanmean(std_err[:,3])
    
    xs = [1,2,3,4]
    rev = [rev_1,rev_2,rev_3,rev_4]
    st = [st_1,st_2,st_3,st_4]
    z = np.polyfit(xs, rev,1)
    p = np.poly1d(z)
    plt.figure()
    plt.plot(xs,p(xs),"--", color = 'grey', label = 'Trend Line')
  
    plt.errorbar(x = xs, y = rev, yerr = st, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    

    #plt.ylim(0,1.6)
    plt.ylabel('Number of Trials Till Threshold')
    plt.xlabel('Reversal Number')
    mean_rev =  np.mean(all_subj_mean,axis = 2)
    std_rev =  np.std(all_subj_mean,axis = 2)

    med_sub = np.mean(mean_rev,axis = 0)
    std_sub = np.std(std_rev, axis = 0)
    sample_size=np.sqrt(7)
    std_err_median= std_sub/sample_size
    x_pos = np.arange(len(med_sub))
    plt.figure(figsize=(10,10))
    sns.set(style="white", palette="muted", color_codes=True)
    plt.errorbar(x = x_pos, y = med_sub, yerr = std_err_median, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    
    
    z = np.polyfit(x_pos, med_sub,1)
    p = np.poly1d(z)
    plt.plot(x_pos,p(x_pos),"--", color = 'grey', label = 'Trend Line')
    #plt.ylim(0,0.9)
    
    return aovrm_es, posthoc

