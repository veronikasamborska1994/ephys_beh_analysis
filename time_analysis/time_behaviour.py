#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:05:33 2020

@author: veronikasamborska
"""

# =============================================================================
# Calculate how much animals are affected by errros in the beginning vs end of blocks (works)
# =============================================================================
import numpy as np
import pylab as plt
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')

from palettable import wesanderson as wes
import heatmap_aligned as ha
from scipy import io
import ephys_beh_import as ep
import seaborn as sns 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)


def load_data():
   
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
    HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
    experiment_aligned_PFC = ha.all_sessions_aligment(PFC, all_sessions)
    experiment_aligned_HP = ha.all_sessions_aligment(HP, all_sessions)
    data_HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    data_PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
#    
    return data_HP, data_PFC,experiment_aligned_PFC,experiment_aligned_HP

def plot_fractions(data_HP,data_PFC):
    session_switch_repeat, session_switch_repeat_1st_half,session_switch_repeat_2nd_half = switch_behaviour(data_HP, data_PFC)
    subject_mean = []; subject_mean_1st = []; subject_mean_2nd = []
    for s, subject in enumerate(session_switch_repeat):
        subject_mean.append(np.mean(session_switch_repeat[s]))
        subject_mean_1st.append(np.mean(session_switch_repeat_1st_half[s]))
        subject_mean_2nd.append(np.mean(session_switch_repeat_2nd_half[s]))

    all_subjects_mean = np.mean(subject_mean)
    all_subjects_std = (np.std(subject_mean))/np.sqrt(len(subject_mean))

    all_subjects_mean_1st = np.mean(subject_mean_1st)
    all_subjects_std_1st = (np.std(subject_mean_1st))/np.sqrt(len(subject_mean_1st))

    all_subjects_mean_2nd = np.mean(subject_mean_2nd)
    all_subjects_std_2nd = (np.std(subject_mean_2nd))/np.sqrt(len(subject_mean_2nd))
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors

    plt.bar([0,1,2],[all_subjects_mean,all_subjects_mean_1st,all_subjects_mean_2nd],0.5,
            yerr = [all_subjects_std,all_subjects_std_1st,all_subjects_std_2nd], color = [c[0], c[1],c[2]])
    plt.xticks([0,1,2],['Average Stay/Switch','First Half of the Block Stay/Switch', 'Second Half of the Block Stay/Switch' ])
    
    
    plt.bar([0,1],[all_subjects_mean_1st-all_subjects_mean, all_subjects_mean_2nd-all_subjects_mean],
            yerr = [all_subjects_std,all_subjects_std], color = [c[0], c[1]])
    
    plt.xticks([0,1],['1st Half Relative to Average','2nd Half Relative to Average'])
    sns.despine()
    plt.ylabel('Fraction Change')
   
def switch_behaviour(data_HP,data_PFC):

    all_subjects = [data_HP['DM'][0][:16], data_HP['DM'][0][16:24],data_HP['DM'][0][24:],data_PFC['DM'][0][:9], data_PFC['DM'][0][9:26],data_PFC['DM'][0][26:40],data_PFC['DM'][0][40:]]

    session_switch_repeat_subject = []; session_switch_repeat_1st_half_subject = []; session_switch_repeat_2nd_halfsubject = []
    for subject in all_subjects: 
        session_switch_repeat = []; session_switch_repeat_1st_half = []; session_switch_repeat_2nd_half = []

        for  s, sess in enumerate(subject):
            DM = subject[s]
            choices = DM[:,1]
            reward = DM[:,2]    
        
            block = DM[:,4]
            block_df = np.diff(block)
            ind_block = np.where(block_df != 0)[0]
    
            if len(ind_block) >= 11:
                
                trials_since_block = []
                t = 0
                
                for st,s in enumerate(block):
                    if block[st-1] != block[st]:
                        t = 0
                    else:
                        t+=1
                    trials_since_block.append(t)
                    
                #block_totals_ind = (np.where(np.asarray(ind_block) == 1)[0]-1)[1:]
                block_totals_ind = ind_block
                block_totals = np.diff(block_totals_ind)-1
                trials_since_block = trials_since_block[:ind_block[11]]
                fraction_list = []
    
    
                for t,trial in enumerate(trials_since_block):
                    
                    if t <= block_totals_ind[0]:
                        fr = trial/block_totals_ind[0]
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[0] and  t <= block_totals_ind[1]:
                        fr = trial/block_totals[0]
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[1] and  t <= block_totals_ind[2]:
                        fr = trial/block_totals[1]               
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[2] and  t <= block_totals_ind[3]:
                        fr = trial/block_totals[2]                
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[3] and  t <= block_totals_ind[4]:
                        fr = trial/block_totals[3]
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[4] and  t <= block_totals_ind[5]:
                        fr = trial/block_totals[4]
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[5] and  t <= block_totals_ind[6]:
                        fr = trial/block_totals[5]
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[6] and  t <= block_totals_ind[7]:
                        fr = trial/block_totals[6]  
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[7] and  t <= block_totals_ind[8]:
                        fr = trial/block_totals[7]
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[8] and  t <= block_totals_ind[9]:
                        fr = trial/block_totals[8]                 
                        fraction_list.append(fr)
    
                    elif t > block_totals_ind[9] and  t <= block_totals_ind[10]:
                        fr = trial/block_totals[9]
                        fraction_list.append(fr)
    
                    elif t >  block_totals_ind[10] and  t <= len(trials_since_block):
                        fr = trial/trials_since_block[-1]
                        fraction_list.append(fr)
                        
                choices = choices[:ind_block[11]]
                reward = reward[:ind_block[11]]
                no_rew = np.where(reward == 0)[0]
                repeat = 0
                switch = 0
                for ch, choice in enumerate(choices):
                    if ch != (len(choices)-1):
                        if ch in no_rew:
                            if choices[ch+1] == choices[ch]:
                                repeat +=1
                            if choices[ch+1] != choices[ch]:
                                switch +=1
                            
                switch_to_repeat = repeat/switch
                
                first_half_repeat = 0
                second_half_repeat = 0
                first_half_switch = 0
                second_half_switch = 0
                fraction_list_first = np.where(np.asarray(fraction_list) <= 0.5)[0]
                fraction_list_second = np.where(np.asarray(fraction_list) >= 0.5)[0]
    
                for ch, choice in enumerate(choices):
                    if ch != (len(choices)-1):
    
                        if ch in no_rew:
                            if ch in fraction_list_first:
                                if choices[ch+1] == choices[ch]:
                                    first_half_repeat +=1
                                if choices[ch+1] != choices[ch]:
                                    first_half_switch +=1
                            elif ch in fraction_list_second:
                                if choices[ch+1] == choices[ch]:
                                    second_half_repeat +=1
                                if choices[ch+1] != choices[ch]:
                                    second_half_switch +=1
          
                switch_to_repeat_first_half = first_half_repeat/first_half_switch
                switch_to_repeat_second_half = second_half_repeat/second_half_switch
                
                session_switch_repeat.append(switch_to_repeat)
    
                session_switch_repeat_1st_half.append(switch_to_repeat_first_half)
                session_switch_repeat_2nd_half.append(switch_to_repeat_second_half)
                
        session_switch_repeat_subject.append(session_switch_repeat); session_switch_repeat_1st_half_subject.append(session_switch_repeat_1st_half)
        session_switch_repeat_2nd_halfsubject.append(session_switch_repeat_2nd_half)

    return session_switch_repeat_subject, session_switch_repeat_1st_half_subject,session_switch_repeat_2nd_halfsubject
            