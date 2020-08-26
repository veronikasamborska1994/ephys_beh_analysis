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
from statsmodels.stats.anova import AnovaRM
import pandas as pd
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')

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
    
    s_switch_to_repeat_1, s_switch_to_repeat_2, s_switch_to_repeat_3, s_switch_to_repeat_4, s_switch_to_repeat_5  = switch_behaviour(data_HP, data_PFC)
    subject_mean_1 = []; subject_mean_2 = []; subject_mean_3 = []; subject_mean_4 = [];subject_mean_5 = []
    s_switch_to_repeat_1, s_switch_to_repeat_2, s_switch_to_repeat_3, s_switch_to_repeat_4  = switch_behaviour_a_in_a_b_in_b(data_HP, data_PFC)
   
   
    
    for s, subject in enumerate(s_switch_to_repeat_1):
        subject_mean_1.append(np.mean(s_switch_to_repeat_1[s])); subject_mean_2.append(np.mean(s_switch_to_repeat_2[s]))
        subject_mean_3.append(np.mean(s_switch_to_repeat_3[s])); subject_mean_4.append(np.mean(s_switch_to_repeat_4[s])) # subject_mean_5.append(np.mean(s_switch_to_repeat_5[s]))
      
    fractions = [np.asarray(subject_mean_1),np.asarray(subject_mean_2),np.asarray(subject_mean_3),\
                                        np.asarray(subject_mean_4)]

    fractions = np.concatenate(fractions,0)
    subject_id = np.tile(np.arange(7), 4)
    fraction_id = np.zeros(28)
    fraction_id[7:14] = 1
    fraction_id[14:21] = 2
    fraction_id[21:28] = 3
    #fraction_id[28:35] = 4
    
    
    data = {'Data':fractions,'Sub_id': subject_id,'cond': fraction_id}
    data = pd.DataFrame.from_dict(data = data)
    
    
    aovrm = AnovaRM(data, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()

    


    all_subjects_mean_1 = np.mean(subject_mean_1); all_subjects_std_1 = (np.std(subject_mean_1))/np.sqrt(len(subject_mean_1))

    all_subjects_mean_2 = np.mean(subject_mean_2); all_subjects_std_2 = (np.std(subject_mean_2))/np.sqrt(len(subject_mean_2))
   
    all_subjects_mean_3 = np.mean(subject_mean_3); all_subjects_std_3 = (np.std(subject_mean_3))/np.sqrt(len(subject_mean_3))

    all_subjects_mean_4 = np.mean(subject_mean_4); all_subjects_std_4 = (np.std(subject_mean_4))/np.sqrt(len(subject_mean_4))
   
    #all_subjects_mean_5 = np.mean(subject_mean_5); all_subjects_std_5 = (np.std(subject_mean_5))/np.sqrt(len(subject_mean_5))
 
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
  
    plt.subplot(1,2,2)
    plt.bar(np.arange(4),[all_subjects_mean_1, all_subjects_mean_2,
                          all_subjects_mean_3, all_subjects_mean_4],
            yerr = [all_subjects_std_1,all_subjects_std_2,all_subjects_std_3,all_subjects_std_4],color = c[3])
            
    sns.despine()
    plt.ylabel('Switch/ (Repeat + Switch)')
    plt.xlabel('Fraction in Block')
    plt.title('Post Reward')
    
   
def switch_behaviour(data_HP,data_PFC):

    all_subjects = [data_HP['DM'][0][:16], data_HP['DM'][0][16:24],data_HP['DM'][0][24:],data_PFC['DM'][0][:9], data_PFC['DM'][0][9:26],data_PFC['DM'][0][26:40],data_PFC['DM'][0][40:]]

    s_switch_to_repeat_1 =[];s_switch_to_repeat_2 =[];s_switch_to_repeat_3 =[]; s_switch_to_repeat_4 =[]; s_switch_to_repeat_5 =[]
        
    for subject in all_subjects: 
        switch_to_repeat_1 =[];switch_to_repeat_2 =[];switch_to_repeat_3 =[];switch_to_repeat_4 =[];switch_to_repeat_5 =[]
        
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
                            
                switch_to_repeat = repeat/(switch+repeat)
                
                repeat_1 = 0; repeat_2 = 0; repeat_3 = 0; repeat_4 = 0;repeat_5 = 0
                switch_1 = 0; switch_2 = 0;switch_3 = 0; switch_4 = 0;switch_5 = 0
                
                fraction_list_1 = np.where(np.asarray(fraction_list) <= 0.2)[0]
                fraction_list_2 = np.where((np.asarray(fraction_list) <=  0.4) & (np.asarray(fraction_list) >= 0.2))[0]
                fraction_list_3 = np.where((np.asarray(fraction_list) <=  0.6) & (np.asarray(fraction_list) >= 0.4))[0]
                fraction_list_4 = np.where((np.asarray(fraction_list) <=  0.8) & (np.asarray(fraction_list) >= 0.6))[0]
                fraction_list_5 = np.where((np.asarray(fraction_list) <=  1) & (np.asarray(fraction_list) >= 0.8))[0]
              
                for ch, choice in enumerate(choices):
                    if ch != (len(choices)-1):
                        if ch in no_rew:
                            if ch in fraction_list_1:
                                if choices[ch+1] == choices[ch]:
                                    repeat_1 +=1
                                if choices[ch+1] != choices[ch]:
                                    switch_1 +=1
                                    
                            elif ch in fraction_list_2:
                                if choices[ch+1] == choices[ch]:
                                    repeat_2 +=1
                                if choices[ch+1] != choices[ch]:
                                    switch_2 +=1
                            
                            elif ch in fraction_list_3:
                                if choices[ch+1] == choices[ch]:
                                    repeat_3 +=1
                                if choices[ch+1] != choices[ch]:
                                    switch_3 +=1
                            
                            
                            elif ch in fraction_list_4:
                                if choices[ch+1] == choices[ch]:
                                    repeat_4 +=1
                                if choices[ch+1] != choices[ch]:
                                    switch_4 +=1
                          
                            elif ch in fraction_list_5:
                                if choices[ch+1] == choices[ch]:
                                    repeat_5 +=1
                                if choices[ch+1] != choices[ch]:
                                    switch_5 +=1
                          
                           
              
                switch_to_repeat_1.append((repeat_1/(switch_1+repeat_1)))
                switch_to_repeat_2.append((repeat_2/(switch_2+repeat_2)))
                switch_to_repeat_3.append((repeat_3/(switch_3+repeat_3)))
                switch_to_repeat_4.append((repeat_4/(switch_4+repeat_4)))
                switch_to_repeat_5.append((repeat_5/(switch_5+repeat_5)))
                

        s_switch_to_repeat_1.append(switch_to_repeat_1); s_switch_to_repeat_2.append(switch_to_repeat_2); s_switch_to_repeat_3.append(switch_to_repeat_3)
        s_switch_to_repeat_4.append(switch_to_repeat_4); s_switch_to_repeat_5.append(switch_to_repeat_5)
        

    return s_switch_to_repeat_1, s_switch_to_repeat_2, s_switch_to_repeat_3, s_switch_to_repeat_4, s_switch_to_repeat_5



def switch_behaviour_a_in_a_b_in_b(data_HP,data_PFC):

    all_subjects = [data_HP['DM'][0][:16], data_HP['DM'][0][16:24],data_HP['DM'][0][24:],data_PFC['DM'][0][:9], data_PFC['DM'][0][9:26],data_PFC['DM'][0][26:40],data_PFC['DM'][0][40:]]

    s_switch_to_repeat_1 =[];s_switch_to_repeat_2 =[];s_switch_to_repeat_3 =[]; s_switch_to_repeat_4 =[]
        
    for subject in all_subjects: 
        switch_to_repeat_1 =[];switch_to_repeat_2 =[];switch_to_repeat_3 =[];switch_to_repeat_4 =[]
        
        for  s, sess in enumerate(subject):
            DM = subject[s]
            state = DM[:,0]
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
                state = state[:ind_block[11]]
                
                state_a_choices = np.where((state == 1) & (choices == 1))[0]
                state_b_choices = np.where((state == 0) & (choices == 0))[0]
                ch_a_b = np.concatenate([state_a_choices,state_b_choices])
                no_rew = np.where(reward == 1)[0]
                repeat = 0
                switch = 0
                for ch, choice in enumerate(choices):
                    if ch != (len(choices)-1):
                        if ch in ch_a_b:
                            if ch in no_rew:
                                if choices[ch+1] == choices[ch]:
                                    repeat +=1
                                if choices[ch+1] != choices[ch]:
                                    switch +=1
                                
                switch_to_repeat = switch/(switch+repeat)
                
                repeat_1 = 0; repeat_2 = 0; repeat_3 = 0; repeat_4 = 0
                switch_1 = 0; switch_2 = 0;switch_3 = 0; switch_4 = 0
                
                fraction_list_1 = np.where(np.asarray(fraction_list) <=  0.166)[0]
                fraction_list_2 = np.where((np.asarray(fraction_list) <=  0.5) & (np.asarray(fraction_list) >=  0.25))[0]
                fraction_list_3 = np.where((np.asarray(fraction_list) <=  0.75) & (np.asarray(fraction_list) >= 0.5))[0]
                fraction_list_4 = np.where((np.asarray(fraction_list) <=  1) & (np.asarray(fraction_list) >= 0.75))[0]
                fraction_list_5 = np.where((np.asarray(fraction_list) <=  1) & (np.asarray(fraction_list) >= 0.75))[0]
                fraction_list_6 = np.where((np.asarray(fraction_list) <=  1) & (np.asarray(fraction_list) >= 0.75))[0]

                for ch, choice in enumerate(choices):
                    if ch != (len(choices)-1):
                        if ch in no_rew:
                            if ch in ch_a_b:
                                if ch in fraction_list_1:
                                    if choices[ch+1] == choices[ch]:
                                        repeat_1 +=1
                                    if choices[ch+1] != choices[ch]:
                                        switch_1 +=1
                                        
                                elif ch in fraction_list_2:
                                    if choices[ch+1] == choices[ch]:
                                        repeat_2 +=1
                                    if choices[ch+1] != choices[ch]:
                                        switch_2 +=1
                                
                                elif ch in fraction_list_3:
                                    if choices[ch+1] == choices[ch]:
                                        repeat_3 +=1
                                    if choices[ch+1] != choices[ch]:
                                        switch_3 +=1
                                
                                
                                elif ch in fraction_list_4:
                                    if choices[ch+1] == choices[ch]:
                                        repeat_4 +=1
                                    if choices[ch+1] != choices[ch]:
                                        switch_4 +=1
                              
                           
              
                switch_to_repeat_1.append((switch_1/(switch_1+repeat_1)))
                switch_to_repeat_2.append((switch_2/(switch_2+repeat_2)))
                switch_to_repeat_3.append((switch_3/(switch_3+repeat_3)))
                switch_to_repeat_4.append((switch_4/(switch_4+repeat_4)))
                

        s_switch_to_repeat_1.append(switch_to_repeat_1); s_switch_to_repeat_2.append(switch_to_repeat_2); s_switch_to_repeat_3.append(switch_to_repeat_3)
        s_switch_to_repeat_4.append(switch_to_repeat_4)

    return s_switch_to_repeat_1, s_switch_to_repeat_2, s_switch_to_repeat_3, s_switch_to_repeat_4


