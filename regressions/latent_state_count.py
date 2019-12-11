#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:01:51 2019

@author: veronikasamborska
"""

from random import randint
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.io
from sklearn import svm
from sklearn import metrics
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import remapping_count as rc 
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import utility as ut

HP = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
PFC = scipy.io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
#
Data_HP = HP['Data'][0]
DM_HP = HP['DM'][0]
Data_PFC = PFC['Data'][0]
DM_PFC = PFC['DM'][0]

session_list_PFC = select_trials(Data_PFC, DM_PFC, 62, ind_time = np.arange(30,40))
correct_A_A_PFC, correct_B_A_PFC, correct_A_B_PFC, correct_B_B_PFC = decode_time_in_block(session_list_PFC,'PFC')

session_list_HP = select_trials(Data_HP, DM_HP, 62, ind_time = np.arange(30,40))

correct_A_A_HP, correct_B_A_HP, correct_A_B_HP, correct_B_B_HP = decode_time_in_block(session_list_HP,'HP')

def perm_test(firing_1,firing_2,n_perm):
    
    activity_perm = np.zeros((n_perm, 1))
    #activity_perm_check = np.zeros((n_perm, 1))
    
    for i in range(n_perm):
        np.random.shuffle(firing_1)   
        np.random.shuffle(firing_2)
        activity_perm[i,:] = np.corrcoef(firing_1, firing_2)[1,0]
        #perm_check_1  = np.flip(firing_1)   
       # perm_check_2 = np.flip(firing_2)
        #activity_perm_check[i,:] = np.corrcoef(firing_1, firing_2)[1,0]

    
    p = np.percentile(activity_perm,95)
    
    return activity_perm,p#, activity_perm_check
   



def interpolate_trials(Data_HP,DM_HP, Data_PFC, DM_PFC, ):
    
    all_sessions_trials_per_block = []
    
    for data,dm in zip(Data_HP, DM_HP):
        trials, neurons, time = data.shape
        block = dm[:,4]
        
        state_change = np.where(np.diff(block)!=0)[0]+1
        state_change = np.append(state_change,0)
        state_change = np.sort(state_change)
           
        state_trial_n = np.diff(state_change)
        all_sessions_trials_per_block.append(state_trial_n)
        
    for data,dm in zip(Data_PFC, DM_PFC):
        trials, neurons, time = data.shape
        block = dm[:,4]

        state_change = np.where(np.diff(block)!=0)[0]+1
        state_change = np.append(state_change,0)
        state_change = np.sort(state_change)
           
        state_trial_n = np.diff(state_change)
        all_sessions_trials_per_block.append(state_trial_n)
        
    all_sessions_trials_per_block = np.concatenate(np.asarray(all_sessions_trials_per_block), axis = 0)
    max_number_per_block = int(np.max(all_sessions_trials_per_block))
    
    return max_number_per_block

def state_behaviour_ind(state_change_t1,state_change_t2,state_change_t3, change, data):
    #Task 1 
    state_change_t1_1_ind = np.arange(state_change_t1[0],state_change_t1[1])
    state_change_t1_2_ind = np.arange(state_change_t1[1],state_change_t1[2])
    state_change_t1_3_ind = np.arange(state_change_t1[2],state_change_t1[3])
    
    ind_st_1_4 = np.where(change == state_change_t1[3])[0]
    
    if (len(change)-1) == ind_st_1_4:
        state_change_t1_4_ind = np.arange(state_change_t1[3],len(data))
    else:
        state_change_t1_4_ind = np.arange(state_change_t1[3],change[ind_st_1_4+1])
    

    #Task 2
    state_change_t2_1_ind = np.arange(state_change_t2[0],state_change_t2[1])
    state_change_t2_2_ind = np.arange(state_change_t2[1],state_change_t2[2])
    state_change_t2_3_ind = np.arange(state_change_t2[2],state_change_t2[3])
    ind_st_2_4 = np.where(change ==state_change_t2[3])[0]
    
    if (len(change)-1) == ind_st_2_4:
        state_change_t2_4_ind = np.arange(state_change_t2[3],len(data))
    else:
        state_change_t2_4_ind = np.arange(state_change_t2[3],change[ind_st_2_4+1])
  
    #Task 3
    state_change_t3_1_ind = np.arange(state_change_t3[0],state_change_t3[1])
    state_change_t3_2_ind = np.arange(state_change_t3[1],state_change_t3[2])
    state_change_t3_3_ind = np.arange(state_change_t3[2],state_change_t3[3])
    ind_st_3_4 = np.where(change ==state_change_t3[3])[0]
    
    if (len(change)-1) == ind_st_3_4:
        state_change_t3_4_ind = np.arange(state_change_t3[3],len(data))
    else:
        state_change_t3_4_ind = np.arange(state_change_t3[3],change[ind_st_3_4+1])
    
    return state_change_t1_1_ind,state_change_t1_2_ind, state_change_t1_3_ind,state_change_t1_4_ind,\
    state_change_t2_1_ind,state_change_t2_2_ind, state_change_t2_3_ind,state_change_t2_4_ind,\
    state_change_t3_1_ind,state_change_t3_2_ind, state_change_t3_3_ind,state_change_t3_4_ind



def choose_a_a_b_b(choice_a_state_a, choice_b_state_b,state_change_t1_1_ind,state_change_t1_2_ind, state_change_t1_3_ind,state_change_t1_4_ind,\
    state_change_t2_1_ind,state_change_t2_2_ind, state_change_t2_3_ind,state_change_t2_4_ind,\
    state_change_t3_1_ind,state_change_t3_2_ind, state_change_t3_3_ind,state_change_t3_4_ind, block, block_ch):
    
    
    #Task 1 
    task_1_st = np.concatenate((state_change_t1_1_ind,state_change_t1_2_ind, state_change_t1_3_ind,state_change_t1_4_ind))
    t1_a = np.intersect1d(choice_a_state_a,task_1_st)
    t1_b = np.intersect1d(choice_b_state_b,task_1_st)
    
    t1_a_1 = t1_a[np.where((t1_a > block_ch[0]) & (t1_a < block_ch[1]))[0]]
    t1_b_1 = t1_b[np.where((t1_b > block_ch[0]) & (t1_b < block_ch[1]))[0]]
    
    t1_a_2 = t1_a[np.where((t1_a > block_ch[1]) & (t1_a < block_ch[2]))[0]]
    t1_b_2 = t1_b[np.where((t1_b > block_ch[1]) & (t1_b < block_ch[2]))[0]]
   
    t1_a_3 = t1_a[np.where((t1_a > block_ch[2]) & (t1_a < block_ch[3]))[0]]
    t1_b_3 = t1_b[np.where((t1_b > block_ch[2]) & (t1_b < block_ch[3]))[0]]
    
    t1_a_4 = t1_a[np.where(t1_a > block_ch[3])]
    t1_b_4 = t1_b[np.where(t1_b > block_ch[3])]
    
    if len(t1_a_1) > 0 :     
        t1_a_state_1 = t1_a_1.astype(int)
    elif len(t1_a_2) > 0:
        t1_a_state_1 = t1_a_2.astype(int)

    if len(t1_b_1) > 0 :     
        t1_b_state_1 = t1_b_1.astype(int)
    elif len(t1_b_2) > 0:
        t1_b_state_1 = t1_b_2.astype(int)

    if len(t1_a_3) > 0 :     
        t1_a_state_2 = t1_a_3.astype(int)
    elif len(t1_a_4) > 0:
        t1_a_state_2 = t1_a_4.astype(int)

    if len(t1_b_3) > 0 :     
        t1_b_state_2 = t1_b_3.astype(int)
    elif len(t1_b_4) > 0:
        t1_b_state_2 = t1_b_4.astype(int)

    
    #Task 2 
    task_2_st = np.concatenate((state_change_t2_1_ind,state_change_t2_2_ind, state_change_t2_3_ind,state_change_t2_4_ind))
    t2_a = np.intersect1d(choice_a_state_a,task_2_st)
    t2_b = np.intersect1d(choice_b_state_b,task_2_st)
    
    t2_a_1 = t2_a[np.where((t2_a > block_ch[4]) & (t2_a < block_ch[5]))[0]]
    t2_b_1 = t2_b[np.where((t2_b > block_ch[4]) & (t2_b < block_ch[5]))[0]]
   
    t2_a_2 = t2_a[np.where((t2_a > block_ch[5]) & (t2_a < block_ch[6]))[0]]
    t2_b_2 = t2_b[np.where((t2_b > block_ch[5]) & (t2_b < block_ch[6]))[0]]
   
    t2_a_3 = t2_a[np.where((t2_a > block_ch[6]) & (t2_a < block_ch[7]))[0]]
    t2_b_3 = t2_b[np.where((t2_b > block_ch[6]) & (t2_b < block_ch[7]))[0]]
    
    t2_a_4 = t2_a[np.where(t2_a > block_ch[7])]
    t2_b_4 = t2_b[np.where(t2_b > block_ch[7])]
    
    if len(t2_a_1) > 0 :     
        t2_a_state_1 = t2_a_1.astype(int)
    elif len(t2_a_2) > 0:
        t2_a_state_1 = t2_a_2.astype(int)

    if len(t2_b_1) > 0 :     
        t2_b_state_1 = t2_b_1.astype(int)
    elif len(t2_b_2) > 0:
        t2_b_state_1 = t2_b_2.astype(int)

    if len(t2_a_3) > 0 :     
        t2_a_state_2 = t2_a_3.astype(int)
    elif len(t2_a_4) > 0:
        t2_a_state_2 = t2_a_4.astype(int)

    if len(t2_b_3) > 0 :     
        t2_b_state_2 = t2_b_3.astype(int)
    elif len(t2_b_4) > 0:
        t2_b_state_2 = t2_b_4.astype(int)
    
    #Task 3
    task_3_st = np.concatenate((state_change_t3_1_ind,state_change_t3_2_ind, state_change_t3_3_ind,state_change_t3_4_ind))
    t3_a = np.intersect1d(choice_a_state_a,task_3_st)
    t3_b = np.intersect1d(choice_b_state_b,task_3_st)
    
    t3_a_1 = t3_a[np.where((t3_a > block_ch[8]) & (t3_a < block_ch[9]))[0]]
    t3_b_1 = t3_b[np.where((t3_b > block_ch[8]) & (t3_b < block_ch[9]))[0]]
   
    t3_a_2 = t3_a[np.where((t3_a > block_ch[9]) & (t3_a < block_ch[10]))[0]]
    t3_b_2 = t3_b[np.where((t3_b > block_ch[9]) & (t3_b < block_ch[10]))[0]]
   
    t3_a_3 = t3_a[np.where((t3_a > block_ch[10]) & (t3_a < block_ch[11]))[0]]
    t3_b_3 = t3_b[np.where((t3_b > block_ch[10]) & (t3_b < block_ch[11]))[0]]
    
    t3_a_4 = t3_a[np.where(t3_a > block_ch[11:])]
    t3_b_4 = t3_b[np.where(t3_b > block_ch[11:])]
    
    if len(t3_a_1) > 0 :     
        t3_a_state_1 = t3_a_1.astype(int)
    elif len(t3_a_2) > 0:
        t3_a_state_1 = t3_a_2.astype(int)

    if len(t3_b_1) > 0 :     
        t3_b_state_1 = t3_b_1.astype(int)
    elif len(t3_b_2) > 0:
        t3_b_state_1 = t3_b_2.astype(int)

    if len(t3_a_3) > 0 :     
        t3_a_state_2 = t3_a_3.astype(int)
    elif len(t3_a_4) > 0:
        t3_a_state_2 = t3_a_4.astype(int)

    if len(t3_b_3) > 0 :     
        t3_b_state_2 = t3_b_3.astype(int)
    elif len(t3_b_4) > 0:
        t3_b_state_2 = t3_b_4.astype(int)
   
    return t1_a_state_1, t1_a_state_2 ,t1_b_state_1, t1_b_state_2 ,t2_a_state_1,t2_a_state_2, t2_b_state_1,t2_b_state_2,t3_a_state_1,t3_a_state_2,\
    t3_b_state_1,t3_b_state_2

def select_trials(Data, DM, max_number_per_block, ind_time = np.arange(0,63)):
    
    all_sessions = []
    for data,dm in zip(Data, DM):
        
        trials, neurons, time = data.shape
        choices = dm[:,1]
        block = dm[:,4]
        task = dm[:,5]
        state = dm[:,0]

        data = np.mean(data[:,:,ind_time], axis = 2)

        b_pokes = dm[:,7]
        a_pokes = dm[:,6]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        
        task_1 = np.where(taskid == 1)
        task_2 = np.where(taskid == 2)
        task_3 = np.where(taskid == 3)
        
       
        correct_a = 1*choices.astype(bool) & state.astype(bool)
        choices_b  = (choices-1)*-1
        state_b  = (state-1)*-1
        correct_b = 1*choices_b.astype(bool) & state_b.astype(bool)
        correct = correct_a+correct_b
        exp_choices = ut.exp_mov_ave(correct, tau = 8, initValue = 0.5, alpha = None)
        ind_choosing_correct = np.where(exp_choices > 0.65)[0]
        
        state_change = np.where(np.diff(block)!=0)[0]+1
        state_change = np.append(state_change,0)
        state_change = np.sort(state_change)               
       
        choice_a_state_a = np.where((choices == 1) & (state == 1))[0]
        choice_b_state_b = np.where((choices == 0) & (state == 0))[0]
        if len(state_change) > 12:
            block_12_ind = state_change[12]
            state_change = state_change[:12]
        
       
        
        data = data[:block_12_ind]
        
        if len(state_change) > 11:
            
            
                
            state_1_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 0)))
            state_2_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 1)))
            state_3_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 2)))    
            state_4_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 3)))
            state_5_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 4)))
            state_6_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 5)))   
            state_7_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 6)))
            
            state_8_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 7))) 
            state_9_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 8)))
            state_10_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 9)))
            state_11_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 10)))
            state_12_correct = np.intersect1d(ind_choosing_correct, (np.where(block == 11)))
            
            change = [np.asarray([state_1_correct[0],state_2_correct[0],state_3_correct[0],state_4_correct[0],\
                                         state_5_correct[0],state_6_correct[0], state_7_correct[0],state_8_correct[0],\
                                         state_9_correct[0],state_10_correct[0], state_11_correct[0], state_12_correct[0]])][0]
            
          
            
            block_ch = np.zeros(12)
            ch = np.zeros(12)
            if task_1[0][-1] < task_2[0][-1]< task_3[0][-1]:
                block_ch[:] = state_change
                ch[:] = change
        
            elif task_1[0][-1]< task_3[0][-1] and task_3[0][-1]< task_2[0][-1]:
                block_ch[:4] = state_change[:4]
                block_ch[4:8] = state_change[8:]
                block_ch[8:12] = state_change[4:8]
                ch[:4] = change[:4]
                ch[4:8] = change[8:]
                ch[8:12] = change[4:8]

                
            elif task_3[0][-1]< task_2[0][-1] and task_2[0][-1]< task_1[0][-1]:
                block_ch[:4] = state_change[8:]
                block_ch[4:8] = state_change[4:8]
                block_ch[8:12] = state_change[:4]
                ch[:4] = change[8:]
                ch[4:8] = change[4:8]
                ch[8:12] = change[:4]
                
            elif task_3[0][-1] < task_1[0][-1] and task_3[0][-1]< task_2[0][-1] and task_1[0][-1] < task_2[0][-1]:
                block_ch[:4] = state_change[8:]
                block_ch[4:8] = state_change[:4]
                block_ch[8:12] = state_change[4:8]
                ch[:4] = change[8:]
                ch[4:8] = change[:4]
                ch[8:12] = change[4:8]
                
                
            elif task_2[0][-1]< task_3[0][-1] and task_3[0][-1]< task_1[0][-1]:
                block_ch[:4] = state_change[4:8]
                block_ch[4:8] = state_change[8:]
                block_ch[8:12] = state_change[:4]
                ch[:4] = change[4:8]
                ch[4:8] = change[8:]
                ch[8:12] = change[:4]
                
            elif task_2[0][-1]< task_1[0][-1] and task_1[0][-1]< task_3[0][-1]:
                block_ch[:4] = state_change[4:8]
                block_ch[4:8] = state_change[:4]
                block_ch[8:12] = state_change[8:]
                ch[:4] = change[4:8]
                ch[4:8] = change[:4]
                ch[8:12] = change[8:]
                
            state_change_t1 = ch[:4]
            state_change_t2 = ch[4:8]
            state_change_t3 = ch[8:]
        

            state_change_t1_1_ind,state_change_t1_2_ind, state_change_t1_3_ind,state_change_t1_4_ind,\
            state_change_t2_1_ind,state_change_t2_2_ind, state_change_t2_3_ind,state_change_t2_4_ind,\
            state_change_t3_1_ind,state_change_t3_2_ind, state_change_t3_3_ind,state_change_t3_4_ind = state_behaviour_ind(state_change_t1,state_change_t2,state_change_t3, change, data)
            
           
            t1_a_state_1, t1_a_state_2 ,t1_b_state_1, t1_b_state_2 ,t2_a_state_1,t2_a_state_2, t2_b_state_1,t2_b_state_2,t3_a_state_1,t3_a_state_2,\
            t3_b_state_1,t3_b_state_2 = choose_a_a_b_b(choice_a_state_a, choice_b_state_b,state_change_t1_1_ind,state_change_t1_2_ind, state_change_t1_3_ind,state_change_t1_4_ind,\
            state_change_t2_1_ind,state_change_t2_2_ind, state_change_t2_3_ind,state_change_t2_4_ind,\
            state_change_t3_1_ind,state_change_t3_2_ind, state_change_t3_3_ind,state_change_t3_4_ind, block, block_ch)
          
            data_t1_1 = data[t1_a_state_1,:]
            data_t1_2 = data[t1_a_state_2,:]
            data_t1_3 = data[t1_b_state_1,:]
            data_t1_4 = data[t1_b_state_2,:]
            
                
            data_t2_1 = data[t2_a_state_1,:]
            data_t2_2 = data[t2_a_state_2,:]
            data_t2_3 = data[t2_b_state_1,:]
            data_t2_4 = data[t2_b_state_2,:]

            data_t3_1 = data[t3_a_state_1,:]
            data_t3_2 = data[t3_a_state_2,:]
            data_t3_3 = data[t3_b_state_1,:]
            data_t3_4 = data[t3_b_state_2,:]
            
        
            dict_names = {'data_t1_1':data_t1_1,'data_t1_2':data_t1_2,'data_t1_3':data_t1_3,'data_t1_4':data_t1_4,\
                         'data_t2_1':data_t2_1,'data_t2_2':data_t2_2,'data_t2_3':data_t2_3,'data_t2_4':data_t2_4,\
                         'data_t3_1':data_t3_1,'data_t3_2':data_t3_2,'data_t3_3':data_t3_3,'data_t3_4':data_t3_4}
           
            all_dict = {}
            for i in dict_names.keys():
                data_dict = {i:np.full((data_t1_1.shape[1],max_number_per_block), np.nan)}
                for n in range(dict_names[i].shape[1]):
                    x = np.arange(dict_names[i][:,n].shape[0])
                    y = dict_names[i][:,n]
                    f = interpolate.interp1d(x, y)
                
                    xnew = np.arange(0, dict_names[i][:,n].shape[0]-1, (dict_names[i][:,n].shape[0]-1)/max_number_per_block)
                    ynew = f(xnew)   # use interpolation function returned by `interp1d`
                    ynew = gaussian_filter1d(ynew, 10)
                    data_dict[i][n,:] = ynew[:max_number_per_block]
            
                all_dict.update(data_dict)
            all_sessions.append(all_dict)
                
    session_list = []
    for s in all_sessions:
        neuron_list = []
        for i in dict_names.keys():
            neuron_list.append(s[i])
        session_list.append(np.asarray(neuron_list))
    session_list = np.concatenate(session_list,1)
    
    return session_list
 

def pairs(st):
    
    comb = combinations(st, 2)
    state_1 = []
    state_2 = []
    for i in comb:
        state_1.append(i[0])
        state_2.append(i[1])
        
    return state_1, state_2

def perm_test_neuron(session_list):
    
    all_sessions_choice_t = np.transpose(session_list, [1,0,2])
    all_p = [] 
    all_corr = []
    all_hist = []
    all_hist_check = []
    for neu,n in enumerate(all_sessions_choice_t):
         if all_sessions_choice_t.shape[1] == 12:
             state_1, state_2 = pairs('ABCDEFGKLMNO')
             
             dict_states =  {'A':n[0], 'B':n[1], 'C':n[2], 'D':n[3], 'E' :n[4], 'F': n[5],\
                         'G':n[6], 'K':n[7], 'L':n[8], 'M':n[9],'N':n[10],'O': n[11]}
         elif all_sessions_choice_t.shape[1] == 6:
             state_1, state_2 = pairs('ABCDEF')
             
             dict_states =  {'A':n[0], 'B':n[1], 'C':n[2], 'D':n[3], 'E' :n[4], 'F': n[5]}
         state_a  = []
         for i in state_1:
             state_a.append(dict_states[i])
         
         state_b  = []
         for i in state_2:
             state_b.append(dict_states[i])
     
         state_a =  np.asarray(state_a).flatten()
         state_b = np.asarray(state_b).flatten()
         activity_perm, p_within_state_task_1 = perm_test(state_a,state_b,1000)
#         all_hist_check.append(activity_perm_check)
         all_hist.append(activity_perm)        
         corr_real_t1 = np.corrcoef(state_a, state_b)[1,0]
         all_p.append(p_within_state_task_1)
         all_corr.append(corr_real_t1)
    return all_corr, all_p, all_hist, dict_states, all_hist_check


def decode_time_in_block(session_list, title):
    
    all_sessions_A  = np.concatenate((session_list[:2],session_list[4:6],session_list[8:10]), 0)
    all_sessions_A_fl  = np.concatenate(all_sessions_A, 1)
    
    all_sessions_B  = np.concatenate((session_list[2:4],session_list[6:8],session_list[10:]), 0)
    all_sessions_B_fl  = np.concatenate(all_sessions_B, 1)

    bins = np.arange(all_sessions_A.shape[2])
    len_bins = int(len(bins)/3)
    bins[:len_bins] = 0
    bins[len_bins:len_bins*2] = 1
    bins[len_bins*2:] = 2

    n_repeats  = int(all_sessions_A.shape[0])
    Y = np.tile(bins,n_repeats)
    
    blocks_train_A = all_sessions_A_fl[:, :int(all_sessions_A_fl.shape[1]/2)]
    blocks_train_B = all_sessions_B_fl[:, :int(all_sessions_B_fl.shape[1]/2)]

    Y_train = Y[:int(all_sessions_A_fl.shape[1]/2)]
    blocks_test_A = all_sessions_A_fl[:, int(all_sessions_A_fl.shape[1]/2):]
    blocks_test_B = all_sessions_B_fl[:, int(all_sessions_B_fl.shape[1]/2):]

    Y_test = Y[int(all_sessions_A_fl.shape[1]/2):]

    model_nb = svm.SVC(gamma='scale',class_weight='balanced')
    model_nb.fit(np.transpose(blocks_train_A),Y_train)  

    pred_class_A = model_nb.predict(np.transpose(blocks_test_A))
    cnf_matrix_A = metrics.confusion_matrix(Y_train,pred_class_A)/blocks_train_A.shape[1]
    
    pred_class_B = model_nb.predict(np.transpose(blocks_test_B))
    cnf_matrix_B_A = metrics.confusion_matrix(Y_train,pred_class_B)/blocks_train_A.shape[1]
    
    model = svm.SVC(gamma='scale',class_weight='balanced')

    model.fit(np.transpose(blocks_train_B),Y_train)  
    pred_class_A_from_B = model.predict(np.transpose(blocks_test_A))
    
    pred_class_B_B = model.predict(np.transpose(blocks_test_B))
    cnf_matrix_B_B = metrics.confusion_matrix(Y_train,pred_class_B_B)/blocks_train_A.shape[1]

    plt.figure()
    plt.imshow(cnf_matrix_A)
    plt.colorbar()

    plt.figure()
    plt.imshow(cnf_matrix_B_B)
    plt.colorbar()

    plt.figure()
    plt.imshow(cnf_matrix_B_A)
    
    plt.colorbar()
    plt.xticks([0,1,2],['start','middle','end'])
    plt.yticks([0,1,2],['start','middle','end'])
    #plt.title(title)
    correct_A_A = metrics.accuracy_score(Y_test, pred_class_A)
    correct_B_A = metrics.accuracy_score(Y_test, pred_class_B)
    correct_A_B = metrics.accuracy_score(Y_test, pred_class_A_from_B)
    correct_B_B = metrics.accuracy_score(Y_test, pred_class_B_B)

    return correct_A_A, correct_B_A, correct_A_B, correct_B_B

    
        
def plotting(all_sessions_choice,all_sessions_init):
    all_corr, all_p, all_hist, dict_states, all_hist_check = perm_test_neuron(session_list_PFC)
    neurons_above_95 = sum(np.greater(all_corr, all_p)*1)
    
    
    ind_neurns = np.greater(all_corr, all_p)
    only_ns = session_list_PFC[:,ind_neurns,:]
    tr = np.transpose(only_ns,[2,0,1])
    corr = np.corrcoef(np.concatenate(tr,0))
    plt.imshow(corr)
    plt.colorbar()
    ticks_n = np.linspace(0, corr.shape[0]-12, 12)
    ticks_label = dict_states.keys()  
    plt.xticks(ticks_n,ticks_label, rotation = 'vertical')  
    plt.yticks(ticks_n,ticks_label)
     
    # Plot histograms
    fig = plt.figure()
    activity_perm = np.asarray(all_hist)[ind_neurns]
    c = np.asarray(all_corr)[ind_neurns]

    for ii,i in enumerate(activity_perm):
        fig.add_subplot(int(neurons_above_95/2/2),int(neurons_above_95/2/2), ii+1)
        perc = np.percentile(i,95)
        plt.hist(i, color = 'red')
        ed, h = np.histogram(i)
        plt.vlines(perc, ymin = 0, ymax = max(ed))
        plt.vlines(c[ii],  ymin = 0, ymax = max(ed), color = 'grey')
        plt.tight_layout()

     
    for i in range(only_ns.shape[1]):
        plt.figure()
        plt.plot(only_ns[:,i].T)
