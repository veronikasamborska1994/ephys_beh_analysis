#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:07:43 2020

@author: veronikasamborska
"""


import numpy as np
import pylab as plt
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')

import warp_trials_number as warp
import palettable
from palettable import wesanderson as wes
import heatmap_aligned as ha
from scipy import io
import ephys_beh_import as ep
from itertools import combinations 
from matplotlib.backends.backend_pdf import PdfPages
import scipy

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
    
    return data_HP, data_PFC,experiment_aligned_PFC,experiment_aligned_HP



def hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, beh = False, block = False, raw_data = True, HP = True, start = 35, end = 42):
   
    if beh  == True:
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,\
        trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_align_beh(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    
    elif block == True:   
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,\
        trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_aligned_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    
    if raw_data == True:
         HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC  = warp.all_sessions_align_beh_raw_data(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end)
        
    if HP == True:
        exp = HP_aligned_time
        state_exp = state_list_HP
        task_exp = task_list_HP
    
    else:
        exp = PFC_aligned_time
        state_exp = state_list_PFC
        task_exp = task_list_PFC
    
    a_a_matrix_t_1_list = []
    b_b_matrix_t_1_list = []
    a_a_matrix_t_2_list = []
    b_b_matrix_t_2_list = []
    a_a_matrix_t_3_list = []
    b_b_matrix_t_3_list = []
     
    block_1_t1 = []
    block_2_t1 = []
    block_3_t1 = []
    block_4_t1 = []

    block_1_t2 = []
    block_2_t2 = []
    block_3_t2 = []
    block_4_t2 = []
 
    block_1_t3 = []
    block_2_t3 = []
    block_3_t3 = []
    block_4_t3 = []

    for s, session in enumerate(exp):
        
        state = state_exp[s]
        task = task_exp[s]
     
        a_s_t_1 = np.where((state == 1) & (task == 1))[0][0]
        a_s_t_2 = np.where((state == 1) & (task == 2))[0][0] - np.where(task == 2)[0][0]
        a_s_t_3 = np.where((state == 1) & (task == 3))[0][0] - np.where(task == 3)[0][0]
        
        # Task 1 
        if a_s_t_1 == 0:
            a_s_t1_1 = session[0]
            a_s_t1_2 = session[2]
            b_s_t1_1 = session[1]
            b_s_t1_2 = session[3]
            a_a_matrix_t_1 = np.hstack((a_s_t1_1,a_s_t1_2)) # At 13 change
            b_b_matrix_t_1 = np.hstack((b_s_t1_1, b_s_t1_2))# At 13 change 
           
           

        elif a_s_t_1 !=0 :
            a_s_t1_1 = session[1]
            a_s_t1_2 = session[3]
            b_s_t1_1 = session[0]
            b_s_t1_2 = session[2]
            a_a_matrix_t_1 = np.hstack((a_s_t1_1,a_s_t1_2)) # At 13 change
            b_b_matrix_t_1 = np.hstack((b_s_t1_1, b_s_t1_2))# At 13 change
          
        if a_s_t_2 == 0:
            
            a_s_t2_1 = session[4]
            a_s_t2_2 = session[6]
            b_s_t2_1 = session[5]
            b_s_t2_2 = session[7]
            a_a_matrix_t_2 = np.hstack((a_s_t2_1,a_s_t2_2)) # At 13 change
            b_b_matrix_t_2 = np.hstack((b_s_t2_1, b_s_t2_2))# At 13 change
           
        elif a_s_t_2 !=0:
            a_s_t2_1 = session[5]
            a_s_t2_2 = session[7]
            b_s_t2_1 = session[4]
            b_s_t2_2 = session[6]
            a_a_matrix_t_2 = np.hstack((a_s_t2_1,a_s_t2_2)) # At 13 change
            b_b_matrix_t_2 = np.hstack((b_s_t2_1, b_s_t2_2))# At 13 change
           
        if a_s_t_3 == 0:
            a_s_t3_1 = session[8]
            a_s_t3_2 = session[10]
            b_s_t3_1 = session[9]
            b_s_t3_2 = session[11]
            a_a_matrix_t_3 = np.hstack((a_s_t3_1,a_s_t3_2)) # At 13 change
            b_b_matrix_t_3 = np.hstack((b_s_t3_1, b_s_t3_2))# At 13 change
           
            
        elif a_s_t_3 != 0:
            a_s_t3_1 = session[9]
            a_s_t3_2 = session[11]
            b_s_t3_1 = session[8]
            b_s_t3_2 = session[10]
            a_a_matrix_t_3 = np.hstack((a_s_t3_1,a_s_t3_2)) # At 13 change
            b_b_matrix_t_3 = np.hstack((b_s_t3_1, b_s_t3_2))# At 13 change
        
        block_1_t1.append(session[0])
        block_2_t1.append(session[1])
        block_3_t1.append(session[2])
        block_4_t1.append(session[3])

        block_1_t2.append(session[4])
        block_2_t2.append(session[5])
        block_3_t2.append(session[6])
        block_4_t2.append(session[7])
 
        block_1_t3.append(session[8])
        block_2_t3.append(session[9])
        block_3_t3.append(session[10])
        block_4_t3.append(session[11])

        a_a_matrix_t_1_list.append(a_a_matrix_t_1)
        b_b_matrix_t_1_list.append(b_b_matrix_t_1)
        a_a_matrix_t_2_list.append(a_a_matrix_t_2) 
        b_b_matrix_t_2_list.append(b_b_matrix_t_2)
        a_a_matrix_t_3_list.append(a_a_matrix_t_3)
        b_b_matrix_t_3_list.append(b_b_matrix_t_3)
        
    # A and B in each task  
    a_a_matrix_t_1_list= np.concatenate(a_a_matrix_t_1_list,0)
    b_b_matrix_t_1_list= np.concatenate(b_b_matrix_t_1_list,0)
    a_a_matrix_t_2_list= np.concatenate(a_a_matrix_t_2_list,0)
    b_b_matrix_t_2_list= np.concatenate(b_b_matrix_t_2_list,0)
    a_a_matrix_t_3_list= np.concatenate(a_a_matrix_t_3_list,0)
    b_b_matrix_t_3_list= np.concatenate(b_b_matrix_t_3_list,0)
    
    block_1_t1 = np.concatenate(block_1_t1,0)
    block_2_t1 = np.concatenate(block_2_t1,0)
    block_3_t1 = np.concatenate(block_3_t1,0)
    block_4_t1 =np.concatenate(block_4_t1,0)

    block_1_t2 = np.concatenate(block_1_t2,0)
    block_2_t2 = np.concatenate(block_2_t2,0)
    block_3_t2 = np.concatenate(block_3_t2,0)
    block_4_t2 = np.concatenate(block_4_t2,0)
 
    block_1_t3 = np.concatenate(block_1_t3,0)
    block_2_t3 = np.concatenate(block_2_t3,0)
    block_3_t3 = np.concatenate(block_3_t3,0)
    block_4_t3 = np.concatenate(block_4_t3,0)

     
    
    return  a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
            a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
            a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
            block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
            block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
            block_1_t3, block_2_t3, block_3_t3, block_4_t3
                
            
def plot():
    ind_above_chance_HP = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, perm = True)
    
    ind_above_chance_PFC = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = False, start = 0, end = 62, perm = True)
   
    
  
def perm_time_in_block(data_HP, data_PFC,experiment_aligned_HP, \
                       experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, perm = True):
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
    switch = int(a_a_matrix_t_1_list.shape[1]/2)
    distance_mean_neuron = []
    perm_mean = []
    
    for i,aa in enumerate(a_a_matrix_t_1_list):
        
        a_1 = a_a_matrix_t_1_list[i][:switch] - np.mean(a_a_matrix_t_1_list[i][:switch])
        a_2 = a_a_matrix_t_1_list[i][switch:] - np.mean(a_a_matrix_t_1_list[i][switch:])
        a_3 = a_a_matrix_t_2_list[i][:switch] - np.mean(a_a_matrix_t_2_list[i][:switch])
        a_4 = a_a_matrix_t_2_list[i][switch:] - np.mean(a_a_matrix_t_2_list[i][switch:])
        a_5 = a_a_matrix_t_3_list[i][:switch] - np.mean(a_a_matrix_t_3_list[i][:switch])
        a_6 = a_a_matrix_t_3_list[i][switch:] - np.mean(a_a_matrix_t_3_list[i][switch:])
      
        b_1 = b_b_matrix_t_1_list[i][:switch] - np.mean(b_b_matrix_t_1_list[i][:switch])
        b_2 = b_b_matrix_t_1_list[i][switch:] - np.mean(b_b_matrix_t_1_list[i][switch:])
        b_3 = b_b_matrix_t_2_list[i][:switch] - np.mean(b_b_matrix_t_2_list[i][switch:])
        b_4 = a_a_matrix_t_2_list[i][switch:] - np.mean(b_b_matrix_t_2_list[i][:switch])
        b_5 = b_b_matrix_t_3_list[i][:switch] - np.mean(b_b_matrix_t_3_list[i][:switch])
        b_6 = b_b_matrix_t_3_list[i][switch:] - np.mean(b_b_matrix_t_3_list[i][switch:])
      
        
        stack_for_perms = np.vstack((a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6))

                        
        blocks_std_a = np.std([a_1,a_2, a_3, a_4, a_5, a_6],0)/(np.sqrt(6))
                
        blocks_std_b = np.std([b_1,b_2, b_3, b_4, b_5,b_6],0)/(np.sqrt(6))
       
        distance = np.sum(np.sqrt(abs(np.mean(stack_for_perms[:6],0)-np.mean(stack_for_perms[6:],0))**2/(blocks_std_a**2 +blocks_std_b**2)))
        distance_mean_neuron.append(distance)
    
        
        m, n = [6,6]
        # more_extreme = np.zeros(len(predictors))

        #diff_perm = np.zeros(int(num_rounds))
        diff_perm = []

        if perm:
            nn = 0
            for ind_a in combinations(range(m + n), m):
                ind_b = [i for i in range(m + n) if i not in ind_a]
                a_s_perm = stack_for_perms[np.array(ind_a)]
                b_s_perm = stack_for_perms[np.array(ind_b)]
                   
                blocks_std_a_perm = np.std(a_s_perm,0)/(np.sqrt(6))
                blocks_std_b_perm= np.std(b_s_perm,0)/(np.sqrt(6))
                distance =np.sum(np.sqrt(abs(np.mean(a_s_perm,0)-np.mean(b_s_perm,0))**2/(blocks_std_a_perm**2 +blocks_std_b_perm**2)))
                diff_perm.append(distance)
                nn+=1
                    
        perm_mean.append(np.percentile(diff_perm,95))
    grand_bud = wes.GrandBudapest1_4.mpl_colors
    plt.ion()
    plt.figure()
    
    his_perm, b_p = np.histogram(perm_mean,100)
    hist_mean_neuron,b = np.histogram(distance_mean_neuron,100)
    plt.bar(b[:-1], hist_mean_neuron, width = 1, color = grand_bud[1],alpha = 0.5, label = 'Neurons')
    plt.bar(b_p[:-1], his_perm, width = 1, color = grand_bud[0], alpha = 0.5, label = 'Permutation')
    plt.legend()
    ind_above_chance = np.where(np.array(distance_mean_neuron) > np.array(perm_mean))[0]
    #plt.hist(perm_mean, color = grand_bud[0], alpha = 0.5, label = 'Permutation')
    #plt.hist(distance_mean_neuron, color = grand_bud[1],alpha = 0.5, label = 'Neurons')

    plt.legend()
    if HP == True:
        plt.title('HP')
    elif HP == False:
        plt.title('PFC')
    return  ind_above_chance
        
        
def perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, perm = True):
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
    distance_mean_neuron = []
    perm_mean = []
    neurons = 0
    for i,aa in enumerate(a_a_matrix_t_1_list):
        switch = int(a_a_matrix_t_1_list.shape[1]/2)

        neurons+=1
        
        a_1 = a_a_matrix_t_1_list[i][:switch] 
        a_2 = a_a_matrix_t_1_list[i][switch:] 
        a_3 = a_a_matrix_t_2_list[i][:switch] 
        a_4 = a_a_matrix_t_2_list[i][switch:]
        a_5 = a_a_matrix_t_3_list[i][:switch] 
        a_6 = a_a_matrix_t_3_list[i][switch:]
      
        b_1 = b_b_matrix_t_1_list[i][:switch] 
        b_2 = b_b_matrix_t_1_list[i][switch:] 
        b_3 = b_b_matrix_t_2_list[i][:switch] 
        b_4 = a_a_matrix_t_2_list[i][switch:] 
        b_5 = b_b_matrix_t_3_list[i][:switch]
        b_6 = b_b_matrix_t_3_list[i][switch:] 
      
        if raw_data == True:
                
            a_1 = (a_1 - np.mean(a_1))
            a_2 = (a_2 - np.mean(a_2))
            a_3 = (a_3 - np.mean(a_3))
            a_4 = (a_4 - np.mean(a_4))
            a_5 = (a_5 - np.mean(a_5))
            a_6 = (a_6 - np.mean(a_6))

            b_1 = (b_1 - np.mean(b_1))
            b_2 = (b_2 - np.mean(b_2))
            b_3 = (b_3 - np.mean(b_3))
            b_4 = (b_4 - np.mean(b_4))
            b_5 = (b_5 - np.mean(b_5))
            b_6 = (b_6 - np.mean(b_6))

        switch = len(a_1)

        blocks_all_tasks =  np.mean([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)
        peak = np.max(blocks_all_tasks)
        troph = np.min(blocks_all_tasks)

        std_blocks_all_tasks =  np.std([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)/(np.sqrt(12))

        
        #distance = np.sqrt((abs(peak-tropclh)**2)/(np.mean(std_blocks_all_tasks))**2)
        distance = abs(peak-troph)/np.max(std_blocks_all_tasks)

        distance_mean_neuron.append(distance)
    
        diff_perm = []
        if perm:
            for p in range(perm):
                perm_array_rolled = np.hstack((a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6))
                np.random.shuffle(perm_array_rolled)

                a_1_perm = perm_array_rolled[:switch]
                a_2_perm = perm_array_rolled[switch:switch*2]
                a_3_perm = perm_array_rolled[switch*2:switch*3]
                a_4_perm = perm_array_rolled[switch*3:switch*4]
                a_5_perm = perm_array_rolled[switch*4:switch*5]
                a_6_perm = perm_array_rolled[switch*5:switch*6]
              
                b_1_perm = perm_array_rolled[switch*6:switch*7]
                b_2_perm = perm_array_rolled[switch*7:switch*8]
                b_3_perm = perm_array_rolled[switch*8:switch*9]
                b_4_perm = perm_array_rolled[switch*9:switch*10]
                b_5_perm = perm_array_rolled[switch*10:switch*11]
                b_6_perm = perm_array_rolled[switch*11:]
              
                blocks_all_tasks_perm =  np.mean([a_1_perm,a_2_perm,a_3_perm,a_4_perm,a_5_perm,a_6_perm,b_1_perm,b_2_perm,b_3_perm,b_4_perm,b_5_perm,b_6_perm],0)
                std_blocks_all_tasks_perm =  np.std([a_1_perm,a_2_perm,a_3_perm,a_4_perm,a_5_perm,a_6_perm,b_1_perm,b_2_perm,b_3_perm,b_4_perm,b_5_perm,b_6_perm],0)/(np.sqrt(12))


                peak_perm = np.max(blocks_all_tasks_perm)
                troph_perm = np.min(blocks_all_tasks_perm)
                
                distance_perm = abs(peak_perm-troph_perm)/np.max(std_blocks_all_tasks_perm)

                #distance_perm = np.sqrt((abs(peak_perm-troph_perm)**2)/(np.mean(std_blocks_all_tasks_perm))**2)
                diff_perm.append(distance_perm)
                    
        perm_mean.append(np.percentile(diff_perm,95))
   
    grand_bud = wes.GrandBudapest1_4.mpl_colors
    plt.ion()
    plt.figure()
    
    his_perm, b_p = np.histogram(perm_mean,1000)
    hist_mean_neuron,b = np.histogram(distance_mean_neuron,1000)
   
    plt.bar(b[:-1], hist_mean_neuron, width = 0.05, color = grand_bud[0],alpha = 0.5, label = 'Neurons')
    plt.bar(b_p[:-1], his_perm, width = 0.05, color = grand_bud[1], alpha = 0.5, label = 'Permutation')
    plt.legend()
    ind_above_chance = np.where(np.array(distance_mean_neuron) > np.array(perm_mean))[0]
    percentage = (len(ind_above_chance)/neurons)*100
    #plt.hist(perm_mean, color = grand_bud[0], alpha = 0.5, label = 'Permutation')
    #plt.hist(distance_mean_neuron, color = grand_bud[1],alpha = 0.5, label = 'Neurons')

    plt.legend()
    if HP == True:
        plt.title('HP')
    elif HP == False:
        plt.title('PFC')
    return ind_above_chance,percentage


def percent():
    
    ind_above_chance_HP_preinit,percentage_HP_preinit  = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20, perm = 1000)  
    
    ind_above_chance_PFC_preinit,percentage_PFC_preinit  = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20, perm = 1000)  
   
    ind_above_chance_HP_choice,percentage_HP_choice  = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 35, end = 42, perm = 1000)  
   
    ind_above_chance_PFC_choice,percentage_PFC_choice  = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = False, start = 35, end = 42, perm = 1000)  
   
    ind_above_chance_HP_rew,percentage_HP_rew = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 42, end = 62, perm = 1000)  
   
    ind_above_chance_PFC_rew,percentage_PFC_rew  = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = False, start = 42, end = 62, perm = 1000)  
   
    ind_above_chance_HP_rew,percentage_HP_init = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 25, end = 35, perm = 1000)  
   
    ind_above_chance_PFC_rew,percentage_PFC_init  = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = False, start = 25, end = 35, perm = 1000)  
      

     
##Block in task code -->cell that fires in First/Second/Third etc block in each task (irrespective of A or B)
            
def plot_task_hierarchy(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, region = 'HP', plot = True):

     a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
   
    
     if HP == True:
         a_list,  b_list, rew_list,  no_rew_list = find_rewards_choices(data_HP, experiment_aligned_HP)
     elif HP == False:
         a_list,  b_list, rew_list,  no_rew_list = find_rewards_choices(data_PFC, experiment_aligned_PFC)
                               
     isl = wes.FantasticFox2_5.mpl_colors
     plt.ioff()

     blocks_t1 = np.hstack((block_1_t1,block_2_t1,block_3_t1, block_4_t1))
     blocks_t2 = np.hstack((block_1_t2,block_2_t2,block_3_t2, block_4_t2))
     blocks_t3 = np.hstack((block_1_t3,block_2_t3,block_3_t3, block_4_t3))
     stack_all_blocks = np.concatenate((blocks_t1,blocks_t2,blocks_t3),1)
     c = palettable.scientific.sequential.LaPaz_5.mpl_colormap
     switch = block_1_t1.shape[1]

     ticks = [0,switch,switch*2,switch*3,switch*4,switch*5,switch*6,switch*7,switch*8,switch*9,switch*10,switch*11]
     tick_n = ['1 T1 ', '2 T1 ', '3 T1', '4 T1', '1 T2 ', '2 T2 ', '3 T2', '4 T2',\
               ' 1 T3 ', ' 2 T3 ', '3 T3', '4 T3']
  
     corr = np.corrcoef(stack_all_blocks.T)
     plt.figure()

     plt.imshow(corr, cmap = c)
     plt.colorbar()
     plt.xticks(ticks, tick_n, rotation = 90)
     plt.yticks(ticks, tick_n)
     isl_1 =  wes.Moonrise1_5.mpl_colors
     grand_bud = wes.GrandBudapest1_4.mpl_colors

     if plot == True:
         pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ region +'time_in_block.pdf')
         plt.ioff()
         count = 0
         plot_new = True
       
         for i,m in enumerate(blocks_t1): 
             
            count +=1
            if count == 7:
                plot_new = True
                count = 1
            if plot_new == True:
                pdf.savefig()      
                plt.clf()
                plt.figure()
                plot_new = False
            plt.subplot(3,4, count)
           
            blocks_mean = np.mean([blocks_t1[i], blocks_t2[i], blocks_t3[i]],0)
            blocks_std = np.std([blocks_t1[i], blocks_t2[i], blocks_t3[i]],0)/(np.sqrt(4))
            
            plt.plot(blocks_mean, color = grand_bud[0], label = 'Task Av')
            plt.fill_between(np.arange(len(blocks_mean)), blocks_mean-blocks_std, blocks_mean+blocks_std, alpha=0.2, color = grand_bud[0])
        
            #plt.plot(blocks_t1[i], color = isl_1[1], label = 'Task 1')
            #plt.plot(blocks_t2[i], color = isl_1[2], label = 'Task 2')
            #plt.plot(blocks_t3[i], color = isl_1[4], label = 'Task 3')

            plt.vlines([switch,switch*2,switch*3], np.min(blocks_mean), np.max(blocks_mean), linestyle = ':', color = 'grey', label = 'Switch')
            
            plt.title(str(count))
            plt.tight_layout()
            
            if count == 1:
                plt.legend()
                
            plt.subplot(3,4, count+6)
            plt.plot(a_list[i], color = isl[1], label = 'A')
            plt.plot(b_list[i], color = isl[2], label = 'B')
            plt.plot(rew_list[i], color = isl[3], label = 'Reward')
            plt.plot(no_rew_list[i], color = isl[4], label = 'No Rew')
            plt.title(str(count))
            plt.vlines([25,36,43], np.min([np.min(a_list[i]),np.min(b_list[i]),np.min(rew_list[i]),np.min(no_rew_list[i])]),\
                                        np.max([np.max(a_list[i]),np.max(b_list[i]),np.max(rew_list[i]),np.max(no_rew_list[i])]),linestyle= '--', color = 'pink')

            if count == 1:
                plt.legend()
         pdf.savefig()      
         pdf.close()
         
       
def find_rewards_choices(data, experiment_aligned_data):

    res_list, list_block_changes, trials_since_block_list, state_list,task_list,reward_list,choice_list = \
    warp.raw_data_time_warp_beh(data, experiment_aligned_data)
    a_list = []
    b_list = []
    rew_list = []
    no_rew_list = []

    for s,session in enumerate(res_list):
        rewards = reward_list[s]
        rew = np.where(rewards == 1)[0]
        no_rew = np.where(rewards == 0)[0]

        choices = choice_list[s]
        a = np.where(choices == 1)[0]
        b = np.where(choices == 0)[0]

        rewarded = np.mean(session[rew],0)
        not_rewarded = np.mean(session[no_rew],0)
        
        a = np.mean(session[a],0)
        b = np.mean(session[b],0)
       
        a_list.append(a)
        b_list.append(b)
        rew_list.append(rewarded)
        no_rew_list.append(not_rewarded)
    
    a_list = np.concatenate(a_list,0)
    b_list = np.concatenate(b_list,0)
    rew_list = np.concatenate(rew_list,0)
    no_rew_list = np.concatenate(no_rew_list,0)
    
    return  a_list,  b_list, rew_list,  no_rew_list

         
def plot():
  
    # A vs B Task
    ind_above_chance_a_b_spec_HP = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20,\
                        region = 'HP A B all Tasks Init Raw', plot = True,\
                        a_b_in_task = True, plot_block_in_task = False, a_b_all_tasks = False,plot_block_all_tasks = False, perm = 1000, plot_all = False)      
 
    ind_above_chance_a_b_spec_PFC = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20,\
                        region = 'PFC A B all Tasks Init Raw', plot = True,\
                        a_b_in_task = True, plot_block_in_task = False, a_b_all_tasks = False, plot_block_all_tasks = False, perm = 1000, plot_all = False)      
 
    # A vs B Av
    ind_above_chance_a_b_HP_ch, per_HP = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62,\
                        region = 'HP A B Ave Check A B', plot = True,\
                        a_b_in_task = False, plot_block_in_task = False, a_b_all_tasks = True, plot_block_all_tasks = False,perm  = 1000, plot_all = False)      
 
    ind_above_chance_a_b_PFC_ch, per_PFC = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = False, HP = False, start = 0, end = 62,\
                        region = 'PFC A B Ave Check A B', plot = True,\
                        a_b_in_task = False, plot_block_in_task = False, a_b_all_tasks = True, plot_block_all_tasks = False, perm = 1000,plot_all = False)      
 
    # Task 1,2,3
    ind_above_chance_tasks_HP =  plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20,\
                        region = 'HP Tasks Init Raw', plot = True,\
                        a_b_in_task = False, plot_block_in_task = True, a_b_all_tasks = False, plot_block_all_tasks = False)      
 
    ind_above_chance_tasks_PFC = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
                        region = 'PFC Tasks Init Raw', plot = True,\
                        a_b_in_task = False, plot_block_in_task = True, a_b_all_tasks = False, plot_block_all_tasks = False)      
    
    # All blocks 
    ind_above_chance_blocks_HP_non_demean_big_ed,neuron_count_HP = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True,start = 0, end = 20,\
                        region = 'HP Block all Tasks Init Roll', plot = True,\
                        a_b_in_task = False, plot_block_in_task = False, a_b_all_tasks = False,plot_block_all_tasks = True,perm = 000, plot_all = False)      
 
    ind_above_chance_blocks_PFC_non_demean_big_ed, neuron_count_PFC = plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20,\
                        region = 'PFC Block all Tasks Init Roll', plot = True,\
                        a_b_in_task = False, plot_block_in_task = False, a_b_all_tasks = False, plot_block_all_tasks = True,perm = 1000,plot_all = False)      
 
    
def create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = False, start = 0, end = 62, title = 'PFC aligned'):
    
     a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1, block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
         
     A_block_1_1 = a_a_matrix_t_1_list[:,:int(a_a_matrix_t_1_list.shape[1]/2)]
     A_block_1_2 = a_a_matrix_t_1_list[:,int(a_a_matrix_t_1_list.shape[1]/2):]

     A_block_2_1 = a_a_matrix_t_2_list[:,:int(a_a_matrix_t_1_list.shape[1]/2)]
     A_block_2_2 = a_a_matrix_t_2_list[:,int(a_a_matrix_t_1_list.shape[1]/2):]

     A_block_3_1 = a_a_matrix_t_3_list[:,:int(a_a_matrix_t_1_list.shape[1]/2)]
     A_block_3_2 = a_a_matrix_t_3_list[:,int(a_a_matrix_t_1_list.shape[1]/2):]

     B_block_1_1 = b_b_matrix_t_1_list[:,:int(a_a_matrix_t_1_list.shape[1]/2)]
     B_block_1_2 = b_b_matrix_t_1_list[:,int(a_a_matrix_t_1_list.shape[1]/2):]

     B_block_2_1 = b_b_matrix_t_2_list[:,:int(a_a_matrix_t_1_list.shape[1]/2)]
     B_block_2_2 = b_b_matrix_t_2_list[:,int(a_a_matrix_t_1_list.shape[1]/2):]

     B_block_3_1 = b_b_matrix_t_3_list[:,:int(a_a_matrix_t_1_list.shape[1]/2)]
     B_block_3_2 = b_b_matrix_t_3_list[:,int(a_a_matrix_t_1_list.shape[1]/2):]
     
     all_blocks = [A_block_1_1,A_block_1_2,A_block_2_1,A_block_2_2,A_block_3_1, A_block_3_2,\
                   B_block_1_1,B_block_1_2,B_block_2_1,B_block_2_2,B_block_3_1, B_block_3_2]

     scipy.io.savemat('/Users/veronikasamborska/Desktop/code_for_data_session/'+ title + '.mat',{'A_Task_1_Block_1': A_block_1_1,
                                                                          'A_Task_1_Block_2': A_block_1_2,
                                                                           'A_Task_2_Block_1': A_block_2_1,
                                                                           'A_Task_2_Block_2': A_block_2_2,
                                                                           'A_Task_3_Block_1': A_block_3_1,
                                                                           'A_Task_3_Block_2': A_block_3_2,                                                                         
                                                                           'B_Task_1_Block_1': B_block_1_1,
                                                                           'B_Task_1_Block_2': B_block_1_2,
                                                                           'B_Task_2_Block_1': B_block_2_1,
                                                                           'B_Task_2_Block_2': B_block_2_2,
                                                                           'B_Task_3_Block_1': B_block_3_1,
                                                                           'B_Task_3_Block_2': B_block_3_2,
                                                                             })


def save_files():
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                 block = False, raw_data = False, HP = False, start = 0, end = 62, title = 'PFC_aligned_on_beh_time_all_trial_time_residuals')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                block = False, raw_data = False, HP = True, start = 0, end = 62, title = 'HP_aligned_on_beh_time_all_trial_time_residuals')  
    
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                 block = False, raw_data = False, HP = False, start = 0, end = 20, title = 'PFC_aligned_on_beh_time_pre_init_time_residuals')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                block = False, raw_data = False, HP = True, start = 0, end = 20, title = 'HP_aligned_on_beh_time_pre_init_time_residuals')
  
    
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                 block = False, raw_data = True, HP = False, start = 0, end = 62, title = 'PFC_aligned_on_beh_time_all_trial_time_raw_firing')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                block = False, raw_data = True, HP = True, start = 0, end = 62, title = 'HP_aligned_on_beh_time_all_trial_time_raw_firing')  
    
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                 block = False, raw_data = True, HP = False, start = 0, end = 20, title = 'PFC_aligned_on_beh_time_pre_init_time_raw_firing')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = True,
                block = False, raw_data = True, HP = True, start = 0, end = 20, title = 'HP_aligned_on_beh_time_pre_init_time_raw_firing')
  
 
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                 block = True, raw_data = False, HP = False, start = 0, end = 62, title = 'PFC_aligned_on_block_change_all_trial_time_residuals')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                block = True, raw_data = False, HP = True, start = 0, end = 62, title = 'HP_aligned_on_block_change_all_trial_time_residuals')  
    
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                 block = True, raw_data = False, HP = False, start = 0, end = 20, title = 'PFC_aligned_on_block_change_pre_init_time_residuals')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                block = True, raw_data = False, HP = True, start = 0, end = 20, title = 'HP_aligned_on_block_change_pre_init_time_residuals')
  
    
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                 block = True, raw_data = True, HP = False, start = 0, end = 62, title = 'PFC_aligned_on_block_change_all_trial_time_raw_firing')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                block = True, raw_data = True, HP = True, start = 0, end = 62, title = 'HP_aligned_on_block_change_all_trial_time_raw_firing')  
    
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                 block = True, raw_data = True, HP = False, start = 0, end = 20, title = 'PFC_aligned_on_block_change_pre_init_time_raw_firing')
   
    create_array(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, beh = False,
                block = True, raw_data = True, HP = True, start = 0, end = 20, title = 'HP_aligned_on_block_change_pre_init_time_raw_firing')
  
    
## A/B Block Number in Task 

def plot_task_hierarchy_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = False, HP = True,start = 0, end = 25,\
                        region = 'HP', plot = True,\
                        a_b_in_task = False, plot_block_in_task = True, a_b_all_tasks = False, plot_block_all_tasks = False, perm = True,\
                            plot_all = False):

     a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
   
    
     if HP == True:
         a_list,  b_list, rew_list,  no_rew_list = find_rewards_choices(data_HP, experiment_aligned_HP)
         if plot_block_all_tasks == False:
             ind_above_chance = perm_time_in_block(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)
         elif  plot_block_all_tasks ==  True:
            ind_above_chance,per = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)
          
     elif HP == False:
        a_list,  b_list, rew_list,  no_rew_list = find_rewards_choices(data_PFC, experiment_aligned_PFC)
        if plot_block_all_tasks == False:
            ind_above_chance = perm_time_in_block(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)
        elif  plot_block_all_tasks ==  True:
           ind_above_chance, per = perm_test_time(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)
          
                       
     # a_blocks_1_2 = np.hstack((a_a_matrix_t_1_list,a_a_matrix_t_2_list, a_a_matrix_t_3_list))
     # b_blocks_1_2 =  np.hstack((b_b_matrix_t_1_list,b_b_matrix_t_2_list, b_b_matrix_t_3_list))
     # stack_all_blocks = np.concatenate((a_blocks_1_2,b_blocks_1_2),1)
     # c = palettable.scientific.sequential.LaPaz_5.mpl_colormap

     # np.arange(a_a_matrix_t_1_list.shape[1])
     # corr = np.corrcoef(stack_all_blocks.T)
     # plt.ion()
     # plt.figure()
     # swtich = a_a_matrix_t_1_list.shape[1]/2
     # ticks = [0,swtich,swtich*2,swtich*3,swtich*4,swtich*5,swtich*6,swtich*7,swtich*8,swtich*9,swtich*10,swtich*11]
     # tick_n = [' A 1 T1 ', ' A 2 T1 ', 'B 1 T1', 'B 2 T1', ' A 1 T2 ', ' A 2 T2 ', 'B 1 T2', 'B 2 T2',\
     #           ' A 1 T3 ', ' A 2 T3 ', 'B 1 T3', 'B 2 T3']
     
     # # plt.ioff()
     # plt.imshow(corr, cmap = c)
     # plt.xticks(ticks, tick_n, rotation = 90)
     # plt.yticks(ticks, tick_n)

     # plt.colorbar()
     switch = int(a_a_matrix_t_1_list.shape[1]/2)
     
     a_s_1 = np.mean([a_a_matrix_t_1_list[:,:switch],a_a_matrix_t_1_list[:,switch:]],0)
     b_s_1 = np.mean([b_b_matrix_t_1_list[:,:switch],b_b_matrix_t_1_list[:,switch:]],0)
     a_s_1_1 = a_a_matrix_t_1_list[:,:switch] 
     a_s_1_2 = a_a_matrix_t_1_list[:,switch:]
     b_s_1_1 = b_b_matrix_t_1_list[:,:switch]
     b_s_1_2 = b_b_matrix_t_1_list[:,switch:]
     
   
     a_s_2 = np.mean([a_a_matrix_t_2_list[:,:switch],a_a_matrix_t_2_list[:,switch:]],0)
     b_s_2 = np.mean([b_b_matrix_t_2_list[:,:switch],b_b_matrix_t_2_list[:,switch:]],0)
     a_s_2_1 = a_a_matrix_t_2_list[:,:switch]
     a_s_2_2 = a_a_matrix_t_2_list[:,switch:]
     b_s_2_1 = b_b_matrix_t_2_list[:,:switch]
     b_s_2_2 = b_b_matrix_t_2_list[:,switch:]
  
     a_s_3 = np.mean([a_a_matrix_t_3_list[:,:switch],a_a_matrix_t_3_list[:,switch:]],0)
     b_s_3 = np.mean([b_b_matrix_t_3_list[:,:switch],b_b_matrix_t_3_list[:,switch:]],0)
     
     a_s_3_1 = a_a_matrix_t_3_list[:,:switch]
     a_s_3_2 = a_a_matrix_t_3_list[:,switch:]
     b_s_3_1 = b_b_matrix_t_3_list[:,:switch]
     b_s_3_2 = b_b_matrix_t_3_list[:,switch:]
     
     # if raw_data == True:
     #     a_s_1_1 = a_s_1_1 - np.mean(a_s_1_1,1)
     #     a_s_1_2 = a_s_1_2 - np.mean(a_s_1_2,1)
     #     a_s_1 = np.mean([a_s_1_1,a_s_1_2],0)

     #     b_s_1_1 = b_s_1_1 - np.mean(b_s_1_1,1)
     #     b_s_1_2 = b_s_1_2 - np.mean(b_s_1_2,1)
     #     b_s_1 = np.mean([b_s_1_1,b_s_1_2],0)

     #     a_s_2_1 = a_s_2_1 - np.mean(a_s_2_1,1)
     #     a_s_2_2 = a_s_2_2 - np.mean(a_s_2_2,1)
     #     a_s_2 = np.mean([a_s_2_1,a_s_2_2],0)

     #     b_s_2_1 = b_s_2_1 - np.mean(b_s_2_1,1)
     #     b_s_2_2 = b_s_2_2 - np.mean(b_s_2_2,1)
     #     b_s_2 = np.mean([b_s_2_1,b_s_2_2],0)

     #     a_s_3_1 = a_s_3_1 - np.mean(a_s_3_1,1)
     #     a_s_3_2 = a_s_3_2 - np.mean(a_s_3_2,1)
     #     a_s_3 = np.mean([a_s_3_1,a_s_3_2],0)

     #     b_s_3_1 = b_s_3_1 - np.mean(b_s_3_1,1)
     #     b_s_3_2 = b_s_3_2 - np.mean(b_s_3_2,1)
     #     b_s_3 = np.mean([b_s_3_1,b_s_3_2],0)

     #     stack_all_blocks = np.concatenate((a_s_1,a_s_2, a_s_3, b_s_1, b_s_2,b_s_3),1)
   
     stack_all_blocks = np.concatenate((a_s_1,a_s_2, a_s_3, b_s_1, b_s_2,b_s_3),1)
     c = palettable.scientific.sequential.LaPaz_5.mpl_colormap

     corr = np.corrcoef(stack_all_blocks.T)
     plt.figure()
     swtich = a_s_1.shape[1]
     ticks = [0,swtich,swtich*2,swtich*3,swtich*4, swtich*5]
     tick_n = [' A 1 ', ' A 2 ', 'A 3', 'B 1 ', ' B 2 ', ' B 3']
     
     plt.ion()
     plt.figure()
     plt.imshow(corr, cmap = c)
     plt.xticks(ticks, tick_n, rotation = 90)
     plt.yticks(ticks, tick_n)

     
     isl_1 =  wes.Moonrise1_5.mpl_colors
     isl  = wes.Royal3_5.mpl_colors
     if plot == True:
         pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ region +'A vs B block 1 vs 2.pdf')
         plt.ioff()
         count = 0
         plot_new = True
       
         #plt.savefig('/Users/veronikasamborska/Desktop/time/'+ region +'corr_time_in_block.pdf')
         switch = int(a_a_matrix_t_1_list.shape[1]/2)
         neuron_count = 0
         for i,m in enumerate(a_a_matrix_t_1_list): 
            count +=1
            neuron_count += 1
            if count == 7:
                plot_new = True
                count = 1
            if plot_new == True:
                pdf.savefig()      
                plt.clf()
                plt.figure()
                plot_new = False
            plt.subplot(3,4, count)
           
            #blocks_mean_b = np.mean([b_b_matrix_t_1_list[i],b_b_matrix_t_2_list[i], b_b_matrix_t_3_list[i]],0)
            #blocks_std_b = np.std([b_b_matrix_t_1_list[i],b_b_matrix_t_2_list[i], b_b_matrix_t_3_list[i]],0)/(np.sqrt(3))
            
            #blocks_mean_a = np.mean([a_a_matrix_t_1_list[i],a_a_matrix_t_2_list[i], a_a_matrix_t_3_list[i]],0)
           # blocks_std_a = np.std([a_a_matrix_t_1_list[i],a_a_matrix_t_2_list[i], a_a_matrix_t_3_list[i]],0)/(np.sqrt(3))
            grand_bud = wes.GrandBudapest1_4.mpl_colors
            grand_bud_1 = wes.GrandBudapest2_4.mpl_colors
            mend = wes.Mendl_4.mpl_colors
            if raw_data == True:
                
                a_1 = (a_s_1_1[i]- np.mean(a_s_1_1[i]))
                a_2 = (a_s_1_2[i]- np.mean(a_s_1_2[i]))
                a_3 = (a_s_2_1[i]- np.mean(a_s_2_1[i]))
                a_4 = (a_s_2_2[i]- np.mean(a_s_2_2[i]))
                a_5 = (a_s_3_1[i]- np.mean(a_s_3_1[i]))
                a_6 = (a_s_3_2[i]- np.mean(a_s_3_2[i]))

                b_1 = (b_s_1_1[i]- np.mean(b_s_1_1[i]))
                b_2 = (b_s_1_2[i]- np.mean(b_s_1_2[i]))
                b_3 = (b_s_2_1[i]- np.mean(b_s_2_1[i]))
                b_4 = (b_s_2_2[i]- np.mean(b_s_2_2[i]))
                b_5 = (b_s_3_1[i]- np.mean(b_s_3_1[i]))
                b_6 = (b_s_3_2[i]- np.mean(b_s_3_2[i]))
            else:
                    
                a_1 = a_s_1_1[i]#- np.mean(a_s_1_1[i])
                a_2 = a_s_1_2[i]#- np.mean(a_s_1_2[i])
                a_3 = a_s_2_1[i]#- np.mean(a_s_2_1[i])
                a_4 = a_s_2_2[i]#- np.mean(a_s_2_2[i])
                a_5 = a_s_3_1[i]#- np.mean(a_s_3_1[i])
                a_6 = a_s_3_2[i]#- np.mean(a_s_3_2[i])
    
                b_1 = b_s_1_1[i]#- np.mean(b_s_1_1[i])
                b_2 = b_s_1_2[i]#- np.mean(b_s_1_2[i])
                b_3 = b_s_2_1[i]#- np.mean(b_s_2_1[i])
                b_4 = b_s_2_2[i]#- np.mean(b_s_2_2[i])
                b_5 = b_s_3_1[i]#- np.mean(b_s_3_1[i])
                b_6 = b_s_3_2[i]#- np.mean(b_s_3_2[i])

            if a_b_in_task == True: 
                
                plt.plot(a_1, color = grand_bud[1], label = 'A 1')
                           
                plt.plot(a_2, color = grand_bud_1[0], label = 'A 2')
    
                plt.plot(a_3, color = isl[1], label = 'A 3')
    
                plt.plot(a_4, color = isl_1[0], label = 'B 1')
    
                plt.plot(a_5, color = mend[1], label = 'B 2')
    
                plt.plot(a_6, color = mend[3], label = 'B 3')
            
            if plot_block_in_task == True:
            
                blocks_mean_task_1 =  np.mean([a_1,a_2, b_1,b_2],0)
                blocks_std_task_1 = np.std([a_1,a_2, b_1,b_2],0)/(np.sqrt(4))

                blocks_mean_task_2 =  np.mean([a_3,a_4, b_3,b_4],0)
                blocks_std_task_2 = np.std([a_3,a_4, b_3,b_4],0)/(np.sqrt(4))

                blocks_mean_task_3 =  np.mean([a_5,a_6, b_5,b_6],0)
                blocks_std_task_3 = np.std([a_5,a_6, b_5,b_6],0)/(np.sqrt(4))
 
                plt.plot(blocks_mean_task_1, color = isl_1[0], label = 'Task 1')
                plt.fill_between(np.arange(len(blocks_mean_task_1)), blocks_mean_task_1-blocks_std_task_1, blocks_mean_task_1+blocks_std_task_1, alpha=0.2, color = isl_1[0])

                plt.plot(blocks_mean_task_2, color = mend[1], label = 'Task 2')
                plt.fill_between(np.arange(len(blocks_mean_task_2)), blocks_mean_task_2 - blocks_std_task_2, blocks_mean_task_2 + blocks_std_task_2, alpha=0.2, color = mend[1])

                plt.plot(blocks_mean_task_3, color = mend[2], label = 'Task 3')
                plt.fill_between(np.arange(len(blocks_mean_task_3)), blocks_mean_task_3 - blocks_std_task_3, blocks_mean_task_3 + blocks_std_task_3, alpha=0.2, color = mend[2])
                
            elif plot_block_all_tasks:
                 
               
                
                 blocks_all_tasls =  np.mean([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)
                 std_blocks_all_tasls =  np.std([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)/(np.sqrt(12))

                 plt.plot(blocks_all_tasls, color = isl_1[0], label = 'All Tasks Block Time')
                 plt.fill_between(np.arange(len(blocks_all_tasls)), blocks_all_tasls-std_blocks_all_tasls, blocks_all_tasls+std_blocks_all_tasls, alpha=0.2, color = isl_1[0])
 
            elif a_b_all_tasks == True:
                
               

                
                blocks_mean_a = np.mean([a_1,a_2,a_3,a_4,a_5,a_6],0)
                blocks_std_a = np.std([a_1,a_2,a_3,a_4,a_5,a_6],0)/(np.sqrt(6))
                
                blocks_mean_b = np.mean([b_1,b_2,b_3,b_4,b_5,b_6],0)
                blocks_std_b = np.std([b_1,b_2,b_3,b_4,b_5,b_6],0)/(np.sqrt(6))
               
                
                # blocks_mean_a = np.mean([a_s_1[i], a_s_2[i],a_s_3[i]],0)
                # blocks_std_a = np.std([a_s_1[i], a_s_2[i],a_s_3[i]],0)/(np.sqrt(3))
                
                # blocks_mean_b = np.mean([b_s_1[i], b_s_2[i], b_s_3[i]],0)
                # blocks_std_b = np.std([b_s_1[i], b_s_2[i], b_s_3[i]],0)/(np.sqrt(3))
               
                plt.plot(blocks_mean_b, color = isl_1[0], label = 'B')
                plt.fill_between(np.arange(len(blocks_mean_b)), blocks_mean_b-blocks_std_b, blocks_mean_b+blocks_std_b, alpha=0.2, color = isl_1[0])
    
                plt.plot(blocks_mean_a, color = isl_1[1], label = 'A')
                plt.fill_between(np.arange(len(blocks_mean_a)), blocks_mean_a-blocks_std_a, blocks_mean_a+blocks_std_a, alpha=0.2, color = isl_1[1])
    
                #plt.vlines(switch, np.min([np.min(blocks_mean_a),np.min(blocks_mean_b)]), np.max([np.max(blocks_mean_a),np.max(blocks_mean_b)]), linestyle = ':', color = 'grey', label = 'Switch')
            if plot_all == True:
                stack = np.vstack((a_1,a_2,a_3,a_4,a_5,a_6,b_1, b_2,b_3,b_4,b_5,b_6))
                
                labels = ['A 1 ', ' A 2 ', 'A 3', 'A 4', ' A 5 ', 'A 6', 'B 1 ', ' B 2 ', ' B 3', 'B 4 ', ' B 5 ', ' B 6']
                for s,ss in enumerate(stack):
                    plt.plot(ss)
                    
            plt.tight_layout()
            
            if count == 1:
                plt.legend()
            if  (neuron_count-1) in ind_above_chance:
                plt.title('Significant')
            else:
                plt.title(str(count))


            plt.subplot(3,4, count+6)
            plt.plot(a_list[i], color = isl[1], label = 'A')
            plt.plot(b_list[i], color = isl[2], label = 'B')
            plt.plot(rew_list[i], color = isl[3], label = 'Reward')
            plt.plot(no_rew_list[i], color = isl[4], label = 'No Rew')
            if  (neuron_count-1) in ind_above_chance:
                plt.title('Significant')
            else:
                plt.title(str(count))
            plt.vlines([25,36,43], np.min([np.min(a_list[i]),np.min(b_list[i]),np.min(rew_list[i]),np.min(no_rew_list[i])]),\
                                        np.max([np.max(a_list[i]),np.max(b_list[i]),np.max(rew_list[i]),np.max(no_rew_list[i])]),linestyle= '--', color = 'pink')

           
            if count == 1:
                plt.legend()
           
         pdf.savefig()      
         pdf.close()
         
         return ind_above_chance,neuron_count
                
     
        