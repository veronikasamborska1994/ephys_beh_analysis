#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:44:30 2020

@author: veronikasamborska
"""
import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import warp_trials_number as warp
from scipy.stats import norm
import palettable
from palettable import wesanderson as wes
import seaborn as sns
import heatmap_aligned as ha
from scipy import io
import ephys_beh_import as ep

from matplotlib.backends.backend_pdf import PdfPages
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



def a_to_a_b_to_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, beh = False, block = False, raw_data = True, HP = True, start = 35, end = 42):
   
    if beh  == True:
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,\
        trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_align_beh(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    elif block == True:   
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,\
        trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_aligned_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    if raw_data == True:
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,\
        trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP,\
        task_list_PFC = warp.all_sessions_align_beh_raw_data(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end)
   
    if HP == True:
        exp = HP_aligned_time
        state_exp = state_list_HP
        task_exp = task_list_HP
    else:
        exp = PFC_aligned_time
        state_exp = state_list_PFC
        task_exp = task_list_PFC
    
    # HP matrices 
    a_a_matrix_t_1 = []
    b_b_matrix_t_1 = []
    a_a_matrix_t_2 = []
    b_b_matrix_t_2 = []
    a_a_matrix_t_3 = []
    b_b_matrix_t_3 = []
    
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

        elif a_s_t_1 !=0 :
            a_s_t1_1 = session[1]
            a_s_t1_2 = session[3]
            b_s_t1_1 = session[0]
            b_s_t1_2 = session[2]

        if a_s_t_2 == 0:
            
            a_s_t2_1 = session[4]
            a_s_t2_2 = session[6]
            b_s_t2_1 = session[5]
            b_s_t2_2 = session[7]

            
        elif a_s_t_2 !=0:
            a_s_t2_1 = session[5]
            a_s_t2_2 = session[7]
            b_s_t2_1 = session[4]
            b_s_t2_2 = session[6]

        if a_s_t_3 == 0:
            a_s_t3_1 = session[8]
            a_s_t3_2 = session[10]
            b_s_t3_1 = session[9]
            b_s_t3_2 = session[11]

            
        elif a_s_t_3 != 0:
            a_s_t3_1 = session[9]
            a_s_t3_2 = session[11]
            b_s_t3_1 = session[8]
            b_s_t3_2 = session[10]
        
        
        if a_s_t_1 == 0 :
            a_a_matrix_t_1.append(np.hstack((a_s_t1_1,a_s_t1_2))) # At 13 change
        
        elif a_s_t_1 != 0 :
            b_b_matrix_t_1.append(np.hstack((b_s_t1_1, b_s_t1_2)))# At 13 change
        
        if a_s_t_2 == 0 :
            a_a_matrix_t_2.append(np.hstack((a_s_t2_1,a_s_t2_2))) # At 13 change
        
        elif a_s_t_2 != 0 :
            b_b_matrix_t_2.append(np.hstack((b_s_t2_1, b_s_t2_2))) # At 13 change
            
        if a_s_t_3 == 0 :
            a_a_matrix_t_3.append(np.hstack((a_s_t3_1,a_s_t3_2))) # At 13 change
        
        elif a_s_t_3 != 0 :
            b_b_matrix_t_3.append(np.hstack((b_s_t3_1, b_s_t3_2)))# At 13 change
           
            
    a_a_matrix_t_1 = np.concatenate(a_a_matrix_t_1,0)
    b_b_matrix_t_1 = np.concatenate(b_b_matrix_t_1,0)
    a_a_matrix_t_2 = np.concatenate(a_a_matrix_t_2,0)
    b_b_matrix_t_2 = np.concatenate(b_b_matrix_t_2,0)
    a_a_matrix_t_3 = np.concatenate(a_a_matrix_t_3,0)
    b_b_matrix_t_3 = np.concatenate(b_b_matrix_t_3,0)
    
    return a_a_matrix_t_1,b_b_matrix_t_1,a_a_matrix_t_2,b_b_matrix_t_2,a_a_matrix_t_3,b_b_matrix_t_3
  
   
     
def a_to_a_sub_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, beh = False, block = False, raw_data = True, HP = True, start = 35, end = 42):
   
    if beh  == True:
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,\
        trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_align_beh(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    elif block == True:   
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,\
        trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_aligned_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    if raw_data == True:
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,\
        trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP,\
        task_list_PFC = warp.all_sessions_align_beh_raw_data(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start, end)
        
    if HP == True:
        exp = HP_aligned_time
        state_exp = state_list_HP
        task_exp = task_list_HP
    else:
        exp = PFC_aligned_time
        state_exp = state_list_PFC
        task_exp = task_list_PFC
    
    a_a_matrix_t_1_list= []
    b_b_matrix_t_1_list= []
    a_b_matrix_t_1_list= []
    b_a_matrix_t_1_list= []
    a_a_matrix_t_2_list= []
    b_b_matrix_t_2_list= []
    a_b_matrix_t_2_list= []
    b_a_matrix_t_2_list= []
    a_a_matrix_t_3_list= []
    b_b_matrix_t_3_list= []
    a_b_matrix_t_3_list= []
    b_a_matrix_t_3_list= []
     
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
            a_b_matrix_t_1 = np.concatenate((np.mean([a_s_t1_1,a_s_t1_2],0),np.mean([b_s_t1_1,b_s_t1_2],0)),1) # At 13 change
            b_a_matrix_t_1 = np.concatenate((np.mean([b_s_t1_1,b_s_t1_2],0),np.mean([a_s_t1_1,a_s_t1_2],0)),1) # At 13 change


        elif a_s_t_1 !=0 :
            a_s_t1_1 = session[1]
            a_s_t1_2 = session[3]
            b_s_t1_1 = session[0]
            b_s_t1_2 = session[2]
            a_a_matrix_t_1 = np.hstack((a_s_t1_1,a_s_t1_2)) # At 13 change
            b_b_matrix_t_1 = np.hstack((b_s_t1_1, b_s_t1_2))# At 13 change
            a_b_matrix_t_1 = np.concatenate((np.mean([a_s_t1_1,a_s_t1_2],0),np.mean([b_s_t1_1,b_s_t1_2],0)),1) # At 13 change
            b_a_matrix_t_1 = np.concatenate((np.mean([b_s_t1_1,b_s_t1_2],0),np.mean([a_s_t1_1,a_s_t1_2],0)),1) # At 13 change

        if a_s_t_2 == 0:
            
            a_s_t2_1 = session[4]
            a_s_t2_2 = session[6]
            b_s_t2_1 = session[5]
            b_s_t2_2 = session[7]
            a_a_matrix_t_2 = np.hstack((a_s_t2_1,a_s_t2_2)) # At 13 change
            b_b_matrix_t_2 = np.hstack((b_s_t2_1, b_s_t2_2))# At 13 change
            a_b_matrix_t_2 = np.concatenate((np.mean([a_s_t2_1,a_s_t2_2],0),np.mean([b_s_t2_1,b_s_t2_2],0)),1) # At 13 change
            b_a_matrix_t_2 = np.concatenate((np.mean([b_s_t2_1,b_s_t2_2],0),np.mean([a_s_t2_1,a_s_t2_2],0)),1) # At 13 change

        elif a_s_t_2 !=0:
            a_s_t2_1 = session[5]
            a_s_t2_2 = session[7]
            b_s_t2_1 = session[4]
            b_s_t2_2 = session[6]
            a_a_matrix_t_2 = np.hstack((a_s_t2_1,a_s_t2_2)) # At 13 change
            b_b_matrix_t_2 = np.hstack((b_s_t2_1, b_s_t2_2))# At 13 change
            a_b_matrix_t_2 = np.concatenate((np.mean([a_s_t2_1,a_s_t2_2],0),np.mean([b_s_t2_1,b_s_t2_2],0)),1) # At 13 change
            b_a_matrix_t_2 = np.concatenate((np.mean([b_s_t2_1,b_s_t2_2],0),np.mean([a_s_t2_1,a_s_t2_2],0)),1) # At 13 change

        if a_s_t_3 == 0:
            a_s_t3_1 = session[8]
            a_s_t3_2 = session[10]
            b_s_t3_1 = session[9]
            b_s_t3_2 = session[11]
            a_a_matrix_t_3 = np.hstack((a_s_t3_1,a_s_t3_2)) # At 13 change
            b_b_matrix_t_3 = np.hstack((b_s_t3_1, b_s_t3_2))# At 13 change
            a_b_matrix_t_3 = np.concatenate((np.mean([a_s_t3_1,a_s_t3_2],0),np.mean([b_s_t3_1,b_s_t3_2],0)),1) # At 13 change
            b_a_matrix_t_3 = np.concatenate((np.mean([b_s_t3_1,b_s_t3_2],0),np.mean([a_s_t3_1,a_s_t3_2],0)),1) # At 13 change

            
        elif a_s_t_3 != 0:
            a_s_t3_1 = session[9]
            a_s_t3_2 = session[11]
            b_s_t3_1 = session[8]
            b_s_t3_2 = session[10]
            a_a_matrix_t_3 = np.hstack((a_s_t3_1,a_s_t3_2)) # At 13 change
            b_b_matrix_t_3 = np.hstack((b_s_t3_1, b_s_t3_2))# At 13 change
            a_b_matrix_t_3 = np.concatenate((np.mean([a_s_t3_1,a_s_t3_2],0),np.mean([b_s_t3_1,b_s_t3_2],0)),1) # At 13 change
            b_a_matrix_t_3 = np.concatenate((np.mean([b_s_t3_1,b_s_t3_2],0),np.mean([a_s_t3_1,a_s_t3_2],0)),1) # At 13 change
        
        a_a_matrix_t_1_list.append(a_a_matrix_t_1)
        b_b_matrix_t_1_list.append(b_b_matrix_t_1)
        a_b_matrix_t_1_list.append(a_b_matrix_t_1)
        b_a_matrix_t_1_list.append(b_a_matrix_t_1)
        a_a_matrix_t_2_list.append(a_a_matrix_t_2) 
        b_b_matrix_t_2_list.append(b_b_matrix_t_2)
        a_b_matrix_t_2_list.append(a_b_matrix_t_2)
        b_a_matrix_t_2_list.append(b_a_matrix_t_2)
        a_a_matrix_t_3_list.append(a_a_matrix_t_3)
        b_b_matrix_t_3_list.append(b_b_matrix_t_3)
        a_b_matrix_t_3_list.append(a_b_matrix_t_3)
        b_a_matrix_t_3_list.append(b_a_matrix_t_3)
        
    a_a_matrix_t_1_list= np.concatenate(a_a_matrix_t_1_list,0)
    b_b_matrix_t_1_list= np.concatenate(b_b_matrix_t_1_list,0)
    a_b_matrix_t_1_list= np.concatenate(a_b_matrix_t_1_list,0)
    b_a_matrix_t_1_list= np.concatenate(b_a_matrix_t_1_list,0)
    a_a_matrix_t_2_list= np.concatenate(a_a_matrix_t_2_list,0)
    b_b_matrix_t_2_list= np.concatenate(b_b_matrix_t_2_list,0)
    a_b_matrix_t_2_list= np.concatenate(a_b_matrix_t_2_list,0)
    b_a_matrix_t_2_list= np.concatenate(b_a_matrix_t_2_list,0)
    a_a_matrix_t_3_list= np.concatenate(a_a_matrix_t_3_list,0)
    b_b_matrix_t_3_list= np.concatenate(b_b_matrix_t_3_list,0)
    a_b_matrix_t_3_list= np.concatenate(a_b_matrix_t_3_list,0)
    b_a_matrix_t_3_list= np.concatenate(b_a_matrix_t_3_list,0)
    
    
    return  a_a_matrix_t_1_list, b_b_matrix_t_1_list,a_b_matrix_t_1_list, b_a_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list, a_b_matrix_t_2_list, b_a_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list, a_b_matrix_t_3_list, b_a_matrix_t_3_list
               

def plot_ab_ba_aa_bb():
    
    plot_ab_a_sub_aa(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, \
                         beh = False, block = False, raw_data = True, HP = True, start = 0, end = 62, t = 'Beh Aligned Firing Rats', region = 'HP', plot_neurons = True)
   
    plot_ab_a_sub_aa(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, \
                         beh = False, block = False, raw_data = True, HP = False, start = 0, end = 62, t = 'Beh Aligned Firing Rats', region = 'PFC', plot_neurons = True)
        
    
def plot_ab_a_sub_aa(data_HP, data_PFC,experiment_aligned_HP,experiment_aligned_PFC, \
                         beh = False, block = True, raw_data = False, HP = True, start = 0, end = 62, t = 'Beh Aligned', region = 'HP', plot_neurons = True):
   
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,a_b_matrix_t_1_list, b_a_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list, a_b_matrix_t_2_list, b_a_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list, a_b_matrix_t_3_list, b_a_matrix_t_3_list =   a_to_a_sub_a_b(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                                          beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
    
    plt.ion()     
    task1_aa = np.corrcoef(a_a_matrix_t_1_list.T)
    task2_aa = np.corrcoef(a_a_matrix_t_2_list.T)
    task3_aa = np.corrcoef(a_a_matrix_t_3_list.T)
    
    task1_bb = np.corrcoef(b_b_matrix_t_1_list.T)
    task2_bb = np.corrcoef(b_b_matrix_t_2_list.T)
    task3_bb = np.corrcoef(b_b_matrix_t_3_list.T)       
    
    task1_ab = np.corrcoef(a_b_matrix_t_1_list.T)
    task2_ab = np.corrcoef(a_b_matrix_t_2_list.T)
    task3_ab = np.corrcoef(a_b_matrix_t_3_list.T)
    
    task1_ba = np.corrcoef(b_a_matrix_t_1_list.T)
    task2_ba = np.corrcoef(b_a_matrix_t_2_list.T)
    task3_ba = np.corrcoef(b_a_matrix_t_3_list.T)
  
               
    if block == False: 
         ticks = np.arange(26)
         ticks_aa = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A1','A2','A3','A4','A5','A6','A7','A8','A9', 'A10', 'A11', 'A12', 'A13',]
         ticks_bb = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                               'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13']
         
         ticks_ab = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13']
         ticks_ba = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                              'A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13']
          
         change = 13
         
    elif block == True:
         ticks = np.arange(52)
         ticks_aa = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                     'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26',
                        'A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                     'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26']
         ticks_bb = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                     'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26',\
                  'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                     'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26']
         ticks_ab = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26',
                           'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',
                           'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26']
          
         ticks_ba = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',
                           'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26',\
                              'A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26' ]
           
         change = 26
          
    plt.ioff()

    c = palettable.scientific.sequential.LaPaz_5.mpl_colormap
    plt.figure(figsize=[10,10])
    plt.subplot(3,2,1)
    plt.imshow(task1_aa, cmap = c)
    
    plt.yticks(ticks,ticks_aa)
    plt.xticks(ticks,ticks_aa,  rotation = 90)
   
    plt.title(t +' ' + 'Task 1')
    plt.colorbar()

    plt.subplot(3,2,2)
    plt.imshow(task2_aa, cmap = c)
    plt.yticks(ticks,ticks_aa)
    plt.xticks(ticks,ticks_aa,  rotation = 90)
     
    plt.title(t +' ' + 'Task 2')
    plt.colorbar()

    plt.subplot(3,2,3)
    plt.imshow(task3_aa, cmap = c)
    plt.yticks(ticks,ticks_aa)
    plt.xticks(ticks,ticks_aa,  rotation = 90)
   
    plt.title(t +' ' + 'Task 3')
    plt.colorbar()

   
    plt.subplot(3,2,4)
    plt.imshow(task1_bb, cmap = c)
    plt.yticks(ticks,ticks_bb)
    plt.xticks(ticks,ticks_bb,  rotation = 90)
   
    plt.title(t+' '  + 'Task 1')
    plt.colorbar()

    
    plt.subplot(3,2,5)
    plt.imshow(task2_bb, cmap = c)
    plt.yticks(ticks,ticks_bb)
    plt.xticks(ticks,ticks_bb,  rotation = 90)
    plt.title(t +' ' + 'Task 2')
    plt.colorbar()

    
    plt.subplot(3,2,6)
    plt.imshow(task3_bb, cmap = c)
    plt.yticks(ticks,ticks_bb)
    plt.xticks(ticks,ticks_bb,  rotation = 90)
    plt.title(t+' '  + 'Task 3')
    plt.colorbar()
    plt.savefig('/Users/veronikasamborska/Desktop/time/'+ t + region +'_corr.pdf')

         
    isl = wes.FantasticFox2_5.mpl_colors
    if plot_neurons == True:
        pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ t + region +'.pdf')
        plt.ioff()
        count = 0
        plot_new = True
        for i,matrix in enumerate(a_a_matrix_t_1_list):
            
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
           
            a_s_tasks = np.mean([a_a_matrix_t_1_list[i], a_a_matrix_t_2_list[i], a_a_matrix_t_3_list[i]],0)
            a_s_tasks_std = np.std([a_a_matrix_t_1_list[i], a_a_matrix_t_2_list[i], a_a_matrix_t_3_list[i]],0)/(np.sqrt(3))
            a_b_s_tasks = np.mean([a_b_matrix_t_1_list[i], a_b_matrix_t_2_list[i], a_b_matrix_t_3_list[i]],0)
            a_b_s_tasks_std = np.std([a_b_matrix_t_1_list[i], a_b_matrix_t_2_list[i], a_b_matrix_t_3_list[i]],0)/(np.sqrt(3))

            b_s_tasks = np.mean([b_b_matrix_t_1_list[i], b_b_matrix_t_2_list[i], b_b_matrix_t_3_list[i]],0)
            b_s_tasks_std = np.std([b_b_matrix_t_1_list[i], b_b_matrix_t_2_list[i], b_b_matrix_t_3_list[i]],0)/(np.sqrt(3))
            b_a_s_tasks = np.mean([b_a_matrix_t_1_list[i], b_a_matrix_t_2_list[i], b_a_matrix_t_3_list[i]],0)
            b_a_s_tasks_std = np.std([b_a_matrix_t_1_list[i], b_a_matrix_t_2_list[i], b_a_matrix_t_3_list[i]],0)/(np.sqrt(3))

           
            plt.plot(a_s_tasks, color = isl[0], label = 'A to A')
            plt.fill_between(np.arange(len(a_s_tasks)), a_s_tasks-a_s_tasks_std, a_s_tasks+a_s_tasks_std, alpha=0.2, color = isl[0])
            plt.plot(a_b_s_tasks, color = isl[4], label = 'A to B')
            plt.fill_between(np.arange(len(a_b_s_tasks)), a_b_s_tasks-a_b_s_tasks_std, a_b_s_tasks+a_b_s_tasks_std, alpha=0.2, color = isl[4])
           
            
            plt.plot(b_s_tasks, color = isl[1], label = 'B to B')
            plt.fill_between(np.arange(len(b_s_tasks)), b_s_tasks-b_s_tasks_std, b_s_tasks+b_s_tasks_std, alpha=0.2, color = isl[1])
            plt.plot(b_a_s_tasks, color = isl[3], label = 'B to A')
            plt.fill_between(np.arange(len(b_a_s_tasks)), b_a_s_tasks-b_a_s_tasks_std, b_a_s_tasks+b_a_s_tasks_std, alpha=0.2, color = isl[3])
            plt.vlines(change,np.min([a_s_tasks,b_s_tasks,b_a_s_tasks,a_b_s_tasks]), np.max([a_s_tasks,b_s_tasks,b_a_s_tasks,a_b_s_tasks]), linestyle = ':', color = 'grey', label = 'Switch')
            
        
            plt.title(str(count))
            plt.subplot(3,4, count+6)
            
            autocorrelation_a_a_1 = np.correlate(a_a_matrix_t_1_list[i],a_a_matrix_t_1_list[i],'same')
            autocorrelation_a_a_2 = np.correlate(a_a_matrix_t_2_list[i],a_a_matrix_t_2_list[i],'same')
            autocorrelation_a_a_3 = np.correlate(a_a_matrix_t_3_list[i],a_a_matrix_t_3_list[i],'same')
            autocorrelation_a_a = np.mean([autocorrelation_a_a_1,autocorrelation_a_a_2,autocorrelation_a_a_3],0)
            autocorrelation_a_a_std = np.std([autocorrelation_a_a_1,autocorrelation_a_a_2,autocorrelation_a_a_3],0)/(np.sqrt(3))

            autocorrelation_b_b_1 = np.correlate(b_b_matrix_t_1_list[i],b_b_matrix_t_1_list[i],'same')
            autocorrelation_b_b_2 = np.correlate(b_b_matrix_t_2_list[i],b_b_matrix_t_2_list[i],'same')
            autocorrelation_b_b_3 = np.correlate(b_b_matrix_t_3_list[i],b_b_matrix_t_3_list[i],'same')
            autocorrelation_b_b = np.mean([autocorrelation_b_b_1,autocorrelation_b_b_2,autocorrelation_b_b_3],0)
            autocorrelation_b_b_std = np.std([autocorrelation_b_b_1,autocorrelation_b_b_2,autocorrelation_b_b_3],0)/(np.sqrt(3))

            plt.plot(autocorrelation_a_a, color = isl[0], linestyle = '--', label = 'AutoCorr A A')
            plt.fill_between(np.arange(len(autocorrelation_a_a)), autocorrelation_a_a-autocorrelation_a_a_std, autocorrelation_a_a+autocorrelation_a_a_std, alpha=0.2, color = isl[0])
           
            plt.plot(autocorrelation_b_b, color = isl[1], linestyle = '--', label = 'AutoCorr B B')
            plt.fill_between(np.arange(len(autocorrelation_b_b)), autocorrelation_b_b-autocorrelation_b_b_std, autocorrelation_b_b+autocorrelation_b_b_std, alpha=0.2, color = isl[1])
          
            autocorrelation_a_b_1 = np.correlate(a_b_matrix_t_1_list[i],a_b_matrix_t_1_list[i],'same')
            autocorrelation_a_b_2 = np.correlate(a_b_matrix_t_2_list[i],a_b_matrix_t_2_list[i],'same')
            autocorrelation_a_b_3 = np.correlate(a_b_matrix_t_3_list[i],a_b_matrix_t_3_list[i],'same')
            autocorrelation_a_b = np.mean([autocorrelation_a_b_1,autocorrelation_a_b_2,autocorrelation_a_b_3],0)
            autocorrelation_a_b_std = np.std([autocorrelation_a_b_1,autocorrelation_a_b_2,autocorrelation_a_b_3],0)/(np.sqrt(3))

            autocorrelation_b_a_1 = np.correlate(b_a_matrix_t_1_list[i],b_a_matrix_t_1_list[i],'same')
            autocorrelation_b_a_2 = np.correlate(b_a_matrix_t_2_list[i],b_a_matrix_t_2_list[i],'same')
            autocorrelation_b_a_3 = np.correlate(b_a_matrix_t_3_list[i],b_a_matrix_t_3_list[i],'same')
            autocorrelation_b_a = np.mean([autocorrelation_b_a_1,autocorrelation_b_a_2,autocorrelation_b_a_3],0)
            autocorrelation_b_a_std = np.std([autocorrelation_b_a_1,autocorrelation_b_a_2,autocorrelation_b_a_3],0)/(np.sqrt(3))
            
            plt.plot(autocorrelation_a_b, color = isl[4], linestyle = '--', label = 'AutoCorr A B')
            plt.fill_between(np.arange(len(autocorrelation_a_b)), autocorrelation_a_b - autocorrelation_a_b_std, autocorrelation_a_b + autocorrelation_a_b_std, alpha=0.2, color = isl[4])
           
            plt.plot(autocorrelation_b_a, color = isl[3], linestyle = '--', label = 'AutoCorr B A')
            plt.fill_between(np.arange(len(autocorrelation_b_a)), autocorrelation_b_a - autocorrelation_b_a_std, autocorrelation_b_a + autocorrelation_b_a_std, alpha=0.2, color = isl[3])
           
            plt.title(str(count))
            plt.tight_layout()
            if count == 1:
                plt.legend()
        pdf.savefig()      
        pdf.close()

                
  
def plot_trial_warped(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, block = False, HP = True, start = 35, end = 42):
    
    if block  == False:
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,\
        trials_since_block_list_HP,trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_align_beh(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
    else:   
        HP_aligned_time, PFC_aligned_time, state_list_HP, state_list_PFC,trials_since_block_list_HP,\
        trials_since_block_list_PFC,task_list_HP, task_list_PFC = warp.all_sessions_aligned_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start,end)
       
    if HP == True:
        exp = HP_aligned_time
        state_exp = state_list_HP
        task_exp = task_list_HP
    else:
        exp = PFC_aligned_time
        state_exp = state_list_PFC
        task_exp = task_list_PFC
        
    # HP matrices 
    a_b_matrix_t_1 = []
    b_a_matrix_t_1 = []
    a_b_matrix_t_2 = []
    b_a_matrix_t_2 = []
    a_b_matrix_t_3 = []
    b_a_matrix_t_3 = []
    
    for s, session in enumerate(exp):
        state = state_exp[s]
        task = task_exp[s]
        a_s_t_1 = np.where((state == 1) & (task == 1))[0][0]
        a_s_t_2 = np.where((state == 1) & (task == 2))[0][0]-np.where(task == 2)[0][0]
        a_s_t_3 = np.where((state == 1) & (task == 3))[0][0]-np.where(task == 3)[0][0]
        
        # Task 1 
        if a_s_t_1 == 0:
            a_s_t1 = np.mean((session[0],session[2]),0)
            b_s_t1 = np.mean((session[1],session[3]),0)
        elif a_s_t_1 !=0 :
            a_s_t1 = np.mean((session[1],session[3]),0)
            b_s_t1 = np.mean((session[0],session[2]),0)
        if a_s_t_2 == 0:
            a_s_t2 = np.mean((session[4],session[6]),0)
            b_s_t2 = np.mean((session[5],session[7]),0)
        elif a_s_t_2 !=0:
            a_s_t2 = np.mean((session[5],session[7]),0)
            b_s_t2 = np.mean((session[4],session[6]),0)
        if a_s_t_3 == 0:
            a_s_t3 = np.mean((session[8],session[10]),0)
            b_s_t3 = np.mean((session[9],session[11]),0)  
        elif a_s_t_3 != 0:
            a_s_t3 = np.mean((session[9],session[11]),0)
            b_s_t3 = np.mean((session[8],session[10]),0)
        
        
        if a_s_t_1 == 0 :
            a_b_matrix_t_1.append(np.hstack((a_s_t1,b_s_t1))) # At 13 change
        
        elif a_s_t_1 != 0 :
            b_a_matrix_t_1.append(np.hstack((b_s_t1, a_s_t1)))# At 13 change
        
        if a_s_t_2 == 0 :
            a_b_matrix_t_2.append(np.hstack((a_s_t2,b_s_t2))) # At 13 change
        
        elif a_s_t_2 != 0 :
            b_a_matrix_t_2.append(np.hstack((b_s_t2, a_s_t2))) # At 13 change
            
        if a_s_t_3 == 0 :
            a_b_matrix_t_3.append(np.hstack((a_s_t3,b_s_t3))) # At 13 change
        
        elif a_s_t_3 != 0 :
            b_a_matrix_t_3.append(np.hstack((b_s_t3, a_s_t3)))# At 13 change
           
            
    a_b_matrix_t_1 = np.concatenate(a_b_matrix_t_1,0)
    b_a_matrix_t_1 = np.concatenate(b_a_matrix_t_1,0)
    a_b_matrix_t_2 = np.concatenate(a_b_matrix_t_2,0)
    b_a_matrix_t_2 = np.concatenate(b_a_matrix_t_2,0)
    a_b_matrix_t_3 = np.concatenate(a_b_matrix_t_3,0)
    b_a_matrix_t_3 = np.concatenate(b_a_matrix_t_3,0)
    
    return  a_b_matrix_t_1, b_a_matrix_t_1, a_b_matrix_t_2, b_a_matrix_t_2, a_b_matrix_t_3, b_a_matrix_t_3
    
def plot():
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = True, t = 'HP Pre Init', s = 20, e = 25,\
                          ba = False, aa = True,plot_neurons = True,region = 'AA')
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'PFC Pre Init', s = 20, e = 25,\
                          ba = False, aa = True,plot_neurons = True,region = 'AA')
       
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = True, t = 'HP Pre Init', s = 20, e = 25,\
                          ba = True, aa = False, plot_neurons = True, region = 'AB')
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'PFC Pre Init', s = 20, e = 25,\
                          ba = True, aa = False, plot_neurons = True, region = 'AB')
       
        
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = True, t = 'HP Choice ', s = 35, e = 40,\
                          ba = False, aa = True,plot_neurons = True,region = 'AA')
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'PFC Choice ', s = 35, e = 40,\
                          ba = False, aa = True,plot_neurons = True,region = 'AA')
       
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = True, t = 'HP Choice ', s = 35, e = 40,\
                          ba = True, aa = False, plot_neurons = True, region = 'AB')
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'PFC Choice ', s = 35, e = 40,\
                          ba = True, aa = False, plot_neurons = True, region = 'AB')
       
        
        
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = True, t = 'HP Reward ', s = 42, e = 62,\
                          ba = False, aa = True,plot_neurons = True,region = 'AA')
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'PFC Reward ', s = 42, e = 62,\
                          ba = False, aa = True,plot_neurons = True,region = 'AA')
       
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = True, t = 'HP Reward ', s = 42, e = 62,\
                          ba = True, aa = False, plot_neurons = True, region = 'AB')
    
    plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'PFC Reward ', s = 42, e = 62,\
                          ba = True, aa = False, plot_neurons = True, region = 'AB')
     
        
        
def plot_a_b_correlations(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'Choice', s = 0, e = 62,\
                          ba = True, aa = False,plot_neurons = False, region = 'HP'):
    
    if ba == True:
        a_b_matrix_t_1, b_a_matrix_t_1, a_b_matrix_t_2, b_a_matrix_t_2, a_b_matrix_t_3, b_a_matrix_t_3 = plot_trial_warped(data_HP, data_PFC,experiment_aligned_HP,\
         experiment_aligned_PFC, block = block, HP = experiment,  start = s, end = e)
        task1_s1 = np.corrcoef(a_b_matrix_t_1.T)
        task2_s1 = np.corrcoef(a_b_matrix_t_2.T)
        task3_s1 = np.corrcoef(a_b_matrix_t_3.T)
        
        task1_s2 = np.corrcoef(b_a_matrix_t_1.T)
        task2_s2 = np.corrcoef(b_a_matrix_t_2.T)
        task3_s2 =  np.corrcoef(b_a_matrix_t_3.T)
        if block == False: 
            ticks = np.arange(26)
            ticks_ab = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                           'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13']
            ticks_ba = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                                  'A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13']
            change = 13
        elif block == True:
            ticks = np.arange(52)
            ticks_ab = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26',
                           'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',
                           'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26']
            ticks_ba = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',
                           'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26',\
                              'A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26' ]
            change = 26
            
    if aa ==True:
        a_a_matrix_t_1,b_b_matrix_t_1,a_a_matrix_t_2,b_b_matrix_t_2,a_a_matrix_t_3,b_b_matrix_t_3 = a_to_a_b_to_b(data_HP, data_PFC,experiment_aligned_HP,\
                                                                                                              experiment_aligned_PFC, block = block, HP = experiment,  start = s, end = e)
        task1_s1 = np.corrcoef(a_a_matrix_t_1.T)
        task2_s1 = np.corrcoef(a_a_matrix_t_2.T)
        task3_s1 = np.corrcoef(a_a_matrix_t_3.T)
        
        task1_s2 = np.corrcoef(b_b_matrix_t_1.T)
        task2_s2 = np.corrcoef(b_b_matrix_t_2.T)
        task3_s2 = np.corrcoef(b_b_matrix_t_3.T)
        
        if block == False: 
            ticks = np.arange(26)
            ticks_ab = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                           'A1','A2','A3','A4','A5','A6','A7','A8','A9', 'A10', 'A11', 'A12', 'A13',]
            ticks_ba = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                                  'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13']
            change = 13
            
        elif block == True:
            ticks = np.arange(52)
            ticks_ab = ['A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26',
                           'A1','A2','A3','A4','A5','A6', 'A7','A8','A9', 'A10', 'A11', 'A12', 'A13',\
                        'A14','A15','A16','A17','A18', 'A19','A20','A21', 'A22', 'A23', 'A24', 'A25', 'A26']
            ticks_ba = ['B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                        'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26',\
                     'B1','B2','B3','B4','B5','B6', 'B7','B8','B9', 'B10', 'B11', 'B12', 'B13',\
                        'B14','B15','B16','B17','B18', 'B19','B20','B21', 'B22', 'B23', 'B24', 'B25', 'B26']
        
            change = 26
            
        
    
        
    plt.ioff()
    c =palettable.scientific.sequential.LaPaz_5.mpl_colormap
    plt.figure(figsize=[10,10])
    plt.subplot(3,2,1)
    plt.imshow(task1_s1, cmap = c)
    
    plt.yticks(ticks,ticks_ab)
    plt.xticks(ticks,ticks_ab,  rotation = 90)
   
    plt.title(t +' ' + 'Task 1')
    plt.colorbar()

    plt.subplot(3,2,2)
    plt.imshow(task2_s1, cmap = c)
    plt.yticks(ticks,ticks_ab)
    plt.xticks(ticks,ticks_ab,  rotation = 90)
     
    plt.title(t +' ' + 'Task 2')
    plt.colorbar()

    plt.subplot(3,2,3)
    plt.imshow(task3_s1, cmap = c)
    plt.yticks(ticks,ticks_ab)
    plt.xticks(ticks,ticks_ab,  rotation = 90)
   
    plt.title(t +' ' + 'Task 3')
    plt.colorbar()

   
    plt.subplot(3,2,4)
    plt.imshow(task1_s2, cmap = c)
    plt.yticks(ticks,ticks_ba)
    plt.xticks(ticks,ticks_ba,  rotation = 90)
   
    plt.title(t+' '  + 'Task 1')
    plt.colorbar()

    
    plt.subplot(3,2,5)
    plt.imshow(task2_s2, cmap = c)
    plt.yticks(ticks,ticks_ab)
    plt.xticks(ticks,ticks_ba,  rotation = 90)
    plt.title(t +' ' + 'Task 2')
    plt.colorbar()

    
    plt.subplot(3,2,6)
    plt.imshow(task3_s2, cmap = c)
    plt.yticks(ticks,ticks_ba)
    plt.xticks(ticks,ticks_ba,  rotation = 90)
    plt.title(t+' '  + 'Task 3')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('/Users/veronikasamborska/Desktop/time/'+ t + region + ' '+'Corr' +'.pdf')
    
    if aa == False:
        av = np.mean([task1_s1, task2_s1,task3_s1, task1_s2, task2_s2,task3_s2],0)
        plt.figure()
        plt.imshow(av,cmap = c)
        #plt.yticks(ticks,ticks_ab)
        #plt.xticks(ticks,ticks_ba,  rotation = 90)
        plt.title('Average' + ' ' + t)
        
        plt.colorbar()
    
        plt.tight_layout()
        #plt.savefig('/Users/veronikasamborska/Desktop/'+ t +'Average' +'.pdf')
    elif aa == True:    
        av_a = np.mean([task1_s1, task2_s1,task3_s1],0)
        av_b = np.mean([task1_s2, task2_s2,task3_s2],0)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(av_a,cmap = c)
        plt.yticks(ticks,ticks_ab)
        plt.xticks(ticks,ticks_ab,  rotation = 90)
  
        #plt.yticks(ticks,ticks_ab)
        #plt.xticks(ticks,ticks_ba,  rotation = 90)
        plt.title('Average A' + ' ' + t)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(av_b,cmap = c)
        plt.yticks(ticks,ticks_ba)
        plt.xticks(ticks,ticks_ba,  rotation = 90)
  
        #plt.yticks(ticks,ticks_ab)
        #plt.xticks(ticks,ticks_ba,  rotation = 90)
        plt.title('Average B' + ' ' + t)
        plt.colorbar()
    
        plt.tight_layout()
        #plt.savefig('/Users/veronikasamborska/Desktop/'+ t +'Average' +'.pdf')
    #Autocorrelations
    
    
    # for m in [a_b_matrix_t_1,a_b_matrix_t_2,a_b_matrix_t_3,b_a_matrix_t_1,b_a_matrix_t_2,b_a_matrix_t_3]:
    #      m =np.corrcoef(m.T)[13:,:13]
    #      autocorrelation = np.correlate(m.flatten(),m.flatten(),'same')
    #      plt.figure()
    #      plt.plot(autocorrelation)
        
    isl = wes.FantasticFox2_5.mpl_colors
    if aa == True:
        plot = a_a_matrix_t_1
    else:
        plot  = a_b_matrix_t_1
    if plot_neurons == True:
        pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ t + region +'.pdf')
        plt.ioff()
        count = 0
        plot_new = True
        for i in plot:
            count +=1
            if count == 11:
                plot_new = True
    
                count = 1
            if plot_new == True:
                pdf.savefig()
                plt.clf()
                plt.figure()
                
                plot_new = False

            autocorrelation = np.correlate(i,i,'same')
            plt.subplot(5,4, count)
            plt.plot(i, color = isl[0])
            #plt.xticks(ticks, ticks_ab, rotation = 90)
            plt.vlines(change,np.min(i), np.max(i), linestyle = ':', color = 'grey', label = 'Switch')
            plt.legend()
            plt.title(str(count))
            plt.subplot(5,4, count+10)
            plt.plot(autocorrelation, color = isl[2], label = 'Auto')
            plt.legend()
            plt.title(str(count))
            plt.tight_layout()
        pdf.close()
                

def plot_surprise():
    plt.ion()
    plt.figure()

    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                            experiment_aligned_PFC,block = False, experiment = True,\
                                t = 'Init Aligned Beh', l = 'HP', i = 1, fig_no = 1, s = 20, e = 25)

    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                            experiment_aligned_PFC,block = False, experiment = True,\
                                t = 'Choice Aligned Beh', l = 'HP', i = 1, fig_no = 2, s = 35, e = 40)

    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                            experiment_aligned_PFC,block = False, experiment = True,\
                                t = 'Reward Aligned Beh', l = 'HP', i = 1, fig_no = 3, s = 42, e = 63)
              
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                                experiment_aligned_PFC,block = False, experiment = False,\
                                    t = 'Init Aligned Beh', l = 'PFC', i = 4, fig_no = 1, s = 20, e = 25)
    
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                                experiment_aligned_PFC,block = False, experiment = False,\
                                    t = 'Choice Aligned Beh', l = 'PFC', i = 4, fig_no = 2, s = 35, e = 40)
    
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                                experiment_aligned_PFC,block = False, experiment = False,\
                                    t = 'Reward Aligned Beh', l = 'PFC', i = 4, fig_no = 3, s = 42, e = 63)
   
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                            experiment_aligned_PFC,block = True, experiment = True,\
                                t = 'Init Aligned Beh', l = 'HP', i = 1, fig_no = 4, s = 20, e = 25)

    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                            experiment_aligned_PFC,block = True, experiment = True,\
                                t = 'Choice Aligned Beh', l = 'HP', i = 1, fig_no = 5, s = 35, e = 40)

    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                            experiment_aligned_PFC,block = True, experiment = True,\
                                t = 'Reward Aligned Beh', l = 'HP', i = 1, fig_no = 6, s = 42, e = 63)
              
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                                experiment_aligned_PFC,block = True, experiment = False,\
                                    t = 'Init Aligned Beh', l = 'PFC', i = 4, fig_no = 4, s = 20, e = 25)
    
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                                experiment_aligned_PFC,block = True, experiment = False,\
                                    t = 'Choice Aligned Beh', l = 'PFC', i = 4, fig_no = 5, s = 35, e = 40)
    
    surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, \
                                experiment_aligned_PFC,block = True, experiment = False,\
                                    t = 'Reward Aligned Beh', l = 'PFC', i = 4, fig_no = 6, s = 42, e = 63)
   

def surprise_between_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,block = False, experiment = False, t = 'Choice', l = 'PFC', i = 1, fig_no = 1, s = 40, e = 63):
    isl = wes.Moonrise5_6.mpl_colors

    c = isl[i]
    a_b_matrix_t_1, b_a_matrix_t_1, a_b_matrix_t_2, b_a_matrix_t_2, a_b_matrix_t_3, b_a_matrix_t_3 = plot_trial_warped(data_HP, data_PFC,experiment_aligned_HP,\
                                                                                                     experiment_aligned_PFC, block = block, HP = experiment,  start = s, end = e)

    block_changes = [a_b_matrix_t_1,b_a_matrix_t_1, a_b_matrix_t_2, b_a_matrix_t_2, a_b_matrix_t_3, b_a_matrix_t_3]
    surprise_list_neurons_a_b_diff_block = []
    surprise_list_neurons_a_b_diff_block_sqrt = []

    for b in block_changes:
        surprise_list_neurons_a_b = []
        for a_b in b:
        # Task 1 Mean rates on the first 20 A trials
            if block == False:
                index_baseline = 6
                index_block = 13
            else:
                index_baseline = 13
                index_block = 26
           
            a_first_mean = np.mean(a_b[:index_baseline], axis = 0)
            a_first_std = np.std(a_b[:index_baseline], axis = 0)   
            a_last = a_b[index_baseline:index_block]
            b_first = a_b[index_block:]
            min_std = 2
                         
            a_within = -norm.logpdf(a_last, a_first_mean, a_first_std + min_std)
            b_between = -norm.logpdf(b_first, a_first_mean, a_first_std + min_std)
               
            surprise_array = np.concatenate([a_within, b_between])                   
                
            surprise_list_neurons_a_b.append(surprise_array)
                
        surprise_list_neurons_mean = np.transpose(np.mean(np.asarray(surprise_list_neurons_a_b), axis = 0))
        surprise_list_neurons_sqrt = np.transpose(np.std(np.asarray(surprise_list_neurons_a_b), axis = 0))/np.sqrt(len(surprise_list_neurons_a_b))
        surprise_list_neurons_a_b_diff_block.append(surprise_list_neurons_mean)
        surprise_list_neurons_a_b_diff_block_sqrt.append(surprise_list_neurons_sqrt)
    
    mean = np.mean(surprise_list_neurons_a_b_diff_block,0)
    std = np.mean(surprise_list_neurons_a_b_diff_block_sqrt,0)
    plt.subplot(2,3,fig_no)
    plt.plot(mean, color = c, label = l)
    plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.2, color = c)
    
    if block == True:
        ind_change = index_baseline-1
    else:
        ind_change = index_baseline
    if experiment == True:
        plt.vlines(ind_change, min(mean), max(mean), linestyle = ':',color = 'grey', label = 'Switch')
    plt.title(t)
    plt.legend()
    #plt.ylim(1.2, 2.35)
    plt.legend()
    plt.ylabel('-log(p)X))')
    plt.xlabel(' Trial N')
    sns.despine()
    
    
    
     
    