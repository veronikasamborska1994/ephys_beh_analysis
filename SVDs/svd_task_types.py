#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:15:22 2020

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from itertools import combinations 
import scipy

def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid

def extract_data(data, a ='PFC'):

    if a == 'PFC':
        all_subjects = [data['DM'][0][:9], data['DM'][0][9:26],data['DM'][0][25:40],data['DM'][0][40:]]
        all_firing = [data['Data'][0][:9], data['Data'][0][9:26],data['Data'][0][25:40],data['Data'][0][40:]]
    else:   
        all_subjects = [data['DM'][0][:16], data['DM'][0][16:24],data['DM'][0][24:]]
        all_firing = [data['Data'][0][:16], data['Data'][0][16:24],data['Data'][0][24:]]
        
    neurons = 0
    animal_neurons = []
    for s in all_firing:
        animal_n = 0
        for ss in s:
            neurons += ss.shape[1]
            animal_n +=ss.shape[1]
        animal_neurons.append(animal_n)

    n_neurons_cum = 0

    flattened_all_clusters_task_1_first_half =  np.zeros((neurons,63*4))
    flattened_all_clusters_task_1_second_half = np.zeros((neurons,63*4))
    flattened_all_clusters_task_2_first_half = np.zeros((neurons,63*4))
    flattened_all_clusters_task_2_second_half =np.zeros((neurons,63*4))
    flattened_all_clusters_task_3_first_half = np.zeros((neurons,63*4))
    flattened_all_clusters_task_3_second_half = np.zeros((neurons,63*4))
    for d,dd in enumerate(all_subjects):
        dm = all_subjects[d]
        firing = all_firing[d]

    
        for  s, sess in enumerate(dm):
            
           
            DM = dm[s]
            firing_rates = firing[s]

            n_trials, n_neurons, n_timepoints = firing_rates.shape
            n_neurons_cum += n_neurons
        
              
            choices = DM[:,1]
            reward = DM[:,2] #[:ind_block[11]]  
            task =  DM[:,5] #[:ind_block[11]]
           
            a_pokes = DM[:,6] #[:ind_block[11]]
            b_pokes = DM[:,7] #[:ind_block[11]]
            
            taskid = task_ind(task, a_pokes, b_pokes)
          
            
            task_1 = np.where(taskid == 1)[0]
            task_2 = np.where(taskid == 2)[0]
            task_3 = np.where(taskid == 3)[0]
            
            task_1_1 = task_1[:int(len(task_1)/2)]
            task_1_2 = task_1[int(len(task_1)/2):]
            
            task_2_1 = task_2[:int(len(task_2)/2)]
            task_2_2 = task_2[int(len(task_2)/2):]
            
            task_3_1 = task_3[:int(len(task_3)/2)]
            task_3_2 = task_3[int(len(task_3)/2):]

            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_1_1))],0)
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_1_1))],0)
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_1_1))],0)
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_1_1))],0)


            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_1_2))],0)
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_1_2))],0)
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_1_2))],0)
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_1_2))],0)
            
            
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_2_1))],0)
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_2_1))],0)
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_2_1))],0)
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_2_1))],0)


            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_2_2))],0)
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_2_2))],0)
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_2_2))],0)
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_2_2))],0)
           
            
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_3_1))],0)
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_3_1))],0)
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_3_1))],0)
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_3_1))],0)


            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_3_2))],0)
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_3_2))],0)
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3]= np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_3_2))],0)
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_3_2))],0)
           
            
    return flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, animal_neurons
  
def permute(PFC, HP, diagonal = False):
    
  
    all_subjects = [PFC['DM'][0][:9], PFC['DM'][0][9:26],PFC['DM'][0][26:40],PFC['DM'][0][40:],HP['DM'][0][:16], HP['DM'][0][16:24],HP['DM'][0][24:]]
    all_subjects_firing = [PFC['Data'][0][:9], PFC['Data'][0][9:26],PFC['Data'][0][26:40],PFC['Data'][0][40:],HP['Data'][0][:16], HP['Data'][0][16:24],HP['Data'][0][24:]]

    animals_PFC = [0,1,2,3]
    animals_HP = [4,5,6]
    m, n = len(animals_PFC), len(animals_HP)
     
    u_v_area_shuffle = []
    u_v_area_1_shuffle  = []
    u_v_area_2_shuffle  = []
    u_v_area_3_shuffle  = []
        
     
    for indices_PFC in combinations(range(m + n), m):
        indices_HP = [i for i in range(m + n) if i not in indices_PFC]
       
        PFC_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_PFC)])
        HP_shuffle_dm = np.concatenate(np.asarray(all_subjects)[np.asarray(indices_HP)])
        
        PFC_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_PFC)])
        HP_shuffle_f = np.concatenate(np.asarray(all_subjects_firing)[np.asarray(indices_HP)])
        HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]
        PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]
       
        u_v_area = []
        u_v_area_1 = []
        u_v_area_2 = []
        u_v_area_3 = []
        
        for d in [HP_shuffle,PFC_shuffle]:
        
            dm = d[0]
            firing_all = d[1]
            neurons = 0
            for s in firing_all:
              
                neurons += s.shape[1]
        
            n_neurons_cum = 0
            flattened_all_clusters_task_1_first_half =  np.zeros((neurons,63*4))
            flattened_all_clusters_task_1_second_half = np.zeros((neurons,63*4))
            flattened_all_clusters_task_2_first_half = np.zeros((neurons,63*4))
            flattened_all_clusters_task_2_second_half =np.zeros((neurons,63*4))
            flattened_all_clusters_task_3_first_half = np.zeros((neurons,63*4))
            flattened_all_clusters_task_3_second_half = np.zeros((neurons,63*4))
          
            
            for  s, sess in enumerate(dm):
                print(s)
                
               
                DM = dm[s]
                firing_rates = firing_all[s]
    
                n_trials, n_neurons, n_timepoints = firing_rates.shape
                n_neurons_cum += n_neurons
            
                  
                choices = DM[:,1]
                reward = DM[:,2] #[:ind_block[11]]  
                task =  DM[:,5] #[:ind_block[11]]
               
                a_pokes = DM[:,6] #[:ind_block[11]]
                b_pokes = DM[:,7] #[:ind_block[11]]
                
                taskid = task_ind(task, a_pokes, b_pokes)
              
                
                task_1 = np.where(taskid == 1)[0]
                task_2 = np.where(taskid == 2)[0]
                task_3 = np.where(taskid == 3)[0]
                
                task_1_1 = task_1[:int(len(task_1)/2)]
                task_1_2 = task_1[int(len(task_1)/2):]
                
                task_2_1 = task_2[:int(len(task_2)/2)]
                task_2_2 = task_2[int(len(task_2)/2):]
                
                task_3_1 = task_3[:int(len(task_3)/2)]
                task_3_2 = task_3[int(len(task_3)/2):]
    
                flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_1_1))],0)
                flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_1_1))],0)
                flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_1_1))],0)
                flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_1_1))],0)
    
    
                flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_1_2))],0)
                flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_1_2))],0)
                flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_1_2))],0)
                flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_1_2))],0)
                
                
                flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_2_1))],0)
                flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_2_1))],0)
                flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_2_1))],0)
                flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_2_1))],0)
    
    
                flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_2_2))],0)
                flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_2_2))],0)
                flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_2_2))],0)
                flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_2_2))],0)
               
                
                flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_3_1))],0)
                flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_3_1))],0)
                flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_3_1))],0)
                flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_3_1))],0)
    
    
                flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:63] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_3_2))],0)
                flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63:63*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_3_2))],0)
                flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*2:63*3]= np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_3_2))],0)
                flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,63*3:63*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_3_2))],0)
       
            trp, average_between_all,average_within_all = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half,task = 0, diagonal= diagonal)
                
            trp_1, average_between_all_1,average_within_all_1 = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = 1, diagonal= diagonal)
            
            trp_2,average_between_all_2,average_within_all_2 = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = 2, diagonal= diagonal)
            
            trp_3,average_between_all_3,average_within_all_3 = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half,  task = 3, diagonal= diagonal)
           
             
            u_v_area.append(trp)
            u_v_area_1.append(trp_1)
            u_v_area_2.append(trp_2)
            u_v_area_3.append(trp_3)
            
        u_v_area_shuffle.append(u_v_area)
        u_v_area_1_shuffle.append(u_v_area_1)
        u_v_area_2_shuffle.append(u_v_area_2)
        u_v_area_3_shuffle.append(u_v_area_3)
        
    return  u_v_area_shuffle, u_v_area_1_shuffle, u_v_area_2_shuffle, u_v_area_3_shuffle



def svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = 1, diagonal = False):
    
    # flattened_all_clusters_task_1_first_half = scipy.stats.zscore(flattened_all_clusters_task_1_first_half,0)
    # flattened_all_clusters_task_1_second_half = scipy.stats.zscore(flattened_all_clusters_task_1_second_half,0)
 
    # flattened_all_clusters_task_2_first_half = scipy.stats.zscore(flattened_all_clusters_task_2_first_half,0)
    # flattened_all_clusters_task_2_second_half = scipy.stats.zscore(flattened_all_clusters_task_2_second_half,0)
     
    # flattened_all_clusters_task_3_first_half = scipy.stats.zscore(flattened_all_clusters_task_3_first_half,0)
    # flattened_all_clusters_task_3_second_half = scipy.stats.zscore(flattened_all_clusters_task_3_second_half,0)
     
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = False)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = False)

    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = False)
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
    t_u = np.transpose(u_t1_1)  
    t_v = np.transpose(vh_t1_1)  

    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_1 = np.transpose(u_t2_1)   
    t_v_t_2_1 = np.transpose(vh_t2_1)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  

    t_u_t_3_2 = np.transpose(u_t3_2)
    t_v_t_3_2 = np.transpose(vh_t3_2)  
    
    n_neurons = flattened_all_clusters_task_1_first_half.shape[0]
    #Compare task 1 Second Half 
    s_task_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_1_first_half, t_v_t_1_2])
    
    if diagonal == False:
        s_1_2 = s_task_1_2.diagonal()
    else:
        s_1_2 = np.sum(s_task_1_2**2, axis = 1)
     
    sum_c_task_1_2 = np.cumsum(abs(s_1_2))/n_neurons
    
   
    #Compare task 2 First Half from task 1 Last Half 
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half, t_v_t_1_2])
    if diagonal == False:
        s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    else:
        s_2_1_from_t_1_2 = np.sum(s_task_2_1_from_t_1_2**2, axis = 1)
    sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/n_neurons


    
    #Compare task 2 Second Half from first half
    s_task_2_2_from_t_2_1 = np.linalg.multi_dot([t_u_t_2_1, flattened_all_clusters_task_2_second_half, t_v_t_2_1])    
    if diagonal == False:
        s_2_2_from_t_2_1 = s_task_2_2_from_t_2_1.diagonal()
    else:
        s_2_2_from_t_2_1 = np.sum(s_task_2_2_from_t_2_1**2, axis = 1)
    sum_c_task_2_2_from_t_2_1 = np.cumsum(abs(s_2_2_from_t_2_1))/n_neurons
    
     #Compare task 2 Firs Half from second half
    s_task_2_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half, t_v_t_2_2])    
    if diagonal == False:
        s_2_1_from_t_2_2 = s_task_2_1_from_t_2_2.diagonal()
    else:
        s_2_1_from_t_2_2 = np.sum(s_task_2_1_from_t_2_2**2, axis = 1)
    sum_c_task_2_1_from_t_2_2 = np.cumsum(abs(s_2_1_from_t_2_2))/n_neurons


        
    #Compare task 3 First Half from Task 2 Last Half 
    s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_3_first_half, t_v_t_2_2])
    if diagonal == False:
        s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
    else:
        s_3_1_from_t_2_2 = np.sum(s_task_3_1_from_t_2_2**2, axis = 1)
    sum_c_task_3_1_from_t_2_2 = np.cumsum(abs(s_3_1_from_t_2_2))/n_neurons


     #Compare task 3 First Half from Task 2 Last Half 
    s_task_3_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_3_first_half, t_v_t_1_2])
    if diagonal == False:
        s_3_1_from_t_1_2 = s_task_3_1_from_t_1_2.diagonal()
    else:
        s_3_1_from_t_1_2 = np.sum(s_task_3_1_from_t_1_2**2, axis = 1)
    sum_c_task_3_1_from_t_1_2 = np.cumsum(abs(s_3_1_from_t_1_2))/n_neurons


    s_task_3_1_from_t_3_2 = np.linalg.multi_dot([t_u_t_3_2, flattened_all_clusters_task_3_first_half, t_v_t_3_2])
    if diagonal == False:
        s_3_1_from_t_3_2 = s_task_3_1_from_t_3_2.diagonal()
    else:
        s_3_1_from_t_3_2 = np.sum(s_task_3_1_from_t_3_2**2, axis = 1)
    sum_c_task_3_1_from_t_3_2 = np.cumsum(abs(s_3_1_from_t_3_2))/n_neurons

    if task == 0:
        average_within_all = np.mean([sum_c_task_1_2, sum_c_task_3_1_from_t_3_2, sum_c_task_2_1_from_t_2_2], axis = 0)
        average_between_all = np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_2_2, sum_c_task_3_1_from_t_1_2], axis = 0)

    elif task == 1:
        average_within_all = sum_c_task_1_2
        average_between_all = sum_c_task_2_1_from_t_1_2
        
    elif task ==2:
        average_within_all = sum_c_task_1_2
        average_between_all = sum_c_task_3_1_from_t_1_2  
    elif task ==3:  
        average_within_all = sum_c_task_2_1_from_t_2_2
        average_between_all = sum_c_task_3_1_from_t_2_2
     
            
    if diagonal == True:
        
        average_within_all = average_within_all/average_within_all[-1]
        average_between_all = average_between_all/average_between_all[-1]
 
    
    trp = (np.trapz(average_within_all) - np.trapz(average_between_all))/average_within_all.shape[0]
    
    return trp,average_between_all,average_within_all

def real_diff(data,a= 'PFC', task = 0, diagonal = False):
    
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, animal_neurons = extract_data(data, a =a) 

    trp, average_between_all,average_within_all = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = task, diagonal = diagonal)
   
    return trp, average_between_all,average_within_all


def animals_test():
    
    u_v_area_shuffle, u_v_area_1_shuffle, u_v_area_2_shuffle, u_v_area_3_shuffle = permute(PFC, HP, diagonal = False)
    
    # Real DIfferences 
    trp_pfc,  average_between_all_pfc,average_within_all_pfc =  real_diff(PFC,a= 'PFC', task = 0,  diagonal = False)
    trp_pfc_1, average_between_all_pfc_1,average_within_all_pfc_1 = real_diff(PFC,a= 'PFC', task = 1, diagonal = False)
    trp_pfc_2,average_between_all_pfc_2,average_within_all_pfc_2 = real_diff(PFC,a= 'PFC', task = 2,  diagonal = False)
    trp_pfc_3, average_between_all_pfc_3,average_within_all_pfc_3 = real_diff(PFC,a= 'PFC', task = 3,  diagonal = False)
    
    trp_hp, average_between_all_hp,average_within_all_hp =  real_diff(HP,a= 'HP', task = 0, diagonal = False)
    trp_hp_1, average_between_all_hp_1,average_within_all_hp_1 =  real_diff(HP,a= 'HP', task = 1, diagonal = False)
    trp_hp_2, average_between_all_hp_2,average_within_all_hp_2 =  real_diff(HP,a= 'HP', task = 2, diagonal = False)
    trp_hp_3, average_between_all_hp_3,average_within_all_hp_3 =  real_diff(HP,a= 'HP', task = 3, diagonal = False)
    
    
    diff_uv  = []
    diff_uv_1 = []
    diff_uv_2 = []
    diff_uv_3 = []
    
    for i,ii in enumerate(u_v_area_shuffle):
        diff_uv.append(u_v_area_shuffle[i][0]- u_v_area_shuffle[i][1])
        diff_uv_1.append(u_v_area_1_shuffle[i][0]- u_v_area_1_shuffle[i][1])
        diff_uv_2.append(u_v_area_2_shuffle[i][0]- u_v_area_2_shuffle[i][1])
        diff_uv_3.append(u_v_area_3_shuffle[i][0]- u_v_area_3_shuffle[i][1])
   
    uv_95 = np.percentile(diff_uv,95)
    uv_95_1 = np.percentile(diff_uv_1,95)
    uv_95_2 = np.percentile(diff_uv_2,95)
    uv_95_3 = np.percentile(diff_uv_3,95)
    
    real_uv = trp_hp-trp_pfc
    real_uv_1 = trp_hp_1-trp_pfc_1
    real_uv_2 = trp_hp_2-trp_pfc_2
    real_uv_3 = trp_hp_3-trp_pfc_3
    
    
    within_pfc = [average_within_all_pfc,average_within_all_pfc_1,average_within_all_pfc_2,average_within_all_pfc_3]
    within_hp = [average_within_all_hp,average_within_all_hp_1,average_within_all_hp_2,average_within_all_hp_3]
    between_pfc = [average_between_all_pfc,average_between_all_pfc_1,average_between_all_pfc_2,average_between_all_pfc_3]
    between_hp = [average_between_all_hp,average_between_all_hp_1,average_between_all_hp_2,average_between_all_hp_3]
  
    for ii,i in enumerate(within_hp):
        plt.figure()
        plt.plot(i, label = 'Within HP', color='black')
        plt.plot(within_pfc[ii], label = 'Within PFC', color = 'green')
        
        plt.plot(between_hp[ii], label = 'Between HP', color='black',linestyle = '--')
        plt.plot(between_pfc[ii], label = 'Between PFC', color = 'green', linestyle = '--')



    

    

    