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
import palettable
from scipy import io
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import EmpiricalCovariance, MinCovDet
def run():
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    HP = io.loadmat('/Users/veronikasamborska/Desktop/data/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/data/PFC.mat')

def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid

def extract_data(data, t = 0, ab = 0):

    
    all_subjects = data['DM'][0]
    all_firing = data['Data'][0]
        
    neurons = 0
    animal_neurons = []
    for s in all_firing:
        
        neurons += s.shape[1]

    n_neurons_cum = 0

    flattened_all_clusters_task_1_first_half =  np.zeros((neurons,63*4))
    flattened_all_clusters_task_1_second_half = np.zeros((neurons,63*4))
    flattened_all_clusters_task_2_first_half = np.zeros((neurons,63*4))
    flattened_all_clusters_task_2_second_half =np.zeros((neurons,63*4))
    flattened_all_clusters_task_3_first_half = np.zeros((neurons,63*4))
    flattened_all_clusters_task_3_second_half = np.zeros((neurons,63*4))
    
    
    for  s, sess in enumerate(all_firing):
            
           
            DM = all_subjects[s]
            firing_rates = all_firing[s]

            n_trials, n_neurons, n_timepoints = firing_rates.shape
            n_neurons_cum += n_neurons
        
              
            choices = DM[:,1]
            reward = DM[:,2] #[:ind_block[11]]  
            task =  DM[:,5] #[:ind_block[11]]
           
            a_pokes = DM[:,6] #[:ind_block[11]]
            b_pokes = DM[:,7] #[:ind_block[11]]
            
        
            taskid = task_ind(task, a_pokes, b_pokes)
            if t ==0:
            
                task_1 = np.where(task == 1)[0]
                task_2 = np.where(task == 2)[0]
                task_3 = np.where(task == 3)[0]
            else:
                task_1 = np.where(taskid == 1)[0]
                task_2 = np.where(taskid == 2)[0]
                task_3 = np.where(taskid == 3)[0]
                
            # task_1_1 = task_1[::2]

            # task_1_2 = task_1[1::2]
            
            # task_2_1 = task_2[::2]
            # task_2_2 = task_2[1::2]
            
            # task_3_1 = task_3[::2]
            # task_3_2 = task_3[1::2]

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
           
    if ab ==0:       
        return flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, animal_neurons
    elif ab ==1:
        return flattened_all_clusters_task_1_first_half[:,:63*2], flattened_all_clusters_task_1_second_half[:,:63*2],\
        flattened_all_clusters_task_2_first_half[:,:63*2], flattened_all_clusters_task_2_second_half[:,:63*2],\
        flattened_all_clusters_task_3_first_half[:,:63*2],flattened_all_clusters_task_3_second_half[:,:63*2], animal_neurons
        
    elif ab ==2:
        return flattened_all_clusters_task_1_first_half[:,63*2:], flattened_all_clusters_task_1_second_half[:,63*2:],\
        flattened_all_clusters_task_2_first_half[:,63*2:], flattened_all_clusters_task_2_second_half[:,63*2:],\
        flattened_all_clusters_task_3_first_half[:,63*2:],flattened_all_clusters_task_3_second_half[:,63*2:], animal_neurons
  
def permute(PFC, HP, diagonal = False, perm = 2, axis = 0):
    
  
   
    all_subjects = np.concatenate((HP['DM'][0],PFC['DM'][0]),0)     
    all_subjects_firing = np.concatenate((HP['Data'][0],PFC['Data'][0]),0)     
        
    n_sessions = np.arange(len(HP['DM'][0])+len(PFC['DM'][0]))
   
  
    u_v_area_shuffle = []
    u_v_area_1_shuffle  = []
    u_v_area_2_shuffle  = []
    u_v_area_3_shuffle  = []
        
     
    for i in range(perm):
        np.random.shuffle(n_sessions) # Shuffle PFC/HP sessions
        indices_HP = n_sessions[:len(HP['DM'][0])]
        indices_PFC = n_sessions[len(HP['DM'][0]):]
        
        PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)]
        HP_shuffle_dm = all_subjects[np.asarray(indices_HP)]
        
        PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)]
        HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]
        
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
       
                
            trp_1, average_between_all_1,average_within_all_1 = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = 1, diagonal= diagonal, axis = axis)
            
            trp_2,average_between_all_2,average_within_all_2 = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = 2, diagonal= diagonal, axis = axis)
            
            trp_3,average_between_all_3,average_within_all_3 = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
            flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half,  task = 3, diagonal= diagonal, axis = axis)
            
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
                
               
                DM = dm[s]
                firing_rates = firing_all[s]
    
                n_trials, n_neurons, n_timepoints = firing_rates.shape
                n_neurons_cum += n_neurons
            
                  
                choices = DM[:,1]
                reward = DM[:,2] #[:ind_block[11]]  
                task =  DM[:,5] #[:ind_block[11]]
                
                task_1 = np.where(task == 1)[0]
                task_2 = np.where(task == 2)[0]
                task_3 = np.where(task == 3)[0]
                
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
            flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half,task = 0, diagonal = diagonal, axis = axis)
          
             
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
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = 1, diagonal = False, axis = 0):
    
    # demean = np.mean([np.mean(flattened_all_clusters_task_1_first_half,1), np.mean(flattened_all_clusters_task_2_first_half,1),np.mean(flattened_all_clusters_task_3_first_half,1),\
    #     np.mean(flattened_all_clusters_task_1_second_half,1),np.mean(flattened_all_clusters_task_2_second_half,1),np.mean(flattened_all_clusters_task_3_second_half,1)],0)
     
    # flattened_all_clusters_task_1_first_half = (flattened_all_clusters_task_1_first_half.T- demean).T
    # flattened_all_clusters_task_2_first_half = (flattened_all_clusters_task_2_first_half.T- demean).T
    # flattened_all_clusters_task_3_first_half = (flattened_all_clusters_task_3_first_half.T- demean).T

    # flattened_all_clusters_task_1_second_half = (flattened_all_clusters_task_1_second_half.T - demean).T
    # flattened_all_clusters_task_2_second_half = (flattened_all_clusters_task_2_second_half.T - demean).T
    # flattened_all_clusters_task_3_second_half = (flattened_all_clusters_task_3_second_half.T - demean).T 
    
    neurons = 345
    flattened_all_clusters_task_1_first_half = flattened_all_clusters_task_1_first_half[:neurons]
    flattened_all_clusters_task_2_first_half = flattened_all_clusters_task_2_first_half[:neurons]
    flattened_all_clusters_task_3_first_half = flattened_all_clusters_task_3_first_half[:neurons]

    flattened_all_clusters_task_1_second_half = flattened_all_clusters_task_1_second_half[:neurons]
    flattened_all_clusters_task_2_second_half = flattened_all_clusters_task_2_second_half[:neurons]
    flattened_all_clusters_task_3_second_half = flattened_all_clusters_task_3_second_half[:neurons]

    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = True)

    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = True)
    
    
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half
  
    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  

    
    n_neurons = flattened_all_clusters_task_1_first_half.shape[0]
    #Compare task 1 Second Half 
    s_task_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_1_first_half, t_v_t_1_2])
    
    if diagonal == False:
        s_1_2 = s_task_1_2.diagonal()
    else:
        s_1_2 = np.sum(s_task_1_2**2, axis = axis)
     
    sum_c_task_1_2 = np.cumsum(abs(s_1_2))/n_neurons
    
   
    #Compare task 2 First Half from task 1 Last Half 
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half, t_v_t_1_2])
    if diagonal == False:
        s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    else:
        s_2_1_from_t_1_2 = np.sum(s_task_2_1_from_t_1_2**2, axis = axis)
    sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/n_neurons

   
     #Compare task 2 Firs Half from second half
    s_task_2_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half, t_v_t_2_2])    
    if diagonal == False:
        s_2_1_from_t_2_2 = s_task_2_1_from_t_2_2.diagonal()
    else:
        s_2_1_from_t_2_2 = np.sum(s_task_2_1_from_t_2_2**2, axis = axis)
    sum_c_task_2_1_from_t_2_2 = np.cumsum(abs(s_2_1_from_t_2_2))/n_neurons


        
    #Compare task 3 First Half from Task 2 Last Half 
    s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_3_first_half, t_v_t_2_2])
    if diagonal == False:
        s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
    else:
        s_3_1_from_t_2_2 = np.sum(s_task_3_1_from_t_2_2**2, axis = axis)
    sum_c_task_3_1_from_t_2_2 = np.cumsum(abs(s_3_1_from_t_2_2))/n_neurons


     #Compare task 3 First Half from Task 2 Last Half 
    s_task_3_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_3_first_half, t_v_t_1_2])
    if diagonal == False:
        s_3_1_from_t_1_2 = s_task_3_1_from_t_1_2.diagonal()
    else:
        s_3_1_from_t_1_2 = np.sum(s_task_3_1_from_t_1_2**2, axis = axis)
    sum_c_task_3_1_from_t_1_2 = np.cumsum(abs(s_3_1_from_t_1_2))/n_neurons


    
    if task == 0:
        average_within_all = np.mean([sum_c_task_1_2, sum_c_task_2_1_from_t_2_2], axis = 0)
        average_between_all = np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_1_2], axis = 0)

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

def real_diff(data,a= 'PFC', task = 0, diagonal = False, axis = 0, ab = 0):
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, animal_neurons = extract_data(data, t = task, ab = ab)

    # u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)
    # u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)
    # u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = False)
    # i = 0
    # plt.figure(figsize = (10,3))
    # plt.subplot(1,3,1)
    # plt.plot(vh_t1_1[i,:63], label = 'A Reward', color = 'pink')
    # plt.plot(vh_t1_1[i,63:63*2], label = 'A No Reward',color = 'pink', linestyle = '--')
    # plt.plot(vh_t1_1[i,63*2:63*3], label = 'B Reward',color = 'grey')
    # plt.plot(vh_t1_1[i,63*3:63*4], label = 'B No Reward',color = 'grey', linestyle = '--')
    # plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

    # plt.subplot(1,3,2)
    # plt.plot(vh_t2_1[i,:63],  label = 'A Reward', color = 'pink')
    # plt.plot(vh_t2_1[i,63:63*2], label = 'A No Reward',color = 'pink', linestyle = '--')
    # plt.plot(vh_t2_1[i,63*2:63*3], label = 'B Reward',color = 'grey')
    # plt.plot(vh_t2_1[i,63*3:63*4], label = 'B No Reward',color = 'grey', linestyle = '--')
    # plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    

    # plt.subplot(1,3,3)
    # plt.plot(vh_t3_1[i,:63], label = 'A Reward', color = 'pink')
    # plt.plot(vh_t3_1[i,63:63*2], label = 'A No Reward',color = 'pink', linestyle = '--')
    # plt.plot(vh_t3_1[i,63*2:63*3], label = 'B Reward',color = 'grey')
    # plt.plot(vh_t3_1[i,63*3:63*4], label = 'B No Reward',color = 'grey', linestyle = '--')
    # plt.xticks([0,10,25,35,42,50,60], ['-1','-0.6','Init', 'Ch','R', '+0.32', '+0.72'])    
    # sns.despine()
    # plt.tight_layout()

    # plt.figure(figsize = (10,3))
    # plt.subplot(1,3,1)
    # plt.plot(u_t1_1[:,i], color = 'pink')
    # plt.subplot(1,3,2)
    # plt.plot(u_t2_1[:,i], color = 'pink')
    # plt.subplot(1,3,3)
    # plt.plot(u_t3_1[:,i], color = 'pink')
    # sns.despine()
    # plt.tight_layout()

  
    trp, average_between_all,average_within_all = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, task = task, diagonal = diagonal, axis = axis)
   
    return trp, average_between_all,average_within_all


def replicate_shoenbaum():
    
    data = PFC
    a = 'PFC'
    task = 0
    diagonal = False
    axis = 0
    ab = 0
    n_comp  = 100
    # M is trials x neuron
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, animal_neurons = extract_data(data, t = task, ab = ab)
        
     
    X = np.concatenate((flattened_all_clusters_task_1_first_half,flattened_all_clusters_task_2_first_half,flattened_all_clusters_task_3_first_half),1)
    
    X = X - np.mean(X,0)
    u, s, v = np.linalg.svd(X, full_matrices = True)
    
    #components by x neurons - first 100 neural components 
    Y = np.dot(X,v[:n_comp,:].T).T # lower dimensional representation of Y (reduce neuronal space in V)
    #Y_rec =  u_t1_1[:,:100] *s_t1_1[:100]
    
    ica = FastICA()
    # A is neural component x component demixing matrix 
    A = ica.fit_transform(Y) # A is a demixing matrix of component x component from the lower dimensional representation of Y (reduce neuronal space in V) 
    # n samples x n independent neural components
    
    B = np.dot(np.linalg.pinv(v[:,:n_comp]).T,A.T) # projecting the demixing component x component matrix onto the trial singular vectors with reduced number of neurons
    # B matrix tells you on which trials the neuronal 
    
    # Select 100 first neural and make them independent from each other, then project back onto trials 
    

    classes = np.concatenate((np.zeros(63)+1,np.ones(63)+1,np.ones(63)+2,(np.ones(63)+3),\
                             np.ones(63)+4,np.ones(63)+5,np.ones(63)+6, np.ones(63)+7,\
                            np.ones(63)+8,np.ones(63)+9,np.ones(63)+10,(np.ones(63)+11)))
        
    lda = LinearDiscriminantAnalysis()
    lda.fit(B,classes)
    x_new = lda.transform(B)
    plt.figure(1)
    var =lda.explained_variance_ratio_
    plt.plot(var)
    
    pre_inits = np.vstack(((np.mean(x_new[:25],0),np.mean(x_new[63:63+25],0),np.mean(x_new[63*2:(63*2+25)],0),\
            np.mean(x_new[63*3:63*3+25],0), np.mean(x_new[63*4:63*4+25],0),\
            np.mean(x_new[63*5:63*5+25],0), np.mean(x_new[63*6:63*6+25],0),\
           np.mean( x_new[63*7:63*7+25],0), np.mean(x_new[63*8:63*8+25],0),\
           np.mean( x_new[63*9:63*9+25],0), np.mean(x_new[63*10:63*10+25],0), np.mean(x_new[63*11:63*11+25],0))))
        
    inits = np.vstack(((np.mean(x_new[20:25],0),np.mean(x_new[63+20:63+25],0),np.mean(x_new[63+20*2:(63+20*2+25)],0),\
            np.mean(x_new[63+20*3:63*3+25],0), np.mean(x_new[63+20*4:63*4+25],0),\
            np.mean(x_new[63+20*5:63*5+25],0), np.mean(x_new[63+20*6:63*6+25],0),\
           np.mean( x_new[63+20*7:63*7+25],0), np.mean(x_new[63+20*8:63*8+25],0),\
           np.mean( x_new[63+20*9:63*9+25],0), np.mean(x_new[63+20*10:63*10+25],0), np.mean(x_new[63+20*11:63*11+25],0))))

    
    choices = np.vstack(((np.mean(x_new[36:42],0),np.mean(x_new[63+36:63+42],0),np.mean(x_new[63+36*2:(63+36*2+42)],0),\
            np.mean(x_new[63+36*3:63*3+42],0), np.mean(x_new[63+36*4:63*4+42],0),\
            np.mean(x_new[63+36*5:63*5+42],0), np.mean(x_new[63+36*6:63*6+42],0),\
           np.mean( x_new[63+36*7:63*7+42],0), np.mean(x_new[63+36*8:63*8+42],0),\
           np.mean( x_new[63+36*9:63*9+42],0), np.mean(x_new[63+36*10:63*10+42],0), np.mean(x_new[63+36*11:63*11+42],0))))

    rewards = np.vstack(((np.mean(x_new[42:50],0),np.mean(x_new[63+42:63+50],0),np.mean(x_new[63+42*2:(63+42*2+50)],0),\
            np.mean(x_new[63+42*3:63*3+50],0), np.mean(x_new[63+42*4:63*4+50],0),\
            np.mean(x_new[63+42*5:63*5+50],0), np.mean(x_new[63+42*6:63*6+50],0),\
           np.mean( x_new[63+42*7:63*7+50],0), np.mean(x_new[63+42*8:63*8+50],0),\
           np.mean( x_new[63+42*9:63*9+50],0), np.mean(x_new[63+42*10:63*10+50],0), np.mean(x_new[63+42*11:63*11+50],0))))
    plt.figure()
     
    trial_stages = np.concatenate((pre_inits,inits,choices,rewards),0)

    plt.imshow(np.corrcoef(trial_stages,trial_stages)[48:,48:])
                                          
    #plt.imshow(np.corrcoef(x_new,x_new)[:756,:756])
    
plt.subplot(3,1,1)   
plt.plot(a)
plt.xticks([0,1,2,3,4,5], [1,2,3,4,5,6])

plt.subplot(3,1,2)   
plt.plot(b)
plt.xticks([0,1,2,3,4,5], [1,2,3,4,5,6])

plt.subplot(3,1,3)   
plt.plot(a-b)
plt.xticks([0,1,2,3,4,5], [1,2,3,4,5,6])

def run_svd():
    
    run(HP, PFC, d = True, p = 2, axis = 0, ab = 2)
    run(HP, PFC, d = True, p = 2, axis = 1,ab = 1)
    run(HP, PFC, d = False, p = 2, ab = 2)

def run(HP, PFC, d = True, p = 1000, axis = 0, ab=0):
    
    u_v_area_shuffle, u_v_area_1_shuffle, u_v_area_2_shuffle, u_v_area_3_shuffle = permute(PFC, HP, diagonal = d, perm = p, axis = axis)
    # Real DIfferences 
    trp_pfc,  average_between_all_pfc,average_within_all_pfc =  real_diff(PFC,a= 'PFC', task = 0,  diagonal = d,axis = axis, ab = ab)
    trp_pfc_1, average_between_all_pfc_1,average_within_all_pfc_1 = real_diff(PFC,a= 'PFC', task = 1, diagonal = d,axis = axis, ab = ab)
    trp_pfc_2,average_between_all_pfc_2,average_within_all_pfc_2 = real_diff(PFC,a= 'PFC', task = 2,  diagonal = d,axis = axis, ab = ab)
    trp_pfc_3, average_between_all_pfc_3,average_within_all_pfc_3 = real_diff(PFC,a= 'PFC', task = 3,  diagonal = d,axis = axis, ab = ab)
    
    trp_hp, average_between_all_hp,average_within_all_hp =  real_diff(HP,a= 'HP', task = 0,  diagonal = d,axis = axis, ab = ab)
    trp_hp_1, average_between_all_hp_1,average_within_all_hp_1 =  real_diff(HP,a= 'HP', task = 1, diagonal = d, axis = axis, ab = ab)
    trp_hp_2, average_between_all_hp_2,average_within_all_hp_2 =  real_diff(HP,a= 'HP', task = 2, diagonal = d, axis = axis, ab = ab)
    trp_hp_3, average_between_all_hp_3,average_within_all_hp_3 =  real_diff(HP,a= 'HP', task = 3, diagonal = d,axis = axis, ab = ab)
    
    
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
    
    plt.figure(figsize = (4,5))
    plt.subplot(4,1,1)
    plt.hist(diff_uv, color = 'grey')
    plt.vlines(real_uv,ymin = 0, ymax = max(np.histogram(diff_uv)[0]))
    plt.vlines(uv_95,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')
  
    plt.subplot(4,1,2)
    plt.hist(diff_uv_1, color = 'grey')
    plt.vlines(real_uv_1,ymin = 0, ymax = max(np.histogram(diff_uv_1)[0]))
    plt.vlines(uv_95_1,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')

    plt.subplot(4,1,3)
    plt.hist(diff_uv_2, color = 'grey')
    plt.vlines(real_uv_2,ymin = 0, ymax = max(np.histogram(diff_uv_2)[0]))
    plt.vlines(uv_95_2,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')

    plt.subplot(4,1,4)
    plt.hist(diff_uv_3, color = 'grey')
    plt.vlines(real_uv_3,ymin = 0, ymax = max(np.histogram(diff_uv_3)[0]))
    plt.vlines(uv_95_3,ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'red')
 
    sns.despine()
  
    within_pfc = [average_within_all_pfc,average_within_all_pfc_1,average_within_all_pfc_2,average_within_all_pfc_3]
    within_hp = [average_within_all_hp,average_within_all_hp_1,average_within_all_hp_2,average_within_all_hp_3]
    between_pfc = [average_between_all_pfc,average_between_all_pfc_1,average_between_all_pfc_2,average_between_all_pfc_3]
    between_hp = [average_between_all_hp,average_between_all_hp_1,average_between_all_hp_2,average_between_all_hp_3]
  
    
    plt.figure(figsize = (4,10))
    l = 0
    for ii,i in enumerate(within_hp):
        l+=1
        plt.subplot(4,1,l)
        plt.plot(i, label = 'Within HP', color='black')
        plt.plot(within_pfc[ii], label = 'Within PFC', color = 'green')
        
        plt.plot(between_hp[ii], label = 'Between HP', color='black',linestyle = '--')
        plt.plot(between_pfc[ii], label = 'Between PFC', color = 'green', linestyle = '--')
        sns.despine()
        #plt.xtitle(str(l))
        
    
        
        
        
        
  # Plot diagonals and corelation matrices
  
    # task_1_1, task_1_2,\
    # task_2_1, task_2_2,\
    # task_3_1, task_3_2, animal_neurons = extract_data(HP)
    
   
    # cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap

    # plt.figure()
    # plt.subplot(2,3,1)
    # plt.imshow(np.corrcoef(task_1_2.T,task_3_2.T), cmap =cmap, aspect = 'auto')

    
    # diag_1_3 = np.diagonal(np.corrcoef(task_1_2.T,task_3_2.T),252)
    # plt.subplot(2,3,2)
    # plt.imshow(np.corrcoef(task_2_1.T,task_3_1.T), cmap =cmap, aspect = 'auto')
    
    # diag_2_3 = np.diagonal(np.corrcoef(task_2_1.T,task_3_1.T),252)

    # plt.subplot(2,3,3)
    # plt.imshow(np.corrcoef(task_1_2.T,task_2_2.T), cmap =cmap, aspect = 'auto')
    # diag_1_2 = np.diagonal(np.corrcoef(task_1_1.T,task_2_1.T),252)

    # ymin = np.min([diag_1_3,diag_1_2,diag_2_3]) - 0.05
    # ymax = np.max([diag_1_3,diag_1_2,diag_2_3]) + 0.05

    # plt.subplot(2,3,4)
    # plt.plot(diag_1_3, color = 'pink')
    # plt.ylim(ymin,ymax)
    # plt.subplot(2,3,5)
    # plt.plot(diag_2_3, color = 'pink')
    # plt.ylim(ymin,ymax)
    # plt.subplot(2,3,6)
    # plt.plot(diag_1_2, color = 'pink')
    # plt.ylim(ymin,ymax)
    # sns.despine()
   



    

    