#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:05:17 2019

@author: veronikasamborska
"""
# =============================================================================
# Script to plot correlation matrices that respect space; 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt


def coor_m_tim(data, r = 1):

    x = data['DM']
    y = data['Data']
     
    init_t1 = []
    init_t2 = []
    init_t3 = []
    
    choice_a_t1_r = []
    choice_a_t2_r = []
    choice_a_t3_r = []
    
    choice_b_t1_r = []
    choice_b_t2_r = []
    choice_b_t3_r = []
    
    choice_a_t1_nr = []
    choice_a_t2_nr = []
    choice_a_t3_nr = []
    
    choice_b_t1_nr = []
    choice_b_t2_nr = []
    choice_b_t3_nr = []
    
    choice_a_t1_rt_r = []
    choice_a_t2_rt_r = []
    choice_a_t3_rt_r = []
    
    choice_b_t1_rt_r = []
    choice_b_t2_rt_r = []
    choice_b_t3_rt_r = []

    choice_a_t1_rt_nr = []
    choice_a_t2_rt_nr = []
    choice_a_t3_rt_nr = []
    
    choice_b_t1_rt_nr = []
    choice_b_t2_rt_nr = []
    choice_b_t3_rt_nr = []



    for s, sess in enumerate(x):
        
        DM = x[s]
        #state =  DM[:,0]
        reward = DM[:,2]
        choices = DM[:,1]
        b_pokes = DM[:,6]
        a_pokes = DM[:,5]
        task = DM[:,4]
        
        firing_rates_all_time = y[s]
    
        taskid = np.zeros(len(task));
            
        #TASK1 -> A and B at leftright extremes; b_pokes = 10-apokes;
        taskid[b_pokes == 10 - a_pokes] = 1
        
        #TASK2 -> b is one of the diagonal ones [2,3,7,8]
        taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2
            
        #TASK3 -> b either top or bottom [2,3,7,8] (1 or 9). 
        taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
        It = np.arange(20 ,25) #Init
        Ct = np.arange(42, 47) #Choice
#        Rt = np.arange(42, 47) #Reward
        
        init_t1.append(np.mean(firing_rates_all_time[np.where(taskid == 1)[0],:,:][:,:,It], axis = 0))
        init_t2.append(np.mean(firing_rates_all_time[np.where(taskid == 2)[0],:,:][:,:,It], axis = 0))
        init_t3.append(np.mean(firing_rates_all_time[np.where(taskid == 3)[0],:,:][:,:,It], axis = 0))
        
            
        choice_a_t1_r.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 1) & (reward == 1))[0],:,:][:,:,Ct], axis = 0))
        choice_a_t2_r.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 1)& (reward == 1))[0],:,:][:,:,Ct], axis = 0))
        choice_a_t3_r.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 1)& (reward == 1))[0],:,:][:,:,Ct], axis = 0))
        
        choice_a_t1_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 1) & (reward == 0))[0],:,:][:,:,Ct], axis = 0))
        choice_a_t2_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 1)& (reward == 0))[0],:,:][:,:,Ct], axis = 0))
        choice_a_t3_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 1)& (reward == 0))[0],:,:][:,:,Ct], axis = 0))
        
        choice_b_t1_r.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 0)& (reward == 1))[0],:,:][:,:,Ct], axis = 0))
        choice_b_t2_r.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 0)& (reward == 1))[0],:,:][:,:,Ct], axis = 0))
        choice_b_t3_r.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 0)& (reward == 1))[0],:,:][:,:,Ct], axis = 0))
       
        choice_b_t1_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 0)& (reward == 0))[0],:,:][:,:,Ct], axis = 0))
        choice_b_t2_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 0)& (reward == 0))[0],:,:][:,:,Ct], axis = 0))
        choice_b_t3_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 0)& (reward == 0))[0],:,:][:,:,Ct], axis = 0))
            
        choice_a_t1_rt_r.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 1)& (reward == 1))[0],:,:][:,:,Rt], axis = 0))
        choice_a_t2_rt_r.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 1)& (reward == 1))[0],:,:][:,:,Rt], axis = 0))
        choice_a_t3_rt_r.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 1)& (reward == 1))[0],:,:][:,:,Rt], axis = 0))
        
        choice_b_t1_rt_r.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 0)& (reward == 1))[0],:,:][:,:,Rt], axis = 0))
        choice_b_t2_rt_r.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 0)& (reward == 1))[0],:,:][:,:,Rt], axis = 0))
        choice_b_t3_rt_r.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 0)& (reward == 1))[0],:,:][:,:,Rt], axis = 0))
        
        choice_a_t1_rt_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 1)& (reward == 0))[0],:,:][:,:,Rt], axis = 0))
        choice_a_t2_rt_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 1)& (reward == 0))[0],:,:][:,:,Rt], axis = 0))
        choice_a_t3_rt_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 1)& (reward == 0))[0],:,:][:,:,Rt], axis = 0))
        
        choice_b_t1_rt_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 1) & (choices == 0)& (reward == 0))[0],:,:][:,:,Rt], axis = 0))
        choice_b_t2_rt_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 2) & (choices == 0)& (reward == 0))[0],:,:][:,:,Rt], axis = 0))
        choice_b_t3_rt_nr.append(np.mean(firing_rates_all_time[np.where((taskid == 3) & (choices == 0)& (reward == 0))[0],:,:][:,:,Rt], axis = 0))
         
        
        

    list_arrays = [init_t1, init_t2, init_t3,\
                   choice_a_t1_r,choice_a_t2_r,choice_a_t3_r,\
                  # choice_a_t1_nr,choice_a_t2_nr, choice_a_t3_nr,\
                  # choice_b_t1_r,choice_b_t2_r, choice_b_t3_r,\
                   choice_b_t1_r, choice_b_t2_r, choice_b_t3_r]
                  # choice_a_t1_rt_r,choice_a_t2_rt_r,choice_a_t3_rt_r,\
                  # choice_a_t1_rt_nr,choice_a_t2_rt_nr, choice_a_t3_rt_nr,\
                  # choice_b_t1_rt_r,choice_b_t2_rt_r, choice_b_t3_rt_r]
                  # choice_b_t1_rt_nr, choice_b_t2_rt_nr, choice_b_t3_rt_nr]
        
    for i,l in enumerate(list_arrays):
        list_arrays[i] = np.concatenate(l, axis = 0)
        
    list_arrays = np.concatenate(list_arrays, axis = 1)
    corr = np.corrcoef(list_arrays.T)
    plt.figure(3)
    plt.imshow(corr)
    plt.imshow(corr[15:30,30:])
    plt.colorbar()

#    plt.imshow(corr[30:45,30:45])
#    plt.colorbar()
    
#    plt.colorbar()
    
    #plt.yticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],['I1','I2','I3','A1', 'A2', 'A3',\
    #                                                         'B1','B2','B3', 'A1RT', 'A2RT', 'A3RT',\
    #                                                        'B1RT', 'B2RT', 'B3RT'])
    #plt.xticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145],['I1','I2','I3','A1', 'A2', 'A3',\
    #                                                         'B1','B2','B3', 'A1RT', 'A2RT', 'A3RT',\
    #                                                        'B1RT', 'B2RT', 'B3RT'])
