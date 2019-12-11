#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:24:11 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 


def correlations(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):    
    
    data = HP
    y = data['DM'][0]
    x = data['Data'][0]

    stack_array = []
    for  s, sess in enumerate(x):
        DM = y[s]
       
        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        state = DM[:,0]
        block = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
       
        if task_1_2 == True:
            
            taskid_1 = 1
            taskid_2 = 2
            
        elif task_2_3 == True:
            
            taskid_1 = 2
            taskid_2 = 3
        
        elif task_1_3 == True:
            
            taskid_1 = 1
            taskid_2 = 3
        
        task_1_a_bad = np.where((taskid == taskid_1) & (choices == 1) & (state == 0))[0] # Find indicies for task 1 A
        task_1_a_good = np.where((taskid == taskid_1) & (choices == 1) & (state == 1))[0] # Find indicies for task 1 A

        task_1_b_bad = np.where((taskid == taskid_1) & (choices == 0) & (state == 1))[0] # Find indicies for task 1 A
        task_1_b_good = np.where((taskid == taskid_1) & (choices == 0) & (state == 0))[0] # Find indicies for task 1 A

        task_2_a_bad = np.where((taskid == taskid_2) & (choices == 1) & (state == 0))[0] # Find indicies for task 1 A
        task_2_a_good = np.where((taskid == taskid_2) & (choices == 1) & (state == 1))[0] # Find indicies for task 1 A

        task_2_b_bad = np.where((taskid == taskid_2) & (choices == 0) & (state == 1))[0] # Find indicies for task 1 A
        task_2_b_good = np.where((taskid == taskid_2) & (choices == 0) & (state == 0))[0] # Find indicies for task 1 A


        trials_since_block = []
        t = 0
        for st,s in enumerate(state):
            if state[st-1] != state[st]:
                t = 0
            else:
                t+=1
            trials_since_block.append(t)
      
        firing_rates_mean_time = sess
        task_1_a_bad_f = np.mean(firing_rates_mean_time[task_1_a_bad[:10]],axis = 2) 
        task_1_a_good_f = np.mean(firing_rates_mean_time[task_1_a_good[:10]],axis = 2) 

        task_1_b_bad_f = np.mean(firing_rates_mean_time[task_1_b_bad[:10]],axis = 2) 
        task_1_b_good_f = np.mean(firing_rates_mean_time[task_1_b_good[:10]],axis = 2) 
        
        task_2_a_bad_f = np.mean(firing_rates_mean_time[task_2_a_bad[:10]],axis = 2) 
        task_2_a_good_f = np.mean(firing_rates_mean_time[task_2_a_good[:10]],axis = 2) 

        task_2_b_bad_f = np.mean(firing_rates_mean_time[task_2_b_bad[:10]],axis = 2) 
        task_2_b_good_f = np.mean(firing_rates_mean_time[task_2_b_good[:10]],axis = 2) 
        
        stack = np.vstack((task_1_a_bad_f,task_1_a_good_f,task_1_b_bad_f,task_1_b_good_f,task_2_a_bad_f,task_2_a_good_f,task_2_b_bad_f,task_2_b_good_f))
        print(stack.shape)
        
        if stack.shape[0] == 80:
            stack_array.append(stack)
   all_conc = np.concatenate(stack_array,1)
   corr  = np.corrcoef(stack)
   
   plt.imshow(corr)
   plt.yticks(np.arange(0,80,10),['A bad T1', 'A good T1', 'B bad T1', 'B good T1', 'A bad T2', 'A good T2', 'B bad T2', 'B good T2'])
  
   plt.xticks(np.arange(0,80,10),['A bad T1', 'A good T1', 'B bad T1', 'B good T1', 'A bad T2', 'A good T2', 'B bad T2', 'B good T2'])

      