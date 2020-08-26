#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:39:16 2020

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm
from scipy import io
from palettable import wesanderson as wes
import seaborn as sns
font = {'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)



def remap_surprise_time(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):
    
    y = data['DM'][0]
    x = data['Data'][0]
    task_time_confound_data = []
    task_time_confound_dm = []
    
    for  s, sess in enumerate(x):
        DM = y[s]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
                
        if task_1_2  == True:
            
            taskid_1 = 1
            taskid_2 = 2
            
        elif task_2_3 == True:
            
            taskid_1 = 2
            taskid_2 = 3
        
        elif task_1_3 == True:
            
            taskid_1 = 1
            taskid_2 = 3
        
        task_1 = np.where(taskid == taskid_1)[0][-1]
        task_2 = np.where(taskid == taskid_2)[0][0]
        if task_1+1 == task_2: #or task_1+1== task_2:
            task_time_confound_data.append(sess)
            task_time_confound_dm.append(y[s])
        task_1_rev = np.where(taskid == taskid_1)[0][0]
        task_2_rev = np.where(taskid == taskid_2)[0][-1]
        if task_2_rev+1 == task_1_rev:

            task_time_confound_data.append(sess)
            task_time_confound_dm.append(y[s])
      
    return task_time_confound_data,task_time_confound_dm



      