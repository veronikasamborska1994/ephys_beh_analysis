#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:27:14 2019

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import remapping_count as rc 

HP = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
PFC = scipy.io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')

Data_HP = HP['Data'][0]
DM_HP = HP['DM'][0]
Data_PFC = PFC['Data'][0]
DM_PFC = PFC['DM'][0]


def block_id(taskid,block_ids):
    
    blockid_1 = taskid[0]
    blockid_3 = taskid[-1]
    
    if blockid_1 == 1 and blockid_3 == 2 or  blockid_1 == 2 and blockid_3 == 1:
        blockid_2 = 3
        
    elif blockid_1 == 3 and blockid_3 == 2 or  blockid_1 == 2 and blockid_3 == 3:
        blockid_2 = 1
        
    elif blockid_1 == 3 and blockid_3 == 1 or blockid_1 == 1 and blockid_3 == 3:
        blockid_2 = 2
        
        
    if blockid_1 == 1 and blockid_2 == 2 and blockid_3 == 3:
        task_1_block = block_ids[:4]
        task_2_block = block_ids[4:8]
        task_3_block = block_ids[8:]
        
    elif blockid_1 == 1 and blockid_2 == 3 and blockid_3 == 2:
        task_1_block = block_ids[:4]
        task_2_block = block_ids[8:]
        task_3_block = block_ids[4:8]

    elif blockid_1 == 2 and blockid_2 == 1 and blockid_3 == 3:
        task_1_block =  block_ids[4:8]
        task_2_block = block_ids[:4]
        task_3_block = block_ids[8:]
    
    elif blockid_1 == 2 and blockid_2 == 3 and blockid_3 == 1:
        task_1_block =  block_ids[8:]
        task_2_block = block_ids[:4]
        task_3_block = block_ids[4:8]

    elif blockid_1 == 3 and blockid_2 == 1 and blockid_3 == 2:
        task_1_block =  block_ids[4:8]
        task_2_block = block_ids[8:]
        task_3_block = block_ids[:4]
    
    elif blockid_1 == 3 and blockid_2 == 2 and blockid_3 == 1:
        task_1_block =  block_ids[8:]
        task_2_block = block_ids[4:8]
        task_3_block = block_ids[:4]
        
    return task_1_block, task_2_block, task_3_block
        
        
        
        
        
def svd_trial(data,dm, n):
   
    all_min = []
    svd_task_1_1 = []
    svd_task_1_2 = []
    svd_task_1_3 = []
    svd_task_1_4 = []

    svd_task_2_1 = []
    svd_task_2_2 = []
    svd_task_2_3 = []
    svd_task_2_4 = []

    svd_task_3_1 = []
    svd_task_3_2 = []
    svd_task_3_3 = []
    svd_task_3_4 = []

            
    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = data[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        #state = DM[:,0]
        #trials = np.arange(len(state))
        #choices = DM[:,1]
        #reward = DM[:,2]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        block = DM[:,4]
        block_df = np.diff(block)
        block_ids = np.where(block_df != 0)[0]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        all_trials = []

        if len(block_ids) > 11:
            
             task_1_block, task_2_block, task_3_block = block_id(taskid,block_ids)

             task_1_1 = int(np.where(np.where(taskid ==1)[0] == task_1_block[0])[0])
             task_1_2 = int(np.where(np.where(taskid ==1)[0] == task_1_block[1])[0])
             task_1_3 = int(np.where(np.where(taskid ==1)[0] == task_1_block[2])[0])
             task_1_4 = int(np.where(np.where(taskid ==1)[0] == task_1_block[3])[0])


             firing_t_1_1= firing_rates[np.where(taskid ==1)][:task_1_1]
             firing_t_1_2 = firing_rates[np.where(taskid ==1)][task_1_1:task_1_2]
             firing_t_1_3 = firing_rates[np.where(taskid ==1)][task_1_2:task_1_3]
             firing_t_1_4 = firing_rates[np.where(taskid ==1)][task_1_3:task_1_4]

            
             task_2_1 = int(np.where(np.where(taskid ==2)[0]== task_2_block[0])[0])
             task_2_2 = int(np.where(np.where(taskid ==2)[0]== task_2_block[1])[0])
             task_2_3 = int(np.where(np.where(taskid ==2)[0]== task_2_block[2])[0])
             task_2_4 = int(np.where(np.where(taskid ==2)[0]== task_2_block[3])[0])

        
             firing_t_2_1 = firing_rates[np.where(taskid ==2)][:task_2_1]
             firing_t_2_2 = firing_rates[np.where(taskid ==2)][task_2_1:task_2_2]
             firing_t_2_3 = firing_rates[np.where(taskid ==2)][task_2_2:task_2_3]
             firing_t_2_4 = firing_rates[np.where(taskid ==2)][task_2_3:task_2_4]
   
            
             task_3_1 = int(np.where(np.where(taskid ==3)[0]== task_3_block[0])[0])
             task_3_2 = int(np.where(np.where(taskid ==3)[0]== task_3_block[1])[0])
             task_3_3 = int(np.where(np.where(taskid ==3)[0]== task_3_block[2])[0])
             task_3_4 = int(np.where(np.where(taskid ==3)[0]== task_3_block[3])[0])


             firing_t_3_1= firing_rates[np.where(taskid == 3)][:task_3_1]
             firing_t_3_2 = firing_rates[np.where(taskid == 3)][task_3_1:task_3_2]
             firing_t_3_3 = firing_rates[np.where(taskid == 3)][task_3_2:task_3_3]
             firing_t_3_4 = firing_rates[np.where(taskid == 3)][task_3_3:task_3_4]
             
             
             all_trials = [len(firing_t_1_1), len(firing_t_1_2), len(firing_t_1_3), len(firing_t_1_4),\
                    len(firing_t_2_1), len(firing_t_2_2), len(firing_t_2_3), len(firing_t_2_4),\
                    len(firing_t_3_1), len(firing_t_3_2), len(firing_t_3_3), len(firing_t_3_4)] 
            
             min_all_trials = np.min(all_trials)
             all_min.append(min_all_trials)
             
             if min_all_trials > n:
            
                 svd_task_1_1.append(firing_t_1_1[:n,:,:])
                 svd_task_1_2.append(firing_t_1_2[:n,:,:])
                 svd_task_1_3.append(firing_t_1_3[:n,:,:])
                 svd_task_1_4.append(firing_t_1_4[:n,:,:])
    
                 svd_task_2_1.append(firing_t_2_1[:n,:,:])
                 svd_task_2_2.append(firing_t_2_2[:n,:,:])
                 svd_task_2_3.append(firing_t_2_3[:n,:,:])
                 svd_task_2_4.append(firing_t_2_4[:n,:,:])
    
                 svd_task_3_1.append(firing_t_3_1[:n,:,:])
                 svd_task_3_2.append(firing_t_3_2[:n,:,:])
                 svd_task_3_3.append(firing_t_3_3[:n,:,:])
                 svd_task_3_4.append(firing_t_3_4[:n,:,:])
                 
    
    svd_task_1_1 = np.concatenate(svd_task_1_1,1)
    svd_task_1_2 = np.concatenate(svd_task_1_2,1)
    svd_task_1_3 = np.concatenate(svd_task_1_3,1)
    svd_task_1_4 = np.concatenate(svd_task_1_4,1)

    svd_task_2_1 = np.concatenate(svd_task_2_1,1)
    svd_task_2_2 = np.concatenate(svd_task_2_2,1)
    svd_task_2_3 = np.concatenate(svd_task_2_3,1)
    svd_task_2_4 = np.concatenate(svd_task_2_4,1)

    svd_task_3_1 = np.concatenate(svd_task_3_1,1)
    svd_task_3_2 = np.concatenate(svd_task_3_2,1)
    svd_task_3_3 = np.concatenate(svd_task_3_3,1)
    svd_task_3_4 = np.concatenate(svd_task_3_4,1)

    return svd_task_1_1, svd_task_1_2, svd_task_1_3, svd_task_1_4, svd_task_2_1, svd_task_2_2, svd_task_2_3,\
    svd_task_2_4, svd_task_3_1 , svd_task_3_2, svd_task_3_3, svd_task_3_4 




def svd(data,dm, c, t, n, start, end):
    
    svd_task_1_1, svd_task_1_2, svd_task_1_3, svd_task_1_4, svd_task_2_1, svd_task_2_2, svd_task_2_3,\
    svd_task_2_4, svd_task_3_1 , svd_task_3_2, svd_task_3_3, svd_task_3_4  = svd_trial(data,dm,n)

    
    task_1_svd_1 = np.vstack((np.mean(svd_task_1_1[:,:,start:end],axis = 2),np.mean(svd_task_1_2[:,:,start:end],axis = 2)))
    task_1_svd_2 = np.vstack((np.mean(svd_task_1_3[:,:,start:end],axis = 2),np.mean(svd_task_1_4[:,:,start:end],axis = 2)))
    
    task_2_svd_1 = np.vstack((np.mean(svd_task_2_1[:,:,start:end], axis = 2),np.mean(svd_task_2_2[:,:,start:end],axis = 2)))
    task_2_svd_2 = np.vstack((np.mean(svd_task_2_3[:,:,start:end],axis = 2),np.mean(svd_task_2_4[:,:,start:end],axis = 2)))
        
    task_3_svd_1 = np.vstack((np.mean(svd_task_2_1[:,:,start:end],axis = 2),np.mean(svd_task_2_2[:,:,start:end],axis = 2)))
    task_3_svd_2 = np.vstack((np.mean(svd_task_2_3[:,:,start:end],axis = 2),np.mean(svd_task_2_4[:,:,start:end],axis = 2)))
  
#    task_1_svd_1 = np.vstack((np.transpose(svd_task_1_1,[0,2,1]).reshape(svd_task_1_1.shape[0]*svd_task_1_1.shape[2],svd_task_1_1.shape[1]),\
#                             np.transpose(svd_task_1_2,[0,2,1]).reshape(svd_task_1_2.shape[0]*svd_task_1_2.shape[2],svd_task_1_2.shape[1])))
#   
#    task_1_svd_2 = np.vstack((np.transpose(svd_task_1_3,[0,2,1]).reshape(svd_task_1_3.shape[0]*svd_task_1_3.shape[2],svd_task_1_3.shape[1]),\
#                             np.transpose(svd_task_1_4,[0,2,1]).reshape(svd_task_1_4.shape[0]*svd_task_1_4.shape[2],svd_task_1_4.shape[1])))
#    
#    task_2_svd_1 = np.vstack((np.transpose(svd_task_2_1,[0,2,1]).reshape(svd_task_2_1.shape[0]*svd_task_2_1.shape[2],svd_task_2_1.shape[1]),\
#                             np.transpose(svd_task_2_2,[0,2,1]).reshape(svd_task_2_2.shape[0]*svd_task_2_2.shape[2],svd_task_2_2.shape[1])))
#    
#    task_2_svd_2 = np.vstack((np.transpose(svd_task_2_3,[0,2,1]).reshape(svd_task_2_3.shape[0]*svd_task_2_3.shape[2],svd_task_2_3.shape[1]),\
#                             np.transpose(svd_task_2_4,[0,2,1]).reshape(svd_task_2_4.shape[0]*svd_task_2_4.shape[2],svd_task_2_4.shape[1])))
#   
#    task_3_svd_1 = np.vstack((np.transpose(svd_task_3_1,[0,2,1]).reshape(svd_task_3_1.shape[0]*svd_task_3_1.shape[2],svd_task_3_1.shape[1]),\
#                             np.transpose(svd_task_3_2,[0,2,1]).reshape(svd_task_3_2.shape[0]*svd_task_3_2.shape[2],svd_task_3_2.shape[1])))
#    
#    task_3_svd_2 = np.vstack((np.transpose(svd_task_3_3,[0,2,1]).reshape(svd_task_3_3.shape[0]*svd_task_3_3.shape[2],svd_task_3_3.shape[1]),\
#                             np.transpose(svd_task_3_4,[0,2,1]).reshape(svd_task_3_4.shape[0]*svd_task_3_4.shape[2],svd_task_3_4.shape[1])))
#   
    
    
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(task_1_svd_1.T, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(task_1_svd_2.T, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(task_2_svd_1.T, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(task_2_svd_2.T, full_matrices = True)
    
     #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(task_3_svd_1.T, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 3 second half
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(task_3_svd_2.T, full_matrices = True)
     
    #Compare Within
    s_task_1 = np.linalg.multi_dot([u_t1_1.T, task_1_svd_2.T, vh_t1_1.T])
    s_1 = s_task_1.diagonal()     
    sum_within_1 = np.cumsum(abs(s_1))/task_1_svd_2.T.shape[0]
    
    s_task_2 = np.linalg.multi_dot([u_t2_1.T, task_2_svd_2.T, vh_t2_1.T])
    s_2 = s_task_2.diagonal()     
    sum_within_2 = np.cumsum(abs(s_2))/task_1_svd_2.T.shape[0]
    
    s_task_3 = np.linalg.multi_dot([u_t3_1.T, task_3_svd_2.T, vh_t3_1.T])
    s_3 = s_task_3.diagonal()     
    sum_within_3 = np.cumsum(abs(s_3))/task_1_svd_2.T.shape[0]
   
    #Compare Between
    s_task_1_2_ = np.linalg.multi_dot([u_t1_2.T, task_2_svd_1.T, vh_t1_2.T])
    s_1_2 = s_task_1_2_.diagonal()     
    sum_c_between_1_2 = np.cumsum(abs(s_1_2))/task_1_svd_2.T.shape[0]

    s_task_2_3 = np.linalg.multi_dot([u_t2_2.T, task_3_svd_1.T, vh_t2_2.T])
    s_2_3 = s_task_2_3.diagonal()     
    sum_c_between_1_2 = np.cumsum(abs(s_2_3))/task_1_svd_2.T.shape[0]

    within = np.mean([sum_within_1, sum_within_2,sum_within_3], 0)
    between = np.mean([sum_c_between_1_2, sum_within_2,sum_c_between_1_2], 0)
    
    plt.figure(1)
    plt.plot(within,  color = c, label = t + ' Within')
    plt.plot(between, color = c,linestyle = '--', label = t + ' Between')
    plt.legend()
    
    fig = plt.figure()
    for i in range(6):
        fig.add_subplot(3,2, i+1)
        plt.plot(vh_t1_2[i],color = c, linestyle = 'dotted', label = 'Task 1 blocks 1,2 ' +t)
        #plt.plot(vh_t1_2[i],color = c, linestyle = 'dashdot', label = 'Task 1 blocks 3,4')
        
        plt.plot(vh_t2_1[i],color = c, linestyle = 'solid',  label = 'Task 2 blocks 1,2 '+t)
       # plt.plot(vh_t2_2[i],color = c, linestyle = 'solid',  label = 'Task 2 blocks 3,4')

        plt.vlines(15, min(vh_t1_1[i]), max(vh_t1_1[i]),label = 'State Change')
        
    plt.legend()
    
    
def plotting():
    
    svd(Data_PFC, DM_PFC, 'red', 'PFC reward', 15, 42,52)
    svd(Data_PFC, DM_PFC, 'pink', 'PFC choice', 15, 30,40)
    svd(Data_PFC, DM_PFC, 'darkred', 'PFC initiation', 15, 20,30)


    svd(Data_HP, DM_HP, 'blue', 'HP reward', 15, 42,52)
    svd(Data_HP, DM_HP, 'lightblue', 'HP choice', 15, 30,40)
    svd(Data_HP, DM_HP, 'darkblue', 'HP initiation', 15, 20,30)

    