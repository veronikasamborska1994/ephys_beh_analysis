#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:24:11 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection
from palettable import wesanderson as wes
import palettable
from collections import OrderedDict
import sys
import itertools
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

import heatmap_aligned as ha
import regression_function as reg_f
from sklearn.linear_model import LinearRegression
from scipy import io
import ephys_beh_import as ep

def load_data():
   
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
    HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
    experiment_aligned_PFC = ha.all_sessions_aligment(PFC, all_sessions)
    experiment_aligned_HP = ha.all_sessions_aligment(HP, all_sessions)
    data_HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    data_PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    return HP, PFC,experiment_aligned_PFC,experiment_aligned_HP


def plot():
    corr,stack_array = correlations(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    a_b_off_diag = corr[:,:80]
    starts_t1_b = a_b_off_diag[40:60,:40]
    ends_t1_b = a_b_off_diag[60:80,:40]
    
    norm_end_t1_b = ends_t1_b-starts_t1_b
    norm_start_t1_b = starts_t1_b-ends_t1_b


    starts_t2_b = a_b_off_diag[120:140,:40]
    ends_t2_b = a_b_off_diag[140:160,:40]
    norm_ends_t2_b = ends_t2_b-starts_t2_b
    norm_start_t2_b = starts_t2_b-ends_t2_b
    
    starts_t2_a = a_b_off_diag[80:100,:40]
    ends_t2_a = a_b_off_diag[100:120,:40]
    norm_end_t1_b_t2_a = ends_t2_a-starts_t2_a
    norm_start_t2_a = starts_t2_a-ends_t2_a

   
    mean_ends = (norm_end_t1_b + norm_ends_t2_b)/2
    mean_starts = (norm_start_t1_b  + norm_start_t2_b)/2
    plt.figure()
    plt.imshow(mean_ends)
    plt.colorbar()
    
    plt.yticks([5, 15],['B good Start', 'B good End'])
    plt.xticks([5,15, 25, 35],['A bad start', 'A bad end', 'A good start', 'A good end'])

    plt.figure()
    plt.imshow(mean_starts)
    plt.colorbar()
    
    
    plt.yticks([5, 15],['B bad Start', 'B bad End'])
    plt.xticks([5,15, 25, 35],['A bad start', 'A bad end', 'A good start', 'A good end'])

   
     
def _CPD(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd


def regression_choice_reward(data,experiment_aligned_data, start,end):
    
    dm = data['DM'][0]
    firing = data['Data'][0]

    res_list = []
    trials_since_block_list = [] 
    cpd = []
    
    for  s, sess in enumerate(dm):
        
        
        DM = dm[s]
        choices = DM[:,1]
        reward = DM[:,2]
        
        firing_rates_s = firing[s]
 
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]
        

        if len(ind_block) >= 12:
            
            trials_since_block = []
            t = 0
            
            #Bug in the state? 
            for st,s in enumerate(block):
                if block[st-1] != block[st]:
                    t = 0
                else:
                    t+=1
                trials_since_block.append(t)
                
            trials_since_block = np.asarray(trials_since_block[:ind_block[11]])
            ones = np.ones(len(trials_since_block))
            choices = choices[:len(trials_since_block)]
            rewards = reward[:len(trials_since_block)]
            firing_rates = np.mean(firing_rates_s[:len(trials_since_block), :,start:end],2)
            n_timepoints,n_neurons = firing_rates.shape
            predictors_all = OrderedDict([#('Time', trials_since_block),
                                            ('Reward', rewards),
                                            ('Choice', choices),
                                            ('ones', ones)])
          
                    
            X = np.vstack(predictors_all.values()).T[:len(ones),:].astype(float)
            y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            
            pdes = np.linalg.pinv(X)

            pe = np.matmul(pdes,y)
            res = y - np.matmul(X,pe)

            res_list.append(res)
            trials_since_block_list.append(trials_since_block)
            
            ind_last_10 = np.where(trials_since_block == 0)[0]
            ind_last_10 = np.append(ind_last_10,len(trials_since_block))  
            minimum_rev = np.min(np.diff(ind_last_10-1))
            if minimum_rev > 5:
                ind_below_10 = np.where(trials_since_block < 10)
                firing_first_10 = firing_rates[ind_below_10]
                ones = ones[ind_below_10]
                choices = choices[ind_below_10]
                rewards = reward[ind_below_10]
                trials_since_block = trials_since_block[ind_below_10]
      
            
                predictors_cpd  = OrderedDict([('Time', trials_since_block),
                                                ('Reward', rewards),
                                                ('Choice', choices),
                                                ('ones', ones)])
              
                        
                X = np.vstack(predictors_cpd.values()).T[:len(ones),:].astype(float)
                y = firing_first_10.reshape([len(firing_first_10),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            
                           
                cpd.append(_CPD(X,y))


            
    cpd = np.mean(np.concatenate(cpd,0),0)
    return res_list,trials_since_block_list,cpd,predictors_cpd
#HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
#PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
  
def corr_m(data,experiment_aligned_data,fig_n_imshow,start,end):
    
    res_list,trials_since_block_list,cpd,predictors_all = regression_choice_reward(data,experiment_aligned_data, start,end)
    plt.figure(10)
    plt.bar(list(predictors_all.keys())[:-1],cpd[:-1], color = 'black')
    stack_array = []
    
    for s in range(len(res_list)):
        session_firing = res_list[s]
        session_trials = np.asarray(trials_since_block_list[s])
        ind_last_10 = np.where(session_trials == 0)[0]
        ind_last_10 = np.append(ind_last_10,len(session_trials))  
        minimum_rev = np.min(np.diff(ind_last_10-1))
        if minimum_rev > 20:
            ind_below_10 = np.where(session_trials < 10)
            session_firing_cut = session_firing[ind_below_10]
         
            index_last_10_list = []
            for i in ind_last_10[1:]:
                index = np.arange(i-10,i)
                index_last_10_list.append(index) 
                
                
            merged = np.asarray(list(itertools.chain(*index_last_10_list)))
            session_first_10 = session_firing[merged]
            both_start_end = np.vstack((session_firing_cut,session_first_10))
            
            if len(both_start_end) == 240:
                stack_array.append(both_start_end)

    all_conc = np.concatenate(stack_array,1)
    length = 10
    _1_st = all_conc[:length]
    _2_st = all_conc[length:length*2]
    _3_st = all_conc[length*2:length*3]
    _4_st = all_conc[length*3:length*4]
    _5_st = all_conc[length*4:length*5]
    _6_st = all_conc[length*5:length*6]
    _7_st = all_conc[length*6:length*7]
    _8_st = all_conc[length*7:length*8]
    _9_st = all_conc[length*8:length*9]
    _10_st = all_conc[length*9:length*10]
    _11_st = all_conc[length*10:length*11]
    _12_st = all_conc[length*11:length*12]

    mean_task_1_st = np.mean([_1_st,_2_st,_3_st,_4_st], axis = 0 )
    mean_task_2_st = np.mean([_5_st,_6_st,_7_st,_8_st], axis = 0)
    mean_task_3_st = np.mean([_9_st,_10_st,_11_st,_12_st], axis = 0)
   
    
    length = 10
    _1_end = all_conc[length*12:length*13]
    _2_end = all_conc[length*13:length*14]
    _3_end = all_conc[length*14:length*15]
    _4_end = all_conc[length*15:length*16]
    _5_end = all_conc[length*16:length*17]
    _6_end = all_conc[length*17:length*18]
    _7_end = all_conc[length*18:length*19]
    _8_end = all_conc[length*19:length*20]
    _9_end = all_conc[length*20:length*21]
    _10_end = all_conc[length*21:length*22]
    _11_end = all_conc[length*22:length*23]
    _12_end = all_conc[length*23:length*24]

    mean_task_1_end = np.mean([_1_end,_2_end,_3_end,_4_end], axis = 0 )
    mean_task_2_end = np.mean([_5_end,_6_end,_7_end,_8_end], axis = 0)
    mean_task_3_end = np.mean([_9_end,_10_end,_11_end,_12_end], axis = 0)
   
    task_start_end = np.concatenate((mean_task_1_st,mean_task_2_st,mean_task_3_st,\
                                     mean_task_1_end,mean_task_2_end,mean_task_3_end))

    corr  = np.corrcoef(task_start_end)
   
    plt.figure(fig_n_imshow)
    plt.imshow(corr)
    plt.yticks(np.arange(0,60,10),['Start','2','3','End','2','3'])
  
    plt.xticks(np.arange(0,60,10),['Start','2','3','End','2','3'], rotation = 90)

    plt.colorbar()
   
    
def corr_m_ends_only(data,experiment_aligned_data, title,fig_n_imshow,fig_n_diag,c):
    res_list,trials_since_block_list = regression_choice_reward(data,experiment_aligned_data)
    stack_array = []
    for s in range(len(res_list)):
        session_firing = res_list[s]
        session_trials = np.asarray(trials_since_block_list[s])
        ind_last_10 = np.where(session_trials == 0)[0]
        ind_last_10 = np.append(ind_last_10,len(session_trials))  
        minimum_rev = np.min(np.diff(ind_last_10-1))

        if minimum_rev > 10: 
            index_last_10_list = []
            for i in ind_last_10[1:]:
                index = np.arange(i-10,i)
                index_last_10_list.append(index) 
                    
                    
            merged = np.asarray(list(itertools.chain(*index_last_10_list)))
            session_first_10 = session_firing[merged]
                
            if len(session_first_10) == 120:
                stack_array.append(session_first_10)

    all_conc = np.concatenate(stack_array,1)
    length = 10
    _1 = all_conc[:length]
    _2 = all_conc[length:length*2]
    _3 = all_conc[length*2:length*3]
    _4 = all_conc[length*3:length*4]
    _5 = all_conc[length*4:length*5]
    _6 = all_conc[length*5:length*6]
    _7 = all_conc[length*6:length*7]
    _8 = all_conc[length*7:length*8]
    _9 = all_conc[length*8:length*9]
    _10 = all_conc[length*9:length*10]
    _11 = all_conc[length*10:length*11]
    _12 = all_conc[length*11:length*12]

    mean_task_1 = np.mean([_1,_2,_3,_4], axis = 0 )
    mean_task_2 = np.mean([_5,_6,_7,_8], axis = 0)
    mean_task_3 = np.mean([_9,_10,_11,_12], axis = 0)
    task_10 = np.concatenate((mean_task_1,mean_task_2,mean_task_3))
    corr  = np.corrcoef(task_10)
   
    plt.figure(fig_n_imshow)
    plt.imshow(corr)

    #plt.yticks(np.arange(0,120,10),['End','2','3','4','5','6','7','8','9','10','11','12'])
  
    #plt.xticks(np.arange(0,120,10),['End','2','3','4','5','6','7','8','9','10','11','12'], rotation = 90)

    plt.colorbar()
    plt.title('Ends 10' +' '+ title)
    off_diag = np.diagonal(corr,10,0)
    plt.figure(fig_n_diag)
    plt.plot(off_diag, color = c)

    
def corr_m_starts_only(data,experiment_aligned_data, title,fig_n_imshow,fig_n_diag,c):
    res_list,trials_since_block_list = regression_choice_reward(data,experiment_aligned_data)
    stack_array = []
    for s in range(len(res_list)):
        session_firing = res_list[s]
        session_trials = np.asarray(trials_since_block_list[s])
        ind_last_10 = np.where(session_trials == 0)[0]
        ind_last_10 = np.append(ind_last_10,len(session_trials))  
        minimum_rev = np.min(np.diff(ind_last_10-1))
        if minimum_rev > 10:
            ind_below_10 = np.where(session_trials < 10)
            session_trials_cut = session_trials[ind_below_10]
            session_firing_cut = session_firing[ind_below_10]
        
            if len(session_trials_cut) == 120:
                stack_array.append(session_firing_cut)

    all_conc = np.concatenate(stack_array,1)
    corr  = np.corrcoef(all_conc)
    length = 10
    _1 = all_conc[:length]
    _2 = all_conc[length:length*2]
    _3 = all_conc[length*2:length*3]
    _4 = all_conc[length*3:length*4]
    _5 = all_conc[length*4:length*5]
    _6 = all_conc[length*5:length*6]
    _7 = all_conc[length*6:length*7]
    _8 = all_conc[length*7:length*8]
    _9 = all_conc[length*8:length*9]
    _10 = all_conc[length*9:length*10]
    _11 = all_conc[length*10:length*11]
    _12 = all_conc[length*11:length*12]

    mean_task_1 = np.mean([_1,_2,_3,_4], axis = 0 )
    mean_task_2 = np.mean([_5,_6,_7,_8], axis = 0)
    mean_task_3 = np.mean([_9,_10,_11,_12], axis = 0)
    task_10 = np.concatenate((mean_task_1,mean_task_2,mean_task_3))
    corr  = np.corrcoef(task_10)
   
    plt.figure(fig_n_imshow)
    plt.imshow(corr)
    
    # plt.figure()
    # plt.imshow(corr)
    # plt.yticks(np.arange(0,120,10),['Start','2','3','4','5','6','7','8','9','10','11','12'])
  
    # plt.xticks(np.arange(0,120,10),['Start','2','3','4','5','6','7','8','9','10','11','12'], rotation = 90)

    plt.colorbar()
    plt.title('Starts 10' +' '+ title)
    off_diag = np.diagonal(corr,10,0)
    plt.figure(fig_n_diag)
    plt.plot(off_diag, color = c)

        
  
def correlations(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):    
    
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
        
        ## Last 10
        
        task_1_a_bad_l = np.mean(firing_rates_mean_time[task_1_a_bad[-10:]],axis = 2) 
        task_1_a_good_l = np.mean(firing_rates_mean_time[task_1_a_good[-10:]],axis = 2) 

        task_1_b_bad_l = np.mean(firing_rates_mean_time[task_1_b_bad[-10:]],axis = 2) 
        task_1_b_good_l = np.mean(firing_rates_mean_time[task_1_b_good[-10:]],axis = 2) 
        
        task_2_a_bad_l = np.mean(firing_rates_mean_time[task_2_a_bad[-10:]],axis = 2) 
        task_2_a_good_l = np.mean(firing_rates_mean_time[task_2_a_good[-10:]],axis = 2) 

        task_2_b_bad_l = np.mean(firing_rates_mean_time[task_2_b_bad[-10:]],axis = 2) 
        task_2_b_good_l = np.mean(firing_rates_mean_time[task_2_b_good[-10:]],axis = 2) 
        
        stack_first_last = np.vstack((task_1_a_bad_f,task_1_a_bad_l, task_1_a_good_f,task_1_a_good_l,task_1_b_bad_f,task_1_b_bad_l,\
                                      task_1_b_good_f,task_1_b_good_l,task_2_a_bad_f,task_2_a_bad_l,\
                                          task_2_a_good_f,task_2_a_good_l,task_2_b_bad_f,task_2_b_bad_l,task_2_b_good_f,task_2_b_good_l))
        print(stack_first_last.shape)
        
        if stack_first_last.shape[0] == 160:
            stack_array.append(stack_first_last)
            
    all_conc = np.concatenate(stack_array,1)
    corr  = np.corrcoef(all_conc)
   
    plt.figure()
    plt.imshow(corr)
    plt.xticks(np.arange(0,160,10),['A bad T1 Start', 'A bad T1 End', 'A good T1 Start', 'A good T1 End', 'B bad T1 Start','B bad T1 End',\
                                   'B good T1 Start','B good T1 End', 'A bad T2 Start', ' bad T2 End','A good T2 Start', 'A good T2 End', 'B bad T2 Start', 'B bad T2 End',\
                                       'B good T2 Start','B good T2 End'], rotation=90)
        
    return corr,stack_array


def pca_time(corr):
    stack_array,corr = correlations(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    all_conc = np.concatenate(stack_array,1)

    u,s,v = np.linalg.svd(all_conc.T)
    
    t_v = np.transpose(v) 
    
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.plot(t_v[:,i])
        
    proj_v =  np.linalg.multi_dot((t_v,all_conc))
    
    PCs = proj_v.T[:5,:]
    
    fig = plt.figure(1, figsize=[14,12], clear=True)
    ax3D_PFC = plt.subplot2grid([3,3], [0,0], rowspan=3, colspan=2, projection='3d')
    ax2Da_PFC = fig.add_subplot(3, 3, 3)
    ax2Db_PFC = fig.add_subplot(3, 3, 6)
    ax2Dc_PFC = fig.add_subplot(3, 3, 9)
    
    
    # First 20 
    #3D plot
    x = PCs[0,:20]
    y = PCs[1,:20]
    z = PCs[2,:20]
    n = 21 # number of data points



    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap

    colors=[cmap(float(ii)/(n-1)) for ii in range(n-1)]
    
    for i in range(n-1):
        
        if i < 10:
            ax3D_PFC.scatter(x[i], y[i], z[i], color=colors[15])
           
        elif i > 10:
            ax3D_PFC.scatter(x[i], y[i], z[i], color=colors[1])
        
        
    for i in range(n-1):  
        if i < 10:
            ax2Da_PFC.scatter(x[i], y[i],color=colors[15],s = 10)
            ax2Db_PFC.scatter(x[i], z[i],color=colors[15],s = 10)
            ax2Dc_PFC.scatter(y[i], z[i],color=colors[15], s = 10)
        elif i > 10:
            ax2Da_PFC.scatter(x[i], y[i],color=colors[1],s = 10)
            ax2Db_PFC.scatter(x[i], z[i],color=colors[1],s = 10)
            ax2Dc_PFC.scatter(y[i], z[i],color=colors[1], s = 10)
       
    x = PCs[0,40:60]
    y = PCs[1,40:60]
    z = PCs[2,40:60]
    n = 21 # number of data points
  
    
    for i in range(n-1):
        if i < 10:
            ax3D_PFC.scatter(x[i], y[i], z[i], color=colors[15])
           
        elif i > 10:
            ax3D_PFC.scatter(x[i], y[i], z[i], color=colors[1])
          

           
        
    for i in range(n-1):  
        if i < 10:
            ax2Da_PFC.scatter(x[i], y[i],color=colors[15],s = 10)
            ax2Db_PFC.scatter(x[i], z[i],color=colors[15],s = 10)
            ax2Dc_PFC.scatter(y[i], z[i],color=colors[15], s = 10)
        elif i >10:
            ax2Da_PFC.scatter(x[i], y[i],color=colors[1],s = 10)
            ax2Db_PFC.scatter(x[i], z[i],color=colors[1],s = 10)
            ax2Dc_PFC.scatter(y[i], z[i],color=colors[1], s = 10)
       
      