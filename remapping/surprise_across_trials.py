#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:11:09 2020

@author: veronikasamborska
"""

import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm
from scipy import io
from palettable import wesanderson as wes
import seaborn as sns
import remap_time_fix as rtf
from  statsmodels.stats.anova import AnovaRM
import pingouin as pg
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel as ttest_rel
from tqdm import tqdm
import palettable
import scipy
from scipy.ndimage import gaussian_filter1d

font = {'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)


def run():
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    data_plot = io.loadmat('/Users/veronikasamborska/Desktop/plotDat.mat')

def trials_surprise(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):    
    
    # y = data['DM'][0]
    # x = data['Data'][0]
    x,y = rtf.remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)

    surprise_list_neurons_a_a = []
    surprise_list_neurons_b_b = []
    surprise_list_neurons_a_a_diff = []
    surprise_list_neurons_b_b_diff = []
           
    ind_pre = 16
    ind_post = 20
    
    #A_ind_pre_sw=Aind_pre_sw(end-ntrials-baselength:end);

    for  s, sess in enumerate(x):
        DM = y[s]
       
        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
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
        
        task_1_a = np.where((taskid == taskid_1) & (choices == 1))[0] # Find indicies for task 1 A
        task_1_a_pre_baseline = task_1_a[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
        task_1_a_pre  = task_1_a[-ind_pre+10:] # Find indicies for task 1 A last 10 
        
        # Reverse
        task_1_a_pre_baseline_rev = task_1_a[-10:] # Find indicies for task 1 A last 10 
        task_1_a_pre_rev  = task_1_a[-ind_pre:-ind_pre+20] # Find indicies for task 1 A last 10 
       
        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_pre_baseline = task_1_b[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
        task_1_b_pre  = task_1_b[-ind_pre+10:] # Find indicies for task 1 A last 10 
     
        task_1_b_pre_baseline_rev = task_1_b[-10:] # Find indicies for task 1 A last 10 
        task_1_b_pre_rev  = task_1_b[-ind_pre:-ind_pre+20]# Find indicies for task 1 A last 10 
     
        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_post = task_2_b[:ind_post] # Find indicies for task 1 A last 10 

        task_2_b_post_rev_baseline = task_2_b[-10:] # Find indicies for task 1 A last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_post = task_2_a[:ind_post] # Find indicies for task 1 A last 10 

        task_2_a_post_rev_baseline = task_2_a[-10:] # Find indicies for task 1 A last 10 

        firing_rates_mean_time = x[s]
       
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        for neuron in range(n_neurons):
            
            n_firing = firing_rates_mean_time[:,neuron, :].T  # Firing rate of each neuron
            n_firing =  gaussian_filter1d(n_firing.astype(float),1,1)
           
            n_firing = n_firing.T
                 
            # Baseline
            task_1_mean_a = np.mean(n_firing[task_1_a_pre_baseline], axis = 0)
            task_1_std_a = np.std(n_firing[task_1_a_pre_baseline], axis = 0)   
           
            # Task 1 Mean rates on the first 20 B trials
            task_1_mean_b = np.mean(n_firing[task_1_b_pre_baseline], axis = 0)
            task_1_std_b = np.std(n_firing[task_1_b_pre_baseline], axis = 0)
            
            min_std = 2
            
            if (len(np.where(n_firing[task_1_a_pre] == 0)[0]) )== 0 and (len(np.where(n_firing[task_1_b_pre] == 0)[0]) == 0)\
                     and (len(np.where(n_firing[task_2_a_post] == 0)[0]) == 0) and (len(np.where(n_firing[task_2_b_post] == 0)[0]) == 0):
               
            #if (len(np.where(task_1_mean_a == 0)[0]) == 0 ) and (len(np.where(task_1_mean_b == 0)[0]) == 0):
                  
                a_within =  - norm.logpdf(n_firing[task_1_a_pre], task_1_mean_a, (task_1_std_a + min_std))
        
                b_within = - norm.logpdf(n_firing[task_1_b_pre], task_1_mean_b,(task_1_std_b + min_std))
        
                a_between = - norm.logpdf(n_firing[task_2_a_post], task_1_mean_a, (task_1_std_a + min_std))
        
                b_between = - norm.logpdf(n_firing[task_2_b_post], task_1_mean_b, (task_1_std_b + min_std))

            else:
                
                 a_within = np.zeros(n_firing[task_1_a_pre].shape); a_within[:] = np.NaN
                 b_within = np.zeros(n_firing[task_1_b_pre].shape); b_within[:] = np.NaN
                 a_between =  np.zeros(n_firing[task_2_a_post].shape); a_between[:] = np.NaN
                 b_between = np.zeros(n_firing[task_2_b_post].shape); b_between[:] = np.NaN
       
          
           
            surprise_array_a = np.concatenate([a_within, a_between], axis = 0)                   
            surprise_array_b = np.concatenate([b_within,b_between], axis = 0)         
               
         
            surprise_list_neurons_a_a.append(surprise_array_a)
            surprise_list_neurons_b_b.append(surprise_array_b)
            
     
    surprise_list_neurons_a_a = np.nanmean(np.asarray(surprise_list_neurons_a_a), axis = 0)
    surprise_list_neurons_b_b = np.nanmean(np.asarray(surprise_list_neurons_b_b), axis = 0)
    
           
    return surprise_list_neurons_b_b,surprise_list_neurons_a_a


def shuffle_block_start_trials(data, task_1_2 = False, task_2_3 = False, task_1_3 = False, n_perms = 5):
    
    x,y = rtf.remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)
    ind_pre = 31
    ind_post = 40
    n_count = 0
    
    surprise_list_neurons_a_a_p = []
    surprise_list_neurons_b_b_p = []
       
    for  s, sess in tqdm(enumerate(x)):
        DM = y[s]
        task = DM[:,5]
        surprise_list_neurons_a_a_perm = []
        surprise_list_neurons_b_b_perm = []
         
        for perm in range(n_perms):

            surprise_list_neurons_a_a = []
            surprise_list_neurons_b_b = []
       
            choices = DM[:,1]
            b_pokes = DM[:,7]
            a_pokes = DM[:,6]
            task = DM[:,5]
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
           
            #np.random.shuffle(taskid)
            taskid  = np.roll(task,np.random.randint(len(task)), axis=0)

            task_1_a = np.where((taskid == taskid_1) & (choices == 1))[0] # Find indicies for task 1 A
            task_1_a_pre_baseline = task_1_a[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
            task_1_a_pre  = task_1_a[-ind_pre+10:] # Find indicies for task 1 A last 10 
            # Reverse
            task_1_a_pre_baseline_rev = task_1_a[-10:] # Find indicies for task 1 A last 10 
            task_1_a_pre_rev  = task_1_a[-ind_pre:-ind_pre+20] # Find indicies for task 1 A last 10 
           
            task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
            task_1_b_pre_baseline = task_1_b[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
            task_1_b_pre  = task_1_b[-ind_pre+10:] # Find indicies for task 1 A last 10 
         
            task_1_b_pre_baseline_rev = task_1_b[-10:] # Find indicies for task 1 A last 10 
            task_1_b_pre_rev  = task_1_b[-ind_pre:-ind_pre+20]# Find indicies for task 1 A last 10 
         
            task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
            task_2_b_post = task_2_b[:ind_post] # Find indicies for task 1 A last 10 
    
            task_2_b_post_rev_baseline = task_2_b[-10:] # Find indicies for task 1 A last 10 
    
            task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
            task_2_a_post = task_2_a[:ind_post] # Find indicies for task 1 A last 10 
    
            task_2_a_post_rev_baseline = task_2_a[-10:] # Find indicies for task 1 A last 10 
    
            firing_rates_mean_time = x[s]
           
            n_trials, n_neurons, n_time = firing_rates_mean_time.shape
            
            for neuron in range(n_neurons):
                
                
                n_firing = firing_rates_mean_time[:,neuron, :].T  # Firing rate of each neuron
                n_firing =  gaussian_filter1d(n_firing.astype(float),2,1)
                n_firing = n_firing.T
           
                # n_firing_pre_init = np.mean(n_firing[:,:20],1)
                # n_firing_init = np.mean(n_firing[:,25:30],1)
                # n_firing_ch = np.mean(n_firing[:,36:41],1)
                # n_firing_rew = np.mean(n_firing[:,42:47],1)
                # n_firing = np.vstack([n_firing_pre_init,n_firing_init,n_firing_ch,n_firing_rew])
                # n_firing = n_firing.T
                # Baseline
                task_1_mean_a = np.mean(n_firing[task_1_a_pre_baseline], axis = 0)
                task_1_std_a = np.std(n_firing[task_1_a_pre_baseline], axis = 0)   
               
                # Task 1 Mean rates on the first 20 B trials
                task_1_mean_b = np.mean(n_firing[task_1_b_pre_baseline], axis = 0)
                task_1_std_b = np.std(n_firing[task_1_b_pre_baseline], axis = 0)
                
                min_std = 2
            
                if (len(np.where(n_firing[task_1_a_pre] == 0)[0]) )== 0 and (len(np.where(n_firing[task_1_b_pre] == 0)[0]) == 0)\
                     and (len(np.where(n_firing[task_2_a_post] == 0)[0]) == 0) and (len(np.where(n_firing[task_2_b_post] == 0)[0]) == 0)\
                     and (len(np.where(task_1_mean_a == 0)[0]) == 0 ) and (len(np.where(task_1_mean_b == 0)[0]) == 0):
                       
                            
                    a_within =  - norm.logpdf(n_firing[task_1_a_pre], task_1_mean_a, (task_1_std_a + min_std))
        
                    b_within = - norm.logpdf(n_firing[task_1_b_pre], task_1_mean_b,(task_1_std_b + min_std))
        
                    a_between = - norm.logpdf(n_firing[task_2_a_post], task_1_mean_a, (task_1_std_a + min_std))
        
                    b_between = - norm.logpdf(n_firing[task_2_b_post], task_1_mean_b, (task_1_std_b + min_std))

                else:
                     a_within = np.zeros(n_firing[task_1_a_pre].shape); a_within[:] = np.NaN
                     b_within = np.zeros(n_firing[task_1_b_pre].shape); b_within[:] = np.NaN
                     a_between =  np.zeros(n_firing[task_2_a_post].shape); a_between[:] = np.NaN
                     b_between = np.zeros(n_firing[task_2_b_post].shape); b_between[:] = np.NaN
    
                surprise_array_a = np.concatenate([a_within, a_between], axis = 0)                   
                surprise_array_b = np.concatenate([b_within,b_between], axis = 0)         
                 
                surprise_list_neurons_a_a.append(surprise_array_a)
                surprise_list_neurons_b_b.append(surprise_array_b)
                
                
            surprise_list_neurons_a_a_perm.append(abs(np.nanmean(surprise_list_neurons_a_a,0)[21] - np.asarray(np.nanmean(surprise_list_neurons_a_a,0)[20])))
            surprise_list_neurons_b_b_perm.append(abs(np.nanmean(surprise_list_neurons_b_b,0)[21] - np.asarray(np.nanmean(surprise_list_neurons_b_b,0)[20])))
            
        surprise_list_neurons_a_a_p.append(np.percentile(np.asarray(surprise_list_neurons_a_a_perm),95, axis = 0))
        surprise_list_neurons_b_b_p.append(np.percentile(np.asarray(surprise_list_neurons_b_b_perm),95, axis = 0))
        
        
    surprise_list_neurons_a_a_p = np.nanmean(surprise_list_neurons_a_a_p,0)
    surprise_list_neurons_b_b_p = np.nanmean(surprise_list_neurons_b_b_p,0)
    
    return surprise_list_neurons_a_a_p, surprise_list_neurons_b_b_p
         



            
def plot_heat_surprise(data_HP, data_PFC):
    
    mean_b_b_t1_2_HP, mean_a_a_t1_t2_HP  = trials_surprise(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_t2_3_HP, mean_a_a_t2_t3_HP = trials_surprise(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)  
    mean_b_b_t1_3_HP, mean_a_a_t1_3_HP = trials_surprise(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)

    mean_b_b_t1_2_PFC, mean_a_a_t1_t2_PFC = trials_surprise(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_t2_3_PFC, mean_a_a_t2_t3_PFC  = trials_surprise(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)  
    mean_b_b_t1_3_PFC, mean_a_a_t1_3_PFC = trials_surprise(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    
     
    n_perms = 10
    _perm_mean_b_b_t1_2_HP,  _perm_mean_a_a_t1_t2_HP = shuffle_block_start_trials(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False, n_perms = n_perms)
    _perm_mean_b_b_t2_3_HP,  _perm_mean_a_a_t2_t3_HP =  shuffle_block_start_trials(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False, n_perms = n_perms)  
    _perm_mean_b_b_t1_3_HP,  _perm_mean_a_a_t1_3_HP = shuffle_block_start_trials(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True, n_perms = n_perms)

    _perm_mean_b_b_t1_2_PFC,  _perm_mean_a_a_t1_t2_PFC = shuffle_block_start_trials(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False, n_perms = n_perms)
    _perm_mean_b_b_t2_3_PFC,  _perm_mean_a_a_t2_t3_PFC  = shuffle_block_start_trials(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False,n_perms = n_perms)  
    _perm_mean_b_b_t1_3_PFC,  _perm_mean_a_a_t1_3_PFC = shuffle_block_start_trials(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True,n_perms = n_perms)
   
    ch = np.arange(36,42)
    init = np.arange(20,25)
    all_inds = []
    
    diff_space_HP_perm_ch = _perm_mean_b_b_t1_2_HP.T
    diff_space_HP_diff_ch = abs(mean_b_b_t1_2_HP[21,:] - mean_b_b_t1_2_HP[20,:])
    all_inds.append(np.where(diff_space_HP_diff_ch > diff_space_HP_perm_ch)) #0
    
    same_space_HP_perm_ch = _perm_mean_a_a_t1_3_HP.T
    same_space_HP_diff_ch = abs(mean_a_a_t1_3_HP[21,:] - mean_a_a_t1_3_HP[20,:])
    all_inds.append(np.where(same_space_HP_diff_ch > same_space_HP_perm_ch)) #1

    b_init_space_HP_perm_ch =_perm_mean_b_b_t2_3_HP.T
    b_init_space_HP_diff_ch = abs(mean_b_b_t2_3_HP[21,:] - mean_b_b_t2_3_HP[20,:])
    all_inds.append(np.where(b_init_space_HP_diff_ch > b_init_space_HP_perm_ch)) #2
  
    
    diff_space_PFC_perm_ch = _perm_mean_b_b_t1_2_PFC.T
    diff_space_PFC_diff_ch = abs(mean_b_b_t1_2_PFC[21,:] - mean_b_b_t1_2_PFC[20,:])
    all_inds.append(np.where(diff_space_PFC_diff_ch > diff_space_PFC_perm_ch)) #3

    same_space_PFC_perm_ch = _perm_mean_a_a_t1_3_PFC.T
    same_space_PFC_diff_ch = abs(mean_a_a_t1_3_PFC[21,:] - mean_a_a_t1_3_PFC[20,:])
    all_inds.append(np.where(same_space_PFC_diff_ch > (same_space_PFC_perm_ch))) #3
  
    b_init_space_PFC_perm_ch = _perm_mean_b_b_t2_3_PFC.T
    b_init_space_PFC_diff_ch = abs(mean_b_b_t2_3_PFC[21,:] - mean_b_b_t2_3_PFC[20,:])
    all_inds.append(np.where(b_init_space_PFC_diff_ch > (b_init_space_PFC_perm_ch))) #5
    
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    fig1, axes1 = plt.subplots(nrows=4, ncols=3, figsize=(10,5))

    im = axes1[0,0].imshow(mean_b_b_t1_2_HP.T,cmap = cmap, aspect = 'auto')
    axes1[0,0].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[0,0].set_xticks( np.arange(0,mean_b_b_t1_2_HP.shape[0],2))
    axes1[0,0].set_xticklabels( np.arange(0,mean_b_b_t1_2_HP.shape[0],2)+1)
    axes1[0,0].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])

    clim=im.properties()['clim']
    axes1[0,1].imshow(mean_b_b_t1_3_HP.T ,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[0,1].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[0,1].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[0,1].set_xticks( np.arange(0,mean_b_b_t1_2_HP.shape[0],2))
    axes1[0,1].set_xticklabels( np.arange(0,mean_b_b_t1_2_HP.shape[0],2)+1)

    axes1[0,2].imshow(mean_b_b_t2_3_HP.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[0,2].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[0,2].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[0,2].set_xticks( np.arange(0,mean_b_b_t1_2_HP.shape[0],2))
    axes1[0,2].set_xticklabels( np.arange(0,mean_b_b_t1_2_HP.shape[0],2)+1)

    axes1[1,0].imshow(mean_b_b_t1_2_PFC.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[1,0].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[1,0].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[0,0].set_xticks( np.arange(0,mean_b_b_t1_2_HP.shape[0],2))
    axes1[0,0].set_xticklabels( np.arange(0,mean_b_b_t1_2_HP.shape[0],2)+1)

    axes1[1,1].imshow(mean_b_b_t1_3_PFC.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[1,1].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[1,1].set_xticks( np.arange(0,mean_b_b_t1_2_HP.shape[0],2))
    axes1[1,1].set_xticklabels( np.arange(0,mean_b_b_t1_2_HP.shape[0],2)+1)
    axes1[1,1].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
 
    axes1[1,2].imshow(mean_b_b_t2_3_PFC.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[1,2].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[1,2].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[1,2].set_xticks( np.arange(0,mean_b_b_t1_2_HP.shape[0],2))
    axes1[1,2].set_xticklabels( np.arange(0,mean_b_b_t1_2_HP.shape[0],2)+1)



    axes1[2,0].imshow(mean_a_a_t1_t2_HP.T ,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[2,0].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[2,0].set_xticks( np.arange(0,mean_a_a_t1_t2_HP.shape[0],2))
    axes1[2,0].set_xticklabels( np.arange(0,mean_a_a_t1_t2_HP.shape[0],2)+1)
    axes1[2,0].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])

    clim=im.properties()['clim']
    axes1[2,1].imshow(mean_a_a_t1_3_HP.T ,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[2,1].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[2,1].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[2,1].set_xticks( np.arange(0,mean_a_a_t1_3_HP.shape[0],2))
    axes1[2,1].set_xticklabels( np.arange(0,mean_a_a_t1_3_HP.shape[0],2)+1)

    axes1[2,2].imshow(mean_a_a_t2_t3_HP.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[2,2].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[2,2].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[2,2].set_xticks( np.arange(0,mean_a_a_t2_t3_HP.shape[0],2))
    axes1[2,2].set_xticklabels( np.arange(0,mean_a_a_t2_t3_HP.shape[0],2)+1)

    axes1[3,0].imshow(mean_a_a_t1_t2_PFC.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[3,0].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[3,0].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[3,0].set_xticks( np.arange(0,mean_a_a_t1_t2_PFC.shape[0],2))
    axes1[3,0].set_xticklabels( np.arange(0,mean_a_a_t1_t2_PFC.shape[0],2)+1)

    axes1[3,1].imshow(mean_a_a_t1_3_PFC.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[3,1].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[3,1].set_xticks( np.arange(0,mean_a_a_t1_3_PFC.shape[0],2))
    axes1[3,1].set_xticklabels( np.arange(0,mean_a_a_t1_3_PFC.shape[0],2)+1)
    axes1[3,1].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
 
    axes1[3,2].imshow(mean_a_a_t2_t3_PFC.T,cmap = cmap,clim=clim, aspect = 'auto')
    axes1[3,2].set_yticks([5,10,15,20,25,36,42,47, 52,57,62])
    axes1[3,2].set_yticklabels(['-800','-600','-400','-200','I','Ch','O','+200', '+400', '+600', '+800'])
    axes1[3,2].set_xticks( np.arange(0,mean_a_a_t2_t3_PFC.shape[0],2))
    axes1[3,2].set_xticklabels( np.arange(0,mean_a_a_t2_t3_PFC.shape[0],2)+1)

    fig1.colorbar(im, ax=axes1.ravel().tolist(), shrink=0.5)

    
    