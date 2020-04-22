#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:02:23 2020

@author: veronikasamborska
"""

## Warping code for interpolating firing rates on neighbouring trials

import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
import utility as ut 
from palettable import wesanderson as wes
from scipy.special import erf
from collections import OrderedDict
import regression_function as reg_f
import regressions as re
import seaborn as sns
from palettable import wesanderson as wes
import align_trial_choices_rewards as ch_rew_align

def corr_betwee():
    
    C_1_a_HP,C_2_a_HP,C_3_a_HP = regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True)
    C_1_b_HP,C_2_b_HP,C_3_b_HP = regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True)
      
    C_1_a_PFC,C_2_a_PFC,C_3_a_PFC = regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
    C_1_b_PFC,C_2_b_PFC,C_3_b_PFC = regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
   
    
    C_HP_a = np.nanmean([C_1_a_HP,C_2_a_HP,C_3_a_HP],0)
    C_HP_b = np.nanmean([C_1_b_HP,C_2_b_HP,C_3_b_HP],0)
    nan_a = ~np.isnan(C_HP_a[2,:]) & ~np.isinf(C_HP_a[2,:])
    nan_b = ~np.isnan(C_HP_b[2,:]) & ~np.isinf(C_HP_b[2,:])
    C_HP_a = C_HP_a[:,nan_a & nan_b]
    C_HP_b = C_HP_b[:,nan_a & nan_b]
   
    C_PFC_a = np.nanmean([C_1_a_PFC,C_2_a_PFC,C_3_a_PFC],0)
    C_PFC_b = np.nanmean([C_1_b_PFC,C_2_b_PFC,C_3_b_PFC],0)
    nan_a = ~np.isnan(C_PFC_a[2,:]) & ~np.isinf(C_PFC_a[2,:])
    nan_b = ~np.isnan(C_PFC_b[2,:]) & ~np.isinf(C_PFC_b[2,:])
    C_PFC_a = C_PFC_a[:,nan_a & nan_b]
    C_PFC_b = C_PFC_b[:,nan_a & nan_b]
   
    pred = 2
    color = 'lightblue'
    plt.figure(figsize = (3,5))
    plt.subplot(2,1,1)
    sns.regplot(C_HP_a[pred], C_HP_b[pred], color = color, label = str(np.around(np.corrcoef(C_HP_a[pred], C_HP_b[pred])[0,1],3)))
    plt.legend()
    plt.xlabel('Block B')
    plt.ylabel('Block A')

    plt.title('HP')


   
    color = 'pink'
    plt.subplot(2,1,2)
    sns.regplot(C_PFC_a[pred], C_PFC_b[pred], color = color, label = str(np.around(np.corrcoef(C_PFC_a[pred], C_PFC_b[pred])[0,1],3)))
    plt.legend()
    plt.xlabel('Block B')
    plt.ylabel('Block A')

    sns.despine()
    plt.title('PFC')
    plt.tight_layout()


    C_1_a_HP,C_2_a_HP,C_3_a_HP = regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True)
    C_1_b_HP,C_2_b_HP,C_3_b_HP = regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True)
      
    C_1_a_PFC,C_2_a_PFC,C_3_a_PFC = regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
    C_1_b_PFC,C_2_b_PFC,C_3_b_PFC = regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
   
    
    C_HP_1 = np.nanmean([C_1_a_HP,C_1_b_HP],0)
    C_HP_2 = np.nanmean([C_2_a_HP,C_2_b_HP],0)
    C_HP_3 =  np.nanmean([C_3_a_HP,C_3_b_HP],0)
                         
    nan_1 = ~np.isnan(C_HP_1[2,:]) & ~np.isinf(C_HP_1[2,:])
    nan_2 = ~np.isnan(C_HP_2[2,:]) & ~np.isinf(C_HP_2[2,:])
    nan_3 = ~np.isnan(C_HP_3[2,:]) & ~np.isinf(C_HP_3[2,:])

    C_HP_1 = C_HP_1[:,nan_1 & nan_2 & nan_3]
    C_HP_2 = C_HP_2[:,nan_1 & nan_2 & nan_3]
    C_HP_3 = C_HP_3[:,nan_1 & nan_2 & nan_3]
    
    C_PFC_1 = np.nanmean([C_1_a_PFC,C_1_b_PFC],0)
    C_PFC_2 = np.nanmean([C_2_a_PFC,C_2_b_PFC],0)
    C_PFC_3 =  np.nanmean([C_3_a_PFC,C_3_b_PFC],0)
                         
    nan_1 = ~np.isnan(C_PFC_1[2,:]) & ~np.isinf(C_PFC_1[2,:])
    nan_2 = ~np.isnan(C_PFC_2[2,:]) & ~np.isinf(C_PFC_2[2,:])
    nan_3 = ~np.isnan(C_PFC_3[2,:]) & ~np.isinf(C_PFC_3[2,:])

    C_PFC_1 = C_PFC_1[:,nan_1 & nan_2 & nan_3]
    C_PFC_2 = C_PFC_2[:,nan_1 & nan_2 & nan_3]
    C_PFC_3 = C_PFC_3[:,nan_1 & nan_2 & nan_3]
          
    
    pred = 0
    color = 'pink'
    plt.subplot(2,3,1)
    sns.regplot(C_PFC_1[pred], C_PFC_2[pred], color = color, label = str(np.around(np.corrcoef(C_PFC_1[pred], C_PFC_2[pred])[0,1],3)))
    plt.legend()
    plt.title('Task 1 & 2')

    plt.subplot(2,3,2)
    sns.regplot(C_PFC_1[pred], C_PFC_3[pred], color = color, label = str(np.around(np.corrcoef(C_PFC_1[pred], C_PFC_3[pred])[0,1],3)))
    plt.legend()
    plt.title('Task 1 & 3')

    plt.subplot(2,3,3)
    sns.regplot(C_PFC_2[pred], C_PFC_3[pred], color = color, label = str(np.around(np.corrcoef(C_PFC_2[pred], C_PFC_3[pred])[0,1],3)))
    plt.legend()
    sns.despine()
    plt.title('Task 2 & 3')

    color = 'lightblue'

    plt.subplot(2,3,4)
    sns.regplot(C_HP_1[pred], C_HP_2[pred], color = color, label = str(np.around(np.corrcoef(C_HP_1[pred], C_HP_2[pred])[0,1],3)))
    plt.legend()
    plt.title('Task 1 & 2')

    plt.subplot(2,3,5)
    sns.regplot(C_HP_1[pred], C_HP_3[pred], color = color, label = str(np.around(np.corrcoef(C_HP_1[pred], C_HP_3[pred])[0,1],3)))
    plt.legend()
    plt.title('Task 1 & 3')

    plt.subplot(2,3,6)
    sns.regplot(C_HP_2[pred], C_HP_3[pred], color = color, label = str(np.around(np.corrcoef(C_HP_2[pred], C_HP_3[pred])[0,1],3)))
    plt.legend()
    sns.despine()
    plt.title('Task 2 & 3')
    plt.tight_layout()


   
def corr_between_C():
    C_1_a_HP,C_2_a_HP,C_3_a_HP = regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True)
    C_1_b_HP,C_2_b_HP,C_3_b_HP = regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True)
      
    C_1_a_PFC,C_2_a_PFC,C_3_a_PFC = regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
    C_1_b_PFC,C_2_b_PFC,C_3_b_PFC = regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
   
    #Bs
    pred = 2

    block_1_nan = ~np.isnan(C_1_b_HP[2,:]) & ~np.isinf(C_1_b_HP[2,:])
    block_2_nan = ~np.isnan(C_2_b_HP[2,:]) & ~np.isinf(C_2_b_HP[2,:])
    block_3_nan = ~np.isnan(C_3_b_HP[2,:]) & ~np.isinf(C_3_b_HP[2,:])
    
    C_1_b_HP = C_1_b_HP[:,block_1_nan & block_2_nan & block_3_nan]
    C_2_b_HP = C_2_b_HP[:,block_1_nan & block_2_nan & block_3_nan]
    C_3_b_HP = C_3_b_HP[:,block_1_nan & block_2_nan & block_3_nan]

    block_1_nan = ~np.isnan(C_1_b_PFC[2,:]) & ~np.isinf(C_1_b_PFC[2,:])
    block_2_nan = ~np.isnan(C_2_b_PFC[2,:]) & ~np.isinf(C_2_b_PFC[2,:])
    block_3_nan = ~np.isnan(C_3_b_PFC[2,:]) & ~np.isinf(C_3_b_PFC[2,:])
    
    C_1_b_PFC= C_1_b_PFC[:,block_1_nan & block_2_nan & block_3_nan]
    C_2_b_PFC = C_2_b_PFC[:,block_1_nan & block_2_nan & block_3_nan]
    C_3_b_PFC = C_3_b_PFC[:,block_1_nan & block_2_nan & block_3_nan]

    #  
    color = 'lightblue'
    # HP 
    plt.figure()
    plt.title('B Block')

    plt.subplot(2,3,1)
    sns.regplot(C_1_b_HP[pred], C_2_b_HP[pred], color = color, label = str(np.around(np.corrcoef(C_1_b_HP[pred], C_2_b_HP[pred])[0,1],3)))
    plt.legend()

    plt.subplot(2,3,2)
    sns.regplot(C_1_b_HP[pred], C_3_b_HP[pred], color = color,label = str(np.around(np.corrcoef(C_1_b_HP[pred], C_3_b_HP[pred])[0,1],3)) )
    plt.legend()

    plt.subplot(2,3,3)
    sns.regplot(C_2_b_HP[pred], C_3_b_HP[pred], color = color ,label = str(np.around(np.corrcoef(C_2_b_HP[pred], C_3_b_HP[pred])[0,1],3)))
    plt.legend()

    color = 'pink'
    plt.subplot(2,3,4)
    sns.regplot(C_1_b_PFC[pred], C_2_b_PFC[pred], color = color, label = str(np.around(np.corrcoef(C_1_b_PFC[pred], C_2_b_PFC[pred])[0,1],3)) )
    plt.legend()

    plt.subplot(2,3,5)
    sns.regplot(C_1_b_PFC[pred], C_3_b_PFC[pred], color = color,label = str(np.around(np.corrcoef(C_1_b_PFC[pred], C_3_b_PFC[pred])[0,1],3)) )
    plt.legend()

    plt.subplot(2,3,6)
    sns.regplot(C_2_b_PFC[pred], C_3_b_PFC[pred], color = color,label = str(np.around(np.corrcoef(C_2_b_PFC[pred], C_3_b_PFC[pred])[0,1],3)) )
    plt.legend()

    sns.despine()




    block_1_nan = ~np.isnan(C_1_a_HP[2,:]) & ~np.isinf(C_1_a_HP[2,:])
    block_2_nan = ~np.isnan(C_2_a_HP[2,:]) & ~np.isinf(C_2_a_HP[2,:])
    block_3_nan = ~np.isnan(C_3_a_HP[2,:]) & ~np.isinf(C_3_a_HP[2,:])
    
    C_1_a_HP = C_1_a_HP[:,block_1_nan & block_2_nan & block_3_nan]
    C_2_a_HP = C_2_a_HP[:,block_1_nan & block_2_nan & block_3_nan]
    C_3_a_HP = C_3_a_HP[:,block_1_nan & block_2_nan & block_3_nan]

    block_1_nan = ~np.isnan(C_1_a_PFC[2,:]) & ~np.isinf(C_1_a_PFC[2,:])
    block_2_nan = ~np.isnan(C_2_a_PFC[2,:]) & ~np.isinf(C_2_a_PFC[2,:])
    block_3_nan = ~np.isnan(C_3_a_PFC[2,:]) & ~np.isinf(C_3_a_PFC[2,:])
    
    C_1_a_PFC= C_1_a_PFC[:,block_1_nan & block_2_nan & block_3_nan]
    C_2_a_PFC = C_2_a_PFC[:,block_1_nan & block_2_nan & block_3_nan]
    C_3_a_PFC = C_3_a_PFC[:,block_1_nan & block_2_nan & block_3_nan]

    # As 
    color = 'lightblue'
    plt.figure()
    plt.title('B Block')

    # HP 
    plt.subplot(2,3,1)
    sns.regplot(C_1_a_HP[pred], C_2_a_HP[pred], color = color, label = str(np.around(np.corrcoef(C_1_a_HP[pred], C_2_a_HP[pred])[0,1],3)))
    plt.legend()

    plt.subplot(2,3,2)
    sns.regplot(C_1_a_HP[pred], C_3_a_HP[pred], color = color,label = str(np.around(np.corrcoef(C_1_a_HP[pred], C_3_a_HP[pred])[0,1],3)) )
    plt.legend()

    plt.subplot(2,3,3)
    sns.regplot(C_2_a_HP[pred], C_3_a_HP[pred], color = color ,label = str(np.around(np.corrcoef(C_2_a_HP[pred], C_3_a_HP[pred])[0,1],3)))
    plt.legend()

    color = 'pink'
    plt.subplot(2,3,4)
    sns.regplot(C_1_a_PFC[pred], C_2_a_PFC[pred], color = color, label = str(np.around(np.corrcoef(C_1_a_PFC[pred], C_2_a_PFC[pred])[0,1],3)) )
    plt.legend()

    plt.subplot(2,3,5)
    sns.regplot(C_1_a_PFC[pred], C_3_a_PFC[pred], color = color,label = str(np.around(np.corrcoef(C_1_a_PFC[pred], C_3_a_PFC[pred])[0,1],3)) )
    plt.legend()

    plt.subplot(2,3,6)
    sns.regplot(C_2_a_PFC[pred], C_3_a_PFC[pred], color = color,label = str(np.around(np.corrcoef(C_2_a_PFC[pred], C_3_a_PFC[pred])[0,1],3)) )
    plt.legend()

    sns.despine()
    
    
    
    
    ## Anti-correlations
    
    #Bs

    block_b_nan = ~np.isnan(C_1_b_HP[2,:]) & ~np.isinf(C_1_b_HP[2,:])
    block_a_nan = ~np.isnan(C_1_a_HP[2,:]) & ~np.isinf(C_1_a_HP[2,:])
    
    C_1_b_HP = C_1_b_HP[:,block_b_nan & block_a_nan]
    C_1_a_HP = C_1_a_HP[:,block_b_nan & block_a_nan]

    block_b_nan = ~np.isnan(C_1_b_PFC[2,:]) & ~np.isinf(C_1_b_PFC[2,:])
    block_a_nan = ~np.isnan(C_1_a_PFC[2,:]) & ~np.isinf(C_1_a_PFC[2,:])
    
    C_1_b_PFC = C_1_b_PFC[:,block_b_nan & block_a_nan]
    C_1_a_PFC = C_1_a_PFC[:,block_b_nan & block_a_nan]
    
    pred = 2

    #  
    color = 'lightblue'
    # HP 
    plt.figure()
    plt.title('B Block')

    plt.subplot(2,1,1)
    sns.regplot(C_1_b_HP[pred], C_1_a_HP[pred], color = color, label = str(np.around(np.corrcoef(C_1_b_HP[pred], C_1_a_HP[pred])[0,1],3)))
    plt.legend()


    color = 'pink'
    plt.subplot(2,1,2)
    sns.regplot(C_1_b_PFC[pred], C_1_a_PFC[pred], color = color, label = str(np.around(np.corrcoef(C_1_b_PFC[pred], C_1_a_PFC[pred])[0,1],3)) )
    plt.legend()
    sns.despine()



    block_b_nan = ~np.isnan(C_2_b_HP[2,:]) & ~np.isinf(C_2_b_HP[2,:])
    block_a_nan = ~np.isnan(C_2_a_HP[2,:]) & ~np.isinf(C_2_a_HP[2,:])
    
    C_2_b_HP = C_2_b_HP[:,block_b_nan & block_a_nan]
    C_2_a_HP = C_2_a_HP[:,block_b_nan & block_a_nan]

    block_b_nan = ~np.isnan(C_2_b_PFC[2,:]) & ~np.isinf(C_2_b_PFC[2,:])
    block_a_nan = ~np.isnan(C_2_a_PFC[2,:]) & ~np.isinf(C_2_a_PFC[2,:])
    
    C_2_b_PFC = C_2_b_PFC[:,block_b_nan & block_a_nan]
    C_2_a_PFC = C_2_a_PFC[:,block_b_nan & block_a_nan]
    

    #  
    color = 'lightblue'
    # HP 
    plt.figure()
    plt.title('B Block')

    plt.subplot(2,1,1)
    sns.regplot(C_2_b_HP[pred], C_2_a_HP[pred], color = color, label = str(np.around(np.corrcoef(C_2_b_HP[pred], C_2_a_HP[pred])[0,1],3)))
    plt.legend()


    color = 'pink'
    plt.subplot(2,1,2)
    sns.regplot(C_2_b_PFC[pred], C_2_a_PFC[pred], color = color, label = str(np.around(np.corrcoef(C_2_b_PFC[pred], C_2_a_PFC[pred])[0,1],3)) )
    plt.legend()
    sns.despine()
   

def regression_time_choices_rewards_a_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True):
    
    C_1 = []
    C_2 = []
    C_3 = []


    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = ch_rew_align.hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
    
    for s, session in enumerate(a_a_matrix_t_1_list):
        
        firing_rates_1 = a_a_matrix_t_1_list[s]
        firing_rates_2 = a_a_matrix_t_2_list[s]
        firing_rates_3 = a_a_matrix_t_3_list[s]
        n_neurons = firing_rates_1.shape[0]
        
        rewards_a_1 = a_a_matrix_t_1_list_rewards[s]
        rewards_a_2 = a_a_matrix_t_2_list_rewards[s]
        rewards_a_3 = a_a_matrix_t_3_list_rewards[s]
        #rewards = np.hstack([rewards_a_1,rewards_a_2,rewards_a_3])

        choices_a_1 = a_a_matrix_t_1_list_choices[s]
        choices_a_2 = a_a_matrix_t_2_list_choices[s]
        choices_a_3 = a_a_matrix_t_3_list_choices[s]
        #choices = np.hstack([choices_a_1,choices_a_2,choices_a_3])

        block_length = np.tile(np.arange(session.shape[1]/2),2)
       # trial_number = np.tile(block_length,3)
        ones_1 = np.ones(len(choices_a_1))
        ones_2 = np.ones(len(choices_a_2))
        ones_3 = np.ones(len(choices_a_3))

        # Task 1 
        
        predictors = OrderedDict([('Reward', rewards_a_1),
                                      ('Choice', choices_a_1),
                                      ('Trial Number',block_length),
                                      ('Constant', ones_1)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices_a_1),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C_1.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
        
        # Task 2
        predictors = OrderedDict([('Reward', rewards_a_2),
                                      ('Choice', choices_a_2),
                                      ('Trial Number',block_length),
                                      ('Constant', ones_2)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices_a_2),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C_2.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
       
        
        # Task 3
        predictors = OrderedDict([('Reward', rewards_a_3),
                                      ('Choice', choices_a_3),
                                      ('Trial Number',block_length),
                                      ('Constant', ones_3)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices_a_3),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C_3.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
       
     
    C_1 = np.concatenate(C_1,1)
    C_2 = np.concatenate(C_2,1)
    C_3 = np.concatenate(C_3,1)

    return C_1,C_2,C_3

def regression_time_choices_rewards_b_blocks_diff_tasks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True):
    
    C_1 = []
    C_2 = []
    C_3 = []


    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = ch_rew_align.hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
    
    for s, session in enumerate(b_b_matrix_t_1_list):
        
        firing_rates_1 = b_b_matrix_t_1_list[s]
        firing_rates_2 = b_b_matrix_t_2_list[s]
        firing_rates_3 = b_b_matrix_t_3_list[s]
        n_neurons = firing_rates_1.shape[0]
        
        rewards_b_1 = b_b_matrix_t_1_list_rewards[s]
        rewards_b_2 = b_b_matrix_t_2_list_rewards[s]
        rewards_b_3 = b_b_matrix_t_3_list_rewards[s]
        #rewards = np.hstack([rewards_a_1,rewards_a_2,rewards_a_3])

        choices_b_1 = b_b_matrix_t_1_list_choices[s]
        choices_b_2 = b_b_matrix_t_2_list_choices[s]
        choices_b_3 = b_b_matrix_t_3_list_choices[s]
        #choices = np.hstack([choices_a_1,choices_a_2,choices_a_3])

        block_length = np.tile(np.arange(session.shape[1]/2),2)
       # trial_number = np.tile(block_length,3)
        ones_1 = np.ones(len(choices_b_1))
        ones_2 = np.ones(len(choices_b_2))
        ones_3 = np.ones(len(choices_b_3))

        # Task 1 
        
        predictors = OrderedDict([('Reward', rewards_b_1),
                                      ('Choice', choices_b_1),
                                      ('Trial Number',block_length),
                                      ('Constant', ones_1)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices_b_1),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C_1.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
        
        # Task 2
        predictors = OrderedDict([('Reward', rewards_b_2),
                                      ('Choice', choices_b_2),
                                      ('Trial Number',block_length),
                                      ('Constant', ones_2)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices_b_2),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C_2.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
       
        
        # Task 3
        predictors = OrderedDict([('Reward', rewards_b_3),
                                      ('Choice', choices_b_3),
                                      ('Trial Number',block_length),
                                      ('Constant', ones_3)])
                
        
        X = np.vstack(predictors.values()).T[:len(choices_b_3),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X)
        
        C_3.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
       
     
    C_1 = np.concatenate(C_1,1)
    C_2 = np.concatenate(C_2,1)
    C_3 = np.concatenate(C_3,1)

    return C_1,C_2,C_3