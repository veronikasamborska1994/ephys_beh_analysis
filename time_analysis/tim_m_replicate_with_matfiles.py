#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:59:44 2020

@author: veronikasamborska
"""

import numpy as np
import pylab as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
import align_trial_choices_rewards as ch_rew_align
import seaborn as sns
from collections import OrderedDict
import regression_function as reg_f
from palettable import wesanderson as wes
from scipy import io
import statsmodels.api as sm
import seaborn as sns
from itertools import combinations 

pfc = io.loadmat('/Users/veronikasamborska/Desktop/code_for_data_session/data/trial_warped_data/PFC/aligned_on_behaviour/all_trial/PFC_aligned_on_beh_time_all_trial_time_raw_firing.mat')

hp = io.loadmat('/Users/veronikasamborska/Desktop/code_for_data_session/data/trial_warped_data/HP/aligned_on_behaviour/all_trial/HP_aligned_on_beh_time_all_trial_time_raw_firing.mat')

def _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True):
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = ch_rew_align.hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
    A1 = []; A2 = []; A3 = []; A4 = []; A5 = []; A6 = []
    B1 = []; B2 = []; B3 = []; B4 = []; B5 = []; B6 = []

    A1_rew = []; A2_rew = []; A3_rew = []; A4_rew = []; A5_rew = []; A6_rew = []
    B1_rew = []; B2_rew = []; B3_rew = []; B4_rew = []; B5_rew = []; B6_rew = []

    A1_ch = []; A2_ch = []; A3_ch = []; A4_ch = []; A5_ch = []; A6_ch = []
    B1_ch = []; B2_ch = []; B3_ch = []; B4_ch = []; B5_ch = []; B6_ch = []

    for s, session in enumerate(a_a_matrix_t_1_list):
        
        A1.append(a_a_matrix_t_1_list[s][:,:17]); A2.append(a_a_matrix_t_1_list[s][:,17:]); A3.append(a_a_matrix_t_2_list[s][:,:17]); A4.append(a_a_matrix_t_2_list[s][:,17:])
        A5.append(a_a_matrix_t_3_list[s][:,:17]); A6.append(a_a_matrix_t_3_list[s][:,17:])
        
        B1.append(b_b_matrix_t_1_list[s][:,:17]); B2.append(b_b_matrix_t_1_list[s][:,17:]); B3.append(b_b_matrix_t_2_list[s][:,:17]); B4.append(b_b_matrix_t_2_list[s][:,17:])
        B5.append(b_b_matrix_t_3_list[s][:,:17]); B6.append(b_b_matrix_t_3_list[s][:,17:])
        
        A1_rew.append(a_a_matrix_t_1_list_rewards[s][:17]); A2_rew.append(a_a_matrix_t_1_list_rewards[s][17:]); A3_rew.append(a_a_matrix_t_2_list_rewards[s][:17]); A4_rew.append(a_a_matrix_t_2_list_rewards[s][17:])
        A5_rew.append(a_a_matrix_t_3_list_rewards[s][:17]); A6_rew.append(a_a_matrix_t_3_list_rewards[s][17:])
        
        B1_rew.append(b_b_matrix_t_1_list_rewards[s][:17]); B2_rew.append(b_b_matrix_t_1_list_rewards[s][17:]); B3_rew.append(b_b_matrix_t_2_list_rewards[s][:17]); B4_rew.append(b_b_matrix_t_2_list_rewards[s][17:])
        B5_rew.append(b_b_matrix_t_3_list_rewards[s][:17]); B6_rew.append(b_b_matrix_t_3_list_rewards[s][17:])
        
        
        A1_ch.append(a_a_matrix_t_1_list_choices[s][:17]); A2_ch.append(a_a_matrix_t_1_list_choices[s][17:]); A3_ch.append(a_a_matrix_t_2_list_choices[s][:17]); A4_ch.append(a_a_matrix_t_2_list_choices[s][17:])
        A5_ch.append(a_a_matrix_t_3_list_choices[s][:17]); A6_ch.append(a_a_matrix_t_3_list_choices[s][17:])
        
        B1_ch.append(b_b_matrix_t_1_list_choices[s][:17]); B2_ch.append(b_b_matrix_t_1_list_choices[s][17:]); B3_ch.append(b_b_matrix_t_2_list_choices[s][:17]); B4_ch.append(b_b_matrix_t_2_list_choices[s][17:])
        B5_ch.append(b_b_matrix_t_3_list_choices[s][:17]); B6_ch.append(b_b_matrix_t_3_list_choices[s][17:])
        
       
        
    return A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,\
    B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,\
    A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch,\
    B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch


     
def regression_time_choices_rewards_a_blocks_reward(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= True):
    
    C_a = []
    C_b = []
    
    p_vals_a = []
    p_vals_b = []

    count_perm   = []
    comb, comb_1 = [6,6]


    for ind_a in combinations(range(comb + comb_1), comb):
        ind_b = [i for i in range(comb + comb_1) if i not in ind_a]
       
        A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
        neuron_count = 0
        perm_count = 0

        for s, session in enumerate(A1):
            A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
            B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
            
            A1_rew_s = A1_rew[s];  A2_rew_s= A2_rew[s]; A3_rew_s = A3_rew[s]; A4_rew_s = A4_rew[s];  A5_rew_s= A5_rew[s]; A6_rew_s = A6_rew[s]
            B1_rew_s= B1_rew[s];  B2_rew_s = B2_rew[s]; B3_rew_s = B3_rew[s]; B4_rew_s = B4_rew[s];  B5_rew_s = B5_rew[s]; B6_rew_s = B6_rew[s]
           
            A1_c_s = A1_ch[s];  A2_c_s= A2_ch[s]; A3_c_s = A3_rew[s]; A4_c_s = A4_rew[s];  A5_c_s= A5_rew[s]; A6_c_s = A6_rew[s]
            B1_c_s = B1_rew[s];  B2_c_s = B2_rew[s]; B3_c_s = B3_rew[s]; B4_c_s = B4_rew[s];  B5_c_s = B5_rew[s]; B6_c_s = B6_rew[s]
            for n,neuron in enumerate(A1_s):
                A_f = np.hstack([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]])
                B_f = np.hstack([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]])
                A_rewards = np.hstack([A1_rew_s,A2_rew_s,A3_rew_s,A4_rew_s,A5_rew_s,A6_rew_s])
                B_rewards = np.hstack([B1_rew_s,B2_rew_s,B3_rew_s,B4_rew_s,B5_rew_s,B6_rew_s])
    
                A_choices = np.hstack([A1_c_s,A2_c_s,A3_c_s,A4_c_s,A5_c_s,A6_c_s])
                B_choices = np.hstack([B1_c_s,B2_c_s,B3_c_s,B4_c_s,B5_c_s,B6_c_s])
    
                block_length = np.tile(np.arange(A1[s].shape[1]),6)
                ones = np.ones(len(block_length))
                
                predictors_a = OrderedDict([('Choice',A_choices),
                                            ('Reward',A_rewards),
                                            ('Trial Number',block_length),
                                            ('Constant', ones)])
                        
                
                X_a = np.vstack(predictors_a.values()).T[:len(block_length),:].astype(float)
        
                results = sm.OLS(A_f, X_a).fit()   
               
                C_a.append(results.tvalues[2]) # Predictor loadings
                p_vals_a.append(results.pvalues[2])
                
        
                predictors_b = OrderedDict([('Choice',B_choices),
                                            ('Reward',B_rewards),
                                            ('Trial Number',block_length),
                                            ('Constant', ones)])
                 
                   
                X_b = np.vstack(predictors_b.values()).T[:len(ones),:].astype(float)
                results = sm.OLS(B_f, X_b).fit()   
                p_vals_b.append(results.pvalues[2])
        
                C_b.append(results.tvalues[2]) # Predictor loadings
    
                if anti_corr:
                    if C_a[n] < 0:
                        if p_vals_a[n] < 0.05:
                            if C_b[n] > 0:
                                if p_vals_b[n] < 0.05:
                                    neuron_count += 1
                    elif C_a[n] > 0:
                        if p_vals_a[n] < 0.05:
                            if C_b[n] < 0:
                                if p_vals_b[n] < 0.05:
                                    neuron_count += 1            
                else:
                    if C_a[n] < 0:
                        if p_vals_a[n] < 0.05:
                            if C_b[n] < 0:
                                if p_vals_b[n] < 0.05:
                                    neuron_count += 1
                    elif C_a[n] > 0:
                        if p_vals_a[n] < 0.05:
                            if C_b[n] > 0:
                                if p_vals_b[n] < 0.05:
                                    neuron_count += 1    
                
                all_firing = np.vstack([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n],B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]])
               
                y_a = all_firing[ind_a,:]
                y_a = y_a.flatten()
                y_b  = all_firing[ind_b,:]
                y_b = y_b.flatten()
        
                C_perm_b = sm.OLS(y_b, X_b).fit()   
                C_perm_a = sm.OLS(y_a, X_a).fit()  
                
                if anti_corr:
                    if C_perm_a.params[2] < 0:
                        if C_perm_a.pvalues[2] < 0.05:
                            if C_perm_b.params[2] > 0:
                                if C_perm_b.pvalues[2] < 0.05:
                                    perm_count += 1
                    elif C_perm_a.params[2] > 0:
                        if C_perm_a.pvalues[2] < 0.05:
                            if C_perm_b.params[2] < 0:
                                if C_perm_b.pvalues[2] < 0.05:
                                    perm_count += 1
                        
                else:
                   if C_perm_a.params[2] < 0:
                        if C_perm_a.pvalues[2] < 0.05:
                            if C_perm_b.params[2] < 0:
                                if C_perm_b.pvalues[2] < 0.05:
                                    perm_count += 1
                   elif C_perm_a.params[2] > 0:
                        if C_perm_a.pvalues[2] < 0.05:
                            if C_perm_b.params[2] > 0:
                                if C_perm_b.pvalues[2] < 0.05:
                                    perm_count += 1
                        
                        
            
        count_perm.append(perm_count)
    return count_perm, neuron_count,C_a,C_b
       
def regression_time_choices_rewards_a_blocks(data_file, anti_corr= True):
    
    C_a = []
    C_b = []
    
    p_vals_a = []
    p_vals_b = []

    count_perm   = []
    data = []
    m, n = [6,6]

    for ind_a in combinations(range(m + n), m):
        ind_b = [i for i in range(m + n) if i not in ind_a]
              

        for key in ('A_Task_1_Block_1', 'A_Task_1_Block_2', 'A_Task_2_Block_1', 'A_Task_2_Block_2', 'A_Task_3_Block_1', 'A_Task_3_Block_2', 'B_Task_1_Block_1', 'B_Task_1_Block_2',\
        'B_Task_2_Block_1', 'B_Task_2_Block_2', 'B_Task_3_Block_1', 'B_Task_3_Block_2'):
            data.append(data_file[key])
        A = data[:6]
        B = data[6:]
        neuron_count = 0
        perm_count = 0

        for s, session in enumerate(A[0]):
            A1 = A[0][s];  A2 = A[1][s]; A3 = A[2][s]; A4 = A[3][s];  A5 = A[4][s]; A6 = A[5][s]
            B1 = B[0][s];  B2 = B[1][s]; B3 = B[2][s]; B4 = B[3][s];  B5 = B[4][s]; B6 = B[5][s]
                
            A_f = np.hstack([A1,A2,A3,A4,A5,A6])
            B_f = np.hstack([B1,B2,B3,B4,B5,B6])
    
            block_length = np.tile(np.arange(A1.shape[0]),6)
            ones = np.ones(len(block_length))
            
            predictors_a = OrderedDict([('Trial Number',block_length),
                                      ('Constant', ones)])
                    
            
            X_a = np.vstack(predictors_a.values()).T[:len(block_length),:].astype(float)
    
            results = sm.OLS(A_f, X_a).fit()   
           
            C_a.append(results.tvalues[0]) # Predictor loadings
            p_vals_a.append(results.pvalues[0])
            
    
            predictors_b = OrderedDict([('Trial Number',block_length),
                                          ('Constant', ones)])
            
               
            X_b = np.vstack(predictors_b.values()).T[:len(ones),:].astype(float)
            results = sm.OLS(B_f, X_b).fit()   
            p_vals_b.append(results.pvalues[0])
    
            C_b.append(results.tvalues[0]) # Predictor loadings

            if anti_corr:
                if C_a[s] < 0:
                    if p_vals_a[s] < 0.05:
                        if C_b[s] > 0:
                            if p_vals_b[s] < 0.05:
                                neuron_count += 1
                elif C_a[s] > 0:
                    if p_vals_a[s] < 0.05:
                        if C_b[s] < 0:
                            if p_vals_b[s] < 0.05:
                                neuron_count += 1            
            else:
                if C_a[s] < 0:
                    if p_vals_a[s] < 0.05:
                        if C_b[s] < 0:
                            if p_vals_b[s] < 0.05:
                                neuron_count += 1
                elif C_a[s] > 0:
                    if p_vals_a[s] < 0.05:
                        if C_b[s] > 0:
                            if p_vals_b[s] < 0.05:
                                neuron_count += 1    
            
            all_firing = np.vstack([A1,A2,A3,A4,A5,A6,B1,B2,B3,B4,B5,B6])
           
            y_a = all_firing[ind_a,:].flatten()
            y_b  = all_firing[ind_b,:].flatten()
    
            C_perm_b = sm.OLS(y_b, X_b).fit()   
            C_perm_a = sm.OLS(y_a, X_a).fit()  
            
            if anti_corr:
                if C_perm_a.params[0] < 0:
                    if C_perm_a.pvalues[0] < 0.05:
                        if C_perm_b.params[0] > 0:
                            if C_perm_b.pvalues[0] < 0.05:
                                perm_count += 1
                elif C_perm_a.params[0] > 0:
                    if C_perm_a.pvalues[0] < 0.05:
                        if C_perm_b.params[0] < 0:
                            if C_perm_b.pvalues[0] < 0.05:
                                perm_count += 1
                    
            else:
               if C_perm_a.params[0] < 0:
                    if C_perm_a.pvalues[0] < 0.05:
                        if C_perm_b.params[0] < 0:
                            if C_perm_b.pvalues[0] < 0.05:
                                perm_count += 1
               elif C_perm_a.params[0] > 0:
                    if C_perm_a.pvalues[0] < 0.05:
                        if C_perm_b.params[0] > 0:
                            if C_perm_b.pvalues[0] < 0.05:
                                perm_count += 1
                    
                    
        
        count_perm.append(perm_count)
    return count_perm, neuron_count,C_a,C_b

def plot():
     
    count_perm_anti_pfc, neuron_count_anti_pfc, C_a_anti_pfc, C_b_anti_pfc = regression_time_choices_rewards_a_blocks_reward(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= True)
    count_perm_anti_hp, neuron_count_anti_hp, C_a_anti_hp, C_b_anti_hp = regression_time_choices_rewards_a_blocks_reward(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= True)

    count_perm_same_pfc, neuron_count_same_pfc, C_a_same_pfc, C_b_same_pfc =regression_time_choices_rewards_a_blocks_reward(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= False)
    count_perm_same_hp, neuron_count_same_hp, C_a_same_hp, C_b_same_hp = regression_time_choices_rewards_a_blocks_reward(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= False)

    plt.subplot(2,2,1)
    plt.hist(count_perm_anti_pfc, 10,color = 'lightblue')      
    plt.vlines(neuron_count_anti_pfc, 0, np.max(np.histogram(count_perm_anti_pfc,10)[0]), color = 'black', label = 'data')
    plt.vlines(np.percentile(count_perm_anti_pfc,99), 0, np.max(np.histogram(count_perm_anti_pfc,10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
    plt.vlines(np.percentile(count_perm_anti_pfc,95), 0, np.max(np.histogram(count_perm_anti_pfc,10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
    plt.legend()
    plt.xlabel('# cells')
    plt.ylabel('Count')
    plt.title('PFC Anticorrelation')


    plt.subplot(2,2,2)
    plt.hist(count_perm_anti_hp, 10,color = 'pink')      
    plt.vlines(neuron_count_anti_hp, 0, np.max(np.histogram(count_perm_anti_hp,10)[0]), color = 'black', label = 'data')
    plt.vlines(np.percentile(count_perm_anti_hp,99), 0, np.max(np.histogram(count_perm_anti_hp,10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
    plt.vlines(np.percentile(count_perm_anti_hp,95), 0, np.max(np.histogram(count_perm_anti_hp,10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
    plt.legend()
    plt.xlabel('# cells')
    plt.ylabel('Count')
    plt.title('HP Anticorrelation')


    plt.subplot(2,2,3)
    plt.hist(count_perm_same_pfc, 10,color = 'lightblue')      
    plt.vlines(neuron_count_same_pfc, 0, np.max(np.histogram(count_perm_same_pfc,10)[0]), color = 'black', label = 'data')
    plt.vlines(np.percentile(count_perm_same_pfc,99), 0, np.max(np.histogram(count_perm_same_pfc,10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
    plt.vlines(np.percentile(count_perm_same_pfc,95), 0, np.max(np.histogram(count_perm_same_pfc,10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
    plt.legend()
    plt.xlabel('# cells')
    plt.ylabel('Count')
    plt.title('PFC Same Sign')


    plt.subplot(2,2,4)
    plt.hist(count_perm_same_hp, 10,color = 'pink')      
    plt.vlines(neuron_count_same_pfc, 0, np.max(np.histogram(count_perm_same_hp,10)[0]), color = 'black', label = 'data')
    plt.vlines(np.percentile(count_perm_same_hp,99), 0, np.max(np.histogram(count_perm_same_hp,10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
    plt.vlines(np.percentile(count_perm_same_hp,95), 0, np.max(np.histogram(count_perm_same_hp,10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
    plt.legend()
    plt.xlabel('# cells')
    plt.ylabel('Count')
    plt.title('HP Same Sign')
    sns.despine()


