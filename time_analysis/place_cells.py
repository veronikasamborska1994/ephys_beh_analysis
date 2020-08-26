#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:49:36 2020

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
from palettable import wesanderson as wes
from scipy import io
import statsmodels.api as sm
import seaborn as sns
from itertools import combinations 
import scipy.stats as stt
from sklearn.feature_selection import f_regression
import regression_function as reg_f
from scipy.stats import zscore


def _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = True):
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = ch_rew_align.hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = HP)
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

  
def regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63,HP = True, anti_corr= True, pred = 0):
    
    pred = pred
  
    count_perm   = []
    comb, comb_1 = [6,6]


    for ind_a in combinations(range(comb + comb_1), comb):
        ind_b = [i for i in range(comb + comb_1) if i not in ind_a]
       
        A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
        neuron_count = 0
        perm_count = 0
        
        C_a = []
        C_b = []
        p_vals_a = []
        p_vals_b = []
    

        neurons = 0
        neuron_id = []

        for s, session in enumerate(A1):
            A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
            B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
            
            A1_rew_s = A1_rew[s];  A2_rew_s= A2_rew[s]; A3_rew_s = A3_rew[s]; A4_rew_s = A4_rew[s];  A5_rew_s= A5_rew[s]; A6_rew_s = A6_rew[s]
            B1_rew_s = B1_rew[s];  B2_rew_s = B2_rew[s]; B3_rew_s = B3_rew[s]; B4_rew_s = B4_rew[s];  B5_rew_s = B5_rew[s]; B6_rew_s = B6_rew[s]
        
            A1_c_s = A1_ch[s];  A2_c_s= A2_ch[s]; A3_c_s = A3_ch[s]; A4_c_s = A4_ch[s];  A5_c_s= A5_ch[s]; A6_c_s = A6_ch[s]
            B1_c_s = B1_ch[s];  B2_c_s = B2_ch[s]; B3_c_s = B3_ch[s]; B4_c_s = B4_ch[s];  B5_c_s = B5_ch[s]; B6_c_s = B6_ch[s]
            for n,neuron in enumerate(A1_s):
                neurons += 1
                basis_funcs = []

                A_f = np.hstack([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]])
                B_f = np.hstack([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]])
                A_rewards = np.hstack([A1_rew_s,A2_rew_s,A3_rew_s,A4_rew_s,A5_rew_s,A6_rew_s])
                B_rewards = np.hstack([B1_rew_s,B2_rew_s,B3_rew_s,B4_rew_s,B5_rew_s,B6_rew_s])
    
                A_choices = np.hstack([A1_c_s,A2_c_s,A3_c_s,A4_c_s,A5_c_s,A6_c_s])
                B_choices = np.hstack([B1_c_s,B2_c_s,B3_c_s,B4_c_s,B5_c_s,B6_c_s])
                    
                BFs_time = [stt.norm(loc=i,scale=.2) for i in np.linspace(0,1,num=4)] #basis functions for time in session
                for i in BFs_time:
                    basis_funcs.append(i.pdf(np.linspace(0,1,num= len(A1_s[n]))))
                
                basis_1 = np.tile(basis_funcs[0],6)
                basis_2 = np.tile(basis_funcs[1],6)
                basis_3 = np.tile(basis_funcs[2],6)
                basis_4 = np.tile(basis_funcs[3],6)
                
                ones = np.ones(len(basis_1))
                
                predictors_a = OrderedDict([('Choice',A_choices),
                                            ('Reward',A_rewards),
                                            ('Start',basis_1),
                                            ('Beginning',basis_2),
                                            ('Middle',basis_3),
                                            ('End',basis_4),
                                            ('Constant', ones)])
                        
                
                X_a = np.vstack(predictors_a.values()).T[:len(ones),:].astype(float)
        
                results_a = sm.OLS(A_f, X_a).fit()   
               
                C_a.append(results_a.tvalues[pred]) # Predictor loadings
                p_vals_a.append(results_a.pvalues[pred])
                
        
                predictors_b = OrderedDict([('Choice',B_choices),
                                           ('Reward',B_rewards),
                                           ('Start',basis_1),
                                            ('Beginning',basis_2),
                                            ('Middle',basis_3),
                                            ('End',basis_4),
                                            ('Constant', ones)])
                 
                   
                X_b = np.vstack(predictors_b.values()).T[:len(ones),:].astype(float)
                results_b = sm.OLS(B_f, X_b).fit()   
                p_vals_b.append(results_b.pvalues[pred])
        
                C_b.append(results_b.tvalues[pred]) # Predictor loadings
    
                if anti_corr:
                    if results_a.params[pred] < 0:
                        if results_a.pvalues[pred] < 0.05:
                            if results_b.params[pred] > 0:
                                if results_b.pvalues[pred] < 0.05:
                                    neuron_count += 1
                                    neuron_id.append(neurons)

                    elif results_a.params[pred] > 0:
                        if results_a.pvalues[pred] < 0.05:
                            if results_b.params[pred] < 0:
                                if results_b.pvalues[pred] < 0.05:
                                    neuron_count += 1    
                                    neuron_id.append(neurons)
                else:
                    if results_a.params[pred] < 0:
                        if results_a.pvalues[pred] < 0.05:
                            if results_b.params[pred] < 0:
                                if results_b.pvalues[pred] < 0.05:
                                    neuron_count += 1
                                    neuron_id.append(neurons)

                    elif results_a.params[pred] > 0:
                        if results_a.pvalues[pred] < 0.05:
                            if results_b.params[pred] > 0:
                                if results_b.pvalues[pred] < 0.05:
                                    neuron_count += 1
                                    neuron_id.append(neurons)

                
                all_firing = np.vstack([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n],B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]])
               
                y_a = all_firing[ind_a,:]
                y_a = y_a.flatten()
                y_b  = all_firing[ind_b,:]
                y_b = y_b.flatten()
        
                C_perm_b = sm.OLS(y_b, X_b).fit()   
                C_perm_a = sm.OLS(y_a, X_a).fit()  
                
                if anti_corr:
                    if C_perm_a.params[pred] < 0:
                        if C_perm_a.pvalues[pred] < 0.05:
                            if C_perm_b.params[pred] > 0:
                                if C_perm_b.pvalues[pred] < 0.05:
                                    perm_count += 1
                    elif C_perm_a.params[pred] > 0:
                        if C_perm_a.pvalues[pred] < 0.05:
                            if C_perm_b.params[pred] < 0:
                                if C_perm_b.pvalues[pred] < 0.05:
                                    perm_count += 1
                        
                else:
                   if C_perm_a.params[pred] < 0:
                        if C_perm_a.pvalues[pred] < 0.05:
                            if C_perm_b.params[pred] < 0:
                                if C_perm_b.pvalues[pred] < 0.05:
                                    perm_count += 1
                   elif C_perm_a.params[pred] > 0:
                        if C_perm_a.pvalues[pred] < 0.05:
                            if C_perm_b.params[pred] > 0:
                                if C_perm_b.pvalues[pred] < 0.05:
                                    perm_count += 1
                        
                        
            
        count_perm.append(perm_count)
    return count_perm, neuron_count,C_a,C_b,neuron_id



def plot(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC):

    plot_neurons_same_place(basis_funcs, neuron_id_PFC_raw_st, data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, cl = 0, i = 0)
    plot_neurons_same_place(basis_funcs, neuron_id_PFC_raw_st, data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, cl = 0, i = 0)

    plot_neurons_same_place(basis_funcs, neuron_id_PFC_raw_beg, data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, cl = 1, i = 3)
  
    plot_neurons_same_place(basis_funcs, neuron_id_PFC_raw_mid, data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, cl = 2, i = 1)
    plot_neurons_same_place(basis_funcs, neuron_id_PFC_raw_end, data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, cl = 3, i = 2)

    count_perm_anti_pfc_st, neuron_count_anti_pfc_st, C_a_anti_pfc_st, C_b_anti_pfc_st, neuron_id_PFC_st = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= True,  pred = 2)
    count_perm_anti_hp_st, neuron_count_anti_hp_st, C_a_anti_hp_st, C_b_anti_hp_st, neuron_id_HP_st = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= True,  pred = 2)

    count_perm_same_pfc_st, neuron_count_same_pfc_st, C_a_same_pfc_st, C_b_same_pfc_st,neuron_id_PFC_raw_st =regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= False,  pred = 2)
    count_perm_same_hp_st, neuron_count_same_hp_st, C_a_same_hp_st, C_b_same_hp_st,neuron_id_HP_raw_st = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= False,  pred = 2)


    count_perm_anti_pfc_beg, neuron_count_anti_pfc_beg, C_a_anti_pfc_beg, C_b_anti_pfc_5, neuron_id_PFC_beg = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= True,  pred = 5)
    count_perm_anti_hp_beg, neuron_count_anti_hp_beg, C_a_anti_hp_beg, C_b_anti_hp_beg, neuron_id_HP_beg = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= True,  pred = 5)

    count_perm_same_pfc_beg, neuron_count_same_pfc_beg, C_a_same_pfc_beg, C_b_same_pfc_beg,neuron_id_PFC_raw_beg =regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= False,  pred = 5)
    count_perm_same_hp_beg, neuron_count_same_hp_beg, C_a_same_hp_beg, C_b_same_hp_beg,neuron_id_HP_raw_beg = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= False,  pred = 5)


    count_perm_anti_pfc_mid, neuron_count_anti_pfc_mid, C_a_anti_pfc_mid, C_b_anti_pfc_mid, neuron_id_PFC_mid = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= True,  pred = 3)
    count_perm_anti_hp_mid, neuron_count_anti_hp_mid, C_a_anti_hp_mid, C_b_anti_hp_mid, neuron_id_HP_mid = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= True,  pred = 3)

    count_perm_same_pfc_mid, neuron_count_same_pfc_mid, C_a_same_pfc_mid, C_b_same_pfc_mid, neuron_id_PFC_raw_mid =regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= False,  pred = 3)
    count_perm_same_hp_mid, neuron_count_same_hp_mid, C_a_same_hp_mid, C_b_same_hp_mid, neuron_id_HP_raw_mid = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= False,  pred = 3)


    count_perm_anti_pfc_end, neuron_count_anti_pfc_end, C_a_anti_pfc_end, C_b_anti_pfc_end, neuron_id_PFC_end = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= True,  pred = 4)
    count_perm_anti_hp_end, neuron_count_anti_hp_end, C_a_anti_hp_end, C_b_anti_hp_end, neuron_id_HP_end = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= True,  pred = 4)

    count_perm_same_pfc_end, neuron_count_same_pfc_end, C_a_same_pfc_end, C_b_same_pfc_end, neuron_id_PFC_raw_end =regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, anti_corr= False,  pred = 4)
    count_perm_same_hp_end, neuron_count_same_hp_end, C_a_same_hp_end, C_b_same_hp_end, neuron_id_HP_raw_end = regression_time_choices_rewards_a_blocks_reward_basis(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, anti_corr= False,  pred = 4)

# Opposite Sign 
    pfc_opp_count_perm = [count_perm_anti_pfc_st,count_perm_anti_pfc_beg,count_perm_anti_pfc_mid,count_perm_anti_pfc_end]
    pfc_opp_count= [neuron_count_anti_pfc_st,neuron_count_anti_pfc_beg,neuron_count_anti_pfc_mid,neuron_count_anti_pfc_end]
    
    hp_opp_count_perm = [count_perm_anti_hp_st,count_perm_anti_hp_beg,count_perm_anti_hp_mid,count_perm_anti_hp_end]
    hp_opp_count = [neuron_count_anti_hp_st,neuron_count_anti_hp_beg,neuron_count_anti_hp_mid,neuron_count_anti_hp_end]
 
  # Same sign 
    
    pfc_same_count_perm = [count_perm_same_pfc_st,count_perm_same_pfc_beg,count_perm_same_pfc_mid,count_perm_same_pfc_end]
    pfc_same_count= [neuron_count_same_pfc_st,neuron_count_same_pfc_beg,neuron_count_same_pfc_mid,neuron_count_same_pfc_end]
    
    hp_same_count_perm = [count_perm_same_hp_st,count_perm_same_hp_beg,count_perm_same_hp_mid,count_perm_same_hp_end]
    hp_same_count = [neuron_count_same_hp_st,neuron_count_same_hp_beg,neuron_count_same_hp_mid,neuron_count_same_hp_end]
   
    for ind, array in enumerate(pfc_opp_count_perm):
        plt.figure()
        plt.subplot(2,2,1)
        plt.hist(pfc_opp_count_perm[ind], 10,color = 'lightblue')      
        plt.vlines(pfc_opp_count[ind], 0, np.max(np.histogram(pfc_opp_count_perm[ind],10)[0]), color = 'black', label = 'data')
        plt.vlines(np.percentile(pfc_opp_count_perm[ind],99), 0, np.max(np.histogram(pfc_opp_count_perm[ind],10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
        plt.vlines(np.percentile(pfc_opp_count_perm[ind],95), 0, np.max(np.histogram(pfc_opp_count_perm[ind],10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
        plt.legend()
        plt.xlabel('# cells')
        plt.ylabel('Count')
        if ind == 0:
            plt.title('PFC Anticorrelation Start')
        elif ind == 1:
            plt.title('PFC Anticorrelation Beg')
        elif ind == 2:
            plt.title('PFC Anticorrelation Mid')
        elif ind == 3:
            plt.title('PFC Anticorrelation End')
   
   
        plt.subplot(2,2,2)
        plt.hist(hp_opp_count_perm[ind], 10,color = 'pink')      
        plt.vlines(hp_opp_count[ind], 0, np.max(np.histogram(hp_opp_count_perm[ind],10)[0]), color = 'black', label = 'data')
        plt.vlines(np.percentile(hp_opp_count_perm[ind],99), 0, np.max(np.histogram(hp_opp_count_perm[ind],10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
        plt.vlines(np.percentile(hp_opp_count_perm[ind],95), 0, np.max(np.histogram(hp_opp_count_perm[ind],10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
        plt.legend()
        plt.xlabel('# cells')
        plt.ylabel('Count')
        if ind == 0:
            plt.title('HP Anticorrelation Start')
        elif ind == 1:
            plt.title('HP Anticorrelation Beg')
        elif ind == 2:
            plt.title('HP Anticorrelation Mid')
        elif ind == 3:
            plt.title('HP Anticorrelation End')
   
    
        plt.subplot(2,2,3)
        plt.hist(pfc_same_count_perm[ind], 10,color = 'lightblue')      
        plt.vlines(pfc_same_count[ind], 0, np.max(np.histogram(pfc_same_count_perm[ind],10)[0]), color = 'black', label = 'data')
        plt.vlines(np.percentile(pfc_same_count_perm[ind],99), 0, np.max(np.histogram(pfc_same_count_perm[ind],10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
        plt.vlines(np.percentile(pfc_same_count_perm[ind],95), 0, np.max(np.histogram(pfc_same_count_perm[ind],10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
        plt.legend()
        plt.xlabel('# cells')
        plt.ylabel('Count')
        if ind == 0:
            plt.title('PFC Same Sign Start')
        elif ind == 1:
            plt.title('PFC Same Sign Beg')
        elif ind == 2:
            plt.title('PFC Same Sign Mid')
        elif ind == 3:
            plt.title('PFC Same Sign End')
    
    
        plt.subplot(2,2,4)
        plt.hist(hp_same_count_perm[ind], 10,color = 'pink')      
        plt.vlines(hp_same_count[ind], 0, np.max(np.histogram(hp_same_count_perm[ind],10)[0]), color = 'black', label = 'data')
        plt.vlines(np.percentile(hp_same_count_perm[ind],99), 0, np.max(np.histogram(hp_same_count_perm[ind],10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
        plt.vlines(np.percentile(hp_same_count_perm[ind],95), 0, np.max(np.histogram(hp_same_count_perm[ind],10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
        plt.legend()
        plt.xlabel('# cells')
        plt.ylabel('Count')
        if ind == 0:
            plt.title('HP Same Sign Start')
        elif ind == 1:
            plt.title('HP Same Sign Beg')
        elif ind == 2:
            plt.title('HP Same Sign Mid')
        elif ind == 3:
            plt.title('HP Same Sign End')
    
        sns.despine()

# =============================================================================
def degrees_of_freedom(f_stats): 
    p_vals_f = []
    
    for f in f_stats:    
        if f >= 3.94 and f < 11.50:
            p_vals_f.append(0.05)
        elif f >= 11.50:
            p_vals_f.append(0.001)
        else: 
            p_vals_f.append(1)
 
    return p_vals_f

def regression_task_specific(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = True):
    
    A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = HP)
       
    A1 = np.concatenate(A1,0); A2 =  np.concatenate(A2,0); A3 =  np.concatenate(A3,0); A4 =  np.concatenate(A4,0); A5 =  np.concatenate(A5,0); A6 =  np.concatenate(A6,0)
    B1 = np.concatenate(B1,0); B2 = np.concatenate(B2,0);  B3 = np.concatenate(B3,0); B4 = np.concatenate(B4,0); B5 = np.concatenate(B5,0); B6 = np.concatenate(B6,0)


    basis_funcs = []
    

    firing_1 = np.hstack([A1,A2,B1,B2])
    firing_2 = np.hstack([A3,A4, B3,B4])
    firing_3 = np.hstack([A5,A6, B5,B6])

          
    BFs_time = [stt.norm(loc=i,scale=.2) for i in np.linspace(0,1,num=4)] #basis functions for time in session
    for i in BFs_time:
        basis_funcs.append(i.pdf(np.linspace(0,1,num= len(A1_s[0]))))
    
    basis_1 = np.tile(basis_funcs[0],4)
    basis_2 = np.tile(basis_funcs[1],4)
    basis_3 = np.tile(basis_funcs[2],4)
    basis_4 = np.tile(basis_funcs[3],4)
    zscore_f1 = zscore(firing_1,0) 
    zscore_f2 = zscore(firing_2,0) 
    zscore_f3 = zscore(firing_3,0) 

    block_a = np.zeros(len(basis_1))
    block_a[:int(len(block_a)/2)] = 1
    
    block_b = np.zeros(len(basis_1))
    block_b[int(len(block_a)/2):] = 1
    
    ones = np.ones(len(basis_1))
    
    predictors = OrderedDict([
                                 ('Block A x Start', basis_1*block_a),
                                 ('Block A x Beginning',basis_2*block_a),
                                 ('Block A x Middle',basis_3*block_a),
                                 ('Block A x End',basis_4*block_a),
                                 ('Block B x Start', basis_1*block_b),
                                 ('Block B x Beginning',basis_2*block_b),
                                 ('Block B x Middle',basis_3*block_b),
                                 ('Block B x End',basis_4*block_b),
                                 ('Constant', ones)])
            
   
    X = np.vstack(predictors.values()).T[:len(ones),:].astype(float)
    Y = firing_1.reshape([len(firing_1),-1]).T
    tc = np.identity(X.shape[1])
    
    pdes = np.linalg.pinv(X)
     
    pe = np.matmul(pdes,Y)
    cope = np.matmul(tc,pe)
    
    Y_2 = firing_2.reshape([len(firing_2),-1]).T
    Y_3 = firing_3.reshape([len(firing_3),-1]).T
    
    implied_basis_2 = np.matmul(Y_2,np.linalg.pinv(cope))
    implied_basis_3 = np.matmul(Y_3,np.linalg.pinv(cope))
    implied_basis = np.mean([implied_basis_2,implied_basis_3],0)

    b=0
    plt.figure()
    for basis in range(implied_basis.shape[1]):
        b+=1
        plt.subplot(9,2,b*2)
        
        plt.plot(implied_basis[:,basis], color = 'red')
        sns.despine()
        
    b=0
    for x in X.T:
        b+=1
        if b > 1:
            plt.subplot(9,2,b*2-1)
        else:
            plt.subplot(9,2,b)
       
        plt.plot(x, color = 'black')
        sns.despine()

    
    
def regression_residuals_blocks_aligned_on_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = True, contrast = np.arange(8)):
    
    A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = HP)
       
    res_a = []
    res_b = []
    p_val = []
    p_val_session_perm = []
    f_test_list  = []
    for s, session in enumerate(A1):
        global A1_s;global A2_s; global A3_s;global A4_s; global A5_s; global A6_s
        global B1_s;global B2_s; global B3_s;global B4_s; global B5_s; global B6_s

        A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
        B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
        
        A1_rew_s = A1_rew[s];  A2_rew_s= A2_rew[s]; A3_rew_s = A3_rew[s]; A4_rew_s = A4_rew[s];  A5_rew_s= A5_rew[s]; A6_rew_s = A6_rew[s]
        B1_rew_s = B1_rew[s];  B2_rew_s = B2_rew[s]; B3_rew_s = B3_rew[s]; B4_rew_s = B4_rew[s];  B5_rew_s = B5_rew[s]; B6_rew_s = B6_rew[s]
    
        A1_c_s = A1_ch[s];  A2_c_s= A2_ch[s]; A3_c_s = A3_ch[s]; A4_c_s = A4_ch[s];  A5_c_s= A5_ch[s]; A6_c_s = A6_ch[s]
        B1_c_s = B1_ch[s];  B2_c_s = B2_ch[s]; B3_c_s = B3_ch[s]; B4_c_s = B4_ch[s];  B5_c_s = B5_ch[s]; B6_c_s = B6_ch[s]

        basis_funcs = []

        firing = np.hstack([A1_s,A2_s,A3_s,A4_s,A5_s,A6_s,B1_s,B2_s,B3_s,B4_s,B5_s,B6_s]).T

        rewards = np.hstack([A1_rew_s,A2_rew_s,A3_rew_s,A4_rew_s,A5_rew_s,A6_rew_s,B1_rew_s,B2_rew_s,B3_rew_s,B4_rew_s,B5_rew_s,B6_rew_s])

        choices = np.hstack([A1_c_s,A2_c_s,A3_c_s,A4_c_s,A5_c_s,A6_c_s,B1_c_s,B2_c_s,B3_c_s,B4_c_s,B5_c_s,B6_c_s])
            
        BFs_time = [stt.norm(loc=i,scale=.2) for i in np.linspace(0,1,num=4)] #basis functions for time in session
        for i in BFs_time:
            basis_funcs.append(i.pdf(np.linspace(0,1,num= len(A1_s[0]))))
        
        basis_1 = np.tile(basis_funcs[0],12)
        basis_2 = np.tile(basis_funcs[1],12)
        basis_3 = np.tile(basis_funcs[2],12)
        basis_4 = np.tile(basis_funcs[3],12)

        block_a = np.zeros(len(rewards))
        block_a[:int(len(block_a)/2)] = 1
        
        block_b = np.zeros(len(rewards))
        block_b[int(len(block_a)/2):] = 1
        
        ones = np.ones(len(basis_1))
        
        predictors = OrderedDict([('Choice',choices),
                                     ('Reward',rewards),
                                     
                                    # ('Linear Time', block_length), 
                                  
                                     ('Block A x Start', basis_1*block_a),
                                     ('Block A x Beginning',basis_2*block_a),
                                     ('Block A x Middle',basis_3*block_a),
                                     ('Block A x End',basis_4*block_a),
                                     ('Block B x Start', basis_1*block_b),
                                     ('Block B x Beginning',basis_2*block_b),
                                     ('Block B x Middle',basis_3*block_b),
                                     ('Block B x End',basis_4*block_b),
                                     ('Constant', ones)])
                
       
        X = np.vstack(predictors.values()).T[:len(ones),:].astype(float)
        Y = firing.reshape([len(firing),-1]) 
        EVs = X.shape[1]

        c = np.eye(EVs)
        c_f = c[contrast,:]
        ones = np.where(c_f == 1)
        c_f[ones[0],ones[1]+1] = -1

        f_stats = reg_f.regression_code_fstats(Y,X, c_f)
        f_test_list.append(degrees_of_freedom(f_stats))
        p_val.append(f_stats)

       
        pdes = np.linalg.pinv(X)
        pe = np.matmul(pdes,Y)
        res = Y - np.matmul(X,pe)

        res_a.append(res[:int(len(res)/2)])
        res_b.append(res[int(len(res)/2):])
        
        p_val_perm   = []
        comb, comb_1 = [6,6]
    
        for ind_a in combinations(range(comb + comb_1), comb):
            ind_b = [i for i in range(comb + comb_1) if i not in ind_a]
           
            shuffle = ['A1_s', 'A2_s', 'A3_s', 'A4_s', 'A5_s', 'A6_s', 'B1_s', 'B2_s','B3_s', 'B4_s', 'B5_s','B6_s']
            a_ind = np.asarray(shuffle)[list(ind_a)]
            b_ind = np.asarray(shuffle)[list(ind_b)]
            
            firing_perm = np.hstack([globals()[a_ind[0]] ,globals()[a_ind[1]], globals()[a_ind[2]],globals()[a_ind[3]],\
                                     globals()[a_ind[4]] ,globals()[a_ind[5]], globals()[b_ind[0]],globals()[b_ind[1]], globals()[b_ind[2]],globals()[b_ind[3]],\
                                     globals()[b_ind[4]] ,globals()[b_ind[5]]]).T
                

            Y_perm = firing_perm.reshape([len(firing_perm),-1]) 
    
            f_stats_perm = reg_f.regression_code_fstats(Y_perm,X, c_f)
            p_val_perm.append(f_stats_perm)
        p_val_session_perm.append(p_val_perm)
        
                            
    p_val_session_perm = np.concatenate(p_val_session_perm,1)
    
    p_val_session_per = np.percentile(p_val_session_perm,95,0)
    p_val = np.concatenate(p_val)
    f_test_list = np.concatenate(f_test_list,0)
    p_val_f05 = np.where(f_test_list == 0.05)[0]
    p_val_f001 = np.where(f_test_list == 0.001)[0]

    where_p_05 = np.where(p_val > p_val_session_per)[0]

    
    return res_a, res_b, p_val, where_p_05, p_val_f05,p_val_f001


def plot_f_significant_cells():
    
    HP = True
    res_a, res_b, p_val, where_p_05, p_val_f05,p_val_f001 =  regression_residuals_blocks_aligned_on_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,HP = HP, contrast = np.arange(2,10))
        
    
    plt.figure()
    
    
    A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = HP)
        
   
    neurons = 0
    n_count = 0
    
    for s, session in enumerate(A1):
        
        A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
        B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
         
        
        for n,neuron in enumerate(A1_s):
             #if neurons in  neuron_id:
            neurons+=1
            if neurons-1 in p_val_f05:
              n_count +=1 
              A_f = np.mean([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]],0)
              B_f = np.mean([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]],0)
              A_std =  (np.std([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]],0))/np.sqrt(6)
              B_std = (np.std([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]],0))/ np.sqrt(6)

              plt.subplot(5,int(len(where_p_05)/4), n_count)
              plt.plot(A_f, color = 'pink', label = 'A')
              plt.plot(B_f, color = 'blue', label = 'B')
              plt.fill_between(np.arange(len(A_f)), A_f-A_std, A_f+A_std, alpha = 0.2, color ='pink')
              plt.fill_between(np.arange(len(B_f)), B_f-B_std, B_f+B_std, alpha = 0.2, color = 'blue')
              plt.ylabel('FR')
              if n_count == len(where_p_05):
                  plt.legend()
              sns.despine()
    #plt.tight_layout()

       


def plot():
    
 isl_1 =  wes.Darjeeling4_5.mpl_colors

 plot_neurons_same_place(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, c1 = isl_1[0], c2 = isl_1[1], c3 =isl_1[2], c4 = isl_1[3])
 plot_neurons_same_place(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = False, c1 = isl_1[0], c2 = isl_1[4], c3 =isl_1[0], c4 = isl_1[3])
 
 
 
def plot_neurons_same_place(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 63, HP = True, c1 = 'grey', c2 = 'pink', c3 ='grey', c4 = 'blue'):
    
     A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, HP = HP)
        
     neurons = 0
     ns = 0
     neurons_hist = []
     neurons_hist_perm = []
     
     for s, session in enumerate(A1):
         A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
         B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
         
        
         for n,neuron in enumerate(A1_s):
             neurons +=1
             #if neurons in  neuron_id:
             corr = []
             ns+=1
             A_f = np.mean([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]],0)
             B_f = np.mean([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]],0)
            
             neurons_hist.append(np.corrcoef(A_f,B_f)[0,1])
             comb, comb_1 = [6,6]


             for ind_a in combinations(range(comb + comb_1), comb):
                ind_b = [i for i in range(comb + comb_1) if i not in ind_a]
               
                all_firing = np.vstack([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n],B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]])
          
                y_a = np.mean(all_firing[ind_a,:],0)
                y_b  = np.mean(all_firing[ind_b,:],0)
                corr.append(np.corrcoef(y_a,y_b)[0,1])
             neurons_hist_perm.append(np.mean(corr))

     plt.figure()
     plt.subplot(1,2,1)
   
     plt.hist(neurons_hist,color = c1,  alpha = 0.8, label = 'data')      
     plt.hist(neurons_hist_perm,color = c2, alpha = 0.8, label =  'shuffle')  
     plt.legend()
     sns.despine()

     if HP == True:
         plt.title('HP Raw')
     else:
         plt.title('PFC Raw')
   
    
         
     res_a, res_b = regression_residuals_blocks_aligned_on_block(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC, start = start, end = end, HP = HP, contrast = np.arange(3,6))
     res_a = np.concatenate(res_a,1).T
     res_b = np.concatenate(res_b,1).T

     neurons_hist_residual = []
     neurons_hist_perm_residual = []
     for s, session in enumerate(res_a):
         A1 = res_a[s].reshape(6, 17)
         B1 = res_b[s].reshape(6, 17)
         
         corr = []
         A_f = np.mean(A1,0)
         B_f = np.mean(B1,0)
       
         neurons_hist_residual.append(np.corrcoef(A_f,B_f)[0,1])
         comb, comb_1 = [6,6]
         

         for ind_a in combinations(range(comb + comb_1), comb):
            ind_b = [i for i in range(comb + comb_1) if i not in ind_a]
           
            all_firing = np.hstack([res_a[s].reshape(17,6),  res_b[s].reshape(17,6)]).T
     
            y_a = np.mean(all_firing[ind_a,],1)
            y_b  = np.mean(all_firing[ind_b,],1)
            corr.append(np.corrcoef(y_a,y_b)[0,1])
         neurons_hist_perm_residual.append(np.mean(corr))

            
     plt.subplot(1,2,2)
     plt.hist(neurons_hist_residual,color = c3, alpha = 0.8, label = 'data')   
    
     
     plt.hist(neurons_hist_perm_residual,color = c4, alpha = 0.8, label = 'shuffle')     
     plt.legend()
     sns.despine()
     if HP == True:
         plt.title('HP Res')
     else:
         plt.title('PFC Res')
     
     plt.figure()
     neuron_ids = np.where(np.asarray(neurons_hist_residual)   < -0.95)[0]
     neuron_ids = neuron_ids[:20]
     neuron_ids_size =int(len(neuron_ids)/4)+1
  
     n_count = 0
     neurons =0
     for n,neuron in enumerate(neurons_hist_residual):
         neurons +=1

         if neurons in neuron_ids:
             n_count+=1
             A1 = res_a[n].reshape(6,17)
             B1 = res_b[n].reshape(6,17)
             plt.subplot(6,neuron_ids_size,n_count)
             plt.plot(np.mean(A1,0), color = 'pink')
             plt.plot(np.mean(B1,0), color = 'blue')
             #a = np.mean(A1,0)
             #b = np.mean(B1,0)
             #plt.plot(np.mean([a,b],0),  color = 'grey')

             sns.despine()
     n_count = 0
     neurons = 0
     A1,A2,A3, A4, A5, A6, B1, B2, B3, B4, B5, B6,A1_rew, A2_rew, A3_rew, A4_rew, A5_rew, A6_rew,B1_rew, B2_rew, B3_rew, B4_rew, B5_rew, B6_rew,A1_ch, A2_ch,A3_ch, A4_ch, A5_ch, A6_ch, B1_ch, B2_ch, B3_ch, B4_ch, B5_ch, B6_ch  = _make_all_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
     plt.figure()

     for s, session in enumerate(A1):
          A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
          B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
             
            
          for n,neuron in enumerate(A1_s):
              neurons +=1
              if neurons in neuron_ids:
                  n_count+=1
               
            
                  A_f = np.mean([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]],0)
                  B_f = np.mean([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]],0)
                
                  plt.subplot(6,neuron_ids_size,n_count)
                  plt.plot(A_f, color = 'pink', alpha = 0.5)
                  plt.plot(B_f, color = 'blue',alpha = 0.5)
                  #plt.plot(np.mean([A_f,B_f],0),  color = 'grey')
                 
                  sns.despine()
      # neurons = 0
      # n_count = 0
      # plt.figure()
 
     # for s, session in enumerate(A1):
     #      A1_s = A1[s];  A2_s = A2[s]; A3_s = A3[s]; A4_s = A4[s];  A5_s = A5[s]; A6_s = A6[s]
     #      B1_s = B1[s];  B2_s = B2[s]; B3_s = B3[s]; B4_s = B4[s];  B5_s = B5[s]; B6_s = B6[s]
             
            
     #      for n,neuron in enumerate(A1_s):
     #          neurons +=1
     #          if neurons in where_p_05:
     #              n_count+=1
               
            
     #              A_f = np.mean([A1_s[n],A2_s[n],A3_s[n],A4_s[n],A5_s[n],A6_s[n]],0)
     #              B_f = np.mean([B1_s[n],B2_s[n],B3_s[n],B4_s[n],B5_s[n],B6_s[n]],0)
                
     #              plt.subplot(6,neuron_ids_size,n_count)
     #              plt.plot(A_f, color = 'pink', alpha = 0.5)
     #              plt.plot(B_f, color = 'blue',alpha = 0.5)
     #              sns.despine()
                 
                 
     # neuron_ids_size = int(int(np.where(np.asarray(neurons_hist_residual) > 0.6)[0].shape[0])/6)+1
     # plt.figure()
     # n_count = 0
     # for n,neuron in enumerate(neurons_hist_residual):
     #      if n in where_p_05:
     #         n_count+=1
     #         A1 = res_a[n].reshape(17,6)
     #         B1 = res_b[n].reshape(17,6)
     #         plt.subplot(6,neuron_ids_size,n_count)
     #         plt.plot(np.mean(A1,1), color = 'pink')
     #         plt.plot(np.mean(B1,1), color = 'blue')
     #         sns.despine()

       