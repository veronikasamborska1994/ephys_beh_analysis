
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:50:34 2020

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


def regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True, perm = False, p_val = 2.40, pred = 2):
    
    C_a = []
    C_b = []
    p_val = p_val
    pred = pred

    if perm:
        count_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.

    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = ch_rew_align.hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = start, end = end, HP = HP)
    neuron_count =0 
    for s, session in enumerate(a_a_matrix_t_1_list):
        
        firing_rates_a = np.concatenate([a_a_matrix_t_1_list[s],a_a_matrix_t_2_list[s],a_a_matrix_t_3_list[s]],1)
        n_neurons = firing_rates_a.shape[0]
        
        for n in range(n_neurons):
            neuron_count +=1
            
        
        rewards_a_1 = a_a_matrix_t_1_list_rewards[s]
        rewards_a_2 = a_a_matrix_t_2_list_rewards[s]
        rewards_a_3 = a_a_matrix_t_3_list_rewards[s]
        rewards = np.hstack([rewards_a_1,rewards_a_2,rewards_a_3])

        choices_a_1 = a_a_matrix_t_1_list_choices[s]
        choices_a_2 = a_a_matrix_t_2_list_choices[s]
        choices_a_3 = a_a_matrix_t_3_list_choices[s]
        choices = np.hstack([choices_a_1,choices_a_2,choices_a_3])

        block_length = np.tile(np.arange(session.shape[1]/2),2)
        trial_number = np.tile(block_length,3)
        ones = np.ones(len(choices))
        
        predictors_a = OrderedDict([#('Reward', rewards),
                                  #('Choice', choices),
                                  ('Trial Number',trial_number),
                                  ('Constant', ones)])
                
        
        X_a = np.vstack(predictors_a.values()).T[:len(choices),:].astype(float)
        n_predictors = X_a.shape[1]
        y = firing_rates_a.reshape([len(firing_rates_a),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X_a)

        C_a.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
        
        
        firing_rates_b = np.concatenate([b_b_matrix_t_1_list[s],b_b_matrix_t_2_list[s],b_b_matrix_t_3_list[s]],1)
        n_neurons = firing_rates_b.shape[0]
        rewards_b_1 = b_b_matrix_t_1_list_rewards[s]
        rewards_b_2 = b_b_matrix_t_2_list_rewards[s]
        rewards_b_3 = b_b_matrix_t_3_list_rewards[s]
        rewards = np.hstack([rewards_b_1,rewards_b_2,rewards_b_3])

        choices_b_1 = b_b_matrix_t_1_list_choices[s]
        choices_b_2 = b_b_matrix_t_2_list_choices[s]
        choices_b_3 = b_b_matrix_t_3_list_choices[s]
        choices = np.hstack([choices_b_1,choices_b_2,choices_b_3])

        block_length = np.tile(np.arange(session.shape[1]/2),2)
        trial_number = np.tile(block_length,3)
        ones = np.ones(len(choices))

        predictors_b = OrderedDict([#('Reward', rewards),
                                     # ('Choice', choices),
                                      ('Trial Number',trial_number),
                                      ('Constant', ones)])
        
           
        X_b = np.vstack(predictors_b.values()).T[:len(choices),:].astype(float)
        n_predictors = X_b.shape[1]
        y = firing_rates_b.reshape([len(firing_rates_b),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        tstats = reg_f.regression_code(y.T, X_b)   

        C_b.append(tstats.reshape(n_predictors,n_neurons)) # Predictor loadings
        
        if perm:
           for i in range(perm):
               all_firing = np.concatenate((firing_rates_a,firing_rates_b),1)
               all_firing_block = all_firing.reshape(all_firing.shape[0],int(all_firing.shape[1]/12), 12)
               all_firing_block = np.transpose(all_firing_block,[2,0,1])
               np.random.shuffle(all_firing_block.T) # Shuffle A / B trials (axis 0 only).
               all_firing_block = np.transpose(all_firing_block,[1,2,0])
               all_firing_block = all_firing_block.reshape(all_firing.shape[0], all_firing.shape[1])
               y_a = all_firing_block[:,:int(all_firing_block.shape[1]/2)]
               y_b  = all_firing_block[:,int(all_firing_block.shape[1]/2):]

               C_perm_a = reg_f.regression_code(y_a.T, X_a)
               C_perm_b = reg_f.regression_code(y_b.T, X_b)
               a_positive_perm = np.where(C_perm_a[pred] > p_val)[0]
               a_negative_perm = np.where(C_perm_a[pred] < -p_val)[0]
               b_positive_perm = np.where(C_perm_b[pred] > p_val)[0]
               b_negative_perm = np.where(C_perm_b[pred] < -p_val)[0]
               
               common_p_a_n_b_perm = [i for i in a_positive_perm if i in b_negative_perm]
               common_n_a_p_b_perm = [i for i in a_negative_perm if i in b_positive_perm]
               count_perm[i].append(len(common_p_a_n_b_perm)+ len(common_n_a_p_b_perm))


     
    C_a = np.concatenate(C_a,1)
    C_b = np.concatenate(C_b,1)  
     
    if perm:
        count_perm = [np.sum(count) for count in count_perm]

           
    return C_a, C_b, count_perm, p_val, neuron_count

def Tim_replicate_perm():
    
    
    #p_val = 1.68 (0.05)
    #p_val = 3.10 (< .001)
    C_a_HP, C_b_HP, count_perm_HP,p_val, neuron_count = regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True, perm = 2, p_val = 1.68  , pred = 0)
    C_a_PFC, C_b_PFC, count_perm_PFC,p_val,neuron_count = regression_time_choices_rewards_a_blocks(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False, perm = 2,  p_val =1.68  , pred = 0)


    
    p_val = 3.10 
    pred = 2
    a_positive_HP = np.where(C_a_HP[pred] > p_val)[0]
    a_negative_HP = np.where(C_a_HP[pred] < -p_val)[0]
    b_positive_HP = np.where(C_b_HP[pred] > p_val)[0]
    b_negative_HP = np.where(C_b_HP[pred] < -p_val)[0]
    common_p_a_n_b_HP = [i for i in a_positive_HP if i in b_negative_HP]
    common_n_a_p_b_HP = [i for i in a_negative_HP if i in b_positive_HP]
    count_HP = len(common_p_a_n_b_HP)+ len(common_n_a_p_b_HP)

    a_positive_PFC = np.where(C_a_PFC[pred] > p_val)[0]
    a_negative_PFC = np.where(C_a_PFC[pred] < -p_val)[0]
    b_positive_PFC = np.where(C_b_PFC[pred] > p_val)[0]
    b_negative_PFC = np.where(C_b_PFC[pred] < -p_val)[0]
    common_p_a_n_b_PFC = [i for i in a_positive_PFC if i in b_negative_PFC]
    common_n_a_p_b_PFC = [i for i in a_negative_PFC if i in b_positive_PFC]
    count_PFC = len(common_p_a_n_b_PFC)+ len(common_n_a_p_b_PFC)   

    plt.subplot(1,2,1)
    plt.hist(count_perm_HP, 10,color = 'grey')      
    plt.vlines(count_HP, 0, np.max(np.histogram(count_perm_HP,10)[0]), color = 'black', label = 'data')
    plt.vlines(np.percentile(count_perm_HP,99), 0, np.max(np.histogram(count_perm_HP,10)[0]), color = 'grey', linestyle  = 'dotted', label =  '<.001')
    plt.vlines(np.percentile(count_perm_HP,95), 0, np.max(np.histogram(count_perm_HP,10)[0]), color = 'grey', linestyle  = '--', label =  '<.05')
    plt.legend()
    plt.xlabel('# cells')
    plt.ylabel('Count')

    plt.title('HP')
    plt.subplot(1,2,2)
    plt.hist(count_perm_PFC,10,color = 'lightblue')      
    plt.vlines(count_PFC, 0, np.max(np.histogram(count_perm_PFC,10)[0]), color = 'black', label = 'data')
    plt.vlines(np.percentile(count_perm_PFC,99), 0, np.max(np.histogram(count_perm_PFC,10)[0]), color = 'lightblue', linestyle  = 'dotted', label =  '<.001')
    plt.vlines(np.percentile(count_perm_PFC,95), 0, np.max(np.histogram(count_perm_PFC,10)[0]), color = 'lightblue', linestyle  = '--', label =  '<.05')

    plt.legend()
    plt.title('PFC')
    plt.xlabel('# cells')
    plt.ylabel('Count')

    sns.despine()
    
# def plot_neurons(common_p_a_n_b_HP,  common_n_a_p_b_HP, common_p_a_n_b_PFC, common_n_a_p_b_PFC,data_HP, data_PFC,\
#                  experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = True):
    
    
    # a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
    # a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
    # a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
    # a_a_matrix_t_1_list_rewards, b_b_matrix_t_1_list_rewards,\
    # a_a_matrix_t_2_list_rewards, b_b_matrix_t_2_list_rewards,\
    # a_a_matrix_t_3_list_rewards, b_b_matrix_t_3_list_rewards,\
    # a_a_matrix_t_1_list_choices, b_b_matrix_t_1_list_choices,\
    # a_a_matrix_t_2_list_choices, b_b_matrix_t_2_list_choices,\
    # a_a_matrix_t_3_list_choices, b_b_matrix_t_3_list_choices = ch_rew_align.hieararchies_extract_rewards_choices(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,start = 0, end = 20, HP = False)
   
    # isl_1 =  wes.Moonrise1_5.mpl_colors
    
    # neuron_count =0 


    # for s, session in enumerate(a_a_matrix_t_1_list):
        
    #     switch = int(b_b_matrix_t_1_list[s].shape[1]/2)
        
    #     firing_rates_b = np.mean([ b_b_matrix_t_1_list[s][:, : switch],b_b_matrix_t_1_list[s][:, switch:],\
    #                                b_b_matrix_t_2_list[s][:, : switch],b_b_matrix_t_2_list[s][:, switch:],\
    #                                b_b_matrix_t_3_list[s][:, : switch],b_b_matrix_t_3_list[s][:, switch:]],0)
            
    #     firing_rates_a = np.mean([ a_a_matrix_t_1_list[s][:, : switch],a_a_matrix_t_1_list[s][:, switch:],\
    #                                a_a_matrix_t_2_list[s][:, : switch],a_a_matrix_t_2_list[s][:, switch:],\
    #                                a_a_matrix_t_3_list[s][:, : switch],a_a_matrix_t_3_list[s][:, switch:]],0)
               
    #     firing_b_std =  (np.std([ b_b_matrix_t_1_list[s][:, : switch],b_b_matrix_t_1_list[s][:, switch:],\
    #                                b_b_matrix_t_2_list[s][:, : switch],b_b_matrix_t_2_list[s][:, switch:],\
    #                                b_b_matrix_t_3_list[s][:, : switch],b_b_matrix_t_3_list[s][:, switch:]],0))
    #     firing_a_std =  (np.std([a_a_matrix_t_1_list[s][:, : switch],a_a_matrix_t_1_list[s][:, switch:],\
    #                                a_a_matrix_t_2_list[s][:, : switch],a_a_matrix_t_2_list[s][:, switch:],\
    #                                a_a_matrix_t_3_list[s][:, : switch],a_a_matrix_t_3_list[s][:, switch:]],0)
               
        
    #     firing_rates = np.concatenate([a_a_matrix_t_1_list[s],a_a_matrix_t_2_list[s],a_a_matrix_t_3_list[s]],1)
    #     n_neurons = firing_rates.shape[0]
    #     print(n_neurons)
    #     print(firing_rates_b.shape)
    #     for n in range(n_neurons):
            
    #         neuron_count += 1
    #         print(neuron_count)
            
    #         if (neuron_count-1) in common_p_a_n_b_PFC or  (neuron_count-1) in common_n_a_p_b_PFC:
    #             print(firing_rates_a.shape)
                
    #             firing_a = firing_rates_a[n]
    #             firing_b =  firing_rates_b[n]
                
    #             firing_a_st = (firing_a_std[n])/(np.sqrt(6))
    #             firing_b_st  = (firing_b_std[n])/(np.sqrt(6))
    #             plt.figure()
    #             plt.plot(firing_a, color = isl_1[0], label = 'Block A Average')
    #             plt.plot(firing_b, color = isl_1[1], label = 'Block B Average')
    #             plt.fill_between(np.arange(len(firing_a)), firing_a-firing_a_st, firing_a+firing_a_st, alpha = 0.2, color = isl_1[0])
    #             plt.fill_between(np.arange(len(firing_b)), firing_b-firing_b_st, firing_b+firing_b_st, alpha = 0.2, color = isl_1[1])

        
        
