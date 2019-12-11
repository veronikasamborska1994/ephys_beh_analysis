#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:22:26 2019

@author: veronikasamborska
"""
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

import remapping_count as rc 
import numpy as np
import matplotlib.pyplot as plt
import regression_function as reg_f
import regressions as re
from collections import OrderedDict
from matplotlib import colors as mcolors



def regression_general(data):
    
    C = []
    cpd = []
    
    C_1 = []
    cpd_1 = []
    
    C_2 = []
    cpd_2 = []
    
    C_3 = []
    cpd_3 = []
    
    #data = HP

    dm = data['DM'][0]
    #dm = dm[:-1]
    firing = data['Data'][0]
    #firing = firing[:-1]
    session_trials_since_block = []
    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        block = DM[:,4]
        block_df = np.diff(block)
        taskid = rc.task_ind(task,a_pokes,b_pokes)

        correct_choice = np.where(choices == state)[0]
        correct = np.zeros(len(choices))
        correct[correct_choice] = 1
        
        a_since_block = []
        trials_since_block = []
        t = 0
        
        #Bug in the state? 
        for st,s in enumerate(block):
            if state[st-1] != state[st]:
                t = 0
            else:
                t+=1
            trials_since_block.append(t)
            
        session_trials_since_block.append(trials_since_block)
              
        t = 0    
        for st,(s,c) in enumerate(zip(block, choices)):
            if state[st-1] != state[st]:
                t = 0
                a_since_block.append(t)

            elif c == 1:
                t+=1
                a_since_block.append(t)
            else:
                a_since_block.append(0)

        negative_reward_count = []
        rew = 0
        block_df = np.append(block_df,0)
        for r,b in zip(reward,block_df):
  
            if r == 0:
                rew += 1
                negative_reward_count.append(rew)
            elif r == 1:
                rew -= 1
                negative_reward_count.append(rew)
            if b != 0:
                rew = 0
                
        positive_reward_count = []
        rew = 0
        block_df = np.append(block_df,0)
        for r,b in zip(reward,block_df):
  
            if r == 1:
                rew += 1
                positive_reward_count.append(rew)
            elif r == 0:
                rew += 0
                positive_reward_count.append(rew)
            if b != 0:
                rew = 0
 
        positive_reward_count = np.asarray(positive_reward_count)
        negative_reward_count = np.asarray(negative_reward_count)
        choices_int = np.ones(len(reward))

        
        choices_int[np.where(choices == 0)] = -1
        reward_choice_int = choices_int * reward
        interaction_trial_latent = trials_since_block * state
        interaction_a_latent = a_since_block * state
        int_a_reward = a_since_block * reward
        
        interaction_trial_choice = trials_since_block*choices_int
        reward_trial_in_block = trials_since_block*positive_reward_count
        negative_reward_count_st = negative_reward_count*correct
        positive_reward_count_st = positive_reward_count*correct
        negative_reward_count_ch = negative_reward_count*choices
        positive_reward_count_ch = positive_reward_count*choices
        ones = np.ones(len(choices))
        

        predictors_all = OrderedDict([('Reward', reward),
                                  ('Choice', choices),
                                  ('Correct', correct),
                                  ('A in Block', a_since_block),   
                                  ('A in Block x Reward', int_a_reward),   
                                  
                                  ('State', state),
                                  ('Trial in Block', trials_since_block),
                                  ('Interaction State x Trial in Block', interaction_trial_latent),
                                  ('Interaction State x A count', interaction_a_latent),

                                  ('Choice x Trials in Block', interaction_trial_choice),
                                  ('Reward x Choice', reward_choice_int),
                                  ('No Reward Count in a Block', negative_reward_count),
                                  ('No Reward x Correct', negative_reward_count_st),
                                  ('Reward Count in a Block', positive_reward_count),
                                  ('Reward Count x Correct', positive_reward_count_st),
                                  ('No reward Count x Choice',negative_reward_count_ch),
                                  ('Reward Count x Choice',positive_reward_count_ch),
                                  ('Reward x Trial in Block',reward_trial_in_block),
    
                                  ('ones', ones)])
        
           
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
    
        C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]

        # Task 1 
        reward_t1 = reward[task_1]
        choices_t1 = choices[task_1]
        correct_t1 = correct[task_1]
        
        a_since_block_t1 = np.asarray(a_since_block)[task_1]
        int_a_reward_t1 = int_a_reward[task_1]
        state_t1 = state[task_1]
        trials_since_block_t1 = np.asarray(trials_since_block)[task_1]
        interaction_trial_latent_t1 = interaction_trial_latent[task_1]
        interaction_a_latent_t1 = interaction_a_latent[task_1]
        interaction_trial_choice_t1 = interaction_trial_choice[task_1]
        reward_choice_int_t1 = reward_choice_int[task_1]
        negative_reward_count_t1 = negative_reward_count[task_1]
        negative_reward_count_st_t1 = negative_reward_count_st[task_1]
        positive_reward_count_t1 = positive_reward_count[task_1]
        positive_reward_count_st_t1 = positive_reward_count_st[task_1]
        negative_reward_count_ch_t1 = negative_reward_count_ch[task_1]
        positive_reward_count_ch_t1 = positive_reward_count_ch[task_1]
        reward_trial_in_block_t1 = reward_trial_in_block[task_1]
        
        firing_rates_t1 = firing_rates[task_1]
        ones = np.ones(len(choices_t1))

        predictors = OrderedDict([('Reward', reward_t1),
                                  ('Choice', choices_t1),
                                  ('Correct', correct_t1),
                                  ('A in Block', a_since_block_t1),   
                                  ('A in Block x Reward', int_a_reward_t1),   
                                  
                                  ('State', state_t1),
                                  ('Trial in Block', trials_since_block_t1),
                                  ('Interaction State x Trial in Block', interaction_trial_latent_t1),
                                  ('Interaction State x A count', interaction_a_latent_t1),

                                  ('Choice x Trials in Block', interaction_trial_choice_t1),
                                  ('Reward x Choice', reward_choice_int_t1),
                                  ('No Reward Count in a Block', negative_reward_count_t1),
                                  ('No Reward x Correct', negative_reward_count_st_t1),
                                  ('Reward Count in a Block', positive_reward_count_t1),
                                  ('Reward Count x Correct', positive_reward_count_st_t1),
                                  ('No reward Count x Choice',negative_reward_count_ch_t1),
                                  ('Reward Count x Choice',positive_reward_count_ch_t1),
                                  ('Reward x Trial in Block',reward_trial_in_block_t1),
    
                                  ('ones', ones)])
           
        X = np.vstack(predictors.values()).T[:len(choices_t1),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_t1.reshape([len(firing_rates_t1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
    
        C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
       
        # Task 2
        reward_t2 = reward[task_2]
        choices_t2 = choices[task_2]
        correct_t2 = correct[task_2]
        
        a_since_block_t2 = np.asarray(a_since_block)[task_2]
        int_a_reward_t2 = int_a_reward[task_2]
        state_t2 = state[task_2]
        trials_since_block_t2 = np.asarray(trials_since_block)[task_2]
        interaction_trial_latent_t2 = interaction_trial_latent[task_2]
        interaction_a_latent_t2 = interaction_a_latent[task_2]
        interaction_trial_choice_t2 = interaction_trial_choice[task_2]
        reward_choice_int_t2 = reward_choice_int[task_2]
        negative_reward_count_t2 = negative_reward_count[task_2]
        negative_reward_count_st_t2 = negative_reward_count_st[task_2]
        positive_reward_count_t2 = positive_reward_count[task_2]
        positive_reward_count_st_t2 = positive_reward_count_st[task_2]
        negative_reward_count_ch_t2 = negative_reward_count_ch[task_2]
        positive_reward_count_ch_t2 = positive_reward_count_ch[task_2]
        reward_trial_in_block_t2 = reward_trial_in_block[task_2]
        
        firing_rates_t2 = firing_rates[task_2]
        ones = np.ones(len(choices_t2))

        predictors = OrderedDict([('Reward', reward_t2),
                                  ('Choice', choices_t2),
                                  ('Correct', correct_t2),
                                  ('A in Block', a_since_block_t2),   
                                  ('A in Block x Reward', int_a_reward_t2),   
                                  
                                  ('State', state_t2),
                                  ('Trial in Block', trials_since_block_t2),
                                  ('Interaction State x Trial in Block', interaction_trial_latent_t2),
                                  ('Interaction State x A count', interaction_a_latent_t2),

                                  ('Choice x Trials in Block', interaction_trial_choice_t2),
                                  ('Reward x Choice', reward_choice_int_t2),
                                  ('No Reward Count in a Block', negative_reward_count_t2),
                                  ('No Reward x Correct', negative_reward_count_st_t2),
                                  ('Reward Count in a Block', positive_reward_count_t2),
                                  ('Reward Count x Correct', positive_reward_count_st_t2),
                                  ('No reward Count x Choice',negative_reward_count_ch_t2),
                                  ('Reward Count x Choice',positive_reward_count_ch_t2),
                                  ('Reward x Trial in Block',reward_trial_in_block_t2),
    
                                  ('ones', ones)])
           
           
        X = np.vstack(predictors.values()).T[:len(choices_t2),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_t2.reshape([len(firing_rates_t2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
    
        C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_2.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
        
        
        # Task 3
        reward_t3 = reward[task_3]
        choices_t3 = choices[task_3]
        trials_since_block_t3 = np.asarray(trials_since_block)[task_3]
        state_t3 = state[task_3]
        interaction_trial_latent_t3 = interaction_trial_latent[task_3]
        interaction_trial_choice_t3 = interaction_trial_choice[task_3]
        reward_choice_int_t3 = reward_choice_int[task_3]
        negative_reward_count_t3 = negative_reward_count[task_3]
        negative_reward_count_st_t3 = negative_reward_count_st[task_3]

        firing_rates_t3 = firing_rates[task_3]
        ones = np.ones(len(choices_t3))

        predictors = OrderedDict([('Reward', reward_t3),
                                  ('Choice', choices_t3),
                                 # ('Reward Rate', reward_exp),
                                 # ('Trial in Block', trials_since_block_t3),
                                 # ('State', state_t3),
                                 # ('Interaction Block x Trial in Block', interaction_trial_latent_t3),
                                  #('Choice x Trials in Block', interaction_trial_choice_t3),
                                  ('Reward x Choice', reward_choice_int_t3),
                                  ('No reward count', negative_reward_count_t3),
                                  ('No Reward x Correct', negative_reward_count_st_t3),

                                  ('ones', ones)])
        
           
        X = np.vstack(predictors.values()).T[:len(choices_t3),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates_t3.reshape([len(firing_rates_t3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
    
        C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_3.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
    
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
   
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    C_1 = np.concatenate(C_1,1)
   
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)
    C_2 = np.concatenate(C_2,1)
    
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    C_3 = np.concatenate(C_3,1)
    
    return C, cpd, C_1, cpd_1,C_2, cpd_2, C_3, cpd_3, predictors_all,session_trials_since_block


#           predictors = OrderedDict([('Reward', reward_t2),
#                                  ('Choice', choices_t2),
#                                  ('Correct', correct_t2),
#                                  ('A in Block', a_since_block_t2),   
#                                  ('A in Block x Reward', int_a_reward_t2),   
#                                  
#                                  ('State', state_t2),
#                                  ('Trial in Block', trials_since_block_t2),
#                                  ('Interaction State x Trial in Block', interaction_trial_latent_t2),
#                                  ('Interaction State x A count', interaction_a_latent_t2),
#
#                                  ('Choice x Trials in Block', interaction_trial_choice_t2),
#                                  ('Reward x Choice', reward_choice_int_t2),
#                                  ('No Reward Count in a Block', negative_reward_count_t2),
#                                  ('No Reward x Correct', negative_reward_count_st_t2),
#                                  ('Reward Count in a Block', positive_reward_count_t2),
#                                  ('Reward Count x Correct', positive_reward_count_st_t2),
#                                  ('No reward Count x Choice',negative_reward_count_ch_t2),
#                                  ('Reward Count x Choice',positive_reward_count_ch_t2),
#                                  ('Reward x Trial in Block',reward_trial_in_block_t2),
    
def svd_on_coefs():
    
    C, cpd, C_1, cpd_1,C_2, cpd_2, C_3, cpd_3, predictors,session_trials_since_block = regression_general(HP)
    C_PFC, cpd_PFC, C_1_PFC, cpd_1_PFC,C_2_PFC, cpd_2_PFC, C_3_PFC, cpd_3_PFC, predictors,session_trials_since_block = regression_general(PFC)

    task_1_HP = C_1[[5,6,7,9],:,:]
    task_2_HP = C_2[[5,6,7,9],:,:]
    
    #HP  
    task_1_HP = np.transpose(task_1_HP,[0,2,1]).reshape(task_1_HP.shape[0]*task_1_HP.shape[2], task_1_HP.shape[1])
    task_2_HP = np.transpose(task_2_HP,[0,2,1]).reshape(task_2_HP.shape[0]*task_2_HP.shape[2], task_2_HP.shape[1])
    #task_3 = np.transpose(task_3,[0,2,1]).reshape(task_3.shape[0]*task_3.shape[2], task_3.shape[1])
    
    
    where_are_NaNs = np.isnan(task_1_HP)
    task_1_HP[where_are_NaNs] = 0
    where_are_NaNs = np.isinf(task_1_HP)
    task_1_HP[where_are_NaNs] = 0
    
    
    where_are_NaNs = np.isnan(task_2_HP)
    task_2_HP[where_are_NaNs] = 0
    where_are_NaNs = np.isinf(task_2_HP)
    task_2_HP[where_are_NaNs] = 0
    
    
    u_t1, s_t1, vh_t1 = np.linalg.svd(np.transpose(task_1_HP), full_matrices = False)
    t_u = np.transpose(u_t1)  
    t_v = np.transpose(vh_t1)  
 
    
    s_task_1 = np.linalg.multi_dot([t_u, np.transpose(task_2_HP), t_v])
    s_1 = s_task_1.diagonal()

   
    sum_c_task_1 = np.cumsum(abs(s_1))/task_1_HP.shape[0]
    
    plt.plot(sum_c_task_1, 'black', label = 'HP')
    plt.legend()
    
    #PFC
    task_1_PFC = C_1_PFC[[5,6,7,9],:,:]
    task_2_PFC = C_2_PFC[[5,6,7,9],:,:]
    
    
    task_1_PFC = np.transpose(task_1_PFC,[0,2,1]).reshape(task_1_PFC.shape[0]*task_1_PFC.shape[2], task_1_PFC.shape[1])
    task_2_PFC = np.transpose(task_2_PFC,[0,2,1]).reshape(task_2_PFC.shape[0]*task_2_PFC.shape[2], task_2_PFC.shape[1])
    #task_3 = np.transpose(task_3,[0,2,1]).reshape(task_3.shape[0]*task_3.shape[2], task_3.shape[1])
    
    
    where_are_NaNs = np.isnan(task_1_PFC)
    task_1_PFC[where_are_NaNs] = 0
    where_are_NaNs = np.isinf(task_1_PFC)
    task_1_PFC[where_are_NaNs] = 0
    
    
    where_are_NaNs = np.isnan(task_2_PFC)
    task_2_PFC[where_are_NaNs] = 0
    where_are_NaNs = np.isinf(task_2_PFC)
    task_2_PFC[where_are_NaNs] = 0
    
    
    u_t1, s_t1, vh_t1 = np.linalg.svd(np.transpose(task_1_PFC), full_matrices = False)
    t_u = np.transpose(u_t1)  
    t_v = np.transpose(vh_t1)  
 
    
    s_task_1 = np.linalg.multi_dot([t_u, np.transpose(where_are_NaNs), t_v])
    s_1 = s_task_1.diagonal()

   
    sum_c_task_1 = np.cumsum(abs(s_1))/task_1_PFC.shape[0]
    
    plt.plot(sum_c_task_1, 'green', label = 'PFC')
    plt.legend()

   
def plot_cpd_gen(data,fig_n,title):
    
    C, cpd, C_1, cpd_1,C_2, cpd_2, C_3, cpd_3, predictors,session_trials_since_block = regression_general(PFC)
    cpd = cpd[:,:-1]
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    c = [*colors]
    #c =  ['violet', 'black', 'red','chocolate', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
       #   'purple','orange', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors]
    plt.figure(fig_n)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.title(title)
    plt.legend()
    plt.ylabel('Coefficient of Partial Determination')
    plt.xlabel('Time (ms)')
    
    firing = PFC['Data'][0]
    dm = PFC['DM'][0]

    coef = C[6]
    coef_average = np.mean(coef, 1)
    index = np.where(coef_average > 4)[0][1]
    #index = 192
    neuron = 0
    for sess,s in enumerate(firing):
        for n in range(s.shape[1]):
            neuron+=1
            if neuron == index:
                block_plot = session_trials_since_block[sess]
                trials = s[:,n,:]
                design = dm[sess]
                
                
    # Plot mean around choice
    ch = np.mean(trials[:,22:26],1)
    plt.figure()
    plt.plot(ch)
    plt.plot(block_plot)
    plt.plot(design[:,0])
    
    ch = np.mean(trials[:,42:50],1)
    plt.figure()
    plt.plot(ch)
    plt.plot(block_plot)
    plt.plot(design[:,0])
                        
        

#   predictors_all = OrderedDict([
#                          ('latent_state',state),
#                          ('choice',choices_forced_unforced ),
#                          ('reward', outcomes),
#                          ('forced_trials',forced_trials),
#                          ('block', block),
#                          ('task',task),
#                          ('A', a_pokes),
#                          ('B', b_pokes),
#                          ('Initiation', i_pokes),
#                          #('Chosen_Simple_RW',chosen_Q1),
#                          #('Chosen_Cross_learning_RW', chosen_Q4),
#                          #('Value_A_RW', Q1_value_a),
#                          #('Value_B_RW', Q1_value_b),
#                          #('Value_A_Cross_learning', Q4_value_a),
#                          ('ones', ones)])

    
   # plt.vlines(32,ymin = 0, ymax = 0.15,linestyles= '--', color = 'grey', label = 'Poke')
#   
#           predictors = OrderedDict([('Reward', reward_t2),
#                                  ('Choice', choices_t2),
#                                  ('Correct', correct_t2),
#                                  ('A in Block', a_since_block_t2),   
#                                  ('A in Block x Reward', int_a_reward_t2),   
#                                  
#                                  ('State', state_t2),
#                                  ('Trial in Block', trials_since_block_t2),
#                                  ('Interaction State x Trial in Block', interaction_trial_latent_t2),
#                                  ('Interaction State x A count', interaction_a_latent_t2),
#
#                                  ('Choice x Trials in Block', interaction_trial_choice_t2),
#                                  ('Reward x Choice', reward_choice_int_t2),
#                                  ('No Reward Count in a Block', negative_reward_count_t2),
#                                  ('No Reward x Correct', negative_reward_count_st_t2),
#                                  ('Reward Count in a Block', positive_reward_count_t2),
#                                  ('Reward Count x Correct', positive_reward_count_st_t2),
#                                  ('No reward Count x Choice',negative_reward_count_ch_t2),
#                                  ('Reward Count x Choice',positive_reward_count_ch_t2),
#                                  ('Reward x Trial in Block',reward_trial_in_block_t2),
#    
#                                  ('ones', ones)])
           
   
    plt.ylim(0, 0.1)
    fig = plt.figure()
    for i in range(len(p)):
        task_1 = np.mean(C_1[i,:], axis = 1).flatten()
        task_2 = np.mean(C_2[i,:], axis = 1).flatten()
       # task_3 = np.mean(C_3[4,:], axis = 1).flatten()
    
        argmax_neuron = np.argsort(-task_1)
        task_2_by_1 = task_2[argmax_neuron]
        task_1 = task_1[argmax_neuron]
        #task_3_by_1 = task_3[argmax_neuron]
        
        y = np.arange(len(task_1))
        fig.add_subplot(5, 4, i+1)
        plt.scatter(y,task_2_by_1,s = 2, color = 'blue', label = 'Task 2 sorted by Task 1')
        plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
        plt.title(p[i])
        plt.tight_layout()

    #plt.scatter(y,task_3_by_1,s = 2,color = 'slateblue', label = 'Task 3 sorted by Task 1')
    
    #plt.scatter(y,task_1,s = 2,color = 'black', label = 'Task 1 sorted')

    #plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
    
    


    