#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:39:27 2020

@author: veronikasamborska
"""

import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from matplotlib import colors as mcolors

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

def tensor_PCA(data):
    n = 1
    fig = plt.figure(1)
    dm = data['DM']
    data = data['Data']
    predictors_correlated = []
    for s,session in enumerate(data): 
        # Fit CP tensor decomposition (two times).
        DM = dm[s]
        R = 20
        
        U = tt.cp_als(session, rank=R, verbose=True)
        #V = tt.cp_als(session, rank=R, verbose=True)
        
        # Compare the low-dimensional factors from the two fits.
        #fig, ax, po = tt.plot_factors(U.factors)
        #tt.plot_factors(V.factors, fig=fig)
        #fig.suptitle("raw models")
        #fig.tight_layout()
        
        # Align the two fits and print a similarity score.
        #sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
        #print(sim)
        
        # Plot the results again to see alignment.
        #fig, ax, po = tt.plot_factors(U.factors)
        #tt.plot_factors(V.factors, fig=fig)
        #fig.suptitle("aligned models")
        #fig.tight_layout()
        
        # Show plots.
        #plt.show()
        
        trial_factors = U.factors.factors[0]
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
        block = DM[:,4]
        block_df = np.diff(block)
        
        correct_choice = np.where(choices == state)[0]
        correct = np.zeros(len(choices))
        correct[correct_choice] = 1
        
        a_since_block = []
        trials_since_block = []
        t = 0
        session_trials_since_block = []
        
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
                                      ('Reward x Trial in Block',reward_trial_in_block)])
        p = list(predictors_all.items()) 
        for t in range(trial_factors.shape[1]):
            
            for i in range(len(p)):
                corr = np.corrcoef(trial_factors[:,t],p[i][1])
                if abs(corr[1,0]) > 0.65:
                    predictors_correlated.append((str(abs(corr[1,0]) ) +  ' ' + p[i][0]))
                    col = list(mcolors.CSS4_COLORS.keys())
                    col = np.asarray(col)
                    if 'x' in p[i][0]: 
                        color = []
                        for ii,value in enumerate(p[i][1]):
                            value = np.int(value)
                            color.append(col[value])
                       
                        
                    else:
                        color_ind =np.asarray(np.where(p[i][1] == 0)[0])
                        color = np.asarray(['green' for i in range(len(p[i][1]))])
                        color[color_ind] = 'black'
                    
                        
                    #fig.add_subplot(5,6,n)
                    #n +=1
                        
                    #plt.scatter(np.arange(len(trial_factors[:,t])),trial_factors[:,t], c = color, s = 1,label=color)
                    #plt.title(p[i][0])

                    if 'x Trial' in p[i][0]: 
                        plt.figure()

                        plt.scatter(np.arange(len(trial_factors[:,t])),trial_factors[:,t], c = color, s = 15,label=color)
                        plt.vlines(np.where(block_df!= 0)[0],ymin = min(trial_factors[:,t]), ymax = max(trial_factors[:,t]), alpha = 0.5)
                        x = trial_factors[:,t]
                        col_x = color
                        state_tial = p[i][1]
                        plt.title(p[i][0])
                    
                    
    fig.tight_layout()                
    return predictors_correlated,x,col_x, state_tial
                   
            
    
