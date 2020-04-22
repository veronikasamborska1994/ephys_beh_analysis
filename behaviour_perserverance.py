#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:14:43 2020

@author: veronikasamborska
"""

import seaborn as sns
import numpy as np
from collections import OrderedDict
import scipy.stats as stt
from sklearn.model_selection import RepeatedKFold
import sklearn.linear_model as lm

def regression_cross_validate_perseverance(data):
    
    dm = data['DM'][0] 
    ccs = []; ccs_ch = []; ccs_rew = []

    for  s, sess in enumerate(dm):        
        DM = dm[s]
        choices = DM[:,1]
        reward = DM[:,2]    
    
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]

        if len(ind_block) >= 11:
            
            trials_since_block = []
            t = 0
            
            for st,s in enumerate(block):
                if block[st-1] != block[st]:
                    t = 0
                else:
                    t+=1
                trials_since_block.append(t)
                
            #block_totals_ind = (np.where(np.asarray(ind_block) == 1)[0]-1)[1:]
            block_totals_ind = ind_block
            block_totals = np.diff(block_totals_ind)-1
            trials_since_block = trials_since_block[:ind_block[11]]
            fraction_list = []


            for t,trial in enumerate(trials_since_block):
                
                if t <= block_totals_ind[0]:
                    fr = trial/block_totals_ind[0]
                    fraction_list.append(fr)

                elif t > block_totals_ind[0] and  t <= block_totals_ind[1]:
                    fr = trial/block_totals[0]
                    fraction_list.append(fr)

                elif t > block_totals_ind[1] and  t <= block_totals_ind[2]:
                    fr = trial/block_totals[1]               
                    fraction_list.append(fr)

                elif t > block_totals_ind[2] and  t <= block_totals_ind[3]:
                    fr = trial/block_totals[2]                
                    fraction_list.append(fr)

                elif t > block_totals_ind[3] and  t <= block_totals_ind[4]:
                    fr = trial/block_totals[3]
                    fraction_list.append(fr)

                elif t > block_totals_ind[4] and  t <= block_totals_ind[5]:
                    fr = trial/block_totals[4]
                    fraction_list.append(fr)

                elif t > block_totals_ind[5] and  t <= block_totals_ind[6]:
                    fr = trial/block_totals[5]
                    fraction_list.append(fr)

                elif t > block_totals_ind[6] and  t <= block_totals_ind[7]:
                    fr = trial/block_totals[6]  
                    fraction_list.append(fr)

                elif t > block_totals_ind[7] and  t <= block_totals_ind[8]:
                    fr = trial/block_totals[7]
                    fraction_list.append(fr)

                elif t > block_totals_ind[8] and  t <= block_totals_ind[9]:
                    fr = trial/block_totals[8]                 
                    fraction_list.append(fr)

                elif t > block_totals_ind[9] and  t <= block_totals_ind[10]:
                    fr = trial/block_totals[9]
                    fraction_list.append(fr)

                elif t >  block_totals_ind[10] and  t <= len(trials_since_block):
                    fr = trial/trials_since_block[-1]
                    fraction_list.append(fr)
                    
            choices = choices[:ind_block[11]]
            reward = reward[:ind_block[11]]
            
            last_reward = []

            for r,rew in enumerate(reward):
                if r > 0 :
                    if reward[r-1] == 1:
                        last_reward.append(1)
                    elif reward[r-1] == 0:
                        last_reward.append(0)
                
            last_choice =  []

            for c,ch in enumerate(choices):
                if c > 0 :
                    if choices[r-1] == 1:
                        last_choice.append(1)
                    elif choices[r-1] == 0:
                        last_choice.append(0)
                    
                       
            fraction_list = np.asarray(fraction_list)[1:]
     
            last_reward = np.asarray(last_reward)
            last_choice = np.asarray(last_choice)
            fraction_reward = last_reward*fraction_list
            fraction_choice = last_reward*fraction_list
            trials = len(fraction_choice)
            
            predictors_all = OrderedDict([('Last Reward', last_reward),
                                          ('Last Choice', last_choice),
                                          ('Block Fraction', fraction_list), 
                                          ('Block Fraction x Choice', fraction_choice),
                                          ('Block Fraction x Reward', fraction_reward)])
            
            X = np.vstack(predictors_all.values()).T[:trials,:].astype(float)
            y = choices[1:]     
            kf = RepeatedKFold(n_splits = 5,n_repeats = 2,random_state = 99)  #initialise repeated K-Fold
             
           
            ccx = []; ccx_ch = []; ccx_rew = [] #initialise contained for storing cross validated fits
               
            
            for train_ix,test_ix in kf.split(y):  
        
                y_train = y[train_ix]; y_test = y[test_ix] #get train and test indices for activity
                    
                #get train and test DM without time in block interactions
                x_train_no_choice_int = X[:,0][train_ix]; x_test_no_choice_int = X[:,0][test_ix]
               
                #get train and test DM with time in block interactions
                x_train_choice_int = X[:,1][train_ix]; x_test_choice_int = X[:,1][test_ix]
                x_train_rew_int = X[train_ix,2:3]; x_test_rew_int = X[test_ix,2:3]

                #fit linear model with regularisation. Ideally would do nested K-fold
                #to select optimal hyper-parameter
                linR = lm.LogisticRegression(fit_intercept = True)
                ft = linR.fit(x_train_no_choice_int,y_train)
                
                ccx.append(np.corrcoef(ft.predict(x_test_no_choice_int),y_test)[0,1])  #get cross validated fit quality
                
                linR = lm.LogisticRegression(fit_intercept = True)

                ft = linR.fit(x_train_choice_int,y_train)
                ccx_ch.append(np.corrcoef(ft.predict(x_test_choice_int),y_test)[0,1])  #get cross validated fit quality
                
                linR = lm.LogisticRegression(fit_intercept = True)

                ft = linR.fit(x_train_rew_int,y_train)
                ccx_rew.append(np.corrcoef(ft.predict(x_test_rew_int),y_test)[0,1])  #get cross validated fit quality
            
            ccs.append(np.nanmean(ccx))
            ccs_ch.append(np.nanmean(ccx_ch))
            ccs_rew.append(np.nanmean(ccx_rew))
            
    c1  = np.array(ccs)**2
    c2  = np.array(ccs_ch)**2

    ixs = np.logical_and.reduce([np.isfinite(c1),
                             np.isfinite(c2)])
    t,p = stt.ttest_rel(c1[ixs],c2[ixs])
    print('Variance explained \nwithout time in block: {:.5f}\nwith time in block: {:.5f}'.format(np.nanmean(c1),np.nanmean(c2)))
    print('t:{:.3f}\np:{:.3e}'.format(t,p))

                  