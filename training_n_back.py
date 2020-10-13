#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:36:48 2020

@author: veronikasamborska
"""


from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import sys 
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pandas as pd

import seaborn as sns
from sklearn.decomposition import PCA
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}
import scipy
from statsmodels.regression.linear_model import OLS
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import ephys_beh_import as ep

plt.rc('font', **font)
sys.path.append( '/Users/veronikasamborska/Desktop/Veronika Backup/2018-12-12-Reversal_learning/code/reversal_learning/')
import data_import as di

def import_data():
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

    #HP_LFP,PFC_LFP, m484_LFP, m479_LFP, m483_LFP, m478_LFP, m486_LFP, m480_LFP, m481_LFP, all_sessions_LFP = ep.import_code(ephys_path,beh_path, lfp_analyse = 'True') 
    HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')

    exp = di.Experiment('/Users/veronikasamborska/Desktop/Veronika Backup/2018-12-12-Reversal_learning/data_pilot3')

def plotting_functions(exp):
   
    n = 6
    
    coef_subj =  runs_length(exp, subject_IDs ='all', n = n)
    coef_subj = np.asarray(coef_subj)
    rewards = coef_subj[:,:,:n]
    choices = coef_subj[:,:,n:n*2]
    choices_X_reward = coef_subj[:,:,n*2:-1]
    
    
    mean_ch = np.mean(choices,0)
    sqrt_ch = np.std(choices,0)/np.sqrt(9)
 
    mean_rew = np.mean(rewards,0)
    sqrt_rew = np.std(rewards,0)/np.sqrt(9)
 
    mean_rew_ch = np.mean(choices_X_reward,0)
    sqrt_rew_ch = np.std(choices_X_reward,0)/np.sqrt(9)
    
   
    _1_back_ch = choices[:, :, 0]#[:,1:]
    _other_back_ch = np.mean(choices[:, :,1:],2)#[:,1:]
    _1_back_rew_ch = choices_X_reward[:,:,0]
    _other_back_rew_ch = np.mean(choices_X_reward[:,:,1:],2)
    _all_back_rew = np.mean(rewards,2)#[:,1:]

    fractions = np.concatenate(_1_back_ch.T,0)
    subject_id = np.tile(np.arange(10), 9)
    fraction_id = np.zeros(90)
    k = 0 
    for n in range(10):
        fraction_id[n*9:n*9+9] = k
        k+=1
        
        
    data = {'Data':fractions,'Sub_id': subject_id,'cond': fraction_id}
    data = pd.DataFrame.from_dict(data = data)
    
    
    aovrm = AnovaRM(data, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()
    print(res)

    
    _1_back_ch = np.mean(choices[:, :, 0],0)

    _1_back_ch_er = np.std(choices[:, :, 0],0)/np.sqrt(9)

    #_other_back_ch = np.mean(choices[:, :, 1:],2)
    _other_back_ch =  np.mean(np.mean(choices[:, :,1:],2),0)
    _other_back_ch_err =  np.std(np.mean(choices[:, :, 1:],2),0)/np.sqrt(9)

    #_all_back_rew = np.mean(rewards,2)
    _all_back_rew = np.mean(np.mean(rewards,2),0)
    _all_back_rew_err =  np.std(np.mean(rewards,2),0)/np.sqrt(9)

    #_all_back_rew_ch = np.mean(choices_X_reward,2)
    _1_back_rew_ch = np.mean(choices_X_reward[:,:,0],0)
    _1_back_rew_ch_err =  np.std(choices_X_reward[:,:,0],0)/np.sqrt(9)
 
    _all_back_rew_ch = np.mean(np.mean(choices_X_reward[:,:,1:],2),0)
    _all_back_rew_ch_err =  np.std(np.mean(choices_X_reward[:,:,1:],2),0)/np.sqrt(9)
 
    
    plt.figure(figsize = (10,3))

    plt.subplot(1,5,1)
    plt.errorbar(np.arange(len(_all_back_rew)), _all_back_rew, yerr=_all_back_rew_err, fmt='o', color = 'purple')

    plt.subplot(1,5,2)
    #sns.boxplot(data =_1_back_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_1_back_ch)), _1_back_ch, yerr=_1_back_ch_er, fmt='o', color = 'purple')

    plt.subplot(1,5,3)
    #sns.boxplot(data=_other_back_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_other_back_ch)), _other_back_ch, yerr=_other_back_ch_err, fmt='o', color = 'purple')
    
    plt.subplot(1,5,4)
    #sns.boxplot(data=_all_back_rew_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_1_back_rew_ch)), _1_back_rew_ch, yerr=_1_back_rew_ch_err, fmt='o', color = 'purple')

  
    plt.subplot(1,5,5)
    #sns.boxplot(data=_all_back_rew_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_all_back_rew_ch)), _all_back_rew_ch, yerr=_all_back_rew_ch_err, fmt='o', color = 'purple')

    sns.despine()


def runs_length(experiment, subject_IDs ='all', n = 12):    
   
    n = n
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        
    coef_subj = []
    se_subj = []
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
        X_subj = []
        choices_current_subj = []
        results_task = []
        task = 0
        config_add = []
        for j, session in enumerate(subject_sessions):
            choices = session.trial_data['choices']
            all_sessions = len(subject_sessions)-1
            configuration = session.trial_data['configuration_i'] 

            if j == 0:
                previous_session_config = configuration[0]

            elif configuration[0]!= previous_session_config:
                task += 1
               
                previous_session_config = configuration[0]  
    
                    
                X_task = np.concatenate(X_subj,0)
                choices_task = np.concatenate(choices_current_subj,0)
    
                choices_current_subj = []
                X_subj = []
                model = sm.Logit(choices_task,X_task)
                #model = OLS(0.5-choices_task,X_task)
                results = model.fit()
                results_task.append(results.params)
                
           
            if len(choices) > n*4:
                #if j >0 :if len(choices) > n+20:
                
                reward = session.trial_data['outcomes']
    
                configuration = session.trial_data['configuration_i'] 
                config_add.append(configuration[0])
                previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
                 
                previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1]
                 
                interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
                 
        
                choices_current = (choices[n:])
               
                ones = np.ones(len(interactions)).reshape(len(interactions),1)
                 
                X = (np.hstack([previous_rewards,previous_choices,interactions,ones]))
                
                X_subj.append(X)
                choices_current_subj.append(choices_current)
                
                if j == all_sessions:
                
                    task += 1
                    previous_session_config = configuration[0]  
        
                        
                    X_task = np.concatenate(X_subj,0)
                    choices_task = np.concatenate(choices_current_subj,0)
        
                    choices_current_subj = []
                    X_subj = []
                    model = sm.Logit(choices_task,X_task)
                    #model = OLS(choices_task,X_task)
                    results = model.fit()
                    results_task.append(results.params)
    
                
        print(task)  
        coef_subj.append(results_task)
        
    mean_t = np.mean(coef_subj,0)
    sqrt = np.std(coef_subj,0)/np.sqrt(9)
 

    plt.figure(figsize = (15,3))
    #plt.figure(figsize = (5,5))

    j = 0
    for i,ii in enumerate(mean_t):
        plt.subplot(1,10,i+1)
        j+=1
        plt.plot(np.arange(len(ii))[:-1], ii[:-1], color = 'grey')
        plt.fill_between(np.arange(len(ii))[:-1], ii[:-1]+sqrt[i][:-1], ii[:-1]- sqrt[i][:-1],alpha = 0.2, color = 'grey')
        plt.hlines(0, xmin = np.arange(len(ii))[:-1][0],xmax = np.arange(len(ii))[:-1][-1])
        length = len(np.arange(len(ii))[:-1])
        plt.xticks(np.arange(len(ii))[:-1],np.arange(length)+1)
        #plt.ylim(-1.2,0.45)
        sns.despine()
        plt.tight_layout()
        plt.xlabel('N back Task' + ' ' + str(j))
    return coef_subj
    
      
def recordings_n_back(m484, m479, m483, m478, m486, m480, m481, n =10):
    subj = [m484, m479, m483, m478, m486, m480, m481]
    n = n
    coef_subj = []
    se_subj = []
    for s in subj:
        s = np.asarray(s)
        date  = []
        for ses in s:
            date.append(ses.datetime)
            
        ind_sort = np.argsort(date)    
        s = s[ind_sort]
    
        X_subj = []
        choices_current_subj = []
        results_task = []
        task = 0
        config_add = []
        for j,session in enumerate(s): 
            choices = session.trial_data['choices']
                
            
            reward = session.trial_data['outcomes']

            configuration = session.trial_data['configuration_i'] 
            config_add.append(configuration[0])
            previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
             
            previous_choices = scipy.linalg.toeplitz(0.5-choices, np.zeros((1,n)))[n-1:-1]
             
            interactions = scipy.linalg.toeplitz((((0.5-choices)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            choices_current = (choices[n:])
           
            ones = np.ones(len(interactions)).reshape(len(interactions),1)
             
            X = (np.hstack([previous_rewards,previous_choices,interactions,ones]))
            
           
            model = sm.Logit(choices_current,X)
           # model = OLS(0.5-choices_current,X)
            results = model.fit()
            results_task.append(results.params)

      
        print(task)  
        coef_subj.append(results_task[:7])
        
    mean_t = np.mean(coef_subj,0)
    sqrt = np.std(coef_subj,0)/np.sqrt(9)
 

    plt.figure(figsize = (15,3))
    j = 0
    for i,ii in enumerate(mean_t):
        plt.subplot(1,7,i+1)
        j+=1
        plt.plot(np.arange(len(ii))[n*2:-1], ii[n*2:-1], color = 'grey')
        plt.fill_between(np.arange(len(ii))[n*2:-1], ii[n*2:-1]+sqrt[i][n*2:-1], ii[n*2:-1]- sqrt[i][n*2:-1],alpha = 0.2, color = 'grey')
        plt.hlines(0, xmin = np.arange(len(ii))[n*2:-1][0],xmax = np.arange(len(ii))[n*2:-1][-1])
        plt.ylim(-1.5,0.35)
        
        length = len(np.arange(len(ii))[n*2:-1])
        plt.xticks(np.arange(len(ii))[n*2:-1],np.arange(length))
        plt.xlabel('session' + ' ' +str(j))
        sns.despine()
        plt.tight_layout()
    