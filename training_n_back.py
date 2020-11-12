#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:36:48 2020

@author: veronikasamborska
"""

from sklearn.linear_model import LogisticRegression

from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import sys 
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pandas as pd
from palettable import wesanderson as wes

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

def rew_prev_behaviour__PFC_HP(PFC,HP,n):
    d = [PFC,HP]
   
    for l in d:
        dm = l['DM'][0]
        results_array = []
        std_err = []
        for  s, sess in enumerate(dm):
                    
            DM = dm[s] 
                   
            choices = DM[:,1]
                 
            reward = DM[:,2]
            
            previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
            
            previous_choices = scipy.linalg.toeplitz(choices-0.5, np.zeros((1,n)))[n-1:-1]
            
            interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
            
   
            choices_current = choices[n:]
            ones = np.ones(len(interactions)).reshape(len(interactions),1)
            
            X = np.hstack([previous_rewards,previous_choices,interactions,ones])
            
            model = sm.Logit(choices_current,X)
            results = model.fit()
            results_array.append(results.params)
            cov = results.cov_params()
            std_err.append(np.sqrt(np.diag(cov)))

   
    average = np.mean(results_array,0)
    std = np.std(results_array,0)/len(dm)
    pv = scipy.stats.ttest_1samp(results_array,0).pvalue
    plt.plot(pv[n*2:-1])
    max_y = np.max(average)
    min_y = np.min(average)-0.05
    
    plt.figure(figsize = (8,4))
    average = np.mean(results_array,0)
    c = wes.Royal2_5.mpl_colors
    plt.subplot(1,3,1)
    plt.plot(np.arange(len(average))[:n], average[:n], color = c[1])
    plt.fill_between(np.arange(len(average))[:n], average[:n]+std[:n], average[:n]- std[:n],alpha = 0.2, color =c[1])
    plt.hlines(0, xmin = np.arange(len(average))[:n][0],xmax = np.arange(len(average))[:n][-1])
    length = len(np.arange(len(average))[:n])
    plt.xticks(np.arange(len(average))[:n],np.arange(length))
    plt.ylim(min_y,max_y)
    plt.ylabel('Coefficient')
    plt.xlabel('n-back')
    sns.despine()
    
    plt.subplot(1,3,2)
    plt.plot(np.arange(len(average))[n:n*2], average[n:n*2], color = c[2])
    plt.fill_between(np.arange(len(average))[n:n*2], average[n:n*2]+std[n*2:-1], average[n:n*2]- std[n:n*2],alpha = 0.2, color =c[2])
    plt.hlines(0, xmin = np.arange(len(average))[n:n*2][0],xmax = np.arange(len(average))[n:n*2][-1])
    length = len(np.arange(len(average))[n:n*2])
    plt.xticks(np.arange(len(average))[n:n*2],np.arange(length))
    plt.ylim(min_y,max_y)
    plt.ylabel('Coefficient')
    plt.xlabel('n-back')
    sns.despine()
    
    plt.subplot(1,3,3)
    plt.plot(np.arange(len(average))[n*2:-1], average[n*2:-1], color = c[3])
    plt.fill_between(np.arange(len(average))[n*2:-1], average[n*2:-1]+std[n*2:-1], average[n*2:-1]- std[n*2:-1],alpha = 0.2, color =c[3])
    plt.hlines(0, xmin = np.arange(len(average))[n*2:-1][0],xmax = np.arange(len(average))[n*2:-1][-1])
    length = len(np.arange(len(average))[n*2:-1])
    plt.xticks(np.arange(len(average))[n*2:-1],np.arange(length))
    plt.ylim(min_y,max_y)
    plt.ylabel('Coefficient')
    plt.xlabel('n-back')
    sns.despine()
    plt.tight_layout()
    
    
    return average


def import_data():
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

    #HP_LFP,PFC_LFP, m484_LFP, m479_LFP, m483_LFP, m478_LFP, m486_LFP, m480_LFP, m481_LFP, all_sessions_LFP = ep.import_code(ephys_path,beh_path, lfp_analyse = 'True') 
    HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')

    exp = di.Experiment('/Users/veronikasamborska/Desktop/Veronika Backup/2018-12-12-Reversal_learning/data_pilot3')

def plotting_functions(m484, m479, m483, m478, m486, m480, m481,exp, n = 5):
   
    
    coef_subj =  runs_length(exp, subject_IDs ='all', n = n)
    
   # coef_subj = recordings_n_back(m484, m479, m483, m478, m486, m480, m481, n = n)
    coef_subj = np.asarray(coef_subj)
    rewards = coef_subj[:,:,:n]
    choices = coef_subj[:,:,n:n*2]
    choices_X_reward = coef_subj[:,:,n*2:-1]
    
     
   
    _1_back_ch = choices[:, :, 0]#[:,1:]
    _other_back_ch = np.mean(choices[:, :,1:],2)#[:,1:]
    _1_back_rew_ch = choices_X_reward[:,:,0]
    _other_back_rew_ch = np.mean(choices_X_reward[:,:,1:],2)
    _all_back_rew = np.mean(rewards,2)#[:,1:]
    
     
    # subject_id = np.tile(np.arange(7), 7)
    # fraction_id = np.zeros(7*7)
    # k = 0 
    # for n in range(10):
    #     fraction_id[n*7:n*7+7] = k
    #     k+=1
        
    
    subject_id = np.tile(np.arange(10), 9)
    fraction_id = np.zeros(90)
    k = 0 
    for n in range(10):
        fraction_id[n*9:n*9+9] = k
        k+=1
        
    _1_back = np.concatenate(_1_back_ch.T,0)
    _1_back = {'Data':_1_back,'Sub_id': subject_id,'cond': fraction_id}
    _1_back = pd.DataFrame.from_dict(data = _1_back)
    aovrm = AnovaRM(_1_back, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()
    _1_back = res.anova_table
    p_val_1_back = np.around(res.anova_table['Pr > F'][0])

    _other_back_ch = np.concatenate(_other_back_ch.T,0)
    _other_back_ch = {'Data':_other_back_ch,'Sub_id': subject_id,'cond': fraction_id}
    _other_back_ch = pd.DataFrame.from_dict(data = _other_back_ch)
    aovrm = AnovaRM(_other_back_ch, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()
    _other_back = res.anova_table

    p_val_other_back_ch = np.around(res.anova_table['Pr > F'][0])

    _1_back_rew_ch = np.concatenate(_1_back_rew_ch.T,0)
    _1_back_rew_ch = {'Data':_1_back_rew_ch,'Sub_id': subject_id,'cond': fraction_id}
    _1_back_rew_ch = pd.DataFrame.from_dict(data = _1_back_rew_ch)
    aovrm = AnovaRM(_1_back_rew_ch, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()
    _1_back_re_ch = res.anova_table

    p_val_1_back_rew_ch = np.around(res.anova_table['Pr > F'][0])

    _other_back_rew_ch = np.concatenate(_other_back_rew_ch.T,0)
    _other_back_rew_ch = {'Data':_other_back_rew_ch,'Sub_id': subject_id,'cond': fraction_id}
    _other_back_rew_ch = pd.DataFrame.from_dict(data = _other_back_rew_ch)
    aovrm = AnovaRM(_other_back_rew_ch, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()
    _other_back_reward_choice = res.anova_table

    p_val_1_back_other_back_rew_ch = np.around(res.anova_table['Pr > F'][0])

    
    
    _all_back_rew = np.concatenate(_all_back_rew.T,0)
    _all_back_rew = {'Data':_all_back_rew,'Sub_id': subject_id,'cond': fraction_id}
    _all_back_rew = pd.DataFrame.from_dict(data = _all_back_rew)
    aovrm = AnovaRM(_all_back_rew, depvar = 'Data',subject = 'Sub_id', within=['cond'])
    res = aovrm.fit()
    _back_rew = res.anova_table

    p_val_1_back_all_back_rew = np.around(res.anova_table['Pr > F'][0])

    
   
    
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
 
    isl = wes.Royal2_5.mpl_colors

    plt.figure(figsize = (10,4))

    plt.subplot(1,5,1)
    plt.errorbar(np.arange(len(_all_back_rew)), _all_back_rew, yerr=_all_back_rew_err, fmt='o', color = isl[0])
    plt.annotate(p_val_1_back_all_back_rew, xy = (10,np.max(_all_back_rew)+0.01))
    plt.xlim(-1,10)
    plt.title(' N Rewards Back')
    plt.xticks(np.arange(10),np.arange(10)+1)
    plt.xlabel('Task')
 
    plt.subplot(1,5,2)
    #sns.boxplot(data =_1_back_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_1_back_ch)), _1_back_ch, yerr=_1_back_ch_er, fmt='o', color = isl[3])
    plt.annotate(p_val_1_back, xy = (10,np.max(_1_back_ch)+0.01))
    plt.xlim(-1,10)
    plt.title(' 1 Choice Back')
    plt.xticks(np.arange(10),np.arange(10)+1)
    plt.xlabel('Task')
    plt.ylabel('Coefficient')

    plt.subplot(1,5,3)
    #sns.boxplot(data=_other_back_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_other_back_ch)), _other_back_ch, yerr=_other_back_ch_err, fmt='o', color = isl[3])
    plt.annotate(p_val_other_back_ch, xy = (10,np.max(_other_back_ch)+0.01))
    plt.xlim(-1,10)
    plt.title(' 2+ Choices Back')
    plt.xticks(np.arange(10),np.arange(10)+1)
    plt.xlabel('Task')
    plt.ylabel('Coefficient')

    plt.subplot(1,5,4)
    #sns.boxplot(data=_all_back_rew_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_1_back_rew_ch)), _1_back_rew_ch, yerr=_1_back_rew_ch_err, fmt='o', color = isl[4])
    plt.annotate(p_val_1_back_rew_ch, xy = (10,np.max(_1_back_rew_ch)+0.01))
    plt.xlim(-1,10)
    plt.title(' 1 Choice x Reward Back')
    plt.xticks(np.arange(10),np.arange(10)+1)
    plt.xlabel('Task')
    plt.ylabel('Coefficient')

  
    plt.subplot(1,5,5)
    #sns.boxplot(data=_all_back_rew_ch, palette="Set3",showfliers = False)
    plt.errorbar(np.arange(len(_all_back_rew_ch)), _all_back_rew_ch, yerr=_all_back_rew_ch_err, fmt='o', color = isl[4])
    plt.annotate(p_val_1_back_other_back_rew_ch, xy = (10,np.max(_all_back_rew_ch)+0.01))
    plt.xlim(-1,10)
    plt.xticks(np.arange(10),np.arange(10)+1)
    plt.xlabel('Task')
    plt.title(' 2 Choices x Rewards Back')
    plt.ylabel('Coefficient')

    sns.despine()
    plt.tight_layout()

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
        results_session =[]
        for j, session in enumerate(subject_sessions):
            choices = session.trial_data['choices']
            all_sessions = len(subject_sessions)-1
            configuration = session.trial_data['configuration_i'] 

            if j == 0:
                previous_session_config = configuration[0]

            elif configuration[0]!= previous_session_config:
                task += 1
               
                previous_session_config = configuration[0]  
    
                    
                # X_task = np.concatenate(X_subj,0)
                # choices_task = np.concatenate(choices_current_subj,0)
    
                # choices_current_subj = []
                # X_subj = []
                # model = sm.Logit(choices_task,X_task)
                # #model = OLS(0.5-choices_task,X_task)
                # results = model.fit()
                results_task.append(np.mean(results_session,0))
                
           
            
            if len(choices) > n*3:
                
                reward = session.trial_data['outcomes']
    
                configuration = session.trial_data['configuration_i'] 
                config_add.append(configuration[0])
                previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1]
                 
                previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1]
                 
                interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
                 
        
                choices_current = (choices[n:])
               
                ones = np.ones(len(interactions)).reshape(len(interactions),1)
                X = (np.hstack([previous_rewards,previous_choices,interactions,ones]))
                #if np.isfinite(np.linalg.cond(X)) == False:
                rank  =np.linalg.matrix_rank(X)
   

                #model = sm.Logit(choices_current,X)
                model = LogisticRegression(fit_intercept = False)
                #model = OLS(X,choices_current)
                results = model.fit(X,choices_current)
                #print(results.params)
                results_session.append(results.coef_[0])
               
                # X_subj.append(X)
                # choices_current_subj.append(choices_current)
            
                if j == all_sessions:
                
                    task += 1
                    previous_session_config = configuration[0]  
        
                        
                    # X_task = np.concatenate(X_subj,0)
                    # choices_task = np.concatenate(choices_current_subj,0)
        
                    # choices_current_subj = []
                    # X_subj = []
                    # model = sm.Logit(choices_task,X_task)
                    # #model = OLS(choices_task,X_task)
                    # results = model.fit()
                    #results_task.append(results.params)
                    results_task.append(np.mean(results_session,0))
                
        print(task)  
        coef_subj.append(results_task)
        
    mean_t = np.mean(coef_subj,0)
    sqrt = np.std(coef_subj,0)/np.sqrt(9)
    
    plt.figure(figsize = (15,3))
    #plt.figure(figsize = (5,5))
    isl = wes.Royal2_5.mpl_colors


    #rewards = coef_subj[:,:,:n]
   # choices = coef_subj[:,:,n:n*2]
    #choices_X_reward = coef_subj[:,:,n*2:-1]
    
     
   
    j = 0
    for i,ii in enumerate(mean_t):
       # plt.subplot(2,5,i+1)
        j += 1
        
        # Rewards
        plt.plot(np.arange(len(ii))[:n]+j*n, ii[:n], color = isl[0])
        plt.fill_between(np.arange(len(ii))[:n]+j*n, ii[:n]+sqrt[i][:n], ii[:n]- sqrt[i][:n],alpha = 0.2, color = isl[0])
        
        plt.plot(np.arange(len(ii))[n:n*2]+j*n+(n*10), ii[n:n*2], color = isl[3])
        plt.fill_between(np.arange(len(ii))[n:n*2]+j*n+(n*10), ii[n:n*2]+sqrt[i][n:n*2], ii[n:n*2]- sqrt[i][n:n*2],alpha = 0.2, color = isl[3])
        
        
        plt.plot(np.arange(len(ii))[n*2:-1]+j*n+(n*20), ii[n*2:-1], color = isl[4])
        plt.fill_between(np.arange(len(ii))[n*2:-1]+j*n+(n*20), ii[n*2:-1]+sqrt[i][n*2:-1], ii[n*2:-1]- sqrt[i][n*2:-1],alpha = 0.2, color = isl[4])
        if j == 10:
            plt.plot(np.arange(len(ii))[:n]+j*n, ii[:n], color = isl[0], label = 'N-back Rewards')
            plt.plot(np.arange(len(ii))[n:n*2]+j*n+(n*10), ii[n:n*2], color = isl[3],label = 'N-back Choics')
            plt.plot(np.arange(len(ii))[n*2:-1]+j*n+(n*20), ii[n*2:-1], color = isl[4], label = 'N-back Choice x Rewards')

            plt.legend()
        plt.ylabel('Coefficient')
        #plt.hlines(0, xmin = np.arange(len(ii))[:-1][0],xmax = np.arange(len(ii))[:-1][-1])
        length = len(np.arange(len(ii))[:-1])
   # plt.xticks(np.arange(10,306,10),[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10])
    #plt.ylim(-1.2,0.45)
    sns.despine()
    plt.xlabel('N back Task' + ' ' + str(j))
    plt.tight_layout()

    return coef_subj
    
      
def recordings_n_back(m484, m479, m483, m478, m486, m480, m481, n = 10):
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
             
            previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1]
             
            interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            choices_current = (choices[n:])
           
            ones = np.ones(len(interactions)).reshape(len(interactions),1)
             
            X = (np.hstack([previous_rewards,previous_choices,interactions,ones]))
            model = LogisticRegression()
            results = model.fit(X,choices_current)
            results_task.append(results.params)
              
           
           #  model = sm.Logit(choices_current,X)
           # # model = OLS(0.5-choices_current,X)
           #  results = model.fit()
           #  results_task.append(results.params)
               
      
        #print(task)  
        coef_subj.append(results_task[:7])
        
    mean_t = np.mean(coef_subj,0)
    sqrt = np.std(coef_subj,0)/np.sqrt(9)
 
    isl = wes.Royal2_5.mpl_colors
    plt.figure(figsize = (15,3))

    j = 0
    for i,ii in enumerate(mean_t):
       # plt.subplot(2,5,i+1)
        j += 1
        
        # Rewards
        plt.plot(np.arange(len(ii))[:n]+j*n, ii[:n], color = isl[0])
        plt.fill_between(np.arange(len(ii))[:n]+j*n, ii[:n]+sqrt[i][:n], ii[:n]- sqrt[i][:n],alpha = 0.2, color = isl[0])
        
        plt.plot(np.arange(len(ii))[n:n*2]+j*n+(n*7), ii[n:n*2], color = isl[3])
        plt.fill_between(np.arange(len(ii))[n:n*2]+j*n+(n*7), ii[n:n*2]+sqrt[i][n:n*2], ii[n:n*2]- sqrt[i][n:n*2],alpha = 0.2, color = isl[3])
        
        
        plt.plot(np.arange(len(ii))[n*2:-1]+j*n+(n*14), ii[n*2:-1], color = isl[4])
        plt.fill_between(np.arange(len(ii))[n*2:-1]+j*n+(n*14), ii[n*2:-1]+sqrt[i][n*2:-1], ii[n*2:-1]- sqrt[i][n*2:-1],alpha = 0.2, color = isl[4])
        if j == 10:
            plt.plot(np.arange(len(ii))[:n]+j*n, ii[:n], color = isl[0], label = 'N-back Rewards')
            plt.plot(np.arange(len(ii))[n:n*2]+j*n+(n*7), ii[n:n*2], color = isl[3],label = 'N-back Choics')
            plt.plot(np.arange(len(ii))[n*2:-1]+j*n+(n*14), ii[n*2:-1], color = isl[4], label = 'N-back Choice x Rewards')

            plt.legend()
        plt.ylabel('Coefficient')
        #plt.hlines(0, xmin = np.arange(len(ii))[:-1][0],xmax = np.arange(len(ii))[:-1][-1])
        length = len(np.arange(len(ii))[:-1])
   # plt.xticks(np.arange(10,306,10),[1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10])
    #plt.ylim(-1.2,0.45)
    sns.despine()
    plt.xlabel('N back Task' + ' ' + str(j))
    plt.tight_layout()

    return coef_subj