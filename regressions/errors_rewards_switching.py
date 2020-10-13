#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:50:45 2020

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
import regression_function as reg_f
from collections import OrderedDict
import regressions as re
import palettable.wesanderson as wes
from matplotlib.backends.backend_pdf import PdfPages

def sequence_rewards_errors_regression(data, perm = True):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    C = []
    cpd = []
   
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape

        choices = DM[:,1]
        reward = DM[:,2]    
        state = DM[:,0]

        cum_error = []
        
        err = 0
        for r in reward:
            if r == 0:   
                err+=1
            else:
                err = 0
            cum_error.append(err)
        
        cum_reward = []
        for r in reward:
            if r == 1:
                err+=1
            else:
                err = 0
            cum_reward.append(err)
             
                
       
        ones = np.ones(len(reward))
        
        predictors_all = OrderedDict([('Reward', reward),
                                      ('Choice', choices),
                                      ('State', state),
                                      ('Errors', cum_error),
                                      ('Rewards',cum_reward),
                                      ('ones', ones)])
            
               
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        
        C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
        
      
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    high_loadings_rewards = np.where(abs(np.mean(C[4,:,:20],1)) > 2.5)[0]
    high_loadings_errors = np.where(abs(np.mean(C[3,:,:20],1)) > 2.5)[0]
    
    return high_loadings_errors,high_loadings_rewards


def sequences_rewards_errors(data):
    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
   
    errors_1 = np.zeros((neurons,63,3));  
    errors_2 = np.zeros((neurons,63,3));  
    errors_3 = np.zeros((neurons,63,3));  

    rewards_1 = np.zeros((neurons,63,3));  
    rewards_2 = np.zeros((neurons,63,3));  
    rewards_3 = np.zeros((neurons,63,3));  
    
    errors_1_std = np.zeros((neurons,63,3));  
    errors_2_std = np.zeros((neurons,63,3));  
    errors_3_std = np.zeros((neurons,63,3));  

    rewards_1_std = np.zeros((neurons,63,3));  
    rewards_2_std = np.zeros((neurons,63,3));  
    rewards_3_std = np.zeros((neurons,63,3));  

    n_neurons_cum = 0
    
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons

        choices = DM[:,1]
        reward = DM[:,2]    

        stay=choices[0:-1]==choices[1:]
        stay = stay*1
        stay = np.insert(stay,0,0)
        lastreward = reward[0:-1]
        lastreward = np.insert(lastreward,0,0)
        cum_error = []
        
        err = 0
        for r in reward:
            if r == 0:
                err+=1
            else:
                err = 0
            cum_error.append(err)
        
        cum_reward = []
        for r in reward:
            if r == 1:
                err+=1
            else:
                err = 0
            cum_reward.append(err)
            
        rewards = np.where(np.asarray(cum_reward)==0)[0]
        errors = np.where(np.asarray(cum_error) == 0)[0]

        _2_rewards_last = []; _2_rewards_first = []
        _1_rewards_last = []; _1_rewards_first = [] 
        _3_rewards_last = []; _3_rewards_first = []
        _1_rewards_1 = [];  _2_rewards_1 = [];  _3_rewards_1 = []

         
        for ind in rewards: 
            
            if ind > 0 and ind < (len(cum_reward)-3):
                
                if cum_reward[ind-1] > 1 and cum_reward[ind+1] == 0 and cum_reward[ind+2] > 0 :
                   #and choices[ind-1] == choices[ind] and choices[ind] == choices[ind+1] and choices[ind] != choices[ind+2]:  # 2 or more rewards/errors before  2 error/reward

                   _2_rewards_first.append(ind+2)
                   _2_rewards_last.append(ind-1)
                   _2_rewards_1.append(ind-(cum_reward[ind-1]))

                if cum_reward[ind-1] > 1 and cum_reward[ind+1] > 0:
                    #and choices[ind-1] == choices[ind] and choices[ind] != choices[ind+1]:  # 2 or more rewards/errors before  2 error/reward
                   
                    _1_rewards_first.append(ind+1)
                    _1_rewards_last.append(ind-1)
                    _1_rewards_1.append(ind-(cum_reward[ind-1]))
                    
                if cum_reward[ind-1] > 1 and cum_reward[ind+1] == 0 and cum_reward[ind+2] == 0 and cum_reward[ind+3] > 0:
                    #and choices[ind-1] == choices[ind] and choices[ind] == choices[ind+1] and choices[ind] == choices[ind+2]\
                    #and choices[ind] != choices[ind+3]:  # 2 or more rewards/errors before  2 error/reward

                    _3_rewards_first.append(ind+3)
                    _3_rewards_last.append(ind-1)
                    _3_rewards_1.append(ind-(cum_reward[ind-1]))

        _2_errors_last = []; _2_errors_first = []
        _1_errors_last = []; _1_errors_first = [] 
        _3_errors_last = []; _3_errors_first = []
        _1_errors_1 = [];  _2_errors_1 = [];  _3_errors_1 = []
        
        
        for ind in errors: 
            
            if ind > 0 and ind < (len(cum_error)-3):
                
                if cum_error[ind-1] > 1 and cum_error[ind+1] == 0 and cum_error[ind+2] > 0:
                    #and choices[ind-1] == choices[ind] and choices[ind] == choices[ind+1] and choices[ind] != choices[ind+2]:  # 2 or more rewards/errors before  2 error/reward

                   _2_errors_first.append(ind+2)
                   _2_errors_last.append(ind-1)
                   _2_errors_1.append(ind-(cum_error[ind-1]))
                   
    
                if cum_error[ind-1] > 1 and cum_error[ind+1] > 0:
                    #and choices[ind-1] == choices[ind] and choices[ind] != choices[ind+1]:  # 2 or more rewards/errors before  2 error/reward

                    _1_errors_first.append(ind+1)
                    _1_errors_last.append(ind-1)
                    _1_errors_1.append(ind-(cum_error[ind-1]))

                    
                if cum_error[ind-1] > 1 and cum_error[ind+1] == 0 and cum_error[ind+2] == 0 and cum_error[ind+3] > 0:
                    #and choices[ind-1] == choices[ind] and choices[ind] == choices[ind+1] and choices[ind] == choices[ind+2]:
                    #and choices[ind] != choices[ind+3]:  # 2 or more rewards/errors before  2 error/reward
                    
                    _3_errors_first.append(ind+3)
                    _3_errors_last.append(ind-1)
                    _3_errors_1.append(ind-(cum_error[ind-1]))

                    
     
    
        errors_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanmean(firing_rates[_1_errors_first],0)
        errors_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanmean(firing_rates[_2_errors_first],0)
        errors_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanmean(firing_rates[_3_errors_first],0)

        errors_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanmean(firing_rates[_1_errors_last],0)
        errors_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanmean(firing_rates[_2_errors_last],0)
        errors_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanmean(firing_rates[_3_errors_last],0)
        
        errors_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,2] = np.nanmean(firing_rates[_1_errors_1],0)
        errors_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,2] = np.nanmean(firing_rates[_2_errors_1],0)
        errors_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,2] = np.nanmean(firing_rates[_3_errors_1],0)

    
  
        rewards_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanmean(firing_rates[_1_rewards_first],0)
        rewards_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanmean(firing_rates[_2_rewards_first],0)
        rewards_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanmean(firing_rates[_3_rewards_first],0)

        rewards_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanmean(firing_rates[_1_rewards_last],0)
        rewards_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanmean(firing_rates[_2_rewards_last],0)
        rewards_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanmean(firing_rates[_3_rewards_last],0)
        
        
        rewards_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,2] = np.nanmean(firing_rates[_1_rewards_1],0)
        rewards_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,2] = np.nanmean(firing_rates[_2_rewards_1],0)
        rewards_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,2] = np.nanmean(firing_rates[_3_rewards_1],0)

        # errors_1_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanstd(firing_rates[_1_errors_first],0)/np.sqrt(firing_rates[_1_errors_first].shape[0])
        # errors_2_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanstd(firing_rates[_2_errors_first],0)/np.sqrt(firing_rates[_2_errors_first].shape[0])
        # errors_3_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanstd(firing_rates[_3_errors_first],0)/np.sqrt(firing_rates[_2_errors_first].shape[0])

        # errors_1_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanstd(firing_rates[_1_errors_last],0)/np.sqrt(firing_rates[_1_errors_last].shape[0])
        # errors_2_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanstd(firing_rates[_2_errors_last],0)/np.sqrt(firing_rates[_2_errors_last].shape[0])
        # errors_3_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanstd(firing_rates[_3_errors_last],0)/np.sqrt(firing_rates[_3_errors_last].shape[0])

    
  
        # rewards_1_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanstd(firing_rates[_1_rewards_first],0)/np.sqrt(firing_rates[_1_rewards_first].shape[0])
        # rewards_2_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanstd(firing_rates[_2_rewards_first],0)/np.sqrt(firing_rates[_2_rewards_first].shape[0])
        # rewards_3_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.nanstd(firing_rates[_3_rewards_first],0)/np.sqrt(firing_rates[_3_rewards_first].shape[0])

        # rewards_1_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanstd(firing_rates[_1_rewards_last],0)/np.sqrt(firing_rates[_1_rewards_last].shape[0])
        # rewards_2_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanstd(firing_rates[_2_rewards_last],0)/np.sqrt(firing_rates[_2_rewards_last].shape[0])
        # rewards_3_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.nanstd(firing_rates[_3_rewards_last],0)/np.sqrt(firing_rates[_3_rewards_last].shape[0])
    
    
    #high_loadings_errors,high_loadings_rewards = sequence_rewards_errors_regression(data, perm = True)
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
  

    plt.figure(figsize = (10,3))
    plt.subplot(2,3,1)
    plt.plot(np.nanmean(errors_1,0)[:,0], label = 'First error after 1 reward', color = pal[1])
    plt.plot(np.nanmean(errors_1,0)[:,1],label = 'Last error before 1 reward', color = pal[3])
    plt.plot(np.nanmean(rewards_1,0)[:,2],label  = 'First reward in a Run', color = pal[3], linestyle = '--')

    plt.legend(loc='lower right',prop={'size': 6})

    plt.subplot(2,3,2)
    plt.plot(np.nanmean(errors_2,0)[:,0],label = 'First error after 2 reward', color = pal[1])
    plt.plot(np.nanmean(errors_2,0)[:,1],label  = 'Last error  before 2 reward', color = pal[3])
    plt.plot(np.nanmean(rewards_2,0)[:,2],label  = 'First reward in a Run', color = pal[3], linestyle = '--')

    plt.legend(loc='lower right',prop={'size': 6})
    sns.despine()
    
    
    plt.subplot(2,3,3)
    plt.plot(np.nanmean(errors_3,0)[:,0],label = 'First error after 3 reward', color = pal[1])
    plt.plot(np.nanmean(errors_3,0)[:,1],label  = 'Last error before 3 reward', color = pal[3])
    plt.plot(np.nanmean(rewards_3,0)[:,2],label  = 'First reward in a run', color = pal[3], linestyle = '--')

    plt.legend(loc='lower right',prop={'size': 6})
    sns.despine()
   
    
    plt.subplot(2,3,4)
    plt.plot(np.nanmean(rewards_1,0)[:,0], label = 'First reward after 1 error', color = pal_c[1])
    plt.plot(np.nanmean(rewards_1,0)[:,1],label = 'Last reward before 1 error', color = pal_c[3])
    plt.plot(np.nanmean(errors_1,0)[:,2],label  = 'First error in a run', color = pal_c[3], linestyle = '--')

    plt.legend(loc='lower right',prop={'size': 6})

    plt.subplot(2,3,5)
    plt.plot(np.nanmean(rewards_2,0)[:,0],label = 'First reward after 2 error', color = pal_c[1])

    # plt.fill_between(np.arange(63),\
    #                     np.nanmean(rewards_3,0)[:,0]+np.nanstd(rewards_2,0)[:,0]/np.sqrt(rewards_2.shape[0]) , np.nanmean(rewards_2,0)[:,0]-np.nanstd(rewards_2,0)[:,0]/np.sqrt(rewards_2.shape[0]), alpha = 0.3,color =  pal_c[1])
  
    plt.plot(np.nanmean(rewards_2,0)[:,1],label  = 'Last reward  before 2 error', color = pal_c[3])
    plt.plot(np.nanmean(errors_2,0)[:,2],label  = 'First error in a run', color = pal_c[3], linestyle = '--')

    # plt.fill_between(np.arange(63),\
    #                     np.nanmean(rewards_2,0)[:,1]+np.nanstd(rewards_2,0)[:,1]/np.sqrt(rewards_2.shape[0]) , np.nanmean(rewards_2,0)[:,1]-np.nanstd(rewards_2,0)[:,1]/np.sqrt(rewards_2.shape[0]), alpha = 0.3,color =  pal_c[3])
  
    plt.legend(loc='lower right',prop={'size': 6})
    sns.despine()
    
    
    plt.subplot(2,3,6)
    plt.plot(np.nanmean(rewards_3,0)[:,0],label = 'First reward after 3 error', color = pal_c[1])
    # plt.fill_between(np.arange(63),\
    #                     np.nanmean(rewards_3,0)[:,0]+np.nanstd(rewards_3,0)[:,0]/np.sqrt(rewards_2.shape[0]) , np.nanmean(rewards_3,0)[:,0]-np.nanstd(rewards_3,0)[:,0]/np.sqrt(rewards_2.shape[0]), alpha = 0.3,color =  pal_c[1])
  
    plt.plot(np.nanmean(rewards_3,0)[:,1],label  = 'Last reward before 3 error', color = pal_c[3])
    # plt.fill_between(np.arange(63),\
    #                     np.nanmean(rewards_3,0)[:,1]+np.nanstd(rewards_3,0)[:,1]/np.sqrt(rewards_2.shape[0]) , np.nanmean(rewards_3,0)[:,1]-np.nanstd(rewards_3,0)[:,1]/np.sqrt(rewards_2.shape[0]), alpha = 0.3,color =  pal_c[3])
    plt.plot(np.nanmean(errors_3,0)[:,2],label  = 'First error in a run', color = pal_c[3], linestyle = '--')

    plt.legend(loc='lower right',prop={'size': 6})
    sns.despine()
    
   
    # for ind,i in enumerate(rewards_3[high_loadings_rewards]):
    #     plt.figure()
    #     plt.plot(i[:,0],label = 'First reward before 3 error', color = pal_c[1])
    #     plt.fill_between(np.arange(len(i[:,0])),\
    #                     i[:,0]+rewards_3_std[ind,:,0] ,i[:,0]-rewards_3_std[ind,:,0], alpha = 0.3,color =  pal_c[1])
  
    #     plt.plot(i[:,1],label = 'Last reward after 3 error', color = pal_c[3])
    #     plt.fill_between(np.arange(len(i[:,1])),\
    #                     i[:,1]+rewards_3_std[ind,:,1] ,i[:,1]-rewards_3_std[ind,:,1], alpha = 0.3,color =  pal_c[3])
  
    return errors_1,errors_2,errors_3,rewards_1,rewards_2,rewards_3


def plot_all_cells(data):
    
    errors_1,errors_2,errors_3,rewards_1,rewards_2,rewards_3 = sequences_rewards_errors(HP)

    area = 'HP_runs_rewards'
    pdf = PdfPages('/Users/veronikasamborska/Desktop/'+ area+'.pdf')

    neuron = errors_1.shape[0]
    plt.ion()
    plt.figure()

    #high_loadings_errors,high_loadings_rewards = sequence_rewards_errors_regression(data, perm = True)
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
   
    subplot = 0
    for i in range(neuron):
       # if i in high_loadings_rewards:
        subplot += 1 
        if subplot == 16:
            #plt.legend()
            pdf.savefig()
            plt.clf()
            subplot -= 15


            
        plt.subplot(4,4,subplot)
         
        
        # plt.plot(rewards_1[i,:,0], color = pal_c[0],label = 'First Reward After 1 Error')
        # plt.plot(rewards_1[i,:,1], color = pal_c[0],label = 'Last Reward Before 1 Error', linestyle = '--')
        # #plt.plot(rewards_1[i,:,2], color = pal_c[0],label = 'First Reward in a run', linestyle = ':')

        # plt.plot(rewards_2[i,:,0], color = pal_c[2],label = 'First Reward After 2 Errors')
        # plt.plot(rewards_2[i,:,1], color = pal_c[2],label = 'Last Reward Before 2 Errors', linestyle = '--')
        # #plt.plot(rewards_2[i,:,2], color = pal_c[2],label = 'First Reward in a run', linestyle = ':')

        plt.plot(rewards_3[i,:,0],label = 'First reward after 3 error', color = pal_c[1])
        plt.plot(rewards_3[i,:,1],label  = 'Last reward before 3 error', color = pal_c[3])
        plt.plot(errors_3[i,:,2], label  = 'First error in a run', color = pal_c[3], linestyle = '--')
        sns.despine()

   
    pdf.close()


   
def svd_rewards_errors(data):
    
    errors_1,errors_2,errors_3,rewards_1,rewards_2,rewards_3 = sequences_rewards_errors(data)
    
    errors_1 = np.transpose(errors_1,[0,2,1]); errors_1 = errors_1.reshape(errors_1.shape[0], errors_1.shape[1]*errors_1.shape[2])
    errors_2 = np.transpose(errors_2,[0,2,1]); errors_2 = errors_2.reshape(errors_2.shape[0], errors_2.shape[1]*errors_2.shape[2])
    errors_3 = np.transpose(errors_3,[0,2,1]); errors_3 = errors_3.reshape(errors_3.shape[0], errors_3.shape[1]*errors_3.shape[2])

    rewards_1 = np.transpose(rewards_1,[0,2,1]); rewards_1 = rewards_1.reshape(rewards_1.shape[0], rewards_1.shape[1]*rewards_1.shape[2])
    rewards_2 = np.transpose(rewards_2,[0,2,1]); rewards_2 = rewards_2.reshape(rewards_2.shape[0], rewards_2.shape[1]*rewards_2.shape[2])
    rewards_3 = np.transpose(rewards_3,[0,2,1]); rewards_3 = rewards_3.reshape(rewards_3.shape[0], rewards_3.shape[1]*rewards_3.shape[2])

    # errors_2 = np.mean(errors_2[:,:25,],1)
    # errors_3 = np.mean(errors_3[:,:25,:],1)
    # rewards_1 = np.mean(rewards_1[:,:25,:],1)
    # rewards_2 = np.mean(rewards_2[:,:25,:],1)
    # rewards_3 = np.mean(rewards_3[:,:25,:],1)
    
    all_err_rew =np.concatenate((errors_1, rewards_1, errors_2, rewards_2, errors_3,rewards_3),1)
    all_err_rew = all_err_rew[~np.isnan(all_err_rew).any(axis=1)]

    u,s,v = np.linalg.svd(all_err_rew)
    #v = v.reshape((v.shape[0],a_ind_error.shape[1], a_ind_error.shape[2]*2))
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    
    ind = 18
    pc = 1
    t_length = 63
    
    plt.figure(figsize = (10,3))
    plt.subplot(1,3,1)
    plt.plot(v[pc,:int(v.shape[1]/ind)], color = pal[0])
    plt.plot(v[pc,t_length*3: t_length*4], color = pal_c[0])
   
    plt.subplot(1,3,2)
    plt.plot(v[pc,int(v.shape[1]/ind)*2:int(v.shape[1]/ind)*3], color = pal[1])
    plt.plot(v[pc,int(v.shape[1]/ind)*3:int(v.shape[1]/ind)*4], color = pal[1], linestyle = '--')
    
    plt.subplot(1,3,3)
    plt.plot(v[pc,int(v.shape[1]/ind)*4:int(v.shape[1]/ind)*5], color = pal[1])
    plt.plot(v[pc,int(v.shape[1]/ind)*5:int(v.shape[1]/ind)*6], color = pal[1], linestyle = '--')
  
    plt.subplot(1,3,1)
    plt.plot(v[pc,int(v.shape[1]/ind)*6:int(v.shape[1]/ind)*7], color = pal_c[1])
    plt.plot(v[pc,int(v.shape[1]/ind)*7:int(v.shape[1]/ind)*8], color = pal_c[1], linestyle = '--')
  
    plt.subplot(1,3,2)
    plt.plot(v[pc,int(v.shape[1]/ind)*8:int(v.shape[1]/ind)*9], color = pal_c[1])
    plt.plot(v[pc,int(v.shape[1]/ind)*9:int(v.shape[1]/ind)*10], color = pal_c[1], linestyle = '--')
    
    plt.subplot(1,3,3)
    plt.plot(v[pc,int(v.shape[1]/ind)*10:int(v.shape[1]/ind)*11], color = pal_c[1])
    plt.plot(v[pc,int(v.shape[1]/ind)*11:int(v.shape[1]/ind)*12], color = pal_c[1], linestyle = '--')
  
       
    sns.despine()
    #plt.legend()
   


    