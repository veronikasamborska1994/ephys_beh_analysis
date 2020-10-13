#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:17:51 2020
X_3.s
@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def switch_number_of_rewards(HP,PFC):
    d = [HP,PFC]
    session = []

    for data in d:
        dm = data['DM'][0]
        firing = data['Data'][0]
        neurons = 0
        for s in firing:
            neurons += s.shape[1]
        tr_runs = 5
        switch = np.zeros((neurons,63,tr_runs));  
      
        n_neurons_cum = 0
        mean_rest  =[]
        for  s, sess in enumerate(dm):
            DM = dm[s]
            firing_rates = firing[s]
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            n_neurons_cum += n_neurons
            forced = DM[:,3]
            forced_ind = np.where(forced == 0)[0]
            choices = DM[:,1]#[forced_ind]
            reward = DM[:,2]#[forced_ind] 
        
            succ = 0 
            cum_reward_side = []
            for r,rew in enumerate(reward):
                if r < len(reward)-1:
                    if rew == 1:
                        if (reward[r] == reward[r-1]) and (choices[r] == choices[r-1]):
                            succ +=1
                        else:
                            succ = 0
                    else:
                        succ = 0
                    cum_reward_side.append(succ)
           
            #switch = 0 
            switch_list = []
            for c,ch in enumerate(choices):
                if c < len(choices)-1: 
                    if (choices[c] == choices[c+1]):
                        switch = 1
                    else:
                        switch = 0
                    switch_list.append(switch)
            rew_uniq = np.arange(11)
            
            probability = np.zeros(len(rew_uniq))
            probability[:] = np.NaN
            for i in rew_uniq:
                trials = (np.where(cum_reward_side==i)[0])+1
                if len(switch_list) in trials:
                    trials = trials[:-1]
                probability[i] = np.sum(np.asarray(switch_list)[trials])/len(np.asarray(switch_list)[trials])
            session.append(probability)
    plt.scatter(np.arange(len(np.nanmean(session,0))),np.nanmean(session,0))    
    plt.scatter(np.arange(len(np.nanstd(session,0))),np.nanmean(session,0)+np.nanstd(session,0))    
    plt.scatter(np.arange(len(np.nanstd(session,0))),np.nanmean(session,0)-np.nanstd(session,0))    

def rrrnrrnn(data):

    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    tr_runs = 5
    errors_1 = np.zeros((neurons,121,2));  
    errors_2 = np.zeros((neurons,121,2));  
    errors_3 = np.zeros((neurons,121,2));  

    n_neurons_cum = 0
    mean_pre_explore = []
    mean_rest  =[]
    
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
            
        no_rewards = np.where(np.asarray(cum_reward)==0)[0]
            
        more_than_1 = []
        just_1 = []
        last_rew_run_2 =[]
        last_rew_run_1 = []
        just_3 = []
        last_rew_run_3 = []
        
        for ind in no_rewards: 
            if ind > 0 and ind < (len(cum_reward)-3):
                if cum_reward[ind-1] >1 and cum_reward[ind+1] == 0 and cum_reward[ind+2] > 0:
                   more_than_1.append(ind+2)
                   last_rew_run_2.append(ind-1)
    
                elif cum_reward[ind-1] >1 and cum_reward[ind+1] > 0:
                    just_1.append(ind+1)
                    last_rew_run_1.append(ind-1)
                    
                elif cum_reward[ind-1] > 1 and cum_reward[ind+1] == 0 and cum_reward[ind+2] == 0 and cum_reward[ind+3] > 0:
                    just_3.append(ind+3)
                    last_rew_run_3.append(ind-1)
                    
        errors_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.mean(firing_rates[just_1],0)
        errors_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.mean(firing_rates[more_than_1],0)
        errors_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,0] = np.mean(firing_rates[just_3],0)

        errors_1[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.mean(firing_rates[last_rew_run_1],0)
        errors_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.mean(firing_rates[last_rew_run_2],0)
        errors_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,1] = np.mean(firing_rates[last_rew_run_3],0)

    
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    pal = sns.cubehelix_palette(8)

    one_error = np.nanmean(errors_1,0)
    two_errors = np.nanmean(errors_2,0)
    three_errors = np.nanmean(errors_3,0)
    
    plt.figure(figsize = (10,3))
    plt.subplot(1,3,1)
    plt.plot(one_error[:,0], label = 'First error after 1 reward', color = pal[1])
    plt.plot(one_error[:,1],label = 'Last error after 1 reward', color = pal[3])
    plt.legend(loc='lower right',prop={'size': 6})

    plt.subplot(1,3,2)
    plt.plot(two_errors[:,0],label = 'First error  2 reward', color = pal[1])
    plt.plot(two_errors[:,1],label  = 'Last error  2 reward', color = pal[3])
    plt.legend(loc='lower right',prop={'size': 6})
    sns.despine()
    
    
    plt.subplot(1,3,3)
    plt.plot(three_errors[:,0],label = 'First error after 3 reward', color = pal[1])
    plt.plot(three_errors[:,1],label  = 'Last error after 3 reward', color = pal[3])
    plt.legend(loc='lower right',prop={'size': 6})
    sns.despine()
   
    
    

def errors_before_switch(data):

    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    tr_runs = 5
    errors = np.zeros((neurons,121,tr_runs));  
    rewardS = np.zeros((neurons,121,tr_runs));  

    n_neurons_cum = 0
    mean_pre_explore = []
    mean_rest  =[]
    
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons
        choices = DM[:,1]
        reward = DM[:,2]    
        state = DM[:,0]

        stay=choices[0:-1]==choices[1:]
        stay = stay*1
        stay = np.insert(stay,0,0)
        lastreward = reward[0:-1]
        lastreward = np.insert(lastreward,0,0)
        cum_error = []
        err = 0
        
        state = DM[:,0]
        
        rl = np.zeros(len(stay))
        rl[0]=1
      
        rl_right = np.zeros(len(stay))
        rl_right[0]=choices[0]==state[0]
        choice_rr_start=-100
         
         
        rl_wrong=np.zeros(len(stay));
        rl_wrong[0]=choices[0]!=state[0];
        choice_rw_start=-100;
        
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

        for tr in range(len(stay)):
            if tr > 0: 
                if stay[tr] == 1:
                    rl[tr] = rl[tr-1]+1
                else:
                    rl[tr]=1
                
                
                if ((choices[tr] == choice_rr_start) & (choices[tr]==state[tr])):
                    rl_right[tr]=rl_right[tr-1]+1
                    
                elif (choices[tr]==state[tr]):
                    
                    rl_right[tr]=1;
                    choice_rr_start=choices[tr]
                else:
                    rl_right[tr]=0;
                    choice_rr_start =-100; #If he made the wrong choice it can't be part of a correct run. 
                
                
                if ((choices[tr]==choice_rw_start) & (choices[tr]!=state[tr])):
                    rl_wrong[tr]=rl_wrong[tr-1]+1
                    
                elif choices[tr]!=state[tr]:
                    rl_wrong[tr]=1
                    choice_rw_start=choices[tr]
                else:
                    rl_wrong[tr] = 0;
                    choice_rw_start=-100 #If he made the right choice it can't be part of a wrong run. 
        
      
        
        explore = np.where(rl == 1)[0]
        explore_not  = np.where(rl!= 1)[0]
        
        mean_pre_explore.append(np.asarray(np.asarray(cum_error)[explore-1]))
        mean_rest.append(np.mean(np.asarray(cum_error)[explore_not]))
        
        for i in range(tr_runs):
            errors[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(cum_error) == i+1))[0]],0)
            rewardS[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(cum_reward) == i+1))[0]],0)

    runs_er = []
    runs_rew = []
    for i in range(tr_runs):
     
        runs_er.append(np.mean(errors[:,:,i],0))
        runs_rew.append(np.mean(rewardS[:,:,i],0))

    #plt.figure(figsize = (7,3))
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    pal = sns.cubehelix_palette(8)

    # plt.subplot(1,2,1)
    # for i,r in enumerate(runs_er):
    #     plt.plot(r, color = pal_c[i], label = str(i) + ' '+'Incorrect Run')
    # plt.ylabel('FR')
    # plt.xlabel('Time in Trial')

    # plt.legend()
    # sns.despine()
    mean_r  =[]; mean_er  =[]

    for r in runs_rew:
        mean_r.append(np.mean(r[:20],0))
    for r in runs_er:
        mean_er.append(np.mean(r[:20],0))
   
    plt.figure(figsize = (7,3))

    # plt.subplot(1,2,2)
    plt.scatter(np.arange(len(mean_r)),mean_er, color = pal[:len(mean_r)],label = 'Error')
    plt.scatter(np.arange(len(mean_r))+5,mean_r, color = pal_c[:len(mean_r)], label = 'Rew')

    sns.despine()
   # corr = np.round(np.corrcoef(np.arange(len(mean_r)),mean_r)[0,1],2)
    #plt.annotate('r =' + str(corr), xy = (0,7))
    plt.ylabel('Pre Init FR')
    plt.xlabel('Run #')
    plt.legend()
    plt.tight_layout()
    plt.xticks([0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,1,2,3,4,5])
    
    # rewardS_svd = np.transpose(rewardS,[0,2,1]); rewardS_svd = rewardS_svd.reshape(rewardS_svd.shape[0], rewardS_svd.shape[1]*rewardS_svd.shape[2])
    # errors_svd = np.transpose(errors,[0,2,1]); errors_svd = errors_svd.reshape(errors_svd.shape[0], errors_svd.shape[1]*errors_svd.shape[2])
   
    rewardS_svd = np.mean(rewardS[:,:25,:],1)
    errors_svd = np.mean(errors[:,:25,:],1)
    
    all_err_rew =np.concatenate((rewardS_svd, errors_svd),1)
    all_err_rew = all_err_rew[~np.isnan(all_err_rew).any(axis=1)]

    u,s,v = np.linalg.svd(all_err_rew)
    #v = v.reshape((v.shape[0],a_ind_error.shape[1], a_ind_error.shape[2]*2))
    pal = sns.cubehelix_palette(10)
    pal_c = sns.cubehelix_palette(20, start=2, rot=0, dark=0, light=.95)
    
    ind = 10
    plt.figure(figsize = [5,10])
    
    for i in range(10):
        
        plt.subplot(5,2,i+1)
        plt.scatter(np.arange(len(v[i,:5])),v[i,5:], color = pal_c[:5],label = 'Errors')
        plt.scatter(np.arange(len(v[i,5:]))+5,v[i,:5], color = pal[:5], label =  'Rewards')

    sns.despine()
    plt.legend()
   

    
    
def errors_training(experiment,subject_IDs ='all'):
    
   # Define variables
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
    sessions_block = []
    tasks = 10 # Maximum number of tasks
    errors_subj = np.ones((9,10))
    errors_subj[:] = np.NaN
    # Looping through subjects and sessions 
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs = [subject_ID])
        task_number = 0 # Current task number
        previous_session_config = 0 
        subject_sessions = experiment.get_sessions(subject_ID)
        
        errors_current = []
        task_number_list = []
        for j, session in enumerate(subject_sessions):
            ft = session.trial_data['forced_trial'] 
            ft_ind = np.where(ft == 0)[0]
            sessions_block = session.trial_data['block'][ft_ind]
            choices = session.trial_data['choices'][ft_ind]
            state = session.trial_data['state'][ft_ind]
            reward = session.trial_data['outcomes'][ft_ind]

            # Find trials where threshold crossed.
            #block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            correct = np.where(state == choices)[0]
            incorrect = np.where(state != choices)[0]
          
            configuration = session.trial_data['configuration_i'] 
            stay=choices[0:-1]==choices[1:]
            stay = stay * 1
            stay = np.insert(stay,0,0)
      
            err = 0
            cum_error = []

            for r in reward:
                if r == 0:
                   err+=1
                else:
                    err = 0
                cum_error.append(err)

            runs_list = [0]
            run = 0
            for c, ch in enumerate(choices):
                if c > 0:
                    if choices[c] == choices[c-1]:
                        run += 1
                    elif choices[c] != choices[c-1]:
                        run = 0
                    runs_list.append(run)
              
            runs_list = np.asarray(runs_list)
            explore = np.where(runs_list == 0)[0]
            
            
            runs_list_corr = []
            runs_list_incorr = []
            corr_run = 0
            run_ind_c =[]
            for c, ch in enumerate(choices):
                if c > 0  and c in correct:
                    if choices[c] == choices[c-1]:
                        if corr_run == 0:
                            run_ind_c.append(c)
                        corr_run +=1
                    elif choices[c] != choices[c-1]:
                        corr_run = 0
                else:
                    corr_run = 0
                runs_list_corr.append(corr_run)
             
            incorr_run = 0
            run_ind_inc = []
            for c, ch in enumerate(choices):
                if c > 0  and c in incorrect:
                    if choices[c] == choices[c-1]:
                        if incorr_run ==0:
                            run_ind_inc.append(c)
                        incorr_run +=1
                    elif choices[c] != choices[c-1]:
                        incorr_run = 0
                else:
                    incorr_run = 0
                    
                runs_list_incorr.append(incorr_run)
           
            if 0 in explore:
                explore = explore[1:]
                

            if configuration[0]!= previous_session_config:
                task_number += 1
                previous_session_config = configuration[0]  
            
            errors_current += list(np.asarray(cum_error)[explore-1])
            task_number_list += [task_number] * len(list(np.asarray(cum_error)[explore-1]))
            
        for t in np.unique(task_number_list):
            errors_subj[n_subj,t-1] = np.nanmean((np.asarray(errors_current)[np.where(np.asarray(task_number_list) == t)]))
    
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    pal = sns.cubehelix_palette(8)
    #plt.subplot(1,2,2)
    pre_explore_subj_mean = np.nanmean(np.asarray(errors_subj),0)
    pre_explore_subj_se = np.nanstd(np.asarray(errors_subj),0)/np.sqrt(9)
    plt.fill_between(np.arange(len(pre_explore_subj_mean)),\
                        pre_explore_subj_mean+pre_explore_subj_se ,pre_explore_subj_mean-pre_explore_subj_se, alpha = 0.3,color =  pal[3])
    plt.plot(np.arange(len(pre_explore_subj_mean)),\
                        pre_explore_subj_mean, color = pal[4])
    
    #sns.regplot(np.arange(len(pre_explore_subj_mean)),pre_explore_subj_mean, color = 'pink',fit_reg = True)
    sns.despine()
    plt.xlabel('Task #')
    plt.ylabel('Error Before Explore')

   # Exporting data as excel files using pandas to reversal_learning folder
      
