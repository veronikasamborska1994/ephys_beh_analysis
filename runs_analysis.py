#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:25:38 2020

@author: veronikasamborska
"""


from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import sys 
from matplotlib.cbook import flatten
#import utility as ut
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
from collections import OrderedDict

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)


def run():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')

def residuals(data):
    dm = data['DM'][0]
    firing = data['Data'][0]
    C_1 = []; C_2 = []; C_3 = []
    res_list = []
    counts_l = []
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
       # firing_rates = firing_rates[:,:,:63]
        n_trials, n_neurons, n_timepoints = firing_rates.shape

        choices = DM[:,1]
        
        reward = DM[:,2]    

        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]
        
        task = task[3:]
        a_pokes = a_pokes[3:]
        b_pokes = b_pokes[3:]
        
        
        reward_2_ago = reward[1:-2]
        reward_3_ago = reward[:-3]
        reward_prev = reward[2:-1]
        reward_current = reward[3:]
        
        ones = np.ones(len(reward_2_ago))
        
        error_count = []
        err = 0
        for r,run in enumerate(reward_current):
            if reward_current[r] == 0 and reward_current[r-1] == 0:
                err+=1
            else:
                err = 0
            error_count.append(err)
            
        reward_count = []
        err = 0
        for r,run in enumerate(reward_current):
            if reward_current[r] == 1 and reward_current[r-1] == 1:
                err+=1
            else:
                err = 0
            reward_count.append(err)
        counts_l.append([reward_count,error_count])
        
        firing_rates = firing_rates[3:]
        predictors_all = OrderedDict([#('Reward', reward_current),
                                      ('1 ago Outcome', reward_prev),
                                      ('2 ago Outcome', reward_2_ago),
                                      ('3 ago Outcome', reward_3_ago),                                 
                                      ('ones', ones)])   
            
        X = np.vstack(predictors_all.values()).T[:len(reward),:].astype(float)
       
        Y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]

        pdes = np.linalg.pinv(X)
        pe = np.matmul(pdes,Y)
        res = Y - np.matmul(X,pe)
        res_list.append(res.reshape(firing_rates.shape[0],firing_rates.shape[1],firing_rates.shape[2])) # Predictor loadings
    return res_list,counts_l


def extract_error_count(data):
    
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    tr_runs = 4
    error_counts = np.zeros((neurons,121,tr_runs));  reward_counts = np.zeros((neurons,121,tr_runs))
    n_neurons_cum = 0
    res_list,counts_l = residuals(data)

    for  s, sess in enumerate(res_list):
        
        firing_rates  = res_list[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape

        counts_s = counts_l[s]
        error_count = counts_s[1]
        reward_count = counts_s[0]
        n_neurons_cum += n_neurons

        for i in range(tr_runs):
         
            error_counts[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(error_count) == i+1))[0]],0)
            reward_counts[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(reward_count) == i+1))[0]],0)

    return error_counts,reward_counts
 
    
 
def tim_extract(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    tr_runs = 4
    a_ind_right_1_2 = np.zeros((neurons,63,tr_runs));  b_ind_right_1_2 = np.zeros((neurons,63,tr_runs))
    a_ind_wr_1_2 = np.zeros((neurons,63,tr_runs));  b_ind_wr_1_2 = np.zeros((neurons,63,tr_runs))
   
    a_ind_right_3 = np.zeros((neurons,63,tr_runs));  b_ind_right_3 = np.zeros((neurons,63,tr_runs))
    a_ind_wr_3 = np.zeros((neurons,63,tr_runs));  b_ind_wr_3 = np.zeros((neurons,63,tr_runs))
   
    
    #ind_right_std =  np.zeros((neurons,63,tr_runs)); ind_wrong_std = np.zeros((neurons,63,tr_runs)); 

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
        state = DM[:,0]
        task = DM[:,5]
        
        task[np.where(task ==2)[0]]=1
        stay=choices[0:-1]==choices[1:]
        stay = stay*1
        stay = np.insert(stay,0,0)
        lastreward = reward[0:-1]
        lastreward = np.insert(lastreward,0,0)


        rl = np.zeros(len(stay))
        rl[0]=1
      
        rl_right = np.zeros(len(stay))
        rl_right[0]=choices[0]==state[0]
        choice_rr_start=-100
         
         
        rl_wrong=np.zeros(len(stay));
        rl_wrong[0]=choices[0]!=state[0];
        choice_rw_start=-100;
        
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
        
       
        for i in range(tr_runs):
           
            a_ind_right_1_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(rl_right) == i+1) & (choices == 1) & (task == 1))[0]],0)
            b_ind_right_1_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(rl_right) == i+1) & (choices == 0) & (task == 1))[0]],0)

            a_ind_wr_1_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl_wrong) == i+1) & (choices == 1) & (task == 1)))[0]],0)
            b_ind_wr_1_2[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl_wrong) == i+1) & (choices == 0) & (task == 1)))[0]],0)
        

            a_ind_right_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(rl_right) == i+1) & (choices == 1) & (task == 3))[0]],0)
            b_ind_right_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(rl_right) == i+1) & (choices == 0) & (task == 3))[0]],0)

            a_ind_wr_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(rl_wrong) == i+1) & (choices == 1) & (task == 3))[0]],0)
            b_ind_wr_3[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where((np.asarray(rl_wrong) == i+1) & (choices == 0) & (task == 3))[0]],0)
    
    return  a_ind_right_1_2,b_ind_right_1_2, a_ind_wr_1_2,b_ind_wr_1_2,a_ind_right_3,b_ind_right_3,a_ind_wr_3,b_ind_wr_3

def tim_extract_ones_skip(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    tr_runs = 6
    a_ind_rew = np.zeros((neurons,63,tr_runs));  b_ind_rew = np.zeros((neurons,63,tr_runs))
    a_ind_no_rew = np.zeros((neurons,63,tr_runs)); b_ind_no_rew = np.zeros((neurons,63,tr_runs))
    a_ind_right = np.zeros((neurons,63,tr_runs));  b_ind_right = np.zeros((neurons,63,tr_runs))
    a_ind_wr = np.zeros((neurons,63,tr_runs));  b_ind_wr = np.zeros((neurons,63,tr_runs))
    ind_right =  np.zeros((neurons,63,tr_runs)); ind_wrong = np.zeros((neurons,63,tr_runs)); 
    ind_right_std =  np.zeros((neurons,63,tr_runs)); ind_wrong_std = np.zeros((neurons,63,tr_runs)); 
    n_neurons_cum = 0
    
    for  s, sess in enumerate(dm):
        # runs_list = []
        # runs_list.append(0)
        # runs_list_corr = []
        # runs_list_incorr = []
        # DM = dm[s]
        # firing_rates = firing[s]
        # n_trials, n_neurons, n_timepoints = firing_rates.shape
        # n_neurons_cum += n_neurons

        # choices = DM[:,1]
        # state = DM[:,0]
        # reward = DM[:,2]    
        
        # correct = np.where(state == choices)[0]
        # incorrect = np.where(state != choices)[0]
      
       
        # run = 0
        # for c, ch in enumerate(choices):
        #     if c > 0:
        #         if choices[c] == choices[c-1]:
        #             run += 1
        #         elif choices[c] != choices[c-1]:
        #             run = 0
        #         runs_list.append(run)
        # corr_run = 0
        # run_ind_c =[]
        # for c, ch in enumerate(choices):
        #     if c > 0  and c in correct:
        #         if choices[c] == choices[c-1]:
        #             if corr_run == 0:
        #                 run_ind_c.append(c)
        #             corr_run +=1
        #         elif choices[c] != choices[c-1]:
        #             corr_run = 0
        #     else:
        #         corr_run = 0
        #     runs_list_corr.append(corr_run)
         
        # incorr_run = 0
        # run_ind_inc = []
        # for c, ch in enumerate(choices):
        #     if c > 0  and c in incorrect:
        #         if choices[c] == choices[c-1]:
        #             if incorr_run ==0:
        #                 run_ind_inc.append(c)
        #             incorr_run +=1
        #         elif choices[c] != choices[c-1]:
        #             incorr_run = 0
        #     else:
        #         incorr_run = 0
                
        #     runs_list_incorr.append(incorr_run)
        # runs_list_corr_arr = np.asarray(runs_list_corr)
        
        # ind_change = []
        # for r,rr in enumerate(runs_list_corr):
        #     if r > 1 and r  < len(runs_list_corr)-1:
        #         if rr == 0:
        #             if runs_list_corr[r-1] > 0 and runs_list_corr[r+1] > 0:
        #                 ind_change.append(r)
                        
        # if len(ind_change) > 0 :
        #     for i in ind_change:
        #         end = runs_list_corr.index(0,i+1)
        #         runs_list_corr_arr[i+1:end] += runs_list_corr_arr[i-1]
                        
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


        rl = np.zeros(len(stay))
        rl[0]=1
      
        rl_right = np.zeros(len(stay))
        rl_right[0]=choices[0]==state[0]
        choice_rr_start=-100
         
         
        rl_wrong=np.zeros(len(stay));
        rl_wrong[0]=choices[0]!=state[0];
        choice_rw_start=-100;
        
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
        
        rl_right_list = list(rl_right)
        ind_change = []
        for r,rr in enumerate(rl_right):
            if r > 1 and r  < len(rl_right)-1:
                if rr == 1:
                    if rl_right[r-1] > 1 and rl_right[r+1] > 1:
                        ind_change.append(r)
                        
        if len(ind_change) > 0 :
            for i in ind_change:
                end = rl_right_list.index(0,i+1)
                rl_right[i+1:end] += rl_right[i-1]
       
        trials = len(rl_right)
        
        # rl_right= np.roll(rl_right,np.random.randint(trials), axis = 0)
        # rl_wrong= np.roll(rl_wrong,np.random.randint(trials), axis = 0)
        # rl = np.roll(rl,np.random.randint(trials), axis = 0)

        for i in range(tr_runs):
            
            a_ind_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl) == i+1) & (choices == 1) & (reward == 1)))[0]],0)
            b_ind_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl) == i+1) & (choices == 0) & (reward == 1)))[0]],0)

            a_ind_no_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl) == i+1) & (choices == 1) & (reward == 0)))[0]],0)
            b_ind_no_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl) == i+1) & (choices == 0) & (reward == 0)))[0]],0)


            a_ind_right[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl_right) == i+1) & (choices == 1)))[0]],0)
            b_ind_right[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl_right) == i+1) & (choices == 0)))[0]],0)

            a_ind_wr[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl_wrong) == i+1) & (choices == 1)))[0]],0)
            b_ind_wr[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[(np.where((np.asarray(rl_wrong) == i+1) & (choices == 0)))[0]],0)
            
            
            ind_right[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.mean(firing_rates[np.where(np.asarray(rl_right) == i+1)[0]],0)
            ind_wrong[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] =np.mean(firing_rates[np.where(np.asarray(rl_wrong) == i+1)[0]],0)
             

            ind_right_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] = np.std(firing_rates[np.where(np.asarray(rl_right) == i+1)[0]],0)/(np.sqrt(firing_rates[np.where(np.asarray(rl_right) == i+1)[0]].shape[0]))
            ind_wrong_std[n_neurons_cum-n_neurons:n_neurons_cum ,:,i] =np.std(firing_rates[np.where(np.asarray(rl_wrong) == i+1)[0]],0)/(np.sqrt(firing_rates[np.where(np.asarray(rl_wrong) == i+1)[0]].shape[0]))
        
        
            
    return  a_ind_rew,  b_ind_rew,\
            a_ind_no_rew, b_ind_no_rew, a_ind_right, b_ind_right, a_ind_wr, b_ind_wr, ind_right,ind_wrong,ind_right_std,ind_wrong_std
        
      
def runs_extract_all(data):    
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    
    ind_1_rew = []; ind_2_rew = []; ind_3_rew = []; ind_4_rew = []; ind_5_rew = []

    ind_1 = []; ind_2 = []; ind_3 = []; ind_4 = []; ind_5= []

    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        reward = DM[:,2]    
        stay=choices[0:-1]==choices[1:]
        lastreward = reward[0:-1]

       
        run = 0
        for c, ch in enumerate(choices):
            if c > 0:
                if choices[c] == choices[c-1]:
                    run += 1
                elif choices[c] != choices[c-1]:
                    run = 0
                runs_list.append(run)
        
        ind_1_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 1) & (reward == 1)))[0]],0))
        ind_2_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 2) & (reward == 1)))[0]],0))  
        ind_3_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 3) & (reward == 1)))[0]],0))
        ind_4_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 4) & (reward == 1)))[0]],0))
        ind_5_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 5) & (reward == 1)))[0]],0))
             
        ind_1.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 1) & (reward == 0)))[0]],0))
        ind_2.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 2) & (reward == 0)))[0]],0))
        ind_3.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 3) & (reward == 0)))[0]],0))
        ind_4.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 4) & (reward == 0)))[0]],0))
        ind_5.append(np.mean(firing_rates[(np.where((np.asarray(runs_list) == 5) & (reward == 0)))[0]],0))

    all_reward = np.hstack([np.concatenate(ind_1_rew),np.concatenate(ind_2_rew),np.concatenate(ind_3_rew)\
                               ,np.concatenate(ind_4_rew),np.concatenate(ind_5_rew)])
    all_no_reward = np.hstack([np.concatenate(ind_1),np.concatenate(ind_2),np.concatenate(ind_3)\
                               ,np.concatenate(ind_4),np.concatenate(ind_5)])
      
    return all_reward, all_no_reward
     
    
def runs_extract(data):    
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    
    correct_ind_1_rew = []; correct_ind_2_rew = []; correct_ind_3_rew = []; correct_ind_4_rew = []; correct_ind_5_rew = []
    incorrect_ind_1_rew = []; incorrect_ind_2_rew = []; incorrect_ind_3_rew = []; incorrect_ind_4_rew = []; incorrect_ind_5_rew = []

    correct_ind_1 = []; correct_ind_2 = []; correct_ind_3 = []; correct_ind_4 = []; correct_ind_5= []
    incorrect_ind_1 = []; incorrect_ind_2= []; incorrect_ind_3 = []; incorrect_ind_4 = []; incorrect_ind_5 = []

    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        runs_list_corr = []
        runs_list_incorr = []
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]
        state = DM[:,0]
        reward = DM[:,2]    
        
        correct = np.where(state == choices)[0]
        incorrect = np.where(state != choices)[0]
      
       
        run = 0
        for c, ch in enumerate(choices):
            if c > 0:
                if choices[c] == choices[c-1]:
                    run += 1
                elif choices[c] != choices[c-1]:
                    run = 0
                runs_list.append(run)
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
        
        correct_ind_1_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 1) & (reward == 0)))[0]],0))
        correct_ind_2_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 2) & (reward == 0)))[0]],0))
        correct_ind_3_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 3) & (reward == 0)))[0]],0))
        correct_ind_4_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 4) & (reward == 0)))[0]],0))
        correct_ind_5_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 5) & (reward == 0)))[0]],0))
        
        incorrect_ind_1_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 1) & (reward == 0)))[0]],0))
        incorrect_ind_2_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 2) & (reward == 0)))[0]],0))
        incorrect_ind_3_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 3) & (reward == 0)))[0]],0))
        incorrect_ind_4_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 4) & (reward == 0)))[0]],0))
        incorrect_ind_5_rew.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 5) & (reward == 0)))[0]],0))


        correct_ind_1.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 1) & (reward == 1)))[0]],0))
        correct_ind_2.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 2) & (reward == 1)))[0]],0))
        correct_ind_3.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 3) & (reward == 1)))[0]],0))
        correct_ind_4.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 4) & (reward == 1)))[0]],0))
        correct_ind_5.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_corr) == 5) & (reward == 1)))[0]],0))
        
        incorrect_ind_1.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 1) & (reward == 1)))[0]],0))
        incorrect_ind_2.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 2) & (reward == 1)))[0]],0))
        incorrect_ind_3.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 3) & (reward == 1)))[0]],0))
        incorrect_ind_4.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 4) & (reward == 1)))[0]],0))
        incorrect_ind_5.append(np.mean(firing_rates[(np.where((np.asarray(runs_list_incorr) == 5) & (reward == 1)))[0]],0))

    correct_all = np.hstack([np.concatenate(correct_ind_1_rew),np.concatenate(correct_ind_2_rew),np.concatenate(correct_ind_3_rew)\
                               ,np.concatenate(correct_ind_4_rew),np.concatenate(correct_ind_5_rew)])
    incorrect_all = np.hstack([np.concatenate(incorrect_ind_1_rew),np.concatenate(incorrect_ind_2_rew),np.concatenate(incorrect_ind_3_rew)\
                               ,np.concatenate(incorrect_ind_4_rew),np.concatenate(incorrect_ind_5_rew)])
    
    correct_all_rew = np.hstack([np.concatenate(correct_ind_1),np.concatenate(correct_ind_2),np.concatenate(correct_ind_3)\
                               ,np.concatenate(correct_ind_4),np.concatenate(correct_ind_5)])
    incorrect_all_rew = np.hstack([np.concatenate(incorrect_ind_1),np.concatenate(incorrect_ind_2),np.concatenate(incorrect_ind_3)\
                               ,np.concatenate(incorrect_ind_4),np.concatenate(incorrect_ind_5)])
    
    return correct_all, incorrect_all, correct_all_rew,incorrect_all_rew
        
     
def svd_runs(data):
    
    all_reward, all_no_reward = runs_extract_all(data)
    u_rew,s_rew,v_rew = np.linalg.svd(all_reward)
    u,s,v = np.linalg.svd(all_no_reward)
    pal = sns.cubehelix_palette(8)
    plt.ion()
    plt.figure(figsize = (10,20))
    for i in range(20):
        
        plt.subplot(5,4,i+1)
        
        plt.plot(v[i,:int(v.shape[1]/5)], color = pal[0],linestyle = '--')
        plt.plot(v[i,int(v.shape[1]/5):int(v.shape[1]/5)*2], color = pal[1],linestyle = '--')
        plt.plot(v[i,int(v.shape[1]/5)*2: int(v.shape[1]/5)*3], color = pal[2],linestyle = '--')
        plt.plot(v[i,int(v.shape[1]/5)*3: int(v.shape[1]/5)*4], color = pal[3],linestyle = '--')
        plt.plot(v[i,int(v.shape[1]/5)*4: int(v.shape[1]/5)*5], color = pal[4],linestyle = '--', label = 'no reward 5')
        
        plt.plot(v_rew[i,:int(v.shape[1]/5)], color = pal[0])
        plt.plot(v_rew[i,int(v.shape[1]/5):int(v.shape[1]/5)*2], color = pal[1])
        plt.plot(v_rew[i,int(v.shape[1]/5)*2: int(v.shape[1]/5)*3], color = pal[2])
        plt.plot(v_rew[i,int(v.shape[1]/5)*3: int(v.shape[1]/5)*4], color = pal[3])
        plt.plot(v_rew[i,int(v.shape[1]/5)*4: int(v.shape[1]/5)*5], color = pal[4],label = 'reward 5')
    sns.despine()
    plt.legend()
   
   

    
    a_ind_rew, b_ind_rew,\
    a_ind_no_rew, b_ind_no_rew, a_ind_right, b_ind_right, a_ind_wr, b_ind_wr,ind_right,ind_wrong,ind_right_std,ind_wrong_std = tim_extract_ones_skip(data)
    
  
    #ind_right = ind_right[:,:20,:];     ind_wrong = ind_wrong[:,:20,:]
    #ind_right = ind_right.reshape(ind_right.shape[0], ind_right.shape[1]*ind_right.shape[2])
    #ind_wrong = ind_wrong.reshape(ind_wrong.shape[0], ind_wrong.shape[1]*ind_wrong.shape[2])
    ind_right = np.mean(ind_right[:,:25,:],1);     ind_wrong = np.mean(ind_wrong[:,:25,:],1)
    ind_right_std  = np.mean(ind_right_std[:,:25,:],1);     ind_wrong_std = np.mean(ind_wrong_std[:,:25,:],1)


    all_runs_pre_init = np.concatenate((ind_right,ind_wrong),1)
    all_runs_pre_init = all_runs_pre_init[~np.isnan(all_runs_pre_init).any(axis=1)]

    u,s,v = np.linalg.svd(all_runs_pre_init)
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(all_runs_pre_init)
    
    print(pca.explained_variance_ratio_)

    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    cmap = np.concatenate((pal_c[:6], pal[:6]))
    plt.figure(figsize = (4,8))
    #plt.annotate('HP')

    for i in range(10):
        
        plt.subplot(5,2,i+1)
        
        plt.scatter(np.arange(len(v[i]))[:6],v[i,:6], color =cmap[:6], s = 50,  label = 'right')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,6:], color =cmap[6:], s = 50, label = 'wrong')
        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,1,2,3,4,5,6,7])
        plt.xlabel('Run #')
        
    
    sns.despine()
    plt.legend()
    plt.tight_layout()
    
    neuron = all_runs_pre_init.shape[0]
    plt.figure()

    subplot = 0
    for i in range(neuron):
        subplot += 1 
        if subplot == 20:
            plt.figure()
            subplot -= 19
            
        plt.subplot(4,5,subplot)
        s = 10
        

        plt.plot(np.arange(len(all_runs_pre_init[i]))[:6],all_runs_pre_init[i,:6], color =cmap[1],  label = 'right')
        plt.fill_between(np.arange(len(all_runs_pre_init[i]))[:6],all_runs_pre_init[i,:6]+ind_wrong_std[i],all_runs_pre_init[i,:6]-ind_wrong_std[i], color =cmap[1], alpha = 0.7)
        plt.plot(np.arange(len(all_runs_pre_init[i]))[:6],all_runs_pre_init[i,6:], color =cmap[8],label = 'wrong')
        plt.fill_between(np.arange(len(all_runs_pre_init[i]))[:6],all_runs_pre_init[i,6:]+ind_wrong_std[i],all_runs_pre_init[i,6:]-ind_wrong_std[i], color =cmap[8], alpha = 0.7)

        
 

    sns.despine()
    plt.legend()
    
    
    
    # Split by A and B 
    a_ind_rew, b_ind_rew,\
    a_ind_no_rew, b_ind_no_rew, a_ind_right, b_ind_right, a_ind_wr, b_ind_wr,ind_right,ind_wrong,ind_right_std,ind_wrong_std = tim_extract_ones_skip(data)
    
  
   
    a_ind_right = np.mean(a_ind_right[:,:25,:],1);     b_ind_right = np.mean(b_ind_right[:,:25,:],1)
   
    a_ind_wr = np.mean(a_ind_wr[:,:25,:],1);     b_ind_wr = np.mean(b_ind_wr[:,:25,:],1)


    all_runs_pre_init = np.concatenate((a_ind_right,b_ind_right,a_ind_wr,b_ind_wr),1)
    all_runs_pre_init = all_runs_pre_init[~np.isnan(all_runs_pre_init).any(axis=1)]

    u,s,v = np.linalg.svd(all_runs_pre_init)
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(all_runs_pre_init)
    
    print(pca.explained_variance_ratio_)

    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    cmap = np.concatenate((pal_c[:6], pal[:6]))
    plt.figure(figsize = (4,8))
    #plt.annotate('HP')

    for i in range(10):
        
        plt.subplot(5,2,i+1)
        
        plt.scatter(np.arange(len(v[i]))[:6],v[i,:6], color = cmap[1], s = 50,  label = 'right A')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,6:12], color = cmap[6], s = 50, label = 'right B')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,12:18], color = cmap[4], s = 50, label = 'wrong A')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,18:], color = cmap[11], s = 50, label = 'wrong B')

        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,1,2,3,4,5,6,7])
        plt.xlabel('Run #')
        
    
    sns.despine()
    plt.legend()
    plt.tight_layout()

    
 
    # Split by Rew A and B
    a_ind_rew, b_ind_rew,\
    a_ind_no_rew, b_ind_no_rew, a_ind_right, b_ind_right, a_ind_wr, b_ind_wr,ind_right,ind_wrong,ind_right_std,ind_wrong_std = tim_extract_ones_skip(data)
    
  
   
    a_ind_rew = np.mean(a_ind_rew[:,:25,:],1);     b_ind_rew = np.mean(b_ind_rew[:,:25,:],1)
   
    a_ind_no_rew = np.mean(a_ind_no_rew[:,:25,:],1);     b_ind_no_rew = np.mean(b_ind_no_rew[:,:25,:],1)


    all_runs_pre_init = np.concatenate((a_ind_rew,b_ind_rew,a_ind_no_rew,b_ind_no_rew),1)
    all_runs_pre_init = all_runs_pre_init[~np.isnan(all_runs_pre_init).any(axis=1)]

    u,s,v = np.linalg.svd(all_runs_pre_init)
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(all_runs_pre_init)
    
    print(pca.explained_variance_ratio_)

    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    cmap = np.concatenate((pal_c[:6], pal[:6]))
    plt.figure(figsize = (4,8))
    #plt.annotate('HP')

    for i in range(10):
        
        plt.subplot(5,2,i+1)
        
        plt.scatter(np.arange(len(v[i]))[:6],v[i,:6], color = cmap[1], s = 50,  label = 'rew A')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,6:12], color = cmap[6], s = 50, label = 'rew B')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,12:18], color = cmap[4], s = 50, label = 'no-rew A')
        plt.scatter(np.arange(len(v[i]))[:6],v[i,18:], color = cmap[11], s = 50, label = 'no-rew B')

        #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,1,2,3,4,5,6,7])
        plt.xlabel('Run #')
        
    
    sns.despine()
    plt.legend()
    plt.tight_layout()
    
    
    a_ind_right_1_2,b_ind_right_1_2, a_ind_wr_1_2,b_ind_wr_1_2,a_ind_right_3,b_ind_right_3,a_ind_wr_3,b_ind_wr_3  = tim_extract(data)

    a_ind_right_1_2 = np.nanmean(a_ind_right_1_2[:,:25,:],1);     b_ind_right_1_2 = np.nanmean(b_ind_right_1_2[:,:25,:],1)
   
    a_ind_wr_1_2 = np.nanmean(a_ind_wr_1_2[:,:25,:],1);     b_ind_wr_1_2 = np.nanmean(b_ind_wr_1_2[:,:25,:],1)

    a_ind_right_3 = np.nanmean(a_ind_right_3[:,:25,:],1);     b_ind_right_3 = np.nanmean(b_ind_right_3[:,:25,:],1)
   
    a_ind_wr_3 = np.nanmean(a_ind_wr_3[:,:25,:],1);     b_ind_wr_3 = np.nanmean(b_ind_wr_3[:,:25,:],1)

    all_runs_pre_init = np.concatenate((a_ind_right_1_2,b_ind_right_1_2,a_ind_wr_1_2,b_ind_wr_1_2,\
                                        a_ind_right_3,b_ind_right_3,a_ind_wr_3,b_ind_wr_3),1)
        
    all_runs_pre_init = all_runs_pre_init[~np.isnan(all_runs_pre_init).any(axis=1)]
    
    u,s,v = np.linalg.svd(all_runs_pre_init)
    
    
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)
    cmap = np.concatenate((pal_c[:6], pal[:6]))
    plt.figure(figsize = (4,8))

    ind = 4
    for i in range(10):
        
        plt.subplot(5,2,i+1)
        
        plt.scatter(np.arange(len(v[i]))[:ind],v[i,:ind], color = cmap[1], s = 50,  label = 'right A T 1 & 2')
        plt.scatter(np.arange(len(v[i]))[:ind],v[i,ind:ind*2], color = cmap[6], s = 50, label = 'right B T 1 & 2')
        plt.scatter(np.arange(len(v[i]))[:ind],v[i,ind*2:ind*3], color = cmap[4], s = 50, label = 'wrong A T 1 & 2')
        plt.scatter(np.arange(len(v[i]))[:ind],v[i,ind*3:ind*4], color = cmap[9], s = 50, label = 'wrong B T  1& 2')
        # plt.scatter(np.arange(len(v[i]))[:ind],v_c[i,:ind], color = cmap[3], s = 50, label = 'right B T 3')
        # plt.scatter(np.arange(len(v[i]))[:ind],v_c[i,ind:ind*2], color = cmap[8], s = 50, label = 'right A T 3')
        # plt.scatter(np.arange(len(v[i]))[:ind],v_c[i,ind*2:ind*3], color = cmap[5], s = 50, label = 'wrong A T 3')
        # plt.scatter(np.arange(len(v[i]))[:ind],v_c[i,ind*3:ind*4], color = cmap[11], s = 50, label = 'wrong B T 3')

        # #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,1,2,3,4,5,6,7])
        plt.xlabel('Run #')
        
    
    sns.despine()
    plt.legend()
    plt.tight_layout()
    
    
    
    a_ind_error,b_ind_error = extract_error_count(data)
    a_ind_error_res = np.transpose(a_ind_error,[0,2,1])
    a_ind_error_res = a_ind_error_res.reshape(a_ind_error_res.shape[0],a_ind_error_res.shape[2]*a_ind_error_res.shape[1])
    b_ind_error_res = np.transpose(b_ind_error,[0,2,1])
    b_ind_error_res = b_ind_error_res.reshape(b_ind_error_res.shape[0],b_ind_error_res.shape[2]*b_ind_error_res.shape[1])

    all_runs_pre_init = np.concatenate((a_ind_error_res,b_ind_error_res),1)
        
    all_runs_pre_init = all_runs_pre_init[~np.isnan(all_runs_pre_init).any(axis=1)]
  
    
    u,s,v = np.linalg.svd(all_runs_pre_init)
    #v = v.reshape((v.shape[0],a_ind_error.shape[1], a_ind_error.shape[2]*2))
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)

    plt.ion()
    plt.figure(figsize = (10,20))
    ind = 8
    for i in range(5):
        
        plt.subplot(5,1,i+1)
        
        plt.plot(v[i,:int(v.shape[1]/ind)], color = pal[0],label = 'error 1')
        plt.plot(v[i,int(v.shape[1]/ind):int(v.shape[1]/ind)*2], color = pal[1],label = ' error 2')
        plt.plot(v[i,int(v.shape[1]/ind)*2: int(v.shape[1]/ind)*3], color = pal[2],label = ' error 3')
        plt.plot(v[i,int(v.shape[1]/ind)*3: int(v.shape[1]/ind)*4], color = pal[3], label = ' error 4')
        
        plt.plot(v[i,int(v.shape[1]/ind)*4: int(v.shape[1]/ind)*5], color = pal_c[0],label = ' reward 1')
        plt.plot(v[i,int(v.shape[1]/ind)*5: int(v.shape[1]/ind)*6], color = pal_c[1],label = 'reward 2')
        plt.plot(v[i,int(v.shape[1]/ind)*6: int(v.shape[1]/ind)*7], color = pal_c[2],label = 'reward 3')
        plt.plot(v[i,int(v.shape[1]/ind)*7: int(v.shape[1]/ind)*8], color = pal_c[3], label = 'reward 4')
        
    sns.despine()
    plt.legend()
   

