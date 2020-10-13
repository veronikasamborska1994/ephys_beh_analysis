#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:13:57 2020

@author: veronikasamborska
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:39:22 2020

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
import scipy
from scipy import io
from sklearn.decomposition import PCA
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from scipy import stats

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)


def load():
    
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    ## Longer trial
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP_RPE.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC_RPE.mat')
  

def task_ind(task, a_pokes, b_pokes):
    
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task));
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
  
    return taskid



def regression_code_session(data, design_matrix): 
    
    tc = np.identity(design_matrix.shape[1])
    
    pdes = np.linalg.pinv(design_matrix)
    tc_pdes = np.matmul(tc,pdes)
    pdes_tc = np.matmul(np.transpose(pdes),np.transpose(tc))
    
    prevar = np.diag(np.matmul(tc_pdes, pdes_tc))
    
    R = np.identity(design_matrix.shape[0]) - np.matmul(design_matrix, pdes)
    tR = np.trace(R)
    
    pe = np.matmul(pdes,data)
    cope = np.matmul(tc,pe)
    
    res = data - np.matmul(design_matrix,pe)
    sigsq = np.sum((res*res)/tR, axis = 0)
    sigsq = np.reshape(sigsq,(1,res.shape[1]))
    prevar = np.reshape(prevar,(tc.shape[0],1))
    varcope = prevar*sigsq
    
    #tstats = cope/np.sqrt(varcope)
    
    return cope,varcope
    


def rew_prev_behaviour(data):
    
    dm = data['DM'][0]
    results_array = []
    std_err = []
   
    for  s, sess in enumerate(dm):
            
            DM = dm[s] 
              
            choices = DM[:,1]
            
            reward = DM[:,2]
        
          
            reward_2_ago = reward[1:-2]
            reward_3_ago =reward[:-3]
            reward_prev = reward[2:-1]
            reward_current = reward[3:]
            
            choices_2_ago = 0.5-choices[1:-2]
            choices_3_ago = 0.5-choices[:-3]
            choices_prev = 0.5-choices[2:-1]
            choices_current = 0.5-choices[3:]
            
           
            choices_2_ago_rew = ((choices_2_ago)*(reward_2_ago-0.5))*2
            choices_3_ago_rew = ((choices_3_ago)*(reward_3_ago-0.5))*2
            choices_prev_rew = ((choices_prev)*(reward_prev-0.5))*2
                    
            
            ones = np.ones(len(choices_current))
            trials = len(choices_current)
            predictors_all = OrderedDict([
                                            ('1 ago Outcome', reward_prev),
                                            ('2 ago Outcome', reward_2_ago),
                                            ('3 ago Outcome', reward_3_ago),
                                      #      ('4 ago Outcome', reward_4_ago),

                                            ('1 ago Choice', choices_prev),
                                            ('2 ago Choice', choices_2_ago),
                                            ('3 ago Choice', choices_3_ago), 
                                       #     ('4 ago Choice', choices_4_ago), 

                                            ('1 ago Choice Rew', choices_prev_rew),
                                            ('2 ago Choice Rew', choices_2_ago_rew),
                                            ('3 ago Choice Rew', choices_3_ago_rew),
                                       #     ('4 ago Choice Rew', choices_4_ago_rew),

                                          ('ones', ones)])
            
            X = np.vstack(predictors_all.values()).T[:trials,:].astype(float)
            #choices_current = choices_current.reshape(trials,1)        
            rank = np.linalg.matrix_rank(X)
            n_predictors = X.shape[1]
            
           #model = sm.Logit(choices_current,X)
            model = OLS(choices_current,X)
            results = model.fit()
            results_array.append(results.params)
            cov = results.cov_params()
            std_err.append(np.sqrt(np.diag(cov)))
    
    average = np.sum((results_array),0)/np.sqrt(np.sum(std_err,0))

      

def within_taks_codes(data, area = 'HP', perm = 5):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    C = []
    cpd = []
    cpd_perm_p = []

    for  s, sess in enumerate(dm):
        
        cpd_perm = [[] for i in range(perm)] # To store permuted predictor loadings for each session.

        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        state = DM[:,0]
        choices = DM[:,1]
        
        reward = DM[:,2]    
        
        reward_2_ago = 0.5-reward[1:-2]
        reward_3_ago = 0.5-reward[:-3]
        reward_prev = 0.5-reward[2:-1]
        reward_current = reward[3:]
       
        # reward_o_1_ago = np.asarray(reward_prev)
        # reward_o_2_ago = np.asarray(reward_2_ago)
        # reward_o_3_ago = np.asarray(reward_3_ago)
       
        firing_rates = firing_rates[3:]
        
        choices_2_ago = 0.5-choices[1:-2]
        choices_3_ago = 0.5-choices[:-3]
        choices_prev = 0.5-choices[2:-1]
        choices_current = choices[3:]
        state = state[3:]
        
     
        cum_reward_orth = np.vstack([reward_current, np.ones(len(reward_current))]).T
        xt = np.linalg.pinv(cum_reward_orth)
        identity = np.identity(len(reward_current))
        id_x = (identity- np.matmul(cum_reward_orth, xt))
             
        # choice_o_1_ago = np.matmul(id_x, np.asarray(choices_prev))
        # choice_o_2_ago = np.matmul(id_x, np.asarray(choices_2_ago))
        # choice_o_3_ago = np.matmul(id_x, np.asarray(choices_3_ago))
        
        ch_rew_int_1 = choices_prev*reward_prev
        ch_rew_int_2 = choices_2_ago*reward_2_ago
        ch_rew_int_3 = choices_3_ago*reward_3_ago

        ones = np.ones(len(choices_3_ago))
        
        predictors_all = OrderedDict([('Reward', reward_current),
                                       ('Choice', choices_current),
                                      
                                       
                                       ('1 ago Outcome', reward_prev),
                                       ('2 ago Outcome', reward_2_ago),
                                       ('3 ago Outcome', reward_3_ago),
                                 
                                        ('1 ago Choice', choices_prev),
                                        ('2 ago Choice', choices_2_ago),
                                        ('3 ago Choice', choices_3_ago),
                                        ('1 Rew x Choice', ch_rew_int_1),
                                        ('2 Rew x Choice', ch_rew_int_2),
                                        ('3 Rew x Choice', ch_rew_int_3),

                                      ('ones', ones)])   
            
        X = np.vstack(predictors_all.values()).T[:len(choices_current),:].astype(float)
        rank = np.linalg.matrix_rank(X)
        print(rank)
        print(X.shape[1])
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd.append(re._CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
        if perm:
           for i in range(perm):
               y_perm = np.roll(y,np.random.randint(len(y)), axis = 0)
               cpd_temp = re._CPD(X,y_perm).reshape(n_neurons, n_timepoints, n_predictors)
               cpd_perm[i].append(np.nanmean(cpd_temp, axis = 0))
 
        cpd_perm_p.append(np.percentile(cpd_perm,95, axis = 0))
    if perm: # Evaluate P values.
        cpd_perm_pval = np.mean(cpd_perm_p,0)[0]
        #cpd_perm_p = np.percentile(cpd_perm,95, axis = 0)
    
    C = np.concatenate(C,1)
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
  
    plt.figure()
    pred = list(predictors_all.keys())
         
    
    array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
   
    for i in range(cpd.shape[1]):
        array_pvals[(np.where(cpd[:,i] > cpd_perm_pval[:,i])[0]),i] = 0.05
 
    ymax = np.max(cpd[:,2:-1].T)
    t = np.arange(0,121)
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors

    for i in np.arange(cpd.shape[1]):
        if i >1 and i < cpd.shape[1]-1:
            plt.plot(cpd[:,i], label =pred[i],color = c[i])
            y = ymax*(1+0.04*i)
            p_vals = array_pvals[:,i]
            t05 = t[p_vals == 0.05]
            plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=5, color=c[i])
            
       

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([24, 35, 42], ['I', 'C', 'R'])
    plt.legend()
    sns.despine()
    plt.title(area)
        

  
def sequence_rewards_errors_regression_generalisation_rew(data, area = 'HP_',c_to_plot = 1,c_to_proj = 2, c_to_proj_3 = 3,ind_rew = 1, plot_a = False,plot_b = False):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    C_1 = []; C_2 = []; C_3 = []
    cpd_1 = []; cpd_2 = []; cpd_3 = []
    c_to_plot = c_to_plot
    c_to_proj = c_to_proj
    c_to_proj_3 = c_to_proj_3
    ind_rew = ind_rew
  
   

    for  s, sess in enumerate(dm):
        
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
       
        # firing_rates = firing_rates[:,:,:63]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        
        state = DM[:,0]
        choices = DM[:,1]-0.5
        reward = DM[:,2] -0.5   

        task =  DM[:,5]
        a_pokes = DM[:,6]
        b_pokes = DM[:,7]
   
        taskid = task_ind(task, a_pokes, b_pokes)
        
        trial_to_start = 7
        
        taskid = taskid[trial_to_start:]

        
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]
        
        # reward_2_ago = reward[1:-2]
        # reward_3_ago = reward[:-3]

        reward_2_ago = np.mean([reward[trial_to_start-2:-2],reward[trial_to_start-3:-3]],0)
        reward_3_ago = np.mean([reward[trial_to_start-4:-4],reward[trial_to_start-5:-5]],0)
        reward_4_ago = np.mean([reward[trial_to_start-6:-6],reward[:-7]],0)
       
        reward_prev = reward[trial_to_start-1:-1]
        reward_current = reward[trial_to_start:]
       
        firing_rates = firing_rates[trial_to_start:]
        
        choices_2_ago = np.mean([choices[trial_to_start-2:-2],choices[trial_to_start-3:-3]],0)
        choices_3_ago = np.mean([choices[trial_to_start-4:-4],choices[trial_to_start-5:-5]],0)
        choices_4_ago = np.mean([choices[trial_to_start-6:-6],choices[:-7]],0)
        
        choices_prev = choices[trial_to_start-1:-1]
        choices_current = choices[trial_to_start:]
        state = state[trial_to_start:]
        
     
        cum_reward_orth = np.vstack([reward_current, np.ones(len(reward_current))]).T
        xt = np.linalg.pinv(cum_reward_orth)
        identity = np.identity(len(reward_current))
        id_x = (identity- np.matmul(cum_reward_orth, xt))
             
        reward_o_1_ago = np.matmul(id_x, np.asarray(reward_prev))
        reward_o_2_ago = np.matmul(id_x, np.asarray(reward_2_ago))
        reward_o_3_ago = np.matmul(id_x, np.asarray(reward_3_ago))
        reward_o_4_ago = np.matmul(id_x, np.asarray(reward_4_ago))

        cum_ch_orth = np.vstack([choices_current, np.ones(len(choices_current))]).T
        xt = np.linalg.pinv(cum_ch_orth)
        identity = np.identity(len(choices_current))
        id_x = (identity- np.matmul(cum_ch_orth, xt))
       
        choice_o_1_ago = np.matmul(id_x, np.asarray(choices_prev))
        choice_o_2_ago = np.matmul(id_x, np.asarray(choices_2_ago))
        choice_o_3_ago = np.matmul(id_x, np.asarray(choices_3_ago))
        choice_o_4_ago = np.matmul(id_x, np.asarray(choices_4_ago))
        
        ones = np.ones(len(reward_current))
        reward_1 = reward_current[task_1]
        choices_1 = choices_current[task_1]
        
        _1_reward_1 = reward_o_1_ago[task_1]
        _2_reward_1 = reward_o_2_ago[task_1]
        _3_reward_1 = reward_o_3_ago[task_1]
        _4_reward_1 = reward_o_4_ago[task_1]

        
        _1_choices_1 = choice_o_1_ago[task_1]
        _2_choices_1 = choice_o_2_ago[task_1]
        _3_choices_1 = choice_o_3_ago[task_1]
        _4_choices_1 = choice_o_4_ago[task_1]
 
        _1_choices_1_for_int = choices_prev[task_1]
        _2_choices_1_for_int = choices_2_ago[task_1]
        _3_choices_1_for_int = choices_3_ago[task_1]
        _4_choices_1_for_int = choices_4_ago[task_1]

            
        _1_reward_1_ch = (_1_reward_1*_1_choices_1_for_int)
        _2_reward_1_ch = (_2_reward_1*_2_choices_1_for_int)
        _3_reward_1_ch = (_3_reward_1*_3_choices_1_for_int)
        _4_reward_1_ch = (_4_reward_1*_4_choices_1_for_int)


        ones_1 = ones[task_1]
        firing_rates_1 = firing_rates[task_1]
        a_1 = np.where(choices_1 == 0.5)[0]
        b_1 = np.where(choices_1 == -0.5)[0]
        
        if plot_a == True:
            reward_1 = reward_1[a_1]

            firing_rates_1 = firing_rates_1[a_1]
            _1_reward_1 = _1_reward_1[a_1]
            _2_reward_1 = _2_reward_1[a_1]
            _3_reward_1 = _3_reward_1[a_1]
            _4_reward_1 = _4_reward_1[a_1]

            _1_choices_1 = _1_choices_1[a_1]
            _2_choices_1 = _2_choices_1[a_1]
            _3_choices_1 = _3_choices_1[a_1]
            _4_choices_1 = _4_choices_1[a_1]
     
            _1_reward_1_ch = _1_reward_1_ch[a_1]
            _2_reward_1_ch = _2_reward_1_ch[a_1]
            _3_reward_1_ch = _3_reward_1_ch[a_1]
            _4_reward_1_ch = _4_reward_1_ch[a_1]

            ones_1  = ones_1[a_1]
           
        elif plot_b == True:
            
            reward_1 = reward_1[b_1]
            
            firing_rates_1 = firing_rates_1[b_1]
            _1_reward_1 = _1_reward_1[b_1]
            _2_reward_1 = _2_reward_1[b_1]
            _3_reward_1 = _3_reward_1[b_1]
            _4_reward_1 = _4_reward_1[b_1]
 
            _1_choices_1 = _1_choices_1[b_1]
            _2_choices_1 = _2_choices_1[b_1]
            _3_choices_1 = _3_choices_1[b_1]
            _4_choices_1 = _4_choices_1[b_1]
             
            _1_reward_1_ch = _1_reward_1_ch[b_1]
            _2_reward_1_ch = _2_reward_1_ch[b_1]
            _3_reward_1_ch = _3_reward_1_ch[b_1]
            _4_reward_1_ch = _4_reward_1_ch[b_1]

            ones_1  = ones_1[b_1]
       
        predictors_all = OrderedDict([('Reward', reward_1),
                                       ('Choice', choices_1),

                                        ('1 ago Outcome', _1_reward_1),
                                        ('2 ago Outcome', _2_reward_1),
                                        ('3 ago Outcome', _3_reward_1),
                                        ('4 ago Outcome', _4_reward_1),

                                          ('1 ago Choice', _1_choices_1),
                                          ('2 ago Choice', _2_choices_1),
                                          ('3 ago Choice', _3_choices_1), 
                                          ('4 ago Choice', _4_choices_1), 
 
                                           ('Prev Rew by Ch ',_1_reward_1_ch),
                                           ('2 Rew ago by Ch ',_2_reward_1_ch),
                                            ('3 Rew ago by Ch ',_3_reward_1_ch),
                                            ('4 Rew ago by Ch ',_4_reward_1_ch),

                                      ('ones', ones_1)])   
            
        X_1 = np.vstack(predictors_all.values()).T[:len(choices_1),:].astype(float)
        rank = np.linalg.matrix_rank(X_1)
       
        n_predictors = X_1.shape[1]
        y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        #tstats = reg_f.regression_code(y_1, X_1)
        tstats,cope = regression_code_session(y_1, X_1)
        C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_1.append(re._CPD(X_1,y_1).reshape(n_neurons, n_timepoints, n_predictors))

        
        
        reward_2 = reward_current[task_2]
        choices_2 = choices_current[task_2]
        _1_reward_2 = reward_o_1_ago[task_2]
        _2_reward_2 = reward_o_2_ago[task_2]
        _3_reward_2 = reward_o_3_ago[task_2]
        _4_reward_2 = reward_o_4_ago[task_2]
 
        _1_choices_2 = choice_o_1_ago[task_2]
        _2_choices_2 = choice_o_2_ago[task_2]
        _3_choices_2 = choice_o_3_ago[task_2]
        _4_choices_2 = choice_o_4_ago[task_2]

        _1_choices_2_for_int = choices_prev[task_2]
        _2_choices_2_for_int = choices_2_ago[task_2]
        _3_choices_2_for_int = choices_3_ago[task_2]
        _4_choices_2_for_int = choices_4_ago[task_2]
  
        _1_reward_2_ch = (_1_reward_2*_1_choices_2_for_int)
        _2_reward_2_ch = (_2_reward_2*_2_choices_2_for_int)
        _3_reward_2_ch = (_3_reward_2*_3_choices_2_for_int)
        _4_reward_2_ch = (_4_reward_2*_4_choices_2_for_int)

        ones_2 = ones[task_2]
        firing_rates_2 = firing_rates[task_2]
        a_2 = np.where(choices_2 == 0.5)[0]
        b_2 = np.where(choices_2 == -0.5)[0]
        
        if plot_a == True:
            reward_2 = reward_2[a_2]

            firing_rates_2 = firing_rates_2[a_2]
            _1_reward_2 = _1_reward_2[a_2]
            _2_reward_2 = _2_reward_2[a_2]
            _3_reward_2 = _3_reward_2[a_2]
            _4_reward_2 = _4_reward_2[a_2]

            _1_choices_2 = _1_choices_2[a_2]
            _2_choices_2 = _2_choices_2[a_2]
            _3_choices_2 = _3_choices_2[a_2]
            _4_choices_2 = _4_choices_2[a_2]

               
            _1_reward_2_ch = _1_reward_2_ch[a_2]
            _2_reward_2_ch = _2_reward_2_ch[a_2]
            _3_reward_2_ch = _3_reward_2_ch[a_2]
            _4_reward_2_ch = _4_reward_2_ch[a_2]

            ones_2  = ones_2[a_2]
           
        elif plot_b == True:
            reward_2 = reward_2[b_2]
            firing_rates_2 = firing_rates_2[b_2]
            _1_reward_2 = _1_reward_2[b_2]
            _2_reward_2 = _2_reward_2[b_2]
            _3_reward_2 = _3_reward_2[b_2]
            _4_reward_2 = _4_reward_2[b_2]
   
            _1_choices_2 = _1_choices_2[b_2]
            _2_choices_2 = _2_choices_2[b_2]
            _3_choices_2 = _3_choices_2[b_2]
            _4_choices_2 = _4_choices_2[b_2]

               
            _1_reward_2_ch = _1_reward_2_ch[b_2]
            _2_reward_2_ch = _2_reward_2_ch[b_2]
            _3_reward_2_ch = _3_reward_2_ch[b_2]
            _4_reward_2_ch = _4_reward_2_ch[b_2]

            ones_2  = ones_2[b_2]
       

    

        
        predictors_all = OrderedDict([('Reward', reward_2),
                                       ('Choice', choices_2),
                                       
                                        ('1 ago Outcome', _1_reward_2),
                                        ('2 ago Outcome', _2_reward_2),
                                        ('3 ago Outcome', _3_reward_2),
                                        ('4 ago Outcome', _4_reward_2),

                                          ('1 ago Choice', _1_choices_2),
                                          ('2 ago Choice', _2_choices_2),
                                          ('3 ago Choice', _3_choices_2),
                                          ('4 ago Choice', _4_choices_2),
 
                                            (' Prev Rew by Ch ',_1_reward_2_ch),
                                            (' 2 Rew ago by Ch ',_2_reward_2_ch),
                                            (' 3 Rew ago by Ch ',_3_reward_2_ch),
                                            (' 4 Rew ago by Ch ',_4_reward_2_ch),

                                       ('ones', ones_2)])
         
               
        X_2 = np.vstack(predictors_all.values()).T[:len(choices_2),:].astype(float)
        y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        #tstats = reg_f.regression_code(y_2, X_2)
        tstats,cope = regression_code_session(y_2, X_2)
        C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        cpd_2.append(re._CPD(X_2,y_2).reshape(n_neurons, n_timepoints, n_predictors))
               
        
        reward_3 = reward_current[task_3]
        choices_3 = choices_current[task_3]
        
        _1_reward_3 = reward_o_1_ago[task_3]
        _2_reward_3 = reward_o_2_ago[task_3]
        _3_reward_3 = reward_o_3_ago[task_3]
        _4_reward_3 = reward_o_4_ago[task_3]
 
        _1_choices_3 = choice_o_1_ago[task_3]
        _2_choices_3 = choice_o_2_ago[task_3]
        _3_choices_3 = choice_o_3_ago[task_3]
        _4_choices_3 = choice_o_4_ago[task_3]
 

        _1_choices_3_for_int = choices_prev[task_3]
        _2_choices_3_for_int = choices_2_ago[task_3]
        _3_choices_3_for_int = choices_3_ago[task_3]
        _4_choices_3_for_int = choices_4_ago[task_3]


        _1_reward_3_ch = (_1_reward_3*_1_choices_3_for_int)
        _2_reward_3_ch = (_2_reward_3*_2_choices_3_for_int)
        _3_reward_3_ch = (_3_reward_3*_3_choices_3_for_int)
        _4_reward_3_ch = (_4_reward_3*_4_choices_3_for_int)

        ones_3 = ones[task_3]

        firing_rates_3 = firing_rates[task_3]
        a_3 = np.where(choices_3 == 0.5)[0]
        b_3 = np.where(choices_3 == -0.5)[0]
        
        if plot_a == True:
            reward_3 = reward_3[a_3]
            firing_rates_3 = firing_rates_3[a_3]
            _1_reward_3 = _1_reward_3[a_3]
            _2_reward_3 = _2_reward_3[a_3]
            _3_reward_3 = _3_reward_3[a_3]
            _4_reward_3 = _4_reward_3[a_3]
  
            _1_choices_3 = _1_choices_3[a_3]
            _2_choices_3 = _2_choices_3[a_3]
            _3_choices_3 = _3_choices_3[a_3]
            _4_choices_3 = _4_choices_3[a_3]
               
            _1_reward_3_ch = _1_reward_3_ch[a_3]
            _2_reward_3_ch = _2_reward_3_ch[a_3]
            _3_reward_3_ch = _3_reward_3_ch[a_3]
            _4_reward_3_ch = _4_reward_3_ch[a_3]

            ones_3  = ones_3[a_3]
           
        elif plot_b == True:
            
            reward_3 = reward_3[b_3]
            firing_rates_3 = firing_rates_3[b_3]
            _1_reward_3 = _1_reward_3[b_3]
            _2_reward_3 = _2_reward_3[b_3]
            _3_reward_3 = _3_reward_3[b_3]
            _4_reward_3 = _4_reward_3[b_3]

            _1_choices_3 = _1_choices_3[b_3]
            _2_choices_3 = _2_choices_3[b_3]
            _3_choices_3 = _3_choices_3[b_3]
            _4_choices_3 = _4_choices_3[b_3]

               
            _1_reward_3_ch = _1_reward_3_ch[b_3]
            _2_reward_3_ch = _2_reward_3_ch[b_3]
            _3_reward_3_ch = _3_reward_3_ch[b_3]
            _4_reward_3_ch = _4_reward_3_ch[b_3]

            ones_3  = ones_3[b_3]
       


        predictors_all = OrderedDict([('Reward', reward_3),
                                        ('Choice', choices_3),
                                
                                        ('1 ago Outcome', _1_reward_3),
                                        ('2 ago Outcome', _2_reward_3),
                                        ('3 ago Outcome', _3_reward_3),
                                        ('4 ago Outcome', _4_reward_3),

                                          ('1 ago Choice', _1_choices_3),
                                        ('2 ago Choice', _2_choices_3),
                                        ('3 ago Choice', _3_choices_3),
                                        ('4 ago Choice', _4_choices_3),
 
                                        (' Prev Rew by Ch',_1_reward_3_ch),
                                        (' 2 Rew ago by Ch',_2_reward_3_ch),
                                        (' 3 Rew ago by Ch',_3_reward_3_ch),
                                        (' 4 Rew ago by Ch',_4_reward_3_ch),

                                       ('ones', ones_3)])
            
        X_3 = np.vstack(predictors_all.values()).T[:len(choices_3),:].astype(float)
        y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        #tstats = reg_f.regression_code(y_3, X_3)
        tstats,cope = regression_code_session(y_3, X_3)

        C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints))# Predictor loadings
        cpd_3.append(re._CPD(X_3,y_3).reshape(n_neurons, n_timepoints, n_predictors))


    C_1 = np.concatenate(C_1,1)
    
    C_2 = np.concatenate(C_2,1)
    
    C_3 = np.concatenate(C_3,1)
    
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    cpd = np.mean([cpd_1,cpd_2,cpd_3],0)

    C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    
    nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
    C_1 = C_1[:,nans[0],:]; C_2 = C_2[:,nans[0],:];  C_3 = C_3[:,nans[0],:]
   
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors
    
    j = 0
    plt.figure()
    pred = list(predictors_all.keys())
    pred = pred[:-1]
    for ii,i in enumerate(cpd.T[:-1]):
        plt.plot(i, color = c[j],label = pred[j])
      
        j+=1
    plt.legend()
    sns.despine()
        
    plt.title( area +'t-values')
   
    cell_id_cum_reward = np.where(np.mean(abs(C_1[1,:,:20]),1) > 1.5)[0]
    cell_id_cum_error = np.where(np.mean(abs(C_1[2,:,:20]),1) > 1.5)[0]
    cell_id_prev_ch = np.where(np.mean(abs(C_1[3,:,:20]),1) > 1.5)[0]
 

    C_1_rew = C_1[ind_rew]; C_2_rew = C_2[ind_rew]; C_3_rew = C_3[ind_rew]
    C_1_rew_count = C_1[c_to_plot]; C_2_rew_count = C_2[c_to_plot]; C_3_rew_count = C_3[c_to_plot]
   
    reward_times_to_choose = np.asarray([20,24,35,41])
    #reward_times_to_choose = np.arange(0,63,10)
    # reward_times_to_choose = np.arange(0,80,10)

    ones = np.ones(len(C_1_rew))
    C_1_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_2_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_3_rew_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
   
    j = 0
    for i in reward_times_to_choose:
        if i ==reward_times_to_choose[0]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-20:i],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-20:i],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-20:i],1)
        elif i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i-5:i+5],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i-5:i+5],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i-5:i+5],1)
        elif i == reward_times_to_choose[3]:
            C_1_rew_proj[:,j] = np.mean(C_1_rew[:,i:i+5],1)
            C_2_rew_proj[:,j] = np.mean(C_2_rew[:,i:i+5],1)
            C_3_rew_proj[:,j] = np.mean(C_3_rew[:,i:i+5],1)
         
        j +=1
    
   
    C_1_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_2_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    C_3_rew_count_proj  = np.ones((C_1_rew.shape[0],reward_times_to_choose.shape[0]+1))
    j = 0
    for i in reward_times_to_choose:
        if i ==reward_times_to_choose[0]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-20:i],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-20:i],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-20:i],1)
        elif i ==reward_times_to_choose[1] or i == reward_times_to_choose[2]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i-5:i+5],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i-5:i+5],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i-5:i+5],1)
        elif i == reward_times_to_choose[3]:
            C_1_rew_count_proj[:,j] = np.mean(C_1_rew_count[:,i:i+5],1)
            C_2_rew_count_proj[:,j] = np.mean(C_2_rew_count[:,i:i+5],1)
            C_3_rew_count_proj[:,j] = np.mean(C_3_rew_count[:,i:i+5],1)
      
        j +=1
        
   
    cpd_1_2_rew, cpd_1_2_rew_var = regression_code_session(C_2_rew_count, C_1_rew_proj);  
    cpd_1_3_rew, cpd_1_3_rew_var = regression_code_session(C_3_rew_count, C_1_rew_proj); 
    cpd_2_3_rew, cpd_2_3_rew_var = regression_code_session(C_3_rew_count,C_2_rew_proj)
    rew_to_count_cpd = (cpd_1_2_rew + cpd_1_3_rew +cpd_2_3_rew)/np.sqrt((cpd_1_2_rew_var+cpd_1_3_rew_var+cpd_2_3_rew_var))

    # cpd_1_2_rew = re._CPD(C_1_rew_proj,C_2_rew_count)
    # cpd_1_3_rew = re._CPD(C_1_rew_proj,C_3_rew_count)
    # cpd_2_3_rew = re._CPD(C_2_rew_proj,C_3_rew_count)
    
    # rew_to_count_cpd = np.mean([cpd_1_2_rew, cpd_1_3_rew, cpd_2_3_rew],0)
    
    cpd_1_rew, cpd_1_rew_var = regression_code_session(C_1_rew_count,C_1_rew_proj)
    cpd_2_rew, cpd_2_rew_var = regression_code_session(C_2_rew_count,C_2_rew_proj)
    cpd_3_rew, cpd_3_rew_var= regression_code_session(C_3_rew_count,C_3_rew_proj)
     
    cpd_1_2_rew_count, cpd_1_2_rew_count_var = regression_code_session(C_2_rew_count, C_1_rew_count_proj); 
    cpd_1_3_rew_count, cpd_1_3_rew_count_var = regression_code_session(C_3_rew_count, C_1_rew_count_proj); 
    cpd_2_3_rew_count, cpd_2_3_rew_count_var = regression_code_session(C_3_rew_count, C_2_rew_count_proj)
    
    within_cpd = (cpd_1_rew + cpd_2_rew + cpd_3_rew)/np.sqrt((cpd_1_rew_var + cpd_2_rew_var + cpd_3_rew_var))
    
    count_to_count_cpd =  (cpd_1_2_rew_count + cpd_1_3_rew_count + cpd_2_3_rew_count)/ np.sqrt((cpd_1_2_rew_count_var + cpd_1_3_rew_count_var + cpd_2_3_rew_count_var))
    df = 2
    count_to_count_p = 1 - stats.t.cdf(abs(count_to_count_cpd),df=df)

    # cpd_1_rew = re._CPD(C_1_rew_proj, C_1_rew_count)
    # cpd_2_rew = re._CPD(C_2_rew_proj, C_2_rew_count)
    # cpd_3_rew= re._CPD(C_3_rew_proj,C_3_rew_count)
     
    # cpd_1_2_rew_count = re._CPD(C_1_rew_count_proj, C_2_rew_count); 
    # cpd_1_3_rew_count = re._CPD(C_1_rew_count_proj, C_3_rew_count); 
    # cpd_2_3_rew_count = re._CPD(C_2_rew_count_proj, C_3_rew_count)
    
    # within_cpd =np.mean([cpd_1_rew, cpd_2_rew, cpd_3_rew],0)
    
    # count_to_count_cpd =  np.mean([cpd_1_2_rew_count + cpd_1_3_rew_count + cpd_2_3_rew_count],0)
    
    cpd_1_rew_bias, cpd_1_rew_bias_var = regression_code_session(C_1_rew, C_1_rew_proj)
    cpd_2_rew_bias, cpd_2_rew_bias_var = regression_code_session(C_2_rew, C_2_rew_proj)
    cpd_3_rew_bias, cpd_3_rew_bias_var = regression_code_session(C_3_rew, C_3_rew_proj)
   

    bias_cpd = (cpd_1_rew_bias + cpd_2_rew_bias + cpd_3_rew_bias)/np.sqrt((cpd_1_rew_bias_var + cpd_2_rew_bias_var + cpd_3_rew_bias_var))
    
    
    # cpd_1_rew_bias = re._CPD(C_1_rew_proj, C_1_rew)
    # cpd_2_rew_bias = re._CPD(C_2_rew_proj, C_2_rew)
    # cpd_3_rew_bias = re._CPD(C_3_rew_proj, C_3_rew)
   

    # bias_cpd = np.mean([cpd_1_rew_bias, cpd_2_rew_bias, cpd_3_rew_bias], 0)
   
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors+wes.Moonrise6_5.mpl_colors
    plt.figure(figsize = (20,3))
    
    
    plt.subplot(2,4,1)   
  
    j = 0
    for i in bias_cpd[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    sns.despine()
    plt.title('Vectors within Task Rewards to Rewards Biased')
    plt.ylabel(' T-stats')
    # plt.ylabel('cpd')

    
    plt.subplot(2,4,2)   
  
    j = 0
    for i in within_cpd[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    sns.despine()
    plt.title(' Vectors within Task Rewards to Reward Counts ')
    plt.ylabel(' T-stats')
    # plt.ylabel('cpd')

    plt.subplot(2,4,3)   
   
    j = 0
    for i in count_to_count_cpd[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    plt.title(str(list(predictors_all.keys())[c_to_plot]) + ' ' + 'between tasks')
    sns.despine()
    plt.ylabel(' T-stats')
    # plt.ylabel('cpd')

    
    plt.subplot(2,4,4)   

    j = 0
    for i in rew_to_count_cpd[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    plt.title('Vectors from Rewards to Reward Counts Tasks ')
    sns.despine()
    # plt.ylabel('cpd')

    
    plt.ylabel(' T-stats')
 
       
    cpd_1_2_rew_rev, cpd_1_2_rew_rev_var = regression_code_session(C_2_rew, C_1_rew_count_proj)
    
    cpd_1_3_rew_rev,cpd_1_3_rew_rev_var  = regression_code_session(C_3_rew, C_1_rew_count_proj)
    
    cpd_2_3_rew_rev,cpd_2_3_rew_rev_var  = regression_code_session(C_3_rew, C_2_rew_count_proj)
  
    count_to_rew_cpd = (cpd_1_2_rew_rev + cpd_1_3_rew_rev + cpd_2_3_rew_rev)/np.sqrt((cpd_1_2_rew_rev_var + cpd_1_3_rew_rev_var + cpd_2_3_rew_rev_var))
    
        
    # cpd_1_2_rew_rev = re._CPD(C_1_rew_count_proj, C_2_rew)
    
    # cpd_1_3_rew_rev = re._CPD(C_1_rew_count_proj, C_3_rew)
    
    # cpd_2_3_rew_rev  = re._CPD(C_2_rew_count_proj, C_3_rew)
  
    # count_to_rew_cpd = np.mean([cpd_1_2_rew_rev, cpd_1_3_rew_rev, cpd_2_3_rew_rev],0)


    cpd_1_rew, cpd_1_rew_var = regression_code_session(C_1_rew, C_1_rew_count_proj)
    cpd_2_rew, cpd_2_rew_var = regression_code_session(C_2_rew, C_2_rew_count_proj)
    cpd_3_rew, cpd_3_rew_var = regression_code_session(C_3_rew, C_3_rew_count_proj)
      
    cpd_1_2_rew_within , cpd_1_2_rew_within_var= regression_code_session(C_2_rew,C_1_rew_proj)
    cpd_1_3_rew_within, cpd_1_3_rew_within_var= regression_code_session(C_3_rew, C_1_rew_proj)
    cpd_2_3_rew_within, cpd_2_3_rew_within_var = regression_code_session(C_3_rew, C_2_rew_proj)
   
    within_cpd_rev = (cpd_1_rew + cpd_2_rew + cpd_3_rew)/ np.sqrt((cpd_1_rew_var + cpd_2_rew_var + cpd_3_rew_var))
    rew_to_count_rew = (cpd_1_2_rew_within+cpd_1_3_rew_within + cpd_2_3_rew_within)/ np.sqrt((cpd_1_2_rew_within_var + cpd_1_3_rew_within_var + cpd_2_3_rew_within_var))
 
    
    # cpd_1_rew = re._CPD(C_1_rew_count_proj,C_1_rew)
    # cpd_2_rew= re._CPD(C_2_rew_count_proj, C_2_rew)
    # cpd_3_rew = re._CPD(C_3_rew_count_proj, C_3_rew)
      
    # cpd_1_2_rew_within = re._CPD(C_1_rew_proj, C_2_rew)
    # cpd_1_3_rew_within = re._CPD(C_1_rew_proj, C_3_rew)
    # cpd_2_3_rew_within = re._CPD(C_2_rew_proj, C_3_rew)
   
    # within_cpd_rev = np.mean([cpd_1_rew, cpd_2_rew , cpd_3_rew],0)
    # rew_to_count_rew = np.mean([cpd_1_2_rew_within,cpd_1_3_rew_within, cpd_2_3_rew_within],0)
    
    cpd_1_rew_bias, cpd_1_rew_bias_var = regression_code_session(C_1_rew_count, C_1_rew_count_proj)
    cpd_2_rew_bias, cpd_2_rew_bias_var = regression_code_session(C_2_rew_count, C_2_rew_count_proj)
    cpd_3_rew_bias, cpd_3_rew_bias_var = regression_code_session(C_3_rew_count, C_3_rew_count_proj)
    
    bias_cpd_rev = (cpd_1_rew_bias+cpd_2_rew_bias+cpd_3_rew_bias)/np.sqrt((cpd_1_rew_bias_var+cpd_2_rew_bias_var+cpd_3_rew_bias_var))
  
    # cpd_1_rew_bias = re._CPD(C_1_rew_count_proj, C_1_rew_count)
    # cpd_2_rew_bias = re._CPD(C_2_rew_count_proj, C_2_rew_count)
    # cpd_3_rew_bias = re._CPD(C_3_rew_count_proj, C_3_rew_count)
    
    # bias_cpd_rev = np.mean([cpd_1_rew_bias,cpd_2_rew_bias,cpd_3_rew_bias],0)
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors+wes.Moonrise1_5.mpl_colors+wes.Moonrise6_5.mpl_colors

    plt.subplot(2,4,5)   
  
    j = 0
    for i in bias_cpd_rev[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    sns.despine()
    plt.title('Vectors within Task Reward Counts to Counts Biased Rev')
    plt.ylabel(' T-stats')
    #plt.ylabel('cpd')

    
    plt.subplot(2,4,6)   
  
    j = 0
    for i in within_cpd_rev[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    sns.despine()
    plt.title(' Vectors within Task Rewards Counts to Reward  ')
    plt.ylabel(' T-stats')
    #plt.ylabel('cpd')

    plt.subplot(2,4,7)   
   
    j = 0
    for i in rew_to_count_rew[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    plt.title('Vectors between Tasks ')
    sns.despine()
    plt.ylabel(' T-stats')
    #plt.ylabel('cpd')

    
    plt.subplot(2,4,8)   

    j = 0
    for i in count_to_rew_cpd[:-1]:
        plt.plot(i, color = c[j], label = str(j))
        j+=1
    plt.legend()  
    plt.title('Vectors from Rewards Counts to Reward Tasks ')
    sns.despine()
    plt.ylabel(' T-stats')
    #plt.ylabel('cpd')
    plt.tight_layout()
    
    C_1_rew_2 = C_1[c_to_proj]; C_2_rew_2 = C_2[c_to_proj]; C_3_rew_2 = C_3[c_to_proj]
    C_1_rew_3 = C_1[c_to_proj_3]; C_2_rew_3 = C_2[c_to_proj_3]; C_3_rew_3 = C_3[c_to_proj_3]

    cpd_1_prev_1_2, cpd_1_prev_1_2_var = regression_code_session(C_1_rew_2, C_1_rew_count_proj)
    cpd_2_prev_1_2, cpd_2_prev_1_2_var = regression_code_session(C_2_rew_2, C_2_rew_count_proj)
    cpd_3_prev_1_2, cpd_3_prev_1_2_var = regression_code_session(C_3_rew_2, C_3_rew_count_proj)
    

    prev_rew_1_2 = (cpd_1_prev_1_2 + cpd_2_prev_1_2 + cpd_3_prev_1_2) / np.sqrt((cpd_1_prev_1_2_var + cpd_2_prev_1_2_var + cpd_3_prev_1_2_var))
    
    cpd_1_prev_1_3, cpd_1_prev_1_3_var = regression_code_session(C_1_rew_3, C_1_rew_count_proj)
    cpd_2_prev_1_3, cpd_2_prev_1_3_var = regression_code_session(C_2_rew_3, C_2_rew_count_proj)
    cpd_3_prev_1_3, cpd_3_prev_1_3_var = regression_code_session(C_3_rew_3, C_3_rew_count_proj)
    
    prev_rew_1_3 = (cpd_1_prev_1_3 + cpd_2_prev_1_3 + cpd_3_prev_1_3)/ np.sqrt((cpd_1_prev_1_3_var + cpd_2_prev_1_3_var + cpd_3_prev_1_3_var))
    
    # plt.figure(figsize = (15,2))
    
    # plt.subplot(1,3,1)

    # j = 0
    # for i in count_to_count_cpd[:-1]:
    #     plt.plot(i, color = c[j], label = str(j))
    #     j+=1
    # plt.legend()  
    # plt.title(str(list(predictors_all.keys())[c_to_plot]) + ' ' + 'between tasks')
    # sns.despine()
    # plt.ylabel(' T-stats')
    # # plt.ylabel('cpd')

 
      
    # plt.subplot(1,3,2)
    # j = 0
    # for i in prev_rew_1_3[:-1]:
    #     plt.plot(i, color = c[j], label = str(j))
    #     j+=1
    # plt.legend()  
    # plt.title('from' + ' '+ str(list(predictors_all.keys())[c_to_plot]) + ' '+ 'to' + ' ' +str(list(predictors_all.keys())[c_to_proj_3]))
    # sns.despine()
    # plt.ylabel(' T-stats')
    # #plt.ylabel('cpd')
    # plt.tight_layout()
    
    # plt.subplot(1,3,3)
    # j = 0
    # for i in prev_rew_1_2[:-1]:
    #     plt.plot(i, color = c[j], label = str(j))
    #     j+=1
    # plt.legend()  
    # plt.title('from' + ' '+ str(list(predictors_all.keys())[c_to_plot]) + ' '+ 'to' + ' ' +str(list(predictors_all.keys())[c_to_proj]))
    # sns.despine()
    # plt.ylabel(' T-stats')
    # #plt.ylabel('cpd')
    # plt.tight_layout()
 
    return cell_id_cum_reward,cell_id_cum_error,cell_id_prev_ch

def runs_firing(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    for s in firing:
        neurons += s.shape[1]
    
    run_1_rew = np.zeros((neurons,121));  run_2_rew = np.zeros((neurons,121))
    run_3_rew = np.zeros((neurons,121));  run_4_rew = np.zeros((neurons,121)); run_5_rew = np.zeros((neurons,121))
    
    run_1_err = np.zeros((neurons,121));  run_2_err = np.zeros((neurons,121))
    run_3_err = np.zeros((neurons,121));  run_4_err = np.zeros((neurons,121)); run_5_err = np.zeros((neurons,121))

    n_neurons_cum = 0
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons
        reward = DM[:,2]  
        err = 0
        cum_error = []
        for r,rew in enumerate(reward):
            if reward[r] == 0 and reward[r-1] == 0:
                err +=1
            else:
                err = 0
            cum_error.append(err)
        err = 0

        cum_reward = []
        for r,rew in enumerate(reward):
            if reward[r] == 1 and reward[r-1] == 1:
                err +=1
            else:
                err = 0
            cum_reward.append(err)
        cum_reward =  np.asarray(cum_reward)
        cum_error =  np.asarray(cum_error)  
        
        run_1_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_reward == 1)[0]],0)
        run_2_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_reward == 2)[0]],0)
        run_3_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_reward == 3)[0]],0)
        run_4_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_reward == 4)[0]],0)
        run_5_rew[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_reward == 5)[0]],0)

        run_1_err[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_error == 1)[0]],0)
        run_2_err[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_error == 2)[0]],0)
        run_3_err[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_error == 3)[0]],0)
        run_4_err[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_error == 4)[0]],0)
        run_5_err[n_neurons_cum-n_neurons:n_neurons_cum ,:] = np.mean(firing_rates[np.where(cum_error == 5)[0]],0)

    return run_1_rew, run_2_rew,run_3_rew, run_4_rew,run_5_rew,run_1_err,run_2_err,run_3_err,run_4_err,run_5_err

def run():

    cell_id_cum_reward,cell_id_cum_error,cell_id_prev_ch = sequence_rewards_errors_regression_generalisation_rew(PFC,\
                area = 'PFC' + ' ', c_to_plot = 6, c_to_proj = 7, c_to_proj_3 = 8, ind_rew = 1, plot_a = False, plot_b = False)


    cell_id_cum_reward,cell_id_cum_error,cell_id_prev_ch = sequence_rewards_errors_regression_generalisation_rew(HP,\
                area = 'HP' + ' ',  c_to_plot = 6, c_to_proj = 7, c_to_proj_3 = 8,ind_rew = 1, plot_a = False, plot_b = False)



    within_taks_codes(HP, area = 'HP', perm = 500)
    within_taks_codes(PFC, area = 'PFC', perm = 500)

       # predictors_all = OrderedDict([('Reward', reward_1),
       #                                 ('Choice', choices_1),
       #                                 ('1 ago Outcome', _1_reward_1),
       #                                 ('2 ago Outcome', _2_reward_1),
       #                                 ('3 ago Outcome', _3_reward_1),
                                 
       #                                #('1 ago Choice', _1_choices_1),
       #                                # ('2 ago Choice', _2_choices_1),
       #                               #  ('3 ago Choice', _3_choices_1), 
       #                                (' Prev Rew by Ch A',_1_reward_1_ch_a),
       #                                  (' 2 Rew ago by Ch A',_2_reward_1_ch_a),
       #                                  (' 3 Rew ago by Ch A',_3_reward_1_ch_a),
       #                                  (' Prev Rew by Ch B',_1_reward_1_ch_b),
       #                                  (' 2 Rew ago by Ch B',_2_reward_1_ch_b),
       #                                  (' 3 Rew ago by Ch B',_3_reward_1_ch_b),
       #                                ('ones', ones_1)])   
               
   
