#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:17:29 2020

@author: veronikasamborska
"""

import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/RSA')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

import trials_extract_RSA as trRS
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import trials_extract_RSA as rsa
from itertools import combinations 
from itertools import permutations
import rsa_no_initiation as rs
import itertools
from math import factorial
import palettable
from palettable import wesanderson as wes
from celluloid import Camera
from matplotlib.animation import FFMpegWriter
import seaborn as sns
import RSAs as rs_in
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)
import poke_aligned_spikes as pos


def raster_process(experiment_aligned_HP,experiment_aligned_PFC):
    
    all_sessions_HP = pos.raster_plot_save(experiment_aligned_HP)
    all_sessions_PFC = pos.raster_plot_save(experiment_aligned_PFC)
    #C_HP,correlation_m_HP,predictors_HP,C_perm_HP,C_p_value = regression_RSA(experiment_aligned_HP, all_sessions_HP, perm = perm)
    #C_PFC,correlation_m_PFC,predictors_HP,C_perm_PFC = regression_RSA(experiment_aligned_PFC, all_sessions_PFC, perm = 100)

    return all_sessions_HP,all_sessions_PFC


def animal_exp(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC, m484,m479,m478,m486,m480):
    
    #HP = m484 + m479 + m483
    #PFC = m478 + m486 + m480 + m481
    # ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    # beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
    # HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
    # experiment_aligned_PFC = ha.all_sessions_aligment(PFC, all_sessions)
    # experiment_aligned_HP = ha.all_sessions_aligment(HP, all_sessions)

    HP_1 = len(m484)
    HP_2 = len(m479)
    
    PFC_1 = len(m478)
    PFC_2= len(m486)
    PFC_3 = len(m480)
    
    HP_all_sessions_1 = all_sessions_HP[:HP_1]
    HP_all_sessions_2 = all_sessions_HP[HP_1:HP_1+HP_2]
    HP_all_sessions_3 = all_sessions_HP[HP_1+HP_2:]

    HP_aligned_1 = experiment_aligned_HP[:HP_1]
    HP_aligned_2 = experiment_aligned_HP[HP_1:HP_1+HP_2]
    HP_aligned_3 = experiment_aligned_HP[HP_1+HP_2:]

    PFC_all_sessions_1 = all_sessions_PFC[:PFC_1]
    PFC_all_sessions_2 = all_sessions_PFC[PFC_1:PFC_1+PFC_2]
    PFC_all_sessions_3 = all_sessions_PFC[PFC_1+PFC_2:PFC_1+PFC_2+PFC_3]
    PFC_all_sessions_4 = all_sessions_PFC[PFC_1+PFC_2+PFC_3:]

    PFC_aligned_1 = experiment_aligned_PFC[:PFC_1]
    PFC_aligned_2 = experiment_aligned_PFC[PFC_1:PFC_1+PFC_2]
    PFC_aligned_3 = experiment_aligned_PFC[PFC_1+PFC_2:PFC_1+PFC_2+PFC_3]
    PFC_aligned_4 = experiment_aligned_PFC[PFC_1+PFC_2+PFC_3:]
    
    return  HP_all_sessions_1, HP_all_sessions_2, HP_all_sessions_3, HP_aligned_1,\
    HP_aligned_2, HP_aligned_3, PFC_all_sessions_1, PFC_all_sessions_2, PFC_all_sessions_3,\
    PFC_all_sessions_4, PFC_aligned_1, PFC_aligned_2, PFC_aligned_3, PFC_aligned_4
    


def permute(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,m484,m479,m478,m486,m480):
    
    HP_all_sessions_1, HP_all_sessions_2, HP_all_sessions_3, HP_aligned_1,\
    HP_aligned_2, HP_aligned_3, PFC_all_sessions_1, PFC_all_sessions_2, PFC_all_sessions_3,\
    PFC_all_sessions_4, PFC_aligned_1, PFC_aligned_2, PFC_aligned_3, PFC_aligned_4 = animal_exp(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                                m484,m479,m478,m486,m480)
     
    all_sessions_perm_HP_PFC = [HP_all_sessions_1,HP_all_sessions_2, HP_all_sessions_3,PFC_all_sessions_1,PFC_all_sessions_2, PFC_all_sessions_3,PFC_all_sessions_4]
    aligned_perm_HP_PFC = [HP_aligned_1,HP_aligned_2,HP_aligned_3,PFC_aligned_1,PFC_aligned_2,PFC_aligned_3,PFC_aligned_4]


    return all_sessions_perm_HP_PFC,aligned_perm_HP_PFC



def extract_trials(experiment, all_sessions, t_start, t_end): 
# =============================================================================
#   This function separates pokes and finds firing rates of each neuron around the pokes.
#   Inputs are:
#        experiment_aligned_HP orexperiment_aligned_PFC,  all_sessions - made in poke_aligned_spikes script
#   whcih contains firing rates around every port entry. 
#       time_window - specifying which time window to use (default is of a width of 1 which is 50ms).
#       For the purposed of animation a different function --> matrices_for_plots(experiment,all_sessions) includes a loop
#       that goes through the full length of the poke aligned histogram (-1.5 to 1.5 sec);
#       this can be changed in poke_aligned_spikes.py script if necessary
#   Outputs are:
#       A in three tasks rewarded, unrewarded;
#       Initiation in three tasks x,y - same physical port, z - different port and also acts as B choice in x task
#       B choices in tasks y and z. These are combined into a big matrix which is the output of the function. 
# =============================================================================    

   session_list_poke_a_task_x_r_spikes = []; session_list_poke_a_task_x_nr_spikes = []

   session_list_poke_a_task_y_r_spikes = []; session_list_poke_a_task_y_nr_spikes = []

   session_list_poke_a_task_z_r_spikes = []; session_list_poke_a_task_z_nr_spikes = []

   session_list_poke_initiation_task_x_spikes = []; session_list_poke_initiation_task_y_spikes = []
   
   session_list_poke_initiation_b_task_z_spikes = []
   
   session_list_poke_choice_b_task_x_spikes_r = []; session_list_poke_choice_b_task_x_spikes_nr = []
   
   session_list_poke_b_task_y_spikes_r = []; session_list_poke_b_task_y_spikes_nr = []
   
   session_list_poke_b_task_z_spikes_r = [];  session_list_poke_b_task_z_spikes_nr = []
   for s,session in enumerate(experiment):
        
        session = experiment[s]
        all_neurons_all_spikes_raster_plot_task = all_sessions[s]
        all_neurons_all_spikes_raster_plot_task = np.asarray(all_neurons_all_spikes_raster_plot_task)
        average_time_spikes = np.mean(all_neurons_all_spikes_raster_plot_task[:,:,t_start:t_end],axis = 2)

        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            
            poke_a_task_x,poke_a_task_y,poke_a_task_z,outcomes,poke_initiation_task_x,poke_initiation_task_y,poke_initiation_b_task_z,\
            poke_choice_b_task_x,poke_4,poke_5 = rsa.seperate_a_into_tasks(session) 
            
            # Get rid off the time dimension        
            n_trials, n_neurons = average_time_spikes.shape   

            # The z-scores of input "a", with any columns including non-finite
            # numbers replaced by all zeros.
            average_time_spikes[:, np.logical_not(np.all(np.isfinite(average_time_spikes), axis=0))] = 0

           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_a_task_x_r_spikes = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 1)), :]
            poke_a_task_x_nr_spikes  = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 0)),:]
           
            poke_a_task_y_r_spikes = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 1)),:]
            poke_a_task_y_nr_spikes  = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 0)),:]
           
            poke_a_task_z_r_spikes = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 1)),:]
            poke_a_task_z_nr_spikes  = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 0)),:]
           
            # Extract spikes for I in three tasks (rewarded and non-rewarded)
            poke_initiation_task_x_spikes = average_time_spikes[np.where(poke_initiation_task_x == 1),:]
            
            poke_initiation_task_y_spikes = average_time_spikes[np.where(poke_initiation_task_y == 1),:]

            poke_initiation_b_task_z_spikes = average_time_spikes[np.where(poke_initiation_b_task_z ==1),:]
            
            poke_choice_b_task_x_spikes_r = average_time_spikes[np.where((poke_choice_b_task_x ==1) & (outcomes == 1)),:]
            poke_choice_b_task_x_spikes_nr = average_time_spikes[np.where((poke_choice_b_task_x ==1) & (outcomes == 0)),:]

            poke_b_task_y_spikes_r = average_time_spikes[np.where((poke_4 ==1) & (outcomes == 1)),:]
            poke_b_task_y_spikes_nr = average_time_spikes[np.where((poke_4 ==1) & (outcomes == 0)),:]
            
            poke_b_task_z_spikes_r = average_time_spikes[np.where((poke_5 ==1) & (outcomes == 1)),:]
            poke_b_task_z_spikes_nr = average_time_spikes[np.where((poke_5 ==1) & (outcomes == 0)),:]
            
            #Find mean firing rates for each neuron on each type of trial
            
            mean_poke_a_task_x_r_spikes = np.mean(poke_a_task_x_r_spikes[0,:,:], axis = 0)
            mean_poke_a_task_x_nr_spikes = np.mean(poke_a_task_x_nr_spikes[0,:,:], axis = 0)
            
            mean_poke_a_task_y_r_spikes = np.mean(poke_a_task_y_r_spikes[0,:,:], axis = 0)
            mean_poke_a_task_y_nr_spikes = np.mean(poke_a_task_y_nr_spikes[0,:,:], axis = 0)

            mean_poke_a_task_z_r_spikes = np.mean(poke_a_task_z_r_spikes[0,:,:], axis = 0)
            mean_poke_a_task_z_nr_spikes = np.mean(poke_a_task_z_nr_spikes[0,:,:], axis = 0)

            mean_poke_initiation_task_x_spikes = np.mean(poke_initiation_task_x_spikes[0,:,:], axis = 0)
            mean_poke_initiation_task_y_spikes = np.mean(poke_initiation_task_y_spikes[0,:,:], axis = 0)
            
            mean_poke_initiation_b_task_z_spikes = np.mean(poke_initiation_b_task_z_spikes[0,:,:],axis = 0)
        
            mean_poke_choice_b_task_x_spikes_r = np.mean(poke_choice_b_task_x_spikes_r[0,:,:], axis = 0)
            mean_poke_choice_b_task_x_spikes_nr = np.mean(poke_choice_b_task_x_spikes_nr[0,:,:], axis = 0)
            
            mean_poke_b_task_y_spikes_r = np.mean(poke_b_task_y_spikes_r[0,:,:], axis = 0)
            mean_poke_b_task_y_spikes_nr = np.mean(poke_b_task_y_spikes_nr[0,:,:], axis = 0)

            mean_poke_b_task_z_spikes_r = np.mean(poke_b_task_z_spikes_r[0,:,:], axis = 0)
            mean_poke_b_task_z_spikes_nr = np.mean(poke_b_task_z_spikes_nr[0,:,:], axis = 0)

            
            session_list_poke_a_task_x_r_spikes.append(mean_poke_a_task_x_r_spikes)
            session_list_poke_a_task_x_nr_spikes.append(mean_poke_a_task_x_nr_spikes)
        
            session_list_poke_a_task_y_r_spikes.append(mean_poke_a_task_y_r_spikes)
            session_list_poke_a_task_y_nr_spikes.append(mean_poke_a_task_y_nr_spikes)
         
            session_list_poke_a_task_z_r_spikes.append(mean_poke_a_task_z_r_spikes)
            session_list_poke_a_task_z_nr_spikes.append(mean_poke_a_task_z_nr_spikes)
        
            session_list_poke_initiation_task_x_spikes.append(mean_poke_initiation_task_x_spikes)
            session_list_poke_initiation_task_y_spikes.append(mean_poke_initiation_task_y_spikes)
           
            session_list_poke_initiation_b_task_z_spikes.append(mean_poke_initiation_b_task_z_spikes)
            
            session_list_poke_choice_b_task_x_spikes_r.append(mean_poke_choice_b_task_x_spikes_r)
            session_list_poke_choice_b_task_x_spikes_nr.append(mean_poke_choice_b_task_x_spikes_nr)
           
            session_list_poke_b_task_y_spikes_r.append(mean_poke_b_task_y_spikes_r)
            session_list_poke_b_task_y_spikes_nr.append(mean_poke_b_task_y_spikes_nr)
            
            session_list_poke_b_task_z_spikes_r.append(mean_poke_b_task_z_spikes_r)
            session_list_poke_b_task_z_spikes_nr.append(mean_poke_b_task_z_spikes_nr)
            
  
   session_list_poke_a_task_x_r_spikes = np.concatenate(session_list_poke_a_task_x_r_spikes,0)
   session_list_poke_a_task_x_nr_spikes = np.concatenate(session_list_poke_a_task_x_nr_spikes,0)
    
   session_list_poke_a_task_y_r_spikes = np.concatenate(session_list_poke_a_task_y_r_spikes,0)
   session_list_poke_a_task_y_nr_spikes = np.concatenate(session_list_poke_a_task_y_nr_spikes,0)
     
   session_list_poke_a_task_z_r_spikes = np.concatenate(session_list_poke_a_task_z_r_spikes,0)
   session_list_poke_a_task_z_nr_spikes = np.concatenate(session_list_poke_a_task_z_nr_spikes,0)
    
   session_list_poke_initiation_task_x_spikes = np.concatenate(session_list_poke_initiation_task_x_spikes,0)
   session_list_poke_initiation_task_y_spikes = np.concatenate(session_list_poke_initiation_task_y_spikes,0)
    
       
   session_list_poke_initiation_b_task_z_spikes = np.concatenate(session_list_poke_initiation_b_task_z_spikes,0)
       
   session_list_poke_choice_b_task_x_spikes_r = np.concatenate(session_list_poke_choice_b_task_x_spikes_r,0)
   session_list_poke_choice_b_task_x_spikes_nr = np.concatenate(session_list_poke_choice_b_task_x_spikes_nr,0)
    
   session_list_poke_b_task_y_spikes_r = np.concatenate(session_list_poke_b_task_y_spikes_r,0)
   session_list_poke_b_task_y_spikes_nr = np.concatenate(session_list_poke_b_task_y_spikes_nr,0)
    
   session_list_poke_b_task_z_spikes_r = np.concatenate(session_list_poke_b_task_z_spikes_r,0)
   session_list_poke_b_task_z_spikes_nr  = np.concatenate(session_list_poke_b_task_z_spikes_nr,0)
   
   matrix_for_correlations = np.vstack([session_list_poke_a_task_x_r_spikes,session_list_poke_a_task_x_nr_spikes,session_list_poke_a_task_y_r_spikes,\
                                       session_list_poke_a_task_y_nr_spikes,session_list_poke_a_task_z_r_spikes,session_list_poke_a_task_z_nr_spikes,\
                                       session_list_poke_initiation_task_x_spikes,session_list_poke_initiation_task_y_spikes,\
                                       session_list_poke_initiation_b_task_z_spikes,session_list_poke_choice_b_task_x_spikes_r,\
                                       session_list_poke_choice_b_task_x_spikes_nr,session_list_poke_b_task_y_spikes_r,\
                                       session_list_poke_b_task_y_spikes_nr,session_list_poke_b_task_z_spikes_r,session_list_poke_b_task_z_spikes_nr])

   return matrix_for_correlations

def _cpd(X,y):
    
    '''Evaluate coefficient of partial determination for each predictor in X'''
    
    pdes = np.linalg.pinv(X)
    pe = np.matmul(pdes,y)
  
    Y_predict = np.matmul(X,pe)
    sse = np.sum((Y_predict - y)**2, axis=0)

    #sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        pdes_i = np.linalg.pinv(X_i)
        pe_i = np.matmul(pdes_i,y)

        Y_predict_i = np.matmul(X_i,pe_i)
        sse_X_i = np.sum((Y_predict_i- y)**2, axis=0)

        #sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[i]=(sse_X_i-sse)/sse_X_i
    return cpd

def _cpd_swap(X,y,ind):
    
    '''Evaluate coefficient of partial determination for each predictor in X perm test'''
    
    pdes = np.linalg.pinv(X)
    pe = np.matmul(pdes,y)
  
    Y_predict = np.matmul(X,pe)
    sse = np.sum((Y_predict - y)**2, axis=0)

    #sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        pdes_i = np.linalg.pinv(X_i)
        pe_i = np.matmul(pdes_i,y)

        Y_predict_i = np.matmul(X_i,pe_i)
        sse_X_i = np.sum((Y_predict_i- y)**2, axis=0)

        #sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[i]=(sse_X_i-sse)/sse_X_i
    return cpd

def permute_labels():
    
    cpd_PFC_labels,cpd_HP_labels, correlation_m_PFC_labels,correlation_m_HP_labels, diff_perm, p_labels = regression_RSA_perm_labels(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,40, 45,perm = True)

def regression_RSA_perm_labels(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,t_start, t_end,perm = True):
    '''Regression of RSA predictors on the actual data matrix'''
    
   
    matrix_for_correlations_HP = extract_trials(experiment_aligned_HP, all_sessions_HP, t_start, t_end)

    correlation_m_HP = np.corrcoef(matrix_for_correlations_HP)
    correlation_m_f_HP = correlation_m_HP.flatten()
    physical_rsa = rs_in.RSA_physical_rdm()
    physical_rsa  = 1*physical_rsa.flatten()
    choice_ab_rsa = rs_in.RSA_a_b_initiation_rdm()
    choice_ab_rsa = 1*choice_ab_rsa.flatten()
    reward_no_reward = rs_in.reward_rdm()
    reward_no_reward = 1*reward_no_reward.flatten()
    reward_at_choices = rs_in.reward_choice_space()
    reward_at_choices = 1*reward_at_choices.flatten()
    choice_initiation_rsa =  rs_in.choice_vs_initiation()
    choice_initiation_rsa = 1*choice_initiation_rsa.flatten()
    a_bs_task_specific_rsa = rs_in.a_bs_task_specific()
    a_bs_task_specific_rsa = 1*a_bs_task_specific_rsa.flatten()
    remapping_a_to_b = 1*(rs_in.remapping_a_to_b()).flatten()
    ones = np.ones(len(choice_ab_rsa))
    

    predictors = OrderedDict([('Space' , physical_rsa),
                              ('A vs B', choice_ab_rsa),
                              ('Reward',reward_no_reward),
                              ('Reward at A vs B',reward_at_choices),
                              ('Choice vs Initiation',choice_initiation_rsa),
                              ('A and B Task Specific',a_bs_task_specific_rsa)])
                              #('constant', ones)])                                        
           
    X = np.vstack(predictors.values()).T[:len(physical_rsa),:].astype(float)
   
    y_HP = correlation_m_f_HP
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_HP)
    cpd_HP = _cpd(X,y_HP)
    
    matrix_for_correlations_PFC = extract_trials(experiment_aligned_PFC, all_sessions_PFC, t_start, t_end)
    correlation_m_PFC = np.corrcoef(matrix_for_correlations_PFC)
    correlation_m_f_PFC = correlation_m_PFC.flatten()
    y_PFC = correlation_m_f_PFC
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_PFC)
    cpd_PFC = _cpd(X,y_PFC)
    
    
    labels = 6

    num_rounds = factorial(6)
    diff_perm = np.zeros((int(num_rounds),len(predictors)))
    perms = list(permutations(range(labels)))
    if perm:
        nn = 0
        for i,ind in enumerate(perms):
            if i < len(perms)-1:
                X_hp = X[:,perms[i]]
                X_pfc = X[:,perms[i+1]]
    
    
               
                ols.fit(X_hp,y_HP)
                cpd_HP_perm = _cpd(X_hp,y_HP)
                 
                ols.fit(X_pfc,y_PFC)
                cpd_PFC_perm = _cpd(X_pfc,y_PFC)
    
                diff_perm[nn,:] = abs(cpd_PFC_perm - cpd_HP_perm)
                nn += 1
            
    p = np.percentile(diff_perm,95, axis = 0)

        
    return cpd_PFC,cpd_HP, correlation_m_PFC,correlation_m_HP, diff_perm, p

   

def regression_RSA_perm(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                              m484,m479,m478,m486,m480,t_start, t_end,perm = True):
    '''Regression of RSA predictors on the actual data matrix'''
    
   
    matrix_for_correlations_HP = extract_trials(experiment_aligned_HP, all_sessions_HP, t_start, t_end)

    correlation_m_HP = np.corrcoef(matrix_for_correlations_HP)
    correlation_m_f_HP = correlation_m_HP.flatten()
    physical_rsa = rs_in.RSA_physical_rdm()
    physical_rsa  = 1*physical_rsa.flatten()
    choice_ab_rsa = rs_in.RSA_a_b_initiation_rdm()
    choice_ab_rsa = 1*choice_ab_rsa.flatten()
    reward_no_reward = rs_in.reward_rdm()
    reward_no_reward = 1*reward_no_reward.flatten()
    reward_at_choices = rs_in.reward_choice_space()
    reward_at_choices = 1*reward_at_choices.flatten()
    choice_initiation_rsa =  rs_in.choice_vs_initiation()
    choice_initiation_rsa = 1*choice_initiation_rsa.flatten()
    a_bs_task_specific_rsa = rs_in.a_bs_task_specific()
    a_bs_task_specific_rsa = 1*a_bs_task_specific_rsa.flatten()
    
    ones = np.ones(len(choice_ab_rsa))
    

    predictors = OrderedDict([('Space' , physical_rsa),
                              ('A vs B', choice_ab_rsa),
                              ('Reward',reward_no_reward),
                              ('Reward at A vs B',reward_at_choices),
                              ('Choice vs Initiation',choice_initiation_rsa),
                              ('A and B Task Specific',a_bs_task_specific_rsa),
                              ('constant', ones)])                                        
           
    X = np.vstack(predictors.values()).T[:len(physical_rsa),:].astype(float)
    # Check if regression is rank deficient 
    #print(X.shape[1])  
    #rank = matrix_rank(X) 
    #print(rank)
    y_HP = correlation_m_f_HP
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_HP)
    cpd_HP = _cpd(X,y_HP)
    C_HP = ols.coef_

    matrix_for_correlations_PFC = extract_trials(experiment_aligned_PFC, all_sessions_PFC, t_start, t_end)
    correlation_m_PFC = np.corrcoef(matrix_for_correlations_PFC)
    correlation_m_f_PFC = correlation_m_PFC.flatten()
    y_PFC = correlation_m_f_PFC
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_PFC)
    cpd_PFC = _cpd(X,y_PFC)
    C_PFC = ols.coef_

    #diff_real = abs(cpd_PFC - cpd_HP)

    
    all_sessions_perm_HP_PFC,aligned_perm_HP_PFC  = permute(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                              m484,m479,m478,m486,m480)
    animals_PFC = [1,2,3,4]
    animals_HP = [5,6,7]
    m, n = len(animals_PFC), len(animals_HP)
   # more_extreme = np.zeros(len(predictors))
    num_rounds = factorial(m + n) / (factorial(m)*factorial(n))
    diff_perm = np.zeros((int(num_rounds),len(predictors)))
    diff_perm_C = np.zeros((int(num_rounds),len(predictors)))
    
    if perm:
        nn = 0
        for indices_PFC in combinations(range(m + n), m):
            indices_HP = [i for i in range(m + n) if i not in indices_PFC]
            PFC_permute_all_sessions = np.asarray(all_sessions_perm_HP_PFC)[np.asarray(indices_PFC)]
            PFC_permute_aligned = [aligned_perm_HP_PFC[np.asarray(indices_PFC)[0]],aligned_perm_HP_PFC[np.asarray(indices_PFC)[1]],\
                                    aligned_perm_HP_PFC[np.asarray(indices_PFC)[2]],aligned_perm_HP_PFC[np.asarray(indices_PFC)[3]]]
           
            
            HP_permute_all_sessions = np.asarray(all_sessions_perm_HP_PFC)[np.asarray(indices_HP)]
            HP_permute_aligned = [aligned_perm_HP_PFC[np.asarray(indices_HP)[0]],aligned_perm_HP_PFC[np.asarray(indices_HP)[1]],aligned_perm_HP_PFC[np.asarray(indices_HP)[2]]]
            
            PFC_permute_all_sessions = list(itertools.chain(*PFC_permute_all_sessions))
            PFC_permute_aligned = list(itertools.chain(*PFC_permute_aligned))

            HP_permute_all_sessions = list(itertools.chain(*HP_permute_all_sessions))
            HP_permute_aligned = list(itertools.chain(*HP_permute_aligned))

            matrix_for_correlations_perm_PFC = extract_trials(PFC_permute_aligned, PFC_permute_all_sessions, t_start, t_end)
           
            y_perm_PFC = np.corrcoef(matrix_for_correlations_perm_PFC).flatten()
            ols.fit(X,y_perm_PFC)
            cpd_PFC_perm = _cpd(X,y_perm_PFC)
            C_PFC_perm = ols.coef_
 
            matrix_for_correlations_perm_HP = extract_trials(HP_permute_aligned, HP_permute_all_sessions,t_start, t_end)
            y_perm_HP = np.corrcoef(matrix_for_correlations_perm_HP).flatten()
            ols.fit(X,y_perm_HP)
            cpd_HP_perm = _cpd(X,y_perm_HP)
            C_HP_perm = ols.coef_

            
            diff_perm[nn,:] = abs(cpd_PFC_perm - cpd_HP_perm)
            diff_perm_C[nn,:] = abs(C_PFC_perm - C_HP_perm)

            nn += 1
            
    #         if len(np.where(diff_perm > diff_real)[0])> 0:
    #             more_extreme[np.where(diff_perm > diff_real)] = more_extreme[np.where(diff_perm > diff_real)]+1

    p = np.percentile(diff_perm,95, axis = 0)
    p_C = np.percentile(diff_perm_C,95, axis = 0)
    # num_rounds = factorial(m + n) / (factorial(m)*factorial(n))
    # p_value = more_extreme / num_rounds
    
        
    return cpd_PFC,cpd_HP, correlation_m_PFC,correlation_m_HP, p, p_C, C_PFC, C_HP

def _CPD(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([X.shape[1]])

    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[i]=(sse_X_i-sse)/sse_X_i
    return cpd

def regression_RSA_perm_against_flip(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                              t_start, t_end,m484,m479,m478,m486,m480):
    '''Regression of RSA predictors on the actual data matrix'''
    
    HP_all_sessions_1, HP_all_sessions_2, HP_all_sessions_3, HP_aligned_1,\
    HP_aligned_2, HP_aligned_3, PFC_all_sessions_1, PFC_all_sessions_2, PFC_all_sessions_3,\
    PFC_all_sessions_4, PFC_aligned_1, PFC_aligned_2, PFC_aligned_3, PFC_aligned_4 = animal_exp(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                m484,m479,m478,m486,m480)
     
    all_sessions_perm_HP_PFC = [HP_all_sessions_1,HP_all_sessions_2, HP_all_sessions_3,PFC_all_sessions_1,PFC_all_sessions_2, PFC_all_sessions_3,PFC_all_sessions_4]
    aligned_perm_HP_PFC = [HP_aligned_1,HP_aligned_2,HP_aligned_3,PFC_aligned_1,PFC_aligned_2,PFC_aligned_3,PFC_aligned_4]

    cpd_PFC_perm = []
    cpd_HP_perm = []
   

    physical_rsa = rs_in.RSA_physical_rdm()
    physical_rsa  = 1*physical_rsa.flatten()
    choice_ab_rsa = rs_in.RSA_a_b_initiation_rdm()
    choice_ab_rsa = 1*choice_ab_rsa.flatten()
    reward_no_reward = rs_in.reward_rdm()
    reward_no_reward = 1*reward_no_reward.flatten()
    reward_at_choices = rs_in.reward_choice_space()
    reward_at_choices = 1*reward_at_choices.flatten()
    choice_initiation_rsa =  rs_in.choice_vs_initiation()
    choice_initiation_rsa = 1*choice_initiation_rsa.flatten()
    a_bs_task_specific_rsa = rs_in.a_bs_task_specific()
    a_bs_task_specific_rsa = 1*a_bs_task_specific_rsa.flatten()
    
    ones = np.ones(len(choice_ab_rsa))
    

    predictors = OrderedDict([('Space' , physical_rsa),
                              ('A vs B', choice_ab_rsa),
                              ('Reward',reward_no_reward),
                              ('Reward at A vs B',reward_at_choices),
                              ('Choice vs Initiation',choice_initiation_rsa),
                              ('A and B Task Specific',a_bs_task_specific_rsa),
                              ('constant', ones)])                                        
           
    X = np.vstack(predictors.values()).T[:len(physical_rsa),:].astype(float)
 
    # HP animals
    matrix_for_correlations_HP_1 = np.corrcoef(extract_trials( HP_aligned_1,HP_all_sessions_1, t_start, t_end)).flatten()
    matrix_for_correlations_HP_2 =  np.corrcoef(extract_trials( HP_aligned_2,HP_all_sessions_2, t_start, t_end)).flatten()
    matrix_for_correlations_HP_3 =  np.corrcoef(extract_trials( HP_aligned_3,HP_all_sessions_3, t_start, t_end)).flatten()

    y_HP = matrix_for_correlations_HP_1
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_HP)
    C_HP_1 = ols.coef_

    y_HP = matrix_for_correlations_HP_2
    ols.fit(X,y_HP)
    C_HP_2 = ols.coef_

    y_HP = matrix_for_correlations_HP_3
    ols.fit(X,y_HP)
    C_HP_3 = ols.coef_
    C_HP_flip = []
    C_HP = np.mean([C_HP_1,C_HP_2,C_HP_3],0)
    perms_HP =  np.arange(0,3)
    for perm in perms_HP:
        HP = np.asarray([C_HP_1,C_HP_2,C_HP_3])
        HP[perm] = HP[perm]* (-1)
        C_HP_flip.append(np.mean(HP,0))



    matrix_for_correlations_PFC_1 = np.corrcoef(extract_trials( PFC_aligned_1,PFC_all_sessions_1, t_start, t_end)).flatten()
    matrix_for_correlations_PFC_2 = np.corrcoef(extract_trials( PFC_aligned_2,PFC_all_sessions_2, t_start, t_end)).flatten()
    matrix_for_correlations_PFC_3 = np.corrcoef(extract_trials( PFC_aligned_3,PFC_all_sessions_3, t_start, t_end)).flatten()
    matrix_for_correlations_PFC_4 = np.corrcoef(extract_trials( PFC_aligned_3,PFC_all_sessions_3, t_start, t_end)).flatten()

    y_PFC = matrix_for_correlations_PFC_1
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_PFC)
    C_PFC_1 = ols.coef_
     
    y_PFC = matrix_for_correlations_PFC_2
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_PFC)
    C_PFC_2 = ols.coef_
     
    y_PFC = matrix_for_correlations_PFC_3
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_PFC)
    C_PFC_3 = ols.coef_

    y_PFC = matrix_for_correlations_PFC_4
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y_PFC)
    C_PFC_4 = ols.coef_
    C_PFC = np.mean([C_PFC_1,C_PFC_2,C_PFC_3,C_PFC_4],0)
    perms_PFC = list(permutations(range(4)))
    
    C_PFC_flip = []
    perms_PFC = np.arange(0,4)
    for perm in perms_PFC:
        PFC = np.asarray([C_PFC_1,C_PFC_2,C_PFC_3,C_PFC_4])
        PFC[perm] = PFC[perm]*(-1)
        C_PFC_flip.append(np.mean(PFC,0))


    return C_PFC,C_HP,C_HP_flip,C_PFC_flip

     
    
def rsa_across_time(experiment,all_sessions):
   
    cue = np.arange(28,50) #100 ms before choice poke entry
    reward =np.arange(29,51) #500 ms after choice poke entry
    
    ind_PFC = []
    ind_HP = []
    C_PFC_list = []
    C_HP_list = []
    p_val_HP_all = []
    p_val_PFC_all = []

    for start,end in zip(cue, reward):
  
        C_PFC,C_HP,C_HP_flip,C_PFC_flip =  regression_RSA_perm_against_flip(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                               start, end, m484,m479,m478,m486,m480)
        C_PFC_list.append(C_PFC)
        C_HP_list.append(C_HP)

        C_HP_flip = np.asarray(C_HP_flip)
        C_PFC_flip = np.asarray(C_PFC_flip)
        
        p_val_PFC = []
        for p, pred in enumerate(C_PFC):
            C_PFC_flip_p = C_PFC_flip[:,p]
            p_val_PFC.append(np.percentile(C_PFC_flip_p,95))
        p_val_HP = []
        for p, pred in enumerate(C_HP):
            C_HP_flip_p = C_HP_flip[:,p]
            p_val_HP.append(np.percentile(C_HP_flip_p,95))
        ind_PFC.append(np.where(C_PFC > p_val_PFC)[0])
        ind_HP.append(np.where(C_HP > p_val_HP)[0])
        p_val_HP_all.append(p_val_HP)
        p_val_PFC_all.append(p_val_PFC)

    space_HP_flip = []; a_b_HP_flip = []; rew_HP_flip = []; rew_a_b_HP_flip = []; ch_init_HP_flip  = []; a_spec_HP_flip = []
    space_PFC_flip = []; a_b_PFC_flip = []; rew_PFC_flip = []; rew_a_b_PFC_flip = []; ch_init_PFC_flip = []; a_spec_PFC_flip = []
                                                        
    for i, ind in enumerate(ind_PFC):
        if 0 in ind_HP[i]:
            space_HP_flip.append(i);
        if 0 in ind_PFC[i]:
            space_PFC_flip.append(i);
        if 1 in ind_HP[i]:
            a_b_HP_flip.append(i);
        if 1 in ind_PFC[i]:
            a_b_PFC_flip.append(i);
        if 2 in ind_HP[i]:
            rew_HP_flip.append(i);
        if 2 in ind_PFC[i]:
            rew_PFC_flip.append(i);
        if 3 in ind_HP[i]:
            rew_a_b_HP_flip.append(i);
        if 3 in ind_PFC[i]:
            rew_a_b_PFC_flip.append(i);
        if 4 in ind_HP[i]:
            ch_init_HP_flip.append(i);
        if 4 in ind_PFC[i]:
            ch_init_PFC_flip.append(i);
        if 5 in ind_HP[i]:
            a_spec_HP_flip.append(i);
        if 5 in ind_PFC[i]:
            a_spec_PFC_flip.append(i);
       
         
    cpd_PFC_list = []
    cpd_HP_list = []
    correlation_m_PFC_list = []
    correlation_m_HP_list = []
    p_C_list = []
    C_PFC_list = []
    C_HP_list = []
    
    p_value_list = []
    for start,end in zip(cue, reward):
        
        cpd_PFC,cpd_HP, correlation_m_PFC,correlation_m_HP, p, p_C, C_PFC, C_HP = regression_RSA_perm(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                              m484,m479,m478,m486,m480,t_start = start, t_end = end ,perm = True)
        p_value_list.append(p)
        p_C_list.append(p_C)
        C_PFC_list.append(C_PFC)
        C_HP_list.append(C_HP)
        cpd_PFC_list.append(cpd_PFC)
        cpd_HP_list.append(cpd_HP)
        correlation_m_PFC_list.append(correlation_m_PFC)
   
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    fig_n = 1
    
    p_value_multiple_comparisons = np.max(p_C_list,0)

    difference_space = []; space_HP = []; space_PFC = []
    
    a_b_HP = []; a_b_PFC = []; difference_a_b = []

    rew_HP = []; rew_PFC= []; difference_rew= []
    
    rew_a_b_HP = []; rew_a_b_PFC = []; difference_rew_a_b = []

    ch_init_HP = []; ch_init_PFC = []; difference_ch_init = []

    a_spec_HP = []; a_spec_PFC = []; difference_a_spec = []
    
    
    for i in range(len(cpd_PFC_list)):
        
        difference_space.append((abs(cpd_HP_list[i][0]- cpd_PFC_list[i][0])))
        space_HP.append(cpd_HP_list[i][0])
        space_PFC.append(cpd_PFC_list[i][0])
        
        # difference_space.append((abs(C_HP_list[i][0]- C_PFC_list[i][0])))
        # space_HP.append(C_HP_list[i][0])
        # space_PFC.append(C_PFC_list[i][0])
        
        a_b_HP.append(cpd_HP_list[i][1])
        a_b_PFC.append(cpd_PFC_list[i][1])
        difference_a_b.append((abs(cpd_HP_list[i][1]- cpd_PFC_list[i][1])))

        # a_b_HP.append(C_HP_list[i][1])
        # a_b_PFC.append(C_PFC_list[i][1])
        # difference_a_b.append((abs(C_HP_list[i][1]- C_PFC_list[i][1])))

        rew_HP.append(cpd_HP_list[i][2])
        rew_PFC.append(cpd_PFC_list[i][2])
        difference_rew.append((abs(cpd_HP_list[i][2]- cpd_PFC_list[i][2])))
        
        # rew_HP.append(C_HP_list[i][2])
        # rew_PFC.append(C_PFC_list[i][2])
        # difference_rew.append((abs(C_HP_list[i][2]- C_PFC_list[i][2])))
       
        rew_a_b_HP.append(cpd_HP_list[i][3])
        rew_a_b_PFC.append(cpd_PFC_list[i][3])
        difference_rew_a_b.append((abs(cpd_HP_list[i][3]- cpd_PFC_list[i][3])))

        # rew_a_b_HP.append(C_HP_list[i][3])
        # rew_a_b_PFC.append(C_PFC_list[i][3])
        # difference_rew_a_b.append((abs(C_HP_list[i][3]- C_PFC_list[i][3])))

        ch_init_HP.append(cpd_HP_list[i][4])
        ch_init_PFC.append(cpd_PFC_list[i][4])
        difference_ch_init.append((abs(cpd_HP_list[i][4]- cpd_PFC_list[i][4])))

        # ch_init_HP.append(C_HP_list[i][4])
        # ch_init_PFC.append(C_PFC_list[i][4])
        # difference_ch_init.append((abs(C_HP_list[i][4]- C_PFC_list[i][4])))

        a_spec_HP.append(cpd_HP_list[i][5])
        a_spec_PFC.append(cpd_PFC_list[i][5])
        difference_a_spec.append((abs(cpd_HP_list[i][5]- cpd_PFC_list[i][5])))
        
        # a_spec_HP.append(C_HP_list[i][5])
        # a_spec_PFC.append(C_PFC_list[i][5])
        # difference_a_spec.append((abs(C_HP_list[i][5]- C_PFC_list[i][5])))
    
    
         
        
    #annotate sig 
        
    space_sig = np.where(difference_space >= p_value_multiple_comparisons[0])[0]
    a_b_sig = np.where(difference_a_b >= p_value_multiple_comparisons[1])[0]
    rew_sig = np.where(difference_rew >= p_value_multiple_comparisons[2])[0]
    rew_a_b_sig = np.where(difference_rew_a_b >= p_value_multiple_comparisons[3])[0]
    ch_init_sig = np.where(difference_ch_init >= p_value_multiple_comparisons[4])[0]
    a_b_spec_sig = np.where(difference_a_spec >= p_value_multiple_comparisons[5])[0]
    x = np.arange(22)
    plt.figure(figsize = (10,2))
    plt.subplot(1,6,1)
    isl = wes.Royal2_5.mpl_colors

    plt.plot(np.asarray(space_HP)*100, color = isl[0])
    plt.plot(np.asarray(space_PFC)*100, color = isl[3])
    plt.title('Space')
    plt.xticks([0,2,4,7,10,12,14,16,18,20,22],['-0.1','C','0.1','O','0.3','0.4','0.5', '0.6','0.8','0.9','1'])
    for s in space_sig:
        plt.annotate('*',xy = [s, (np.max([space_HP,space_PFC])+0.01)*100])    
    # for s in space_HP_flip:
    #     plt.annotate('*',xy = [s, (np.max([space_HP,space_PFC])+0.02)], color = 'pink')
    # for s in space_PFC_flip:
    #     plt.annotate('*',xy = [s, (np.max([space_HP,space_PFC])+0.03)], color = 'grey')
    plt.ylim((np.min([space_HP,space_PFC])-0.03)*100,(np.max([space_HP,space_PFC])+0.03)*100)
   
    plt.ylabel('CPD %')
    
   
    plt.subplot(1,6,2)

    plt.plot(np.asarray(a_b_HP)*100, color = isl[0])
    plt.plot(np.asarray(a_b_PFC)*100, color = isl[3])
    plt.title('Choice A vs B')
    plt.xticks([0,2,4,7,10,12,14,16,18,20,22],['-0.1','C','0.1','O','0.3','0.4','0.5', '0.6','0.8','0.9','1'])
    for s in a_b_sig:
        plt.annotate('*',xy = [s, (np.max([a_b_HP,a_b_PFC])+0.01)*100])
    # for s in a_b_HP_flip:
    #     plt.annotate('*',xy = [s, (np.max([a_b_HP,a_b_PFC])+0.02)], color = 'pink')
    # for s in a_b_PFC_flip:
    #     plt.annotate('*',xy = [s, (np.max([a_b_HP,a_b_PFC])+0.03)], color = 'grey')
    plt.ylim((np.min([a_b_HP,a_b_PFC])-0.03)*100,(np.max([a_b_HP,a_b_PFC])+0.03)*100)


    plt.subplot(1,6,3)

    plt.plot(np.asarray(rew_HP)*100, color = isl[0])
    plt.plot(np.asarray(rew_PFC)*100, color = isl[3])
    plt.title('Reward')
    plt.xticks([0,2,4,7,10,12,14,16,18,20,22],['-0.1','C','0.1','O','0.3','0.4','0.5', '0.6','0.8','0.9','1'])
    for s in rew_sig:
        plt.annotate('*',xy = [s, (np.max([rew_HP,rew_PFC])+0.01)*100])
    # for s in rew_HP_flip:
    #     plt.annotate('*',xy = [s, (np.max([rew_HP,rew_PFC])+0.02)], color = 'pink')
    # for s in rew_PFC_flip:
    #     plt.annotate('*',xy = [s, (np.max([rew_HP,rew_PFC])+0.03)], color = 'grey')
    plt.ylim((np.min([rew_HP,rew_PFC])-0.03)*100,(np.max([rew_HP,rew_PFC])+0.03)*100)

  
    plt.subplot(1,6,4)

    plt.plot(np.asarray(rew_a_b_HP)*100, color = isl[0])
    plt.plot(np.asarray(rew_a_b_PFC)*100, color = isl[3])
    plt.title('Reward at A vs B')
    plt.xticks([0,2,4,7,10,12,14,16,18,20,22],['-0.1','C','0.1','O','0.3','0.4','0.5', '0.6','0.8','0.9','1'])
    for s in rew_a_b_sig:
        plt.annotate('*',xy = [s, (np.max([rew_a_b_HP,rew_a_b_PFC])+0.01)*100]) 
    # for s in rew_a_b_HP_flip:
    #     plt.annotate('*',xy = [s, (np.max([rew_a_b_HP,rew_a_b_PFC])+0.02)], color = 'pink')
    # for s in rew_a_b_PFC_flip:
    #     plt.annotate('*',xy = [s, (np.max([rew_a_b_HP,rew_a_b_PFC])+0.03)], color = 'grey')
    plt.ylim((np.min([rew_a_b_HP,rew_a_b_PFC])-0.03)*100,(np.max([rew_a_b_HP,rew_a_b_PFC])+0.03)*100)
 
       
    plt.subplot(1,6,5)

    plt.plot(np.asarray(ch_init_HP)*100, color = isl[0])
    plt.plot(np.asarray(ch_init_PFC)*100, color = isl[3])
    plt.title('Choice vs Init')
    plt.xticks([0,2,4,7,10,12,14,16,18,20,22],['-0.1','C','0.1','O','0.3','0.4','0.5', '0.6','0.8','0.9','1'])
    for s in ch_init_sig:
        plt.annotate('*',xy = [s, (np.max([ch_init_HP,ch_init_PFC])+0.01)*100])
    # for s in ch_init_HP_flip:
    #     plt.annotate('*',xy = [s, (np.max([ch_init_HP,ch_init_PFC])+0.02)], color = 'pink')
    # for s in ch_init_PFC_flip:
    #     plt.annotate('*',xy = [s, (np.max([ch_init_HP,ch_init_PFC])+0.03)], color = 'grey')
    plt.ylim((np.min([ch_init_HP,ch_init_PFC])-0.03)*100,(np.max([ch_init_HP,ch_init_PFC])+0.03)*100)

 
    plt.subplot(1,6,6)

    plt.plot(np.asarray(a_spec_HP)*100, color = isl[0], label = 'HP')
    plt.plot(np.asarray(a_spec_PFC)*100, color = isl[3], label = 'PFC')
    plt.title('A Task Specific')
    for s in a_b_spec_sig:
        plt.annotate('*',xy = [s, (np.max([a_spec_HP,a_spec_PFC])+0.01)*100])
    # for s in a_spec_HP_flip:
    #     plt.annotate('*',xy = [s, (np.max([a_spec_HP,a_spec_PFC])+0.02)], color = 'pink')
    # for s in a_spec_PFC_flip:
    #     plt.annotate('*',xy = [s, (np.max([a_spec_HP,a_spec_PFC])+0.03)], color = 'grey')
    plt.ylim((np.min([a_spec_HP,a_spec_PFC])-0.03)*100,(np.max([a_spec_HP,a_spec_PFC])+0.03)*100)
    sns.despine()
    plt.legend()
    plt.xticks([0,2,4,7,10,12,14,16,18,20,22],['-0.1','C','0.1','O','0.3','0.4','0.5', '0.6','0.8','0.9','1'])
    plt.tight_layout()

    for i in range(len(cpd_PFC_list)):
        # Predictor RSA Matrices 
        physical_rsa = rs_in.RSA_physical_rdm()
        choice_ab_rsa = rs_in.RSA_a_b_initiation_rdm()
        reward_no_reward = rs_in.reward_rdm() 
        reward_at_choices = rs_in.reward_choice_space()
        choice_initiation_rsa =  rs_in.choice_vs_initiation()
        a_bs_task_specific_rsa = rs_in.a_bs_task_specific()
    
      
        # Set up the axes with gridspec
        fig = plt.figure(figsize=(6, 25))
        grid = plt.GridSpec(13, 3, hspace=0.5, wspace=1)
        
        space_plt = fig.add_subplot(grid[1:3, 0])
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR','2 I T1',\
                   '2 I T2', '3 I T3',  '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
        plt.xticks([])
        plt.title('Space')
    
        choice_plt = fig.add_subplot(grid[3:5, 0])
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR','2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
        plt.xticks([])
        plt.title('A vs B')
    
        reward_no_reward_plt = fig.add_subplot(grid[5:7, 0])
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', '2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
        plt.xticks([])
        plt.title('Reward')
    
        reward_at_choices_plt = fig.add_subplot(grid[7:9, 0])
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', '2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR')) 
        plt.xticks([])
        plt.title('Reward at Choice')   
        
        
        choice_initiation_plt = fig.add_subplot(grid[9:11, 0])
        
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', ' 2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
        plt.xticks([])
        plt.title('Choice vs Initiation')
          
        
        a_bs_task_specific_rsa_plt = fig.add_subplot(grid[11:13, 0])
        
        plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', ' 2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', ' 2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
        plt.title('As and Bs within Task')
        
        
        ind = np.arange(len(cpd_HP)-1)
    
        bar_plot = fig.add_subplot(grid[1:4, 1:3])
        plt.xticks(ind,('Space','A vs B','Reward','Reward at Choice', 'Choice vs Init', 'A and B Task'), rotation = 'vertical')
    
        plt.ylabel('Regression Coefficient')
         
        trial_corr_plot_PFC = fig.add_subplot(grid[ 5:9, 1:3])
    
        plt.xticks([])
        plt.title('PFC')
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', '2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
        
        trial_corr_plot_HP = fig.add_subplot(grid[ 9:13, 1:3])
        plt.title('HP')
    
        plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR', ' 2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
       
        plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                   '1 A T3 R','1 A T3 NR',' 2 I T1',\
                   '2 I T2', '3 I T3', '3 B T1 R',\
                   '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
        
        isl = wes.Royal2_5.mpl_colors
    
        # bar_plot.bar(ind, cpd_HP[:-1], color = isl[0], label = 'HP')
        # bar_plot.bar(ind, cpd_PFC[:-1],bottom=cpd_HP[:-1], color = isl[2], label = 'PFC')
        # x = -0.2
        # for i in p_value[:-1]:
        #     bar_plot.annotate(np.round(i,3),xy = (x,0.8))
        #     x += 1
    
            
        # bar_plot.set_ylim((0,1.1))
        # bar_plot.legend()
    
        
        space_plt.imshow(physical_rsa,aspect = 'auto', cmap = cmap)
        choice_plt.imshow(choice_ab_rsa,aspect = 'auto',cmap = cmap)
        reward_no_reward_plt.imshow(reward_no_reward,aspect = 'auto',cmap = cmap)
        reward_at_choices_plt.imshow(reward_at_choices,aspect = 'auto',cmap = cmap)
        choice_initiation_plt.imshow(choice_initiation_rsa,aspect = 'auto',cmap = cmap)
        a_bs_task_specific_rsa_plt.imshow(a_bs_task_specific_rsa,aspect = 'auto',cmap = cmap)
        trial_corr_plot_PFC.imshow(correlation_m_PFC_list[i], aspect = 1,cmap = cmap)
        trial_corr_plot_HP.imshow(correlation_m_HP_list[i], aspect = 1,cmap = cmap)

        
        
        trial_corr_plot_PFC.imshow(correlation_m_PFC_list[i], aspect = 1,cmap = cmap, vmin = 0, vmax = 1)
        hp = trial_corr_plot_HP.imshow(correlation_m_HP_list[i], aspect = 1,cmap = cmap, vmin = 0, vmax = 1)
        plt.colorbar(hp)

        width = 0.35  # the width of the bars
        difference = abs(cpd_HP_list[i][:-1]- cpd_PFC_list[i][:-1])
        p_values_annotate = difference>p_value_multiple_comparisons[:-1]
        
        
        bar_plot.bar(ind - width/2, cpd_HP_list[i][:-1], width, color = isl[0],label = 'HP')
        bar_plot.bar(ind + width/2, cpd_PFC_list[i][:-1],width, color = isl[3], label = 'PFC')
        
        #bar_plot.bar(ind, cpd_HP_list[i][:-1], color = isl[0], label = 'HP')
        #bar_plot.bar(ind, cpd_PFC_list[i][:-1],bottom=cpd_HP[:-1], color = isl[3], label = 'PFC')
      
        x = - 0.2
        for i in p_values_annotate:
            print(i)
            bar_plot.annotate(i,xy = (x,1))
            x += 1
        bar_plot.legend()
        
        
        space_plt.imshow(physical_rsa,aspect = 'auto',cmap = cmap)
        choice_plt.imshow(choice_ab_rsa,aspect = 'auto',cmap = cmap)
        reward_no_reward_plt.imshow(reward_no_reward,aspect = 1,cmap = cmap)
        reward_at_choices_plt.imshow(reward_at_choices,aspect = 1,cmap = cmap)
        choice_initiation_plt.imshow(choice_initiation_rsa,aspect = 'auto',cmap = cmap)
        a_bs_task_specific_rsa_plt.imshow(a_bs_task_specific_rsa,aspect = 'auto',cmap = cmap)
        bar_plot.set_ylim((0,1.2))
        fig_n +=1
        plt.savefig('/Users/veronikasamborska/Desktop/RSA_pic/RSA_new'+str(fig_n)+'.pdf')
        
    
    #30,50
        
    camera = Camera(fig)
    cpd_PFC_list = []
    cpd_HP_list = []
    correlation_m_PFC_list = []
    correlation_m_HP_list = []
    
    # 10 for 0.5 second  
    #start = [30,35]
    #end =[50,50]

    cue = np.arange(30,60)
    reward =np.arange(31,61)
    
    p_value_list = []
    for start,end in zip(cue, reward):
        
        cpd_PFC,cpd_HP, correlation_m_PFC,correlation_m_HP, p = regression_RSA_perm(all_sessions_HP,all_sessions_PFC, experiment_aligned_HP, experiment_aligned_PFC,\
                                                                                              m484,m479,m478,m486,m480,t_start = start, t_end = end ,perm = True)
        p_value_list.append(p)

        cpd_PFC_list.append(cpd_PFC)
        cpd_HP_list.append(cpd_HP)
        correlation_m_PFC_list.append(correlation_m_PFC)
        correlation_m_HP_list.append(correlation_m_HP)

        #slider_plot.vlines(30, ymin= 0, ymax = 1, color = 'red')
        # slider_plot.vlines(i, ymin = 0 , ymax = 1,linewidth = 4)
            
        trial_corr_plot_PFC.imshow(correlation_m_PFC, aspect = 1,cmap = cmap, vmin = 0, vmax = 1)
        hp = trial_corr_plot_HP.imshow(correlation_m_HP, aspect = 1,cmap = cmap, vmin = 0, vmax = 1)
        width = 0.35  # the width of the bars

        bar_plot.bar(ind - width/2, cpd_HP[:-1], width, color = isl[0],label = 'HP')
        bar_plot.bar(ind + width/2, cpd_PFC[:-1],width, color = isl[3], label = 'PFC')
        
        #bar_plot.bar(ind, cpd_HP_list[i][:-1], color = isl[0], label = 'HP')
        #bar_plot.bar(ind, cpd_PFC_list[i][:-1],bottom=cpd_HP[:-1], color = isl[3], label = 'PFC')
      
       # x = - 0.2
        
        #for i in p_value[:-1]:
        #    bar_plot.annotate(np.round(i,3),xy = (x,1))
        #    x += 1
          

        space_plt.imshow(physical_rsa,aspect = 'auto',cmap = cmap)
        choice_plt.imshow(choice_ab_rsa,aspect = 'auto',cmap = cmap)
        reward_no_reward_plt.imshow(reward_no_reward,aspect = 1,cmap = cmap)
        reward_at_choices_plt.imshow(reward_at_choices,aspect = 1,cmap = cmap)
        choice_initiation_plt.imshow(choice_initiation_rsa,aspect = 'auto',cmap = cmap)
        a_bs_task_specific_rsa_plt.imshow(a_bs_task_specific_rsa,aspect = 'auto',cmap = cmap)
        bar_plot.set_ylim((0,1.2))
        bar_plot.legend()
        camera.snap()        

    p_value_multiple_comparisons = np.max(p_value_list,0)
    animation = camera.animate(interval = 200)
    FFwriter = FFMpegWriter(fps = 1, bitrate=2000) 
    animation.save('/Users/veronikasamborska/Desktop/HP_PFC_rsa.mp4', writer=FFwriter)




