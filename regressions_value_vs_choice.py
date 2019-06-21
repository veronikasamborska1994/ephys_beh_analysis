#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:12:33 2019

@author: veronikasamborska
"""
import regressions as re 
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

plt.rc('font', **font)

def _CPD(X,y):
    
    '''Evaluate coefficient of partial determination for each predictor in X'''   
    ols = LinearRegression(fit_intercept = False)
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd


# Better correlated with the *previous* choice, or the *current choice?

def regression_Q_values_choices(experiment):
    C_1_choice = []
    cpd_1_choice = []
    C_2_choice = []
    cpd_2_choice= []
    C_3_choice = []
    cpd_3_choice  = []

    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
            reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
            same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1,\
            different_outcome_task_2, different_outcome_task_3 = re.predictors_include_previous_trial(session)     
            
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            # Task indicies 
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']

            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]        
            
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_1 = predictor_A_Task_1[1:len(task_1)]
            previous_trial_task_1 = previous_trial_task_1[1:len(task_1)]
            aligned_spikes_task_1 = aligned_spikes[1:len(task_1)]
            
            ones = np.ones(len(predictor_A_Task_1))
            
            # Task 1 
            predictors = OrderedDict([('Current Choice', predictor_A_Task_1),
                                      ('Previous Choice', previous_trial_task_1),             
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_1),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_1.reshape([len(aligned_spikes_task_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_1_choice.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_1_choice.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
            # Task 2 
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_2 = predictor_A_Task_2[len(task_1)+1:len(task_1)+len(task_2)]
            previous_trial_task_2 = previous_trial_task_2[len(task_1)+1:len(task_1)+len(task_2)]
            aligned_spikes_task_2 = aligned_spikes[len(task_1)+1:len(task_1)+len(task_2)]
            ones = np.ones(len(predictor_A_Task_2))

            predictors = OrderedDict([('Current Choice', predictor_A_Task_2),
                                      ('Previous CHoice', previous_trial_task_2),             
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_2),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_2.reshape([len(aligned_spikes_task_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_2_choice.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_2_choice.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
            # Task 3
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_3 = predictor_A_Task_3[len(task_1)+len(task_2)+1:]
            previous_trial_task_3 = previous_trial_task_3[len(task_1)+len(task_2)+1:]
            aligned_spikes_task_3 = aligned_spikes[len(task_1)+len(task_2)+1:]
            aligned_spikes_task_3 = aligned_spikes_task_3[:len(predictor_A_Task_3)]
            ones = np.ones(len(predictor_A_Task_3))

            predictors = OrderedDict([('Current Choice', predictor_A_Task_3),
                                      ('Previous Choice', previous_trial_task_3),             
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_3),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_3.reshape([len(aligned_spikes_task_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_3_choice.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_3_choice.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            

           

    cpd_1_choice = np.nanmean(np.concatenate(cpd_1_choice,0), axis = 0) # Population CPD is mean over neurons.
    cpd_2_choice = np.nanmean(np.concatenate(cpd_2_choice,0), axis = 0) # Population CPD is mean over neurons.
    cpd_3_choice = np.nanmean(np.concatenate(cpd_3_choice,0), axis = 0) # Population CPD is mean over neurons.


    return C_1_choice, C_2_choice, C_3_choice, cpd_1_choice, cpd_2_choice, cpd_3_choice, predictors

# =============================================================================
# Better correlated with the *value* of the choice (Choice[1 -1] X Value) or the *identity of the choice* (Choice[1 -1])
# Find simulated values from two models     
# #experiment_sim_Q1_HP, experiment_sim_Q4_HP=  simulate_Qtd_experiment(fits_Q1_HP, fits_Q4_HP, experiment_aligned_HP)  
# #experiment_sim_Q1_PFC, experiment_sim_Q4_PFC =  simulate_Qtd_experiment(fits_Q1_PFC, fits_Q4_PFC, experiment_aligned_PFC)  
#     
# =============================================================================

def regression_value_vs_choice(experiment,experiment_sim_Q1):
    C_1 = []
    cpd_1 = []
    C_2 = []
    cpd_2 = []
    C_3 = []
    cpd_3  = []

    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
            reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
            same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1,\
            different_outcome_task_2, different_outcome_task_3 = re.predictors_include_previous_trial(session)     
            
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            
            Q_1 = np.asarray(experiment_sim_Q1[s])
            Q_1 = Q_1[:-1]
           

            # Task indicies 
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']

            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]        
            
            Q_1 = Q_1[non_forced_array]
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_1 = predictor_A_Task_1[:len(task_1)]
            predictor_A_Task_1[np.where(predictor_A_Task_1 == 0)] = -1           
            aligned_spikes_task_1 = aligned_spikes[:len(task_1)]
            Q_1_task_1 = Q_1[:len(task_1)]
            value_choice_int = predictor_A_Task_1* Q_1_task_1

            ones = np.ones(len(predictor_A_Task_1))
            
            # Task 1 
            predictors = OrderedDict([('Current Choice', predictor_A_Task_1),
                                      ('Q_1', Q_1_task_1),  
                                      #('Choice by Value', value_choice_int),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_1),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_1.reshape([len(aligned_spikes_task_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_1.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_1.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
            # Task 2 
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_2 = predictor_A_Task_2[len(task_1):len(task_1)+len(task_2)]
            predictor_A_Task_2[np.where(predictor_A_Task_2 == 0)] = -1           
            aligned_spikes_task_2 = aligned_spikes[len(task_1):len(task_1)+len(task_2)]
            Q_1_task_2 = Q_1[len(task_1):len(task_1)+len(task_2)]
            value_choice_int_task_2 = predictor_A_Task_2* Q_1_task_2

            ones = np.ones(len(predictor_A_Task_2))

            predictors = OrderedDict([('Current Choice', predictor_A_Task_2),
                                      ('Q_1', Q_1_task_2),  
                                      #('Choice by Value', value_choice_int_task_2),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_2),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_2.reshape([len(aligned_spikes_task_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_2.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_2.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
            # Task 3
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_3 = predictor_A_Task_3[len(task_1)+len(task_2):]
            predictor_A_Task_3[np.where(predictor_A_Task_3 == 0)] = -1           
            Q_1_task_3 = Q_1[len(task_1)+len(task_2):]

            aligned_spikes_task_3 = aligned_spikes[len(task_1)+len(task_2):]
            aligned_spikes_task_3 = aligned_spikes_task_3[:len(predictor_A_Task_3)]
            
            value_choice_int_task_3 = predictor_A_Task_3* Q_1_task_3

            ones = np.ones(len(predictor_A_Task_3))

            predictors = OrderedDict([('Current Choice', predictor_A_Task_3),
                                      ('Q_1', Q_1_task_3),  
                                      #('Choice by Value', value_choice_int_task_3),
                                      ('ones', ones)])
        
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_3),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_3.reshape([len(aligned_spikes_task_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_3.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_3.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0) # Population CPD is mean over neurons.
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0) # Population CPD is mean over neurons.
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0) # Population CPD is mean over neurons.

    return C_1, C_2, C_3, cpd_1, cpd_2, cpd_3,predictors
                       

# How much of the reward effect is choice specific ( r1 = reward; r2 = reward X choice[1 -1])

def regression_reward_choice(experiment):
    C_1_reward = []
    cpd_1_reward = []
    C_2_reward = []
    cpd_2_reward = []
    C_3_reward = []
    cpd_3_reward  = []

    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
            reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
            same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1,\
            different_outcome_task_2, different_outcome_task_3 = re.predictors_include_previous_trial(session)     
            
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            
            # Task indicies 
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']

            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]        
            
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_1 = predictor_A_Task_1[:len(task_1)]
            predictor_A_Task_1[np.where(predictor_A_Task_1 == 0)] = -1           
            aligned_spikes_task_1 = aligned_spikes[:len(task_1)]
            reward_task_1 = reward[:len(task_1)]
            reward_choice_int = predictor_A_Task_1* reward_task_1

            ones = np.ones(len(predictor_A_Task_1))
            
            # Task 1 
            predictors = OrderedDict([('Reward', reward_task_1),
                                      ('Choice', predictor_A_Task_1),
                                      ('Reward x Choice', reward_choice_int),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_1),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_1.reshape([len(aligned_spikes_task_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_1_reward.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_1_reward.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
            # Task 2 
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_2 = predictor_A_Task_2[len(task_1):len(task_1)+len(task_2)]
            predictor_A_Task_2[np.where(predictor_A_Task_2 == 0)] = -1           
            aligned_spikes_task_2 = aligned_spikes[len(task_1):len(task_1)+len(task_2)]
            reward_task_2 = reward[len(task_1):len(task_1)+len(task_2)]
            reward_choice_int_task_2 = predictor_A_Task_2* reward_task_2

            ones = np.ones(len(predictor_A_Task_2))

            predictors = OrderedDict([('Reward', reward_task_2),
                                      ('Choice', predictor_A_Task_2),
                                      ('Reward x Choice', reward_choice_int_task_2),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_2),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_2.reshape([len(aligned_spikes_task_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_2_reward.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_2_reward.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
            # Task 3
            # (1) Better correlated with the *previous* choice, or the *current choice*
            predictor_A_Task_3 = predictor_A_Task_3[len(task_1)+len(task_2):]
            predictor_A_Task_3[np.where(predictor_A_Task_3 == 0)] = -1           
            reward_task_3 = reward[len(task_1)+len(task_2):]

            aligned_spikes_task_3 = aligned_spikes[len(task_1)+len(task_2):]
            aligned_spikes_task_3 = aligned_spikes_task_3[:len(predictor_A_Task_3)]
            
            reward_choice_int_task_3 = predictor_A_Task_3* reward_task_3

            ones = np.ones(len(predictor_A_Task_3))

            predictors = OrderedDict([('Reward', reward_task_3),
                                      ('Choice', predictor_A_Task_3),
                                      ('Reward x Choice', reward_choice_int_task_3),
                                      ('ones', ones)])
        
            X = np.vstack(predictors.values()).T[:len(predictor_A_Task_3),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes_task_3.reshape([len(aligned_spikes_task_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_3_reward.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_3_reward.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
    cpd_1_reward = np.nanmean(np.concatenate(cpd_1_reward,0), axis = 0) # Population CPD is mean over neurons.
    cpd_2_reward = np.nanmean(np.concatenate(cpd_2_reward,0), axis = 0) # Population CPD is mean over neurons.
    cpd_3_reward = np.nanmean(np.concatenate(cpd_3_reward,0), axis = 0) # Population CPD is mean over neurons.
    
    return C_1_reward,C_2_reward,C_3_reward,cpd_1_reward,cpd_2_reward, cpd_3_reward, predictors 


def plot():
    experiment = experiment_aligned_PFC
    experiment_sim_Q1 = experiment_sim_Q1_PFC
    C_1_reward,C_2_reward,C_3_reward,cpd_1_reward,cpd_2_reward, cpd_3_reward, predictors_reward  = regression_reward_choice(experiment)
    C_1, C_2, C_3, cpd_1, cpd_2, cpd_3,predictors = regression_value_vs_choice(experiment,experiment_sim_Q1)
    C_1_choice, C_2_choice, C_3_choice, cpd_1_choice, cpd_2_choice, cpd_3_choice, predictors_choice = regression_Q_values_choices(experiment)
    plt.figure(figsize=(10, 20))
    plt.subplot(231)   
    plotting_cpds(cpd_1_reward,cpd_2_reward,cpd_3_reward, predictors_reward,experiment)
    plt.subplot(232)   
    plotting_cpds(cpd_1,cpd_2,cpd_3, predictors,experiment)
    plt.subplot(233)   
    plotting_cpds(cpd_1_choice,cpd_2_choice,cpd_3_choice, predictors_choice,experiment)
    
    experiment = experiment_aligned_HP
    experiment_sim_Q1 = experiment_sim_Q1_HP
    C_1_reward,C_2_reward,C_3_reward,cpd_1_reward,cpd_2_reward, cpd_3_reward, predictors_reward  = regression_reward_choice(experiment)
    C_1, C_2, C_3, cpd_1, cpd_2, cpd_3,predictors = regression_value_vs_choice(experiment,experiment_sim_Q1)
    C_1_choice, C_2_choice, C_3_choice, cpd_1_choice, cpd_2_choice, cpd_3_choice, predictors_choice = regression_Q_values_choices(experiment)
    plt.subplot(234)   
    plotting_cpds(cpd_1_reward,cpd_2_reward,cpd_3_reward, predictors_reward,experiment)
    plt.subplot(235)   
    plotting_cpds(cpd_1,cpd_2,cpd_3, predictors,experiment)
    plt.subplot(236)   
    plotting_cpds(cpd_1_choice,cpd_2_choice,cpd_3_choice, predictors_choice,experiment)
    
    plt.tight_layout()


def plotting_cpds(cpd_1,cpd_2,cpd_3, predictors,experiment):
    session = experiment[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
            
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()

    cpd = np.mean([cpd_1,cpd_2,cpd_3], axis = 0)
    cpd = cpd[:,:-1]
    c =  ['black', 'turquoise', 'darkblue']
    p = [*predictors]

    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
    plt.legend()
    plt.ylabel('cpd')
    plt.xlabel('Time')
    plt.vlines(ind_reward,ymin = 0, ymax = 0.15,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 0.15,linestyles= '--', color = 'pink', label = 'Choice')
    plt.ylim(0)
    plt.legend()

