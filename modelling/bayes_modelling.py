#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:43:36 2019

@author: veronikasamborska
"""

# Bayesian Hidden Markov Model fitting to behaviour 

import numpy as np
import sys
import regressions as re 
import scipy.optimize as op
import math 
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from numba import jit
import matplotlib.pyplot as plt


def simulate_experiment(params, experiment):
    
    bayes_posterior = []
    bayes_prior = []

    for s,session in enumerate(experiment):
        

        Posterior_correct_incorrect, Prior_correct_incorrect = simulate_bayes(session, params)
        
        bayes_posterior.append(Posterior_correct_incorrect)
        bayes_prior.append(Prior_correct_incorrect)
        
    return bayes_posterior, bayes_prior

 
def run_script():
    
    HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
    
    experiment_aligned_m484 = ha.all_sessions_aligment(m484, all_sessions)
    experiment_aligned_m479 = ha.all_sessions_aligment(m479, all_sessions)
    experiment_aligned_m483 = ha.all_sessions_aligment(m483, all_sessions)

    experiment_aligned_m478 = ha.all_sessions_aligment(m478, all_sessions)
    experiment_aligned_m486 = ha.all_sessions_aligment(m486, all_sessions)
    experiment_aligned_m480 = ha.all_sessions_aligment(m480, all_sessions)
    experiment_aligned_m481 = ha.all_sessions_aligment(m481, all_sessions)

     
    fits_484 = fit_sessions(experiment_aligned_m484, model())    
    fits_479 = fit_sessions(experiment_aligned_m479, model())  
    fits_483 = fit_sessions(experiment_aligned_m483, model())
    
    fits_m478 = fit_sessions(experiment_aligned_m478, model())
    fits_m486 = fit_sessions(experiment_aligned_m486, model())
    fits_m480 = fit_sessions(experiment_aligned_m480, model())
    fits_m481 = fit_sessions(experiment_aligned_m481, model())
    
    
    fits_484_mean = np.mean(fits_484['params'], axis = 0)
    fits_479_mean = np.mean(fits_479['params'], axis = 0)
    fits_483_mean = np.mean(fits_483['params'], axis = 0)
    
    fits_478_mean = np.mean(fits_m478['params'], axis = 0)
    fits_486_mean = np.mean(fits_m486['params'], axis = 0)
    fits_480_mean = np.mean(fits_m480['params'], axis = 0)
    fits_481_mean = np.mean(fits_m481['params'], axis = 0)
    
    # HP
    bayes_posterior_m484, bayes_prior_m484  =  simulate_experiment(fits_484_mean, experiment_aligned_m484)  
    bayes_posterior_m479, bayes_prior_m479 =   simulate_experiment(fits_479_mean, experiment_aligned_m479)  
    bayes_posterior_m483, bayes_prior_m483  =  simulate_experiment(fits_483_mean, experiment_aligned_m483)  
    
    # PFC
    bayes_posterior_m478, bayes_prior_m478 =  simulate_experiment(fits_478_mean, experiment_aligned_m478)  
    bayes_posterior_m486, bayes_prior_m486 =  simulate_experiment(fits_486_mean, experiment_aligned_m486) 
    bayes_posterior_m480, bayes_prior_m480 =  simulate_experiment(fits_480_mean, experiment_aligned_m480)  
    bayes_posterior_m481, bayes_prior_m481 =  simulate_experiment(fits_481_mean, experiment_aligned_m481)  

    bayes_prior_PFC = bayes_prior_m478+bayes_prior_m486+bayes_prior_m480+bayes_prior_m481
    bayes_posterior_PFC = bayes_posterior_m478+bayes_posterior_m486+bayes_posterior_m480+bayes_posterior_m481
    
    bayes_prior_HP = bayes_prior_m484+bayes_prior_m479+bayes_prior_m483
    bayes_posterior_HP = bayes_posterior_m484+bayes_posterior_m479+bayes_posterior_m483
    
    cpd_PFC, predictors_PFC, C_PFC, C_sq_PFC = regression_bayes(experiment_aligned_PFC,bayes_prior_PFC, bayes_posterior_PFC)
    cpd_HP, predictors_HP, C_HP, C_sq_HP = regression_bayes(experiment_aligned_HP,bayes_prior_HP, bayes_posterior_HP)
    
    

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number.
    

def array_sigmoid(T,P,alpha):
    
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    
    Probs = np.zeros(P.shape)
    TdP = -T*(P[:,1]-alpha)
    TdP[TdP > log_max_float] = log_max_float # Protection against overflow in exponential.    
    Probs[:,0] = 1./(1. + np.exp(TdP))
    Probs[:,1] = 1. - Probs[:,0]

    return Probs


def trans_UC(values_U, param_ranges):
    
    'Transform parameters from unconstrained to constrained space.'
    
    if param_ranges[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, param_ranges):
        
        if rng   == 'unit':  # Range: 0 - 1.
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + math.exp(-value)))  # Don't allow values smaller than 1e-
            
        elif rng   == 'half':  # Range: 0 - 0.5
            if value < -16.:
                value = -16.
            values_T.append(0.5/(1. + math.exp(-value)))  # Don't allow values smaller than 1e-7
            
        elif rng == 'pos':  # Range: 0 - inf
            if value > 16.:
                value = 16.
            values_T.append(math.exp(value))  # Don't allow values bigger than ~ 1e7.
            
        elif rng == 'unc': # Range: - inf - inf.
            values_T.append(value)
            
        elif rng =='cross_term':
             if value < -16.:
                 value = -16.
             values_T.append(-1./(1. + math.exp(-value)))  # Don't allow values smaller than -1
             
    return np.array(values_T)

def protected_log(x):
    
    'Return log of x protected against giving -inf for very small values of x.'
    
    return np.log(((1e-200)/2)+(1-(1e-200))*x)


class model():
    
    def __init__(self):
        
        self.name = 'Bayes'
        self.param_names  = ['alpha', 'iTemp', 'sigma']
        self.params       = [ 0.5   ,  5. , 0.1 ]  
        self.param_ranges = ['unit' ,'pos', 'unit']
        self.n_params = 3

    @jit
    def session_likelihood(self, session, params):
       
        # Unpack trial events.
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        
        forced_trials = session.trial_data['forced_trial']
        non_forced_array = np.where(forced_trials == 0)[0]
        choices = choices[non_forced_array]
        outcomes = outcomes[non_forced_array]
        
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
        predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
        predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
        reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
        same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1, different_outcome_task_2, different_outcome_task_3, switch = re.predictors_include_previous_trial(session)     
        stay = np.asarray(switch)
      
        n_trials = choices.shape[0]
        
        #Unpack parameters.
        
        alpha, iTemp, sigma = params  
        
        reward_prob_correct  = 0.75
        reward_prob_incorrect  = 0.25
        no_reward_prob_correct  = 0.25
        no_reward_prob_incorrect  = 0.75

        #iTemp  =  2.46759374
        #alpha = 0.35039014

        # Variables.
        Posterior_correct_incorrect = np.zeros([n_trials, 2])  # Array for correct and incorrect posteriors
        Posterior_correct_incorrect[0,:] =  0.5 # First choices start  with 0.5 priors for both correct and incorrect choice 

        Prior_correct_incorrect = np.zeros([n_trials, 2])  # Array for correct and incorrect priors
        
        # Switch array --> ones when  choice  is the same; reverse such that 1s mean there was a switch between current and last trial
        switch_choice = stay-1
        switch_choice = switch_choice*(-1)
        outcomes = outcomes[1:]
        
        for i, (o, c) in enumerate(zip(outcomes, switch_choice)): # loop over trials.    
            
            if c  == 1:
                # Prior that the decision is correct if a switch occured; 
                # Reversal P *  Posterior past choice  was correct + (1-Reversal P)* Posterior past choice was incorrect
                Prior_correct_incorrect[i, 0] = ((sigma)* Posterior_correct_incorrect[i, 0])  + ((1-sigma)*Posterior_correct_incorrect[i, 1])
                Prior_correct_incorrect[i, 1] =  1-Prior_correct_incorrect[i, 0]
                
                # Prior that the decision is correct if no  switch occured; 
                # Choice is correct with  probability (1-Reversal P) & given past choice being correct + (Reversal P)* Posterior past choice was incorrect
            
            elif c  == 0:                
                Prior_correct_incorrect[i, 0] = ((1-sigma)* Posterior_correct_incorrect[i, 0])  + ((sigma)*Posterior_correct_incorrect[i, 1])
                Prior_correct_incorrect[i, 1] =  1-Prior_correct_incorrect[i, 0]
                
            if o == 1:                
                # Decision is correct based on the probability of getting a reward weighted by the prior of the trial being correct
                #/Probability of getting a reward in correct prior; probability of getting no reward given the incorrect prior 
                Posterior_correct_incorrect[i+1, 0] = reward_prob_correct * Prior_correct_incorrect[i, 0]/((reward_prob_correct*Prior_correct_incorrect[i,0])+ (reward_prob_incorrect*Prior_correct_incorrect[i,1]))
                Posterior_correct_incorrect[i+1, 1] = 1-Posterior_correct_incorrect[i+1, 0]
           
            elif o == 0:
                 # Decision is correct based on the probability of getting no reward weighted by the prior of the trial being correct
                #/Probability of getting a reward in an correct prior; probability of getting no reward given the incorrect prior 
                Posterior_correct_incorrect[i+1, 0] = no_reward_prob_correct*Prior_correct_incorrect[i, 0]/(((no_reward_prob_correct)*Prior_correct_incorrect[i,0])+ ((no_reward_prob_incorrect)*Prior_correct_incorrect[i,1]))
                Posterior_correct_incorrect[i+1, 1] = 1-Posterior_correct_incorrect[i+1, 0]
           
        

        # Evaluate choice probabilities and likelihood.     
        Posterior_correct_incorrect = Posterior_correct_incorrect[:-1,:]
        
        choice_probs = array_sigmoid(iTemp,Posterior_correct_incorrect, alpha)
        
        trial_log_likelihood_switch = switch_choice*protected_log(choice_probs[:,0])
        trial_log_likelihood_stay = stay*protected_log(choice_probs[:,1])
        session_log_likelihood = (np.sum(trial_log_likelihood_switch)/np.sum(switch_choice))+(np.sum(trial_log_likelihood_stay)/sum(stay))

        ## Checking everything is working okay  
#        plt.figure(3)
#        plt.plot(outcomes, "v", color = 'red', alpha = 0.7, markersize=3)
#        plt.plot(switch_choice,"x", color = 'green', alpha = 0.7, markersize=3)
#        plt.plot(Posterior_correct_incorrect[:,1], color = 'grey', alpha = 0.7)
#        plt.plot(trial_log_likelihood_switch, color = 'red')
#        plt.plot(trial_log_likelihood_stay, color = 'blue')
#        plt.plot(choice_probs[:,0],'black', alpha = 0.4)
#        plt.plot(Posterior_correct_incorrect[:,1], color = 'grey', alpha = 0.7)       
#        plt.title(str(session_log_likelihood))
        
        return session_log_likelihood
    

def simulate_bayes(session, params):

    # Unpack trial events.
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    choices = choices[non_forced_array]
    outcomes = outcomes[non_forced_array]
    
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
    reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
    same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1,\
    different_outcome_task_2, different_outcome_task_3, switch = re.predictors_include_previous_trial(session)     
    
    stay = np.asarray(switch)
  
    n_trials = choices.shape[0]
    
    #Unpack parameters.
    
    alpha, iTemp, sigma = params  
    
    reward_prob_correct  = 0.75
    reward_prob_incorrect  = 0.25
    no_reward_prob_correct  = 0.25
    no_reward_prob_incorrect  = 0.75

  
    # Variables.
    Posterior_correct_incorrect = np.zeros([n_trials, 2])  # Array for correct and incorrect posteriors
    Posterior_correct_incorrect[0,:] =  0.5 # First choices start  with 0.5 priors for both correct and incorrect choice 

    Prior_correct_incorrect = np.zeros([n_trials, 2])  # Array for correct and incorrect priors
    
    # Switch array --> ones when  choice  is the same; reverse such that 1s mean there was a switch between current and last trial
    switch_choice = stay-1
    switch_choice = switch_choice*(-1)
    outcomes = outcomes[1:]
    
    for i, (o, c) in enumerate(zip(outcomes, switch_choice)): # loop over trials.          
        if c  == 1:
            # Prior that the decision is correct if a switch occured; 
            # Reversal P *  Posterior past choice  was correct + (1-Reversal P)* Posterior past choice was incorrect
            Prior_correct_incorrect[i, 0] = ((sigma)* Posterior_correct_incorrect[i, 0])  + ((1-sigma)*Posterior_correct_incorrect[i, 1])
            Prior_correct_incorrect[i, 1] =  1-Prior_correct_incorrect[i, 0]
            
            # Prior that the decision is correct if no  switch occured; 
            # Choice is correct with  probability (1-Reversal P) & given past choice being correct + (Reversal P)* Posterior past choice was incorrect
        
        elif c  == 0:                
            Prior_correct_incorrect[i, 0] = ((1-sigma)* Posterior_correct_incorrect[i, 0])  + ((sigma)*Posterior_correct_incorrect[i, 1])
            Prior_correct_incorrect[i, 1] =  1-Prior_correct_incorrect[i, 0]
            
        if o == 1:                
            # Decision is correct based on the probability of getting a reward weighted by the prior of the trial being correct
            #/Probability of getting a reward in correct prior; probability of getting no reward given the incorrect prior 
            Posterior_correct_incorrect[i+1, 0] = reward_prob_correct * Prior_correct_incorrect[i, 0]/((reward_prob_correct*Prior_correct_incorrect[i,0])+ (reward_prob_incorrect*Prior_correct_incorrect[i,1]))
            Posterior_correct_incorrect[i+1, 1] = 1-Posterior_correct_incorrect[i+1, 0]
       
        elif o == 0:
             # Decision is correct based on the probability of getting no reward weighted by the prior of the trial being correct
            #/Probability of getting a reward in an correct prior; probability of getting no reward given the incorrect prior 
            Posterior_correct_incorrect[i+1, 0] = no_reward_prob_correct*Prior_correct_incorrect[i, 0]/(((no_reward_prob_correct)*Prior_correct_incorrect[i,0])+ ((no_reward_prob_incorrect)*Prior_correct_incorrect[i,1]))
            Posterior_correct_incorrect[i+1, 1] = 1-Posterior_correct_incorrect[i+1, 0]
       
    Posterior_correct_incorrect = Posterior_correct_incorrect[:-1]
    Prior_correct_incorrect = Prior_correct_incorrect[:-1]
    return Posterior_correct_incorrect, Prior_correct_incorrect
 


    
def fit_session(session, agent, repeats = 2, brute_init = True, verbose = False):
    
    '''Find maximum likelihood parameter estimates for a session or list of sessions.'''
    
    # RL models require parameter transformation from unconstrained to true space.
    method = 'Nelder-Mead'
    calculates_gradient = False
    fit_func   = lambda params: -agent.session_likelihood(session, trans_UC(params, agent.param_ranges))
    n_trials = len(session.trial_data['choices'])
    fits = []
    for i in range(repeats): # Perform fitting. 

        if agent.n_params <= 2 and i == 0 and brute_init: 
           # Initialise minimisation with brute force search.
           ranges = tuple([(-5,5) for i in range(agent.n_params)])
           init_params = op.brute(fit_func, ranges, Ns =  20, 
                                  full_output = True, finish = None)[0]
        else:
            
            init_params = np.random.normal(0, 3., agent.n_params)

        fits.append(op.minimize(fit_func, init_params, jac = calculates_gradient,
                                method = method, options = {'disp': verbose}))           

    fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.

    session_fit = {'likelihood' : - fit['fun'],
                   'param_names': agent.param_names,
                   'n_trials': n_trials} 
    
    session_fit['params'] = trans_UC(fit['x'], agent.param_ranges)

    return session_fit

def session_fits(experiment, agent, repeats = 2, brute_init = True, verbose = False):
        
    '''Find maximum likelihood parameter estimates for a session or list of sessions.'''
    for session in experiment:
        # RL models require parameter transformation from unconstrained to true space.
        method = 'Nelder-Mead'
        calculates_gradient = False
        fit_func   = lambda params: -agent.session_likelihood(session, trans_UC(params, agent.param_ranges))
        n_trials = len(session.trial_data['choices'])
        fits = []
        for i in range(repeats): # Perform fitting. 
    
            if agent.n_params <= 2 and i == 0 and brute_init: 
               # Initialise minimisation with brute force search.
               ranges = tuple([(-5,5) for i in range(agent.n_params)])
               init_params = op.brute(fit_func, ranges, Ns =  20, 
                                      full_output = True, finish = None)[0]
            else:
                
                init_params = np.random.normal(0, 3., agent.n_params)
    
            fits.append(op.minimize(fit_func, init_params, jac = calculates_gradient,
                                    method = method, options = {'disp': verbose}))           

    fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.

    session_fit = {'likelihood' : - fit['fun'],
                   'param_names': agent.param_names,
                   'n_trials': n_trials} 
    
    session_fit['params'] = trans_UC(fit['x'], agent.param_ranges)

    
def fit_sessions(sessions, agent):
    '''Perform maximum likelihood fitting on a list of sessions and return
    dictionary with fit information.  For logistic regression agents a one 
    sample ttest is used to check if each parameter loading is significantly
    different from zero.'''
    
    fit_list = [fit_session(session, agent) for session in tqdm(sessions)]
    fits = {'param_names': agent.param_names,
            'n_params'   : agent.n_params,
            'likelihood' : [f['likelihood'] for f in fit_list],
            'params'     : np.array([f['params'] for f in fit_list]),
            'n_trials'     : np.array([f['n_trials'] for f in fit_list])}
    
    return fits



def regression_bayes(experiment,bayes_prior, bayes_posterior):
    C = []
    cpd = []
    C_sq  = []
    
    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            prior = np.asarray(bayes_prior[s][:,0])
            posterior = np.asarray(bayes_posterior[s][:,0])

            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[1:len(choices),:,:]
            outcomes = outcomes[non_forced_array]
            outcomes = outcomes[1:]
            choices  = choices[1:]
            # Getting out task indicies

            ones = np.ones(len(choices))
            
            predictors = OrderedDict([('Prior', prior),
                                      ('Posterior', posterior),
                                      ('choice', choices),
                                      ('reward', outcomes),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X,y)
            C.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            sq = ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)
            sq = sq**2
            C_sq.append(sq)
            cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))


    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0) # Population CPD is mean over neurons.
    C_sq_all = np.concatenate(C,0)

    C = np.nanmean(np.concatenate(C,0), axis = 0) 

    C_sq = np.nanmean(np.concatenate(C_sq,0), axis = 0) # Population CPD is mean over neurons.

    return cpd, predictors, C_sq,C_sq_all


def plotting_beta_sq():
    session = experiment_aligned_PFC[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
            
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()

    cpd, predictors, C_sq, C_sq_all = regression_bayes(experiment_aligned_PFC,bayes_prior_PFC, bayes_posterior_PFC)
    C_sq_all_mean = np.mean(C_sq_all, axis = 1)
    C_sq_all_mean_prior = C_sq_all_mean[:,3]
    C_sq_all_mean_prior = sorted(C_sq_all_mean_prior)
    
    #corr = np.correlate(C_sq_all_mean_prior)
    
    C_sq = C_sq[:,:-1]
    p = [*predictors]
    plt.figure(2)
    colors = ['red', 'darkblue', 'black', 'green']
    for i in np.arange(C_sq.shape[1]):
        plt.plot(C_sq[:,i], label =p[i], color = colors[i])
        #plt.title('PFC')
    plt.vlines(ind_reward,ymin = 0, ymax = 70,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 70,linestyles= '--', color = 'pink', label = 'Choice')

    plt.legend()
    plt.ylabel('Beta Sq')
    plt.xlabel('Time (ms)')

def plotting_bayes():
    session = experiment_aligned_PFC[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
            
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()

    cpd, predictors, C, C_sq = regression_bayes(experiment_aligned_PFC,bayes_prior_PFC, bayes_posterior_PFC)
    C_sq = C_sq[:,:-1]
    cpd = cpd[:,:-1]

    p = [*predictors]
    plt.figure(2)
    colors = ['red', 'darkblue', 'black', 'green']
    plt.subplot(221)   
    for i in np.arange(C_sq.shape[1]):
        plt.plot(C_sq[:,i], label =p[i], color = colors[i])
    plt.vlines(ind_reward,ymin = 0, ymax = 70,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 70,linestyles= '--', color = 'pink', label = 'Choice')

   # plt.legend()
    plt.ylabel('Beta Squared')
    plt.xlabel('Time (ms)')
    plt.title('PFC')

    plt.subplot(222)   
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = colors[i])
    plt.vlines(ind_reward,ymin = 0, ymax = 0.1,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 0.1,linestyles= '--', color = 'pink', label = 'Choice')

   # plt.legend()
    plt.ylabel('Coefficient of Partial Determination')
    plt.xlabel('Time (ms)')
    plt.title('PFC')

    cpd, predictors, C_sq = regression_bayes(experiment_aligned_HP,bayes_prior_HP, bayes_posterior_HP)

    C_sq  = C_sq[:,:-1]
    cpd = cpd[:,:-1]
    p = [*predictors]
    plt.figure(2)
    colors = ['red', 'darkblue', 'black', 'green']
    plt.subplot(223)   
    for i in np.arange(C_sq.shape[1]):
        plt.plot(C_sq[:,i], label =p[i], color = colors[i])
    plt.vlines(ind_reward,ymin = 0, ymax = 70,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 70,linestyles= '--', color = 'pink', label = 'Choice')

   # plt.legend()
    plt.ylabel('Beta Squared')
    plt.xlabel('Time (ms)')
    plt.title('HP')


    plt.subplot(224)   
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = colors[i])
        #plt.title('PFC')
    plt.vlines(ind_reward,ymin = 0, ymax = 0.1,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 0.1,linestyles= '--', color = 'pink', label = 'Choice')

    plt.legend()
    plt.ylabel('Coefficient of Partial Determination')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.title('HP')

    