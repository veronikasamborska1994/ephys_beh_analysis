#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:43:36 2019

@author: veronikasamborska
"""
import numpy as np
import sys
import regressions as re 
import scipy.optimize as op
import math 
from tqdm import tqdm

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number.

#1/(1 - exp(T(Pincorrect- indecision point))))
#  alpha is the indecision point



def array_softmax(T,P,alpha):
    
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
        switch = np.asarray(switch)
      
        n_trials = choices.shape[0]
        
        #Unpack parameters.
        
        #alpha, iTemp, sigma = params  
        sigma =  2.87943965e-05
        alpha = 0.498897255
        reward  = 0.75
        iTemp  =  1.70424019
        
        # Variables.
        Posterior_correct_incorrect = np.zeros([n_trials, 2])  # Array for correct and incorrect posteriors
        
        Prior_correct_incorrect = np.zeros([n_trials, 2])  # Array for correct and incorrect priors
        Prior_correct_incorrect[0,:] =  0.5 # First choices start  with 0.5 priors for both correct and incorrect choice 
        
        # Switch array --> ones when  choice  is the same; reverse such that 1s mean there was a switch between current and last trial
        switch_choice = switch-1
        switch_choice = switch_choice*(-1)
        outcomes = outcomes[1:]
        
        for i, (o, c) in enumerate(zip(outcomes, switch_choice)): # loop over trials.
            if o == 1:
                Prior_correct_incorrect[i, 0]
                
                # Decision is correct based on the probability of getting a reward weighted by the prior of the trial being correct
                #/Probability of getting a reward in correct prior; probability of getting no reward given the incorrect prior 
                Posterior_correct_incorrect[i, 0] = reward_prob_given_correct * Prior_correct_incorrect[i, 0]/((reward_prob_given_correct*Prior_correct_incorrect[i,0])+ ((1-reward)*Prior_correct_incorrect[i,1]))
                Posterior_correct_incorrect[i, 1] = 1-Posterior_correct_incorrect[i, 0]
           
            elif o == 0:
                 # Decision is correct based on the probability of getting no reward weighted by the prior of the trial being correct
                #/Probability of getting a reward in an correct prior; probability of getting no reward given the incorrect prior 
                Posterior_correct_incorrect[i, 0] = (1-reward)*Prior_correct_incorrect[i, 0]/(((reward)*Prior_correct_incorrect[i,0])+ ((1-reward)*Prior_correct_incorrect[i,1]))
                Posterior_correct_incorrect[i, 1] = 1-Posterior_correct_incorrect[i, 0]
           
            if c  == 1:
                # Prior that the decision is correct if a switch occured; 
                # Reversal P *  Posterior past choice  was correct + (1-Reversal P)* Posterior past choice was incorrect
                Prior_correct_incorrect[i+1, 0] = ((sigma)* Posterior_correct_incorrect[i, 0])  + ((1-sigma)*Posterior_correct_incorrect[i, 1])
                Prior_correct_incorrect[i+1, 1] =  1-Prior_correct_incorrect[i+1, 0]
                
                # Prior that the decision is correct if no  switch occured; 
                # Choice is correct with  probability (1-Reversal P) & given past choice being correct + (Reversal P)* Posterior past choice was incorrect
            
            elif c  == 0:                
                Prior_correct_incorrect[i+1, 0] = ((1-sigma)* Posterior_correct_incorrect[i, 0])  + ((sigma)*Posterior_correct_incorrect[i, 1])
                Prior_correct_incorrect[i+1, 1] =  1-Prior_correct_incorrect[i+1, 0]
        
        # Evaluate choice probabilities and likelihood.     
        Posterior_correct_incorrect = Posterior_correct_incorrect[:-1,:]
        
        choice_probs = array_softmax(iTemp,Posterior_correct_incorrect, alpha)
        
        stay  =  switch
        trial_log_likelihood_switch = switch_choice*protected_log(choice_probs[:,0])
        trial_log_likelihood_stay = stay*protected_log(choice_probs[:,1])
        #session_log_likelihood = np.sum(trial_log_likelihood_switch)+np.sum(trial_log_likelihood_stay)#/sum(stay))

        session_log_likelihood = (np.sum(trial_log_likelihood_switch)/np.sum(switch_choice))+(np.sum(trial_log_likelihood_stay)/sum(stay))
        
        return session_log_likelihood

def fit_session(session, agent, repeats = 5, brute_init = True, verbose = False):
    
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

session_fit_Q1 = fit_session(session = session, agent = model(), repeats = 2, brute_init = True, verbose = False)



#all_sessions = experiment_aligned_HP + experiment_aligned_PFC

#fits_Q1_HP = fit_sessions(experiment_aligned_HP, model())
#fits_Q1_PFC = fit_sessions(experiment_aligned_PFC, model())

#experiment_sim_Q1_HP,  =  simulate_Qtd_experiment(fits_Q1_HP, fits_Q4_HP, experiment_aligned_HP)  
#experiment_sim_Q1_PFC, =  simulate_Qtd_experiment(fits_Q1_PFC, fits_Q4_PFC, experiment_aligned_PFC)  
 