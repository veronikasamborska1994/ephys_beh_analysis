#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:47:21 2019

@author: veronikasamborska
"""

import numpy as np
from numba import jit
import sys
import math
import scipy.optimize as op
import matplotlib.pyplot as plt
from tqdm import tqdm
import regressions as re
from collections import OrderedDict
from sklearn.linear_model import LinearRegression


def run():
    fits_Q1_HP = fit_sessions(experiment_aligned_HP, Q1())
    fits_Q1_PFC = fit_sessions(experiment_aligned_PFC, Q1())
    
    fits_Q4_HP = fit_sessions(experiment_aligned_HP, Q4())
    fits_Q4_PFC = fit_sessions(experiment_aligned_PFC, Q4())
    
    #fits_Q1 = fit_sessions(all_sessions, Q1())
    #fits_Q4 = fit_sessions(all_sessions, Q4())        
    experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP ,experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP =  simulate_Qtd_experiment(fits_Q1_HP, fits_Q4_HP, experiment_aligned_HP)  
    experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC =  simulate_Qtd_experiment(fits_Q1_PFC, fits_Q4_PFC, experiment_aligned_PFC)  
        

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number.

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------

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


def softmax(Q,T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[QT > log_max_float] = log_max_float # Protection against overflow in exponential.    
    expQT = np.exp(QT)
    return expQT/expQT.sum()


def array_softmax(Q,T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    TdQ = -T*(Q[:,0]-Q[:,1])
    TdQ[TdQ > log_max_float] = log_max_float # Protection against overflow in exponential.    
    P[:,0] = 1./(1. + np.exp(TdQ))
    P[:,1] = 1. - P[:,0]
    return P

def protected_log(x):
    
    'Return log of x protected against giving -inf for very small values of x.'
    
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

# -------------------------------------------------------------------------------------
# Agents
# -------------------------------------------------------------------------------------
    
class Q1():
    'A Q1 (simple RW) in which action values are updated only after a respective action is taken'
    
    def __init__(self):
        
        self.name = 'Q1'
        self.param_names  = ['alpha', 'iTemp', 'k']
        self.params       = [ 0.5   ,  5. , 0.1 ]  
        self.param_ranges = ['unit' ,'pos', 'unit' ]
        self.n_params = 3

    @jit
    def session_likelihood(self, session, params, return_Qs = False):
        choice_a_ind = []
        choice_b_ind = []
        
        # Unpack trial events.
        choices, outcomes,task = (session.trial_data['choices'], session.trial_data['outcomes'], session.trial_data['task'])
        task = session.trial_data['task']
        
        #forced_trials = session.trial_data['forced_trial']
        #non_forced_array = np.where(forced_trials == 0)[0]
        #task_non_forced = task[non_forced_array]
        #choices = choices[non_forced_array]
        #outcomes = outcomes[non_forced_array]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]  
        
        ind_task_2 = len(task_1)
        ind_task_3 = len(task_1)+len(task_2)

        n_trials = choices.shape[0]
        
        # Unpack parameters.
        alpha, iTemp, k = params  

        # Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
            if i == ind_task_2 or i == ind_task_3: 
                nc = 1 - c # Not chosen action.
                # update action values simple RW
                Q_td[i+1, c] = 0
                Q_td[i+1,nc] = 0 
            else:
                nc = 1 - c # Not chosen action.
                # update action values simple RW
                Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
                Q_td[i+1,nc] = Q_td[i,nc]
           

        # Evaluate choice probabilities and likelihood.     
        choice_probs = array_softmax(Q_td, iTemp)
        for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
                    
        Q_td[choice_a_ind,0] *= k 
        Q_td[choice_b_ind,1] *= k 
        choice_probs = array_softmax(Q_td, iTemp)

        trial_log_likelihood = protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        
        if return_Qs:
            return Q_td
        else:
            return session_log_likelihood

class Q2():
    ''' A Q2 (one variable RW) in which outcomes on A update both A and B action values simultaneously'''
    
    def __init__(self):

        self.name = 'Q2'
        self.param_names  = ['alpha', 'iTemp', 'k']
        self.params       = [ 0.5   ,  5. , 0.1 ]  
        self.param_ranges = ['unit' ,'pos', 'unit']
        self.n_params = 3

    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choice_a_ind = []
        choice_b_ind = []
        
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, iTemp, k  = params  

        # Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen action.

            # update action values simple RW
            Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
            Q_td[i+1,nc] = 1-Q_td[i+1, c]

        # Evaluate choice probabilities and likelihood. 
        
        for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
        Q_td[choice_a_ind,0] *= k 
        Q_td[choice_b_ind,1] *= k 
        
        choice_probs = array_softmax(Q_td, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        if return_Qs:
            return Q_td
        else:
            return session_log_likelihood

class Q3():
    ''' A Q3 (cross-learning RW) in which outcomes on A update A and include a parameter which tells how much B action values are learnt from A'''

    def __init__(self):

        self.name = 'Q3'
        self.param_names  = ['alpha','beta','iTemp', 'h', 'k']
        self.params       = [ 0.5 , 0.5, 5. , 0.5,  0.5]  
        self.param_ranges = ['unit','unit', 'pos', 'unit', 'unit']
        self.n_params = 5


    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choice_a_ind = []
        choice_b_ind = []
        
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, beta, iTemp,h, k = params  

        #Variables.
        Q_td_standard = np.zeros([n_trials + 1, 2])  # Model free action values.
        Q_td_one_variable = np.zeros([n_trials + 1, 2]) 
        Q_td = np.zeros([n_trials + 1, 2]) 

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
            nc = 1 - c # Not chosen action.
            # update action values simple RW
            Q_td_standard[i+1, c] = Q_td_standard[i, c] + alpha*(o - Q_td_standard[i, c])
            Q_td_standard[i+1,nc] = Q_td_standard[i,nc]
                 
            Q_td_one_variable[i+1, c] = Q_td_one_variable[i, c] + beta*(o - Q_td_one_variable[i, c])
            Q_td_one_variable[i+1,nc] = 1-Q_td_one_variable[i+1, c]
            
            Q_td[i+1, c] = (1-abs(h))* Q_td_standard[i+1, c] + h*Q_td_one_variable[i+1, c]                
            Q_td[i+1, nc] = (1-abs(h))* Q_td_standard[i+1, nc] + h*Q_td_one_variable[i+1, nc]                
      
        # Evaluate choice probabilities
        for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
        Q_td[choice_a_ind,0] *= k 
        Q_td[choice_b_ind,1] *= k 
        
        choice_probs = array_softmax(Q_td, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        if return_Qs:
            return Q_td
        else:
            return session_log_likelihood         
                

class Q4():
    ''' A Q4 (cross-learning RW) in which outcomes on A update A and include a parameter which tells how much B action values are learnt from A'''

    def __init__(self):

        self.name = 'Q4'
        self.param_names  = ['alpha','iTemp', 'h', 'k']
        self.params       = [ 0.5   ,  5. , 0.5 , 0.1 ]  
        self.param_ranges = ['unit', 'pos', 'cross_term', 'unit']
        self.n_params = 4


    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choice_a_ind = []
        choice_b_ind = []
        
        # Unpack trial events.
        choices, outcomes,task = (session.trial_data['choices'], session.trial_data['outcomes'], session.trial_data['task'])
        task = session.trial_data['task']
        
        #forced_trials = session.trial_data['forced_trial']
        #non_forced_array = np.where(forced_trials == 0)[0]
        #task_non_forced = task[non_forced_array]
        #choices = choices[non_forced_array]
        #outcomes = outcomes[non_forced_array]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]  
        ind_task_2 = len(task_1)
        ind_task_3 = len(task_1)+len(task_2)
        
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, iTemp, h, k = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.
        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
            if i == ind_task_2 or i == ind_task_3:             
                nc = 1 - c # Not chosen action.
                # update action values simple RW
                Q_td[i+1, c] = 0
                Q_td[i+1,nc] = 0
                
            else:
                nc = 1 - c # Not chosen action.
                # update action values simple RW
                Q_td[i+1, c] = (1-alpha)*Q_td[i, c]+alpha*o
                Q_td[i+1, nc] = (1-alpha*abs(h))*Q_td[i, nc]+alpha*h*o
                                    
        # Evaluate choice probabilities
        for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
                    
        Q_td[choice_a_ind,0] *= k 
        Q_td[choice_b_ind,1] *= k 
        
        choice_probs = array_softmax(Q_td, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        
        if return_Qs:
            return Q_td
        else:
            return session_log_likelihood         
                
                                
# -------------------------------------------------------------------------------------
#  Model fitting
# -------------------------------------------------------------------------------------
                
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



def simulate_Q1(session, params):
    # Unpack trial events.
    choice_a_ind = []
    choice_b_ind = []
    # Unpack trial events.
    choices, outcomes,task = (session.trial_data['choices'], session.trial_data['outcomes'], session.trial_data['task'])
    task = session.trial_data['task']
    #forced_trials = session.trial_data['forced_trial']
    #non_forced_array = np.where(forced_trials == 0)[0]
    #task_non_forced = task[non_forced_array]
    #choices = choices[non_forced_array]
    #outcomes = outcomes[non_forced_array]
    task_1 = np.where(task == 1)[0]
    task_2 = np.where(task == 2)[0]  
    ind_task_2 = len(task_1)
    ind_task_3 = len(task_1)+len(task_2)
    
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha, iTemp, k = params  

    #Variables.
    Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.
    Q_chosen = [0]
    for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
        if i == ind_task_2 or i == ind_task_3: 
            nc = 1 - c # Not chosen action.
            # update action values simple RW
            Q_td[i+1, c] = 0
            Q_td[i+1,nc] = 0 
            Q_chosen.append(Q_td[i+1,c])

        else:           
            nc = 1 - c # Not chosen action.           
            # update action values simple RW
            Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
            Q_td[i+1,nc] = Q_td[i,nc]
            Q_chosen.append(Q_td[i+1,c])

                  
    # Evaluate choice probabilities
    for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
                    
    Q_td[choice_a_ind,0] *= k 
    Q_td[choice_b_ind,1] *= k 
    
    choice_probs = array_softmax(Q_td, iTemp)

    return choice_probs, Q_td, Q_chosen


def simulate_Q4(session, params):
        # Unpack trial events.
        choice_a_ind = []
        choice_b_ind = []
        
        # Unpack trial events.
        choices, outcomes,task = (session.trial_data['choices'], session.trial_data['outcomes'], session.trial_data['task'])
        #forced_trials = session.trial_data['forced_trial']
        #non_forced_array = np.where(forced_trials == 0)[0]
        #task_non_forced = task[non_forced_array]
        #choices = choices[non_forced_array]
        #outcomes = outcomes[non_forced_array]
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]  
        
        ind_task_2 = len(task_1)
        ind_task_3 = len(task_1)+len(task_2)
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, iTemp, h, k = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.    
        Q_chosen = [0]
        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
            if i == ind_task_2 or i == ind_task_3: 
                nc = 1 - c # Not chosen action.
                # update action values simple RW
                Q_td[i+1, c] = 0
                Q_td[i+1,nc] = 0
                Q_chosen.append(Q_td[i+1, c])

            else:
                nc = 1 - c # Not chosen action
                # update action values simple RW
                Q_td[i+1, c] = (1-alpha)*Q_td[i, c]+alpha*o
                Q_td[i+1, nc] = (1-alpha*abs(h))*Q_td[i, nc]+alpha*h*o
                Q_chosen.append(Q_td[i+1, c])
                                            
        # Evaluate choice probabilities
        for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
                    
        Q_td[choice_a_ind,0] *= k 
        Q_td[choice_b_ind,1] *= k 
        
        choice_probs = array_softmax(Q_td, iTemp)
        
        return choice_probs, Q_td, Q_chosen
 
        
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



def BIC_all_sessions():
    
    all_sessions = experiment_aligned_HP + experiment_aligned_PFC
    
    fits_Q1_HP = fit_sessions(experiment_aligned_HP, Q1())
    fits_Q1_PFC = fit_sessions(experiment_aligned_PFC, Q1())

    fits_Q4_HP = fit_sessions(experiment_aligned_HP, Q4())
    fits_Q4_PFC = fit_sessions(experiment_aligned_PFC, Q4())

    fits_Q1 = fit_sessions(all_sessions, Q1())
    fits_Q4 = fit_sessions(all_sessions, Q4())

    BIC_Q1_list = []
    for t,l,p in zip(fits_Q1['n_trials'],fits_Q1['likelihood'],fits_Q1['params']):
        n_trials = t 
        likelihood = l
        n_params = len(p)
        
        BIC_Q1 =  BIC(n_trials,likelihood, n_params)
        BIC_Q1_list.append(BIC_Q1)

        
    BIC_Q4_list = []
    for t,l,p in zip(fits_Q4['n_trials'],fits_Q4['likelihood'],fits_Q4['params']):
        n_trials = t 
        likelihood = l
        n_params = len(p)
      
        BIC_Q4 =  BIC(n_trials,likelihood, n_params)
        BIC_Q4_list.append(BIC_Q4)
        
    cross_term_list = []  
    for p in fits_Q4['params']:
        cross_term_list.append(p[-2])
        
        
    m_cross_term = np.mean(cross_term_list)
        
    BIC_Q1_sum = np.mean(BIC_Q1_list)
   
    BIC_Q4_sum = np.mean(BIC_Q4_list)
    
    return BIC_Q1_sum, BIC_Q4_sum, m_cross_term


def BIC(n_trials,likelihood, n_params):
   
    BIC = -2 * likelihood + n_params * np.log(n_trials)
    
    return BIC

def plotting(session):    
    
    session_fit_Q1 = fit_session(session = session, agent = Q1(), repeats = 5, brute_init = True, verbose = False)
    
    session_fit_Q4 = fit_session(session = session, agent = Q4(), repeats = 5, brute_init = True, verbose = False)
    
    task = session.trial_data['task']

    #choices = session.trial_data['choices']
    #forced_trials = session.trial_data['forced_trial']
    #non_forced_array = np.where(forced_trials == 0)[0]
    #task_non_forced = task[non_forced_array]
    #choices = choices[non_forced_array]
    
    task_1 = np.where(task == 1)[0]
    task_2 = np.where(task == 2)[0]  
    ind_task_2 = len(task_1)
    ind_task_3 = len(task_1)+len(task_2)

    params_Q1 = session_fit_Q1['params']
   
    params_Q4 = session_fit_Q4['params']

    choice_probs_Q1, Q_td_Q1, Q_chosen_1 = simulate_Q1(session, params_Q1)
    
    choice_probs_Q4, Q_td_Q4 ,Q_chosen_4 = simulate_Q4(session, params_Q4)

    fig = plt.figure(figsize=(6, 40))
    grid = plt.GridSpec(8, 3, hspace=6, wspace=1)
    fig.add_subplot(grid[0:2,:3]) 
    plt.plot(Q_td_Q1[:,0], label = 'Value A', color = 'green')
    plt.plot(Q_td_Q1[:,1], label = 'Value B', color = 'black')
    plt.plot(choices, '--', color = 'grey',linewidth = 0.5)
    plt.vlines(ind_task_2, 0, 1, color = 'red')
    plt.vlines(ind_task_3, 0, 1, color = 'red')

    plt.ylabel('Q')
    plt.title('Q_a(t) = Q_a(t-1) + alpha*(outcome -Q_a(t-1))\nQ_b(t) = Q_b(t-1)')


    fig.add_subplot(grid[2:4,:3]) 
    plt.plot(Q_td_Q4[:,0], label = 'Value A',  color = 'green')
    plt.plot(Q_td_Q4[:,1], label = 'Value B', color = 'black')
    plt.plot(choices, '--', color = 'grey', linewidth = 0.5)
    plt.vlines(ind_task_2, -1, 1, color = 'red')
    plt.vlines(ind_task_3, -1, 1, color = 'red')

    plt.ylabel('Q')
    plt.xlabel('Trial N')
    plt.title('Q_a(t) = (1-alpha)*Q(t-1)+alpha*outcome\nQ_b(t)= (1-alpha*abs(h))*Q_b(t-1)+alpha*h*outcome')


    plt.legend()
    
    dict_BIC = {'RW':BIC_Q1_sum, 'RW cross-learning rates':BIC_Q4_sum}
    BICs = np.array(list(dict_BIC.values()))
    p = dict_BIC.keys()
    
    fig.add_subplot(grid[4:7,:2]) 
    plt.bar(np.arange(len(BICs)), BICs, color = 'pink')
    plt.xticks(np.arange(len(BICs)),p, rotation = 'vertical')
    plt.ylabel('Mean BIC score')
    fig.add_subplot(grid[4:7,2]) 
    plt.bar([0,1,2], [0,m_cross_term,0], color = 'grey')
    plt.xticks([0,1,2], ['','Cross-Term',''])

    
    
def simulate_Qtd_experiment(fits_Q1, fits_Q4, experiment):
    
    experiment_sim_Q1 = []
    experiment_sim_Q4 = []
    experiment_sim_Q1_value_a = []
    experiment_sim_Q1_value_b = []
    experiment_sim_Q4_values = []

    for s,session in enumerate(experiment):
        
        params_Q1 = fits_Q1['params'][s]
        params_Q4 = fits_Q4['params'][s]

        choice_probs_Q1, Q_td_Q1, Q_chosen_Q1 = simulate_Q1(session, params_Q1)
        choice_probs_Q4, Q_td_Q4, Q_chosen_Q4 = simulate_Q4(session, params_Q4)
        
        experiment_sim_Q1.append(Q_chosen_Q1)
        experiment_sim_Q4.append(Q_chosen_Q4)
        experiment_sim_Q1_value_a.append(Q_td_Q1[:,0])
        experiment_sim_Q1_value_b.append(Q_td_Q1[:,1])

        experiment_sim_Q4_values.append(Q_td_Q4[:,0])

        
    return experiment_sim_Q1, experiment_sim_Q4, experiment_sim_Q1_value_a, experiment_sim_Q1_value_b, experiment_sim_Q4_values

        



def regression_on_Q_values(experiment,experiment_sim_Q1_value_a,experiment_sim_Q1_value_b, experiment_sim_Q4_values):
    
    C_1 = []
    cpd = []
    C_sq = []

    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            Q_1_a = np.asarray(experiment_sim_Q1_value_a[s])
            Q_1_b = np.asarray(experiment_sim_Q1_value_b[s])

            Q_4 = np.asarray(experiment_sim_Q4_values[s])
            Q_1_a = Q_1_a[:-1]
            Q_1_b = Q_1_b[:-1]
            Q_4 = Q_4[:-1]

            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            
            Q_1_a = Q_1_a[non_forced_array]
            Q_1_b = Q_1_b[non_forced_array]

            Q_4 = Q_4[non_forced_array]
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:,:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies

            ones = np.ones(len(choices))
            
            predictors = OrderedDict([('Q1_a', Q_1_a),
                                      ('Q1_b', Q_1_b),
                                      ('Q_4', Q_4), 
                                      ('choice', choices),
                                      ('reward', outcomes),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_1.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            C_sq.append((ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)**2))

            cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))


    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0) # Population CPD is mean over neurons.
    C_sq = np.nanmean(np.concatenate(C_sq,0), axis = 0) # Population CPD is mean over neurons.

    return cpd, predictors,C_sq, C_1


def regression_on_Q_values_split_by_task(experiment,experiment_sim_Q1_value_a,experiment_sim_Q1_value_b, experiment_sim_Q4_values):
    
    C_1 = []
    C_1_sq = []

    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]    
            
            
            Q_1_a = np.asarray(experiment_sim_Q1_value_a[s])
            Q_1_b = np.asarray(experiment_sim_Q1_value_b[s])

            Q_4 = np.asarray(experiment_sim_Q4_values[s])
            Q_1_a = Q_1_a[:-1]
            Q_1_b = Q_1_b[:-1]
            Q_4 = Q_4[:-1]

            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            
            Q_1_a = Q_1_a[non_forced_array]
            Q_1_b = Q_1_b[non_forced_array]

            Q_4 = Q_4[non_forced_array]
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:,:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies
            
            ones = np.ones(len(choices))
            Q_1_a = Q_1_a[:len(task_1)]
            Q_1_b = Q_1_b[:len(task_1)]
            Q_4 = Q_4[:len(task_1)]
            choices = choices[:len(task_1)]
            outcomes = outcomes[:len(task_1)]
            ones = ones[:len(task_1)]
            aligned_spikes = aligned_spikes[:len(task_1)]

            predictors = OrderedDict([('Q1_a', Q_1_a),
                                      ('Q1_b', Q_1_b),
                                      ('Q_4', Q_4), 
                                      ('choice', choices),
                                      ('reward', outcomes)])
                                      #('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X,y)
            C_1.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            C_1_sq.append((ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)**2))

        
    C_1 = np.concatenate(C_1, axis = 0) # Population CPD is mean over neurons.
    C_1_sq = np.concatenate(C_1_sq, axis = 0) # Population CPD is mean over neurons.

    C_2 = []
    C_2_sq = []
    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]    
            
            Q_1_a = np.asarray(experiment_sim_Q1_value_a[s])
            Q_1_b = np.asarray(experiment_sim_Q1_value_b[s])

            Q_4 = np.asarray(experiment_sim_Q4_values[s])
            Q_1_a = Q_1_a[:-1]
            Q_1_b = Q_1_b[:-1]
            Q_4 = Q_4[:-1]

            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            
            Q_1_a = Q_1_a[non_forced_array]
            Q_1_b = Q_1_b[non_forced_array]

            Q_4 = Q_4[non_forced_array]
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:,:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies

            ones = np.ones(len(choices))
            
            Q_1_a = Q_1_a[len(task_1):len(task_1)+len(task_2)]
            Q_1_b = Q_1_b[len(task_1):len(task_1)+len(task_2)]
            Q_4 = Q_4[len(task_1):len(task_1)+len(task_2)]
            choices = choices[len(task_1):len(task_1)+len(task_2)]
            outcomes = outcomes[len(task_1):len(task_1)+len(task_2)]
            ones = ones[len(task_1):len(task_1)+len(task_2)]
            aligned_spikes = aligned_spikes[len(task_1):len(task_1)+len(task_2)]
            
            predictors = OrderedDict([('Q1_a', Q_1_a),
                                      ('Q1_b', Q_1_b),
                                      ('Q_4', Q_4), 
                                      ('choice', choices),
                                      ('reward', outcomes)])
                                      #('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X,y)
            C_2.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            C_2_sq.append((ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)**2))

    C_2 = np.concatenate(C_2, axis = 0) # Population CPD is mean over neurons.
    C_2_sq = np.concatenate(C_2_sq, axis = 0) # Population CPD is mean over neurons.


    C_3 = []
    C_3_sq = []
    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]    
            
            Q_1_a = np.asarray(experiment_sim_Q1_value_a[s])
            Q_1_b = np.asarray(experiment_sim_Q1_value_b[s])

            Q_4 = np.asarray(experiment_sim_Q4_values[s])
            Q_1_a = Q_1_a[:-1]
            Q_1_b = Q_1_b[:-1]
            Q_4 = Q_4[:-1]

            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            
            Q_1_a = Q_1_a[non_forced_array]
            Q_1_b = Q_1_b[non_forced_array]

            Q_4 = Q_4[non_forced_array]
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:,:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies

            ones = np.ones(len(choices))
            
  
            Q_1_a = Q_1_a[len(task_1)+len(task_2):]
            Q_1_b = Q_1_b[len(task_1)+len(task_2):]
            Q_4 = Q_4[len(task_1)+len(task_2):]
            choices = choices[len(task_1)+len(task_2):]
            outcomes = outcomes[len(task_1)+len(task_2):]
            ones = ones[len(task_1)+len(task_2):]
            aligned_spikes = aligned_spikes[len(task_1)+len(task_2):]
            
            predictors = OrderedDict([('Q1_a', Q_1_a),
                                      ('Q1_b', Q_1_b),
                                      ('Q_4', Q_4), 
                                      ('choice', choices),
                                      ('reward', outcomes)])
                                      #('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(copy_X = True,fit_intercept= False)
            ols.fit(X,y)
            C_3.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            C_3_sq.append((ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)**2))

    C_3 = np.concatenate(C_3, axis = 0) # Population CPD is mean over neurons.
    C_3_sq = np.concatenate(C_3_sq, axis = 0) # Population CPD is mean over neurons.

    return C_1, C_2, C_3

def plotting_coef():
    
    C_1_HP, C_2_HP, C_3_HP = regression_on_Q_values_split_by_task(experiment_aligned_HP,experiment_sim_Q1_value_a_HP,experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP)

    C_1_PFC, C_2_PFC, C_3_PFC = regression_on_Q_values_split_by_task(experiment_aligned_PFC,experiment_sim_Q1_value_a_PFC,experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC)
    
    C_1_HP = np.mean(C_1_HP, axis = 1)
    C_2_HP = np.mean(C_2_HP, axis = 1)
    C_3_HP = np.mean(C_3_HP, axis = 1)
    
    
    task_1 = C_1_HP[:,0].flatten()
    task_2 = C_2_HP[:,0].flatten()
    task_3 = C_3_HP[:,0].flatten()
    
    argmax_neuron = np.argsort(-task_1)
    task_2_by_1 = task_2[argmax_neuron]
    task_1 = task_1[argmax_neuron]
    task_3_by_1 = task_3[argmax_neuron]
    
    y = np.arange(len(task_1))
    plt.figure(3)
    plt.scatter(y,task_2_by_1,s = 2, color = 'red', label = 'Task 2 sorted by Task 1')
    plt.plot(y,task_2_by_1, color = 'grey', label = 'Task 2 sorted by Task 1')
    
    #plt.scatter(y,task_3_by_1,s = 2,color = 'slateblue', label = 'Task 3 sorted by Task 1')
    
    #plt.scatter(y,task_1,s = 2,color = 'black', label = 'Task 1 sorted')
    
    plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
    
    plt.legend()
    plt.title('HP Q1')
    
    C_1_PFC = np.mean(C_1_PFC, axis = 1)
    C_2_PFC = np.mean(C_2_PFC, axis = 1)
    C_3_PFC = np.mean(C_3_PFC, axis = 1)
    
    
    task_1_PFC = C_1_PFC[:,0].flatten()
    task_2_PFC = C_2_PFC[:,0].flatten()
    task_3_PFC = C_3_PFC[:,0].flatten()
    
    argmax_neuron = np.argsort(-task_1_PFC)
    task_2_by_1 = task_2_PFC[argmax_neuron]
    task_1 = task_1_PFC[argmax_neuron]
    task_3_by_1 = task_3_PFC[argmax_neuron]
    
    y = np.arange(len(task_1))
    plt.figure(4)
    plt.scatter(y,task_2_by_1,s = 2, color = 'red', label = 'Task 2 sorted by Task 1')
    plt.plot(y,task_2_by_1, color = 'grey', label = 'Task 2 sorted by Task 1')
    
    #plt.scatter(y,task_3_by_1,s = 2,color = 'slateblue', label = 'Task 3 sorted by Task 1')
    
    #plt.scatter(y,task_1,s = 2,color = 'black', label = 'Task 1 sorted')
    
    plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
    
    plt.legend()
    plt.title('PFC Q1')

def plotting_cpd(): 
    session = experiment_aligned_PFC[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
            
    ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()

    cpd, predictors, C_sq= regression_on_Q_values(experiment_aligned_PFC, experiment_sim_Q1_value_a_PFC,experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC)
    cpd = cpd[:,:-1]
    p = [*predictors]
    plt.figure(1)
    colors = ['red', 'darkblue', 'black', 'green', 'pink']
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = colors[i])
    plt.title('PFC')

    plt.vlines(ind_reward,ymin = 0, ymax = 0.1,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 0.1,linestyles= '--', color = 'pink', label = 'Choice')

    plt.legend()
    plt.ylabel('Coefficient of Partial Determination')
    plt.xlabel('Time (ms)')
