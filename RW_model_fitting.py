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
import utility as ut

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
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]
        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen action.

            # update action values simple RW
            Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
            Q_td[i+1,nc] = Q_td[i,nc]
           

        # Evaluate choice probabilities and likelihood. 
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

        self.name = 'Q1'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen action.

            # update action values simple RW
            Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
            Q_td[i+1,nc] = 1-Q_td[i+1, c]

        # Evaluate choice probabilities and likelihood. 
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

        self.name = 'Q1'
        self.param_names  = ['alpha', 'beta','iTemp', 'h']
        self.params       = [ 0.5   , 0.5,  5. , 0.5   ]  
        self.param_ranges = ['unit', 'unit' , 'pos', 'unit']
        self.n_params = 4


    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, beta, iTemp,h = params  

        #Variables.
        Q_td_standard = np.zeros([n_trials + 1, 2])  # Model free action values.
        Q_td_one_variable = np.zeros([n_trials + 1, 2]) 
        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
            nc = 1 - c # Not chosen action.
            # update action values simple RW
            Q_td_standard[i+1, c] = Q_td_standard[i, c] + alpha*(o - Q_td_standard[i, c])
            Q_td_standard[i+1,nc] = Q_td_standard[i,nc]
                 
            Q_td_one_variable[i+1, c] = Q_td_one_variable[i, c] + beta*(o - Q_td_one_variable[i, c])
            Q_td_one_variable[i+1,nc] = 1-Q_td_one_variable[i+1, c]
                             
        # Evaluate choice probabilities
        Q_td = (1-h)*Q_td_standard + h*Q_td_one_variable
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
                   'param_names': agent.param_names} 
   
    session_fit['params'] = trans_UC(fit['x'], agent.param_ranges)

    return session_fit



def simulate_Q1(session, params):

    # Unpack trial events.
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha, iTemp = params  

    #Variables.
    Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

    for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
        nc = 1 - c # Not chosen action.
        
        # update action values simple RW
        Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
        Q_td[i+1,nc] = Q_td[i,nc]
                  
    # Evaluate choice probabilities
    choice_probs = array_softmax(Q_td, iTemp)
    exp_average= ut.exp_mov_ave(choices,initValue = 0.5,tau = 8)
    return choice_probs, Q_td, exp_average
    

def simulate_Q2(session, params):

    # Unpack trial events.
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha, iTemp = params  

    #Variables.
    Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

    for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
        nc = 1 - c # Not chosen action.
        # update action values simple RW
        Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
        Q_td[i+1,nc] = 1-Q_td[i+1, c]
                  
    # Evaluate choice probabilities
    choice_probs = array_softmax(Q_td, iTemp)
    exp_average= ut.exp_mov_ave(choices,initValue = 0.5,tau = 8)
    
    return choice_probs, Q_td, exp_average

def simulate_Q3(session, params):

    # Unpack trial events.
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha,beta, iTemp, h = params  
    
    #Variables.
    Q_td_standard = np.zeros([n_trials + 1, 2])  # Model free action values.
    Q_td_one_variable = np.zeros([n_trials + 1, 2]) 
    for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
        nc = 1 - c # Not chosen action.
        # update action values simple RW
        Q_td_standard[i+1, c] = Q_td_standard[i, c] + alpha*(o - Q_td_standard[i, c])
        Q_td_standard[i+1,nc] = Q_td_standard[i,nc]
             
        Q_td_one_variable[i+1, c] = Q_td_one_variable[i, c] + beta*(o - Q_td_one_variable[i, c])
        Q_td_one_variable[i+1,nc] = 1-Q_td_one_variable[i+1, c]
                         
    # Evaluate choice probabilities
    Q_td = (1-h)*Q_td_standard + h*Q_td_one_variable
    choice_probs = array_softmax(Q_td, iTemp)
    exp_average= ut.exp_mov_ave(choices,initValue = 0.5,tau = 8)
    return choice_probs, Q_td,exp_average

session = experiment_aligned_PFC[0]
    
session_fit_Q1 = fit_session(session = session, agent = Q1(), repeats = 1000, brute_init = True, verbose = False)

session_fit_Q2 = fit_session(session = session, agent = Q2(), repeats = 1000, brute_init = True, verbose = False)

session_fit_Q3 = fit_session(session = session, agent = Q3(), repeats = 1000, brute_init = True, verbose = False)


params_Q1 = session_fit_Q1['params']
params_Q2 = session_fit_Q2['params']
params_Q3 = session_fit_Q3['params']

choice_probs_Q1, Q_td_Q1,exp_average_Q1 = simulate_Q1(session, params_Q1)
choice_probs_Q2, Q_td_Q2,exp_average_Q2 = simulate_Q2(session, params_Q2)
choice_probs_Q3, Q_td_Q3,exp_average_Q3 = simulate_Q3(session, params_Q3)
#session = experiment_aligned_PFC[0]

fig = plt.figure(figsize=(8, 25))
grid = plt.GridSpec(3, 1, hspace=0.7, wspace=0.4)
fig.add_subplot(grid[0]) 
plt.plot(Q_td_Q1[:,0], label = 'Value A')
plt.plot(Q_td_Q1[:,1], label = 'Value B')
#plt.plot(exp_average_Q1,'--')
#plt.plot(choice_probs_Q1[:,1])

fig.add_subplot(grid[1]) 
plt.plot(Q_td_Q2[:,0], label = 'Value A')
plt.plot(Q_td_Q2[:,1], label = 'Value B')

#plt.plot(exp_average_Q2,'--')
#plt.plot(choice_probs_Q2[:,1])

fig.add_subplot(grid[2]) 
plt.plot(Q_td_Q3[:,0], label = 'Value A')
plt.plot(Q_td_Q3[:,1], label = 'Value B')

plt.legend()

#plt.plot(exp_average_Q3,'--')
#plt.plot(choice_probs_Q3[:,1])
