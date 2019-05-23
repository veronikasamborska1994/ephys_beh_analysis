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
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]
        # Unpack parameters.
        alpha, iTemp, k = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.

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

        #Variables.
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
        
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, iTemp, h, k = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.
        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
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
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha, iTemp, k = params  

    #Variables.
    Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

    for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
        nc = 1 - c # Not chosen action.
        
        # update action values simple RW
        Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
        Q_td[i+1,nc] = Q_td[i,nc]
                  
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

    return choice_probs, Q_td

def simulate_Q2(session, params):
    # Unpack trial events.
    choice_a_ind = []
    choice_b_ind = []
    
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha, iTemp, k = params  

    #Variables.
    Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.

    for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
        nc = 1 - c # Not chosen action.
        # update action values simple RW
        Q_td[i+1, c] = Q_td[i, c] + alpha*(o - Q_td[i, c])
        Q_td[i+1,nc] = 1-Q_td[i+1, c]
                  
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
    
    return choice_probs, Q_td

    
def simulate_Q3(session, params):

    # Unpack trial events.
    choice_a_ind = []
    choice_b_ind = []
    
    choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])
    n_trials = choices.shape[0]

    # Unpack parameters.
    alpha,beta, iTemp, h, k = params  
    
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
      
    
    for c,choice in enumerate(choices):
            if choice == 0:
                if choices[c] == choices[c-1]:
                    choice_a_ind.append(c-1)
            elif choice == 1:
                if choices[c] == choices[c-1]:
                    choice_b_ind.append(c-1)
                    
    Q_td[choice_a_ind,0] *= k 
    Q_td[choice_b_ind,1] *= k 
 
    # Evaluate choice probabilities
    choice_probs = array_softmax(Q_td, iTemp)    
    
    return choice_probs, Q_td

def simulate_Q4(session, params):
        # Unpack trial events.
        choice_a_ind = []
        choice_b_ind = []
        
        choices, outcomes = (session.trial_data['choices'], session.trial_data['outcomes'])

        n_trials = choices.shape[0]

        # Unpack parameters.
        alpha, iTemp, h, k = params  

        #Variables.
        Q_td = np.zeros([n_trials + 1, 2])  # Model free action values.
        
      
        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.
            nc = 1 - c # Not chosen action
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
        
        return choice_probs, Q_td
 
        
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
    
    sessions = experiment_aligned_HP + experiment_aligned_PFC
    
    fits_Q1 = fit_sessions(sessions, Q1())
    fits_Q2 = fit_sessions(sessions, Q2())
    fits_Q3 = fit_sessions(sessions, Q3())
    fits_Q4 = fit_sessions(sessions, Q4())

    
    BIC_Q1_list = []
    for t,l,p in zip(fits_Q1['n_trials'],fits_Q1['likelihood'],fits_Q1['params']):
        n_trials = t 
        likelihood = l
        n_params = len(p)
        
        BIC_Q1 =  BIC(n_trials,likelihood, n_params)
        BIC_Q1_list.append(BIC_Q1)
        
    BIC_Q2_list = []
    for t,l,p in zip(fits_Q2['n_trials'],fits_Q2['likelihood'],fits_Q2['params']):
        n_trials = t 
        likelihood = l
        n_params = len(p)
      
        BIC_Q2 =  BIC(n_trials,likelihood, n_params)
        BIC_Q2_list.append(BIC_Q2)
        
            
    BIC_Q3_list = []
    for t,l,p in zip(fits_Q3['n_trials'],fits_Q3['likelihood'],fits_Q3['params']):
        n_trials = t 
        likelihood = l
        n_params = len(p)
      
        BIC_Q3 =  BIC(n_trials,likelihood, n_params)
        BIC_Q3_list.append(BIC_Q3)
        
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
        
    h_Q3 = []  
    for p in fits_Q3['params']:
        h_Q3.append(p[-1])
        
    m_cross_term = np.mean(cross_term_list)
        
    BIC_Q1_sum = np.mean(BIC_Q1_list)
    BIC_Q2_sum = np.mean(BIC_Q2_list)
    BIC_Q3_sum = np.mean(BIC_Q3_list)
    BIC_Q4_sum = np.mean(BIC_Q4_list)
    
    return BIC_Q1_sum, BIC_Q2_sum, BIC_Q3_sum, BIC_Q4_sum, m_cross_term


def BIC(n_trials,likelihood, n_params):
   
    BIC = -2 * likelihood + n_params * np.log(n_trials)
    
    return BIC

def plotting(session):    
    
    session_fit_Q1 = fit_session(session = session, agent = Q1(), repeats = 5, brute_init = True, verbose = False)
    
    session_fit_Q2 = fit_session(session = session, agent = Q2(), repeats = 5, brute_init = True, verbose = False)
    
    session_fit_Q3 = fit_session(session = session, agent = Q3(), repeats = 5, brute_init = True, verbose = False)
    
    session_fit_Q4 = fit_session(session = session, agent = Q4(), repeats = 5, brute_init = True, verbose = False)


    params_Q1 = session_fit_Q1['params']
    params_Q2 = session_fit_Q2['params']
    params_Q3 = session_fit_Q3['params']
    params_Q4 = session_fit_Q4['params']

    choice_probs_Q1, Q_td_Q1 = simulate_Q1(session, params_Q1)
    choice_probs_Q2, Q_td_Q2 = simulate_Q2(session, params_Q2)
    choice_probs_Q3, Q_td_Q3 = simulate_Q3(session, params_Q3)
    choice_probs_Q4, Q_td_Q4 = simulate_Q4(session, params_Q4)

    fig = plt.figure(figsize=(6, 40))
    grid = plt.GridSpec(12, 3, hspace=6, wspace=1)
    fig.add_subplot(grid[0:2,:3]) 
    plt.plot(Q_td_Q1[:,0], label = 'Value A', color = 'green')
    plt.plot(Q_td_Q1[:,1], label = 'Value B', color = 'black')
    plt.plot(session.trial_data['choices'], '--', color = 'grey',linewidth = 0.5)
    plt.ylabel('Q')
    plt.title('Q_a(t) = Q_a(t-1) + alpha*(outcome -Q_a(t-1))\nQ_b(t) = Q_b(t-1)')
                 

    fig.add_subplot(grid[2:4,:3]) 
    plt.plot(Q_td_Q2[:,0], label = 'Value A', color = 'green')
    plt.plot(Q_td_Q2[:,1], label = 'Value B', color = 'black')
    plt.plot(session.trial_data['choices'], '--', color = 'grey',linewidth = 0.5)
    plt.ylabel('Q')
    plt.title('Q_a(t) = Q_a(t-1) + alpha*(outcome -Q_a(t-1))\nQ_b(t) = 1-Q_a')
          

    
    fig.add_subplot(grid[4:6,:3]) 
    plt.plot(Q_td_Q3[:,0], label = 'Value A',  color = 'green')
    plt.plot(Q_td_Q3[:,1], label = 'Value B', color = 'black')
    plt.plot(session.trial_data['choices'], '--', color = 'grey',linewidth = 0.5)
    plt.ylabel('Q') 
    plt.title('Q_td = (1-h)*Q_classic + h*Q_one_variable')



    fig.add_subplot(grid[6:8,:3]) 
    plt.plot(Q_td_Q4[:,0], label = 'Value A',  color = 'green')
    plt.plot(Q_td_Q4[:,1], label = 'Value B', color = 'black')
    plt.plot(session.trial_data['choices'], '--', color = 'grey', linewidth = 0.5)
    plt.ylabel('Q')
    plt.xlabel('Trial N')
    plt.title('Q_a(t) = (1-alpha)*Q(t-1)+alpha*outcome\nQ_b(t)= (1-alpha*abs(h))*Q_b(t-1)+alpha*h*outcome')


    plt.legend()
    
    dict_BIC = {'RW':BIC_Q1_sum, 'RW 1 Variable':BIC_Q2_sum,'Mix RW and 1 Variable':BIC_Q3_sum, 'RW cross-learning rates':BIC_Q4_sum}
    BICs = np.array(list(dict_BIC.values()))
    p = dict_BIC.keys()
    
    fig.add_subplot(grid[8:11,:2]) 
    plt.bar(np.arange(len(BICs)), BICs, color = 'pink')
    plt.xticks(np.arange(len(BICs)),p, rotation = 'vertical')
    plt.ylabel('Mean BIC score')
    fig.add_subplot(grid[8:11,2]) 
    plt.bar([0,1,2], [0,m_cross_term,0], color = 'grey')
    plt.xticks([0,1,2], ['','Cross-Term',''])

    
    
    

