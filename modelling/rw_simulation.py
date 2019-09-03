#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:31:11 2019

@author: veronikasamborska
"""
# =============================================================================
# Rescorla - Wagner simple simulation 
# =============================================================================

from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt


     
class exp_mov_ave:
    # Exponential moving average class.
    def __init__(self, tau, init_value=0):
        self.tau = tau
        self.init_value = init_value
        self.reset()

    def reset(self, init_value=None, tau=None):
        if tau:
            self.tau = tau
        if init_value:
            self.init_value = init_value
        self.value = self.init_value
        self._m = math.exp(-1./self.tau)
        self._i = 1 - self._m

    def update(self, sample):
        self.value = (self.value * self._m) + (self._i * sample)
        

def bandit_simulation(k_arms = 2, initial_Q_a = 0.5, step_size = 0.1, epsilon = 0.1, n_steps = 1000, n_runs = 1000, halfway_change = False,\
                      update_values_seperately = False, update_values_together = False, cross_learning = False):
    
    reward_history = np.zeros((n_steps,n_runs))
    mov_av = np.zeros((n_steps,n_runs))
    best_action_count = np.zeros((n_steps,n_runs))
    value_a = 0.8
    value_b = 0.2
    Q_a_estimates_a = np.zeros((n_steps,n_runs))
    Q_a_estimates_b = np.zeros((n_steps,n_runs))
    
    for run in tqdm(range(n_runs)):

        count_for_reversal = 100
        count  = False

        true_action_values = np.asarray([value_a,value_b]) # create a new 2-armed bandit test bed for every run
        best_action = true_action_values.argmax() # and identify the best action

        Q_a_estimates = np.full((k_arms),float(initial_Q_a)) 
        action_count = np.zeros(k_arms)
        correct_mov_ave = exp_mov_ave(tau = 8, init_value = 0.5)

        for step in range(n_steps):
            
            #Softmax policy adjustments
            P = np.exp(Q_a_estimates)/(np.exp(epsilon*Q_a_estimates[0])+ np.exp(epsilon*Q_a_estimates[1]))            
            action = P.argmax()
            action_not_chosen = P.argmin()
            
            action_count[action] += 1
            reward = np.random.normal(loc=true_action_values[action], scale = 0)
            if update_values_seperately == True:
                Q_a_estimates[action] = Q_a_estimates[action] + step_size*(reward - Q_a_estimates[action])
                Q_a_estimates_a[step,run] = Q_a_estimates[0]
                Q_a_estimates_b[step,run] = Q_a_estimates[1]
                
            elif update_values_together == True:
                Q_a_estimates[action] = Q_a_estimates[action] + step_size*(reward - Q_a_estimates[action])
                Q_a_estimates[action_not_chosen] = Q_a_estimates[action_not_chosen] + step_size*(1-reward - Q_a_estimates[action_not_chosen])
                Q_a_estimates_a[step,run] = Q_a_estimates[0]
                Q_a_estimates_b[step,run] = Q_a_estimates[1]
            elif cross_learning == True:
                p = 0.1
                Q_a_estimates[action] = Q_a_estimates[action] + step_size*(reward - Q_a_estimates[action])
                Q_a_estimates[action_not_chosen] = Q_a_estimates[action_not_chosen] + step_size*p*(1-reward - Q_a_estimates[action_not_chosen])

                Q_a_estimates_a[step,run] = Q_a_estimates[0]
                Q_a_estimates_b[step,run] = Q_a_estimates[1]
                
            
            reward_history[step,run] = reward # keep track of rewards (for plotting)
 
            if best_action == action:
                best_action_count[step,run] = 1 # keep track of when the true best action is chosen (for plotting)
                correct_mov_ave.update(1)
            else:
                correct_mov_ave.update(0)
    
            if halfway_change and correct_mov_ave.value > 0.75:
                count  = True
                if count == True:
                    count_for_reversal -=1
                    if count_for_reversal == 0:
                        if true_action_values[0] == value_b:
                            true_action_values =  np.asarray([value_a,value_b])
                            best_action = true_action_values.argmax()
                        elif true_action_values[0] == value_a: 
                            true_action_values =  np.asarray([value_b,value_a])
                            best_action = true_action_values.argmax()
                        correct_mov_ave.value = 1 - correct_mov_ave.value
                        count_for_reversal = 100
                           
            mov_av[step,run] = correct_mov_ave.value 

    reward_history = reward_history.mean(axis=1)
    best_action_count = best_action_count.mean(axis=1)
    mov_av = mov_av.mean(axis = 1)
    Q_a =  Q_a_estimates_a.mean(axis = 1)
    Q_b =  Q_a_estimates_b.mean(axis = 1)
    
    return reward_history, best_action_count, mov_av,Q_a,Q_b


def plotting():
    reward_history, best_action_count, mov_av,Q_a,Q_b = bandit_simulation(k_arms = 2, initial_Q_a = 0.5, step_size = 0.1, epsilon = 0.1, n_steps = 1000, n_runs = 1000, halfway_change = True,\
                                                                      update_values_seperately = True, update_values_together = False,cross_learning = False)

    fig = plt.figure(figsize=(8, 25))
    grid = plt.GridSpec(3, 1, hspace=0.7, wspace=0.4)
    fig.add_subplot(grid[0]) 
    plt.plot(Q_a, label  = 'Value A', color = 'green')
    plt.plot(Q_b, label  = 'Value B', color = 'grey')
    plt.title('Two Variables Indepedent')
    
    
    
    reward_history, best_action_count, mov_av,Q_a,Q_b = bandit_simulation(k_arms = 2, initial_Q_a = 0.5, step_size = 0.1, epsilon = 0.1, n_steps = 1000, n_runs = 1000, halfway_change = True,\
                                                                         update_values_seperately = False,update_values_together = True, cross_learning = False)
    fig.add_subplot(grid[1]) 
    plt.plot(Q_a, label  = 'Value A', color = 'green')
    plt.plot(Q_b, label  = 'Value B', color = 'grey')
    plt.title('One variable')
    
    
    reward_history, best_action_count, mov_av,Q_a,Q_b = bandit_simulation(k_arms = 2, initial_Q_a = 0.5, step_size = 0.1, epsilon = 0.1, n_steps = 1000, n_runs = 1000, halfway_change = True,\
                                                                         update_values_seperately = False,update_values_together = False, cross_learning = True)
    fig.add_subplot(grid[2]) 
    plt.plot(Q_a, label  = 'Value A', color = 'green')
    plt.plot(Q_b, label  = 'Value B', color = 'grey')
    plt.title('Cross - Learning') 