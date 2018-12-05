#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:11:58 2018

@author: veronikasamborska
"""

import regressions as re
import data_import as di
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict

session = di.Session('/Users/veronikasamborska/Desktop/data_3_tasks_ephys/m479/m479-2018-08-10-173911.txt')
choices = session.trial_data['choices']
forced_trials = session.trial_data['forced_trial']
non_forced_array = np.where(forced_trials == 0)[0]
choice_non_forced = choices[non_forced_array]
n_trials = len(choice_non_forced)

predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3, predictor_B_Task_1,\
 predictor_B_Task_2, predictor_B_Task_3, reward = re.predictors_pokes(session)

A = predictor_A_Task_1 + predictor_A_Task_2 + predictor_A_Task_3
B = predictor_B_Task_1 + predictor_B_Task_2 + predictor_B_Task_3
A_reward = A*reward
B_reward = B*reward
A_task_1_reward = predictor_A_Task_1* reward
A_task_2_reward = predictor_A_Task_2* reward
A_task_3_reward = predictor_A_Task_3* reward
B_task_1_reward = predictor_B_Task_1* reward
B_task_2_reward = predictor_B_Task_2* reward
B_task_3_reward = predictor_B_Task_3* reward 
 
previous_trial = []

for i,ii in enumerate(A): 
    if i > 0:
        if A[i-1] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    previous_trial.append(trial)
        

two_trials = []

for i,ii in enumerate(A): 
    if i > 0:
        if A[i-2] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    two_trials.append(trial)
 
    
three_trials = []

for i,ii in enumerate(A): 
    if i > 0:
        if A[i-3] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    three_trials.append(trial)

r_previous_trial = []  
for i,ii in enumerate(reward): 
    if i > 0:
        if reward[i-1] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    r_previous_trial.append(trial)
        

r_two_trials = []

for i,ii in enumerate(reward): 
    if i > 0:
        if reward[i-2] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    r_two_trials.append(trial)
 
    
r_three_trials = []

for i,ii in enumerate(reward): 
    if i > 0:
        if reward[i-3] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    r_three_trials.append(trial)
        
ra_previous_trial = []  
for i,ii in enumerate(A_reward): 
    if i > 0:
        if A_reward[i-1] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    ra_previous_trial.append(trial)
        

ra_two_trials = []

for i,ii in enumerate(A_reward): 
    if i > 0:
        if A_reward[i-2] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    ra_two_trials.append(trial)
 
    
ra_three_trials = []

for i,ii in enumerate(A_reward): 
    if i > 0:
        if A_reward[i-3] == 1:
            trial = 1
        else:
            trial = 0
    else:
        trial = 0
    ra_three_trials.append(trial)
    

predictors = OrderedDict([('previous_trial' , previous_trial),
                          ('two_trials' , two_trials),
                          ('three_trials' , three_trials),
                          ('r_one_trial', r_previous_trial),
                          ('r_two_trial', r_two_trials),
                          ('r_three_trial', r_three_trials),
                          ('ra_one_trial', ra_previous_trial),
                          ('ra_two_trial', ra_two_trials),
                          ('ra_three_trial', ra_three_trials)])

X = np.vstack(predictors.values()).T[:n_trials,:].astype(float)
ols = LogisticRegression()
model = ols.fit(X, A)
coef = ols.coef_
predictors = [3,2,1]
scatter(predictors,coef[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')