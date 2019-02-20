#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:11:58 2018

@author: veronikasamborska
"""

import regressions as re
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np

all_sessions = HP + PFC

# Empty lists for predictors  

reg_m478_p = []
reg_m479_p = []
reg_m480_p = []
reg_m481_p = []
reg_m483_p = []
reg_m484_p = []
reg_m486_p = []

# Empty lists for dependent variable 
reg_m478_dv = []
reg_m479_dv = []
reg_m480_dv = []
reg_m481_dv = []
reg_m483_dv = []
reg_m484_dv = []
reg_m486_dv = []


reg_m478_fr = []
reg_m479_fr = []
reg_m480_fr = []
reg_m481_fr = []
reg_m483_fr = []
reg_m484_fr = []
reg_m486_fr = []

for s,session in enumerate(all_sessions):
    aligned_spikes= session.aligned_rates[:]

    subj_ID = str(session.subject_ID)
    choices = session.trial_data['choices']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    choice_non_forced = choices[non_forced_array]
    n_trials = len(choice_non_forced)
    
    n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
    t_out = session.t_out
    initiate_choice_t = session.target_times #Initiation and Choice Times
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
    ind_after_choice = ind_choice + 7 # 1 sec after choice
    spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
    mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =2)
    
    predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
    predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
    predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
    # Check if a choice happened before the end of the session
    if len(predictor_A_Task_1) != len(choice_non_forced):
        predictor_A_Task_1 = predictor_A_Task_1[:len(choice_non_forced)]
        predictor_A_Task_2 = predictor_A_Task_2[:len(choice_non_forced)]
        predictor_A_Task_3 = predictor_A_Task_3[:len(choice_non_forced)]
        predictor_B_Task_1 = predictor_B_Task_1[:len(choice_non_forced)]
        predictor_B_Task_2 = predictor_B_Task_2[:len(choice_non_forced)]
        predictor_B_Task_3 = predictor_B_Task_3[:len(choice_non_forced)]
        reward = reward[:len(choice_non_forced)]
        
    A = predictor_A_Task_1 + predictor_A_Task_2 + predictor_A_Task_3
    if len(A) != len(mean_spikes_around_choice):
        mean_spikes_around_choice = mean_spikes_around_choice[:len(A)]
        
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
    y = mean_spikes_around_choice.reshape([len(mean_spikes_around_choice),-1])
   
    if subj_ID == '478':
        reg_m478_p.append(X)
        reg_m478_dv.append(A)
        reg_m478_fr.append(y)
    elif subj_ID == '479':
        reg_m479_p.append(X)
        reg_m479_dv.append(A)
        reg_m479_fr.append(y)
    elif subj_ID == '480':
        reg_m480_p.append(X)
        reg_m480_dv.append(A)
        reg_m480_fr.append(y)
    elif subj_ID == '481':
        reg_m481_p.append(X)
        reg_m481_dv.append(A)
        reg_m481_fr.append(y)
    elif subj_ID == '483':
        reg_m483_p.append(X)
        reg_m483_dv.append(A)
        reg_m483_fr.append(y)
    elif subj_ID == '484':
        reg_m484_p.append(X)
        reg_m484_dv.append(A) 
        reg_m484_fr.append(y)
    elif subj_ID == '486':
        reg_m486_p.append(X)
        reg_m486_dv.append(A)
        reg_m486_fr.append(y)


flat_reg_m486_p = [y for x in reg_m486_p for y in x]
flat_reg_m486_dv = [y for x in reg_m486_dv for y in x]
flat_reg_m486_fr = [y in reg_m486_fr for y in x]

flat_reg_m484_p = [y for x in reg_m484_p for y in x]
flat_reg_m484_dv = [y for x in reg_m484_dv for y in x]
flat_reg_m484_fr = [y for x in reg_m484_fr for y in x]

flat_reg_m481_p = [y for x in reg_m481_p for y in x]
flat_reg_m481_dv = [y for x in reg_m481_dv for y in x]
flat_reg_m481_fr = [y for x in reg_m481_fr for y in x]

flat_reg_m483_p = [y for x in reg_m483_p for y in x]
flat_reg_m483_dv = [y for x in reg_m483_dv for y in x]
flat_reg_m483_fr = [y for x in reg_m483_fr for y in x]

flat_reg_m480_p = [y for x in reg_m480_p for y in x]
flat_reg_m480_dv = [y for x in reg_m480_dv for y in x]
flat_reg_m480_fr = [y for x in reg_m480_fr for y in x]

flat_reg_m479_p = [y for x in reg_m479_p for y in x]
flat_reg_m479_dv = [y for x in reg_m479_dv for y in x]
flat_reg_m479_fr = [y for x in reg_m479_fr for y in x]

flat_reg_m478_p = [y for x in reg_m478_p for y in x]
flat_reg_m478_dv = [y for x in reg_m478_dv for y in x]
flat_reg_m478_fr = [y for x in reg_m478_fr for y in x]



ols = LogisticRegression() 
model_m486 = ols.fit(flat_reg_m486_p, flat_reg_m486_fr)
coef_m486 = ols.coef_
model_m486 = ols.fit(flat_reg_m484_p, flat_reg_m484_fr)
coef_m484 = ols.coef_
model_m483 = ols.fit(flat_reg_m483_p, flat_reg_m483_fr)
coef_m483 = ols.coef_
model_m481 = ols.fit(flat_reg_m481_p, flat_reg_m481_fr)
coef_m481 = ols.coef_
model_m480 = ols.fit(flat_reg_m480_p, flat_reg_m480_fr)
coef_m480 = ols.coef_
model_m479 = ols.fit(flat_reg_m479_p, flat_reg_m479_fr)
coef_m479 = ols.coef_
model_m478 = ols.fit(flat_reg_m478_p, flat_reg_m478_fr)
coef_m478 = ols.coef_


all_animals = np.concatenate([coef_m486,coef_m484,coef_m483, coef_m480, coef_m479, coef_m478])
means = np.mean(all_animals, axis = 0) 
stds = np.std(all_animals, axis = 0) 

pdf = PdfPages('/Users/veronikasamborska/Desktop/regressions.pdf')
predictors = [3,2,1]
plt.figure()
plt.scatter(predictors,coef_m486[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m486[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m486[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m486')
pdf.savefig()

plt.figure()
plt.scatter(predictors,coef_m484[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m484[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m484[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m484')
pdf.savefig()

plt.figure()
plt.scatter(predictors,coef_m483[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m483[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m483[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m483')
pdf.savefig()

plt.figure()
plt.scatter(predictors,coef_m481[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m481[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m481[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m481')
pdf.savefig()

plt.figure()
plt.scatter(predictors,coef_m480[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m480[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m480[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m480')
pdf.savefig()

plt.figure()
plt.scatter(predictors,coef_m479[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m479[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m479[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m479')
pdf.savefig()

plt.figure()
plt.scatter(predictors,coef_m478[0][:3], c = 'black', label = 'Choice A Predictor')
plt.scatter(predictors,coef_m478[0][3:6], c = 'grey', label = 'Reward  Predictor')
plt.scatter(predictors,coef_m478[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('m478')
pdf.savefig()

plt.figure()
plt.errorbar(predictors,means[:3],yerr =  stds[:3], c = 'black', label = 'Choice A Predictor', marker='o', linestyle = '')
plt.errorbar(predictors,means[3:6], yerr =stds[3:6], c = 'grey', label = 'Reward  Predictor', marker='o', linestyle = '')
plt.errorbar(predictors,means[6:9],yerr =stds[6:9], c = 'red', label = 'Choice x Reward Interaction', marker='o', linestyle = '')
plt.legend()
plt.xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
plt.ylabel('Coefficient')
plt.title('Mean')
pdf.savefig()
pdf.close()