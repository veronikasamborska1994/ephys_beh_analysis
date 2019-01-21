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
from matplotlib.backends.backend_pdf import PdfPages



all_sessions = HP + PFC
reg_m478_p = []
reg_m479_p = []
reg_m480_p = []
reg_m481_p = []
reg_m483_p = []
reg_m484_p = []
reg_m486_p = []

reg_m478_dv = []
reg_m479_dv = []
reg_m480_dv = []
reg_m481_dv = []
reg_m483_dv = []
reg_m484_dv = []
reg_m486_dv = []

for s,session in enumerate(all_sessions):
    subj_ID = str(session.subject_ID)
    choices = session.trial_data['choices']
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    choice_non_forced = choices[non_forced_array]
    n_trials = len(choice_non_forced)
    
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
                              ('r_three_trial', r_three_trials)])
                              #('ra_one_trial', ra_previous_trial),
                              #('ra_two_trial', ra_two_trials),
                              #('ra_three_trial', ra_three_trials)])
    
    X = np.vstack(predictors.values()).T[:n_trials,:].astype(float)
    
    if subj_ID == '478':
        reg_m478_p.append(X)
        reg_m478_dv.append(A)
    elif subj_ID == '479':
        reg_m479_p.append(X)
        reg_m479_dv.append(A)
    elif subj_ID == '480':
        reg_m480_p.append(X)
        reg_m480_dv.append(A)
    elif subj_ID == '481':
        reg_m481_p.append(X)
        reg_m481_dv.append(A)
    elif subj_ID == '483':
        reg_m483_p.append(X)
        reg_m483_dv.append(A)
    elif subj_ID == '484':
        reg_m484_p.append(X)
        reg_m484_dv.append(A)     
    elif subj_ID == '486':
        reg_m486_p.append(X)
        reg_m486_dv.append(A)

flat_reg_m486_p = [y for x in reg_m486_p for y in x]
flat_reg_m486_dv = [y for x in reg_m486_dv for y in x]

flat_reg_m484_p = [y for x in reg_m484_p for y in x]
flat_reg_m484_dv = [y for x in reg_m484_dv for y in x]

flat_reg_m481_p = [y for x in reg_m481_p for y in x]
flat_reg_m481_dv = [y for x in reg_m481_dv for y in x]

flat_reg_m483_p = [y for x in reg_m483_p for y in x]
flat_reg_m483_dv = [y for x in reg_m483_dv for y in x]

flat_reg_m480_p = [y for x in reg_m480_p for y in x]
flat_reg_m480_dv = [y for x in reg_m480_dv for y in x]

flat_reg_m479_p = [y for x in reg_m479_p for y in x]
flat_reg_m479_dv = [y for x in reg_m479_dv for y in x]

flat_reg_m478_p = [y for x in reg_m478_p for y in x]
flat_reg_m478_dv = [y for x in reg_m478_dv for y in x]



ols = LogisticRegression() 
model_m486 = ols.fit(flat_reg_m486_p, flat_reg_m486_dv)
coef_m486 = ols.coef_
model_m486 = ols.fit(flat_reg_m484_p, flat_reg_m484_dv)
coef_m484 = ols.coef_
model_m483 = ols.fit(flat_reg_m483_p, flat_reg_m483_dv)
coef_m483 = ols.coef_
model_m481 = ols.fit(flat_reg_m481_p, flat_reg_m481_dv)
coef_m481 = ols.coef_
model_m480 = ols.fit(flat_reg_m480_p, flat_reg_m480_dv)
coef_m480 = ols.coef_
model_m479 = ols.fit(flat_reg_m479_p, flat_reg_m479_dv)
coef_m479 = ols.coef_
model_m478 = ols.fit(flat_reg_m478_p, flat_reg_m478_dv)
coef_m478 = ols.coef_


all_animals = np.concatenate([coef_m486,coef_m484,coef_m483, coef_m480, coef_m479, coef_m478])
means = np.mean(all_animals, axis = 0) 
stds = np.std(all_animals, axis = 0) 

pdf = PdfPages('/Users/veronikasamborska/Desktop/regressions.pdf')
predictors = [3,2,1]
figure()
scatter(predictors,coef_m486[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m486[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m486[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m486')
pdf.savefig()

figure()
scatter(predictors,coef_m484[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m484[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m484[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m484')
pdf.savefig()

figure()
scatter(predictors,coef_m483[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m483[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m483[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m483')
pdf.savefig()

figure()
scatter(predictors,coef_m481[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m481[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m481[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m481')
pdf.savefig()

figure()
scatter(predictors,coef_m480[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m480[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m480[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m480')
pdf.savefig()

figure()
scatter(predictors,coef_m479[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m479[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m479[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m479')
pdf.savefig()

figure()
scatter(predictors,coef_m478[0][:3], c = 'black', label = 'Choice A Predictor')
scatter(predictors,coef_m478[0][3:6], c = 'grey', label = 'Reward  Predictor')
scatter(predictors,coef_m478[0][6:9], c = 'red', label = 'Choice x Reward Interaction')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('m478')
pdf.savefig()

figure()
errorbar(predictors,means[:3],yerr =  stds[:3], c = 'black', label = 'Choice A Predictor', marker='o', linestyle = '')
errorbar(predictors,means[3:6], yerr =stds[3:6], c = 'grey', label = 'Reward  Predictor', marker='o', linestyle = '')
errorbar(predictors,means[6:9],yerr =stds[6:9], c = 'red', label = 'Choice x Reward Interaction', marker='o', linestyle = '')
legend()
xticks([1,2,3], ['Three Trials Ago', 'Two Trials Ago', 'Previous Trial']) 
ylabel('Coefficient')
title('Mean')
pdf.savefig()
clf()
pdf.close()