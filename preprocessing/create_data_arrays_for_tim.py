#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:52:02 2019

@author: veronikasamborska
"""
# =============================================================================
# Creating firing rate arrays with forced trials for Tim
# =============================================================================

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warping_code_all_trials as wp
import ephys_beh_import as ep
import forced_trials_extract_data as ft

import sys 
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_import/plotting/')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_import/regressions/')

import heatmap_aligned as ha 
import regressions as re 
import scipy
import neuron_firing_all_pokes as nef 


def tim_create_mat(experiment,title):# experiment_sim_Q1, experiment_sim_Q4, experiment_sim_Q1_value_a, experiment_sim_Q1_value_b, experiment_sim_Q4_values, title):
    
    all_sessions_list = []
    firing_rates = []
    for s,session in enumerate(experiment):

        firing_rate_non_forced = session.aligned_rates
        firing_rate_forced = session.aligned_rates_forced
        
        choices = session.trial_data['choices']
        trials, neurons, time = firing_rate_non_forced.shape
        firing_rate = np.zeros((len(choices), neurons, time))
        
        events = session.events
        forced_a_b = []
        for event in events:
            if 'a_forced_state' in event:
                forced_a_b.append(1)
            elif 'b_forced_state' in event:
                forced_a_b.append(0)
        forced_a_b = np.asarray(forced_a_b)
  
        index_non_forced = np.where(session.trial_data['forced_trial'] == 0)[0]
        index_forced = np.where(session.trial_data['forced_trial'] == 1)[0]

        
        task = session.trial_data['task']
        forced_trials = session.trial_data['forced_trial']
        block = session.trial_data['block']
        non_forced_array = np.where(forced_trials == 0)[0]  
        non_forced_choices = choices[non_forced_array]            
        
        # Getting out task indicies and choices
        task = session.trial_data['task']
        forced_trials = session.trial_data['forced_trial']
        non_forced_array = np.where(forced_trials == 0)[0]
        task_non_forced = task[non_forced_array]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0] 
        task_3 = np.where(task == 3)[0]
        
        task_2_non_forced = np.where(task_non_forced == 2)[0]
        task_3_non_forced = np.where(task_non_forced == 3)[0]

        forced_trials = session.trial_data['forced_trial']
        outcomes = session.trial_data['outcomes']
    
        predictor_A_Task_1_forced, predictor_A_Task_2_forced, predictor_A_Task_3_forced,\
        predictor_B_Task_1_forced, predictor_B_Task_2_forced, predictor_B_Task_3_forced, reward_forced,\
        predictor_a_good_task_1_forced, predictor_a_good_task_2_forced, predictor_a_good_task_3_forced = ft.predictors_forced(session)
        
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
        predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
        predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3,\
        reward_previous,previous_trial_task_1,previous_trial_task_2,previous_trial_task_3,\
        same_outcome_task_1, same_outcome_task_2, same_outcome_task_3,different_outcome_task_1, different_outcome_task_2, different_outcome_task_3, switch = re.predictors_include_previous_trial(session)     
           
        non_forced_choices = predictor_A_Task_1 + predictor_A_Task_2 + predictor_A_Task_3
        forced_choices = predictor_A_Task_1_forced + predictor_A_Task_2_forced + predictor_A_Task_3_forced
    
        choices_forced_unforced = np.zeros(len(choices))
        choices_forced_unforced[index_forced] = forced_choices[:len(index_forced)]
        choices_forced_unforced[index_non_forced] = non_forced_choices
    
        state = np.zeros(len(choices))
        forced_state = predictor_a_good_task_1_forced + predictor_a_good_task_2_forced + predictor_a_good_task_3_forced
        non_forced_state = np.zeros(len(non_forced_array))
        non_forced_state[predictor_a_good_task_1] = 1
        non_forced_state[predictor_a_good_task_2+task_2_non_forced[0]] = 1
        non_forced_state[predictor_a_good_task_3+task_3_non_forced[0]] = 1
        
        state[index_forced] = forced_state[:len(index_forced)]
        state[index_non_forced] = non_forced_state
        
        choices_forced_unforced[index_forced] = forced_choices[:len(index_forced)]
        choices_forced_unforced[index_non_forced] = non_forced_choices
       
        ones = np.ones(len(choices))

        firing_rate[index_forced] = firing_rate_forced[:len(index_forced)]
        firing_rate[index_non_forced] = firing_rate_non_forced[:len(index_non_forced)]

# =============================================================================
#           Extracting identity of pokes in each task                  
# =============================================================================

        poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
        poke_A_task_2 = 'poke_'+str(session.trial_data['poke_A'][task_2[0]])
        poke_A_task_3 = 'poke_'+str(session.trial_data['poke_A'][task_3[0]])
        poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
        poke_B_task_2  = 'poke_'+str(session.trial_data['poke_B'][task_2[0]])
        poke_B_task_3 = 'poke_'+str(session.trial_data['poke_B'][task_3[0]])
        configuration = session.trial_data['configuration_i']
    
        i_pokes = np.unique(configuration)
        #print('These are I pokes')
        i_poke_task_1 = configuration[0]
        i_poke_task_2 = configuration[task_2[0]]
        i_poke_task_3 = configuration[task_3[0]]
        #print(i_poke_task_1, i_poke_task_2, i_poke_task_3)
        
        poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)
        
        if poke_A1_A2_A3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_B_task_3
            
        if poke_A1_B2_B3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_A1_B2_A3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_B_task_3  
            
        if poke_A1_A2_B3 == True:
            constant_poke_a = poke_A
            poke_b_1 = poke_B
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_B1_B2_B3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_B1_A2_A3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_B_task_3
            
        if poke_B1_A2_B3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_B_task_2
            poke_b_3 = poke_A_task_3
            
        if poke_B1_B2_A3 == True:
            constant_poke_a = poke_B
            poke_b_1 = poke_A
            poke_b_2 = poke_A_task_2
            poke_b_3 = poke_B_task_3
        
        a_pokes = np.zeros(len(choices))
        a_pokes[:] = constant_poke_a[-1]
    
        b_pokes = np.zeros(len(choices))
        b_pokes[:task_1[-1]+1] = poke_b_1[-1]
        b_pokes[task_1[-1]+1:task_2[-1]+1] = poke_b_2[-1]
        b_pokes[task_2[-1]+1:] = poke_b_3[-1]
        
        i_pokes = np.zeros(len(choices))
        i_pokes[:task_1[-1]+1] = i_poke_task_1
        i_pokes[task_1[-1]+1:task_2[-1]+1] = i_poke_task_2
        i_pokes[task_2[-1]+1:] = i_poke_task_3
        
#        chosen_Q1 = experiment_sim_Q1[s][:len(choices)]
#        chosen_Q4 = experiment_sim_Q4[s][:len(choices)]
#        Q1_value_a = experiment_sim_Q1_value_a[s][:len(choices)]
#        Q1_value_b = experiment_sim_Q1_value_b[s][:len(choices)]
#        Q4_value_a = experiment_sim_Q4_values[s][:len(choices)]
#            
        predictors_all = OrderedDict([
                          ('latent_state',state),
                          ('choice',choices_forced_unforced ),
                          ('reward', outcomes),
                          ('forced_trials',forced_trials),
                          ('block', block),
                          ('task',task),
                          ('A', a_pokes),
                          ('B', b_pokes),
                          ('Initiation', i_pokes),
                          #('Chosen_Simple_RW',chosen_Q1),
                          #('Chosen_Cross_learning_RW', chosen_Q4),
                          #('Value_A_RW', Q1_value_a),
                          #('Value_B_RW', Q1_value_b),
                          #('Value_A_Cross_learning', Q4_value_a),
                          ('ones', ones)])
            
        X = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)
        
        # Save all sessions
        all_sessions_list.append(X)
        firing_rates.append(firing_rate)
   
    scipy.io.savemat('/Users/veronikasamborska/Desktop/'+ title + '.mat',{'Data': firing_rates, 'DM': all_sessions_list})
    data = {'Data': firing_rates, 'DM': all_sessions_list}
    
    return data


def x_y_coords_pokes(a_poke_task_1, b_poke_task_1, b_poke_task_2, b_poke_task_3, i_poke_task_1, i_poke_task_2,i_poke_task_3):
    one_x = 332
    one_y = 4.8
    two_x = 232
    two_y = 3.8
    three_x = 432
    three_y = 3.8
    four_x = 132
    four_y = 2.8
    six_x = 532
    six_y = 2.8
    seven_x =  232
    seven_y = 1.8
    eight_x = 432
    eight_y = 1.8
    nine_x = 332
    nine_y = 0.8
    
   
# A pokes     
    if a_poke_task_1 == 4:
        a_x = four_x
        a_y = four_y
        a_x_2 = four_x
        a_y_2 = four_y
        a_x_3 = four_x
        a_y_3 = four_y

    elif a_poke_task_1 == 6:
        a_x = six_x
        a_y = six_y
        a_x_2 = six_x
        a_y_2 = six_y
        a_x_3 = six_x
        a_y_3 = six_y
        
### I pokes 
    if i_poke_task_1 == 1:
        i_x = one_x
        i_y = one_y
    elif i_poke_task_1 == 9:
        i_x = nine_x
        i_y = nine_y

    if i_poke_task_2 == 1:
        i_x_2 = one_x
        i_y_2 = one_y
    elif i_poke_task_2 == 9:
        i_x_2 = nine_x
        i_y_2 = nine_y
    
    if i_poke_task_3 == 1:
        i_x_3 = one_x
        i_y_3 = one_y
    elif i_poke_task_3 == 9:
        i_x_3 = nine_x
        i_y_3 = nine_y

#### B pokes
    if b_poke_task_1 == 2:
        b_x = two_x
        b_y = two_y
    elif b_poke_task_1 == 3:
        b_x = three_x
        b_y = three_y
    elif b_poke_task_1 == 7:
        b_x = seven_x
        b_y = seven_y
    elif b_poke_task_1 == 8:
        b_x = eight_x
        b_y = eight_y
    elif b_poke_task_1 == 1:
        b_x = one_x
        b_y = one_y
    elif b_poke_task_1 == 9:
        b_x = nine_x
        b_y = nine_y   
    elif b_poke_task_1 == 4:
        b_x = four_x
        b_y = four_y
    elif b_poke_task_1 == 6:
        b_x = six_x
        b_y = six_y
        
    if b_poke_task_2 == 2:
        b_x_2 = two_x
        b_y_2 = two_y
    elif b_poke_task_2 == 3:
        b_x_2 = three_x
        b_y_2 = three_y
    elif b_poke_task_2 == 7:
        b_x_2 = seven_x
        b_y_2 = seven_y
    elif b_poke_task_2 == 8:
        b_x_2 = eight_x
        b_y_2 = eight_y   
    elif b_poke_task_2 == 1:
        b_x_2 = one_x
        b_y_2 = one_y
    elif b_poke_task_2 == 9:
        b_x_2 = nine_x
        b_y_2 = nine_y
    elif b_poke_task_2 == 4:
        b_x_2 = four_x
        b_y_2 = four_y
    elif b_poke_task_2 == 6:
        b_x_2 = six_x
        b_y_2 = six_y
        
            
    if b_poke_task_3 == 2:
        b_x_3 = two_x
        b_y_3 = two_y
    elif b_poke_task_3 == 3:
        b_x_3 = three_x
        b_y_3 = three_y
    elif b_poke_task_3 == 7:
        b_x_3 = seven_x
        b_y_3 = seven_y
    elif b_poke_task_3 == 8:
        b_x_3 = eight_x
        b_y_3 = eight_y
    elif b_poke_task_3 == 1:
        b_x_3 = one_x
        b_y_3 = one_y
    elif b_poke_task_3 == 9:
        b_x_3 = nine_x
        b_y_3 = nine_y
        
    elif b_poke_task_3 == 4:
        b_x_3 = four_x
        b_y_3 = four_y
    elif b_poke_task_3 == 6:
        b_x_3 = six_x
        b_y_3 = six_y
        
        
    x_points_task_1 = [a_x,b_x,i_x]
    y_points_task_1 = [a_y,b_y,i_y]

    x_points_task_2 = [a_x_2,b_x_2,i_x_2]
    y_points_task_2 = [a_y_2,b_y_2,i_y_2]

    x_points_task_3 = [a_x_3,b_x_3,i_x_3]
    y_points_task_3 = [a_y_3,b_y_3,i_y_3]
    x_all = [132,232,232,332,332,432,432,532]
    y_all = [2.8,3.8,1.8,4.8,0.8,3.8,1.8,2.8]
    
    return x_points_task_1,y_points_task_1, x_points_task_2,y_points_task_2,x_points_task_3,y_points_task_3,x_all, y_all

    
 
def plotting_trial_by_trial(data, experiment, title):
    pdf = PdfPages('/Users/veronikasamborska/Desktop/' + title +'_spikes.pdf')
    X = data['DM']
    firing_rate = data['Data']
    plt.ioff()

    for s in range(len(X)):
        DM = X[s]
        FR = firing_rate[s]
        trials, neurons, time = FR.shape
        state =  DM[:,0]
        choices =  DM[:,1]
        reward = DM[:,2]
        forced_trials  = DM[:,3]
        task = DM[:,4]
        a_pokes = DM[:,5]
        b_pokes = DM[:,6]
        i_pokes = DM[:,7]
        chosen_Q1 = DM[:,8]
        chosen_Q4  = DM[:,9]
        Q1_value_a = DM[:,10]
        Q1_value_b = DM[:,11]
        Q4_value_a = DM[:,12]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]
        task_3 = np.where(task == 3)[0]

       
        firing_rate_n = np.mean(FR, axis = 2)
       
        a_poke_task_1 = a_pokes[task_1[0]]
      
        b_poke_task_1 = b_pokes[task_1[0]]
        b_poke_task_2 = b_pokes[task_2[0]]
        b_poke_task_3 = b_pokes[task_3[0]]

        i_poke_task_1 = i_pokes[task_1[0]]
        i_poke_task_2 = i_pokes[task_2[0]]
        i_poke_task_3 = i_pokes[task_3[0]]
        x_points_task_1,y_points_task_1,\
        x_points_task_2,y_points_task_2,x_points_task_3,\
        y_points_task_3, x_all, y_all = x_y_coords_pokes(a_poke_task_1, b_poke_task_1,\
        b_poke_task_2, b_poke_task_3, i_poke_task_1, i_poke_task_2,i_poke_task_3)
        
        x_coordinates_A_1_R, x_coordinates_A_2_R, x_coordinates_A_3_R,y_coordinates_A_1_R,\
        y_coordinates_A_2_R, y_coordinates_A_3_R, x_coordinates_B_1_R, x_coordinates_B_2_R,\
        x_coordinates_B_3_R, y_coordinates_B_1_R, y_coordinates_B_2_R, y_coordinates_B_3_R, x_coordinates_I_1_R,\
        x_coordinates_I_2_R, x_coordinates_I_3_R, y_coordinates_I_1_R, y_coordinates_I_2_R, y_coordinates_I_3_R,\
        x_coordinates_A_1_nR, x_coordinates_A_2_nR, x_coordinates_A_3_nR,y_coordinates_A_1_nR,\
        y_coordinates_A_2_nR, y_coordinates_A_3_nR, x_coordinates_B_1_nR, x_coordinates_B_2_nR,\
        x_coordinates_B_3_nR, y_coordinates_B_1_nR, y_coordinates_B_2_nR, y_coordinates_B_3_nR, x_coordinates_I_1_nR,\
        x_coordinates_I_2_nR, x_coordinates_I_3_nR, y_coordinates_I_1_nR, y_coordinates_I_2_nR, y_coordinates_I_3_nR = nef.coordinates_for_plots(experiment[s])
    

        for n in range(neurons):
                        
            fig = plt.figure(figsize=(18, 9))
            grid = plt.GridSpec(3, 6, hspace=0.5, wspace=1)
            t1 = fig.add_subplot(grid[0, 0:4])
            
            # Plotting Task 1 
            fr_task_1 = firing_rate_n[:task_1[-1]]
            max_y_task_1 = np.int(np.max(fr_task_1[:,n])+ 5)
            reward_task_1 = reward[:task_1[-1]]
            reward_ind_task_1 = np.where(reward_task_1 == 1)[0]

            t1.plot(reward_ind_task_1,reward_task_1[reward_ind_task_1]+max_y_task_1+9, "v", color = 'red', alpha = 0.7, markersize=1, label = 'reward')
            
            choices_task_1 = choices[:task_1[-1]]
            choices_ind_task_1= np.where(choices_task_1 == 1 )[0]

            t1.plot(choices_ind_task_1, choices_task_1[choices_ind_task_1]+max_y_task_1+7,"x", color = 'green', alpha = 0.7, markersize=3, label = 'choice')
            
            state_task_1 = state[:task_1[-1]]*5
            t1.plot(state_task_1+max_y_task_1, color = 'black', alpha = 0.7, label = 'State')
            t1.plot(fr_task_1[:,n], color = 'black')
            forced_trials_task_1 = forced_trials[:task_1[-1]]*3
            t1.plot(forced_trials_task_1+max_y_task_1+13, 'x', color = 'orange', alpha = 0.7, markersize=3, label = 'forced')

            
            conj_a_reward =  np.where((reward_task_1 == 1) & (choices_task_1 == 1))[0]
            a_no_reward = np.where((reward_task_1 == 0) & (choices_task_1 == 1))[0]
            conj_b_reward =  np.where((reward_task_1 == 1) & (choices_task_1 == 0))[0]
            b_no_reward = np.where((reward_task_1 == 0) & (choices_task_1 == 0))[0]
            
            t1.vlines(conj_a_reward,ymin = 0, ymax = max_y_task_1, alpha = 0.3, color = 'darkblue', label = 'A reward')    
            t1.vlines(a_no_reward,ymin = 0, ymax = max_y_task_1, alpha = 0.3,color = 'cyan', label = 'A no reward')            
            t1.vlines(conj_b_reward,ymin = 0, ymax = max_y_task_1, alpha = 0.3, color = 'red', label = 'B reward') 
            t1.vlines(b_no_reward,ymin = 0, ymax = max_y_task_1, alpha = 0.3,color = 'pink', label = 'B no reward')
           
            config_1  = fig.add_subplot(grid[0, 4:6])
            config_1.scatter(x_all,y_all,s = 100, c = ['black'])
            config_1.scatter(x_points_task_1,y_points_task_1,s = 100, c = ['blue','red','green'])
            config_1.plot(x_coordinates_A_1_R, y_coordinates_A_1_R[n], color = 'firebrick')   
            config_1.plot(x_coordinates_B_1_R, y_coordinates_B_1_R[n], color = 'firebrick')    
            config_1.plot(x_coordinates_I_1_R, y_coordinates_I_1_R[n], color = 'red', linestyle = ':')   
            config_1.plot(x_coordinates_A_1_nR, y_coordinates_A_1_nR[n], color = 'red', linestyle='dashed')   
            config_1.plot(x_coordinates_B_1_nR, y_coordinates_B_1_nR[n], color = 'red',linestyle='dashed')    
            config_1.plot(x_coordinates_I_1_nR, y_coordinates_I_1_nR[n], color = 'red',linestyle='dashed')   
            config_1.axis('off')


            # Plotting Task 2
            t2 = fig.add_subplot(grid[1, 0:4])
            fr_task_2 = firing_rate_n[task_1[-1]:task_2[-1]]
            max_y_task_2 = np.int(np.max(fr_task_2[:,n])+ 5)
            reward_task_2 = reward[task_1[-1]:task_2[-1]]
            reward_ind_task_2 = np.where(reward_task_2 == 1)[0]

            t2.plot(reward_ind_task_2,reward_task_2[reward_ind_task_2]+max_y_task_2+9, "v", color = 'red', alpha = 0.7, markersize=1, label = 'reward')
            
            choices_task_2 = choices[task_1[-1]:task_2[-1]]
            choices_ind_task_2 = np.where(choices_task_2 == 1 )[0]

            t2.plot(choices_ind_task_2, choices_task_2[choices_ind_task_2]+max_y_task_2+7,"x", color = 'green', alpha = 0.7, markersize=3, label = 'choice')
            
            state_task_2 = state[task_1[-1]:task_2[-1]]*5
            t2.plot(state_task_2+max_y_task_2, color = 'black', alpha = 0.7, label = 'State')
            t2.plot(fr_task_2[:,n], color = 'black')
            
            forced_trials_task_2 = forced_trials[task_1[-1]:task_2[-1]]*3
            t2.plot(forced_trials_task_2+max_y_task_2+13,"x", color = 'orange', alpha = 0.7,  markersize=3,label = 'forced')

           
            conj_a_reward =  np.where((reward_task_2 == 1) & (choices_task_2 == 1))[0]
            a_no_reward = np.where((reward_task_2 == 0) & (choices_task_2 == 1))[0]
            conj_b_reward =  np.where((reward_task_2 == 1) & (choices_task_2 == 0))[0]
            b_no_reward = np.where((reward_task_2 == 0) & (choices_task_2 == 0))[0]
            
            t2.vlines(conj_a_reward,ymin = 0, ymax = max_y_task_2, alpha = 0.3, color = 'darkblue', label = 'A reward')    
            t2.vlines(a_no_reward,ymin = 0, ymax = max_y_task_2, alpha = 0.3, color = 'cyan', label = 'A no reward')            
            t2.vlines(conj_b_reward,ymin = 0, ymax = max_y_task_2, alpha = 0.3, color = 'red', label = 'B reward') 
            t2.vlines(b_no_reward,ymin = 0, ymax = max_y_task_2, alpha = 0.3, color = 'pink', label = 'B no reward')
            
            config_2  = fig.add_subplot(grid[1, 4:6])
            config_2.scatter(x_all,y_all,s = 100, c = ['black'])
            config_2.scatter(x_points_task_2,y_points_task_2,s = 100, c = ['blue','red','green'])
            config_2.plot(x_coordinates_A_2_R, y_coordinates_A_2_R[n], color = 'cadetblue')      
            config_2.plot(x_coordinates_B_2_R, y_coordinates_B_2_R[n], color = 'cadetblue' )       
            config_2.plot(x_coordinates_I_2_R, y_coordinates_I_2_R[n], color = 'blue',  linestyle = ':')      
            config_2.plot(x_coordinates_A_2_nR, y_coordinates_A_2_nR[n], color = 'blue', linestyle='dashed')      
            config_2.plot(x_coordinates_B_2_nR, y_coordinates_B_2_nR[n], color = 'blue',linestyle='dashed' )       
            config_2.plot(x_coordinates_I_2_nR, y_coordinates_I_2_nR[n], color = 'blue',  linestyle='dashed')      
            config_2.axis('off')
            
            

            # Plotting Task 2
            t3 = fig.add_subplot(grid[2, 0:4])
            fr_task_3 = firing_rate_n[task_2[-1]:task_3[-1]]
            max_y_task_3 = np.int(np.max(fr_task_3[:,n])+ 5)
            reward_task_3 = reward[task_2[-1]:task_3[-1]]
            reward_ind_task_3 = np.where(reward_task_3 == 1)[0]

            t3.plot(reward_ind_task_3,reward_task_3[reward_ind_task_3]+max_y_task_3+9, "v", color = 'red', alpha = 0.7, markersize=1, label = 'reward')
            
            choices_task_3 = choices[task_2[-1]:task_3[-1]]
            choices_ind_task_3 = np.where(choices_task_3 == 1 )[0]

            t3.plot(choices_ind_task_3, choices_task_3[choices_ind_task_3]+max_y_task_3+7,"x", color = 'green', alpha = 0.7, markersize=3, label = 'choice')
           
            state_task_3 = state[task_2[-1]:task_3[-1]]*5
            t3.plot(state_task_3 +  max_y_task_3, color = 'black', alpha = 0.7, label = 'State')
            t3.plot(fr_task_3[:,n], color = 'black')
            
            forced_trials_task_3 = forced_trials[task_2[-1]:task_3[-1]]*3
            t3.plot(forced_trials_task_3+max_y_task_3+13,"x", color = 'orange', alpha = 0.7,  markersize=3,label = 'forced')

            conj_a_reward =  np.where((reward_task_3 == 1) & (choices_task_3 == 1))[0]
            a_no_reward = np.where((reward_task_3 == 0) & (choices_task_3 == 1))[0]
            conj_b_reward =  np.where((reward_task_3 == 1) & (choices_task_3 == 0))[0]
            b_no_reward = np.where((reward_task_3 == 0) & (choices_task_3 == 0))[0]
            
            t3.vlines(conj_a_reward,ymin = 0, ymax = max_y_task_3, alpha = 0.3, color = 'darkblue', label = 'A reward')    
            t3.vlines(a_no_reward,ymin = 0, ymax = max_y_task_3, alpha = 0.3,color = 'cyan', label = 'A no reward')            
            t3.vlines(conj_b_reward,ymin = 0, ymax = max_y_task_3, alpha = 0.3, color = 'red', label = 'B reward') 
            t3.vlines(b_no_reward,ymin = 0, ymax = max_y_task_3, alpha = 0.3,color = 'pink', label = 'B no reward')
            t3.legend(loc=2, fontsize = 'x-small')
            
            config_3  = fig.add_subplot(grid[2, 4:6])
            config_3.scatter(x_all,y_all,s = 100, c = ['black'])
            config_3.scatter(x_points_task_3,y_points_task_3,s = 100, c = ['blue','red','green'])
            config_3.axis('off')
            config_3.plot(x_coordinates_A_3_R, y_coordinates_A_3_R[n], color = 'olive')  
            config_3.plot(x_coordinates_B_3_R, y_coordinates_B_3_R[n], color = 'olive')   
            config_3.plot(x_coordinates_I_3_R, y_coordinates_I_3_R[n], color = 'green',  linestyle = ':')     
            config_3.plot(x_coordinates_A_3_nR, y_coordinates_A_3_nR[n], color = 'green',linestyle='dashed')      
            config_3.plot(x_coordinates_B_3_nR, y_coordinates_B_3_nR[n], color = 'green', linestyle='dashed')   
            config_3.plot(x_coordinates_I_3_nR, y_coordinates_I_3_nR[n], color = 'green',  linestyle='dashed')    
                
            pdf.savefig()
            plt.clf()
            
    pdf.close()