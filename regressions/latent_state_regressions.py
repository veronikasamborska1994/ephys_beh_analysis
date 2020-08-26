#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:36:48 2019

@author: veronikasamborska
"""
import regressions as re
import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import regression_function as reg_f
from scipy.io import loadmat


HP_mat = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
PFC_mat = scipy.io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')


def regression_latent_state(experiment, experiment_sim_Q4_values):  
    
    C_1 = []
    C_coef = []
    cpd_1 = []
    
    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            #aligned_spikes = np.mean(aligned_spikes, axis =  2) 
            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]    
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)    
            # Getting out task indicies
            Q4 = experiment_sim_Q4_values[s]
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
                       
            choices = choices[non_forced_array]
            Q4 = Q4[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies
            
            ones = np.ones(len(choices))
            choices = choices[:len(task_1)]
            outcomes = outcomes[:len(task_1)]
            latent_state = np.ones(len(task_1))
            latent_state[predictor_a_good_task_1] = -1
            ones = ones[:len(task_1)]
            aligned_spikes = aligned_spikes[:len(task_1)]
            Q4 = Q4[:len(task_1)]
            choice_Q4 = choices*Q4


            predictors = OrderedDict([#('latent_state',latent_state), 
                                      ('choice', choices),
                                      ('reward', outcomes),
                                      ('Q4', Q4),
                                      ('choice_Q4',choice_Q4),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats = reg_f.regression_code(y, X)
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_coef.append(ols.coef_.reshape(n_neurons, n_predictors,n_timepoints)) # Predictor loadings     
            C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            cpd_1.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))

    C_1 = np.concatenate(C_1, axis = 1) # 
    C_coef = np.concatenate(C_coef, axis = 0) #
    cpd_1 = np.nanmean(np.concatenate(cpd_1,0), axis = 0)

    C_2 = []
    C_coef_2 = []
    cpd_2 = []

    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            #aligned_spikes = np.mean(aligned_spikes, axis =  2) 
            Q4 = experiment_sim_Q4_values[s]

            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]    
            
            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)    

            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            Q4 = Q4[non_forced_array]

            
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies

            ones = np.ones(len(choices))
            
            choices = choices[len(task_1):len(task_1)+len(task_2)]
            latent_state = np.ones(len(choices))
            latent_state[predictor_a_good_task_2] = -1
            
            outcomes = outcomes[len(task_1):len(task_1)+len(task_2)]
            ones = ones[len(task_1):len(task_1)+len(task_2)]
            aligned_spikes = aligned_spikes[len(task_1):len(task_1)+len(task_2)]
            Q4 = Q4[len(task_1):len(task_1)+len(task_2)]
            choice_Q4 = choices*Q4

            predictors = OrderedDict([#('latent_state',latent_state),
                                      ('choice', choices),
                                      ('reward', outcomes),
                                      ('Q4',Q4),
                                      ('choice_Q4',choice_Q4),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats = reg_f.regression_code(y, X)
            C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_coef_2.append(ols.coef_.reshape(n_neurons, n_predictors,n_timepoints)) # Predictor loadings
            cpd_2.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))


    C_2 = np.concatenate(C_2, axis = 1) # Population CPD is mean over neurons.
    C_coef_2 = np.concatenate(C_coef_2, axis = 0) # Population CPD is mean over neurons.
    cpd_2 = np.nanmean(np.concatenate(cpd_2,0), axis = 0)

    C_3 = []
    C_coef_3 = []
    cpd_3 = []
    # Finding correlation coefficients for task 1 
    for s,session in enumerate(experiment):
        aligned_spikes= session.aligned_rates[:]
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            #aligned_spikes = np.mean(aligned_spikes, axis =  2) 

            
            # Getting out task indicies   
            task = session.trial_data['task']
            forced_trials = session.trial_data['forced_trial']
            non_forced_array = np.where(forced_trials == 0)[0]
            task_non_forced = task[non_forced_array]
            task_1 = np.where(task_non_forced == 1)[0]
            task_2 = np.where(task_non_forced == 2)[0]    
            Q4 = experiment_sim_Q4_values[s]

            predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
            predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
            predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)    


            # Getting out task indicies
            forced_trials = session.trial_data['forced_trial']
            outcomes = session.trial_data['outcomes']

            choices = session.trial_data['choices']
            non_forced_array = np.where(forced_trials == 0)[0]
            
            Q4 = Q4[non_forced_array]
            choices = choices[non_forced_array]
            aligned_spikes = aligned_spikes[:len(choices),:]
            outcomes = outcomes[non_forced_array]
            # Getting out task indicies

            ones = np.ones(len(choices))
  
            choices = choices[len(task_1)+len(task_2):]
            latent_state = np.ones(len(choices))
            latent_state[predictor_a_good_task_3] = -1
            
            outcomes = outcomes[len(task_1)+len(task_2):]
            ones = ones[len(task_1)+len(task_2):]
            Q4 = Q4[len(task_1)+len(task_2):]
            choice_Q4 = choices*Q4
            aligned_spikes = aligned_spikes[len(task_1)+len(task_2):]
            
            predictors = OrderedDict([#('latent_state', latent_state),
                                      ('choice', choices),
                                      ('reward', outcomes),
                                      ('Q4', Q4),
                                      ('choice_Q4',choice_Q4),
                                      ('ones', ones)])
        
           
            X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
            n_predictors = X.shape[1]
            y = aligned_spikes.reshape([len(aligned_spikes),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats = reg_f.regression_code(y, X)

            C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            
            ols = LinearRegression(copy_X = True,fit_intercept= True)
            ols.fit(X,y)
            C_coef_3.append(ols.coef_.reshape(n_neurons,n_timepoints, n_predictors)) # Predictor loadings
            cpd_3.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))


    C_3 = np.concatenate(C_3, axis = 1) # Population CPD is mean over neurons.
    C_coef_3 = np.concatenate(C_coef_3, axis = 0) # Population CPD is mean over neurons.
    cpd_3 = np.nanmean(np.concatenate(cpd_3,0), axis = 0)
    
    return C_1, C_2, C_3, C_coef,C_coef_2,C_coef_3,cpd_1,cpd_2,cpd_3,predictors




def load_Q4s():
    # predictors_all = OrderedDict([
    #                           ('latent_state',state),
    #                           ('choice',choices_forced_unforced ),
    #                           ('reward', outcomes),
    #                           ('forced_trials',forced_trials),
    #                           ('block', block),
    #                           ('task',task),
    #                           ('A', a_pokes),
    #                           ('B', b_pokes),
    #                           ('Initiation', i_pokes),
    #                           ('Chosen_Simple_RW',chosen_Q1),
    #                           ('Chosen_Cross_learning_RW', chosen_Q4),
    #                           ('Value_A_RW', Q1_value_a),
    #                           ('Value_B_RW', Q1_value_b),
    #                           ('Value_A_Cross_learning', Q4_value_a),
    #                           ('ones', ones)])
    
    
    data_PFC = loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    data_HP= loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    
    experiment_HP = data_HP['DM'][0]
    experiment_PFC = data_PFC['DM'][0]
    
    experiment_sim_Q4_values_HP = []
    
    for exp in experiment_HP:
        experiment_sim_Q4_values_HP.append(exp[:,13])
        
        
    experiment_sim_Q4_values_PFC = []
    
    for exp in experiment_PFC:
        experiment_sim_Q4_values_PFC.append(exp[:,13])
        
def plot():
    
    C_1_HP, C_2_HP, C_3_HP,  C_coef_HP ,C_coef_2_HP, C_coef_3_HP,cpd_1_HP,cpd_2_HP,cpd_3_HP,predictors = regression_latent_state(experiment_aligned_HP, experiment_sim_Q4_values_HP)
    C_1_PFC, C_2_PFC, C_3_PFC, C_coef_PFC, C_coef_2_PFC, C_coef_3_PFC, cpd_1_PFC,cpd_2_PFC,cpd_3_PFC,predictors= regression_latent_state(experiment_aligned_PFC, experiment_sim_Q4_values_PFC)

    session = experiment_aligned_HP[0]
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
            
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()

    cpd = np.mean([cpd_1_HP,cpd_2_HP,cpd_3_HP], axis = 0)
    cpd = cpd[:,:-1]
    c =  ['black', 'turquoise', 'darkblue', 'yellow', 'green', 'orange']
    p = [*predictors]

    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
    plt.legend()
    plt.ylabel('cpd')
    plt.xlabel('Time')
    plt.vlines(ind_reward,ymin = 0, ymax = 0.15,linestyles= '--', color = 'grey', label = 'Outcome')
    plt.vlines(ind_choice,ymin = 0, ymax = 0.15,linestyles= '--', color = 'pink', label = 'Choice')
    plt.ylim(0)
    plt.legend()


    C_1_C2 = np.concatenate((C_2_HP,C_3_HP), axis = 2)
    C_1_C2 = C_1_C2[4,:,:]
    C_1_C2 = C_1_C2[~np.isnan(C_1_C2).any(axis=1)]
    cross_corr = np.corrcoef(np.transpose(C_1_C2))
    
    #plt.imshow(cross_corr)
    
    task_1 = C_1_HP[4,:]
    task_2 = C_2_HP[4,:]
    task_3 = C_3_HP[4,:]
    
    ind_n_HP = np.where(task_1 > 5)

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
    plt.title('HP')
        
    # Coef from OLS
  
    task_1 = C_coef_HP[:,4]
    task_2 = C_coef_2_HP[:,4]
    task_3 = C_coef_3_HP[:,4]

    argmax_neuron = np.argsort(-task_1)
    task_2_by_1 = task_2[argmax_neuron]
    task_1 = task_1[argmax_neuron]
    task_3_by_1 = task_3[argmax_neuron]
    

    y = np.arange(len(task_1))
    plt.figure(5)
    plt.scatter(y,task_2_by_1,s = 2, color = 'red', label = 'Task 2 sorted by Task 1')
    plt.plot(y,task_2_by_1, color = 'grey', label = 'Task 2 sorted by Task 1')
    
    #plt.scatter(y,task_3_by_1,s = 2,color = 'slateblue', label = 'Task 3 sorted by Task 1') 
    #plt.scatter(y,task_1,s = 2,color = 'black', label = 'Task 1 sorted')
    
    plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
    
    plt.legend()
    plt.title('HP Coef from Python OLS')

    
    task_1_PFC = C_1_PFC[4,:].flatten()
    task_2_PFC = C_2_PFC[4,:].flatten()
    task_3_PFC = C_3_PFC[4,:].flatten()
    
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
    plt.title('PFC')
    
    # Coef from OLS
    
    
    task_1_PFC = C_coef_PFC[:,4].flatten()
    task_2_PFC = C_coef_2_PFC[:,4].flatten()
    task_3_PFC = C_coef_3_PFC[:,4].flatten()
    ind_n_PFC = np.where(task_1_PFC > 5)

    argmax_neuron = np.argsort(-task_1_PFC)
    task_2_by_1 = task_2_PFC[argmax_neuron]
    task_1 = task_1_PFC[argmax_neuron]
    task_3_by_1 = task_3_PFC[argmax_neuron]
    

    y = np.arange(len(task_1))
    plt.figure(6)
    plt.scatter(y,task_2_by_1,s = 2, color = 'red', label = 'Task 2 sorted by Task 1')
    plt.plot(y,task_2_by_1, color = 'grey', label = 'Task 2 sorted by Task 1')

    #plt.scatter(y,task_3_by_1,s = 2,color = 'slateblue', label = 'Task 3 sorted by Task 1')
    
    plt.plot(y,task_1,color = 'black', label = 'Task 1 sorted')
    
    plt.legend()
    plt.title('PFC Coef from OLS')

    return ind_n_PFC,ind_n_HP

#ind_n_PFC, ind_n_HP = plot()
#    
def block_plot():  
    
    neuron_count_HP = 0
    neuron_count_PFC = 0
   
    for s,session in enumerate(experiment_aligned_HP):
        aligned_spikes = session.aligned_rates[:]        
        n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
        for n in range(n_neurons):
            neuron_count_HP += 1           
            if neuron_count_HP == ind_n_HP[0][0]+1:
                spikes = aligned_spikes[:,n,:]
                spikes = np.mean(spikes,axis = 1)
                # Getting out task indicies   
                task = session.trial_data['task']
                forced_trials = session.trial_data['forced_trial']
                non_forced_array = np.where(forced_trials == 0)[0]
                task_non_forced = task[non_forced_array]
                task_1 = np.where(task_non_forced == 1)[0]
                task_2 = np.where(task_non_forced == 2)[0]    
                predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
                predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
                predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)    
    
                # Getting out task indicies
                forced_trials = session.trial_data['forced_trial']
                outcomes = session.trial_data['outcomes']
    
                choices = session.trial_data['choices']
                non_forced_array = np.where(forced_trials == 0)[0]
                states  = session.trial_data['state']
                states = states[non_forced_array]
                
                choices = choices[non_forced_array]
                outcomes = outcomes[non_forced_array]
                ones = np.ones(len(choices))
                # Getting out task indicies
                predictors_all = OrderedDict([('latent_state',states),
                                  ('choice', choices),
                                  ('reward', outcomes),
                                  ('ones', ones)])
                X_all = np.vstack(predictors_all.values()).T[:len(choices),:].astype(float)


                choices = choices[:len(task_1)]
                outcomes = outcomes[:len(task_1)]
                latent_state = np.ones(len(task_1))
                latent_state[predictor_a_good_task_1] = -1
                ones = np.ones(len(task_1))
                spikes = spikes[:len(task_1)]

                
                predictors = OrderedDict([('latent_state',latent_state),
                                  ('choice', choices),
                                  ('reward', outcomes),
                                  ('ones', ones)])
    
                X = np.vstack(predictors.values()).T[:len(choices),:].astype(float)
                t = regression_code(spikes[:,np.newaxis], X)
                

                plt.figure(1)
                spikes = aligned_spikes[:,n,:]
                spikes = np.mean(spikes,axis = 1)


                x = np.arange(len(spikes))
                plt.plot(x,spikes)
                
                max_y = np.int(np.max(spikes)+ 5)
        
     
                forced_trials = session.trial_data['forced_trial']
                outcomes = session.trial_data['outcomes']
    
                choices = session.trial_data['choices']
                non_forced_array = np.where(forced_trials == 0)[0]
                           
                choices = choices[non_forced_array]
                aligned_spikes = aligned_spikes[:len(choices),:,:]
                outcomes = outcomes[non_forced_array]
                states  = session.trial_data['state']
                states = states[non_forced_array]
                
                task = session.trial_data['task']
                task_non_forced = task[non_forced_array]
                task_1 = np.where(task_non_forced == 1)[0]
                task_2 = np.where(task_non_forced == 2)[0] 
                task_3 = np.where(task_non_forced == 3)[0]  

                # Getting out task indicies
                reward_ind = np.where(outcomes == 1)
                plt.plot(reward_ind[0],outcomes[reward_ind]+max_y+2, "v", color = 'red', alpha = 0.7, markersize=1, label = 'reward')
                choices_ind = np.where(choices == 1 )
                conj_a_reward =  np.where((outcomes == 1) & (choices == 1))
                a_no_reward = np.where((outcomes == 0) & (choices == 1))
                conj_b_reward =  np.where((outcomes == 1) & (choices == 0))
                b_no_reward = np.where((outcomes == 0) & (choices == 0))
                
                plt.plot(choices_ind[0], choices[choices_ind]+max_y+5,"x", color = 'green', alpha = 0.7, markersize=3, label = 'choice')
                plt.plot(states+max_y, color = 'black', alpha = 0.7, label = 'State')
                plt.plot(task_1,np.zeros(len(task_1))+max_y+7, 'pink', label = 'Task')
                plt.plot(task_2,np.zeros(len(task_2))+max_y+9, 'pink')
                plt.plot(task_3,np.zeros(len(task_3))+max_y+11, 'pink')
                
                plt.vlines(conj_a_reward,ymin = 0, ymax = max_y, alpha = 0.3, color = 'grey', label = 'A reward')

                plt.vlines(a_no_reward,ymin = 0, ymax = max_y, alpha = 0.3,color = 'darkblue', label = 'A no reward')
                
                plt.vlines(conj_b_reward,ymin = 0, ymax = max_y, alpha = 0.3, color = 'orange', label = 'B reward')

                plt.vlines(b_no_reward,ymin = 0, ymax = max_y, alpha = 0.3,color = 'yellow', label = 'B no reward')

                plt.title('HP neuron above 5')

                plt.legend()
                    
    for s,session in enumerate(experiment_aligned_PFC):
        aligned_spikes = session.aligned_rates[:]
        
        if aligned_spikes.shape[1] > 0: # sessions with neurons? 
            n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
            
            for n in range(n_neurons):
                neuron_count_PFC += 1
                
                if neuron_count_PFC == ind_n_PFC[0][0]:
                    spikes = aligned_spikes[:,n,:]
                    spikes = np.mean(spikes, axis = 1)
                    x = np.arange(len(spikes))
                    plt.figure(2)
                    plt.plot(x,spikes)
                    
                    max_y = np.int(np.max(spikes)+ 5)
            
         
                    forced_trials = session.trial_data['forced_trial']
                    outcomes = session.trial_data['outcomes']
        
                    choices = session.trial_data['choices']
                    non_forced_array = np.where(forced_trials == 0)[0]
                               
                    choices = choices[non_forced_array]
                    aligned_spikes = aligned_spikes[:len(choices),:,:]
                    outcomes = outcomes[non_forced_array]
                    states  = session.trial_data['state']
                    states = states[non_forced_array]
                    
                    task = session.trial_data['task']
                    task_non_forced = task[non_forced_array]
                    task_1 = np.where(task_non_forced == 1)[0]
                    task_2 = np.where(task_non_forced == 2)[0] 
                    task_3 = np.where(task_non_forced == 3)[0]  

                    # Getting out task indicies
                    reward_ind = np.where(outcomes == 1)
                    plt.plot(reward_ind[0],outcomes[reward_ind]+max_y+2, "v", color = 'red', alpha = 0.7, markersize=1, label = 'reward')
                    choices_ind = np.where(choices == 1 )
                    conj_a_reward =  np.where((outcomes == 1) & (choices == 1))
                    a_no_reward = np.where((outcomes == 0) & (choices == 1))
                    conj_b_reward =  np.where((outcomes == 1) & (choices == 0))
                    b_no_reward = np.where((outcomes == 0) & (choices == 0))
                    plt.plot(choices_ind[0], choices[choices_ind]+max_y+5,"x", color = 'green', alpha = 0.7, markersize=3, label = 'choice')
                    plt.plot(states+max_y, color = 'black', alpha = 0.7, label = 'State')
                    plt.plot(task_1,np.zeros(len(task_1))+max_y+7, 'pink', label = 'Task')
                    plt.plot(task_2,np.zeros(len(task_2))+max_y+9, 'pink')
                    plt.plot(task_3,np.zeros(len(task_3))+max_y+11, 'pink')
                    
                    plt.vlines(conj_a_reward,ymin = 0, ymax = max_y, alpha = 0.3, color = 'grey', label = 'A reward')

                    plt.vlines(a_no_reward,ymin = 0, ymax = max_y, alpha = 0.3,color = 'darkblue', label = 'A no reward')

                    plt.vlines(conj_b_reward,ymin = 0, ymax = max_y, alpha = 0.3, color = 'orange', label = 'B reward')

                    plt.vlines(b_no_reward,ymin = 0, ymax = max_y, alpha = 0.3,color = 'yellow', label = 'B no reward')

                    plt.title('PFC neuron above 6')
                
                    plt.legend()
                    

                    
           