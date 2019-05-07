#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:48:01 2019

@author: veronikasamborska
"""

import regressions as re
import numpy as np
import matplotlib.pyplot as plt
import ephys_beh_import as ep


def plotting_reward(experiment):
    pdf = PdfPages('/Users/veronikasamborska/Desktop/spikes_HP.pdf')

    for session in experiment:
        poke_I_1_2 = False
        poke_I_1_3 = False
        poke_I_2_3 = False
        
        predictor_A_Task_1, predictor_A_Task_2, predictor_A_Task_3,\
        predictor_B_Task_1, predictor_B_Task_2, predictor_B_Task_3, reward,\
        predictor_a_good_task_1,predictor_a_good_task_2, predictor_a_good_task_3 = re.predictors_pokes(session)
        poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3 = ep.extract_choice_pokes(session)
        
        forced_trials = session.trial_data['forced_trial']
        non_forced_array = np.where(forced_trials == 0)[0]
        task = session.trial_data['task']
        task_non_forced = task[non_forced_array]
             
        #task_2_change = np.where(task_non_forced ==2)[0]
        #task_3_change = np.where(task_non_forced ==3)[0]
        
        spikes = session.aligned_rates

        if poke_I == poke_I_task_2: 
            poke_I_1_2 = True
        elif poke_I == poke_I_task_3:
            poke_I_1_3 = True
        elif poke_I_task_2 == poke_I_task_3:
            poke_I_2_3 = True

        
        neurons = spikes.shape[1]
        for i in range(neurons):
            t_out = session.t_out
            initiate_choice_t = session.target_times 
            reward_time = initiate_choice_t[-2] +250
        
            ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
            ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
            ind_reward = (np.abs(t_out-reward_time)).argmin()
            
            #reward_t1 = reward[:task_2_change[0]]
            #reward_t2 = reward[task_2_change[0]:task_3_change[0]]
            #reward_t3 = reward[task_3_change[0]:]
        
        
            
            A_task_1_r = np.mean(spikes[np.where((predictor_A_Task_1 == 1) &( reward == 1 )),i,:], axis = 1)
            
            A_task_1_nr = np.mean(spikes[np.where((predictor_A_Task_1 == 1) &( reward == 0 )),i,:], axis = 1)

            B_task_1_r = np.mean(spikes[np.where((predictor_A_Task_1 == 0) &( reward == 1 )),i,:], axis = 1)
            
            B_task_1_nr = np.mean(spikes[np.where((predictor_A_Task_1 == 0) &( reward == 0 )),i,:], axis = 1)

            A_task_2_r = np.mean(spikes[np.where((predictor_A_Task_2 == 1) &( reward == 1 )),i,:], axis = 1)
            
            A_task_2_nr = np.mean(spikes[np.where((predictor_A_Task_2 == 1) &( reward == 0 )),i,:], axis = 1)
            
            B_task_2_r = np.mean(spikes[np.where((predictor_A_Task_2 == 0) &( reward == 1 )),i,:], axis = 1)
            
            B_task_2_nr = np.mean(spikes[np.where((predictor_A_Task_2 == 0) &( reward == 0 )),i,:], axis = 1)

            A_task_3_r = np.mean(spikes[np.where((predictor_A_Task_3 == 1) &( reward == 1 )),i,:], axis = 1)
            
            A_task_3_nr = np.mean(spikes[np.where((predictor_A_Task_3 == 1) &( reward == 0 )),i,:], axis = 1)
            
            B_task_3_r = np.mean(spikes[np.where((predictor_A_Task_3 == 0) &( reward == 1 )),i,:], axis = 1)
            
            B_task_3_nr = np.mean(spikes[np.where((predictor_A_Task_3 == 0) &( reward == 0 )),i,:], axis = 1)


            fig = plt.figure(figsize=(8, 25))
            grid = plt.GridSpec(3, 1, hspace=0.7, wspace=0.4)
            if poke_I_1_3 == True:
                fig.add_subplot(grid[0]) 
                plt.plot(A_task_1_r[0,:], color = 'grey', label =  'A reward t1 ')
                plt.plot(A_task_1_nr[0,:], color = 'grey', linestyle = '--',label =  'A No reward t2')
            
                plt.plot(B_task_1_r[0,:], color = 'darkblue',  label =  'B reward t1 ')
                plt.plot(B_task_1_nr[0,:], color = 'darkblue',linestyle = '--',  label =  'B No reward t2')
                
                fig.add_subplot(grid[1]) 
                plt.plot(A_task_3_r[0,:], color = 'grey',  label =  'A reward t3 ')
                plt.plot(A_task_3_nr[0,:], color = 'grey',linestyle = '--',  label =  'A No reward t3')
            
                plt.plot(B_task_3_r[0,:], color = 'darkblue',  label =  'B reward t3')
                plt.plot(B_task_3_nr[0,:], color = 'darkblue',linestyle = '--',  label =  'B No reward t3')
           
                fig.add_subplot(grid[2]) 
                plt.plot(A_task_2_r[0,:], color = 'grey',  label =  'A reward t2 ')
                plt.plot(A_task_2_nr[0,:], color = 'grey', linestyle = '--', label =  'A No reward t2')
                    
                plt.plot(B_task_2_r[0,:], color = 'darkblue',  label =  'B reward t2 ')
                plt.plot(B_task_2_nr[0,:], color = 'darkblue', linestyle = '--', label =  'B No reward t2')
                 
             
            if poke_I_1_2 == True:
                fig.add_subplot(grid[1])   
                plt.plot(A_task_2_r[0,:], color = 'grey',  label =  'A reward t2 ')
                plt.plot(A_task_2_nr[0,:], color = 'grey', linestyle = '--', label =  'A No reward t2')
                    
                plt.plot(B_task_2_r[0,:], color = 'darkblue',  label =  'B reward t2 ')
                plt.plot(B_task_2_nr[0,:], color = 'darkblue', linestyle = '--', label =  'B No reward t2')
                
                fig.add_subplot(grid[2]) 
                plt.plot(A_task_3_r[0,:], color = 'grey',  label =  'A reward t3 ')
                plt.plot(A_task_3_nr[0,:], color = 'grey',linestyle = '--',  label =  'A No reward t3')
            
                plt.plot(B_task_3_r[0,:], color = 'darkblue',  label =  'B reward t3')
                plt.plot(B_task_3_nr[0,:], color = 'darkblue',linestyle = '--',  label =  'B No reward t3')
           
                fig.add_subplot(grid[1]) 
                plt.plot(A_task_2_r[0,:], color = 'grey',  label =  'A reward t2 ')
                plt.plot(A_task_2_nr[0,:], color = 'grey', linestyle = '--', label =  'A No reward t2')
                    
                plt.plot(B_task_2_r[0,:], color = 'darkblue',  label =  'B reward t2 ')
                plt.plot(B_task_2_nr[0,:], color = 'darkblue', linestyle = '--', label =  'B No reward t2')
                 
            elif poke_I_2_3 == True:  
                fig.add_subplot(grid[2]) 
                plt.plot(A_task_1_r[0,:], color = 'grey', label =  'A reward t1 ')
                plt.plot(A_task_1_nr[0,:], color = 'grey', linestyle = '--',label =  'A No reward t2')
            
                plt.plot(B_task_1_r[0,:], color = 'darkblue',  label =  'B reward t1 ')
                plt.plot(B_task_1_nr[0,:], color = 'darkblue',linestyle = '--',  label =  'B No reward t2')
                
                
                fig.add_subplot(grid[0]) 
                plt.plot(A_task_3_r[0,:], color = 'grey',  label =  'A reward t3 ')
                plt.plot(A_task_3_nr[0,:], color = 'grey',linestyle = '--',  label =  'A No reward t3')
            
                plt.plot(B_task_3_r[0,:], color = 'darkblue',  label =  'B reward t3')
                plt.plot(B_task_3_nr[0,:], color = 'darkblue',linestyle = '--',  label =  'B No reward t3')
           
                fig.add_subplot(grid[1]) 
                plt.plot(A_task_2_r[0,:], color = 'grey',  label =  'A reward t2 ')
                plt.plot(A_task_2_nr[0,:], color = 'grey', linestyle = '--', label =  'A No reward t2')
                    
                plt.plot(B_task_2_r[0,:], color = 'darkblue',  label =  'B reward t2 ')
                plt.plot(B_task_2_nr[0,:], color = 'darkblue', linestyle = '--', label =  'B No reward t2')
                 
            fig.add_subplot(grid[0]) 
            plt.title('Task 1')
            fig.add_subplot(grid[1]) 
            plt.title('Task 2')
            fig.add_subplot(grid[2]) 
            plt.title('Task 3')
             
              
            
            plt.legend()
            plt.xticks([ind_init,ind_choice,ind_reward], ('Initiation', 'Choice', 'Reward' ), rotation = 'vertical')  
            pdf.savefig()
            plt.clf()
    pdf.close()
            
            
