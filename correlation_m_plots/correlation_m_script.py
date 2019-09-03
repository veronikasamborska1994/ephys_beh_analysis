#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:05:02 2019

@author: veronikasamborska
"""
import numpy as np
import sys 
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/decoding')
import pseudo_sim_classifier as ps


## Less elegant script for making correlation matrices that respect space


def time_ind(session):
    # Finding the indices of initiations, choices and rewards 
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()        
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    
    ind_around_init = np.arange(initiation, initiation+10)
    ind_around_choice = np.arange(ind_choice, ind_choice+10)
    
    reward_time = initiate_choice_t[-2] +250
    ind_reward = (np.abs(t_out-reward_time)).argmin()
    ind_around_reward =  np.arange(ind_reward, ind_reward+10)
    
    return ind_around_init,ind_around_choice,ind_around_reward

def  corr_m_plots(data, condition, session):

    y = data['DM']
    X = data['Data']

    all_fr_i_1= []
    all_fr_i_1R= []
    all_fr_b_1= []
    all_fr_b_1R= []
    all_fr_a_1= []
    all_fr_a_1R= []
    
    
    all_fr_i_2= []
    all_fr_i_2R = []
    all_fr_b_2 = []
    all_fr_b_2R = []
    all_fr_a_2 = []
    all_fr_a_2R = []
    
    pokes_init_b = []
    pokes_same_space= []
    pokes_change_init = []
    poke_change_b = []
    
    
    for s, sess in enumerate(X):
        
        firing_rates_all_time = X[s]

        # Design matrix for the session
        DM = y[s]
        state =  DM[:,0]
        reward = DM[:,2]
        choices =  DM[:,1]
        b_pokes = DM[:,6]
        i_pokes = DM[:,7]
        a_pokes = DM[:,5]

        choices_b = np.where((choices == 0) & (state == 0))[0]
        choices_a = np.where((choices == 1) & (state == 1))[0]
        
        ind_choices, ind_init,int_init_same = ps.search_for_tasks_where_init_becomes_choice(DM)
        ind_around_init,ind_around_choice,ind_around_reward = time_ind(session)
        
        i_becomes_b = i_pokes[ind_init][0]
        b_when_i = b_pokes[ind_init][0]
        i_when_b = i_pokes[ind_choices][0]
       
        a_when_i = a_pokes[np.intersect1d(ind_init,choices_a)][0]
        i_a = a_pokes[np.intersect1d(ind_choices,choices_a)][0]

      
        if [i_becomes_b,b_when_i,a_when_i,i_when_b,i_becomes_b, i_a] not in pokes_init_b:
            pokes_init_b.append([i_becomes_b,b_when_i,a_when_i,i_when_b,i_becomes_b, i_a])
    
        i_poke_space = i_pokes[int_init_same][0]
        a_poke_space = a_pokes[0]
        b_i =  np.intersect1d(choices_b, int_init_same)
        b_space_1 = np.unique(b_pokes[b_i])[0]
        b_space_2 = np.unique(b_pokes[b_i])[1]

        if [i_poke_space,a_poke_space,b_space_1,b_space_2] not in pokes_same_space:
            pokes_same_space.append([i_poke_space,a_poke_space, b_space_1,b_space_2])
        
        a_i = np.intersect1d(choices_a, int_init_same)
        a_i_change = np.intersect1d(choices_a, ind_init)
        t_1_b = np.intersect1d(choices_b, int_init_same)
        t_2_b = np.intersect1d(choices_b, ind_init)

        init_t1 = i_pokes[a_i][0]
        init_t2 = i_pokes[a_i_change][0]
        b_change_1 = b_pokes[t_1_b][0]
        b_change_2 = b_pokes[t_2_b][0]

        if [init_t1,a_poke_space, init_t2,a_poke_space,b_change_1,b_change_2] not in pokes_change_init:
            pokes_change_init.append([init_t1,a_poke_space,b_change_1,init_t2,a_poke_space,b_change_2])
    
        b_i = np.intersect1d(choices_b, int_init_same)
        init_b = i_pokes[b_i][0]
        change_b_1 = np.unique(b_pokes[b_i])[0]
        change_b_2 = np.unique(b_pokes[b_i])[1]
        a_i = np.intersect1d(choices_a, int_init_same)

        a_1 = np.unique(a_pokes[a_i])[0]
        a_2 = np.unique(a_pokes[a_i])[0]
        
        if [init_b,change_b_1,a_1,init_b,change_b_2,a_2] not in poke_change_b:
            poke_change_b.append([init_b,change_b_1,a_1, init_b,change_b_2,a_2 ])
        
        
        min_trials_in_task = 24
        
        if condition == 'Initiation_B':
            
            # Indicies of B choices when Initiation is B in another task 
            i_vs_b_I = np.intersect1d(ind_init,choices_b)  
            i_vs_a_I = np.intersect1d(ind_init,choices_a)  

            # Indicies of B choices B is I in another task 
            b_vs_i_B = np.intersect1d(ind_choices, choices_b) 
            a_vs_i_B = np.intersect1d(ind_choices,choices_a)  
            
            t_1_b = i_vs_b_I
            t_1_a = i_vs_a_I
            t_2_b = b_vs_i_B
            t_2_a = a_vs_i_B

        
        elif  condition == 'Space':
            
            task = DM[:,4]
            task_1 = np.where(task == 1)[0]
            task_2 = np.where(task == 2)[0]
            task_3 = np.where(task == 3)[0]

            a_i = np.intersect1d(choices_a, int_init_same)
            b_i =  np.intersect1d(choices_b, int_init_same)
            
            task_1_a = np.intersect1d(task_1, a_i)
            task_2_a = np.intersect1d(task_2, a_i)
            task_3_a = np.intersect1d(task_3, a_i)
            
            task_1_b = np.intersect1d(task_1, b_i)
            task_2_b = np.intersect1d(task_2, b_i)
            task_3_b = np.intersect1d(task_3, b_i)
            
            if task_3_a.shape[0] == 0:
                t_1_a = task_1_a
                t_2_a = task_2_a

                t_1_b = task_1_b
                t_2_b = task_2_b

            elif task_2_a.shape[0] == 0:
                t_1_a = task_1_a
                t_2_a = task_3_a
                
                t_1_b = task_1_b
                t_2_b = task_3_b

                
            elif task_1_a.shape[0] == 0:
                t_1_a = task_2_a
                t_2_a = task_3_a
    
                t_1_b = task_2_b
                t_2_b = task_3_b

           
        elif  condition == 'Init_Change':
           
            t_1_a = np.intersect1d(choices_a, int_init_same)
            t_2_a = np.intersect1d(choices_a, ind_init)
            
            t_1_b = np.intersect1d(choices_b, int_init_same)
            t_2_b = np.intersect1d(choices_b, ind_init)

            
        elif  condition == 'B_Change':
            
            task = DM[:,4]
            task_1 = np.where(task == 1)[0]
            task_2 = np.where(task == 2)[0]
            task_3 = np.where(task == 3)[0]

            b_i = np.intersect1d(choices_b, int_init_same)
            a_i = np.intersect1d(choices_a, int_init_same)

            task_1_b = np.intersect1d(task_1, b_i)
            task_2_b = np.intersect1d(task_2, b_i)
            task_3_b = np.intersect1d(task_3, b_i)
        
            task_1_a = np.intersect1d(task_1, a_i)
            task_2_a = np.intersect1d(task_2, a_i)
            task_3_a = np.intersect1d(task_3, a_i)
            
            if task_3_b.shape[0] == 0:
                t_1_b = task_1_b
                t_2_b = task_2_b
                
                t_1_a = task_1_a
                t_2_a = task_2_a
                
            elif task_2_b.shape[0] == 0:
                t_1_b = task_1_b
                t_2_b = task_3_b
                
                t_1_a = task_1_a
                t_2_a = task_3_a
                
            elif task_1_b.shape[0] == 0:
                t_1_b = task_2_b
                t_2_b = task_3_b
                
                t_1_a = task_2_a
                t_2_a = task_3_a
        
        fr_i_1 = firing_rates_all_time[t_1_b]
        fr_i_1 = np.mean(fr_i_1[:,:,ind_around_init][:min_trials_in_task], axis = 0)
        
        fr_i_1R = firing_rates_all_time[t_1_b]
        fr_i_1R = np.mean(fr_i_1R[:,:,ind_around_reward] [:min_trials_in_task], axis = 0)

        fr_b_1 =  firing_rates_all_time[t_1_b]
        fr_b_1 = np.mean(fr_b_1[:,:,ind_around_choice][:min_trials_in_task], axis = 0)
        
        fr_b_1R =  firing_rates_all_time[t_1_b]
        fr_b_1R = np.mean(fr_b_1R[:,:,ind_around_reward][:min_trials_in_task], axis = 0)

        fr_a_1 =  firing_rates_all_time[t_1_a]
        fr_a_1 = np.mean(fr_a_1[:,:,ind_around_choice][:min_trials_in_task], axis = 0)

        fr_a_1R =  firing_rates_all_time[t_1_a]
        fr_a_1R = np.mean(fr_a_1R[:,:,ind_around_reward][:min_trials_in_task], axis  = 0)

        fr_i_2 = firing_rates_all_time[t_2_b]
        fr_i_2 = np.mean(fr_i_2[:,:,ind_around_init][:min_trials_in_task],axis = 0)

        fr_i_2R = firing_rates_all_time[t_2_b]
        fr_i_2R = np.mean(fr_i_2R[:,:,ind_around_reward][:min_trials_in_task], axis = 0)

        fr_b_2 =  firing_rates_all_time[t_2_b]
        fr_b_2 = np.mean(fr_b_2[:,:,ind_around_choice][:min_trials_in_task], axis =  0)

        fr_b_2R =  firing_rates_all_time[t_2_b]
        fr_b_2R = np.mean(fr_b_2R[:,:,ind_around_reward][:min_trials_in_task], axis = 0)

        fr_a_2 =  firing_rates_all_time[t_2_a]
        fr_a_2 = np.mean(fr_a_2[:,:,ind_around_choice][:min_trials_in_task], axis = 0)

        fr_a_2R =  firing_rates_all_time[t_2_a]
        fr_a_2R = np.mean(fr_a_2R[:,:,ind_around_reward][:min_trials_in_task], axis = 0)
        
        all_fr_i_1.append(fr_i_1)
        all_fr_i_1R.append(fr_i_1R)
        all_fr_b_1.append(fr_b_1)
        all_fr_b_1R.append(fr_b_1R)
        all_fr_a_1.append(fr_a_1)
        all_fr_a_1R.append(fr_a_1R)
     
        all_fr_i_2.append(fr_i_2)
        all_fr_i_2R.append(fr_i_2R)
        all_fr_b_2.append(fr_b_2)
        all_fr_b_2R.append(fr_b_2R)
        all_fr_a_2.append(fr_a_2)
        all_fr_a_2R.append(fr_a_2R)
    
    all_fr_i_1 = np.concatenate(all_fr_i_1, axis = 0)
    all_fr_i_1R = np.concatenate(all_fr_i_1R, axis = 0)
    all_fr_b_1 = np.concatenate(all_fr_b_1, axis = 0)
    all_fr_b_1R = np.concatenate(all_fr_b_1R, axis = 0)
    all_fr_a_1 = np.concatenate(all_fr_a_1, axis = 0)
    all_fr_a_1R = np.concatenate(all_fr_a_1R, axis = 0)
 
    all_fr_i_2 = np.concatenate(all_fr_i_2, axis = 0)
    all_fr_i_2R = np.concatenate(all_fr_i_2R, axis = 0)
    all_fr_b_2 = np.concatenate(all_fr_b_2, axis = 0)
    all_fr_b_2R = np.concatenate(all_fr_b_2R, axis = 0)
    all_fr_a_2 = np.concatenate(all_fr_a_2, axis = 0)
    all_fr_a_2R = np.concatenate(all_fr_a_2R, axis = 0)
    
    
    task_1 = np.hstack((all_fr_i_1,all_fr_i_1R,all_fr_a_1,all_fr_a_1R,all_fr_b_1,all_fr_b_1R))
    task_2 = np.hstack((all_fr_i_2,all_fr_i_2R,all_fr_a_2,all_fr_a_2R,all_fr_b_2,all_fr_b_2R))


    c_m = np.corrcoef(task_1.T,task_2.T)
    
    
    return c_m



def plot():
    
    c_m = corr_m_plots(data_HP, 'Space', session)

    plt.figure(7)
    plt.subplot(141)
    plt.imshow(c_m)
    plt.xticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])
    #plt.axis('off')
    plt.yticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])

    c_m = corr_m_plots(data_HP, 'Init_Change', session)

    plt.subplot(142)
    plt.imshow(c_m)
    plt.xticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])
    #plt.axis('off')
    plt.yticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])

    c_m = corr_m_plots(data_HP, 'B_Change', session)

    plt.subplot(143)
    plt.imshow(c_m)
    plt.xticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])
    #plt.axis('off')
    plt.yticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])

    c_m = corr_m_plots(data_HP, 'Initiation_B', session)

    plt.subplot(144)
    plt.imshow(c_m)
    plt.xticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])
    #plt.axis('off')
    plt.yticks([5,15,25,35,45, 55, 65, 75, 85, 95, 105, 115],['I1','I1 R','A1','A1R', 'B1', 'B1R', 'I2','I2 R','A2', 'A2R', 'B2', 'B2R'])

    plt.tight_layout()

