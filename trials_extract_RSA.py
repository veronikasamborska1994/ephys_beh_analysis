#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:26:20 2019

@author: veronikasamborska
"""


import regression_poke_based_include_a as re
import ephys_beh_import as ep
import numpy as np
from scipy.stats import zscore
import RSAs as rsa
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from numpy.linalg import matrix_rank
from celluloid import Camera
import matplotlib.pyplot as plt
import poke_aligned_spikes as pos
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

# Scipt for extracting data for RSAs 

def make_pokes_consistent(session):
    
    # A/B choices in the original behavioual files do not correspond to As being the same port in all three tasks
    # The function below finds which ones are A choices 
    poke_A1_A2_A3, poke_A1_B2_B3, poke_A1_B2_A3, poke_A1_A2_B3, poke_B1_B2_B3, poke_B1_A2_A3, poke_B1_A2_B3,poke_B1_B2_A3 = ep.poke_A_B_make_consistent(session)
    
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])

    # Finds the identity of the A port 
    if poke_A1_A2_A3 == True:
        constant_poke_a = poke_A
    elif poke_A1_B2_B3 == True:
        constant_poke_a = poke_A
    elif poke_A1_B2_A3 == True:
        constant_poke_a = poke_A
    elif poke_A1_A2_B3 == True:
        constant_poke_a = poke_A
    elif poke_B1_B2_B3 == True:
        constant_poke_a = poke_B
    elif poke_B1_A2_A3 == True:
        constant_poke_a = poke_B
    elif poke_B1_A2_B3 == True:
        constant_poke_a = poke_B
    elif poke_B1_B2_A3 == True:
        constant_poke_a = poke_B
        
    # This function gets out all the poke identities in tasks 1, 2 and 3 and the time stamps for all poke events 
    poke_identity,outcomes_non_forced,initation_choice,initiation_time_stamps,poke_list_A, poke_list_B,all_events,constant_poke_a,choices,trial_times = re.extract_poke_times_include_a(session)
    unique_pokes = np.unique(poke_identity)
    
    # Make poke 1 to always be A and create an array for later filling in when As happen 
    poke_1_id = constant_poke_a
    poke_1 = np.zeros(len(poke_identity))
    
    
    # One port is an initiation port in two out of three tasks; find which one it is 
    poke_I1_I2,poke_I1_I3, poke_I2_I3 = ep.poke_Is_make_consistent(session)
    
    #Find identities of I pokes in three tasks
    poke_A, poke_A_task_2, poke_A_task_3, poke_B, poke_B_task_2, poke_B_task_3,poke_I, poke_I_task_2,poke_I_task_3  = ep.extract_choice_pokes(session)

    
    # Finds the identity of the repeating I port and the I port that can be a B port
    if poke_I1_I2 == True:
        intitiation_in_2_tasks = poke_I
        poke_b_and_initiiation = poke_I_task_3
    elif poke_I1_I3 == True:
        intitiation_in_2_tasks = poke_I
        poke_b_and_initiiation = poke_I_task_2

    elif poke_I2_I3 == True:
        intitiation_in_2_tasks = poke_I_task_2
        poke_b_and_initiiation = poke_I

    poke_2_id = intitiation_in_2_tasks
    poke_2 = np.zeros(len(poke_identity))
    
    poke_3_id = poke_b_and_initiiation
    poke_3 = np.zeros(len(poke_identity))
    
    ind_a = np.where(unique_pokes == constant_poke_a)

    unique_pokes = np.delete(unique_pokes,ind_a)
     
    ind_init = np.where(unique_pokes == intitiation_in_2_tasks)
    unique_pokes = np.delete(unique_pokes,ind_init)

    ind_int_ch = np.where(unique_pokes == poke_b_and_initiiation)
    unique_pokes = np.delete(unique_pokes,ind_int_ch)

    # Poke 4 and 5 are always B ports than can only be B ports 
    poke_4_id = unique_pokes[0]
    poke_5_id = unique_pokes[1]
    poke_4 = np.zeros(len(poke_identity))
    poke_5 = np.zeros(len(poke_identity))
  

    for p,poke in enumerate(poke_identity):
        if poke == poke_1_id:
            poke_1[p] = 1
        if poke == poke_2_id:
            poke_2[p] = 1
        elif poke == poke_3_id:
            poke_3[p] = 1
        elif poke == poke_4_id:
            poke_4[p] = 1
        elif poke == poke_5_id:
            poke_5[p] = 1 
            
    # Indices for outcomes which codes initiation as 0 
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    outcomes = session.trial_data['outcomes']
    outcomes_non_forced = outcomes[non_forced_array]
    
    outcomes =[]
    for o,outcome in enumerate(outcomes_non_forced):
        outcomes.append(0)
        if outcome == 1:
            outcomes.append(1)
        elif outcome == 0:
            outcomes.append(0)
    outcomes = np.asarray(outcomes)  
    
    return poke_1,poke_2,poke_3,poke_4,poke_5, outcomes 


def seperate_a_into_tasks(session):  
    
    poke_1,poke_2,poke_3,poke_4,poke_5, outcomes =  make_pokes_consistent(session)
    poke_I1_I2,poke_I1_I3, poke_I2_I3 = ep.poke_Is_make_consistent(session)

    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
     
    task_2_change = np.where(task_non_forced ==2)[0]
    task_3_change = np.where(task_non_forced ==3)[0]
    
    start_task_2_index_for_pokes = task_2_change[0]*2
    start_task_3_index_for_pokes = task_3_change[0]*2
    
    # Make indicies 0 in irrelevant tasks     
    p1 = poke_1[:start_task_2_index_for_pokes]
    poke_a_task_x  = np.hstack([p1,np.zeros(len(poke_1)-start_task_2_index_for_pokes)])
    
    p2 = poke_1[start_task_2_index_for_pokes:start_task_3_index_for_pokes]
    poke_a_task_y = np.hstack([np.zeros(start_task_2_index_for_pokes),p2, np.zeros(len(poke_1)-start_task_3_index_for_pokes)])
    
    p3 = poke_1[start_task_3_index_for_pokes:]
    poke_a_task_z = np.hstack([np.zeros(start_task_3_index_for_pokes),p3])
    
    # Extract poke indicies for initations and choice B that's an initiation in one of the tasks
    # when the initiation ports are the same in task 1 and & 2
    

    if poke_I1_I2 == True:
       
        poke_initiation_task_x = poke_2[:start_task_2_index_for_pokes]
        poke_initiation_task_x = np.hstack([poke_initiation_task_x,np.zeros(len(poke_1)-start_task_2_index_for_pokes)])
        
        poke_initiation_task_y = poke_2[start_task_2_index_for_pokes:start_task_3_index_for_pokes]
        poke_initiation_task_y = np.hstack([np.zeros(start_task_2_index_for_pokes),poke_initiation_task_y, np.zeros(len(poke_1)-start_task_3_index_for_pokes)])
        
        # Initiation port that's also a B --> acts initation is in the third task 

        poke_initiation_b_task_z = poke_3[start_task_3_index_for_pokes:]
        poke_initiation_b_task_z = np.hstack([np.zeros(start_task_3_index_for_pokes),poke_initiation_b_task_z])
        
        # Initiation port that's also a B --> acts as a choice B in either the first or the second task 
        # Therefore get both the first and the second tasks (one will have non-zero values) and stack array of zeros for 
        # task 3 where it's initiation
        
        poke_choice_b_task_x = poke_3[:start_task_3_index_for_pokes]
        poke_choice_b_task_x = np.hstack([poke_choice_b_task_x, np.zeros(len(poke_1)-start_task_3_index_for_pokes)])
    
    # Extract poke indicies for initations and choice B that's an initiation in one of the tasks
    # when the initiation ports are the same in task 1 and & 3

    elif poke_I1_I3 == True:
        
        poke_initiation_task_x = poke_2[:start_task_2_index_for_pokes]
        poke_initiation_task_x = np.hstack([poke_initiation_task_x,np.zeros(len(poke_1)-start_task_2_index_for_pokes)])
        
        poke_initiation_task_y =  poke_2[start_task_3_index_for_pokes:]
        poke_initiation_task_y = np.hstack([np.zeros(start_task_3_index_for_pokes),poke_initiation_task_y])
        
        # Initiation port that's also a B --> acts initation is in the second task 

        poke_initiation_b_task_z = poke_3[start_task_2_index_for_pokes:start_task_3_index_for_pokes]
        poke_initiation_b_task_z = np.hstack([np.zeros(start_task_2_index_for_pokes),poke_initiation_b_task_z, np.zeros(len(poke_1)-start_task_3_index_for_pokes)])
        
        # Initiation port that's also a B --> acts as a choice B in either the first or the third task 
        # Therefore get both the first and the last tasks (one will have non-zero values) and stack array of zeros for 
        # task 2 where it's initiation

        poke_choice_b_task_x_1 = poke_3[:start_task_2_index_for_pokes]
        poke_choice_b_task_x_3 = poke_3[start_task_3_index_for_pokes:]
        
        poke_choice_b_task_x = np.hstack([poke_choice_b_task_x_1, np.zeros(start_task_3_index_for_pokes-start_task_2_index_for_pokes),poke_choice_b_task_x_3])
    
    # Extract poke indicies for initations and choice B that's an initiation in one of the tasks
    # when the initiation ports are the same in task 2 and & 3

    elif poke_I2_I3 == True:
        
        poke_initiation_task_x = poke_2[start_task_2_index_for_pokes:start_task_3_index_for_pokes]
        poke_initiation_task_x = np.hstack([np.zeros(start_task_2_index_for_pokes),poke_initiation_task_x,np.zeros(len(poke_1)-start_task_3_index_for_pokes)])
        
        poke_initiation_task_y = poke_2[start_task_3_index_for_pokes:]
        poke_initiation_task_y = np.hstack([np.zeros(start_task_3_index_for_pokes),poke_initiation_task_y])
        
        # Initiation port that's also a B --> acts initation is in the first task 

        poke_initiation_b_task_z =  poke_3[:start_task_2_index_for_pokes]
        poke_initiation_b_task_z = np.hstack([poke_initiation_b_task_z,np.zeros(len(poke_1)-start_task_2_index_for_pokes)])
       
        # Initiation port that's also a B --> acts as a choice B in either the second or the third task 
        # Therefore get both the second and the last tasks (one will have non-zero values) and stack array of zeros for 
        # task 1 where it's initiation

        poke_choice_b_task_x = poke_3[start_task_2_index_for_pokes:]
        poke_choice_b_task_x = np.hstack([np.zeros(start_task_2_index_for_pokes),poke_choice_b_task_x])
        
        
    return poke_a_task_x,poke_a_task_y,poke_a_task_z,outcomes,poke_initiation_task_x,poke_initiation_task_y,poke_initiation_b_task_z,\
        poke_choice_b_task_x,poke_4,poke_5



def extract_trials(experiment, all_sessions): 
   
   session_list_poke_a_task_x_r_spikes = []
   session_list_poke_a_task_x_nr_spikes = []

   session_list_poke_a_task_y_r_spikes = []
   session_list_poke_a_task_y_nr_spikes = []

   session_list_poke_a_task_z_r_spikes = []
   session_list_poke_a_task_z_nr_spikes = []

   session_list_poke_initiation_task_x_spikes = []
   session_list_poke_initiation_task_y_spikes = []
   
   session_list_poke_initiation_b_task_z_spikes = []
   
   session_list_poke_choice_b_task_x_spikes_r = []
   session_list_poke_choice_b_task_x_spikes_nr = []
   
   session_list_poke_b_task_y_spikes_r = []
   session_list_poke_b_task_y_spikes_nr = []
   
   session_list_poke_b_task_z_spikes_r = []
   session_list_poke_b_task_z_spikes_nr = []
   

   for s,session in enumerate(experiment):
        
        session = experiment[s]
        all_neurons_all_spikes_raster_plot_task = all_sessions[s]
        all_neurons_all_spikes_raster_plot_task = np.asarray(all_neurons_all_spikes_raster_plot_task)
        
        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            
            poke_a_task_x,poke_a_task_y,poke_a_task_z,outcomes,poke_initiation_task_x,poke_initiation_task_y,poke_initiation_b_task_z,\
            poke_choice_b_task_x,poke_4,poke_5 = seperate_a_into_tasks(session) 
            
            # Get rid off the time dimension        
            average_time_spikes = np.mean(all_neurons_all_spikes_raster_plot_task, axis = 2)
            n_trials, n_neurons = average_time_spikes.shape   
            average_time_spikes = zscore(average_time_spikes,axis = 0)

            # The z-scores of input "a", with any columns including non-finite
            # numbers replaced by all zeros.
            #    zscore[:, np.logical_not(np.all(np.isfinite(zscore), axis=0))] = 0

           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_a_task_x_r_spikes = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 1)), :]
            poke_a_task_x_nr_spikes  = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 0)),:]
           
            poke_a_task_y_r_spikes = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 1)),:]
            poke_a_task_y_nr_spikes  = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 0)),:]
           
            poke_a_task_z_r_spikes = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 1)),:]
            poke_a_task_z_nr_spikes  = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 0)),:]
           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_initiation_task_x_spikes = average_time_spikes[np.where(poke_initiation_task_x == 1),:]
            
            poke_initiation_task_y_spikes = average_time_spikes[np.where(poke_initiation_task_y == 1),:]

            poke_initiation_b_task_z_spikes = average_time_spikes[np.where(poke_initiation_b_task_z ==1),:]
            
            poke_choice_b_task_x_spikes_r = average_time_spikes[np.where((poke_choice_b_task_x ==1) & (outcomes == 1)),:]
            poke_choice_b_task_x_spikes_nr = average_time_spikes[np.where((poke_choice_b_task_x ==1) & (outcomes == 0)),:]

            poke_b_task_y_spikes_r = average_time_spikes[np.where((poke_4 ==1) & (outcomes == 1)),:]
            poke_b_task_y_spikes_nr = average_time_spikes[np.where((poke_4 ==1) & (outcomes == 0)),:]
            
            poke_b_task_z_spikes_r = average_time_spikes[np.where((poke_5 ==1) & (outcomes == 1)),:]
            poke_b_task_z_spikes_nr = average_time_spikes[np.where((poke_5 ==1) & (outcomes == 0)),:]
            
            #Find mean firing rates for each neuron on each type of trial
            
            mean_poke_a_task_x_r_spikes = np.mean(poke_a_task_x_r_spikes[0,:,:], axis = 0)
            mean_poke_a_task_x_nr_spikes = np.mean(poke_a_task_x_nr_spikes[0,:,:], axis = 0)
            
            mean_poke_a_task_y_r_spikes = np.mean(poke_a_task_y_r_spikes[0,:,:], axis = 0)
            mean_poke_a_task_y_nr_spikes = np.mean(poke_a_task_y_nr_spikes[0,:,:], axis = 0)

            mean_poke_a_task_z_r_spikes = np.mean(poke_a_task_z_r_spikes[0,:,:], axis = 0)
            mean_poke_a_task_z_nr_spikes = np.mean(poke_a_task_z_nr_spikes[0,:,:], axis = 0)

            mean_poke_initiation_task_x_spikes = np.mean(poke_initiation_task_x_spikes[0,:,:], axis = 0)
            mean_poke_initiation_task_y_spikes = np.mean(poke_initiation_task_y_spikes[0,:,:], axis = 0)
            
            mean_poke_initiation_b_task_z_spikes = np.mean(poke_initiation_b_task_z_spikes[0,:,:],axis = 0)
        
            mean_poke_choice_b_task_x_spikes_r = np.mean(poke_choice_b_task_x_spikes_r[0,:,:], axis = 0)
            mean_poke_choice_b_task_x_spikes_nr = np.mean(poke_choice_b_task_x_spikes_nr[0,:,:], axis = 0)
            
            mean_poke_b_task_y_spikes_r = np.mean(poke_b_task_y_spikes_r[0,:,:], axis = 0)
            mean_poke_b_task_y_spikes_nr = np.mean(poke_b_task_y_spikes_nr[0,:,:], axis = 0)

            mean_poke_b_task_z_spikes_r = np.mean(poke_b_task_z_spikes_r[0,:,:], axis = 0)
            mean_poke_b_task_z_spikes_nr = np.mean(poke_b_task_z_spikes_nr[0,:,:], axis = 0)

            
            session_list_poke_a_task_x_r_spikes.append(mean_poke_a_task_x_r_spikes)
            session_list_poke_a_task_x_nr_spikes.append(mean_poke_a_task_x_nr_spikes)
        
            session_list_poke_a_task_y_r_spikes.append(mean_poke_a_task_y_r_spikes)
            session_list_poke_a_task_y_nr_spikes.append(mean_poke_a_task_y_nr_spikes)
         
            session_list_poke_a_task_z_r_spikes.append(mean_poke_a_task_z_r_spikes)
            session_list_poke_a_task_z_nr_spikes.append(mean_poke_a_task_z_nr_spikes)
        
            session_list_poke_initiation_task_x_spikes.append(mean_poke_initiation_task_x_spikes)
            session_list_poke_initiation_task_y_spikes.append(mean_poke_initiation_task_y_spikes)
           
            session_list_poke_initiation_b_task_z_spikes.append(mean_poke_initiation_b_task_z_spikes)
            
            session_list_poke_choice_b_task_x_spikes_r.append(mean_poke_choice_b_task_x_spikes_r)
            session_list_poke_choice_b_task_x_spikes_nr.append(mean_poke_choice_b_task_x_spikes_nr)
           
            session_list_poke_b_task_y_spikes_r.append(mean_poke_b_task_y_spikes_r)
            session_list_poke_b_task_y_spikes_nr.append(mean_poke_b_task_y_spikes_nr)
            
            session_list_poke_b_task_z_spikes_r.append(mean_poke_b_task_z_spikes_r)
            session_list_poke_b_task_z_spikes_nr.append(mean_poke_b_task_z_spikes_nr)
            
  
   session_list_poke_a_task_x_r_spikes = np.concatenate(session_list_poke_a_task_x_r_spikes,0)
   session_list_poke_a_task_x_nr_spikes = np.concatenate(session_list_poke_a_task_x_nr_spikes,0)
    
   session_list_poke_a_task_y_r_spikes = np.concatenate(session_list_poke_a_task_y_r_spikes,0)
   session_list_poke_a_task_y_nr_spikes = np.concatenate(session_list_poke_a_task_y_nr_spikes,0)
     
   session_list_poke_a_task_z_r_spikes = np.concatenate(session_list_poke_a_task_z_r_spikes,0)
   session_list_poke_a_task_z_nr_spikes = np.concatenate(session_list_poke_a_task_z_nr_spikes,0)
    
   session_list_poke_initiation_task_x_spikes = np.concatenate(session_list_poke_initiation_task_x_spikes,0)
   session_list_poke_initiation_task_y_spikes = np.concatenate(session_list_poke_initiation_task_y_spikes,0)
    
       
   session_list_poke_initiation_b_task_z_spikes = np.concatenate(session_list_poke_initiation_b_task_z_spikes,0)
       
   session_list_poke_choice_b_task_x_spikes_r = np.concatenate(session_list_poke_choice_b_task_x_spikes_r,0)
   session_list_poke_choice_b_task_x_spikes_nr = np.concatenate(session_list_poke_choice_b_task_x_spikes_nr,0)
    
   session_list_poke_b_task_y_spikes_r = np.concatenate(session_list_poke_b_task_y_spikes_r,0)
   session_list_poke_b_task_y_spikes_nr = np.concatenate(session_list_poke_b_task_y_spikes_nr,0)
    
   session_list_poke_b_task_z_spikes_r = np.concatenate(session_list_poke_b_task_z_spikes_r,0)
   session_list_poke_b_task_z_spikes_nr  = np.concatenate(session_list_poke_b_task_z_spikes_nr,0)
   
   matrix_for_correlations = np.vstack([session_list_poke_a_task_x_r_spikes,session_list_poke_a_task_x_nr_spikes,session_list_poke_a_task_y_r_spikes,\
                                       session_list_poke_a_task_y_nr_spikes,session_list_poke_a_task_z_r_spikes,session_list_poke_a_task_z_nr_spikes,\
                                       session_list_poke_initiation_task_x_spikes,session_list_poke_initiation_task_y_spikes,\
                                       session_list_poke_initiation_b_task_z_spikes,session_list_poke_choice_b_task_x_spikes_r,\
                                       session_list_poke_choice_b_task_x_spikes_nr,session_list_poke_b_task_y_spikes_r,\
                                       session_list_poke_b_task_y_spikes_nr,session_list_poke_b_task_z_spikes_r,session_list_poke_b_task_z_spikes_nr])
   
   return matrix_for_correlations
  

def regression_RSA(matrix_for_correlations):
    
    C = []
    indicies_for_nans = []
    
    if np.isnan(np.mean(matrix_for_correlations)) == True:
        for i in range(matrix_for_correlations.shape[1]):
            if np.isnan(matrix_for_correlations[0,i]) == True:
               indicies_for_nans.append(i)
    
    matrix_for_correlations = np.delete(matrix_for_correlations,indicies_for_nans, axis = 1)

    correlation_m = np.corrcoef(matrix_for_correlations)
    correlation_m_f = correlation_m.flatten()
    physical_rsa = rsa.RSA_physical_rdm()
    physical_rsa  = 1*physical_rsa.flatten()
    choice_ab_rsa = rsa.RSA_a_b_initiation_rdm()
    choice_ab_rsa = 1*choice_ab_rsa.flatten()
    reward_no_reward = rsa.reward_rdm()
    reward_no_reward = 1*reward_no_reward.flatten()
    reward_at_choices = rsa.reward_choice_space()
    reward_at_choices = 1*reward_at_choices.flatten()
    choice_initiation_rsa =  rsa.choice_vs_initiation()
    choice_initiation_rsa = 1*choice_initiation_rsa.flatten()
    
    ones = np.ones(len(choice_ab_rsa))
    

    predictors = OrderedDict([('Space' , physical_rsa),
                              ('A vs B', choice_ab_rsa),
                              ('Reward',reward_no_reward),
                              ('Reward at A vs B',reward_at_choices),
                              ('Choice vs Initiation',choice_initiation_rsa),
                              ('constant', ones)])                                        
           
    X = np.vstack(predictors.values()).T[:len(physical_rsa),:].astype(float)
    print(X.shape[1])  
    rank = matrix_rank(X) 
    print(rank)
    y = correlation_m_f
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    C.append(ols.coef_) # Predictor loadings
    
    return C,correlation_m

def matrices_for_different_times():
    all_sessions_1s_500ms = pos.raster_plot_save(experiment_aligned_HP, time_window = 1)
    all_sessions_500ms_0 = pos.raster_plot_save(experiment_aligned_HP, time_window = 2)
    
    all_sessions_0_500ms = pos.raster_plot_save(experiment_aligned_HP, time_window = 3)
    all_sessions_500ms_1_sec = pos.raster_plot_save(experiment_aligned_HP, time_window = 4)
    all_sessions_1_sec_500ms  = pos.raster_plot_save(experiment_aligned_HP, time_window = 5)
    
    matrix_for_correlations_trials_min_1_500 = extract_trials(experiment_aligned_HP, all_sessions_1s_500ms) 
    matrix_for_correlations_trials_min_500_0 = extract_trials(experiment_aligned_HP, all_sessions_500ms_0) 
    matrix_for_correlations_trials_0_plus_500 = extract_trials(experiment_aligned_HP, all_sessions_0_500ms) 
    matrix_for_correlations_trials__500_1sec = extract_trials(experiment_aligned_HP, all_sessions_500ms_1_sec) 
    matrix_for_correlations_trials__1sec_500ms = extract_trials(experiment_aligned_HP, all_sessions_1_sec_500ms) 
    
    C_matrix_for_correlations_trials_min_1_500, correlation_m_1_50 = regression_RSA(matrix_for_correlations_trials_min_1_500)
    C_matrix_for_correlations_trials_min_500_0, correlation_m_500_0= regression_RSA(matrix_for_correlations_trials_min_500_0)

    C_matrix_for_correlations_trials_0_plus_500, correlation_m_0_plus_500= regression_RSA(matrix_for_correlations_trials_0_plus_500)
    C_matrix_for_correlations_trials_500_1sec,correlation_m_500_1sec = regression_RSA(matrix_for_correlations_trials__500_1sec)

    C_matrix_for_correlations_trials_1sec_500ms, correlation_m_1sec_500ms = regression_RSA(matrix_for_correlations_trials__1sec_500ms)

    physical_rsa = rsa.RSA_physical_rdm()
    choice_ab_rsa = rsa.RSA_a_b_initiation_rdm()
    reward_no_reward = rsa.reward_rdm() 
    reward_at_choices = rsa.reward_choice_space()
    choice_initiation_rsa =  rsa.choice_vs_initiation()
    
    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 25))
    grid = plt.GridSpec(10, 3, hspace=0.5, wspace=1)
    space_plt = fig.add_subplot(grid[0:2, 0])
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('Space')

    choice_plt = fig.add_subplot(grid[2:4, 0])
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('A vs B')

    reward_no_reward_plt = fig.add_subplot(grid[4:6, 0])
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('Reward')

    reward_at_choices_plt = fig.add_subplot(grid[6:8, 0])
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR')) 
    plt.xticks([])
    plt.title('Reward at Choice')

    choice_initiation_plt = fig.add_subplot(grid[8:10, 0])
    
    plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
    plt.title('Choice vs Initiation')

    trial_corr_plot = fig.add_subplot(grid[ 5:9, 1:3])
    plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
   
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
    

   
    len_C = range(len(C_matrix_for_correlations_trials_min_1_500[0]))
    bar_plot = fig.add_subplot(grid[0:3, 1:3])
    plt.ylabel('Regression Coefficient')
    plt.xticks(len_C,('Space','A vs B','Reward','Reward at Choice','Choice vs Initiation', 'Constant'), rotation = 'vertical')
    
    camera = Camera(fig)
    list_correlation = correlation_m_1_50 + correlation_m_500_0, correlation_m_0_plus_500,correlation_m_500_1sec,correlation_m_1sec_500ms
    list_bars_coef = C_matrix_for_correlations_trials_min_1_500 + C_matrix_for_correlations_trials_min_500_0\
    + C_matrix_for_correlations_trials_0_plus_500+C_matrix_for_correlations_trials_500_1sec+C_matrix_for_correlations_trials_1sec_500ms
    #title = ['-1000 to - 500 ms', '- 500ms to poke entry', 'poke entry to + 500ms', '+500ms to 1000ms']
    for i in range(len(list_correlation)):
        #plt.title(title[i])
        trial_corr_plot.imshow(list_correlation[i], aspect = 1)
        plt.colorbar()
        bar_plot.bar(len_C,list_bars_coef[i])
        space_plt.imshow(physical_rsa,aspect = 'auto')
        choice_plt.imshow(choice_ab_rsa,aspect = 'auto')
        reward_no_reward_plt.imshow(reward_no_reward,aspect = 'auto')
        reward_at_choices_plt.imshow(reward_at_choices,aspect = 'auto')
        choice_initiation_plt.imshow(choice_initiation_rsa,aspect = 'auto')
        camera.snap()
    animation = camera.animate(interval=2000)
    FFwriter = FFMpegWriter(fps=1, bitrate=2000)
    animation.save('/Users/veronikasamborska/Desktop/celluloid_minimal.mp4',writer=FFwriter)
   

    
   
    
    
