#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:26:20 2019

@author: veronikasamborska
"""
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/RSA')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

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
from matplotlib.widgets import Slider, Button, RadioButtons
import rsa_no_initiation as rs

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

# Scipt for extracting data for RSAs 

#all_sessions_HP = pos.raster_plot_save(experiment_aligned_HP)
#all_sessions_PFC = pos.raster_plot_save(experiment_aligned_PFC)

def make_pokes_consistent(session):
# =============================================================================
#     
#      A/B choices in the original behavioual files do not correspond to As being the same port in all three tasks
#      The function below finds which ones are A choices 
#     
# =============================================================================
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
# =============================================================================
#   This function identifies which Initiation pokes are in the same physical position in 3 tasks to extract them consistently across sessions. Also makes sure Initiation 
#   port that also acts as a B poke is extracted consistently and separates A pokes into 3 tasks. 
# =============================================================================    
#   Outputs are:  
#   A pokes in 3 tasks : poke_a_task_x, poke_a_task_y,poke_a_task_z
#   Rewards: outcomes,
#   Initiation Pokes that only ever initiation pokes: poke_initiation_task_x,poke_initiation_task_y
#   Initiation Poke that can be an initiation poke or a B poke: poke_initiation_b_task_z,poke_choice_b_task_x
#   B Choices: poke_4,poke_5
#
# =============================================================================
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



def extract_trials_all_time_points(experiment,all_sessions):
    
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
        all_neurons_all_spikes_raster_plot_task = all_neurons_all_spikes_raster_plot_task[:,:,30:60]
        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            
            poke_a_task_x,poke_a_task_y,poke_a_task_z,outcomes,poke_initiation_task_x,poke_initiation_task_y,poke_initiation_b_task_z,\
            poke_choice_b_task_x,poke_4,poke_5 = seperate_a_into_tasks(session) 
            
            # Get rid off the time dimension        
            #average_time_spikes = np.mean(all_neurons_all_spikes_raster_plot_task, axis = 2)
            n_trials, n_neurons, n_timepoints = all_neurons_all_spikes_raster_plot_task.shape   
            average_time_spikes = zscore(all_neurons_all_spikes_raster_plot_task,axis = 0)

            # The z-scores of input "a", with any columns including non-finite
            # numbers replaced by all zeros.
            average_time_spikes[:, np.logical_not(np.all(np.isfinite(average_time_spikes), axis=0))] = 0

           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_a_task_x_r_spikes = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 1)), :,:]
            poke_a_task_x_nr_spikes  = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 0)),:,:]
           
            poke_a_task_y_r_spikes = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 1)),:,:]
            poke_a_task_y_nr_spikes  = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 0)),:,:]
           
            poke_a_task_z_r_spikes = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 1)),:,:]
            poke_a_task_z_nr_spikes  = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 0)),:,:]
           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_initiation_task_x_spikes = average_time_spikes[np.where(poke_initiation_task_x == 1),:,:]
            
            poke_initiation_task_y_spikes = average_time_spikes[np.where(poke_initiation_task_y == 1),:,:]

            poke_initiation_b_task_z_spikes = average_time_spikes[np.where(poke_initiation_b_task_z ==1),:,:]
            
            poke_choice_b_task_x_spikes_r = average_time_spikes[np.where((poke_choice_b_task_x ==1) & (outcomes == 1)),:,:]
            poke_choice_b_task_x_spikes_nr = average_time_spikes[np.where((poke_choice_b_task_x ==1) & (outcomes == 0)),:,:]

            poke_b_task_y_spikes_r = average_time_spikes[np.where((poke_4 ==1) & (outcomes == 1)),:,:]
            poke_b_task_y_spikes_nr = average_time_spikes[np.where((poke_4 ==1) & (outcomes == 0)),:,:]
            
            poke_b_task_z_spikes_r = average_time_spikes[np.where((poke_5 ==1) & (outcomes == 1)),:,:]
            poke_b_task_z_spikes_nr = average_time_spikes[np.where((poke_5 ==1) & (outcomes == 0)),:,:]
            
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
   
   ####Flatten 
   session_list_poke_a_task_x_r_spikes = session_list_poke_a_task_x_r_spikes.flatten()
   session_list_poke_a_task_x_nr_spikes = session_list_poke_a_task_x_nr_spikes.flatten()
    
   session_list_poke_a_task_y_r_spikes = session_list_poke_a_task_y_r_spikes.flatten()
   session_list_poke_a_task_y_nr_spikes = session_list_poke_a_task_y_nr_spikes.flatten()
     
   session_list_poke_a_task_z_r_spikes = session_list_poke_a_task_z_r_spikes.flatten()
   session_list_poke_a_task_z_nr_spikes = session_list_poke_a_task_z_nr_spikes.flatten()
    
   session_list_poke_initiation_task_x_spikes = session_list_poke_initiation_task_x_spikes.flatten()
   session_list_poke_initiation_task_y_spikes = session_list_poke_initiation_task_y_spikes.flatten()
    
       
   session_list_poke_initiation_b_task_z_spikes =session_list_poke_initiation_b_task_z_spikes.flatten()
       
   session_list_poke_choice_b_task_x_spikes_r = session_list_poke_choice_b_task_x_spikes_r.flatten()
   session_list_poke_choice_b_task_x_spikes_nr = session_list_poke_choice_b_task_x_spikes_nr.flatten()
    
   session_list_poke_b_task_y_spikes_r = session_list_poke_b_task_y_spikes_r.flatten()
   session_list_poke_b_task_y_spikes_nr = session_list_poke_b_task_y_spikes_nr.flatten()
    
   session_list_poke_b_task_z_spikes_r = session_list_poke_b_task_z_spikes_r.flatten()
   session_list_poke_b_task_z_spikes_nr  = session_list_poke_b_task_z_spikes_nr.flatten()
   matrix_for_correlations = np.vstack([session_list_poke_a_task_x_r_spikes,session_list_poke_a_task_x_nr_spikes,session_list_poke_a_task_y_r_spikes,\
                                       session_list_poke_a_task_y_nr_spikes,session_list_poke_a_task_z_r_spikes,session_list_poke_a_task_z_nr_spikes,\
                                       session_list_poke_initiation_task_x_spikes,session_list_poke_initiation_task_y_spikes,\
                                       session_list_poke_initiation_b_task_z_spikes,session_list_poke_choice_b_task_x_spikes_r,\
                                       session_list_poke_choice_b_task_x_spikes_nr,session_list_poke_b_task_y_spikes_r,\
                                       session_list_poke_b_task_y_spikes_nr,session_list_poke_b_task_z_spikes_r,session_list_poke_b_task_z_spikes_nr])
   
   
#   matrix_for_correlations = np.concatenate([session_list_poke_a_task_x_r_spikes,session_list_poke_a_task_x_nr_spikes,session_list_poke_a_task_y_r_spikes,\
#                                       session_list_poke_a_task_y_nr_spikes,session_list_poke_a_task_z_r_spikes,session_list_poke_a_task_z_nr_spikes,\
#                                       session_list_poke_initiation_task_x_spikes,session_list_poke_initiation_task_y_spikes,\
#                                       session_list_poke_initiation_b_task_z_spikes,session_list_poke_choice_b_task_x_spikes_r,\
#                                       session_list_poke_choice_b_task_x_spikes_nr,session_list_poke_b_task_y_spikes_r,\
#                                       session_list_poke_b_task_y_spikes_nr,session_list_poke_b_task_z_spikes_r,session_list_poke_b_task_z_spikes_nr],axis = 1)
#   
    
   return matrix_for_correlations


def arrange_x_y_z_tasks(session): 
    
    forced_trials = session.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0]
    task = session.trial_data['task']
    task_non_forced = task[non_forced_array]
 
    task_2_change = np.where(task_non_forced ==2)[0]
    task_3_change = np.where(task_non_forced ==3)[0]

    start_task_2_index_for_pokes = task_2_change[0]*2
    start_task_3_index_for_pokes = task_3_change[0]*2
    
    poke_a_task_x,poke_a_task_y,poke_a_task_z,outcomes,poke_initiation_task_x,poke_initiation_task_y,poke_initiation_b_task_z,\
    poke_choice_b_task_x,poke_4,poke_5 = seperate_a_into_tasks(session) 
    
    # Check in which task it was initiation port 
    if sum(poke_initiation_task_x[:start_task_2_index_for_pokes]) > 0:
        poke_initiation_task_1 = poke_initiation_task_x
    elif sum(poke_initiation_task_x[start_task_2_index_for_pokes:start_task_3_index_for_pokes]) > 0:
        poke_initiation_task_2 = poke_initiation_task_x
    elif sum(poke_initiation_task_x[start_task_3_index_for_pokes:]) > 0:
        poke_initiation_task_3 = poke_initiation_task_x

   #Check in which task it was initiation port 
    if sum(poke_initiation_task_y[:start_task_2_index_for_pokes]) > 0:
        poke_initiation_task_1 = poke_initiation_task_y
    elif sum(poke_initiation_task_y[start_task_2_index_for_pokes:start_task_3_index_for_pokes]) > 0:
        poke_initiation_task_2 = poke_initiation_task_y
    elif sum(poke_initiation_task_y[start_task_3_index_for_pokes:]) > 0:
        poke_initiation_task_3 = poke_initiation_task_y

    #Check in which task it was initiation port 
    if sum(poke_initiation_b_task_z[:start_task_2_index_for_pokes]) > 0:
        poke_initiation_task_1 = poke_initiation_b_task_z
    elif sum(poke_initiation_b_task_z[start_task_2_index_for_pokes:start_task_3_index_for_pokes]) > 0:
        poke_initiation_task_2 = poke_initiation_b_task_z
    elif sum(poke_initiation_b_task_z[start_task_3_index_for_pokes:]) > 0:
        poke_initiation_task_3 = poke_initiation_b_task_z


  # Check in which task it was B port 
    if sum(poke_choice_b_task_x[:start_task_2_index_for_pokes]) > 0:
        poke_b_task_1 = poke_choice_b_task_x
    elif sum(poke_choice_b_task_x[start_task_2_index_for_pokes:start_task_3_index_for_pokes]) > 0:
        poke_b_task_2 = poke_choice_b_task_x
    elif sum(poke_choice_b_task_x[start_task_3_index_for_pokes:]) > 0:
        poke_b_task_3 = poke_choice_b_task_x

  # Check in which task it was B port 
    if sum(poke_4[:start_task_2_index_for_pokes]) > 0:
        poke_b_task_1 = poke_4
    elif sum(poke_4[start_task_2_index_for_pokes:start_task_3_index_for_pokes]) > 0:
        poke_b_task_2 = poke_4
    elif sum(poke_4[start_task_3_index_for_pokes:]) > 0:
        poke_b_task_3 = poke_4

  # Check in which task it was B port 
    if sum(poke_5[:start_task_2_index_for_pokes]) > 0:
        poke_b_task_1 = poke_5
    elif sum(poke_5[start_task_2_index_for_pokes:start_task_3_index_for_pokes]) > 0:
        poke_b_task_2 = poke_5
    elif sum(poke_5[start_task_3_index_for_pokes:]) > 0:
        poke_b_task_3 = poke_5
        
        
    return poke_a_task_x,poke_a_task_y,poke_a_task_z,poke_initiation_task_1,poke_initiation_task_2,poke_initiation_task_3,poke_b_task_1,poke_b_task_2,poke_b_task_3, outcomes

def extract_trials_pokes_task_arranged(experiment, all_sessions):
    session_list_poke_a_task_x_r_spikes_1 = []
    session_list_poke_a_task_x_nr_spikes_1 = []
    
    session_list_poke_a_task_y_r_spikes_1 = []
    session_list_poke_a_task_y_nr_spikes_1 = []

    session_list_poke_a_task_z_r_spikes_1 = []
    session_list_poke_a_task_z_nr_spikes_1 = []

    session_list_poke_initiation_task_x_spikes_1 = []
    session_list_poke_initiation_task_y_spikes_1 = []
   
    session_list_poke_initiation_b_task_z_spikes_1 = []
   
    session_list_poke_choice_b_task_x_spikes_r_1 = []
    session_list_poke_choice_b_task_x_spikes_nr_1 = []
   
    session_list_poke_b_task_y_spikes_r_1 = []
    session_list_poke_b_task_y_spikes_nr_1 = []
   
    session_list_poke_b_task_z_spikes_r_1 = []
    session_list_poke_b_task_z_spikes_nr_1 = []
    
    
    #########
    
    session_list_poke_a_task_x_r_spikes_2 = []
    session_list_poke_a_task_x_nr_spikes_2 = []
    
    session_list_poke_a_task_y_r_spikes_2 = []
    session_list_poke_a_task_y_nr_spikes_2 = []

    session_list_poke_a_task_z_r_spikes_2 = []
    session_list_poke_a_task_z_nr_spikes_2 = []

    session_list_poke_initiation_task_x_spikes_2 = []
    session_list_poke_initiation_task_y_spikes_2 = []
   
    session_list_poke_initiation_b_task_z_spikes_2 = []
   
    session_list_poke_choice_b_task_x_spikes_r_2 = []
    session_list_poke_choice_b_task_x_spikes_nr_2 = []
   
    session_list_poke_b_task_y_spikes_r_2 = []
    session_list_poke_b_task_y_spikes_nr_2 = []
   
    session_list_poke_b_task_z_spikes_r_2 = []
    session_list_poke_b_task_z_spikes_nr_2 = []
    
    for s,session in enumerate(experiment):
        session = experiment[s]
        all_neurons_all_spikes_raster_plot_task = all_sessions[s]
        all_neurons_all_spikes_raster_plot_task = np.asarray(all_neurons_all_spikes_raster_plot_task)
        all_neurons_all_spikes_raster_plot_task = all_neurons_all_spikes_raster_plot_task[:,:,30:60]
        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            poke_a_task_x,poke_a_task_y,poke_a_task_z,poke_initiation_task_1,poke_initiation_task_2,poke_initiation_task_3,poke_b_task_1,poke_b_task_2,poke_b_task_3, outcomes = arrange_x_y_z_tasks(session)

         # Get rid off the time dimension        
            #average_time_spikes = np.mean(all_neurons_all_spikes_raster_plot_task, axis = 2)
            n_trials, n_neurons, n_timepoints = all_neurons_all_spikes_raster_plot_task.shape   
            #average_time_spikes = zscore(all_neurons_all_spikes_raster_plot_task,axis = 0)
            average_time_spikes = all_neurons_all_spikes_raster_plot_task

            # The z-scores of input "a", with any columns including non-finite
            # numbers replaced by all zeros.
            #average_time_spikes[:, np.logical_not(np.all(np.isfinite(average_time_spikes), axis=0))] = 0

           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_a_task_x_r_spikes = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 1)), :,:]
            poke_a_task_x_nr_spikes  = average_time_spikes[np.where((poke_a_task_x == 1) & (outcomes == 0)),:,:]
           
            poke_a_task_y_r_spikes = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 1)),:,:]
            poke_a_task_y_nr_spikes  = average_time_spikes[np.where((poke_a_task_y == 1) & (outcomes == 0)),:,:]
           
            poke_a_task_z_r_spikes = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 1)),:,:]
            poke_a_task_z_nr_spikes  = average_time_spikes[np.where((poke_a_task_z == 1) & (outcomes == 0)),:,:]
           
            # Extract spikes for A in three tasks (rewarded and non-rewarded)
            poke_initiation_task_x_spikes = average_time_spikes[np.where(poke_initiation_task_1 == 1),:,:]
            
            poke_initiation_task_y_spikes = average_time_spikes[np.where(poke_initiation_task_2 == 1),:,:]

            poke_initiation_b_task_z_spikes = average_time_spikes[np.where(poke_initiation_task_3 ==1),:,:]
            
            poke_choice_b_task_x_spikes_r = average_time_spikes[np.where((poke_b_task_1 == 1) & (outcomes == 1)),:,:]
            poke_choice_b_task_x_spikes_nr = average_time_spikes[np.where((poke_b_task_1 == 1) & (outcomes == 0)),:,:]

            poke_b_task_y_spikes_r = average_time_spikes[np.where((poke_b_task_2 ==1) & (outcomes == 1)),:,:]
            poke_b_task_y_spikes_nr = average_time_spikes[np.where((poke_b_task_2 ==1) & (outcomes == 0)),:,:]
            
            poke_b_task_z_spikes_r = average_time_spikes[np.where((poke_b_task_3 ==1) & (outcomes == 1)),:,:]
            poke_b_task_z_spikes_nr = average_time_spikes[np.where((poke_b_task_3 ==1) & (outcomes == 0)),:,:]
            
            #Find mean firing rates for each neuron on each type of trial split
           
         
            mean_poke_a_task_x_r_spikes_1 = np.mean(poke_a_task_x_r_spikes[0,:int(poke_a_task_x_r_spikes.shape[1]/2),:], axis = 0)
            mean_poke_a_task_x_r_spikes_2 = np.mean(poke_a_task_x_r_spikes[0,int(poke_a_task_x_r_spikes.shape[1]/2):,:], axis = 0)

            mean_poke_a_task_x_nr_spikes_1 = np.mean(poke_a_task_x_nr_spikes[0,:int(poke_a_task_x_nr_spikes.shape[1]/2),:], axis = 0)
            mean_poke_a_task_x_nr_spikes_2 = np.mean(poke_a_task_x_nr_spikes[0,int(poke_a_task_x_nr_spikes.shape[1]/2):,:], axis = 0)

            mean_poke_a_task_y_r_spikes_1 = np.mean(poke_a_task_y_r_spikes[0,:int(poke_a_task_y_r_spikes.shape[1]/2),:], axis = 0)
            mean_poke_a_task_y_r_spikes_2 = np.mean(poke_a_task_y_r_spikes[0,int(poke_a_task_y_r_spikes.shape[1]/2):,:], axis = 0)

            mean_poke_a_task_y_nr_spikes_1 = np.mean(poke_a_task_y_nr_spikes[0,:int(poke_a_task_y_nr_spikes.shape[1]/2),:], axis = 0)
            mean_poke_a_task_y_nr_spikes_2 = np.mean(poke_a_task_y_nr_spikes[0,int(poke_a_task_y_nr_spikes.shape[1]/2):,:], axis = 0)

            mean_poke_a_task_z_r_spikes_1 = np.mean(poke_a_task_z_r_spikes[0,:int(poke_a_task_z_r_spikes.shape[1]/2),:], axis = 0)
            mean_poke_a_task_z_r_spikes_2 = np.mean(poke_a_task_z_r_spikes[0,int(poke_a_task_z_r_spikes.shape[1]/2):,:], axis = 0)

            mean_poke_a_task_z_nr_spikes_1 = np.mean(poke_a_task_z_nr_spikes[0,:int(poke_a_task_z_nr_spikes.shape[1]/2),:], axis = 0)
            mean_poke_a_task_z_nr_spikes_2 = np.mean(poke_a_task_z_nr_spikes[0,int(poke_a_task_z_nr_spikes.shape[1]/2):,:], axis = 0)

          
            mean_poke_initiation_task_x_spikes_1 = np.mean(poke_initiation_task_x_spikes[0,:int(poke_initiation_task_x_spikes.shape[1]/2),:], axis = 0)
            mean_poke_initiation_task_x_spikes_2 = np.mean(poke_initiation_task_x_spikes[0,int(poke_initiation_task_x_spikes.shape[1]/2):,:], axis = 0)
            
            mean_poke_initiation_task_y_spikes_1 = np.mean(poke_initiation_task_y_spikes[0,:int(poke_initiation_task_y_spikes.shape[1]/2),:], axis = 0)
            mean_poke_initiation_task_y_spikes_2 = np.mean(poke_initiation_task_y_spikes[0,int(poke_initiation_task_y_spikes.shape[1]/2):,:], axis = 0)

            mean_poke_initiation_b_task_z_spikes_1 = np.mean(poke_initiation_b_task_z_spikes[0,:int(poke_initiation_b_task_z_spikes.shape[1]/2),:], axis = 0)
            mean_poke_initiation_b_task_z_spikes_2 = np.mean(poke_initiation_b_task_z_spikes[0,int(poke_initiation_b_task_z_spikes.shape[1]/2):,:], axis = 0)

           
            mean_poke_choice_b_task_x_spikes_r_1 = np.mean(poke_choice_b_task_x_spikes_r[0,:int(poke_choice_b_task_x_spikes_r.shape[1]/2),:], axis = 0)
            mean_poke_choice_b_task_x_spikes_r_2 = np.mean(poke_choice_b_task_x_spikes_r[0,:int(poke_choice_b_task_x_spikes_r.shape[1]/2),:], axis = 0)

            mean_poke_choice_b_task_x_spikes_nr_1 = np.mean(poke_choice_b_task_x_spikes_nr[0,:int(poke_choice_b_task_x_spikes_nr.shape[1]/2),:], axis = 0)
            mean_poke_choice_b_task_x_spikes_nr_2 = np.mean(poke_choice_b_task_x_spikes_nr[0,:int(poke_choice_b_task_x_spikes_nr.shape[1]/2),:], axis = 0)

            mean_poke_b_task_y_spikes_r_1 = np.mean(poke_b_task_y_spikes_r[0,:int(poke_b_task_y_spikes_r.shape[1]/2),:], axis = 0)
            mean_poke_b_task_y_spikes_r_2 = np.mean(poke_b_task_y_spikes_r[0,:int(poke_b_task_y_spikes_r.shape[1]/2),:], axis = 0)

            mean_poke_b_task_y_spikes_nr_1 = np.mean(poke_b_task_y_spikes_nr[0,:int(poke_b_task_y_spikes_nr.shape[1]/2),:], axis = 0)
            mean_poke_b_task_y_spikes_nr_2 = np.mean(poke_b_task_y_spikes_nr[0,:int(poke_b_task_y_spikes_nr.shape[1]/2),:], axis = 0)

            mean_poke_b_task_z_spikes_r_1 = np.mean(poke_b_task_z_spikes_r[0,:int(poke_b_task_z_spikes_r.shape[1]/2),:], axis = 0)
            mean_poke_b_task_z_spikes_r_2 = np.mean(poke_b_task_z_spikes_r[0,:int(poke_b_task_z_spikes_r.shape[1]/2),:], axis = 0)

            mean_poke_b_task_z_spikes_nr_1 = np.mean(poke_b_task_z_spikes_nr[0,:int(poke_b_task_z_spikes_nr.shape[1]/2),:], axis = 0)
            mean_poke_b_task_z_spikes_nr_2 = np.mean(poke_b_task_z_spikes_nr[0,:int(poke_b_task_z_spikes_nr.shape[1]/2),:], axis = 0)

                        
            # First Half

            session_list_poke_a_task_x_r_spikes_1.append(mean_poke_a_task_x_r_spikes_1)
            session_list_poke_a_task_x_nr_spikes_1.append(mean_poke_a_task_x_nr_spikes_1)
        
            session_list_poke_a_task_y_r_spikes_1.append(mean_poke_a_task_y_r_spikes_1)
            session_list_poke_a_task_y_nr_spikes_1.append(mean_poke_a_task_y_nr_spikes_1)
         
            session_list_poke_a_task_z_r_spikes_1.append(mean_poke_a_task_z_r_spikes_1)
            session_list_poke_a_task_z_nr_spikes_1.append(mean_poke_a_task_z_nr_spikes_1)
        
            session_list_poke_initiation_task_x_spikes_1.append(mean_poke_initiation_task_x_spikes_1)
            session_list_poke_initiation_task_y_spikes_1.append(mean_poke_initiation_task_y_spikes_1)
           
            session_list_poke_initiation_b_task_z_spikes_1.append(mean_poke_initiation_b_task_z_spikes_1)
            
            session_list_poke_choice_b_task_x_spikes_r_1.append(mean_poke_choice_b_task_x_spikes_r_1)
            session_list_poke_choice_b_task_x_spikes_nr_1.append(mean_poke_choice_b_task_x_spikes_nr_1)
           
            session_list_poke_b_task_y_spikes_r_1.append(mean_poke_b_task_y_spikes_r_1)
            session_list_poke_b_task_y_spikes_nr_1.append(mean_poke_b_task_y_spikes_nr_1)
            
            session_list_poke_b_task_z_spikes_r_1.append(mean_poke_b_task_z_spikes_r_1)
            session_list_poke_b_task_z_spikes_nr_1.append(mean_poke_b_task_z_spikes_nr_1)
            
            # Second Half
            
            session_list_poke_a_task_x_r_spikes_2.append(mean_poke_a_task_x_r_spikes_2)
            session_list_poke_a_task_x_nr_spikes_2.append(mean_poke_a_task_x_nr_spikes_2)
        
            session_list_poke_a_task_y_r_spikes_2.append(mean_poke_a_task_y_r_spikes_2)
            session_list_poke_a_task_y_nr_spikes_2.append(mean_poke_a_task_y_nr_spikes_2)
         
            session_list_poke_a_task_z_r_spikes_2.append(mean_poke_a_task_z_r_spikes_2)
            session_list_poke_a_task_z_nr_spikes_2.append(mean_poke_a_task_z_nr_spikes_2)
        
            session_list_poke_initiation_task_x_spikes_2.append(mean_poke_initiation_task_x_spikes_2)
            session_list_poke_initiation_task_y_spikes_2.append(mean_poke_initiation_task_y_spikes_2)
           
            session_list_poke_initiation_b_task_z_spikes_2.append(mean_poke_initiation_b_task_z_spikes_2)
            
            session_list_poke_choice_b_task_x_spikes_r_2.append(mean_poke_choice_b_task_x_spikes_r_2)
            session_list_poke_choice_b_task_x_spikes_nr_2.append(mean_poke_choice_b_task_x_spikes_nr_2)
           
            session_list_poke_b_task_y_spikes_r_2.append(mean_poke_b_task_y_spikes_r_2)
            session_list_poke_b_task_y_spikes_nr_2.append(mean_poke_b_task_y_spikes_nr_2)
            
            session_list_poke_b_task_z_spikes_r_2.append(mean_poke_b_task_z_spikes_r_2)
            session_list_poke_b_task_z_spikes_nr_2.append(mean_poke_b_task_z_spikes_nr_2)
            
                      
    #################### First Half          
    session_list_poke_a_task_x_r_spikes_1 = np.concatenate(session_list_poke_a_task_x_r_spikes_1,0)
    session_list_poke_a_task_x_nr_spikes_1 = np.concatenate(session_list_poke_a_task_x_nr_spikes_1,0)
        
    session_list_poke_a_task_y_r_spikes_1 = np.concatenate(session_list_poke_a_task_y_r_spikes_1,0)
    session_list_poke_a_task_y_nr_spikes_1 = np.concatenate(session_list_poke_a_task_y_nr_spikes_1,0)
     
    session_list_poke_a_task_z_r_spikes_1 = np.concatenate(session_list_poke_a_task_z_r_spikes_1,0)
    session_list_poke_a_task_z_nr_spikes_1 = np.concatenate(session_list_poke_a_task_z_nr_spikes_1,0)
    
    session_list_poke_initiation_task_x_spikes_1 = np.concatenate(session_list_poke_initiation_task_x_spikes_1,0)
    session_list_poke_initiation_task_y_spikes_1 = np.concatenate(session_list_poke_initiation_task_y_spikes_1,0) 
       
    session_list_poke_initiation_b_task_z_spikes_1 = np.concatenate(session_list_poke_initiation_b_task_z_spikes_1,0)
       
    session_list_poke_choice_b_task_x_spikes_r_1 = np.concatenate(session_list_poke_choice_b_task_x_spikes_r_1,0)
    session_list_poke_choice_b_task_x_spikes_nr_1 = np.concatenate(session_list_poke_choice_b_task_x_spikes_nr_1,0)
    
    session_list_poke_b_task_y_spikes_r_1 = np.concatenate(session_list_poke_b_task_y_spikes_r_1,0)
    session_list_poke_b_task_y_spikes_nr_1 = np.concatenate(session_list_poke_b_task_y_spikes_nr_1,0)
    
    session_list_poke_b_task_z_spikes_r_1 = np.concatenate(session_list_poke_b_task_z_spikes_r_1,0)
    session_list_poke_b_task_z_spikes_nr_1  = np.concatenate(session_list_poke_b_task_z_spikes_nr_1,0)
   
    matrix_for_correlations_task_1_1 = np.concatenate([session_list_poke_a_task_x_r_spikes_1,session_list_poke_a_task_x_nr_spikes_1,\
                                                     session_list_poke_initiation_task_x_spikes_1,session_list_poke_choice_b_task_x_spikes_r_1,\
                                                     session_list_poke_choice_b_task_x_spikes_nr_1],axis = 1)
    matrix_for_correlations_task_2_1 = np.concatenate([session_list_poke_a_task_y_r_spikes_1,session_list_poke_a_task_y_nr_spikes_1,\
                                                     session_list_poke_initiation_task_y_spikes_1,
                                                     session_list_poke_b_task_y_spikes_r_1,\
                                                     session_list_poke_b_task_y_spikes_nr_1],axis = 1)
    matrix_for_correlations_task_3_1 = np.concatenate([session_list_poke_a_task_z_r_spikes_1,session_list_poke_a_task_z_nr_spikes_1,\
                                                     session_list_poke_initiation_b_task_z_spikes_1,
                                                     session_list_poke_b_task_z_spikes_r_1,\
                                                     session_list_poke_b_task_z_spikes_nr_1],axis = 1)
   #################### Second Half 
    session_list_poke_a_task_x_r_spikes_2 = np.concatenate(session_list_poke_a_task_x_r_spikes_2,0)
    session_list_poke_a_task_x_nr_spikes_2 = np.concatenate(session_list_poke_a_task_x_nr_spikes_2,0)
        
    session_list_poke_a_task_y_r_spikes_2 = np.concatenate(session_list_poke_a_task_y_r_spikes_2,0)
    session_list_poke_a_task_y_nr_spikes_2 = np.concatenate(session_list_poke_a_task_y_nr_spikes_2,0)
     
    session_list_poke_a_task_z_r_spikes_2 = np.concatenate(session_list_poke_a_task_z_r_spikes_2,0)
    session_list_poke_a_task_z_nr_spikes_2 = np.concatenate(session_list_poke_a_task_z_nr_spikes_2,0)
    
    session_list_poke_initiation_task_x_spikes_2 = np.concatenate(session_list_poke_initiation_task_x_spikes_2,0)
    session_list_poke_initiation_task_y_spikes_2 = np.concatenate(session_list_poke_initiation_task_y_spikes_2,0) 
       
    session_list_poke_initiation_b_task_z_spikes_2 = np.concatenate(session_list_poke_initiation_b_task_z_spikes_2,0)
       
    session_list_poke_choice_b_task_x_spikes_r_2 = np.concatenate(session_list_poke_choice_b_task_x_spikes_r_2,0)
    session_list_poke_choice_b_task_x_spikes_nr_2 = np.concatenate(session_list_poke_choice_b_task_x_spikes_nr_2,0)
    
    session_list_poke_b_task_y_spikes_r_2 = np.concatenate(session_list_poke_b_task_y_spikes_r_2,0)
    session_list_poke_b_task_y_spikes_nr_2 = np.concatenate(session_list_poke_b_task_y_spikes_nr_2,0)
    
    session_list_poke_b_task_z_spikes_r_2 = np.concatenate(session_list_poke_b_task_z_spikes_r_2,0)
    session_list_poke_b_task_z_spikes_nr_2  = np.concatenate(session_list_poke_b_task_z_spikes_nr_2,0)
   
    matrix_for_correlations_task_1_2 = np.concatenate([session_list_poke_a_task_x_r_spikes_2,session_list_poke_a_task_x_nr_spikes_2,\
                                                     session_list_poke_initiation_task_x_spikes_2,
                                                     session_list_poke_choice_b_task_x_spikes_r_2,\
                                                     session_list_poke_choice_b_task_x_spikes_nr_2],axis = 1)
    matrix_for_correlations_task_2_2 = np.concatenate([session_list_poke_a_task_y_r_spikes_2,session_list_poke_a_task_y_nr_spikes_2,\
                                                     session_list_poke_initiation_task_y_spikes_2,
                                                     session_list_poke_b_task_y_spikes_r_2,\
                                                     session_list_poke_b_task_y_spikes_nr_2],axis = 1)
    matrix_for_correlations_task_3_2 = np.concatenate([session_list_poke_a_task_z_r_spikes_2,session_list_poke_a_task_z_nr_spikes_2,\
                                                     session_list_poke_initiation_b_task_z_spikes_2,
                                                     session_list_poke_b_task_z_spikes_r_2,\
                                                     session_list_poke_b_task_z_spikes_nr_2],axis = 1)
   
    return matrix_for_correlations_task_1_1,matrix_for_correlations_task_2_1,matrix_for_correlations_task_3_1,\
    matrix_for_correlations_task_1_2,matrix_for_correlations_task_2_2,matrix_for_correlations_task_3_2



def svd(experiment, all_sessions, diagonal = False, HP = False):
    
    flattened_all_clusters_task_1_first_half,flattened_all_clusters_task_2_first_half,flattened_all_clusters_task_3_first_half,\
    flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_second_half,flattened_all_clusters_task_3_second_half\
    = extract_trials_pokes_task_arranged(experiment, all_sessions)
    
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(flattened_all_clusters_task_1_first_half, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 first half
    u_t2_1, s_t2_1, vh_t2_1 = np.linalg.svd(flattened_all_clusters_task_2_first_half, full_matrices = True)
    
    #SVDsu.shape, s.shape, vh.shape for task 2 second half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = True)
 
    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_1, s_t3_1, vh_t3_1 = np.linalg.svd(flattened_all_clusters_task_3_first_half, full_matrices = True)

    #SVDsu.shape, s.shape, vh.shape for task 3 first half
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = True)

    #Finding variance explained in second half of task 1 using the Us and Vs from the first half

    t_u_t_1_2 = np.transpose(u_t1_2)   
    t_v_t_1_2 = np.transpose(vh_t1_2)  

    t_u_t_2_2 = np.transpose(u_t2_2)  
    t_v_t_2_2 = np.transpose(vh_t2_2)  

    t_u_t_3_2 = np.transpose(u_t3_2)
    t_v_t_3_2 = np.transpose(vh_t3_2)  

  
    #Compare task 2 First Half from task 1 Last Half 
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half, t_v_t_1_2])
    if diagonal == False:
        s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    else:
        s_2_1_from_t_1_2 = np.sum(s_task_2_1_from_t_1_2**2, axis = 1)
    sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/flattened_all_clusters_task_2_first_half.shape[0]

     #Compare task 2 Firs Half from second half
    s_task_2_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half, t_v_t_2_2])    
    if diagonal == False:
        s_2_1_from_t_2_2 = s_task_2_1_from_t_2_2.diagonal()
    else:
        s_2_1_from_t_2_2 = np.sum(s_task_2_1_from_t_2_2**2, axis = 1)
    sum_c_task_2_1_from_t_2_2 = np.cumsum(abs(s_2_1_from_t_2_2))/flattened_all_clusters_task_2_first_half.shape[0]


    
        
    #Compare task 3 First Half from Task 2 Last Half 
    s_task_3_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_3_first_half, t_v_t_2_2])
    if diagonal == False:
        s_3_1_from_t_2_2 = s_task_3_1_from_t_2_2.diagonal()
    else:
        s_3_1_from_t_2_2 = np.sum(s_task_3_1_from_t_2_2**2, axis = 1)
    sum_c_task_3_1_from_t_2_2 = np.cumsum(abs(s_3_1_from_t_2_2))/flattened_all_clusters_task_3_first_half.shape[0]


    s_task_3_1_from_t_3_2 = np.linalg.multi_dot([t_u_t_3_2, flattened_all_clusters_task_3_first_half, t_v_t_3_2])
    if diagonal == False:
        s_3_1_from_t_3_2 = s_task_3_1_from_t_3_2.diagonal()
    else:
        s_3_1_from_t_3_2 = np.sum(s_task_3_1_from_t_3_2**2, axis = 1)
    sum_c_task_3_1_from_t_3_2 = np.cumsum(abs(s_3_1_from_t_3_2))/flattened_all_clusters_task_3_first_half.shape[0]

    average_within = np.mean([sum_c_task_2_1_from_t_2_2, sum_c_task_3_1_from_t_3_2], axis = 0)
    average_between = np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_2_2], axis = 0)
 
    
    if HP == True:
        plt.plot(average_within, label = 'Within Tasks_HP', color='green')
        plt.plot(average_between, label = 'Between Tasks_HP',linestyle = '--', color = 'green')
        
    if HP == False:
        plt.plot(average_within, label = 'Within Tasks_PFC', color='black')
        plt.plot(average_between, label = 'Between Tasks_PFC',linestyle = '--', color = 'black')
      
    plt.title('SVD')
    plt.legend()
    


def plotting_correlations_all_time_points(experiment,all_sessions, figure_number = 1):

    matrix_for_correlations = extract_trials_all_time_points(experiment,all_sessions)
    corr_m = np.corrcoef(np.transpose(matrix_for_correlations))
    v,n = np.linalg.eig(corr_m)
    ticks_n = np.linspace(0, 450, 15)
    
    for i in range(len(v)):
        if i < 16:
            fig = plt.subplot(4, 4, i+1)
            fig.plot(np.real(n[:,i]))
            plt.xticks(ticks_n, ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
                       '1 A T3 R','1 A T3 NR', ' 2 I T1',\
                       '2 I T2', '3 I T3', '3 B T1 R',\
                       '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')  
            
    plt.figure(figure_number)
    plt.imshow(corr_m)
    plt.xticks(ticks_n, ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
   
    plt.yticks(ticks_n, ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
    
    plt.colorbar()
    plt.tight_layout()
    
    

  
#HP_m = extract_trials(experiment_aligned_HP, all_sessions_HP, time_window = 30)

def regression_RSA(matrix_for_correlations):
# =============================================================================
#     Regression of RSA predictors on the actual data matrix
# =============================================================================
    C = [] 
    correlation_m = np.corrcoef(matrix_for_correlations)
    correlation_m_f = correlation_m.flatten()
    physical_rsa = rs.RSA_physical_rdm()
    physical_rsa  = 1*physical_rsa.flatten()
    choice_ab_rsa = rs.RSA_a_b_initiation_rdm()
    choice_ab_rsa = 1*choice_ab_rsa.flatten()
    reward_no_reward = rs.reward_rdm()
    reward_no_reward = 1*reward_no_reward.flatten()
    reward_at_choices = rs.reward_choice_space()
    reward_at_choices = 1*reward_at_choices.flatten()
    #choice_initiation_rsa =  rsa.choice_vs_initiation()
    #choice_initiation_rsa = 1*choice_initiation_rsa.flatten()
    #a_bs_task_specific_rsa = rsa.a_bs_task_specific()
    #a_bs_task_specific_rsa = 1*a_bs_task_specific_rsa.flatten()

    ones = np.ones(len(choice_ab_rsa))
    

    predictors = OrderedDict([('Space' , physical_rsa),
                              ('A vs B', choice_ab_rsa),
                              ('Reward',reward_no_reward),
                              ('Reward at A vs B',reward_at_choices),
                              #('Choice vs Initiation',choice_initiation_rsa),
                              #('A and B Task Specific',a_bs_task_specific_rsa),
                              ('constant', ones)])                                        
           
    X = np.vstack(predictors.values()).T[:len(physical_rsa),:].astype(float)
    # Check if regression is rank deficient 
    #print(X.shape[1])  
    rank = matrix_rank(X) 
    #print(rank)
    y = correlation_m_f
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    C.append(ols.coef_) 
    
    return C,correlation_m,predictors

def extract_trials(experiment, all_sessions, time_window = 1): 
# =============================================================================
#   This function separates pokes and finds firing rates of each neuron around the pokes.
#   Inputs are:
#        experiment_aligned_HP orexperiment_aligned_PFC,  all_sessions - made in poke_aligned_spikes script
#   whcih contains firing rates around every port entry. 
#       time_window - specifying which time window to use (default is of a width of 1 which is 50ms).
#       For the purposed of animation a different function --> matrices_for_plots(experiment,all_sessions) includes a loop
#       that goes through the full length of the poke aligned histogram (-1.5 to 1.5 sec);
#       this can be changed in poke_aligned_spikes.py script if necessary
#   Outputs are:
#       A in three tasks rewarded, unrewarded;
#       Initiation in three tasks x,y - same physical port, z - different port and also acts as B choice in x task
#       B choices in tasks y and z. These are combined into a big matrix which is the output of the function. 
# =============================================================================    

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
        average_time_spikes = all_neurons_all_spikes_raster_plot_task[:,:,time_window]
        if  all_neurons_all_spikes_raster_plot_task.shape[1] > 0: 
            
            poke_a_task_x,poke_a_task_y,poke_a_task_z,outcomes,poke_initiation_task_x,poke_initiation_task_y,poke_initiation_b_task_z,\
            poke_choice_b_task_x,poke_4,poke_5 = seperate_a_into_tasks(session) 
            
            # Get rid off the time dimension        
            #average_time_spikes = np.mean(all_neurons_all_spikes_raster_plot_task, axis = 2)
            n_trials, n_neurons = average_time_spikes.shape   
            average_time_spikes = zscore(average_time_spikes,axis = 0)

            # The z-scores of input "a", with any columns including non-finite
            # numbers replaced by all zeros.
            average_time_spikes[:, np.logical_not(np.all(np.isfinite(average_time_spikes), axis=0))] = 0

           
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
   
   #matrix_for_correlations = np.vstack([session_list_poke_a_task_x_r_spikes,session_list_poke_a_task_x_nr_spikes,session_list_poke_a_task_y_r_spikes,\
   #                                    session_list_poke_a_task_y_nr_spikes,session_list_poke_a_task_z_r_spikes,session_list_poke_a_task_z_nr_spikes,\
   #                                    session_list_poke_initiation_task_x_spikes,session_list_poke_initiation_task_y_spikes,\
   #                                    session_list_poke_initiation_b_task_z_spikes,session_list_poke_choice_b_task_x_spikes_r,\
   #                                   session_list_poke_choice_b_task_x_spikes_nr,session_list_poke_b_task_y_spikes_r,\
   #                                    session_list_poke_b_task_y_spikes_nr,session_list_poke_b_task_z_spikes_r,session_list_poke_b_task_z_spikes_nr])
   
   matrix_for_correlations = np.vstack([session_list_poke_a_task_x_r_spikes,session_list_poke_a_task_x_nr_spikes,session_list_poke_a_task_y_r_spikes,\
                                       session_list_poke_a_task_y_nr_spikes,session_list_poke_a_task_z_r_spikes,session_list_poke_a_task_z_nr_spikes,\
                                       session_list_poke_choice_b_task_x_spikes_r,\
                                       session_list_poke_choice_b_task_x_spikes_nr,session_list_poke_b_task_y_spikes_r,\
                                       session_list_poke_b_task_y_spikes_nr,session_list_poke_b_task_z_spikes_r,session_list_poke_b_task_z_spikes_nr])
   
   return matrix_for_correlations


def rsa_across_time(experiment,all_sessions):
    matrix_for_correlations = extract_trials(experiment_aligned_HP,all_sessions_HP,time_window = 38)
    C,correlation_m,predictors = regression_RSA(matrix_for_correlations)
    
    #plt.imshow(correlation_m)
    ### Predictor RSA Matrices 
    physical_rsa = rs.RSA_physical_rdm()
    choice_ab_rsa = rs.RSA_a_b_initiation_rdm()
    reward_no_reward = rs.reward_rdm() 
    reward_at_choices = rs.reward_choice_space()
    #choice_initiation_rsa =  rsa.choice_vs_initiation()
    #a_bs_task_specific_rsa = rsa.a_bs_task_specific()

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 25))
    grid = plt.GridSpec(12, 3, hspace=0.5, wspace=1)
    space_plt = fig.add_subplot(grid[1:3, 0])
    plt.yticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('Space')

    choice_plt = fig.add_subplot(grid[3:5, 0])
    plt.yticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('A vs B')

    reward_no_reward_plt = fig.add_subplot(grid[5:7, 0])
    plt.yticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('Reward')

    reward_at_choices_plt = fig.add_subplot(grid[7:9, 0])
    plt.yticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR')) 
    plt.xticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'),rotation = 'vertical' ) 
    
    plt.title('Reward at Choice')    
    
    len_C = range(len(C[0][:-1]))
    bar_plot = fig.add_subplot(grid[1:4, 1:3])
    plt.ylabel('Regression Coefficient')
    plt.xticks(len_C,('Space','A vs B','Reward','Reward at Choice'), rotation = 'vertical')
     
    trial_corr_plot = fig.add_subplot(grid[ 6:10, 1:3])
    plt.xticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
   
    plt.yticks(range(12), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
    
    bar_plot.bar(len_C,C[0][:-1])
    space_plt.imshow(physical_rsa,aspect = 'auto')
    choice_plt.imshow(choice_ab_rsa,aspect = 'auto')
    reward_no_reward_plt.imshow(reward_no_reward,aspect = 'auto')
    reward_at_choices_plt.imshow(reward_at_choices,aspect = 'auto')
    #choice_initiation_plt.imshow(choice_initiation_rsa,aspect = 'auto')
   # a_bs_task_specific_rsa_plt.imshow(a_bs_task_specific_rsa,aspect = 'auto')
    sh = trial_corr_plot.imshow(correlation_m, aspect = 1)
    plt.colorbar(sh)

    
def matrices_for_plots(experiment,all_sessions):
    #all_sessions_HP = pos.raster_plot_save(experiment_aligned_HP,time_window_start = 50, time_window_end = 110)
    #all_sessions_PFC = pos.raster_plot_save(experiment_aligned_PFC,time_window_start = 50, time_window_end = 110)
# =============================================================================
#   This is functionthat goes through the full length of the poke aligned histogram (-1.5 to 1.5 sec)
#   and finds the data matrix and computes regression coefficient for each data point around the poke

#   Default time window is 60 from -1500 to 1500 around poke
# =============================================================================
    
    C_list =  []
    correlation_m_list = []
    
    for i in range(60):
        matrix_for_correlations = extract_trials(experiment, all_sessions, time_window = i)
        C,correlation_m, predictors = regression_RSA(matrix_for_correlations)
        C_list.append(C)
        correlation_m_list.append(correlation_m)
    C_list = np.concatenate(C_list,0)
    
    return C_list,correlation_m_list

#C_list_PFC,correlation_m_list_PFC = matrices_for_plots(experiment_aligned_PFC,all_sessions_PFC)
#C_list_HP,correlation_m_list_HP = matrices_for_plots(experiment_aligned_HP,all_sessions_HP)

def matrices_for_different_times(C_list,correlation_m_list, HP = True):
# =============================================================================
#    This function plots predictor RSA matrices; and creates animations of 
#   the actual correlation matrix of the data at each data point between -1.5 to 1.5 seconds around the poke entry
# =============================================================================
    ### Predictor RSA Matrices 
    #physical_rsa = rsa.RSA_physical_rdm()
    #choice_ab_rsa = rsa.RSA_a_b_initiation_rdm()
    reward_no_reward = rsa.reward_rdm() 
    reward_at_choices = rsa.reward_choice_space()
    #choice_initiation_rsa =  rsa.choice_vs_initiation()
    #a_bs_task_specific_rsa = rsa.a_bs_task_specific()

    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 25))
    grid = plt.GridSpec(12, 4, hspace=0.5, wspace=1)
#    space_plt = fig.add_subplot(grid[0:2, 0])
#    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
#               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
#               '2 I T2', '3 I T3', '3 B T1 R',\
#               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
#    plt.xticks([])
#    plt.title('Space')
#
#    choice_plt = fig.add_subplot(grid[2:4, 0])
#    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
#               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
#               '2 I T2', '3 I T3', '3 B T1 R',\
#               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
#    plt.xticks([])
#    plt.title('A vs B')

    reward_no_reward_plt = fig.add_subplot(grid[3:6, 0:2])
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))
    plt.xticks([])
    plt.title('Reward')

    reward_at_choices_plt = fig.add_subplot(grid[6:9, 0:2])
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR')) 
    plt.xticks([])
    plt.title('Reward at Choice')

#    choice_initiation_plt = fig.add_subplot(grid[8:10, 0])
#    
#    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
#               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
#               '2 I T2', '3 I T3', '3 B T1 R',\
#               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
#    plt.xticks([])
#    plt.title('Choice vs Initiation')
#      
#    
#    a_bs_task_specific_rsa_plt = fig.add_subplot(grid[10:12, 0])
#    
#    plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
#               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
#               '2 I T2', '3 I T3', '3 B T1 R',\
#               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
#    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
#               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
#               '2 I T2', '3 I T3', '3 B T1 R',\
#               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
#    plt.title('As and Bs within Task')
#    
    C_list = C_list[:,:-1]
    len_C = range(C_list.shape[1])
    bar_plot = fig.add_subplot(grid[0:3, 2:3])
    plt.ylabel('Regression Coefficient')
    #plt.xticks(len_C,('Space','A vs B','Reward','Reward at Choice','Choice vs Initiation', 'A and Bs Task Specific','Constant'), rotation = 'vertical')
    plt.xticks(len_C,('Reward','Reward at Choice'), rotation = 'vertical')

    # Coordinates of the slider # 50 ms window 
    
    slider_plot  = fig.add_subplot(grid[10:11, 1:3])
    plt.yticks([])
    plt.xticks([0,10,20,30,40,50,60],['- 1.5s','- 1s', '- 0.5s', 'Poke Entry', '+ 0.5s', '1s', '+ 1.5s'])  
    
    trial_corr_plot = fig.add_subplot(grid[ 5:9, 2:4])
    plt.xticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'), rotation = 'vertical')
   
    plt.yticks(range(15), ('1 A T1 R', '1 A T1 NR','1 A T2 R', '1 A T2 NR',\
               '1 A T3 R','1 A T3 NR', ' 2 I T1',\
               '2 I T2', '3 I T3', '3 B T1 R',\
               '3 B T1 NR','4 B T2 R', '4 B T2 NR', '5 B T3 R', '5 B T3 NR'))  
    
    
    camera = Camera(fig)
    for i in range(C_list.shape[0]):
        slider_plot.vlines(30, ymin= 0, ymax = 1, color = 'red')
        slider_plot.vlines(i, ymin = 0 , ymax = 1,linewidth = 4)
        sh = trial_corr_plot.imshow(correlation_m_list[i], aspect = 1)
        bar_plot.bar(len_C,C_list[i,:])
        #space_plt.imshow(physical_rsa,aspect = 'auto')
        #choice_plt.imshow(choice_ab_rsa,aspect = 'auto')
        reward_no_reward_plt.imshow(reward_no_reward,aspect = 1)
        reward_at_choices_plt.imshow(reward_at_choices,aspect = 1)
        #choice_initiation_plt.imshow(choice_initiation_rsa,aspect = 'auto')
        #a_bs_task_specific_rsa_plt.imshow(a_bs_task_specific_rsa,aspect = 'auto')
        camera.snap()
        
    plt.colorbar(sh)

    animation = camera.animate(interval = 200)
    FFwriter = FFMpegWriter(fps = 1, bitrate=2000) 
    if HP == True:
        animation.save('/Users/veronikasamborska/Desktop/HP_rsa.mp4', writer=FFwriter)
    elif HP == False:
        animation.save('/Users/veronikasamborska/Desktop/PFC_rsa.mp4', writer=FFwriter)
   

    
   
    
