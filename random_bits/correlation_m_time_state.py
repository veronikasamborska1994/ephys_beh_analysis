#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:54:38 2020

@author: veronikasamborska
"""

from scipy import io
import numpy as np 



def load_mat_files():
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    Data_HP = HP['Data'][0]
    DM_HP = HP['DM'][0]
    Data_PFC = PFC['Data'][0]
    DM_PFC = PFC['DM'][0]
    
    return Data_HP,DM_HP,Data_PFC,DM_PFC

def correlation_matrices_time_block(Data,DM):
    ''' Design Matrix  =  ('latent_state',state),
                          ('choice',choices_forced_unforced),
                          ('reward', outcomes),
                          ('forced_trials',forced_trials),
                          ('block', block),
                          ('task',task),
                          ('A', a_pokes),
                          ('B', b_pokes),
                          ('Initiation', i_pokes),
                          ('Chosen_Simple_RW',chosen_Q1),
                          ('Chosen_Cross_learning_RW', chosen_Q4),
                          ('Value_A_RW', Q1_value_a),
                          ('Value_B_RW', Q1_value_b),
                          ('Value_A_Cross_learning', Q4_value_a),
                          ('ones', ones)'''

            
    
    for s,session in enumerate(Data):
        firing = np.mean(session,2)
        dm_session = DM[s]
        state = dm_session[:,0]
        choice = dm_session[:,1]
        reward = dm_session[:,2]
        forced = dm_session[:,3]
        block = dm_session[:,4]
        task = dm_session[:,5]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]
        task_3 = np.where(task == 3)[0]

        state_a = np.where(state == 1)[0]
        state_b = np.where(state == 0)[0]
        
        