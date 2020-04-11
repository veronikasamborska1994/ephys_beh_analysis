#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:13:24 2019

@author: veronikasamborska
"""
from scipy.ndimage import gaussian_filter
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt



def svd_on_trials():
    all_hp = []
    for data in HP['Data'][0]:
        data_array = np.asarray(data).reshape(data.shape[0]*data.shape[1],data.shape[2])
        all_hp.append(data_array)
    all_hp_ar = np.concatenate(all_hp, 0)
    u, s, v = np.linalg.svd(all_hp_ar, full_matrices = False)

   
    fig = plt.figure()
    for i in range(15):
        fig.add_subplot(5, 4, i+1)
        plt.plot(v[i], color = 'red')
        
        
        
        
def svd_all_trials():
    all_hp = []
    for data in HP['Data'][0]:
        data_array = np.asarray(data).reshape(data.shape[0]*data.shape[1],data.shape[2])
        all_hp.append(data_array)
    all_hp_ar = np.concatenate(all_hp, 0)
    u, s, v = np.linalg.svd(all_hp_ar, full_matrices = False)

    all_pfc = []
    for data in PFC['Data'][0]:
        data_array = np.asarray(data).reshape(data.shape[0]*data.shape[1],data.shape[2])
        all_pfc.append(data_array)
    all_pfc_ar = np.concatenate(all_pfc, 0)
    u_pfc, s_pfc, v_pfc = np.linalg.svd(all_pfc_ar, full_matrices = False)
    
    fig = plt.figure()
    for i in range(15):
        fig.add_subplot(5, 4, i+1)
        plt.plot(v_pfc[i], color = 'black')
        plt.plot(v[i], color = 'red')

    
def projection_trial_by_trial(session_list,a_is_like_b = True):
    
    #session_list_PFC = np.transpose(session_list_PFC,[0,2,1])
    n_blocks, n_neurons, n_trials = session_list.shape
    
    # Make Place Cell like Matrix
    DD = np.identity(n_trials)
    smoothed_positive_corr =  gaussian_filter(DD,2, mode = 'wrap')    
    
    all_blocks = np.zeros((n_trials*4, n_trials*4))
    all_blocks[:n_trials,:n_trials] = smoothed_positive_corr
    
    all_blocks[n_trials:(2*n_trials),n_trials:(2*n_trials)] = smoothed_positive_corr
    
    all_blocks[(n_trials*2):(3*n_trials),(n_trials*2):(3*n_trials)] = smoothed_positive_corr
    all_blocks[(n_trials*3):(4*n_trials),(n_trials*2):(3*n_trials)] = smoothed_positive_corr
    
    all_blocks[(n_trials*3):(4*n_trials),(n_trials*3):(4*n_trials)] = smoothed_positive_corr
    all_blocks[(n_trials*2):(3*n_trials), (n_trials*3):(4*n_trials)] = smoothed_positive_corr
    
    all_blocks[:n_trials,n_trials:(2*n_trials)] = smoothed_positive_corr
    all_blocks[n_trials:(2*n_trials), :n_trials] = smoothed_positive_corr
    
    
    
    if a_is_like_b  == False:
    
        all_blocks[:n_trials,n_trials:(2*n_trials)] = -smoothed_positive_corr
        
        all_blocks[n_trials:(2*n_trials), :n_trials] = -smoothed_positive_corr
        
    
        all_blocks[(n_trials*2):(3*n_trials), n_trials*3:] = -smoothed_positive_corr
        all_blocks[ n_trials*3:, (n_trials*2):(3*n_trials)] = -smoothed_positive_corr
        
    
    matrix_for_projection =  np.zeros((n_trials*8, n_trials*8))
    matrix_for_projection[:n_trials*4,:n_trials*4] = all_blocks
    matrix_for_projection[n_trials*4:,n_trials*4:] = all_blocks
    
    D,V = la.eig(matrix_for_projection)
    diag_D = np.diag(D)
    
    #a_1 = session_list_PFC[0,:,:]
    #a_2 = session_list_PFC[1,:,:]
    #
    #b_1 = session_list_PFC[2,:,:]
    #b_2 = session_list_PFC[3,:,:]
    #
    #a_3 = session_list_PFC[4,:,:]
    #a_4 = session_list_PFC[5,:,:]
    #
    #b_3 = session_list_PFC[6,:,:]
    #b_4 = session_list_PFC[7,:,:]
    
    #first_4_abab =  np.stack([a_1,b_1,  a_2, b_2, a_3, b_3, a_4, b_4])
    first_4_abab = session_list[:8,:,:]
    reshape_session_list = np.transpose(first_4_abab,[2,0,1])
    fl_session_list = np.concatenate(reshape_session_list,axis = 0)
    rand = np.random.random((fl_session_list.shape[0], fl_session_list.shape[1]))
    rand_1 =  np.random.randn(reshape_session_list.shape[0], reshape_session_list.shape[1],reshape_session_list.shape[2])
    rand_1 = np.concatenate(rand_1,axis = 0)
    
    proj = np.real(np.matmul(np.matmul(np.matmul(V,np.sqrt(diag_D)),np.matmul(rand_1,rand_1.T)),np.matmul(np.sqrt(diag_D),V.T)))
    plt.figure()
    plt.imshow(proj)
