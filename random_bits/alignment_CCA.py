#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:53:25 2020

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/SVDs')

import remapping_count as rc 
import utility as ut
from scipy.sparse.linalg import svds
import latent_state_count as lt
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import regression_function as reg_f
import regressions as re
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from matplotlib import colors as mcolors
from scipy.spatial import procrustes as pr
import SVDs as sv

import hypertools as hyp

def import_data():
    Data_HP = data_HP['Data']
    DM_HP = data_HP['DM']
    Data_PFC = data_PFC['Data']
    DM_PFC = data_PFC['DM']
    



def procrustes(a,b, all_sessions_firing, all_session_dm):
        
    a = 0
    b = 1
    
    data_set_1 = all_sessions_firing[a]
    data_set_2 = all_sessions_firing[b]
    
    session_a = np.transpose(data_set_1,[1,0,2]).reshape(data_set_1.shape[1], data_set_1.shape[0]*data_set_1.shape[2])
    session_b = np.transpose(data_set_2,[1,0,2]).reshape(data_set_2.shape[1], data_set_2.shape[0]*data_set_2.shape[2])

     
    # PCA on the neuronal data with n dimensions 
    n = 8
    u_1, s_1, v_1 =svds(session_a, n) # u is n x m 
    u_2, s_2, v_2 =svds(session_b, n) 


    proj_a = np.linalg.multi_dot([u_1.T, session_a]) # project onto the m manifolds
    proj_b = np.linalg.multi_dot([u_2.T, session_b])
    data = [proj_a,proj_b]
    aligned_data = hyp.align(data)
    
    aligned_a = aligned_data[0].reshape(aligned_data[0].shape[0],data_set_1.shape[0],data_set_1.shape[2])
    aligned_b = aligned_data[1].reshape(aligned_data[1].shape[0],data_set_2.shape[0],data_set_2.shape[2])
    
    aligned_corr = np.corrcoef(aligned_a, aligned_b)
    misaligned_corr = np.corrcoef(proj_a, proj_b)

    #Hyperalignment between tasks 
    task_1 = aligned_b[:,:40,:]
    task_2 = aligned_b[:,40:80,:]
    task_1_reshaped = task_1.reshape(task_1.shape[0],task_1.shape[1]*task_1.shape[2])
    task_2_reshaped = task_2.reshape(task_2.shape[0],task_2.shape[1]*task_2.shape[2])
    tasks = [task_1_reshaped, task_2_reshaped]
    aligned_tasks = hyp.align(tasks)
    task_1_aligned  = aligned_tasks[0].reshape(task_1.shape[0],task_1.shape[1],task_1.shape[2])


    original_a = proj_a.reshape(aligned_data[0].shape[0],data_set_1.shape[0],data_set_1.shape[2])
    original_b = proj_b.reshape(aligned_data[1].shape[0],data_set_2.shape[0],data_set_2.shape[2])
    
    
    return aligned_a,aligned_b,original_a,original_b, task_1_aligned
     

def between_tasks(Data, DM, PFC = True):
    

    cpd_true = []
    C_true = []
    cpd_aligned = []
    C_aligned = []
    
    C_sq_true = []
    C_sq_aligned = []
    
    if PFC == True:
        ind_1 = np.arange(0,26)
        ind_2 =  np.arange(1,27)
        ind_a = np.hstack((ind_1,ind_2))
        ind_b = np.hstack((ind_2,ind_1))
    
    else:
        ind_1 = np.arange(0,15)
        ind_2 =  np.arange(1,16)
        ind_a = np.hstack((ind_1,ind_2))
        ind_b = np.hstack((ind_2,ind_1))
    
   
    all_sessions_firing, all_session_dm = select_trials(Data, DM, 10)

    for a,b in zip(ind_a, ind_b):
        aligned_a, aligned_b, original_a, original_b, task_1_aligned = procrustes(a,b,all_sessions_firing,all_session_dm)
        
          
        
        target =  np.transpose(task_1_aligned, [1,0,2])
        source = np.transpose(original_a, [1,0,2])[40:80,:,:]
        
        #  Session 1 Task 2
        dm_test  = all_session_dm[a]

        trials ,n_neurons, n_timepoints = source.shape
   
        reward_test = dm_test[:,2][40:80]
        state_test = dm_test[:,0][40:80]        
        choices_test = dm_test[:,1][40:80]
        ones_test = np.ones(len(choices_test))
         
        reward_choice = choices_test*reward_test      
      
        # Aligned Task 2 from Task 1 
        predictors_train = OrderedDict([('State', state_test),
                     ('Reward', reward_test),
                     ('Choice', choices_test),
                     ('Reward Choice Int', reward_choice),
                     ('ones', ones_test)])
          
        X = np.vstack(predictors_train.values()).T[:trials,:].astype(float)
        n_predictors = X.shape[1]
        y = source.reshape([len(source),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_true.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_true.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_true.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
          
                   
        # Aligned Using Neurons        
        n_predictors = X.shape[1]
        y = target.reshape([len(target),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_aligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_aligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_aligned.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
    
    cpd_true = np.nanmean(np.concatenate(cpd_true,0), axis = 0)
    C_sq_true = np.concatenate(C_sq_true,1)
    
    
    cpd_aligned = np.nanmean(np.concatenate(cpd_aligned,0), axis = 0)
    C_sq_aligned = np.concatenate(C_sq_aligned,1)
    
    
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors_train]
    
    plt.figure()
    
    for i in np.arange(cpd_true.shape[1]-1):
        plt.plot(cpd_true[:,i], label =p[i] + 'Real', color = c[i])
        plt.plot(cpd_aligned[:,i], label =p[i] + 'Aligned', color = c[i], linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    
    return cpd_aligned,cpd_true
   


def regression_hyperalignment(Data, DM, PFC = True):

    cpd_true = []
    C_true = []
    cpd_misaligned = []
    C_misaligned = []
    cpd_aligned = []
    C_aligned = []
    
    C_sq_true = []
    C_sq_misaligned = []
    C_sq_aligned = []
    
    if PFC == True:
        ind_1 = np.arange(0,26)
        ind_2 =  np.arange(1,27)
        ind_a = np.hstack((ind_1,ind_2))
        ind_b = np.hstack((ind_2,ind_1))
    
    else:
        ind_1 = np.arange(0,15)
        ind_2 =  np.arange(1,16)
        ind_a = np.hstack((ind_1,ind_2))
        ind_b = np.hstack((ind_2,ind_1))
    
    all_sessions_firing, all_session_dm = select_trials(Data, DM, 10)

    for a,b in zip(ind_a, ind_b):
        aligned_a, aligned_b, original_a, original_b, task_1_aligned = procrustes(a,b,all_sessions_firing,all_session_dm)
        
          
        dm_test  = all_session_dm[a]
        
        session_training =  np.transpose(original_a, [1,0,2])
        session_misaligned = np.transpose(original_b, [1,0,2])
        session_aligned =   np.transpose(aligned_b, [1,0,2])
        
        # c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
        #   'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
 
        # for i in range(0,5):
        #     plt.plot(session_training[i,0,:],color = c[i], linestyle =':')
        #     plt.plot(session_aligned[i,0,:],color = c[i])

        trials ,n_neurons, n_timepoints = session_training.shape
        
        
        
        reward_test = dm_test[:,2]
        state_test = dm_test[:,0]        
        choices_test = dm_test[:,1]
        ones_test = np.ones(len(choices_test))
        
        trials_since_block = np.arange(0,10)
        trials_since_block = np.tile(trials_since_block,12)
           
        reward_choice = choices_test*reward_test      
        trial_sq = (np.asarray(trials_since_block)-0.5)**2
        choice_trials_sq = choices_test*trial_sq
        interaction_trials_choice = trials_since_block*choices_test

     

        # Original Using Neurons
        predictors_train = OrderedDict([('State', state_test),
                     ('Reward', reward_test),
                     ('Choice', choices_test),
                     ('Reward Choice Int', reward_choice),
                     ('Trials in Block', trials_since_block),
                     ('Squared Time in Block', trial_sq),
                     ('Trials x Choice', interaction_trials_choice),
                     ('Trials x Choice Sq',choice_trials_sq),                                           
                     ('ones', ones_test)])
          
        X = np.vstack(predictors_train.values()).T[:trials,:].astype(float)
        n_predictors = X.shape[1]
        y = session_training.reshape([len(session_training),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_true.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_true.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_true.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
          
         
        trials ,n_neurons, n_timepoints = session_misaligned.shape
     
        # Misaligned Using Neurons
                  
        y = session_misaligned.reshape([len(session_misaligned),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_misaligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_misaligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_misaligned.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
                 
        # Aligned Using Neurons
      
             
        n_predictors = X.shape[1]
        y = session_aligned.reshape([len(session_aligned),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_aligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_aligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_aligned.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
    

    cpd_true = np.nanmean(np.concatenate(cpd_true,0), axis = 0)
    C_true = np.concatenate(C_true,1)
    
    cpd_misaligned = np.nanmean(np.concatenate(cpd_misaligned,0), axis = 0)
    C_misaligned = np.concatenate(C_misaligned,1)
    
    
    cpd_aligned = np.nanmean(np.concatenate(cpd_aligned,0), axis = 0)
    C_aligned = np.concatenate(C_aligned,1)
    
    
    C_sq_true = np.concatenate(C_sq_true,1)
    C_sq_misaligned =np.concatenate(C_sq_misaligned,1)
    C_sq_aligned = np.concatenate(C_sq_aligned,1)
    
    C_sq_true[np.isfinite(C_sq_true) == False] = np.NaN
    C_sq_true = np.nanmean(C_sq_true,1)[:-1,:]
    C_sq_misaligned = np.nanmean(C_sq_misaligned,1)[:-1,:]
    C_sq_aligned = np.nanmean(C_sq_aligned,1)[:-1,:]

    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors_train]
    
    plt.figure()
    
    for i in np.arange(C_sq_true.shape[0]):
        plt.plot(C_sq_true[i,:], label =p[i] + 'Real', color = c[i])
        plt.plot(C_sq_misaligned[i,:], label =p[i] + 'Misaligned', color = c[i], linestyle = '--')
        plt.plot(C_sq_aligned[i,:], label =p[i] + 'Aligned', color = c[i], linestyle = ':')

    plt.legend()
    plt.ylabel('Coef Sq')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
   
    
    cpd_true = cpd_true[:,:-1]
    cpd_misaligned = cpd_misaligned[:,:-1]
    cpd_aligned = cpd_aligned[:,:-1]
    
    plt.figure()
    
    for i in np.arange(cpd_true.shape[1]):
        plt.plot(cpd_true[:,i], label =p[i] + 'Real', color = c[i])
        plt.plot(cpd_misaligned[:,i], label =p[i] + 'Misaligned', color = c[i], linestyle = '--')
        plt.plot(cpd_aligned[:,i], label =p[i] + 'Aligned', color = c[i], linestyle = ':')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
   
    
   


def select_trials(Data, DM,max_number_per_block, ind_time = np.arange(0,63)):
    
    all_sessions_firing = []
    all_session_dm = []
    for data,dm in zip(Data, DM):
        
        
        trials, neurons, time = data.shape
        if neurons > 10:
            block = dm[:,4]
                       
            state_change = np.where(np.diff(block)!=0)[0]+1
            state_change = np.append(state_change,0)
            state_change = np.sort(state_change)     
            
            #if len(state_change) > 12:
           
            
            if len(state_change) > 11:
                block_12_ind = state_change[12]
                state_change = state_change[:12]

                data = data[:block_12_ind]  
                
                data_t1_1 = data[state_change[0]:state_change[1]][-max_number_per_block:]
                data_t1_2 = data[state_change[1]:state_change[2]][-max_number_per_block:]
                data_t1_3 = data[state_change[2]:state_change[3]][-max_number_per_block:]
                data_t1_4 = data[state_change[3]:state_change[4]][-max_number_per_block:]
              
                data_t2_1 = data[state_change[4]:state_change[5]][-max_number_per_block:]
                data_t2_2 = data[state_change[5]:state_change[6]][-max_number_per_block:]
                data_t2_3 = data[state_change[6]:state_change[7]][-max_number_per_block:]
                data_t2_4 = data[state_change[7]:state_change[8]][-max_number_per_block:]
    
                data_t3_1 = data[state_change[8]:state_change[9]][-max_number_per_block:]
                data_t3_2 = data[state_change[9]:state_change[10]][-max_number_per_block:]
                data_t3_3 = data[state_change[10]:state_change[11]][-max_number_per_block:]
                data_t3_4 = data[state_change[11]:][-max_number_per_block:]
                
                #DM matrix
                
                dm_t1_1 = dm[state_change[0]:state_change[1]][-max_number_per_block:]
                dm_t1_2 = dm[state_change[1]:state_change[2]][-max_number_per_block:]
                dm_t1_3 = dm[state_change[2]:state_change[3]][-max_number_per_block:]
                dm_t1_4 = dm[state_change[3]:state_change[4]][-max_number_per_block:]
                
                    
                dm_t2_1 = dm[state_change[4]:state_change[5]][-max_number_per_block:]
                dm_t2_2 = dm[state_change[5]:state_change[6]][-max_number_per_block:]
                dm_t2_3 = dm[state_change[6]:state_change[7]][-max_number_per_block:]
                dm_t2_4 = dm[state_change[7]:state_change[8]][-max_number_per_block:]
    
                dm_t3_1 = dm[state_change[8]:state_change[9]][-max_number_per_block:]
                dm_t3_2 = dm[state_change[9]:state_change[10]][-max_number_per_block:]
                dm_t3_3 = dm[state_change[10]:state_change[11]][-max_number_per_block:]
                dm_t3_4 = dm[state_change[11]:][-max_number_per_block:]
                
                
                data_list = np.concatenate((data_t1_1,data_t1_2,data_t1_3,data_t1_4,data_t2_1,data_t2_2,data_t2_3,data_t2_4,data_t3_1,data_t3_2,data_t3_3,data_t3_4))
                dm = np.concatenate((dm_t1_1,dm_t1_2,dm_t1_3,dm_t1_4,dm_t2_1,dm_t2_2,dm_t2_3,dm_t2_4,dm_t3_1,dm_t3_2,dm_t3_3,dm_t3_4))
                
                if data_list.shape[0] == 120:
                    all_sessions_firing.append(data_list)
                    all_session_dm.append(dm)
    return all_sessions_firing,all_session_dm

 
    
def align_within_task():
    
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)
    corr_aligned_list = []
    
    if between_tasks: 
        
        for task_1, task_2 in zip([flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_second_half],[flattened_all_clusters_task_2_first_half,flattened_all_clusters_task_3_first_half]):
           
            # PCA on the neuronal data with n dimensions 
            n = 200
            u_1, s_1, v_1 =svds(task_1, n) # u is n x m 
            u_2, s_2, v_2 =svds(task_2, n) 
            
            proj_a = np.linalg.multi_dot([u_1.T, task_1]) # project onto the m manifolds
            proj_b = np.linalg.multi_dot([u_2.T, task_2])
            
            Q_a, R_a = np.linalg.qr(proj_a.T)
            Q_b, R_b = np.linalg.qr(proj_b.T)
            
            
            product_Qs = np.linalg.multi_dot([Q_a.T, Q_b]) # Take equal numbers of trials from each sessions !
            
            
            U,S,V = np.linalg.svd(product_Qs, full_matrices = True) # maximize correlations between manifolds
            
            M_a = np.linalg.multi_dot([np.linalg.inv(R_a),U]) # project latent dynamics onto new directions
            M_b = np.linalg.multi_dot([np.linalg.inv(R_b),V.T]) 
            
            aligned_U_a =  np.linalg.multi_dot([u_1, M_a]) #latent dynamics projected on to new manifold axes
            aligned_U_b =  np.linalg.multi_dot([u_2, M_b])
            
            corr_missaligned = np.linalg.multi_dot([proj_a,proj_b.T])
            
            corr_aligned = np.linalg.multi_dot([aligned_U_a.T,aligned_U_b])
        
            u_aligned, s_aligned, v_aligned = np.linalg.svd(np.linalg.multi_dot([aligned_U_a.T,aligned_U_b]))
            aligned = np.linalg.multi_dot([proj_b.T,M_b,np.linalg.inv(M_a)])
            
            
            
            #corr_aligned = np.linalg.multi_dot([aligned_U_a,aligned_U_b.T])
            aligned_by_trial = np.transpose(aligned, [1,0])
            original_a = proj_a
            original_b = proj_b 
            plt.figure()
            corr_misaligned = np.corrcoef(original_a,original_b)[200:,:200]
            
            
            plt.imshow(corr_misaligned)
            plt.colorbar()
            plt.figure()
            corr_aligned = np.corrcoef(original_a,aligned_by_trial)[200:,:200]
            corr_aligned_list.append(np.diagonal(corr_aligned))
            plt.imshow(corr_aligned)
            plt.title('Between Tasks')
            plt.colorbar()
            
    mean_between = np.mean(corr_aligned_list)   
   # plt.figure(12)
    #plt.plot(mean_between,color = 'black', linestyle = '--')
    corr_aligned_list_within = []
    if within_tasks: 
        for task_1, task_2 in zip([flattened_all_clusters_task_1_first_half,flattened_all_clusters_task_2_first_half],[flattened_all_clusters_task_1_second_half,flattened_all_clusters_task_2_second_half]):
           # PCA on the neuronal data with n dimensions 
            n = 200
            u_1, s_1, v_1 =svds(task_1, n) # u is n x m 
            u_2, s_2, v_2 =svds(task_2, n) 
            
            proj_a = np.linalg.multi_dot([u_1.T, task_1]) # project onto the m manifolds
            proj_b = np.linalg.multi_dot([u_2.T, task_2])
            
            Q_a, R_a = np.linalg.qr(proj_a.T)
            Q_b, R_b = np.linalg.qr(proj_b.T)
            
            
            product_Qs = np.linalg.multi_dot([Q_a.T, Q_b]) # Take equal numbers of trials from each sessions !
            
            
            U,S,V = np.linalg.svd(product_Qs, full_matrices = True) # maximize correlations between manifolds
            
            M_a = np.linalg.multi_dot([np.linalg.inv(R_a),U]) # project latent dynamics onto new directions
            M_b = np.linalg.multi_dot([np.linalg.inv(R_b),V.T]) 
            
            aligned_U_a =  np.linalg.multi_dot([u_1, M_a]) #latent dynamics projected on to new manifold axes
            aligned_U_b =  np.linalg.multi_dot([u_2, M_b])
            
            corr_missaligned = np.linalg.multi_dot([proj_a,proj_b.T])
            
            corr_aligned = np.linalg.multi_dot([aligned_U_a.T,aligned_U_b])
        
            u_aligned, s_aligned, v_aligned = np.linalg.svd(np.linalg.multi_dot([aligned_U_a.T,aligned_U_b]))
            aligned = np.linalg.multi_dot([proj_b.T,M_b,np.linalg.inv(M_a)])
            
            
            
            #corr_aligned = np.linalg.multi_dot([aligned_U_a,aligned_U_b.T])
            aligned_by_trial = np.transpose(aligned, [1,0])
            original_a = proj_a
            original_b = proj_b 
            plt.figure()
            corr_misaligned = np.corrcoef(original_a,original_b)[200:,:200]
            
            
            plt.imshow(corr_misaligned)
            plt.colorbar()
            plt.figure()
            corr_aligned = np.corrcoef(original_a,aligned_by_trial)[200:,:200]
            corr_aligned_list_within.append(np.diagonal(corr_aligned))
            plt.imshow(corr_aligned)
            plt.title('Between Tasks')
            plt.colorbar()
            
    mean_within = np.mean(corr_aligned_list_within)
    plt.figure()
    plt.bar([1,2], [mean_within,mean_between], color =['green', 'lightgreen'])
    plt.title('HP')
    #plt.figure(12)
    #plt.plot(mean_within,color = 'black')    
    #difff = corr_aligned**2-corr_misaligned**2


def task_modes(ind_a,ind_b,Data, DM,misaligned_list,aligned_list):
    
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_HP, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)
    
    task_1 = flattened_all_clusters_task_1_second_half
    task_2 = flattened_all_clusters_task_2_first_half

    # PCA on the neuronal data with n dimensions 
    u_1, s_1, v_1 =np.linalg.svd(task_1) # u is n x m 
    u_2, s_2, v_2 =np.linalg.svd(task_2) 
           
    proj_a = np.linalg.multi_dot([v_1.T, task_1.T]) # project onto the m manifolds
    proj_b = np.linalg.multi_dot([v_2.T, task_2.T])
    
    Q_a, R_a = np.linalg.qr(proj_a.T)
    Q_b, R_b = np.linalg.qr(proj_b.T)

    product_Qs = np.linalg.multi_dot([Q_a.T, Q_b]) # Take equal numbers of trials from each sessions !
    
    
    U,S,V = np.linalg.svd(product_Qs, full_matrices = True) # maximize correlations between manifolds
    
    M_a = np.linalg.multi_dot([np.linalg.inv(R_a),U]) # project latent dynamics onto new directions
    M_b = np.linalg.multi_dot([np.linalg.inv(R_b),V.T]) 
    
    aligned_U_a =  np.linalg.multi_dot([v_1, M_a]) #latent dynamics projected on to new manifold axes
    aligned_U_b =  np.linalg.multi_dot([v_2, M_b])
    
    corr_missaligned = np.linalg.multi_dot([proj_a,proj_b.T])
    
    corr_aligned = np.linalg.multi_dot([aligned_U_a.T,aligned_U_b])

    u_aligned, s_aligned, v_aligned = np.linalg.svd(np.linalg.multi_dot([aligned_U_a.T,aligned_U_b]))
    aligned = np.linalg.multi_dot([proj_b.T,M_b,np.linalg.inv(M_a)])
     
      
    aligned_by_trial = np.transpose(aligned)
    original_a = proj_a
    original_b = proj_b
    
    #.figure()
    corr_misaligned = np.corrcoef(original_a,original_b)[:252,252:]
    
    
    #plt.imshow(corr_misaligned)
   #plt.colorbar()
    #plt.figure()
    corr_aligned = np.corrcoef(original_a,aligned_by_trial)[:252,252:]
    #plt.imshow(corr_aligned)
    #plt.colorbar()
    difff = corr_aligned**2-corr_misaligned**2
    misaligned_corr  = np.diagonal(corr_misaligned)
    aligned_corr = np.diagonal(corr_aligned)

    plt.plot(misaligned_corr, color = 'red')
    plt.plot(aligned_corr, color = 'black')

    return aligned_by_trial, original_a, original_b

    #u_2, s_2, v_2 = np.linalg.svd(session_b, full_matrices = True)
    

def in_progress(ind_a,ind_b,Data, DM,misaligned_list,aligned_list):
    
    all_sessions,all_session_dm =  select_trials(Data, DM,10, ind_time = np.arange(0,63))
   
    #session_a_dm = all_session_dm[ind_a]
    #session_b_dm = all_session_dm[ind_b]

    session_a = np.transpose(all_sessions[ind_a],[1,0,2]).reshape(all_sessions[ind_a].shape[1],all_sessions[ind_a].shape[0]*all_sessions[ind_a].shape[2])
    session_b = np.transpose(all_sessions[ind_b],[1,0,2]).reshape(all_sessions[ind_b].shape[1],all_sessions[ind_b].shape[0]*all_sessions[ind_b].shape[2])
    session_a = session_a[:10,:]
    session_b = session_b[:10,:]

    # PCA on the neuronal data with n dimensions 
    n = 8
    u_1, s_1, v_1 =svds(session_a, n) # u is n x m 
    u_2, s_2, v_2 =svds(session_b, n) 
    
    proj_a = np.linalg.multi_dot([u_1.T, session_a]) # project onto the m manifolds
    proj_b = np.linalg.multi_dot([u_2.T, session_b])
    
    Q_a, R_a = np.linalg.qr(proj_a.T)
    Q_b, R_b = np.linalg.qr(proj_b.T)
    
    
    product_Qs = np.linalg.multi_dot([Q_a.T, Q_b]) # Take equal numbers of trials from each sessions !
    
    
    U,S,V = np.linalg.svd(product_Qs, full_matrices = True) # maximize correlations between manifolds
    
    M_a = np.linalg.multi_dot([np.linalg.inv(R_a),U]) # project latent dynamics onto new directions
    M_b = np.linalg.multi_dot([np.linalg.inv(R_b),V.T]) 
    
    aligned_U_a =  np.linalg.multi_dot([u_1, M_a]) #latent dynamics projected on to new manifold axes
    aligned_U_b =  np.linalg.multi_dot([u_2, M_b])
    
    corr_missaligned = np.linalg.multi_dot([proj_a,proj_b.T])
    
    corr_aligned = np.linalg.multi_dot([aligned_U_a.T,aligned_U_b])

    u_aligned, s_aligned, v_aligned = np.linalg.svd(np.linalg.multi_dot([aligned_U_a.T,aligned_U_b]))
    aligned = np.linalg.multi_dot([proj_b.T,M_b,np.linalg.inv(M_a)])
     
    aligned_by_trial = np.transpose(aligned, [1,0])
    original_a = proj_a
    original_b = proj_b
    
    plt.figure()
    corr_misaligned = np.corrcoef(original_a,original_b)[8:,:8]
    
    
    plt.imshow(corr_misaligned)
    plt.colorbar()
    plt.figure()
    corr_aligned = np.corrcoef(original_a,aligned_by_trial)[8:,:8]
    plt.imshow(corr_aligned)
    plt.colorbar()
    difff = corr_aligned**2-corr_misaligned**2
    misaligned_list.append(np.diagonal(corr_misaligned))
    aligned_list.append(np.diagonal(corr_aligned))

    aligned_by_trial = aligned_by_trial.reshape(aligned_by_trial.shape[0],all_sessions[ind_a].shape[0],all_sessions[ind_a].shape[2])
    original_a = original_a.reshape(aligned_by_trial.shape[0],all_sessions[ind_a].shape[0],all_sessions[ind_a].shape[2])
    original_b = original_b.reshape(aligned_by_trial.shape[0],all_sessions[ind_a].shape[0],all_sessions[ind_a].shape[2])
                                     
    return all_sessions, all_session_dm, aligned_by_trial, original_a, original_b

    #u_2, s_2, v_2 = np.linalg.svd(session_b, full_matrices = True)
    


def regression(Data, DM, PFC = True):

    cpd_true = []
    C_true = []
    cpd_misaligned = []
    C_misaligned = []
    cpd_aligned = []
    C_aligned = []
    
    C_sq_true = []
    C_sq_misaligned = []
    C_sq_aligned = []
    
    if PFC == True:
        ind_1 = np.arange(0,26)
        ind_2 =  np.arange(1,27)
        ind_a = np.hstack((ind_1,ind_2))
        ind_b = np.hstack((ind_2,ind_1))
    
    else:
        ind_1 = np.arange(0,15)
        ind_2 =  np.arange(1,16)
        ind_a = np.hstack((ind_1,ind_2))
        ind_b = np.hstack((ind_2,ind_1))
    
    misaligned_list = []
    aligned_list = []

    for a,b in zip(ind_a, ind_b):
      
        all_sessions, all_session_dm, aligned_by_trial, original_a, original_b = in_progress(a,b,Data, DM, misaligned_list,aligned_list)
        # plt.figure() 

        # for i in np.arange(aligned_by_trial.shape[0]):
        #     plt.plot(np.mean(aligned_by_trial,1)[i,:])
    
        # plt.figure() 
        # for i in np.arange(original_b.shape[0]):
        #     plt.plot(np.mean(original_b,1)[i,:])
    
        #session_a = all_sessions[ind_a]
          
        dm_test  = all_session_dm[a]
        
        session_training =  np.transpose(original_a, [1,0,2])
        session_misaligned = np.transpose(original_b, [1,0,2])
        session_aligned =   np.transpose(aligned_by_trial, [1,0,2])
        
        trials ,n_neurons, n_timepoints = session_training.shape
        
        
        
        reward_test = dm_test[:,2]
        state_test = dm_test[:,0]        
        choices_test = dm_test[:,1]
        ones_test = np.ones(len(choices_test))
        
        trials_since_block = np.arange(0,10)
        trials_since_block = np.tile(trials_since_block,12)
           
        reward_choice = choices_test*reward_test      
        trial_sq = (np.asarray(trials_since_block)-0.5)**2
        choice_trials_sq = choices_test*trial_sq
        interaction_trials_choice = trials_since_block*choices_test

     

        # Original Using Neurons
        predictors_train = OrderedDict([('State', state_test),
                     ('Reward', reward_test),
                     ('Choice', choices_test),
                     #('Reward Choice Int', reward_choice),
                     #('Trials in Block', trials_since_block),
                     #('Squared Time in Block', trial_sq),
                     #('Trials x Choice', interaction_trials_choice),
                     #('Trials x Choice Sq',choice_trials_sq),                                           
                     ('ones', ones_test)])
          
        X = np.vstack(predictors_train.values()).T[:trials,:].astype(float)
        n_predictors = X.shape[1]
        y = session_training.reshape([len(session_training),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_true.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_true.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_true.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
          
         
        trials ,n_neurons, n_timepoints = session_misaligned.shape
     
        # Misaligned Using Neurons
                  
        y = session_misaligned.reshape([len(session_misaligned),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_misaligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_misaligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_misaligned.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
                 
        # Aligned Using Neurons
      
             
        n_predictors = X.shape[1]
        y = session_aligned.reshape([len(session_aligned),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
        tstats = reg_f.regression_code(y, X)
        C_aligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
        C_sq_aligned.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)**2)
        cpd_aligned.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
    

    cpd_true = np.nanmean(np.concatenate(cpd_true,0), axis = 0)
    C_true = np.concatenate(C_true,1)
    
    cpd_misaligned = np.nanmean(np.concatenate(cpd_misaligned,0), axis = 0)
    C_misaligned = np.concatenate(C_misaligned,1)
    
    
    cpd_aligned = np.nanmean(np.concatenate(cpd_aligned,0), axis = 0)
    C_aligned = np.concatenate(C_aligned,1)
    
    
    C_sq_true = np.concatenate(C_sq_true,1)
    C_sq_misaligned =np.concatenate(C_sq_misaligned,1)
    C_sq_aligned = np.concatenate(C_sq_aligned,1)
    
    C_sq_true[np.isfinite(C_sq_true) == False] = np.NaN
    C_sq_true = np.nanmean(C_sq_true,1)[:-1,:]
    C_sq_misaligned = np.nanmean(C_sq_misaligned,1)[:-1,:]
    C_sq_aligned = np.nanmean(C_sq_aligned,1)[:-1,:]

    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors_train]
    
    plt.figure()
    
    for i in np.arange(C_sq_true.shape[0]):
        plt.plot(C_sq_true[i,:], label =p[i] + 'Real', color = c[i])
        plt.plot(C_sq_misaligned[i,:], label =p[i] + 'Misaligned', color = c[i], linestyle = '--')
        plt.plot(C_sq_aligned[i,:], label =p[i] + 'Aligned', color = c[i], linestyle = ':')

    plt.legend()
    plt.ylabel('Coef Sq')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
   
    
    cpd_true = cpd_true[:,:-1]
    cpd_misaligned = cpd_misaligned[:,:-1]
    cpd_aligned = cpd_aligned[:,:-1]
    
    plt.figure()
    
    for i in np.arange(cpd_true.shape[1]):
        plt.plot(cpd_true[:,i], label =p[i] + 'Real', color = c[i])
        plt.plot(cpd_misaligned[:,i], label =p[i] + 'Misaligned', color = c[i], linestyle = '--')
        plt.plot(cpd_aligned[:,i], label =p[i] + 'Aligned', color = c[i], linestyle = ':')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
   
    
    return misaligned_list,aligned_list

def plot_corr(misaligned_list,aligned_list):
    mis = np.mean(misaligned_list_PFC,0).reshape(8,1)
    al = np.mean(aligned_list_PFC,0).reshape(8,1)
    

    
def decoder(Data, DM):
   
    ind_a = [0,1,2,3,4,5]
    ind_b = [1,2,3,4,5,6]
    fig = plt.figure()

    for a,b in zip(ind_a, ind_b):
        print()
        
        all_sessions, all_session_dm, aligned_by_trial, original_a, original_b = in_progress(a,b,Data, DM)
        # plt.figure() 

        # for i in np.arange(aligned_by_trial.shape[0]):
        #     plt.plot(np.mean(aligned_by_trial,1)[i,:])
    
        # plt.figure() 
        # for i in np.arange(original_b.shape[0]):
        #     plt.plot(np.mean(original_b,1)[i,:])
        dm_train = all_session_dm[b]

        dm_test = all_session_dm[a]
        session_misaligned = np.mean(np.transpose(original_b,[1,0,2]),2)
        session_aligned =   np.mean(np.transpose(aligned_by_trial, [1,0,2]),2)
        target = np.mean(np.transpose(original_a,[1,0,2]),2)

        trials ,n_neurons, n_timepoints = original_a.shape
        
        reward_test = dm_test[:,0]
        reward_train = dm_train[:,0]

         
        session_testing_within = np.mean(original_a,2)[:,60:]
        session_training_within = np.mean(original_a,2)[:,:60]

        dm_within_train = reward_test[:60]
        dm_within_test = reward_test[60:]

        #model_nb = GaussianNB()
        model_nb = svm.SVC(gamma='scale',class_weight='balanced')
        #model_nb = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial')

        # Decoding within task using PCA
    
        model_nb.fit(np.transpose(session_training_within),dm_within_train)  
        y_pred_pca = model_nb.predict(np.transpose(session_testing_within))
        correct_pca = metrics.accuracy_score(dm_within_test, y_pred_pca)
    
        # Decoding within task using PCA
    
        model_nb.fit(session_misaligned,reward_train)  
        y_pred_misaligned = model_nb.predict(target)
        correct_misaligned = metrics.accuracy_score(reward_test, y_pred_misaligned)
    
    
        model_nb.fit(session_aligned,reward_train)  
        y_pred_aligned = model_nb.predict(target)
        correct_aligned = metrics.accuracy_score(reward_test, y_pred_aligned)
        
        fig.add_subplot(3,2,b)
        plt.bar([1,2,3], [correct_pca,correct_aligned,correct_misaligned], color = 'black')
        plt.xticks([1,2,3], ('Within Task', 'Aligned', 'Misaligned'))
    plt.tight_layout()
   
