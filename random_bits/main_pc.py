#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:48:46 2019

@author: veronikasamborska
"""
import pylab as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d



def trajectory_analysis(data_HP,data_PFC, experiment_aligned_HP,log=True, fig_no=1, PCs=[0,1,2], remove_nonspecific=True, task_plt = 1):
    '''Plot trajectories showing the average activity for each trial type defined by 
    choice, transition and outcome in a low dimensional space obtained by PCA on the
    data matrix [n_neurons, n_trial_types*n_timepoints].  If the remove_nonspecific 
    argument is True, the cross condition mean activity is subtrated from each conditions
    activity before PCA so only variance across trial types remains. '''
    # Extract average activity for each trial type.
    
    condition_ave_activity_HP = []
    condition_ave_activity_PFC = []

    y_HP = data_HP['DM'][0]
    X_HP = data_HP['Data'][0]
    
    y_PFC = data_PFC['DM'][0]
    X_PFC = data_PFC['Data'][]
    session = experiment_aligned_HP[0]
    
    for s,sess in enumerate(y_HP):
        # HP
        DM = y_HP[s]
        aligned_spikes = X_HP[s]
        task = DM[:,4]
        
        task_1 = np.where(task == 1)[0]
        task_2 = np.where(task == 2)[0]
        task_3 = np.where(task == 3)[0]
        
        if task_plt == 1:
            aligned_spikes = aligned_spikes[task_1,:,:]
            c =  DM[:,1][task_1].astype(bool)
            o = DM[:,2][task_1].astype(bool)
       
        elif task_plt == 2:
            aligned_spikes = aligned_spikes[task_2,:,:]
            c =  DM[:,1][task_2].astype(bool)
            o = DM[:,2][task_2].astype(bool)
            
        elif task_plt ==3:
            aligned_spikes = aligned_spikes[task_3,:,:]
            c =  DM[:,1][task_3].astype(bool)
            o = DM[:,2][task_3].astype(bool)
            
        #n_trials = aligned_spikes.shape[0]
        
        if log:
            aligned_spikes = np.log2(aligned_spikes+0.01)
            
       
        trial_type_masks = [ c &  o,
                             c & ~o,
                            ~c &  o,
                            ~c & ~o ]
        ses_cond_aves = np.concatenate([np.mean(aligned_spikes[ttm,:,:],0, keepdims=True)
                                        for ttm in trial_type_masks], 0) 
        condition_ave_activity_HP.append(ses_cond_aves)
        
    for s,sess in enumerate(y_PFC):
  
        # PFC
        DM_PFC= y_PFC[s]
        aligned_spikes_PFC = X_PFC[s]
        task_PFC = DM[:,4]
        
        task_1_PFC = np.where(task_PFC == 1)[0]
        task_2_PFC = np.where(task_PFC == 2)[0]
        task_3_PFC = np.where(task_PFC == 3)[0]
        
        if task_plt == 1:
            aligned_spikes_PFC = aligned_spikes_PFC[task_1_PFC,:,:]
            c_PFC =  DM_PFC[:,1][task_1].astype(bool)
            o_PFC = DM_PFC[:,2][task_1].astype(bool)
       
        elif task_plt == 2:
            aligned_spikes_PFC = aligned_spikes_PFC[task_2_PFC,:,:]
            c_PFC =  DM_PFC[:,1][task_2].astype(bool)
            o_PFC = DM_PFC[:,2][task_2].astype(bool)
            
        elif task_plt ==3:
            aligned_spikes_PFC = aligned_spikes_PFC[task_3_PFC,:,:]
            c_PFC =  DM_PFC[:,1][task_3].astype(bool)
            o_PFC = DM_PFC[:,2][task_3].astype(bool)
            
        #n_trials = aligned_spikes.shape[0]
        
        if log:
            aligned_spikes_PFC = np.log2(aligned_spikes_PFC+0.01)
            
       
        trial_type_masks_PFC = [ c_PFC &  o_PFC,
                             c_PFC & ~o_PFC,
                            ~c_PFC &  o_PFC,
                            ~c_PFC & ~o_PFC ]
        ses_cond_aves_PFC = np.concatenate([np.mean(aligned_spikes_PFC[ttm,:,:],0, keepdims=True)
                                        for ttm in trial_type_masks_PFC], 0) 
        condition_ave_activity_PFC.append(ses_cond_aves_PFC)
    
    condition_ave_activity_PFC = np.concatenate(condition_ave_activity_PFC,1)  # [trial_type, n_neurons, n_timepoint]
    condition_ave_activity_HP = np.concatenate(condition_ave_activity_HP,1)  # [trial_type, n_neurons, n_timepoint]
    condition_ave_activity = np.concatenate((condition_ave_activity_PFC,condition_ave_activity_HP),1)
    condition_ave_activity = condition_ave_activity_PFC
    
    if remove_nonspecific: # Subtract mean across conditions from each condition.
        condition_ave_activity = condition_ave_activity - np.mean(condition_ave_activity,0)
        
     
    # Do PCA.
    X = np.hstack([caa for caa in condition_ave_activity]) # [n_neurons, n_timepoints*n_trial_types]
    pca = PCA(n_components = 10)
    pca.fit(X.T)
    
    fig = plt.figure(fig_no+2, figsize=[14,12], clear=True)
    ax3D_all = plt.subplot2grid([3,3], [0,0], rowspan=3, colspan=2, projection='3d')
    ax2Da_all = fig.add_subplot(3, 3, 3)
    ax2Db_all = fig.add_subplot(3, 3, 6)
    ax2Dc_all = fig.add_subplot(3, 3, 9)
    ax3D_all.set_xlabel('PC{}'.format(PCs[0]))
    ax3D_all.set_ylabel('PC{}'.format(PCs[1]))
    ax3D_all.set_zlabel('PC{}'.format(PCs[2]))
    ax2Da_all.set_xlabel('PC{}'.format(PCs[0]))
    ax2Da_all.set_ylabel('PC{}'.format(PCs[1]))
    ax2Db_all.set_xlabel('PC{}'.format(PCs[0]))
    ax2Db_all.set_ylabel('PC{}'.format(PCs[2]))
    ax2Dc_all.set_xlabel('PC{}'.format(PCs[1]))
    ax2Dc_all.set_ylabel('PC{}'.format(PCs[2]))
    labels = ['C:1 O:1', 'C:1 O:0', 'C:0 O:1', 'C:0 O:0']
    colors = ['b','b','r','r','c','c','orange','orange']
    styles = ['-','--','-','--','-','--','-','--']
    for i, caa in enumerate(condition_ave_activity):
        #3D plot
        traj_x_all = caa.T @ pca.components_[PCs[0],:]
        traj_y_all = caa.T @ pca.components_[PCs[1],:]
        traj_z_all = caa.T @ pca.components_[PCs[2],:]
        ax3D_all.plot3D(traj_x_all, traj_y_all, traj_z_all, color=colors[i], linestyle=styles[i], label=labels[i])
        # 2D projections.
        ax2Da_all.plot(traj_x_all, traj_y_all, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Db_all.plot(traj_x_all, traj_z_all, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Dc_all.plot(traj_y_all, traj_z_all, color=colors[i], linestyle=styles[i], label=labels[i])
        # Event markers.
        for j, sn, m in zip(range(3), mksn, ['$I$','$C$','$O$']):
            ax3D_all.scatter3D(traj_x_all[sn],traj_y_all[sn],traj_z_all[sn], color=colors[i], marker=m, s=80)
            ax2Da_all.scatter(traj_x_all[sn],traj_y_all[sn], color=colors[i], marker=m, s=80)
            ax2Db_all.scatter(traj_x_all[sn],traj_z_all[sn], color=colors[i], marker=m, s=80)
            ax2Dc_all.scatter(traj_y_all[sn],traj_z_all[sn], color=colors[i], marker=m, s=80)
            
    ax3D.legend()
    fig.tight_layout()
    
    # Plot trajectories for each trial type.
    fig = plt.figure(fig_no, figsize=[14,12], clear=True)
    ax3D = plt.subplot2grid([3,3], [0,0], rowspan=3, colspan=2, projection='3d')
    ax2Da = fig.add_subplot(3, 3, 3)
    ax2Db = fig.add_subplot(3, 3, 6)
    ax2Dc = fig.add_subplot(3, 3, 9)
    ax3D.set_xlabel('PC{}'.format(PCs[0]))
    ax3D.set_ylabel('PC{}'.format(PCs[1]))
    ax3D.set_zlabel('PC{}'.format(PCs[2]))
    ax2Da.set_xlabel('PC{}'.format(PCs[0]))
    ax2Da.set_ylabel('PC{}'.format(PCs[1]))
    ax2Db.set_xlabel('PC{}'.format(PCs[0]))
    ax2Db.set_ylabel('PC{}'.format(PCs[2]))
    ax2Dc.set_xlabel('PC{}'.format(PCs[1]))
    ax2Dc.set_ylabel('PC{}'.format(PCs[2]))
    labels = ['C:1 O:1', 'C:1 O:0', 'C:0 O:1', 'C:0 O:0']
    colors = ['b','b','r','r','c','c','orange','orange']
    styles = ['-','--','-','--','-','--','-','--']
    
    t_out = session.t_out
    initiate_choice_t = session.target_times 
    reward_time = initiate_choice_t[-2] +250
  
    initiation = (np.abs(t_out-initiate_choice_t[1])).argmin()     
    ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin()
    ind_reward = (np.abs(t_out-reward_time)).argmin()
    mksn =[initiation,ind_choice,ind_reward]
    
    for i, caa in enumerate(condition_ave_activity_HP):
        #3D plot
        traj_x_HP = caa.T @ pca.components_[PCs[0],condition_ave_activity_PFC.shape[1]:]
        traj_y_HP = caa.T @ pca.components_[PCs[1],condition_ave_activity_PFC.shape[1]:]
        traj_z_HP = caa.T @ pca.components_[PCs[2],condition_ave_activity_PFC.shape[1]:]
        ax3D.plot3D(traj_x_HP, traj_y_HP, traj_z_HP, color=colors[i], linestyle=styles[i], label=labels[i])
        # 2D projections.
        ax2Da.plot(traj_x_HP, traj_y_HP, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Db.plot(traj_x_HP, traj_z_HP, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Dc.plot(traj_y_HP, traj_z_HP, color=colors[i], linestyle=styles[i], label=labels[i])
        # Event markers.
        for j, sn, m in zip(range(3), mksn, ['$I$','$C$','$O$']):
            ax3D.scatter3D(traj_x_HP[sn],traj_y_HP[sn],traj_z_HP[sn], color=colors[i], marker=m, s=80)
            ax2Da.scatter(traj_x_HP[sn],traj_y_HP[sn], color=colors[i], marker=m, s=80)
            ax2Db.scatter(traj_x_HP[sn],traj_z_HP[sn], color=colors[i], marker=m, s=80)
            ax2Dc.scatter(traj_y_HP[sn],traj_z_HP[sn], color=colors[i], marker=m, s=80)
    
    
    fig = plt.figure(fig_no+1, figsize=[14,12], clear=True)
    ax3D_PFC = plt.subplot2grid([3,3], [0,0], rowspan=3, colspan=2, projection='3d')
    ax2Da_PFC = fig.add_subplot(3, 3, 3)
    ax2Db_PFC = fig.add_subplot(3, 3, 6)
    ax2Dc_PFC = fig.add_subplot(3, 3, 9)
    ax3D_PFC.set_xlabel('PC{}'.format(PCs[0]))
    ax3D_PFC.set_ylabel('PC{}'.format(PCs[1]))
    ax3D_PFC.set_zlabel('PC{}'.format(PCs[2]))
    ax2Da_PFC.set_xlabel('PC{}'.format(PCs[0]))
    ax2Da_PFC.set_ylabel('PC{}'.format(PCs[1]))
    ax2Db_PFC.set_xlabel('PC{}'.format(PCs[0]))
    ax2Db_PFC.set_ylabel('PC{}'.format(PCs[2]))
    ax2Dc_PFC.set_xlabel('PC{}'.format(PCs[1]))
    ax2Dc_PFC.set_ylabel('PC{}'.format(PCs[2]))
    labels = ['C:1 O:1', 'C:1 O:0', 'C:0 O:1', 'C:0 O:0']
    colors = ['b','b','r','r','c','c','orange','orange']
    styles = ['-','--','-','--','-','--','-','--']
    for i, caa in enumerate(condition_ave_activity_PFC):
        #3D plot
        traj_x_PFC = caa.T @ pca.components_[PCs[0],:condition_ave_activity_PFC.shape[1]]
        traj_y_PFC = caa.T @ pca.components_[PCs[1],:condition_ave_activity_PFC.shape[1]]
        traj_z_PFC = caa.T @ pca.components_[PCs[2],:condition_ave_activity_PFC.shape[1]]
        ax3D_PFC.plot3D(traj_x_PFC, traj_y_PFC, traj_z_PFC, color=colors[i], linestyle=styles[i], label=labels[i])
        # 2D projections.
        ax2Da_PFC.plot(traj_x_PFC, traj_y_PFC, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Db_PFC.plot(traj_x_PFC, traj_z_PFC, color=colors[i], linestyle=styles[i], label=labels[i])
        ax2Dc_PFC.plot(traj_y_PFC, traj_z_PFC, color=colors[i], linestyle=styles[i], label=labels[i])
        # Event markers.
        for j, sn, m in zip(range(3), mksn, ['$I$','$C$','$O$']):
            ax3D_PFC.scatter3D(traj_x_PFC[sn],traj_y_PFC[sn],traj_z_PFC[sn], color=colors[i], marker=m, s=80)
            ax2Da_PFC.scatter(traj_x_PFC[sn],traj_y_PFC[sn], color=colors[i], marker=m, s=80)
            ax2Db_PFC.scatter(traj_x_PFC[sn],traj_z_PFC[sn], color=colors[i], marker=m, s=80)
            ax2Dc_PFC.scatter(traj_y_PFC[sn],traj_z_PFC[sn], color=colors[i], marker=m, s=80)
            
      
