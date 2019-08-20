#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:13:18 2019

@author: veronikasamborska
"""

#import heatmap_aligned as ha
import math
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import forced_trials_extract_data as ft
import numpy as np 
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import create_data_arrays_for_tim as cda
import ephys_beh_import as ep
import heatmap_aligned as ha

from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import gaussian_filter

#ephys_path = '/Users/veronikasamborska/Desktop/neurons'
#beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
#HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
#experiment_aligned_PFC = ha.all_sessions_aligment(PFC, all_sessions)
#experiment_aligned_HP = ha.all_sessions_aligment(HP, all_sessions)
#
#
#PFC_forced = ft.all_sessions_aligment_forced(PFC,all_sessions )
#HP_forced = ft.all_sessions_aligment_forced(HP,all_sessions)

#data_PFC = cda.tim_create_mat(experiment_aligned_PFC, experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC, 'PFC') 
#data_HP = cda.tim_create_mat(experiment_aligned_HP, experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP, experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP, 'HP')

y = data_PFC['DM']
X = data_PFC['Data']

DM = y[47]
choices =  DM[:,1]
reward = DM[:,2]
forced_trials  = DM[:,3]
chosen_Q1 = DM[:,8]
chosen_Q4  = DM[:,9]
Q1_value_a = DM[:,10]
Q1_value_b = DM[:,11]
Q4_value_a = DM[:,12]

c = gaussian_filter(choices, 1)

def classifier_GPR(data, session):    
    
    y = data_HP['DM']
    X = data_HP['Data']
    length = []
    correct_list_within = []
    correct_list_between = []
    all_y_within_1 = []
    all_y_between_1 = []
    all_Ys = []
   

    for s, sess in enumerate(X):
        
        # Design matrix for the session
        DM = y[s]
     
        firing_rates_all_time = X[s]
        
        if firing_rates_all_time.shape[1]>20:
            # Tasks indicies 
            choices =  DM[:,1]
            task = DM[:,4]
            
            
            task_1 = np.where(task == 1)[0] #& (choices == 1))[0]
            task_2 = np.where(task == 2)[0] #& (choices == 1))[0]
            task_3 = np.where(task == 3)[0] #& (choices == 1))[0]
            
            # Find the maximum length of any of the tasks in a session
            length.append(len(task_1))
            length.append(len(task_2))
            length.append(len(task_3))     
            min_trials_in_task = int(np.min(length)/2)
            
            #firing_rates_all_time = (firing_rates_all_time-firing_rates_all_time_mean)/firing_rates_all_time_std
            
            # Select the first min_trials_in_task in task one
            firing_rates_mean_task_1_1 = firing_rates_all_time[task_1]
            firing_rates_mean_task_1_1 = firing_rates_mean_task_1_1[:min_trials_in_task,:]
            # Select the last min_trials_in_task in task one
            firing_rates_mean_task_1_2 = firing_rates_all_time[task_1]
            firing_rates_mean_task_1_2 = firing_rates_all_time[task_2[0]-1-min_trials_in_task:task_2[0]-1,:]
           
            # Select the first min_trials_in_task in task two
            firing_rates_mean_task_2_1 = firing_rates_all_time[task_2]
            firing_rates_mean_task_2_1 = firing_rates_all_time[task_2[0]:task_2[0]+min_trials_in_task,:]
            firing_rates_mean_task_2_2 = firing_rates_all_time[task_2]
            firing_rates_mean_task_2_2 = firing_rates_all_time[task_3[0]-1-min_trials_in_task:task_3[0]-1,:]
    
            # Select the first min_trials_in_task in task three
            firing_rates_mean_task_3_1 = firing_rates_all_time[task_3]
            firing_rates_mean_task_3_1 = firing_rates_all_time[task_3[0]:task_3[0]+min_trials_in_task,:]
            firing_rates_mean_task_3_2 = firing_rates_all_time[task_3]
            firing_rates_mean_task_3_2 = firing_rates_all_time[task_3[-1]-min_trials_in_task:task_3[-1],:]
    
            
            # Finding the angle between initiation and every ms          
            # C = 2πr;  Circumference of a circle
            
            C = session.aligned_rates.shape[2]
            p = math.pi
            r =  C/(2*p)
            
            angle_sin_list = []
            angle_cos_list = []
            for i in range(C):
                L = 0+ (i+1)
                ang = (180*L)/(p*r)
                ang_sin = np.sin(np.deg2rad(ang))
                ang_cos = np.cos(np.deg2rad(ang))
            
                angle_sin_list.append(ang_sin)
                angle_cos_list.append(ang_cos)
         
            firing_rates_mean_1_1 = np.concatenate(firing_rates_mean_task_1_1, axis = 1)
            firing_rates_mean_1_2 = np.concatenate(firing_rates_mean_task_1_2, axis = 1)
            firing_rates_mean_2_1 = np.concatenate(firing_rates_mean_task_2_1, axis = 1)
            firing_rates_mean_2_2 = np.concatenate(firing_rates_mean_task_2_2, axis = 1)
            firing_rates_mean_3_1 = np.concatenate(firing_rates_mean_task_3_1, axis = 1)
            firing_rates_mean_3_2 = np.concatenate(firing_rates_mean_task_3_2, axis = 1)
    
            l = firing_rates_mean_1_1.shape[1]
            
            # Creating a vector which identifies trial stage in the firing rate vector
            Y_cos = np.tile(angle_cos_list,int(l/len(angle_cos_list)))
            Y_sin = np.tile(angle_sin_list,int(l/len(angle_sin_list)))
            Y = np.vstack((Y_cos,Y_sin))
            
            #kernel = RBF(length_scale = 2)
            kernel = Matern(nu = 3/2)
            model_nb = GPR(kernel = kernel)
            #model_nb = LinearRegression()
            
            model_nb.fit(np.transpose(firing_rates_mean_1_2), np.transpose(Y))
            y_pred_class_between_t_1_2 = model_nb.predict(np.transpose(firing_rates_mean_2_1))
            correct_between_t_1 = model_nb.score(np.transpose(firing_rates_mean_2_1),np.transpose(Y))
            
            model_nb.fit(np.transpose(firing_rates_mean_1_1),np.transpose(Y))     
            y_pred_class_within_t_1_2 = model_nb.predict(np.transpose(firing_rates_mean_1_2))
            correct_within_t_1 = model_nb.score(np.transpose(firing_rates_mean_1_2),np.transpose(Y))

            model_nb.fit(np.transpose(firing_rates_mean_2_2),np.transpose(Y))
            y_pred_class_between_t_2_3 = model_nb.predict(np.transpose(firing_rates_mean_3_1))
            correct_between_t_2 =  model_nb.score(np.transpose(firing_rates_mean_3_1),np.transpose(Y))

            model_nb.fit(np.transpose(firing_rates_mean_2_1),np.transpose(Y))
            y_pred_class_within_t_2_3 = model_nb.predict(np.transpose(firing_rates_mean_2_2))
            correct_within_t_2 =  model_nb.score(np.transpose(firing_rates_mean_2_2),np.transpose(Y))
            
            model_nb.fit(np.transpose(firing_rates_mean_3_1),np.transpose(Y))
            y_pred_class_within_t_3 = model_nb.predict(np.transpose(firing_rates_mean_3_2))
            correct_within_t_3 =  model_nb.score(np.transpose(firing_rates_mean_3_2),np.transpose(Y))

            correct_list_within.append(correct_within_t_1)
            correct_list_within.append(correct_within_t_2)
            correct_list_within.append(correct_within_t_3)
    
            correct_list_between.append(correct_between_t_1)
            correct_list_between.append(correct_between_t_2)
            
            all_y_within_1.append(y_pred_class_within_t_1_2)
            all_y_between_1.append(y_pred_class_between_t_1_2)
            all_Ys.append(Y)
            
    print(correct_list_within)
    print(correct_list_between)

    return correct_list_within, correct_list_between,all_y_within_1,all_y_between_1,all_Ys
