#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:18:50 2019

@author: veronikasamborska
"""
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import SVDs as sv 
import heatmap_aligned as ha
import numpy as np
import ephys_beh_import as ep
import svds_u_only as svdu
from scipy import stats 
import seaborn as sns
import matplotlib.pyplot as plt

def stats_svd(data_loaded = True, d = True):
    
    if data_loaded == False:
        ephys_path = '/Users/veronikasamborska/Desktop/neurons'
        beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
        
        HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
        
        experiment_aligned_m484 = ha.all_sessions_aligment(m484, all_sessions)
        experiment_aligned_m479 = ha.all_sessions_aligment(m479, all_sessions)
        experiment_aligned_m483 = ha.all_sessions_aligment(m483, all_sessions)
        experiment_aligned_m478 = ha.all_sessions_aligment(m478, all_sessions)
        experiment_aligned_m486 = ha.all_sessions_aligment(m486, all_sessions)
        experiment_aligned_m480 = ha.all_sessions_aligment(m480, all_sessions)
        experiment_aligned_m481 = ha.all_sessions_aligment(m481, all_sessions)
        
    
    #average_within_HP, average_between_HP = sv.svd_plotting(experiment_aligned_HP, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = True, demean_all_tasks = False)
    

    
    average_within_m484, average_between_m484 = sv.svd_plotting(experiment_aligned_m484, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = d, demean_all_tasks = False)
    average_within_m479, average_between_m479 = sv.svd_plotting(experiment_aligned_m479, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = d, demean_all_tasks = False)
    average_within_m483, average_between_m483 = sv.svd_plotting(experiment_aligned_m483, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = d, demean_all_tasks = False)
    average_within_m478, average_between_m478 = sv.svd_plotting(experiment_aligned_m478, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = d, demean_all_tasks = False)
    average_within_m481, average_between_m481 = sv.svd_plotting(experiment_aligned_m481, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = d, demean_all_tasks = False)
    average_within_m486, average_between_m486 = sv.svd_plotting(experiment_aligned_m486, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = d, demean_all_tasks = False)
    average_within_m480, average_between_m480 = sv.svd_plotting(experiment_aligned_m480, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = d, demean_all_tasks = False)
    
    first, average_between_m484,average_between_y_m484,average_within_x_m484, average_within_m484 = svdu.svd_u_and_v_separately(experiment_aligned_m484, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = False, z_score = False)
    first, average_between_m479,average_between_y_m479,average_within_x_m479, average_within_m479 = svdu.svd_u_and_v_separately(experiment_aligned_m479, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = False, z_score = False)
    first, average_between_m483,average_between_y_m483,average_within_x_m483, average_within_m483 = svdu.svd_u_and_v_separately(experiment_aligned_m483, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = False, z_score = False)
    
    first, average_between_m478,average_between_y_m478,average_within_x_m478, average_within_m478 = svdu.svd_u_and_v_separately(experiment_aligned_m478, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, demean_all_tasks = False, z_score = False)
    first, average_between_m481,average_between_y_m481,average_within_x_m481, average_within_m481 = svdu.svd_u_and_v_separately(experiment_aligned_m481, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, demean_all_tasks = False, z_score = False)
    
    first, average_between_m486,average_between_y_m486,average_within_x_m486, average_within_m486 = svdu.svd_u_and_v_separately(experiment_aligned_m486, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, demean_all_tasks = False, z_score = False)
    first, average_between_m480,average_between_y_m480,average_within_x_m480, average_within_m480 = svdu.svd_u_and_v_separately(experiment_aligned_m480, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, demean_all_tasks = False, z_score = False)
    
    m484 = (np.trapz(average_within_m484) - np.trapz(average_between_y_m484))/average_within_m484.shape[0]
    m479 = (np.trapz(average_within_m479) - np.trapz(average_between_y_m479))/average_within_m479.shape[0]
    m483 = (np.trapz(average_within_m483) - np.trapz(average_between_y_m483))/average_within_m483.shape[0]
    
    HP_area = [m484,m479,m483]
    m478 = (np.trapz(average_within_m478) - np.trapz(average_between_y_m478))/average_within_m478.shape[0]
    m481 = (np.trapz(average_within_m481) - np.trapz(average_between_y_m481))/average_within_m481.shape[0]
    m486 = (np.trapz(average_within_m486) - np.trapz(average_between_y_m486))/average_within_m486.shape[0]
    m480 = (np.trapz(average_within_m480) - np.trapz(average_between_y_m480))/average_within_m480.shape[0]
    PFC_area = [m478,m481,m486,m480]
    
    s = stats.ttest_ind(PFC_area,HP_area)
    
    plt.figure()
    sns.barplot(data=[HP_area,PFC_area], capsize=.1, ci="sd",  palette="Blues_d")
    
    return HP_area,PFC_area,s

    #sns.swarmplot(data=[HP_area,PFC_area], color="0", alpha=.35)
   
def svd_predict_another_animal(animal_1,animal_2, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, demean_all_tasks = True, z_score = False):
    
    if tasks_unchanged == True:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = svdu.demean_data(animal_1, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, z_score = False)
        
        flattened_all_clusters_task_1_first_half_2, flattened_all_clusters_task_1_second_half_2,\
        flattened_all_clusters_task_2_first_half_2, flattened_all_clusters_task_2_second_half_2,\
        flattened_all_clusters_task_3_first_half_2,flattened_all_clusters_task_3_second_half_2 = svdu.demean_data(animal_2, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, z_score = False)

    else:
        flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half = svdu.demean_data(animal_1, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, demean_all_tasks = demean_all_tasks, z_score = False)
   
        flattened_all_clusters_task_1_first_half_2, flattened_all_clusters_task_1_second_half_2,\
        flattened_all_clusters_task_2_first_half_2, flattened_all_clusters_task_2_second_half_2 = svdu.demean_data(animal_2, tasks_unchanged = tasks_unchanged, plot_a = plot_a, plot_b = plot_b, average_reward = average_reward, demean_all_tasks = demean_all_tasks, z_score = False)
   
    all_data_animal_1 = np.concatenate([flattened_all_clusters_task_1_first_half,flattened_all_clusters_task_1_second_half,\
                                        flattened_all_clusters_task_2_first_half,flattened_all_clusters_task_2_second_half,\
                                        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half], axis = 1)
    
    all_data_animal_2 = np.concatenate([flattened_all_clusters_task_1_first_half_2,flattened_all_clusters_task_1_second_half_2,\
                                        flattened_all_clusters_task_2_first_half_2,flattened_all_clusters_task_2_second_half_2,\
                                        flattened_all_clusters_task_3_first_half_2,flattened_all_clusters_task_3_second_half_2], axis = 1)
   
    #SVDsu.shape, s.shape, vh.shape for task 1 first half
    u_1, s_1, vh_1 = np.linalg.svd(all_data_animal_1, full_matrices = True)
        
    #SVDsu.shape, s.shape, vh.shape for task 1 second half
    u_2, s_2, vh_2 = np.linalg.svd(all_data_animal_2, full_matrices = True)
    
    
    #Finding variance explained in second half of task 1 using the Us and Vs from the first half   
    t_u = np.transpose(u_1)  
    t_v = np.transpose(vh_1)  

    t_u_t_2 = np.transpose(u_2)   
    t_v_t_2 = np.transpose(vh_2)  
    dd = np.linalg.multi_dot([t_u_t_2,t_v_t_2_2])

# =============================================================================
#     Between Task 1 and Task 2
# =============================================================================
      
    # Variance task 2 First Half from Task 1 Second Half
    # Using U
    x_task_2_from_task_1 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half])
    # Using V
    y_task_2_from_task_1 = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half,t_v_t_1_2])
    
    var_x_task_2_from_task_1 = np.sum(x_task_2_from_task_1**2, axis = 1)
    cum_var_x_task_2_from_task_1 = np.cumsum(var_x_task_2_from_task_1)/np.sqrt(flattened_all_clusters_task_2_first_half.shape[0])
    cum_var_x_task_2_from_task_1 = cum_var_x_task_2_from_task_1/cum_var_x_task_2_from_task_1[-1]
    
    var_y_task_2_from_task_1 = np.sum(y_task_2_from_task_1**2, axis = 0)
    cum_var_y_task_2_from_task_1 = np.cumsum(var_y_task_2_from_task_1)/flattened_all_clusters_task_2_first_half.shape[0]
    cum_var_y_task_2_from_task_1 = cum_var_y_task_2_from_task_1/cum_var_y_task_2_from_task_1[-1]
    
    # Variance task 2 First Half from First Half
    # Using U
    x_1_task_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half])
    # Using V
    y_1_task_2 = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half,t_v_t_2_2])
    
    var_rows_1_task_2 = np.sum(x_1_task_2**2, axis = 1)
    cum_x_1_task_2 = np.cumsum(var_rows_1_task_2)/np.sqrt(flattened_all_clusters_task_2_first_half.shape[0])
    cum_x_1_task_2 = cum_x_1_task_2/cum_x_1_task_2[-1]
    
    var_rows_y_1_task_2 = np.sum(y_1_task_2**2, axis = 0)
    cum_y_1_task_2 = np.cumsum(var_rows_y_1_task_2)/flattened_all_clusters_task_2_first_half.shape[0]    
    cum_y_1_task_2 = cum_y_1_task_2/cum_y_1_task_2[-1]
    
    area_x_task_2_from_task_1 = []
    for i in cum_var_x_task_2_from_task_1:
        if i == 0:
            area_x_task_2_from_task_1.append(np.trapz([0,i]))
        else:
            area_x_task_2_from_task_1.append(np.trapz([0,i]))
    
    #Calculating area proportion within  Task 1 and 2
    area_x_1_task_2 = []
    for i in cum_x_1_task_2:
        if i == 0:
            area_x_1_task_2.append(np.trapz([0,i]))
        else:
            area_x_1_task_2.append(np.trapz([0,i]))

#average_within_HP = np.mean([np.mean(average_within_m484),np.mean(average_within_m479), np.mean(average_within_m483)])
#average_within_PFC = np.mean([np.mean(average_within_m478),np.mean(average_within_m486), np.mean(average_within_m480),np.mean(average_within_m481)])

#average_between_HP = np.mean([np.mean(average_between_m484),np.mean(average_between_m479), np.mean(average_between_m483)])
#average_between_PFC = np.mean([np.mean(average_between_m478),np.mean(average_between_m486), np.mean(average_between_m480),np.mean(average_between_m481)])
