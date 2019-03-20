#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:18:50 2019

@author: veronikasamborska
"""

import SVDs as sv 
import heatmap_aligned as ha
import numpy as np
import ephys_beh_import as ep

    
#HP = m484 + m479 + m483
#PFC = m478 + m486 + m480 + m481

#ephys_path = '/Users/veronikasamborska/Desktop/neurons'
#beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
#
#HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
#
#experiment_aligned_m484 = ha.all_sessions_aligment(m484, all_sessions)
#experiment_aligned_m479 = ha.all_sessions_aligment(m479, all_sessions)
#experiment_aligned_m483 = ha.all_sessions_aligment(m483, all_sessions)
#experiment_aligned_m478 = ha.all_sessions_aligment(m478, all_sessions)
#experiment_aligned_m486 = ha.all_sessions_aligment(m486, all_sessions)
#experiment_aligned_m480 = ha.all_sessions_aligment(m480, all_sessions)
#experiment_aligned_m481 = ha.all_sessions_aligment(m481, all_sessions)


average_within_m484, average_between_m484 = sv.svd_plotting(experiment_aligned_m484, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = False, demean_all_tasks = False)
average_within_m479, average_between_m479 = sv.svd_plotting(experiment_aligned_m479, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = True, demean_all_tasks = False)
average_within_m483, average_between_m483 = sv.svd_plotting(experiment_aligned_m483, tasks_unchanged = True, plot_a = False, plot_b = False, HP = True, average_reward = False, diagonal = True, demean_all_tasks = False)
average_within_m478, average_between_m478 = sv.svd_plotting(experiment_aligned_m478, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = True, demean_all_tasks = False)
average_within_m481, average_between_m481 = sv.svd_plotting(experiment_aligned_m481, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = True, demean_all_tasks = False)
average_within_m486, average_between_m486 = sv.svd_plotting(experiment_aligned_m486, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = True, demean_all_tasks = False)
average_within_m480, average_between_m480 = sv.svd_plotting(experiment_aligned_m480, tasks_unchanged = True, plot_a = False, plot_b = False, HP = False, average_reward = False, diagonal = True, demean_all_tasks = False)


average_within_HP = np.mean([np.mean(average_within_m484),np.mean(average_within_m479), np.mean(average_within_m483)])
average_within_PFC = np.mean([np.mean(average_within_m478),np.mean(average_within_m486), np.mean(average_within_m480),np.mean(average_within_m481)])

average_between_HP = np.mean([np.mean(average_between_m484),np.mean(average_between_m479), np.mean(average_between_m483)])
average_between_PFC = np.mean([np.mean(average_between_m478),np.mean(average_between_m486), np.mean(average_between_m480),np.mean(average_between_m481)])
