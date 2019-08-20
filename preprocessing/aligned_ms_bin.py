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

#experiment_aligned_PFC_ms_bin = ha.all_sessions_aligment(PFC, all_sessions, fs = 1000)
#experiment_aligned_HP_ms_bin = ha.all_sessions_aligment(HP, all_sessions, fs = 1000)

#data_PFC_ms = cda.tim_create_mat(experiment_aligned_PFC_ms_bin, experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC, 'PFC') 
#data_HP_ms = cda.tim_create_mat(experiment_aligned_HP_ms_bin, experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP, experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP, 'HP')

#PFC_forced = ft.all_sessions_aligment_forced(PFC,all_sessions, fs = 1000)
#HP_forced = ft.all_sessions_aligment_forced(HP,all_sessions, fs = 1000)


# C = 2πr; 

session =experiment_aligned_HP_ms_bin[0]
C = session.aligned_rates.shape[2]
p = math.pi
r =  C/(2*p)

angle = []
for i in range(C):
    L = 0+ (i+1)
    ang = (180*L)/(p*r)
    angle.append(ang)