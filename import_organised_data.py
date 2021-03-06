## Sctipt to load organised data matrix

import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/modelling')

import scipy.io
import pylab as plt
import heatmap_aligned as ha
import RW_model_fitting as mfit
import create_data_arrays_for_tim as cda
import forced_trials_extract_data as ft
import ephys_beh_import as ep
import numpy as np

ephys_path = '/Users/veronikasamborska/Desktop/neurons'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')

experiment_aligned_PFC = ha.all_sessions_aligment(PFC, all_sessions)
experiment_aligned_HP = ha.all_sessions_aligment(HP, all_sessions)

#PFC_forced = ft.all_sessions_aligment_forced(PFC,all_sessions) 
#HP_forced = ft.all_sessions_aligment_forced(HP,all_sessions)


# experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP ,experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP,\
# experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC = mfit.run(experiment_aligned_HP,experiment_aligned_PFC)


# data_PFC = cda.tim_create_mat(experiment_aligned_PFC,experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC, 'PFC_RPE') 
# data_HP = cda.tim_create_mat(experiment_aligned_HP, experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP, experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP,  'HP_RPE')

#data_PFC = cda.tim_create_mat(experiment_aligned_PFC,'PFC')
#data_HP = cda.tim_create_mat(experiment_aligned_HP, 'HP')

initiate_choice_t = experiment_aligned_PFC[0].target_times 
t_out = experiment_aligned_PFC[0].t_out 

reward = initiate_choice_t[2] + 250
ind_init = (np.abs(t_out-initiate_choice_t[1])).argmin()
ind_choice = (np.abs(t_out-initiate_choice_t[2])).argmin()
ind_reward = (np.abs(t_out-reward)).argmin()
ind_iti = (np.abs(t_out-initiate_choice_t[3])).argmin()

HP = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
PFC = scipy.io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat') 
         
    