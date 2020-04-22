## Sctipt to load organised data matrix

import sys
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/modelling')

import scipy.io

import heatmap_aligned as ha
import RW_model_fitting as mfit
import create_data_arrays_for_tim as cda
import forced_trials_extract_data as ft
import ephys_beh_import as ep


ephys_path = '/Users/veronikasamborska/Desktop/neurons'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = ep.import_code(ephys_path,beh_path,lfp_analyse = 'False')
experiment_aligned_PFC = ha.all_sessions_aligment(PFC, all_sessions)
experiment_aligned_HP = ha.all_sessions_aligment(HP, all_sessions)

PFC_forced = ft.all_sessions_aligment_forced(PFC,all_sessions)
PFC_forced = ft.all_sessions_aligment_forced(HP,all_sessions)
#
#experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP ,experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP,\

#experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC = mfit.run(experiment_aligned_HP,experiment_aligned_PFC)
#
data_PFC = cda.tim_create_mat(experiment_aligned_PFC,experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC,experiment_sim_Q1_prediction_error_chosen_PFC, 'PFC_RPE') 
data_HP = cda.tim_create_mat(experiment_aligned_HP, experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP, experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP,bayes_prior_HP,experiment_sim_Q1_prediction_error_chosen_HP,  'HP_RPE')

