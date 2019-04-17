#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:28:36 2019

@author: veronikasamborska
"""
import scipy.cluster.hierarchy as shc
import numpy as np
import SVDs as sv 
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Hiearachical clustering (doesn't work reliably with our data)

flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)

flattened_all_clusters_task_1_first_half_HP, flattened_all_clusters_task_1_second_half_HP,\
flattened_all_clusters_task_2_first_half_HP, flattened_all_clusters_task_2_second_half_HP,\
flattened_all_clusters_task_3_first_half_HP,flattened_all_clusters_task_3_second_half_HP = sv.flatten(experiment_aligned_HP, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)

z = zscore(flattened_all_clusters_task_1_first_half_HP, axis = 1)
z_score_list = []
for i in z:
    if not any(np.isnan(i)):
        z_score_list.append(i)
 
z_score_list = np.asarray(z_score_list)   
#data = np.concatenate([flattened_all_clusters_task_1_first_half,flattened_all_clusters_task_1_first_half_HP], axis =0)

dend = shc.dendrogram(shc.linkage(z_score_list,method = 'ward'))

index_sorting = dend['leaves']

sorted_data = z_score_list[index_sorting]
colours = dend['color_list']
np.unique(colours)
plt.figure()
plt.imshow(sorted_data, aspect = 'auto')




