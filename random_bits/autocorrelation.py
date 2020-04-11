#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:28:44 2019

@author: veronikasamborska
"""

#Autocorrelation plots 

import numpy as np
import SVDs as sv 
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_acf

# Hiearachical clustering (doesn't work reliably with our data)

#flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
#flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
#flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)

flattened_all_clusters_task_1_first_half_HP, flattened_all_clusters_task_1_second_half_HP,\
flattened_all_clusters_task_2_first_half_HP, flattened_all_clusters_task_2_second_half_HP,\
flattened_all_clusters_task_3_first_half_HP,flattened_all_clusters_task_3_second_half_HP = sv.flatten(experiment_aligned_HP, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)


def autocorr(x):
    result = np.correlate(x, x, mode = 2)
    
    return result[int(result.shape[0]/2):]

coef_list_n = []
data = np.concatenate([flattened_all_clusters_task_1_first_half_HP,flattened_all_clusters_task_2_first_half_HP,flattened_all_clusters_task_3_first_half_HP], axis = 1)
for i in range(data.shape[0]):
    x = data[i]#,:64]
    x = (x - np.nanmean(x))#/ (np.nanstd(x))

    n_time = x.shape
    coef_list = []
    a = autocorr(x)
    a /= a[0]

    coef_list_n.append(a)
    
coef_list_n = np.asarray(coef_list_n)
mean_corr = np.nanmedian(coef_list_n, axis = 0)

plt.plot(mean_corr)
#plot_acf(x,lags=255)
