#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:59:35 2019

@author: veronikasamborska
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore


def extract_rest_periods(session):
    index_start_rest = []
    index_end_rest = []
    pyControl_choice = [event.time for event in session.events if event.name in ['choice_state']]
    pyControl_choice = np.array(pyControl_choice)
          
    for c,choice in enumerate(pyControl_choice):
        if c < (len(pyControl_choice)-1):
            if (pyControl_choice[c+1] -pyControl_choice[c])/36000 > 0:
                index_start_rest.append(pyControl_choice[c])
                index_end_rest.append(pyControl_choice[c+1])
    return index_start_rest, index_end_rest

 
    
def find_big_breaks_in_tasks(experiment):   
    hist_list = []
    for session in experiment:
        spike_list = [] 
        neurons = np.unique(session.ephys[0])
        index_start_rest, index_end_rest = extract_rest_periods(session)
        if len(index_start_rest) > 0:
            index_end_rest = index_start_rest[0]+1080000
            for neuron in neurons:
                spikes_times = session.ephys[1][np.where(session.ephys[0] == neuron)]
                for spike in spikes_times:
                    if spike > index_start_rest[0] and spike < index_end_rest:
                        spike_list.append(spike)
            hist, bin_edges = np.histogram(spike_list, bins = 1800)
            hist_list.append(hist)
    return hist_list


hist_list_pfc = find_big_breaks_in_tasks(HP)
hist_list_pfc = np.asarray(hist_list_pfc)

z = zscore(hist_list_pfc, axis = 1)

z_score_list = []
for i in z:
    if not any(np.isnan(i)):
        z_score_list.append(i)
 
z_score_list = np.asarray(z_score_list)   
correlation_pfc = np.linalg.multi_dot([z_score_list,np.transpose(z_score_list)])

w_pfc,pc_pfc = np.linalg.eig(correlation_pfc)

sorting = np.argsort(abs(pc_pfc[:,0]))

sort_z_score = z_score_list[sorting]
plt.figure(2)
plt.imshow(sort_z_score, aspect= 'auto')

plt.figure(figsize=(20,10))
plt.plot(pc_pfc[0], color = 'green')     
#plt.xlabel ('Time (sec)')
#plt.title('PC1 PFC')