#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:22:57 2019

@author: veronikasamborska
"""
import numpy as np
import SVDs as sv
import matplotlib.pyplot as plt

# Spectral Ordering 

flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)


task_1 = flattened_all_clusters_task_1_first_half+flattened_all_clusters_task_1_second_half/2
task_2 = flattened_all_clusters_task_2_first_half+flattened_all_clusters_task_2_second_half/2
task_3 = flattened_all_clusters_task_3_first_half+flattened_all_clusters_task_3_second_half/2

all_data = np.concatenate([task_1, task_2,task_3], axis = 1)                         

A = all_data

G = np.dot(A, np.transpose(A))/2
G = np.power(G,3)
Q = -G
Q = np.triu(Q,1) + np.tril(Q,-1)
Q = Q - np.diag(sum(Q))
 
t = 1/np.sqrt(sum(G))
Q = np.dot(np.diag(t),Q)
Q = np.dot(Q,np.diag(t))

v,d = np.linalg.eig(Q)

D = np.diag(v)
 
v2 = d[:,1]
 
v2scale = t*v2
 
ind = np.argsort(v2scale)
sorted_data = all_data[ind]

plt.imshow(sorted_data)

#Plot chunks of 10 neurons

trial_length = 64
n_neurons = 10
HP_range = np.arange(40)
HP_range = HP_range[1:]
HP_cut = np.arange(39)

PFC_range = np.arange(57)
PFC_range = PFC_range[1:]
PFC_cut = np.arange(56)

fig = plt.figure(figsize=(30,60))

for end, chunks in zip(PFC_range,PFC_cut):
    print(chunks)
    print(end)

    mean_task_1_a_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons,:trial_length],axis = 0)
    mean_task_1_a_nr =  np.mean(sorted_data[chunks*n_neurons:end*n_neurons,trial_length:trial_length*2],axis = 0)
    
    mean_task_1_b_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons,trial_length*2:trial_length*3],axis = 0)
    mean_task_1_b_nr =  np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*3:trial_length*4],axis = 0)
    
    mean_task_2_a_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*4: trial_length*5],axis = 0)
    mean_task_2_a_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*5: trial_length*6],axis = 0)
    
    mean_task_2_b_r= np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*6: trial_length*7],axis = 0)
    mean_task_2_b_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*7: trial_length*8],axis = 0)
    
    mean_task_3_a_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*8: trial_length*9],axis = 0)
    mean_task_3_a_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*9: trial_length*10],axis = 0)
    
    mean_task_3_b_r = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*10: trial_length*11],axis = 0)
    mean_task_3_b_nr = np.mean(sorted_data[chunks*n_neurons:end*n_neurons, trial_length*11: trial_length*12],axis = 0)

    fig.add_subplot(6,10,end)
    plt.plot(mean_task_1_a_r,  color = 'darkblue')
    plt.plot(mean_task_1_a_nr,linestyle ='--', color = 'darkblue')
    
    plt.plot(mean_task_1_b_r, color = 'darkred')
    plt.plot(mean_task_1_b_nr,linestyle =  '--', color = 'darkred')
    
    plt.plot(mean_task_2_a_r, color = 'lightblue')
    plt.plot(mean_task_2_a_nr, linestyle = '--', color = 'lightblue')
    
    plt.plot(mean_task_2_b_r, color = 'pink')
    plt.plot(mean_task_2_b_nr, linestyle = '--', color = 'pink')
    
    plt.plot(mean_task_3_a_r, color = 'royalblue')
    plt.plot(mean_task_3_a_nr,linestyle = '--', color = 'royalblue')
    
    plt.plot(mean_task_3_b_r, color = 'coral')
    plt.plot(mean_task_3_b_nr,linestyle = '--', color = 'coral')
    
    if end == 56:     
        mean_task_1_a_r = np.mean(sorted_data[end*n_neurons:,:trial_length],axis = 0)
        mean_task_1_a_nr =  np.mean(sorted_data[end*n_neurons:,trial_length:trial_length*2],axis = 0)
        
        mean_task_1_b_r = np.mean(sorted_data[end*n_neurons:,trial_length*2:trial_length*3],axis = 0)
        mean_task_1_b_nr =  np.mean(sorted_data[end*n_neurons:, trial_length*3:trial_length*4],axis = 0)
        
        mean_task_2_a_r = np.mean(sorted_data[end*n_neurons:, trial_length*4: trial_length*5],axis = 0)
        mean_task_2_a_nr = np.mean(sorted_data[end*n_neurons:, trial_length*5: trial_length*6],axis = 0)
        
        mean_task_2_b_r= np.mean(sorted_data[end*n_neurons:, trial_length*6: trial_length*7],axis = 0)
        mean_task_2_b_nr = np.mean(sorted_data[end*n_neurons:, trial_length*7: trial_length*8],axis = 0)
        
        mean_task_3_a_r = np.mean(sorted_data[end*n_neurons:, trial_length*8: trial_length*9],axis = 0)
        mean_task_3_a_nr = np.mean(sorted_data[end*n_neurons:, trial_length*9: trial_length*10],axis = 0)
        
        mean_task_3_b_r = np.mean(sorted_data[end*n_neurons:, trial_length*10: trial_length*11],axis = 0)
        mean_task_3_b_nr = np.mean(sorted_data[end*n_neurons:, trial_length*11: trial_length*12],axis = 0)
    
        fig.add_subplot(6,10,57)
        plt.plot(mean_task_1_a_r, label = 'A T1 R', color = 'darkblue')
        plt.plot(mean_task_1_a_nr,label = 'A T1 NR',linestyle ='--', color = 'darkblue')
        
        plt.plot(mean_task_1_b_r, label = 'B T1 R', color = 'darkred')
        plt.plot(mean_task_1_b_nr, label = 'B T1 NR',linestyle =  '--', color = 'darkred')
        
        plt.plot(mean_task_2_a_r, label = 'A T2 R', color = 'lightblue')
        plt.plot(mean_task_2_a_nr, label = 'A T2 NR', linestyle = '--', color = 'lightblue')
        
        plt.plot(mean_task_2_b_r,label = 'B T2 R', color = 'pink')
        plt.plot(mean_task_2_b_nr, label = 'B T2 NR', linestyle = '--', color = 'pink')
        
        plt.plot(mean_task_3_a_r, label = 'A T3 R', color = 'royalblue')
        plt.plot(mean_task_3_a_nr, label = 'A T3 NR',linestyle = '--', color = 'royalblue')
        
        plt.plot(mean_task_3_b_r, label = 'B T3 R', color = 'coral')
        plt.plot(mean_task_3_b_nr, label = 'B T3 NR',linestyle = '--', color = 'coral')
#plt.subplots_adjust(left=0.01, bottom=0.1 , right=0.9 , top=0.09)
#plt.legend()