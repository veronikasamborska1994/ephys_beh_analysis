#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:31:52 2018

@author: veronikasamborska
"""

import numpy as np
#import ephys_beh_import as ep
import heatmap_aligned as ha
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d as gs


# Function for finding the dot product of two vectors 
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))




def coefficient_projection(experiment):
     # Choice Beta Loadings Difference between A and B 
     predictors, C, X, y,cpd = ha.regression(experiment)
     for i,session in enumerate(experiment):
         aligned_spikes= session.aligned_rates[:]
         n_trials, n_neurons, n_timepoints = aligned_spikes.shape 
         t_out = session.t_out
         initiate_choice_t = session.target_times #Initiation and Choice Times
         ind_choice = (np.abs(t_out-initiate_choice_t[-2])).argmin() # Find firing rates around choice
         ind_after_choice = ind_choice + 7 # 1 sec after choice
         spikes_around_choice = aligned_spikes[:,:,ind_choice-2:ind_after_choice] # Find firing rates only around choice      
         mean_spikes_around_choice  = np.mean(spikes_around_choice,axis =2)
         
    
        

