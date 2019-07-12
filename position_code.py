#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:31:30 2018

@author: behrenslab
"""

# =============================================================================
#  Script for plotting openfield data; spikes as a scatter superimposed on
#  animal's trajectory
# =============================================================================

from preprocessing import data_import as di
import numpy as np
import matplotlib.pyplot as plt
from os import path
import datetime
from datetime import datetime
import os
import re
from scipy import interpolate
from scipy.ndimage import gaussian_filter

def converttime(time):
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds

def uncycle(time):
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

subject = 'm478'

ephys_path = '/Users/veronikasamborska/Desktop/open_field_neurons/'+ subject
ephys_folder = os.listdir(ephys_path)
ephys_folder = [file for file in ephys_folder if file !='time' and file!='xy_coords' and file !='.DS_Store']

xy_path = '/Users/veronikasamborska/Desktop/open_field_neurons/'+ subject+'/xy_coords'
xy_folder = os.listdir(xy_path)
time_path = '/Users/veronikasamborska/Desktop/open_field_neurons/'+ subject+'/time'
time_folder = os.listdir(time_path)

for file_ephys in ephys_folder:
    match_ephys = re.search(r'\d{4}-\d{2}-\d{2}', file_ephys)
    date_ephys = datetime.strptime(match_ephys.group(), '%Y-%m-%d').date()
    date_ephys = match_ephys.group()
    for file_xy in xy_folder:
        match_behaviour = re.search(r'\d{4}-\d{2}-\d{2}', file_xy)
        date_xy = datetime.strptime(match_behaviour.group(), '%Y-%m-%d').date()
        date_xy = match_behaviour.group()
        
        for file_time in time_folder:
            match_time = re.search(r'\d{4}-\d{2}-\d{2}', file_time)
            date_time = datetime.strptime(match_time.group(), '%Y-%m-%d').date()
            date_time = match_time.group()
            
            if date_ephys == date_xy and date_ephys == date_time:

                neurons = np.load(ephys_path+'/'+file_ephys)
                neurons = neurons[:,~np.isnan(neurons[1,:])]
                
                ## Extract neuron identities 
                clusters =np.unique(neurons[0,:])
     
                xy = np.loadtxt(open(xy_path+'/'+file_xy))
                time_camera = np.loadtxt(open(time_path+'/'+file_time))
                frames = np.arange(len(time_camera))*(1000/30)
                
                x =[]
                y = []
                for line in xy:
                    x.append(line[0])
                    y.append(line[1])
                x = np.asarray(x)
                y =np.asarray(y)
                mask = ~np.isnan(x)

                x = x[mask]
                y = y[mask]
                uncycled_time = frames[mask]

                mask_zero = np.nonzero(x)
               
                x = x[mask_zero]
                y = y[mask_zero]
                uncycled_time = frames[mask_zero]

                for i,cluster in enumerate(clusters): 
                    spikes_to_plot = []
                    spikes_per_frame_list =[]
                    assign_spike_to_xy = 0      
                    #Indicies where this neuron fired
                    cluster_ind = np.where(neurons[0,:]== cluster)
                    #Spike times in bonsai time 
                    spikes = neurons[1,cluster_ind]
                    
                    for s,spike in enumerate(spikes[0,:]):
                        assign_spike_to_xy = (np.abs(uncycled_time - spike)).argmin()
                        spikes_per_frame_list.append(assign_spike_to_xy)
                   
                    x_new = x[spikes_per_frame_list]
                    y_new = y[spikes_per_frame_list]
                    plt.figure()
                    plt.scatter(x,y, s= 2, c = 'grey')
                    plt.scatter(x_new, y_new, s = 2, c = 'green')
                   
#                    H_occupancy, xedges_occupancy, yedges_occupancy = np.histogram2d(x,y, bins = 100)
#                    H, xedges, yedges = np.histogram2d(x_new,y_new, bins = 100)
#                    normalise = H/H_occupancy
#                    V=normalise.copy()
#                    V[np.isnan(normalise)]=0
#                    V[np.isinf(normalise)]=0
#
#                    VV=gaussian_filter(V,sigma = 2) 
#                    plt.figure()
#                    plt.imshow(VV)
       
                    
                    
                    
    