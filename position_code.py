#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:31:30 2018

@author: behrenslab
"""
import funcs as fu
import data_import as di
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from os import path
import matplotlib.patches as mpatches
import seaborn as sns

def converttime(time):
    #offset = time & 0xFFF
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


ephys_session = fu.load_data('m484_openfield_2018-08-03_15-54-12','/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Processed Spikes/m484/openfield','/',True)
clusters = ephys_session['spike_cluster'].unique()
xy = np.loadtxt(open('/media/behrenslab/My Book/video_PC_backup/m478/03_08/openfield/openfield_m478_xy2018-08-03T16_40_41.csv'))
time_camera = np.loadtxt(open('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/Video_data/m484/03_08/openfield/openfield_m484_timestamps2018-08-03T15_54_11.csv'))
time_camera = np.asarray(time_camera,dtype=uint32)
converted_time = converttime(time_camera)
uncycled_time = uncycle(converted_time)
uncycled_time =uncycled_time*1000

x =[]
y = []
for line in xy:
    x.append(line[0])
    y.append(line[1])
x= np.asarray(x)
y =np.asarray(y)
#x = x[np.nonzero(x)]  
#y = y[np.nonzero(y)]
    

for i,cluster in enumerate(clusters): 
    spikes_to_plot = []
    spikes_per_frame_list =[]
    assign_spike_to_xy = 0
    spikes = ephys_session.loc[ephys_session['spike_cluster'] == cluster]
    spikes_times = np.array(spikes['time'])
    spikes_times = spikes_times[~np.isnan(spikes_times)]
    for s,spike in enumerate(spikes_times):
        assign_spike_to_xy = (np.abs(uncycled_time - spike)).argmin()
        spikes_per_frame_list.append(assign_spike_to_xy)
    x_new = x[spikes_per_frame_list]
    y_new = y[spikes_per_frame_list]
    plt.figure()
    sns.set(style="white", palette="muted", color_codes = True)
    plt.grid(False)
    plt.plot(x,y,linewidth=0.5, color = 'lightgray', alpha = 0.7, zorder = 0)
    plt.scatter(x_new, y_new, s = 20, color = 'red', zorder = 1)
    plt.title('{}'.format(cluster))
   # plt.hist2d(x_new,y_new, bins = 100)