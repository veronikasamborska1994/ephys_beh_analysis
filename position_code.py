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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from os import path
import matplotlib.patches as mpatches


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

ephys_session = fu.load_data('m486_2018-07-23_19-25-22','/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/Ephys_Reversal_Learning/data/Ephys 3 Tasks Processed Spikes/m486/openfield','/',True)
clusters = ephys_session['spike_cluster'].unique()
xy = np.loadtxt(open('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/m486_open_fielxyz/openfield_m486_xy2018-07-23T19_25_14.csv'))
time_camera = np.loadtxt(open('/media/behrenslab/90c50efc-05cf-4045-95e4-9dabd129fb47/m486_open_fielxyz/openfield_m486_timestamps2018-07-23T19_25_14.csv'))
time_camera = np.asarray(time_camera,dtype=uint32)
converted_time = converttime(time_camera)
uncycled_time = uncycle(converted_time)

bin_edges_trial = np.arange(0 ,100, 50)
bin_width_ms = 50 
smooth_sd_ms = 100

for i,cluster in enumerate(clusters): 
    spikes_to_plot = []
    spikes = ephys_session.loc[ephys_session['spike_cluster'] == 50]
    spikes_times = np.array(spikes['time'])
    for c,coordinate in enumerate(xy):
        time_at_xy = time_camera[c]
        spikes_ind = spikes_times[(spikes_times >= -100) & (spikes_times<= +100)]        
        spikes_to_plot.append(spikes_ind) 
        hist,edges = np.histogram(spikes_to_plot, bins= bin_edges_trial)
        normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
