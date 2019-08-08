#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:26:20 2018

@author: behrenslab
"""


import data_import as di
import OpenEphys as op 
import Esync as es
import numpy as np

session = di.Session('/home/behrenslab/ephys/2018-06-05-reversal_learning_3_tasks_recording/m483-2018-06-07-161545.txt')

rsync_events = [event.time for event in session.events if event.name in ['Rsync']]
rsync_ephys = op.loadEvents('/home/behrenslab/ephys/2018-06-07_16-15-43/all_channels.events')
rsync_timestamps = rsync_ephys['timestamps']
rsync_timestamps= np.array(rsync_timestamps)[::2]

rsync_events = np.array(rsync_events)
rsync_timestamps =rsync_timestamps/30
aligner = es.Rsync_aligner(rsync_events, rsync_timestamps, plot=True)
times_B = aligner.A_to_B(rsync_events) #A pycontrol
times_A = aligner.B_to_A(rsync_timestamps)
                         
pycontrol_to_ephys = aligner.A_to_B