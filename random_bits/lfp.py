p#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:26:09 2018

@author: veronikasamborska
"""
from scipy import fftpack
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import freqs
import ephys_beh_import as ep
import neuron_firing_all_pokes as nf
import ephys_beh_import as ep
from scipy.signal import hilbert, chirp
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')

import neurodsp
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time

#HP,PFC, m484, m479, m483, m478, m486, m480, m481 = ep.import_code(ephys_path,beh_path)
#experiment_aligned_HP = ha.all_sessions_aligment(HP)
def trial_start_end_times(s):
    task = s.trial_data['task']
    forced_trials = s.trial_data['forced_trial']
    non_forced_array = np.where(forced_trials == 0)[0] 
    
    task_non_forced = task[non_forced_array]
    task_1 = np.where(task_non_forced == 1)[0]
    task_2 = np.where(task_non_forced == 2)[0] 
    task_3 = np.where(task_non_forced == 3)[0]
    task_2_change = np.where(task ==2)[0]
    task_3_change = np.where(task ==3)[0]

    poke_A = 'poke_'+str(s.trial_data['poke_A'][0])
    poke_A_task_2 = 'poke_'+str(s.trial_data['poke_A'][task_2_change[0]])
    poke_A_task_3 = 'poke_'+str(s.trial_data['poke_A'][task_3_change[0]])
    poke_B = 'poke_'+str(s.trial_data['poke_B'][0])
    poke_B_task_2  = 'poke_'+str(s.trial_data['poke_B'][task_2_change[0]])
    poke_B_task_3 = 'poke_'+str(s.trial_data['poke_B'][task_3_change[0]])
    
    events_and_times = [[event.name, event.time] for event in s.events if event.name in ['choice_state',poke_B, poke_A, poke_A_task_2,poke_A_task_3, poke_B_task_2,poke_B_task_3]]
    poke_B_list = []
    poke_A_list = []
    choice_state = False 
    for event in events_and_times:
        choice_state_count = 0
        if 'choice_state' in event:
            choice_state_count +=1 
            choice_state = True   
        if choice_state_count <= len(task_1):
            if poke_B in event: 
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False
        elif choice_state_count > len(task_1) and choice_state_count <= (len(task_1) +len(task_2)):
            if poke_B_task_2 in event:
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A_task_2 in event:
                if choice_state == True:    
                    poke_A_list.append(event[1])
                    choice_state = False     
        elif choice_state_count > (len(task_1) +len(task_2)) and choice_state_count <= (len(task_1) + len(task_2) + len(task_3)):
            if poke_B_task_3 in event:
                if choice_state == True:
                    poke_B_list.append(event[1])
                    choice_state = False
            elif poke_A_task_3 in event:
                if choice_state == True:
                    poke_A_list.append(event[1])
                    choice_state = False
                    
    all_events = poke_A_list + poke_B_list
    return all_events 
    
    

#LFP analysis 

#Low and high pass filters 
def butter_lowpass(cutoff_low, fs, order = 10):
    nyq = 0.5 * fs
    normalCutoff = cutoff_low / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutoff_low, fs, order = 9):
    b, a = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff_high, fs, order = 10):
    nyq = 0.5 * fs
    normalCutoff = cutoff_high / nyq
    b, a = butter(order, normalCutoff, btype='high', analog = True)
    return b, a

def butter_highpass_filter(data, cutoff_high, fs, order = 9):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    y = lfilter(b, a, data)
    return y

session_n = 1
# Filter requirements.
fs = 500       # sample rate, Hz
cutoff_low = 10   # desired cutoff frequency of the filter, Hz
cutoff_high = 6   # desired cutoff frequency of the filter, Hz

s = HP[session_n]
lfp = s.lfp
lfp_time = s.lfp_time
channel_1 = lfp[10,:]
channel_1_time = lfp_time[10,:]
frequency = fftpack.fft(channel_1)
signalPSD = np.abs(frequency) ** 2

#y = butter_lowpass_filter(channel_1, cutoff_low, fs)

#channel_1 = butter_highpass_filter(y,cutoff_high,fs)

#Frequency power over time 
#f, t, Zxx = signal.stft(y,fs = 500)
#plt.pcolormesh(t, f, np.abs(Zxx))

Fs = 500
f_range = (6,10)

raw_spikes = s.ephys
neurons = np.unique(raw_spikes[0])
pyControl_choice = [event.time for event in s.events if event.name in ['choice_state']]
spikes_list,session_duration_ms =  nf.session_spikes_vs_trials_plot(raw_spikes,pyControl_choice)
all_events  = trial_start_end_times(s)

lfp_trials_list = []
lfp_times_list = []

session_spike_list = []

for choice in all_events:
    ind_min = choice - 1000
    ind_max = choice + 1000
    lfp_ind_min = np.where(channel_1_time == channel_1_time.flat[np.abs(channel_1_time - ind_min).argmin()])
    lfp_ind_max = np.where(channel_1_time == channel_1_time.flat[np.abs(channel_1_time - ind_max).argmin()])
    lfp_trial = channel_1[lfp_ind_min[0][0]:lfp_ind_max[0][0]] 
    lfp_time = channel_1_time[lfp_ind_min[0][0]:lfp_ind_max[0][0]] 
    lfp_times_list.append(lfp_time)
    lfp_trials_list.append(lfp_trial)
    neurons_range = range(shape(spikes_list)[0])
    neuron_list= []
    

    for neuron in neurons_range:
        spikes = spikes_list[neuron] 
        neuron_spike_list = []
        for spike in spikes:
            if spike > ind_min and spike < ind_max:  
                neuron_spike_list.append(spike)
        neuron_list.append(neuron_spike_list) 
    session_spike_list.append(neuron_list)

trial = 295
session_spike_list = np.asarray(session_spike_list)
one_trial = session_spike_list[trial,:]
pha = phase_by_time(lfp_times_list[trial], Fs, f_range)

min_time = min(lfp_times_list[trial])
max_time = max(lfp_times_list[trial])
colors = ["red", "orange", "grey", "green", "blue", "purple", "black"]
axmin = 500
axmax = 600
for neuron in neurons_range:
    if len(one_trial[neuron])> 0:
        if neuron == 2:
            axmin+=150
            axmax+=150
            vlines(one_trial[neuron], ymin = axmin, ymax = axmax,color= colors[neuron] )
            xlim(min_time,max_time)
        
plot(lfp_times_list[trial],lfp_trials_list[trial])
