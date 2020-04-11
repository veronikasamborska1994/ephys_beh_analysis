#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:33:52 2019

@author: veronikasamborska
"""
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import hilbert
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing/')
import ephys_beh_import as ep
from scipy.signal import filtfilt, hilbert, remez
import numpy as np
import scipy.io
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
#import remapping_count as rc 
from scipy import stats
import psutil
from scipy.ndimage import gaussian_filter1d



from matplotlib import colors as mcolors
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

def import_data():
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

    HP_LFP,PFC_LFP, m484_LFP, m479_LFP, m483_LFP, m478_LFP, m486_LFP, m480_LFP, m481_LFP, all_sessions_LFP = ep.import_code(ephys_path,beh_path, lfp_analyse = 'True') 
    HP = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = scipy.io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    Data_HP = HP['Data'][0]
    DM_HP = HP['DM'][0]
    Data_PFC = PFC['Data'][0]
    DM_PFC = PFC['DM'][0]
    
def theta_bandpass_filter(sampling_frequency):
    ORDER = 9
    nyquist = 0.5 * sampling_frequency
    TRANSITION_BAND = 2
    RIPPLE_BAND = [4, 12]
    desired = [0, RIPPLE_BAND[0] - TRANSITION_BAND, RIPPLE_BAND[0],
               RIPPLE_BAND[1], RIPPLE_BAND[1] + TRANSITION_BAND, nyquist]
    return remez(ORDER, desired, [0, 1, 0], Hz=sampling_frequency), 1.0

def filter_theta_band(data, sampling_frequency=1500):
    '''Returns a bandpass filtered signal between 150-250 Hz

    Parameters
    ----------
    data : array_like, shape (n_time,)

    Returns
    -------
    filtered_data : array_like, shape (n_time,)

    '''
    filter_numerator, filter_denominator = theta_bandpass_filter(
        sampling_frequency)
    is_nan = np.isnan(data)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, data[~is_nan], axis=0)
    return filtered_data

def get_theta_envelope(data, axis=0):
    '''Extracts the instantaneous amplitude (envelope) of an analytic
    signal using the Hilbert transform'''
    n_samples = data.shape[axis]
    instantaneous_amplitude = np.abs(
        hilbert(data, N=next_fast_len(n_samples), axis=axis))
    return np.take(instantaneous_amplitude, np.arange(n_samples), axis=axis)


def plot_tuning_curves(aligned_rates_choices, session_DM):
    
  
    choices = session_DM[:,1]
    task =  session_DM[:,5]
    a_pokes =  session_DM[:,6]
    b_pokes =  session_DM[:,7]
    taskid = rc.task_ind(task,a_pokes,b_pokes)
       
    forced = session_DM[:,3]
    choices = choices[np.where(forced == 0)]
    taskid = taskid[np.where(forced == 0)]

    a1_ind = [np.where((choices == 1) & (taskid ==1))[0]]
    b1_ind = [np.where((choices == 0) & (taskid ==1))[0]]
    
    a2_ind = [np.where((choices == 1) & (taskid ==2))[0]]
    b2_ind = [np.where((choices == 0) & (taskid ==2))[0]]

    a3_ind = [np.where((choices == 1) & (taskid ==3))[0]]
    b3_ind = [np.where((choices == 0) & (taskid ==3))[0]]


    return a1_ind, b1_ind,a2_ind, b2_ind,a3_ind, b3_ind,taskid,task

def theta(LFP,HP, subj = 'm484'):
    
    subj = 'm484'
      
    if subj == 'm484':  
        Data = HP['Data'][0,:16]
        DM = HP['DM'][0,:16]
    elif subj == 'm479':  
        Data = HP['Data'][0,16:24]
        DM = HP['DM'][0,16:24]
    elif subj == 'm483':  
        Data = HP['Data'][0,24:]
        DM = HP['DM'][0,24:]
        
   
    for s,session in enumerate(m484_LFP):
        
        session_spikes = Data[s]
        dm_session = DM[s]
        LFPs = session.lfp.T
        t = session.lfp_time
        spikes = session.ephys
        
        times_start, times_end = trial_start_end_times(session)
        
        
        a1_ind, b1_ind, a2_ind, b2_ind, a3_ind, b3_ind,taskid,task = plot_tuning_curves(session_spikes, dm_session)
       
        task_1 = np.where(taskid == 1)[0]
        task_2 = np.where(taskid == 2)[0]
        task_3 = np.where(taskid == 3)[0]
        
      
        # Pick a channel of LFP 
        x = LFPs[:,0]    
        
        # Sampling Frequency
        fs = 1000
        x_filt = filter_theta_band(x, fs)
        x_a = hilbert(x_filt)                    
        x_phase = np.angle(x_a)
        plt.plot(x_phase)

        theta_array_neuron_neuron = []
        phase_neuron_neuron = []

        # Last 3 choices are not aligned ---> not sure why yet; everything else is aligned
        #print(np.max(t)-np.max(times_end[-3]))

       # times_start = times_start[:-3]
       # times_end = times_end[:-3]
        task_1_st = times_start[task_1]
        task_1_end = times_end[task_1]
        
        

        starts_neuron = []
        for neuron,n in enumerate(np.unique(spikes[0,:])):
            theta_array_neuron_event = []
            phase_neuron_event = []
            starts_event = []
            for start,end in zip(task_1_st,task_1_end): 
                starts_event.append(start)
                nearest_start = (np.abs(t - start)).argmin()
                nearest_end = (np.abs(t - end)).argmin()

                lfp_event = x[nearest_start:nearest_end]
                time_trial = t[nearest_start:nearest_end]
                if len(lfp_event) > 0:
                    x_filt = filter_theta_band(lfp_event, fs)
                    x_a = hilbert(x_filt)
                    
                    x_phase = np.angle(x_a)
                    spikes_n = spikes[1,np.where(spikes[0,:] == n)][0,:]
                    spikes_per_trial = []
    
                    for spike in spikes_n:
                        if spike > t[nearest_start] and spike < t[nearest_end]:
                            spikes_per_trial.append(spike)
                    phase_ind  = []
                    for spike in spikes_per_trial:
                        phase_ind.append((np.abs(time_trial - spike)).argmin())
                    phase  = x_phase[phase_ind]
                    theta_array_neuron_event.append(spikes_per_trial)
                    phase_neuron_event.append(phase)
                
            theta_array_neuron_neuron.append(theta_array_neuron_event)
            phase_neuron_neuron.append(phase_neuron_event)
            starts_neuron.append(starts_event)
            
        neuron = 0
        for n,p,s in zip(theta_array_neuron_neuron,phase_neuron_neuron,starts_neuron):
            plt.figure()
            reg_array = []
            
            for event_spike,event_phase, start in zip(n,p, s):
                if len(event_spike) > 0:
                    reg_array.append([(event_spike- start),event_phase])
                    
            r = np.concatenate(reg_array,1)
            
            plt.subplot(121)
            plt.scatter(r[0,:],r[1,:], 5)
            slope, intercept, r_value, p_value, std_err = stats.linregress(r[0,:],r[1,:])
            xs = np.arange(0,int(np.max(r[0,:])))
            regression_line = [(slope*x)+intercept for x in xs]
            plt.plot(regression_line, label = 'R value = ' + str(round(r_value,3)) + '   '+'P-value = ' + str(round(p_value,3)))
            plt.legend()
            plt.subplot(122)

            plt.plot(np.mean(session_spikes[task_1,neuron,:],  axis = 0),color = 'red', linestyle = '--', label = 'Task 1')
            plt.plot(np.mean(session_spikes[task_2,neuron,:],  axis = 0),color = 'pink',linestyle = '--', label = 'Task 2')
            plt.plot(np.mean(session_spikes[task_3,neuron,:],  axis = 0),color = 'purple',linestyle = '--', label =  'Task 3')

            neuron += 1
            plt.legend()  
            
 

def trial_start_end_times(s):
 
    choice_states = [event.time for event in s.events if event.name in ['choice_state']]
                 
    state_start =  np.asarray(choice_states) - 500
    state_end =  np.asarray(choice_states) + 500

    return state_start,state_end