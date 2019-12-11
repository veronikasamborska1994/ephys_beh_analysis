#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:56:24 2019

@author: veronikasamborska
"""

from sklearn.linear_model import LinearRegression
import sys
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis')
from ripple_times import detectors as dt
import numpy as np
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
import ephys_beh_import as ep
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_import/regressions/')
import regressions as reg 
import regression_function as re
import scipy.io
import poke_aligned_spikes as pk
from matplotlib import colors as mcolors
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
import remapping_count as rc 
from ripple_times import core
import itertools

#all_sessions_1, all_sessions_2, all_sessions_3 =  ripple_plot(peak_power_all_m483, m483_LFP,HP)

#   predictors_all = OrderedDict([
#                          ('latent_state',state),
#                          ('choice',choices_forced_unforced ),
#                          ('reward', outcomes),
#                          ('forced_trials',forced_trials),
#                          ('block', block),
#                          ('task',task),
#                          ('A', a_pokes),
#                          ('B', b_pokes),
#                          ('Initiation', i_pokes),
#                          #('Chosen_Simple_RW',chosen_Q1),
#                          #('Chosen_Cross_learning_RW', chosen_Q4),
#                          #('Value_A_RW', Q1_value_a),
#                          #('Value_B_RW', Q1_value_b),
#                          #('Value_A_Cross_learning', Q4_value_a),
#                          ('ones', ones)])
#            
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)

def import_data():
    ephys_path = '/Users/veronikasamborska/Desktop/neurons'
    beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'
    HP = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
#   HP = m484 + m479 + m483

    HP_LFP,PFC_LFP, m484_LFP, m479_LFP, m483_LFP, m478_LFP, m486_LFP, m480_LFP, m481_LFP, all_sessions_LFP = ep.import_code(ephys_path,beh_path, lfp_analyse = 'True') 
    all_times_m484_LFP, filtered_LFP_m484_LFP, peak_power_all_m484 = ripple_detect(m484_LFP)
    all_times_m483_LFP, filtered_LFP_m483_LFP, peak_power_all_m483 = ripple_detect(m483_LFP)
    HP = scipy.io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = scipy.io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    Data_HP = HP['Data'][0]
    DM_HP = HP['DM'][0]
    Data_PFC = PFC['Data'][0]
    DM_PFC = PFC['DM'][0]
    
    all_sessions_1, all_sessions_2, all_sessions_3 =  ripple_plot(peak_power_all_m484, m484_LFP,HP)

def ripple_detect(LFP):
    
    all_times = []
    peak_power_all = []
    for session in LFP:
        LFPs = session.lfp.T
        time = session.lfp_time

        sampling_frequency = 1000
        
        
        filtered_LFP,times = dt.Kay_ripple_detector(time, LFPs, sampling_frequency,
                                     minimum_duration=0.015,
                                     zscore_threshold = 3.5, smoothing_sigma=0.004,
                                     close_ripple_threshold=0.0)
        all_times.append(times)
        peak_power = []
        for start, end in zip(times.start_time,times.end_time):
            st = np.where(time == start)[0][0]
            en = np.where(time == end)[0][0]
            ripple_time = np.arange(st,en)
            
            signal = core.get_envelope(filtered_LFP[:,st:en].T)
            max_signal_channel = np.max(signal, axis = 1)
            max_signal =  np.max(max_signal_channel)
            ind_max = np.where(max_signal_channel== max_signal)
            peak_power.append(ripple_time[ind_max][0])
        
        peak_power_all.append(peak_power)
        
    return all_times, filtered_LFP ,peak_power_all


def plot_check_detect(all_times, HP_LFP, filtered_LFP):
    for times, session in zip(all_times_m484_LFP,m484_LFP):
        
        f, ax = plt.subplots()
        for ripple in times.itertuples():
            ax.axvspan(ripple.start_time, ripple.end_time, alpha=0.3, color='red', zorder=1000)
        
        #channel_n = 0
        #plt.plot(time,filtered_LFP[channel_n,:]+4000)
        
        time = session.lfp_time

        bin_width_ms = 1000
        smooth_sd_ms = 20000
        
        pyControl_choice = [event.time for event in session.events if event.name in ['choice_state']]
        pyControl_choice = np.array(pyControl_choice)
        
        session_duration_ms = int(np.nanmax(time))- int(np.nanmin(time))
        bin_edges = np.arange(0,session_duration_ms, bin_width_ms)
        
        trial_rate,edges_py = np.histogram(pyControl_choice, bins=bin_edges)
        trial_rate = gaussian_filter1d(trial_rate.astype(float), smooth_sd_ms/bin_width_ms)
        plt.plot(bin_edges[:-1], trial_rate/max(trial_rate), label='Rate', color = 'lightblue')
        plt.xlabel('Time (ms)')
        plt.ylabel('Smoothed Firing Rate')
        plt.legend()
        
        lfp_start = []
        lfp_end = []
        
        for ripple in times.itertuples():
           lfp_start.append(np.where(time == ripple.start_time)[0])
           lfp_end.append(np.where(time == ripple.end_time)[0])
           
        max_time = np.max(np.asarray(lfp_end)-np.asarray(lfp_start))/2
        mean_list = []   
         
        for lfp_st, lfp_en in zip(lfp_start,lfp_end):
            ripples = filtered_LFP_m484_LFP[:,lfp_st[0]:lfp_en[0]]
            mean_ripple = np.mean(ripples, axis = 0)
            mean_list.append(mean_ripple)
            
        fig = plt.figure()
        for ii,i in enumerate(mean_list[0:25]):
            fig.add_subplot(5,5,ii+1)
            plt.plot(i)

def plot_tuning_curves(HP,subj,s):
    
    if subj == 'm484':  
        Data = HP['Data'][0,:16]
        DM = HP['DM'][0,:16]
    elif subj == 'm479':  
        Data = HP['Data'][0,16:24]
        DM = HP['DM'][0,16:24]
    elif subj == 'm483':  
        Data = HP['Data'][0,24:]
        DM = HP['DM'][0,24:]
        
    aligned_rates_choices =  Data[s]
    session_DM = DM[s]
    choices = session_DM[:,1]
    task =  session_DM[:,5]
    a_pokes =  session_DM[:,6]
    b_pokes =  session_DM[:,7]
    taskid = rc.task_ind(task,a_pokes,b_pokes)
       

    
    a1_ind = aligned_rates_choices[np.where((choices == 1) & (taskid ==1))[0]]
    b1_ind = aligned_rates_choices[np.where((choices == 0) & (taskid ==1))[0]]
    
    a2_ind = aligned_rates_choices[np.where((choices == 1) & (taskid ==2))[0]]
    b2_ind = aligned_rates_choices[np.where((choices == 0) & (taskid ==2))[0]]

    a3_ind = aligned_rates_choices[np.where((choices == 1) & (taskid ==3))[0]]
    b3_ind = aligned_rates_choices[np.where((choices == 0) & (taskid ==3))[0]]


    return a1_ind, b1_ind,a2_ind, b2_ind,a3_ind, b3_ind,taskid,task

def ripple_plot(peak_power_all, LFP, data):
    
    all_sessions_1 = []
    all_sessions_2 = []
    all_sessions_3 = []

    for s,session in enumerate(LFP):
        
        a1_ind, b1_ind,a2_ind, b2_ind,a3_ind, b3_ind, taskid, task = plot_tuning_curves(data, LFP,s)
        
        task_id_1 = task[(np.where(taskid == 1)[0])][0]
        task_id_2 = task[(np.where(taskid == 2)[0])][0]
        task_id_3 = task[(np.where(taskid == 3)[0])][0]
       
        spikes  = session.ephys
        ripple_times = peak_power_all[s]
        neurons = np.unique(spikes[0,:])

        times_free_reward = np.asarray([event.time for event in session.events if event.name == 'free_reward_trial'])
        iti = np.asarray([event.time for event in session.events if event.name == 'inter_trial_interval'])-100
        trial_init = iti+200
        inter_trial_ripples_peak = []
        
        for peak in ripple_times:
            for i, ti in zip(iti,trial_init):
                if peak >= i and peak <= ti:
                    inter_trial_ripples_peak.append(peak)

        task_2_start = times_free_reward[2]
        task_3_start = times_free_reward[4]
       
        all_neurons_1 = []
        all_neurons_2 = []
        all_neurons_3 = []

        for neuron,n in enumerate(neurons):
            spikes_ind = np.where(spikes[0,:] == n)[0]
            spikes_n = spikes[1,spikes_ind]
            
            spikes_list_ripple_task_1 = []
            spikes_list_ripple_task_2 = []
            spikes_list_ripple_task_3 = []

            for peak in ripple_times:
                start = peak - 100
                end = peak + 100

           # for start, end in zip(inter_trial_ripples_start, inter_trial_ripples_end):

                if peak < task_2_start:
                    spikes_ripples_ind =  np.where((spikes_n > start) & (spikes_n < end))[0]
                    ripples = spikes_n[spikes_ripples_ind]
                    if len(ripples) > 0:
                        ripples = ripples-start
                    spikes_list_ripple_task_1.append(ripples)
                elif peak > task_2_start and start < task_3_start:
                    spikes_ripples_ind =  np.where((spikes_n > start) & (spikes_n < end))[0]
                    ripples = spikes_n[spikes_ripples_ind]
                    if len(ripples) > 0:
                        ripples = ripples-start                   
                    spikes_list_ripple_task_2.append(ripples)
                elif peak > task_3_start:
                    spikes_ripples_ind =  np.where((spikes_n > start) & (spikes_n < end))[0]
                    ripples = spikes_n[spikes_ripples_ind]
                    if len(ripples) > 0:
                        ripples = ripples-start
                    spikes_list_ripple_task_3.append(ripples)
      
            if (task_id_1 == 1) & (task_id_2 == 2) & (task_id_3 == 3):
                all_neurons_1.append(np.asarray(spikes_list_ripple_task_1))
                all_neurons_2.append(np.asarray(spikes_list_ripple_task_2))
                all_neurons_3.append(np.asarray(spikes_list_ripple_task_3))
           
            elif (task_id_1 == 1) & (task_id_2 == 3) & (task_id_3 == 2):
                all_neurons_1.append(np.asarray(spikes_list_ripple_task_1))
                all_neurons_2.append(np.asarray(spikes_list_ripple_task_3))
                all_neurons_3.append(np.asarray(spikes_list_ripple_task_2))
                
            elif (task_id_1 == 2) & (task_id_2 == 3) & (task_id_3 == 1):
                all_neurons_1.append(np.asarray(spikes_list_ripple_task_2))
                all_neurons_2.append(np.asarray(spikes_list_ripple_task_3))
                all_neurons_3.append(np.asarray(spikes_list_ripple_task_1))
           
            elif (task_id_1 == 3) & (task_id_2 == 1) & (task_id_3 == 2):
                all_neurons_1.append(np.asarray(spikes_list_ripple_task_3))
                all_neurons_2.append(np.asarray(spikes_list_ripple_task_1))
                all_neurons_3.append(np.asarray(spikes_list_ripple_task_2))
           
            elif (task_id_1 == 3) & (task_id_2 == 2) & (task_id_3 == 1):
                all_neurons_1.append(np.asarray(spikes_list_ripple_task_3))
                all_neurons_2.append(np.asarray(spikes_list_ripple_task_1))
                all_neurons_3.append(np.asarray(spikes_list_ripple_task_2))
           
            elif (task_id_1 == 2) & (task_id_2 == 1) & (task_id_3 == 3):
                all_neurons_1.append(np.asarray(spikes_list_ripple_task_2))
                all_neurons_2.append(np.asarray(spikes_list_ripple_task_1))
                all_neurons_3.append(np.asarray(spikes_list_ripple_task_3))
          
            len_ripple = []
            
            for ii,i in enumerate(all_neurons_1):
                for r,ripple in enumerate(i):
                    len_ripple.append(len(ripple))
            max_ripple_length = np.max(len_ripple)   
            task_1 = np.full((len(all_neurons_1),len(i),max_ripple_length), np.nan)
            for ii,i in enumerate(all_neurons_1):
                for r,ripple in enumerate(i):
                    task_1[ii,r,:len(ripple)] = ripple
            
            len_ripple = []
            for ii,i in enumerate(all_neurons_2):
                for r,ripple in enumerate(i):
                    len_ripple.append(len(ripple))
            max_ripple_length = np.max(len_ripple)   
            task_2 = np.full((len(all_neurons_2),len(i),max_ripple_length), np.nan)
            for ii,i in enumerate(all_neurons_2):
                for r,ripple in enumerate(i):
                    task_2[ii,r,:len(ripple)] = ripple
            
            len_ripple = []
            for ii,i in enumerate(all_neurons_3):
                for r,ripple in enumerate(i):
                    len_ripple.append(len(ripple))
            max_ripple_length = np.max(len_ripple)   
            task_3 = np.full((len(all_neurons_3),len(i),max_ripple_length), np.nan)
            for ii,i in enumerate(all_neurons_3):
                for r,ripple in enumerate(i):
                    task_3[ii,r,:len(ripple)] = ripple
            
        all_sessions_1.append(task_1)
        all_sessions_2.append(task_2)
        all_sessions_3.append(task_3)
    
    return all_sessions_1,all_sessions_2,all_sessions_3

  
   
def crosscorr_weighted(all_sessions,HP, Data, DM, subj = 'm484'):
    
    all_session_b1, all_session_a1, all_session_i1, all_session_b2, all_session_a2,\
    all_session_i2, all_session_b3, all_session_a3, all_session_i3 = a_b_i_coding(Data,DM)

    if subj == 'm484':  
        all_session_b1 = all_session_b1[:16]
        all_session_a1 = all_session_a1[:16]
        all_session_i1 = all_session_i1[:16]
        all_session_b2 = all_session_b2[:16]
        all_session_a2 = all_session_a2[:16]
        all_session_i2 = all_session_i2[:16]
        all_session_b3 = all_session_b3[:16]
        all_session_a3 = all_session_a3[:16]
        all_session_i3 = all_session_i3[:16]
        
    elif subj == 'm479': 
        all_session_b1 = all_session_b1[16:24]
        all_session_a1 = all_session_a1[16:24]
        all_session_i1 = all_session_i1[16:24]
        all_session_b2 = all_session_b2[16:24]
        all_session_a2 = all_session_a2[16:24]
        all_session_i2 = all_session_i2[16:24]
        all_session_b3 = all_session_b3[16:24]
        all_session_a3 = all_session_a3[16:24]
        all_session_i3 = all_session_i3[16:24]
      
    elif subj == 'm483': 
        all_session_b1 = all_session_b1[24:]
        all_session_a1 = all_session_a1[24:]
        all_session_i1 = all_session_i1[24:]
        all_session_b2 = all_session_b2[24:]
        all_session_a2 = all_session_a2[24:]
        all_session_i2 = all_session_i2[24:]
        all_session_b3 = all_session_b3[24:]
        all_session_a3 = all_session_a3[24:]
        all_session_i3 = all_session_i3[24:]
        
    session_mean = []
    
    for s,session in enumerate(all_sessions):
        
        
        a1_ind, b1_ind,a2_ind, b2_ind,a3_ind, b3_ind, taskid, task = plot_tuning_curves(HP,subj,s)
        
        all_session_b1_s = all_session_b1[s]
        all_session_a1_s = all_session_a1[s]
        all_session_i1_s = all_session_i1[s]
        all_session_b2_s = all_session_b2[s]
        all_session_a2_s = all_session_a2[s]
        all_session_i2_s = all_session_i2[s]
        all_session_b3_s = all_session_b3[s]
        all_session_a3_s = all_session_a3[s]
        all_session_i3_s = all_session_i3[s]
        
        task_id_1 = task[(np.where(taskid == 1)[0])][0]
        task_id_2 = task[(np.where(taskid == 2)[0])][0]
        task_id_3 = task[(np.where(taskid == 3)[0])][0]
        task = str(task_id_1) + '_' + str(task_id_2)  + '_'+str(task_id_3)
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        by_hsv = [*colors]
        by_hsv = [color for color in by_hsv if 'dark' in color or 'medium' in color]
        by_hsv = [color for color in by_hsv if not 'grey' in color]
        fig = plt.figure()

        for n in range(a1_ind.shape[1]):
            fig.add_subplot(3,5, n+1)
            plt.plot(np.mean(b2_ind[:,n,:],  axis = 0),color=by_hsv[n], linestyle = '--', label = 'B2')
            plt.plot(np.mean(b1_ind[:,n,:],  axis = 0),color=by_hsv[n],linestyle = ':', label = 'B1')
            plt.plot(np.mean(b3_ind[:,n,:],  axis = 0),color=by_hsv[n],linestyle = '-.', label = 'B3')

            plt.vlines(25,  ymin = 0,  ymax = np.max([np.mean(b2_ind[:,n,:], axis = 0),np.mean(b1_ind[:,n,:],  axis = 0)]), color = 'grey', alpha = 0.5)
            plt.vlines(36,  ymin = 0, ymax = np.max([np.mean(b2_ind[:,n,:],  axis = 0),np.mean(b1_ind[:,n,:],  axis = 0)]), color = 'pink', alpha = 0.5)
      
        plt.legend()  
        plt.title(s)

        round_session = np.round(session)
        x_max = 200
        smooth_sd_ms = 0.5
        bin_width_ms = 1
        bin_edges_trial = np.arange(0,x_max, bin_width_ms)
        hist_array = np.zeros((session.shape[0],session.shape[1],len(bin_edges_trial)-1))
        ticks_label = []
            
        for n,neuron in enumerate(round_session):
            ticks_label.append(str(n+1))

            for r,ripple in enumerate(neuron):
                hist,edges = np.histogram(ripple, bins= bin_edges_trial)
                smoothed = gaussian_filter1d(hist.astype(float), smooth_sd_ms)         
                hist_array[n,r,:] = smoothed
                
        hist_mean = np.mean(hist_array, axis = 1)
        figav = plt.figure()
        
        for i,ii in enumerate(hist_mean):
           figav.add_subplot(5,4,i+1)
           plt.plot(ii, color = 'black')
        
       
        combinations = list(itertools.combinations(range(hist_array.shape[0]), 2))
        
        pos_1 = []
        pos_2 = []
        
        for i in combinations:
            pos_1.append(i[0])
            pos_2.append(i[1])

        for i in combinations:
            pos_1.append(i[1])
            pos_2.append(i[0])

        hist_array_cut = hist_array[:,:50,:]
        c = []
        corr_n_r = []
        for i1,i2 in zip(pos_1,pos_2):
            i = [i1,i2]
            p = []
            corr_n = []
            for r,rr in zip(hist_array_cut[i1],hist_array_cut[i2]):
                ripple = []
                corr = []
                correlation = np.correlate(r,rr, 'full')
                for lag in range(1,50):
                    Design = np.ones((3,len(r)))
                    Design[1,:]  = r
                    Design[2,:] = rr
                    model = LinearRegression().fit(Design[:,:-lag].T, rr[lag:])
                    corr.append(correlation[1])
                    ripple.append(model.coef_[1])
                p.append(ripple)
                corr_n.append(correlation)
            if i == [1,0]:
                plt.plot(np.mean(corr_n,axis = 0))
            elif i == [0,1]:
                plt.plot(np.mean(corr_n,axis = 0))
            c.append(p)
            corr_n_r.append(corr_n)
        c = np.asarray(c)
        mean_c = np.nanmean(c, axis = 1)
        
        session_mean.append(mean_c)
       
        BA = []
        AB = []
        AI = []
        IA = []
        BI = []
        IB = []

        BA_t2 = []
        AB_t2 = []
        AI_t2 = []
        IA_t2 = []
        BI_t2 = []
        IB_t2 = []
        
        BA_t3 = []
        AB_t3 = []
        AI_t3 = []
        IA_t3 = []
        BI_t3 = []
        IB_t3 = []

        for i1,i2 in zip(pos_1,pos_2):

            BA.append(all_session_b1_s[i1]*all_session_a1_s[i2])
            AB.append(all_session_a1_s[i1]*all_session_b1_s[i2])
            AI.append(all_session_a1_s[i1]*all_session_i1_s[i2])
            IA.append(all_session_i1_s[i1]*all_session_a1_s[i2])
            BI.append(all_session_b1_s[i1]*all_session_i1_s[i2])
            IB.append(all_session_i1_s[i1]*all_session_b1_s[i2])

            BA_t2.append(all_session_b2_s[i1]*all_session_a2_s[i2])
            AB_t2.append(all_session_a2_s[i1]*all_session_b2_s[i2])
            AI_t2.append(all_session_a2_s[i1]*all_session_i2_s[i2])
            IA_t2.append(all_session_i2_s[i1]*all_session_a2_s[i2])
            BI_t2.append(all_session_b2_s[i1]*all_session_i2_s[i2])
            IB_t2.append(all_session_i2_s[i1]*all_session_b2_s[i2])

            BA_t3.append(all_session_b3[i1]*all_session_a3_s[i2])
            AB_t3.append(all_session_a3[i1]*all_session_b3_s[i2])
            AI_t3.append(all_session_a3[i1]*all_session_i3_s[i2])
            IA_t3.append(all_session_i3[i1]*all_session_a3_s[i2])
            BI_t3.append(all_session_b3[i1]*all_session_i3_s[i2])
            IB_t3.append(all_session_i3[i1]*all_session_b3_s[i2])
        
        Design_ports = np.asarray([np.ones(len(BA_t2)),BA_t2,AB_t2,AI_t2,IA_t2,BI_t2,IB_t2])
        coefs = re.regression_code(mean_c,Design_ports.T)

    plt.figure()
    plt.plot(coefs[1], label = 'BA')
    plt.plot(coefs[2], label = 'AB')

    plt.plot(coefs[3], label = 'AI')
    plt.plot(coefs[4], label = 'IA')

    plt.plot(coefs[5], label = 'BI')
    plt.plot(coefs[6], label = 'IB')

    plt.legend()


def simulate_replay():
    
    fake_cells = np.zeros((2,50,199))
    
    replays = np.arange(1)
    for i in range(15):
        fake_cells[0,:5,replays] = 1    
        replays += 10
    
    replays = np.arange(1)
    for i in range(15):
        fake_cells[0,5:10,replays+1] = 1    
        replays += 10

    replays = np.arange(1)
    for i in range(15):
        fake_cells[0,30:35,replays+29] = 1    
        replays += 10 
        
        
    replays = np.arange(1)
    for i in range(15):
        fake_cells[1,:5,replays+1] = 1    
        replays += 10
          
    replays = np.arange(1)
    for i in range(15):
        fake_cells[1,5:10,replays+2] = 1    
        replays += 10
    
    replays = np.arange(1)
    for i in range(15):
        fake_cells[1,30:35,replays+30] = 1    
        replays += 10 
        
    noise = np.random.normal(0,0.04,[2,50,199])
    fake_cells = fake_cells+noise
    pos_1 = [1,0]
    pos_2 = [0,1]
    
    one_ripple = fake_cells[:,1,:]   
    zscore = (one_ripple-np.mean(one_ripple))/np.std(one_ripple)
    
    c_all = [] 
    corr_all   = []     
    for i1,i2 in zip(pos_1,pos_2):
        corr = []
        c = []
       # for r,rr in zip(one_ripple[i1],one_ripple[i2]):
        #    ripple = []
        for lag in range(1,198):
            r = zscore[i1]
            rr = zscore[i2]
            #Design = np.ones((2,len(r)))
            #Design[1,:] = rr

            Design = np.ones((3,len(r)))
            Design[1,:]  = r
            Design[2,:] = rr
            model = LinearRegression().fit(Design[:,:-lag].T, rr[lag:])
            correlation = np.correlate(r[:-lag],rr[lag:], 'full')
            corr.append(correlation)
            c.append(model.coef_[1])
        c_all.append(c)
        corr_all.append(corr)

    c_all = np.asarray(c_all)
    mean_c = np.mean(c, axis = 1)
        

def plot(HP, LFP,all_sessions):
    
    for s,session in enumerate(all_sessions):
        
        a1_ind, b1_ind,a2_ind, b2_ind,a3_ind, b3_ind, taskid, task = plot_tuning_curves(HP,m484_LFP,s)
            
        task_id_1 = task[(np.where(taskid == 1)[0])][0]
        task_id_2 = task[(np.where(taskid == 2)[0])][0]
        task_id_3 = task[(np.where(taskid == 3)[0])][0]
        task = str(task_id_1) + '_' + str(task_id_2)  + '_'+str(task_id_3)
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        by_hsv = [*colors]
        by_hsv = [color for color in by_hsv if 'dark' in color or 'medium' in color]
        by_hsv = [color for color in by_hsv if not 'grey' in color]
        fig = plt.figure()

        for n in range(a1_ind.shape[1]):
            fig.add_subplot(3,5, n+1)
            plt.plot(np.mean(b2_ind[:,n,:],  axis = 0),color=by_hsv[n], linestyle = '--', label = 'B2')
            plt.plot(np.mean(b1_ind[:,n,:],  axis = 0),color=by_hsv[n],linestyle = ':', label = 'B1')
            plt.plot(np.mean(b3_ind[:,n,:],  axis = 0),color=by_hsv[n],linestyle = '-.', label = 'B3')

            plt.vlines(25,  ymin = 0,  ymax = np.max([np.mean(b2_ind[:,n,:], axis = 0),np.mean(b1_ind[:,n,:],  axis = 0)]), color = 'grey', alpha = 0.5)
            plt.vlines(36,  ymin = 0, ymax = np.max([np.mean(b2_ind[:,n,:],  axis = 0),np.mean(b1_ind[:,n,:],  axis = 0)]), color = 'pink', alpha = 0.5)
      
        plt.legend()  
        plt.title(s)

        round_session = np.round(session)
        x_max = 225
        smooth_sd_ms = 0.5
        bin_width_ms = 25
        bin_edges_trial = np.arange(0,x_max, bin_width_ms)
        hist_array = np.zeros((session.shape[0],session.shape[1],len(bin_edges_trial)-1))
        ticks_label = []
        for n,neuron in enumerate(round_session):
            ticks_label.append(str(n+1))

            for r,ripple in enumerate(neuron):
                hist,edges = np.histogram(ripple, bins= bin_edges_trial)
                smoothed = gaussian_filter1d(hist.astype(float), smooth_sd_ms)         
                hist_array[n,r,:] = smoothed
        hist_mean = np.mean(hist_array, axis = 1)
        figav = plt.figure()
        
        for i,ii in enumerate(hist_mean):
           figav.add_subplot(5,4,i+1)
           plt.plot(ii, color = 'black')
           
        hist_resh  = np.transpose(hist_array,[0,2,1])
        time_unr  = np.reshape(hist_resh,(hist_resh.shape[0]*hist_resh.shape[1],hist_resh.shape[2]))
       # correlate = np.corrcoef(np.nanmean(hist_array, 2)) 
        correlate = np.corrcoef(time_unr)
        plt.figure()
        plt.imshow(correlate)
        ticks_n = np.arange(0, correlate.shape[0], hist_resh.shape[1])
        
        plt.xticks(ticks_n,ticks_label, rotation = 'vertical')  
        plt.yticks(ticks_n,ticks_label)  
 
        plt.colorbar()
        plt.title(str(s)+'_'+task)

        
        #mean_all_ripples = np.mean(hist_array, 1) 
        #for i in mean_all_ripples:
       #     plt.plot(i)
        
      
        fig1 = plt.figure()
        for ii,i in enumerate(session):
            for r,ripple in enumerate(i):    
                if r <80:
                    fig1.add_subplot(8,10,r+1)
                    plt.vlines(i[r], ymin = 0+ii, ymax = 1+ii, colors=by_hsv[ii])
                    plt.xlim(0, 200)
        plt.title(s)


                    
  
def a_b_i_coding(Data,Design):
    
    all_session_b1 =  []
    all_session_a1 =  []
    all_session_i1 =  []
        
    all_session_b2 =  []
    all_session_a2 =  []
    all_session_i2 =  []
        
    all_session_b3 =  []
    all_session_a3 =  []
    all_session_i3 =  []


  
    for  s, sess in enumerate(Data):
     

        DM = Design[s]
        x = Data[s]
        #state =  DM[:,0]
       
        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        
        task_1 = np.where((taskid == 1))[0]       
        task_2 = np.where((taskid == 2))[0]
        task_3 = np.where((taskid == 3))[0]
       
        task_1_a = np.where((taskid == 1) & (choices == 0))[0]       
        task_2_a = np.where((taskid == 2) & (choices == 0))[0]
        task_3_a = np.where((taskid == 3) & (choices == 0))[0]
        
        task_1_b = np.where((taskid == 1) & (choices == 1))[0]       
        task_2_b = np.where((taskid == 2) & (choices == 1))[0]
        task_3_b = np.where((taskid == 3) & (choices == 1))[0]
        
        
        It = np.arange(23 ,27) #Init
        Ct = np.arange(33, 37) #Choice

        firing_rates_mean_time = x
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        # Numpy arrays to fill the firing rates of each neuron where the A choice was made
                    
        ## Bs          
        b1_fr = np.mean(np.mean(firing_rates_mean_time[task_1_b][:,:,Ct], axis = 0), axis = 1)
        b2_fr = np.mean(np.mean(firing_rates_mean_time[task_2_b][:,:,Ct], axis = 0), axis = 1)
        b3_fr = np.mean(np.mean(firing_rates_mean_time[task_3_b][:,:,Ct], axis = 0), axis = 1)
   
        ## As
        a1_fr = np.mean(np.mean(firing_rates_mean_time[task_1_a][:,:,Ct],axis = 0), axis = 1)
        a2_fr = np.mean(np.mean(firing_rates_mean_time[task_2_a][:,:,Ct],axis = 0), axis = 1)
        a3_fr = np.mean(np.mean(firing_rates_mean_time[task_3_a][:,:,Ct],axis = 0), axis = 1)
        
        ## Is
        i1_fr = np.mean(np.mean(firing_rates_mean_time[task_1][:,:,It], axis = 0), axis = 1)
        i2_fr = np.mean(np.mean(firing_rates_mean_time[task_2][:,:,It], axis = 0), axis = 1)
        i3_fr = np.mean(np.mean(firing_rates_mean_time[task_3][:,:,It], axis = 0), axis = 1)
        
        fr_av_t1 = np.mean(np.mean(firing_rates_mean_time[task_1], axis = 0), axis = 1)
        fr_av_t2 = np.mean(np.mean(firing_rates_mean_time[task_2], axis = 0), axis = 1)
        fr_av_t3 = np.mean(np.mean(firing_rates_mean_time[task_3], axis = 0), axis = 1)
        
        b1_prop = b1_fr/fr_av_t1
        b2_prop = b2_fr/fr_av_t2
        b3_prop = b3_fr/fr_av_t3

        a1_prop = a1_fr/fr_av_t1
        a2_prop = a2_fr/fr_av_t2
        a3_prop = a3_fr/fr_av_t3


        i1_prop = i1_fr/fr_av_t1
        i2_prop = i2_fr/fr_av_t2
        i3_prop = i3_fr/fr_av_t3
        
        all_session_b1.append(b1_prop)
        all_session_a1.append(a1_prop)
        all_session_i1.append(i1_prop)
        
        all_session_b2.append(b2_prop)
        all_session_a2.append(a2_prop)
        all_session_i2.append(i2_prop)
        
        all_session_b3.append(b3_prop)
        all_session_a3.append(a3_prop)
        all_session_i3.append(i3_prop)
        
    return all_session_b1, all_session_a1, all_session_i1, all_session_b2, all_session_a2,\
    all_session_i2, all_session_b3, all_session_a3, all_session_i3
        
        
