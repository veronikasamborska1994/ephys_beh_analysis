#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:22:08 2020

@author: veronikasamborska
"""
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/plotting')
from mpl_toolkits import mplot3d
from matplotlib.cbook import flatten
import scipy.io
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import regression_function as reg_f
import regressions as re
from collections import OrderedDict
from matplotlib import colors as mcolors
from scipy.ndimage import gaussian_filter1d
from matplotlib.cbook import flatten
import create_data_arrays_for_tim as cda
from scipy.interpolate import interp1d
import utility as ut 
from scipy.ndimage import gaussian_filter1d
from palettable import wesanderson as wes
import palettable
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import utility as ut

import palettable.wesanderson as we
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)

def import_code():
    data_PFC = cda.tim_create_mat(experiment_aligned_PFC, 'PFC')# experiment_sim_Q1_PFC, experiment_sim_Q4_PFC, experiment_sim_Q1_value_a_PFC, experiment_sim_Q1_value_b_PFC, experiment_sim_Q4_values_PFC, 'PFC') 
    data_HP = cda.tim_create_mat(experiment_aligned_HP,'HP')# experiment_sim_Q1_HP, experiment_sim_Q4_HP, experiment_sim_Q1_value_a_HP, experiment_sim_Q1_value_b_HP, experiment_sim_Q4_values_HP, 'HP')


def time_and_firing_extract(s_aligned,events_block):
    neurons = np.unique(s_aligned.ephys[0])
    spikes = s_aligned.ephys[1]
    bin_width_ms = 100
    smooth_sd_ms = 20000
    
    session_duration_ms = s_aligned.events[-1].time
    bin_edges = np.arange(0,session_duration_ms, bin_width_ms)
       
    length = np.diff(events_block)
    first = np.arange(0, len(np.histogram(np.arange(0,events_block[0]), bins= (np.arange(0,events_block[0], bin_width_ms)))[0]))
    second = np.arange(0, len(np.histogram(np.arange(0,length[0]), bins= (np.arange(0,length[0], bin_width_ms)))[0]))    
    third = np.arange(0, len(np.histogram(np.arange(0,length[1]), bins= (np.arange(0,length[1], bin_width_ms)))[0]))
    
    
    fourth = np.arange(0, len(np.histogram(np.arange(0,length[2]), bins= (np.arange(0,length[2], bin_width_ms)))[0]))    
    fifth = np.arange(0, len(np.histogram(np.arange(0,length[3]), bins= (np.arange(0,length[3], bin_width_ms)))[0]))    
    six = np.arange(0, len(np.histogram(np.arange(0,length[4]), bins= (np.arange(0,length[4], bin_width_ms)))[0]))
    
    seventh = np.arange(0, len(np.histogram(np.arange(0,length[5]), bins= (np.arange(0,length[5], bin_width_ms)))[0]))    
    eight = np.arange(0, len(np.histogram(np.arange(0,length[6]), bins= (np.arange(0,length[6], bin_width_ms)))[0]))    
    nine = np.arange(0, len(np.histogram(np.arange(0,length[7]), bins= (np.arange(0,length[7], bin_width_ms)))[0]))
    
    ten = np.arange(0, len(np.histogram(np.arange(0,length[8]), bins= (np.arange(0,length[8], bin_width_ms)))[0]))    
    eleven = np.arange(0, len(np.histogram(np.arange(0,length[9]), bins= (np.arange(0,length[9], bin_width_ms)))[0]))    
    twelve = np.arange(0, len(np.histogram(np.arange(0,length[10]), bins= (np.arange(0,length[10], bin_width_ms)))[0]))
    time_in_block = np.hstack((first,second,third,fourth,fifth,six,seventh,eight,nine,ten,eleven,twelve))                                   

    all_neurons = []
    for i,neuron in enumerate(neurons):  
        spikes_ind = np.where(s_aligned.ephys[0] == neuron)
        spikes_n = spikes[spikes_ind]
        hist,edges = np.histogram(spikes_n, bins= bin_edges)# histogram per second
        normalised = gaussian_filter1d(hist.astype(float), smooth_sd_ms/bin_width_ms)
        #plt.figure()
        #plt.plot(bin_edges[:-1], normalised/max(normalised), label='Firing Rate', color ='navy') 
        
        normalised_cut_12 = normalised[:len(time_in_block)]
        all_neurons.append(normalised_cut_12)  
    all_neurons = np.asarray(all_neurons)
    return time_in_block, all_neurons
   



def function_for_size_numpy(data):
    firing = data['Data']
    session_size_list = []
    for s in firing:
        
        session_size = s.shape
        session_size_list.append(session_size)
    session_size_list = np.asarray(session_size_list)
    
    return session_size_list


def tuning_curve(data,experiment_aligned_data):
    dm = data['DM']
    firing = data['Data']
    
    for  s, sess in enumerate(dm):
        DM = dm[s]
        choices = DM[:,1]
        reward = DM[:,2]
        
        exp_rew = ut.exp_mov_ave(reward, 3)
        firing_rates = firing[s]
        s_aligned = experiment_aligned_data[s]
        events_all_trials = [event.time for event in s_aligned.events if event.name in ['init_trial']]

        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]

        if len(ind_block) >= 12:
            fig = plt.figure(figsize=(5,10))
            plt.tight_layout()
            events_block = np.asarray(events_all_trials)[ind_block]
 

            
            one = events_all_trials[:ind_block[0]]
            two = events_all_trials[ind_block[0]:ind_block[1]] - events_block[0]
            three = events_all_trials[ind_block[1]:ind_block[2]] - events_block[1]
            
            four = events_all_trials[ind_block[2]:ind_block[3]] - events_block[2]
            five = events_all_trials[ind_block[3]:ind_block[4]] - events_block[3]
    
            six = events_all_trials[ind_block[4]:ind_block[5]] - events_block[4]
            seven = events_all_trials[ind_block[5]:ind_block[6]] - events_block[5]
    
            eight = events_all_trials[ind_block[6]:ind_block[7]] - events_block[6]
            nine = events_all_trials[ind_block[7]:ind_block[8]] - events_block[7]
            
            ten = events_all_trials[ind_block[8]:ind_block[9]] - events_block[8]
            eleven = events_all_trials[ind_block[9]:ind_block[10]] - events_block[9]    
            twelve = events_all_trials[ind_block[10]:ind_block[11]] - events_block[10]
            
            time_in_block = np.hstack((one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve))       
            ind_block =  np.where(time_in_block==0)[0]
            time_in_block_secs = np.round(time_in_block/1000)
            min_time_ind = np.where(time_in_block_secs ==0)[0]-1
            min_block = np.min(time_in_block_secs[min_time_ind])
            ind_time_minimum = np.where(time_in_block_secs < min_block)[0]
            
            # Blocks
            b_1 = ind_time_minimum[np.asarray(ind_time_minimum < ind_block[0]*1)]    
            b_2 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[0]*1) & (ind_time_minimum < ind_block[1]*1))]
            b_3 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[1]*1) & (ind_time_minimum < ind_block[2]*1))]
            b_4 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[2]*1) & (ind_time_minimum < ind_block[3]*1))]
            b_5 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[3]*1) & (ind_time_minimum < ind_block[4]*1))]
            b_6 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[4]*1) & (ind_time_minimum < ind_block[5]*1))]
            b_7 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[5]*1) & (ind_time_minimum < ind_block[6]*1))]
            b_8 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[6]*1) & (ind_time_minimum < ind_block[7]*1))]
            b_9 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[7]*1) & (ind_time_minimum < ind_block[8]*1))]
            b_10 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[8]*1) & (ind_time_minimum < ind_block[9]*1))]
            b_11 = ind_time_minimum[np.asarray((ind_time_minimum >ind_block[9]*1) & (ind_time_minimum < ind_block[10]*1))]
            b_12 = ind_time_minimum[np.asarray(ind_time_minimum >ind_block[10]*1)]

            firing_rates = np.mean(firing_rates[:len(time_in_block), :,:],2)
            
            # Rewards
            
            rewards_1 = exp_rew[b_1]
            rw_1_list = np.zeros(int(min_block))
            rw_1_list[:] = np.NaN
            rw_1_list[np.asarray(time_in_block_secs[b_1], int)] = rewards_1
            
            rewards_2 = exp_rew[b_2]
            rw_2_list = np.zeros(int(min_block))
            rw_2_list[:] = np.NaN
            rw_2_list[np.asarray(time_in_block_secs[b_2], int)] = rewards_2
            
            rewards_3 = exp_rew[b_3]
            rw_3_list = np.zeros(int(min_block))
            rw_3_list[:] = np.NaN
            rw_3_list[np.asarray(time_in_block_secs[b_3], int)] = rewards_3
           
            rewards_4 = exp_rew[b_4]
            rw_4_list = np.zeros(int(min_block))
            rw_4_list[:] = np.NaN
            rw_4_list[np.asarray(time_in_block_secs[b_4], int)] = rewards_4
           
            rewards_5 = exp_rew[b_5]
            rw_5_list = np.zeros(int(min_block))
            rw_5_list[:] = np.NaN
            rw_5_list[np.asarray(time_in_block_secs[b_5], int)] = rewards_5
           
            rewards_6 = exp_rew[b_6]
            rw_6_list = np.zeros(int(min_block))
            rw_6_list[:] = np.NaN
            rw_6_list[np.asarray(time_in_block_secs[b_6], int)] = rewards_6         

            rewards_7 = exp_rew[b_7]
            rw_7_list = np.zeros(int(min_block))
            rw_7_list[:] = np.NaN
            rw_7_list[np.asarray(time_in_block_secs[b_7], int)] = rewards_7
           
            rewards_8 = exp_rew[b_8]
            rw_8_list = np.zeros(int(min_block))
            rw_8_list[:] = np.NaN
            rw_8_list[np.asarray(time_in_block_secs[b_8], int)] = rewards_8
                 
            rewards_9 = exp_rew[b_9]
            rw_9_list = np.zeros(int(min_block))
            rw_9_list[:] = np.NaN
            rw_9_list[np.asarray(time_in_block_secs[b_9], int)] = rewards_9
           
            rewards_10 = exp_rew[b_10]
            rw_10_list = np.zeros(int(min_block))
            rw_10_list[:] = np.NaN
            rw_10_list[np.asarray(time_in_block_secs[b_10], int)] = rewards_10
        
            
            rewards_11 = exp_rew[b_11]
            rw_11_list = np.zeros(int(min_block))
            rw_11_list[:] = np.NaN
            rw_11_list[np.asarray(time_in_block_secs[b_11], int)] = rewards_11
           
            rewards_12 = exp_rew[b_12]
            rw_12_list = np.zeros(int(min_block))
            rw_12_list[:] = np.NaN
            rw_12_list[np.asarray(time_in_block_secs[b_12], int)] = rewards_12
            av_r = np.nanmean([rw_1_list,rw_2_list,rw_3_list,rw_4_list,rw_5_list, rw_6_list, rw_7_list, rw_8_list, rw_9_list, rw_10_list,rw_11_list, rw_12_list],0)

            av_r = av_r[~np.isnan(av_r)]
            av_r_smooth = gaussian_filter1d(av_r,2)
               
            fig.add_subplot(firing_rates.shape[1]+1,1,1)

            plt.plot(av_r_smooth, color = 'black', alpha = 0.7, linestyle = '--')
            plt.ylabel('Reward Rate')
               
            #Colorscheme
            c = we.GrandBudapest3_6.mpl_colors
            c2 = we.IsleOfDogs3_4.mpl_colors
            c3 = we.Mendl_4.mpl_colors
            c4 = we.Moonrise1_5.mpl_colors
            c5 = we.Moonrise5_6.mpl_colors
            c6 = we.Moonrise6_5.mpl_colors
            for cc,ccc,cccc,ccccc,ccccccc in zip(c2,c3,c4,c5,c6):
                c.append(cc)
                c.append(ccc)
                c.append(cccc)
                c.append(ccccc)
                c.append(ccccccc)
            
            for n in range(firing_rates.shape[1]):
                
                #Block 1 
                firing_block_1 = np.zeros(int(min_block))
                firing_block_1[:] = np.NaN
                f1_b1 = firing_rates[b_1,n]
                firing_block_1[np.asarray(time_in_block_secs[b_1], int)] = f1_b1
                
                #Block 2
                firing_block_2 = np.zeros(int(min_block))
                firing_block_2[:] = np.NaN
                f2_b2 = firing_rates[b_2,n]
                firing_block_2[np.asarray(time_in_block_secs[b_2], int)] = f2_b2
                
                #Block 3
                firing_block_3 = np.zeros(int(min_block))
                firing_block_3[:] = np.NaN
                f3_b3 = firing_rates[b_3,n]
                firing_block_3[np.asarray(time_in_block_secs[b_3], int)] = f3_b3
                
                #Block 4
                firing_block_4 = np.zeros(int(min_block))
                firing_block_4[:] = np.NaN
                f4_b4 = firing_rates[b_4,n]
                firing_block_4[np.asarray(time_in_block_secs[b_4], int)] = f4_b4
 
                #Block 5 
                firing_block_5 = np.zeros(int(min_block))
                firing_block_5[:] = np.NaN
                f5_b5 = firing_rates[b_5,n]
                firing_block_5[np.asarray(time_in_block_secs[b_5], int)] = f5_b5
                
                #Block 6
                firing_block_6 = np.zeros(int(min_block))
                firing_block_6[:] = np.NaN
                f6_b6 = firing_rates[b_6,n]
                firing_block_6[np.asarray(time_in_block_secs[b_6], int)] = f6_b6
                
                #Block 7
                firing_block_7 = np.zeros(int(min_block))
                firing_block_7[:] = np.NaN
                f7_b7 = firing_rates[b_7,n]
                firing_block_7[np.asarray(time_in_block_secs[b_7], int)] = f7_b7
                
                #Block 8
                firing_block_8 = np.zeros(int(min_block))
                firing_block_8[:] = np.NaN
                f8_b8 = firing_rates[b_8,n]
                firing_block_8[np.asarray(time_in_block_secs[b_8], int)] = f8_b8     
            
                #Block 9
                firing_block_9 = np.zeros(int(min_block))
                firing_block_9[:] = np.NaN
                f9_b9 = firing_rates[b_9,n]
                firing_block_9[np.asarray(time_in_block_secs[b_9], int)] = f9_b9
                
                #Block 10
                firing_block_10 = np.zeros(int(min_block))
                firing_block_10[:] = np.NaN
                f10_b10 = firing_rates[b_10,n]
                firing_block_10[np.asarray(time_in_block_secs[b_10], int)] = f10_b10

                #Block 11
                firing_block_11 = np.zeros(int(min_block))
                firing_block_11[:] = np.NaN
                f11_b11 = firing_rates[b_11,n]
                firing_block_11[np.asarray(time_in_block_secs[b_11], int)] = f11_b11
                
                #Block 12
                firing_block_12 = np.zeros(int(min_block))
                firing_block_12[:] = np.NaN
                f12_b12 = firing_rates[b_12,n]
                firing_block_12[np.asarray(time_in_block_secs[b_12], int)] = f12_b12
            
                av = np.nanmean([firing_block_1,firing_block_2,firing_block_3,firing_block_4,firing_block_5, firing_block_6, firing_block_7, firing_block_8, firing_block_9, firing_block_10,firing_block_11, firing_block_12],axis = 0)
               
                av = av[~np.isnan(av)]
                av_smooth = gaussian_filter1d(av,2)
                norm_mean = (av_smooth - np.mean(av_smooth))/np.std(av_smooth)
                fig.add_subplot(firing_rates.shape[1]+1,1,n+2)
                corr_r = np.corrcoef(av_smooth,av_r_smooth)[0,1]
                plt.plot(av_smooth, color = c[n], label  = str(corr_r))
                plt.legend()                
                plt.ylabel('FR')
                #plt.title(str(n))
                plt.tight_layout()

                sns.despine()
            
def regression_time_of_trial(data,experiment_aligned_data,perm = False, trial_hypothesis = False, time_hypothesis = False, place_cells = False):
    runs = 1
    #session_size_list =  function_for_size_numpy(data)
    #print(runs)
    #C = np.zeros((9,session_size_list[1],session_size_list[2]))
    #C[:] = np.NaN
    #cpd = np.zeros((session_size_list[1],session_size_list[0],session_size_list))
    #cpd[:] = np.NaN
        
    C = []
    cpd = []
    firing_rates_time = []
    dm_time =[]
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    
    if perm:
        C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
    
    for  s, sess in enumerate(dm):
        runs +=1
        
        
        DM = dm[s]
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
        
        firing_rates = firing[s]
        s_aligned = experiment_aligned_data[s]
        events_all_trials = [event.time for event in s_aligned.events if event.name in ['init_trial']]
 
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]

        if len(ind_block) >= 12:
            events_block = np.asarray(events_all_trials)[ind_block]
            

            firing_rates_time.append(firing_rates)
            dm_time.append(DM)
            
            one = events_all_trials[:ind_block[0]]
            two = events_all_trials[ind_block[0]:ind_block[1]] - events_block[0]
            three = events_all_trials[ind_block[1]:ind_block[2]] - events_block[1]
            
            four = events_all_trials[ind_block[2]:ind_block[3]] - events_block[2]
            five = events_all_trials[ind_block[3]:ind_block[4]] - events_block[3]
    
            six = events_all_trials[ind_block[4]:ind_block[5]] - events_block[4]
            seven = events_all_trials[ind_block[5]:ind_block[6]] - events_block[5]
    
            eight = events_all_trials[ind_block[6]:ind_block[7]] - events_block[6]
            nine = events_all_trials[ind_block[7]:ind_block[8]] - events_block[7]
            
            ten = events_all_trials[ind_block[8]:ind_block[9]] - events_block[8]
            eleven = events_all_trials[ind_block[9]:ind_block[10]] - events_block[9]
            
            twelve = events_all_trials[ind_block[10]:ind_block[11]] - events_block[10]
            
            time_in_block = np.hstack((one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve))             
    
            firing_rates = firing_rates[:len(time_in_block), :,:] 
            trials, n_neurons, n_timepoints = firing_rates.shape
            state = state[:len(time_in_block)]
            choices = choices[:len(time_in_block)]
            reward = reward[:len(time_in_block)]
            ones = np.ones(len(time_in_block))


            trials_since_block = []
            t = 0
            
            #Bug in the state? 
            for st,s in enumerate(block):
                if block[st-1] != block[st]:
                    t = 0
                else:
                    t+=1
                trials_since_block.append(t)
                
            #block_totals_ind = (np.where(np.asarray(ind_block) == 1)[0]-1)[1:]
            block_totals_ind = ind_block
            block_totals = np.diff(block_totals_ind)-1
            trials_since_block = trials_since_block[:ind_block[11]]
            fraction_list = []


            for t,trial in enumerate(trials_since_block):
                
                if t <= block_totals_ind[0]:
                    fr = trial/block_totals_ind[0]
                    fraction_list.append(fr)

                elif t > block_totals_ind[0] and  t <= block_totals_ind[1]:
                    fr = trial/block_totals[0]
                    fraction_list.append(fr)

                elif t > block_totals_ind[1] and  t <= block_totals_ind[2]:
                    fr = trial/block_totals[1]               
                    fraction_list.append(fr)

                elif t > block_totals_ind[2] and  t <= block_totals_ind[3]:
                    fr = trial/block_totals[2]                
                    fraction_list.append(fr)

                elif t > block_totals_ind[3] and  t <= block_totals_ind[4]:
                    fr = trial/block_totals[3]
                    fraction_list.append(fr)

                elif t > block_totals_ind[4] and  t <= block_totals_ind[5]:
                    fr = trial/block_totals[4]
                    fraction_list.append(fr)

                elif t > block_totals_ind[5] and  t <= block_totals_ind[6]:
                    fr = trial/block_totals[5]
                    fraction_list.append(fr)

                elif t > block_totals_ind[6] and  t <= block_totals_ind[7]:
                    fr = trial/block_totals[6]  
                    fraction_list.append(fr)

                elif t > block_totals_ind[7] and  t <= block_totals_ind[8]:
                    fr = trial/block_totals[7]
                    fraction_list.append(fr)

                elif t > block_totals_ind[8] and  t <= block_totals_ind[9]:
                    fr = trial/block_totals[8]                 
                    fraction_list.append(fr)

                elif t > block_totals_ind[9] and  t <= block_totals_ind[10]:
                    fr = trial/block_totals[9]
                    fraction_list.append(fr)

                elif t >  block_totals_ind[10] and  t <= len(trials_since_block):
                    fr = trial/trials_since_block[-1]
                    fraction_list.append(fr)

                
            #Block lengths calculations
            block_1 = ind_block[0]
            block_2_12 = np.diff(ind_block[:12])    
            block_lengths = np.hstack((block_1,block_2_12))
           
            #Fractions squared, interactions
            fraction_list = np.asarray(fraction_list)
            fraction_sq = (fraction_list-0.5)**2
            int_fraction = fraction_list* choices
            
            reward_choice = choices*reward
            reward_state =reward*state
            reward_state_choice  = reward*state*choices
            #Time squared, interactions
            time_sq = (np.asarray(time_in_block)-0.5)**2
            choice_time_sq = choices*time_sq
            time_choice_int = time_in_block*choices
            
            #Trial squared, interactions

            trial_sq = (np.asarray(trials_since_block)-0.5)**2
            choice_trials_sq = choices*trial_sq
            choice_state_sq = state*trial_sq
            state_choice = state*choices

            interaction_trials_choice = trials_since_block*choices
            interaction_trials_state = trials_since_block*state
            interaction_trials_state_choice = trials_since_block*state*choices

            block_df = np.diff(block)
            ind_block = np.where(block_df ==1)[0]
            ind_block = ind_block[:10]
            block_length = np.append(np.diff(ind_block), ind_block[0])

            min_ind = np.int(np.min(block_length))
            block_index_min = np.arange(0,min_ind)
            for i in ind_block:
                block_index_min = np.append(block_index_min,np.arange(i,i+min_ind))
            
            # Predictors Select 
# =============================================================================
#             state = state[block_index_min]
#             reward = reward[block_index_min]
#             choices = choices[block_index_min]
#             reward_choice = reward_choice[block_index_min]
#             time_in_block = time_in_block[block_index_min]
#             time_sq = time_sq[block_index_min]
#             time_choice_int = time_choice_int[block_index_min]
#             choice_time_sq = choice_time_sq[block_index_min]
#             ones = ones[block_index_min]
#             firing_rates = firing_rates[block_index_min]
#             trials = len(ones)       
# =============================================================================
            
            
            if trial_hypothesis == True:
                predictors_all = OrderedDict([#('Time', trials_all),
                                              
                                              ('State', state),
                                              ('Reward', reward),
                                              ('Choice', choices),
                                              
                                             # ('Trials in Block', trials_since_block),
                                              ('Reward Choice Int', reward_choice),
                                            
                                              
                                              ('Time in Block', time_in_block), 
                                              ('Squared Time in Block', time_sq),
                                              ('Time x Choice', time_choice_int),
                                              ('Choice x Time Sq',choice_time_sq),

                                             # ('Fraction of Block', fraction_list),
                                             # ('Fraction Squared', fraction_sq),
                                             # ('Fraction Interaction', int_fraction),
                                              ('ones', ones)])
            elif time_hypothesis == True:
                predictors_all = OrderedDict([#('Time', trials_all),
                                              #('Time in Block', time_in_block), 
                                              ('State', state),
                                              ('Reward', reward),
                                              ('Choice', choices),
                                              ('State x Choice', state_choice),
                                              ('Reward x Choice', reward_choice),
                                              ('Trials in Block', trials_since_block),
                                            #  ('Squared Time in Block', trial_sq),
                                              ('Trials x State', interaction_trials_state),
                                              ('Trials x State x Choice', interaction_trials_state_choice),
                                              ('Trials x Choice', interaction_trials_choice),
                                              ('Reward x State', reward_state),
                                              ('Reward x State x Choice', reward_state_choice),

                                              #('Trials x Choice Sq',choice_trials_sq),
                                             # ('Trials x State Sq',choice_state_sq),

                                             # ('Choice x Time in Block Int',choice_time),
                                             # ('Fraction of Block', fraction_list),
                                             # ('Fraction Squared', fraction_sq),
                                             # ('Fraction Interaction', int_fraction),
                                              ('ones', ones)])
            elif place_cells == True:
                predictors_all = OrderedDict([#('Time', trials_all),
                                              #('Time in Block', time_in_block), 
                                              ('State', state),
                                              ('Reward', reward),
                                              ('Choice', choices),
                                              #('Trials in Block', trials_since_block),
                                              ('Reward Choice Int', reward_choice),
                                             # ('Trials in Block x Choice',interaction_trials_choice)
                                             # ('Choice x Time in Block Int',choice_time),
                                              ('Fraction of Block', fraction_list),
                                              ('Fraction Squared', fraction_sq),
                                              ('Fraction Interaction', int_fraction),
                                              ('ones', ones)])
                    
            X = np.vstack(predictors_all.values()).T[:trials,:].astype(float)
            n_predictors = X.shape[1]
            y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats = reg_f.regression_code(y, X)
            C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
            
          
            if perm:
                for i in range(perm):
                    # # shuffle the start of all blocks
                     starts = []
                     for l in block_lengths:
                         start = np.random.randint(trials-l)
                         start_list = np.arange(start,start+l)
                         starts.append(start_list)
                        
                     ind = list(flatten(starts))
                     X_perm = X[ind,:]
                    
                    #X_perm = np.roll(X,np.random.randint(trials), axis=0)
                    #tstats = reg_f.regression_code(y, X_perm)
        
                     C_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
                     cpd_perm[i].append(re._CPD(X_perm,y).reshape(n_neurons, n_timepoints, n_predictors))

     
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        C_perm   = np.stack([np.concatenate(C_i,1) for C_i in C_perm],1)
        cpd_p_value  = np.mean(cpd_perm > cpd,0)
        p = np.percentile(cpd_perm,95, axis = 0)

    
    return C, cpd, predictors_all, cpd_perm, cpd_p_value, p,firing_rates_time,dm_time


def regression_time_of_trial_reversal_triggered(data,experiment_aligned_data, perm = False):
    runs = 1
    #session_size_list =  function_for_size_numpy(data)
    #print(runs)
    #C = np.zeros((9,session_size_list[1],session_size_list[2]))
    #C[:] = np.NaN
    #cpd = np.zeros((session_size_list[1],session_size_list[0],session_size_list))
    #cpd[:] = np.NaN
        
    C = []
    cpd = []
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    
    if perm:
        C_perm   = [[] for i in range(perm)] # To store permuted predictor loadings for each session.
        cpd_perm = [[] for i in range(perm)] # To store permuted cpd for each session.
    
    for  s, sess in enumerate(dm):
        runs +=1
        
        
        DM = dm[s]
        state = DM[:,0]
        choices = DM[:,1]
        reward = DM[:,2]
        task = DM[:,5]
        task_ind = np.where(np.diff(task)!=0)[0]
        
        firing_rates = firing[s] 
        block = DM[:,4]
        block_df = np.diff(block)
        ind_block = np.where(block_df != 0)[0]

        if len(ind_block) >= 12:
        
            #Block lengths calculations
            block_1 = ind_block[0]
            block_2_12 = np.diff(ind_block[:12])    
            block_lengths = np.hstack((block_1,block_2_12))

            #Because moving average resets --> calucate corrects for all tasks
            
            task_1_state = state[:task_ind[0]]
            task_2_state=  state[task_ind[0]:task_ind[1]]
            task_3_state = state[task_ind[1]:]
            task_1_choice = choices[:task_ind[0]]
            task_2_choice=  choices[task_ind[0]:task_ind[1]]
            task_3_choice = choices[task_ind[1]:]
            correct_ind_task_1 = np.where(task_1_state == task_1_choice)
            correct_ind_task_2 = np.where(task_2_state == task_2_choice)
            correct_ind_task_3 = np.where(task_3_state == task_3_choice)

            correct_task_1 = np.zeros(len(task_1_state))
            correct_task_1[correct_ind_task_1] = 1
            correct_task_2 = np.zeros(len(task_2_state))
            correct_task_2[correct_ind_task_2] = 1
            correct_task_3 = np.zeros(len(task_3_state))
            correct_task_3[correct_ind_task_3] = 1

            # Calculate movign average to determine behavioural switches
            mov_av_task_1 = ut.exp_mov_ave(correct_task_1,initValue = 0.5,tau = 8)
            mov_av_task_2 = ut.exp_mov_ave(correct_task_2,initValue = 0.5,tau = 8)
            mov_av_task_3 = ut.exp_mov_ave(correct_task_3,initValue = 0.5,tau = 8)
            mov_av = np.concatenate((mov_av_task_1,mov_av_task_2,mov_av_task_3))
            moving_av_0_6 = np.where(mov_av > 0.63)[0]
            
            
           
            b_1 = np.arange(moving_av_0_6[np.where((moving_av_0_6 < ind_block[0].astype(int)) ==1)[0][0]],moving_av_0_6[np.where((moving_av_0_6 < ind_block[0].astype(int)) ==1)[0][0]]+9)
            b_2 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[0]) & (moving_av_0_6 < ind_block[1]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[0]) & (moving_av_0_6 < ind_block[1]).astype(int)) ==1)[0][0]]+9)
                
            b_3 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[1]) & (moving_av_0_6 < ind_block[2]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[1]) & (moving_av_0_6 < ind_block[2]).astype(int)) ==1)[0][0]]+9)
                            
            b_4 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[2]) & (moving_av_0_6 < ind_block[3]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[2]) & (moving_av_0_6 < ind_block[3]).astype(int)) ==1)[0][0]]+9)
                
            b_5 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[3]) & (moving_av_0_6 < ind_block[4]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[3]) & (moving_av_0_6 < ind_block[4]).astype(int)) ==1)[0][0]]+9)
                
            b_6 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[4]) & (moving_av_0_6 < ind_block[5]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[4]) & (moving_av_0_6 < ind_block[5]).astype(int)) ==1)[0][0]]+9)
                
            b_7 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[5]) & (moving_av_0_6 < ind_block[6]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[5]) & (moving_av_0_6 < ind_block[6]).astype(int)) ==1)[0][0]]+9)
                
            b_8 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[6]) & (moving_av_0_6 < ind_block[7]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[6]) & (moving_av_0_6 < ind_block[7]).astype(int)) ==1)[0][0]]+9)
                
            b_9 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[7]) & (moving_av_0_6 < ind_block[8]).astype(int)) ==1)[0][0]],\
                            moving_av_0_6[np.where(((moving_av_0_6 > ind_block[7]) & (moving_av_0_6 < ind_block[8]).astype(int)) ==1)[0][0]]+9)
                
            b_10 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[8]) & (moving_av_0_6 < ind_block[9]).astype(int)) ==1)[0][0]],\
                             moving_av_0_6[np.where(((moving_av_0_6 > ind_block[8]) & (moving_av_0_6 < ind_block[9]).astype(int)) ==1)[0][0]]+9)
            
            b_11 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[9]) & (moving_av_0_6 < ind_block[10]).astype(int)) ==1)[0][0]],\
                             moving_av_0_6[np.where(((moving_av_0_6 > ind_block[9]) & (moving_av_0_6 < ind_block[10]).astype(int)) ==1)[0][0]]+9)
            
            b_12 = np.arange(moving_av_0_6[np.where(((moving_av_0_6 > ind_block[10]) & (moving_av_0_6 < ind_block[11]).astype(int)) ==1)[0][0]],\
                             moving_av_0_6[np.where(((moving_av_0_6 > ind_block[10]) & (moving_av_0_6 < ind_block[11]).astype(int)) ==1)[0][0]]+9)
            
            all_ind_triggered_on_beh = np.concatenate((b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_10,b_11,b_12))
            firing_rates  = firing_rates[all_ind_triggered_on_beh]
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            
            state = state[all_ind_triggered_on_beh]
            choices = choices[all_ind_triggered_on_beh]            
            reward = reward[all_ind_triggered_on_beh]
            reward_choice = choices*reward            
            trials_since_block = np.tile(np.arange(9),12)
            interaction_trials_choice = trials_since_block*choices
            ones = np.ones(len(interaction_trials_choice))
            
            predictors_all = OrderedDict([('State', state),
                                          ('Reward', reward),
                                          ('Choice', choices),
                                          ('Reward Choice Int', reward_choice),
                                          ('Trials in Block', trials_since_block),
                                          ('Trials x Choice', interaction_trials_choice),
                                          ('ones', ones)])
            
            X = np.vstack(predictors_all.values()).T[:n_trials,:].astype(float)
            n_predictors = X.shape[1]
            y = firing_rates.reshape([len(firing_rates),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats = reg_f.regression_code(y, X)
            C.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            cpd.append(re._CPD(X,y).reshape(n_neurons,n_timepoints, n_predictors))
              
            if perm:
                for i in range(perm):
                    # shuffle the start of all blocks
                    starts = []
                    for iu in range(12):
                        start = np.random.randint(n_trials-9)
                        start_list = np.arange(start,start+9)
                        starts.append(start_list)
                        
                    ind = list(flatten(starts))
                    X_perm = X[ind,:]
                    tstats = reg_f.regression_code(y, X_perm)
        
                    C_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
                    cpd_perm[i].append(re._CPD(X_perm,y).reshape(n_neurons, n_timepoints, n_predictors))

  
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.nanmean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        p = np.percentile(cpd_perm,95, axis = 0)
           
     
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    
    return C, cpd, predictors_all, cpd_perm, p

def regression_behavior_triggered_plot():
    C, cpd, predictors_all, cpd_perm, p =  regression_time_of_trial_reversal_triggered(data_HP,experiment_aligned_HP, perm = 1)
    cpd = cpd[:,:-1]

    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    pred = [*predictors]#[4:]
    
    plt.figure()
    t = np.arange(0,63)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =pred[i], color = c[i])
        #plt.plot(p[:,i], label =pred[i], color = c[i], linestyle = '--')
        plt.fill_between(t, np.zeros(t.shape),
            np.max(np.percentile(cpd_perm[:,:,i],95,axis=0),axis=0),alpha=0.2, facecolor=c[i])
    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    
    
def pca(data, experiment_aligned):
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value,firing_rates_time,dm_time = regression_time_of_trial(data,experiment_aligned,perm = False,trial_hypothesis = False, time_hypothesis = True, place_cells = False)

    # s is sesssion
    design = dm_time[s]
    block = design[:,4]
    block_df = np.diff(block)
    ind_block = np.where(block_df ==1)[0]
    for firing in firing_rates_time:
        
        time = np.transpose(firing,[1,0,2]).reshape(firing.shape[1],firing.shape[0]*firing.shape[2])
        u,s,v = np.linalg.svd(time)
      
        proj_u =  np.linalg.multi_dot((u.T,time))
        proj_u = proj_u.reshape(firing.shape[1],firing.shape[0],firing.shape[2])
    
        #proj_v =  np.linalg.multi_dot((v.T,time.T))
        #proj_v = proj_v.reshape(firing.shape[0],firing.shape[2],firing.shape[1])
        fig = plt.figure(1, figsize=[14,12], clear=True)
        ax3D = plt.subplot2grid([3,3], [0,0], rowspan=3, colspan=2, projection='3d')
       
        cmap =  palettable.scientific.sequential.Bilbao_12.mpl_colormap
  
        colors=[cmap(float(ii)/(n-1)) for ii in range(n-1)][0::15]
        c = 0
        for i in range(proj_u.shape[0]):
            if i > 0:
                if i < ind_block[0]:
                    c += 1
                    x = proj_u[1,i,:]
                    y = proj_u[2,i,:]
                    z = proj_u[3,i,:]
                    ax3D.scatter(x, y, z, color=colors[c])
                elif i == ind_block[0]:
                    c = 0
                elif i > ind_block[0] and i < ind_block[1]:
                    c += 1
                    x = proj_u[1,i,:]
                    y = proj_u[2,i,:]
                    z = proj_u[3,i,:]
                    ax3D.scatter(x, y, z, color=colors[c])
 

def _plot_P_values(p_values, t, ax, y0):
    '''Indicate significance levels with dots of different sizes above plot.
    Arguments:
    p_values   : matrix of P values [n_timepoints, n_predictors]
    t          : time points used as x values for plotting
    ax         : matplotlib axes with traces already drawn on them.
    y0         : The y value at which to start plotting the p value indicators.
    multi_comp : Boolean indicating whether to use Benjamini-Hochberg multiple 
                 comparison correction
    '''
    for i in range(p_values.shape[1]):
        y = y0*(1+0.04*i)
        p_vals = p_values[:,i]
        t05 = t[p_vals == 0.05]
        t00 = t[p_vals == 0.001]
        #print(t05)
        #color = ax.get_lines()[i].get_color()
        plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color='red')
        plt.plot(t00, np.ones(t00.shape)*y, '.', markersize=9, color='black')     
        
def plot_neurons(data, experiment_aligned, title):
    
    C_HP_roll, cpd_HP_rol, predictors_HP_rol, cpd_perm_HP_rol, cpd_p_value_HP_rol, p_value_HP_rol,firing_rates_time_HP_rol,dm_time_HP_rol = regression_time_of_trial(data_HP,experiment_aligned_HP,perm = 1000,trial_hypothesis = False, time_hypothesis = True, place_cells = False)
    C_PFC_rol, cpd_PFC_rol, predictors_PFC_rol, cpd_perm_PFC_rol, cpd_p_value_PFC_rol, p_value_PFC_rol,firing_rates_time_PFC_rol,dm_time_PFC_rol = regression_time_of_trial(data_PFC,experiment_aligned_PFC,perm = 1000,trial_hypothesis = False, time_hypothesis = True, place_cells = False)
    #C, cpd, predictors, cpd_perm, cpd_p_value, p_value,firing_rates_time,dm_time = regression_time_of_trial(data,experiment_aligned,perm = 1,trial_hypothesis = False, time_hypothesis = True, place_cells = False)

    
    c =  wes.Darjeeling2_5.mpl_colors + wes.Mendl_4.mpl_colors +wes.GrandBudapest1_4.mpl_colors


    plt.figure()
    t = np.arange(0,63)
    cpd = cpd_PFC_rol[:,:-1]
    cpd_perm = cpd_perm_PFC_rol[:,:-1]
    p = [*predictors_PFC_rol]#[4:]
    values_95 = np.max(np.percentile(cpd_perm,95,axis = 0),axis=0)
    values_99 = np.max(np.percentile(cpd_perm,99,axis = 0),axis=0)
    array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
    for i in range(cpd.shape[1]):
        array_pvals[(np.where(cpd[:,i] > values_95[i])[0]),i] = 0.05
        array_pvals[(np.where(cpd[:,i] > values_99[i])[0]),i] = 0.001
 
    ymax = np.max(cpd)
    
    for i in np.arange(cpd.shape[1]):
      #  perm_plot = np.max(np.percentile(cpd_perm[:,:,i],95,axis = 0),axis=0)
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        y = ymax*(1+0.04*i)
        p_vals = array_pvals[:,i]
        t05 = t[p_vals == 0.05]
        t00 = t[p_vals == 0.001]
        plt.plot(t05, np.ones(t05.shape)*y, '.', markersize=3, color=c[i])
        plt.plot(t00, np.ones(t00.shape)*y, '.', markersize=9, color=c[i])     
        
       

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    sns.despine()
    C_mean_time = np.mean(C[:,:,36:40],2)
    C_time = C_mean_time[4]
    ind_time_neurons = []
    
    for n,nn in enumerate(C_time):
        
        if abs(nn) > 3:
            ind_time_neurons.append(n)
            
    dims = int(len(ind_time_neurons)/4)+1
    for i,ind in enumerate(ind_time_neurons):
        
        plt.subplot(dims,4,i+1)
        time = C[4,:,:]
        reward = C[1,:,:]
        choice = C[2,:,:]
        state = C[0,:,:]
        reward_choice_int = C[3,:,:]

        isl = wes.Moonrise5_6.mpl_colors
        plt.plot(time[ind_time_neurons[i]], color = isl[0], label = 'Time')
        plt.plot(reward[ind_time_neurons[i]], color = isl[1], label = 'Reward')
        plt.plot(reward_choice_int[ind_time_neurons[i]], color = isl[2], label = 'Reward X Choice')

        plt.plot(choice[ind_time_neurons[i]], color = isl[3], label = 'Choice')
        plt.plot(state[ind_time_neurons[i]], color = isl[4], label = 'State')
        plt.title(str(ind))
    plt.tight_layout()
    plt.legend()

    pdf = PdfPages('/Users/veronikasamborska/Desktop/Cells_time_code/'+ title +'.pdf')
        
    n_count = 0  
    i = 0
    ind_time_neurons = np.arange(0,C_time.shape[0])
    plt.ioff()
    
    for s,session in enumerate(firing_rates_time):
        neurons = session.shape[1]
        for n in range(neurons):
            n_count += 1
            if n_count in ind_time_neurons:
                i+=1
                plot_neuron = np.mean(session[:,n,:],1)

                design = dm_time[s]
                block = design[:,4]
                choice = design[:,1]
                reward = design[:,2]
                state = design[:,0]
                block_df = np.diff(block)
                ind_block = np.where(block_df ==1)[0]
                block_length = np.append(np.diff(ind_block), ind_block[0])

                min_ind = np.int(np.min(block_length))-1
    
                plt_gaus = gaussian_filter1d(plot_neuron,1)
                #plt_gaus = plot_neuron

                plt.figure(figsize =(10,10))
                plt.subplot(4,4,1)
                plt.plot(plt_gaus[:ind_block[0]][-min_ind:], color = 'black', label = 'Time B 1')
                rew = reward+np.max(plt_gaus[:ind_block[0]][-min_ind:])
                plt.plot(rew[:ind_block[0]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[:ind_block[0]][-min_ind:])+2
                plt.plot(ch[:ind_block[0]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[:ind_block[0]][-min_ind:])+ 4
                if state[:ind_block[0]][-min_ind:][0]== 1:
                    plt.plot(st[:ind_block[0]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[:ind_block[0]][-min_ind:][0] == 0:
                    plt.plot(st[:ind_block[0]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

               # plt.plot(plt_gaus[:ind_block[0]], color = 'black')

                plt.ylabel('Firing Rate')
                plt.title(str(n_count))

               # plt.ylim(np.min(plt_gaus),np.max(plt_gaus))

                plt.subplot(4,4,2)
                plt.plot(plt_gaus[ind_block[0]:ind_block[1]][-min_ind:], color = 'black', label = 'Time B 2')     
                rew = reward+np.max(plt_gaus[ind_block[0]:ind_block[1]][-min_ind:])
                plt.plot(rew[ind_block[0]:ind_block[1]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[0]:ind_block[1]][-min_ind:])+2
                plt.plot(ch[ind_block[0]:ind_block[1]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[0]:ind_block[1]][-min_ind:])+ 4
                if state[ind_block[0]:ind_block[1]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[0]:ind_block[1]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[0]:ind_block[1]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[0]:ind_block[1]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

                plt.ylabel('Firing Rate')
             #   plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                plt.subplot(4,4,3)
                plt.plot(plt_gaus[ind_block[1]:ind_block[2]][-min_ind:], color = 'black', label = 'Time B 3')
                rew = reward+np.max(plt_gaus[ind_block[1]:ind_block[2]][-min_ind:])
                plt.plot(rew[ind_block[1]:ind_block[2]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[1]:ind_block[2]][-min_ind:])+2
                plt.plot(ch[ind_block[1]:ind_block[2]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[1]:ind_block[2]][-min_ind:])+ 4
                if state[ind_block[1]:ind_block[2]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[1]:ind_block[2]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[1]:ind_block[2]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[1]:ind_block[2]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

               # plt.plot(plt_gaus[ind_block[1]:ind_block[2]], color = 'black')

                plt.ylabel('Firing Rate')
              #  plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                plt.subplot(4,4,4)
                plt.plot(plt_gaus[ind_block[2]:ind_block[3]][-min_ind:], color = 'black' , label = 'Time B 4')
                rew = reward+np.max(plt_gaus[ind_block[2]:ind_block[3]][-min_ind:])
                plt.plot(rew[ind_block[2]:ind_block[3]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[2]:ind_block[3]][-min_ind:])+2
                plt.plot(ch[ind_block[2]:ind_block[3]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[2]:ind_block[3]][-min_ind:])+ 4
                if state[ind_block[2]:ind_block[3]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[2]:ind_block[3]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[2]:ind_block[3]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[2]:ind_block[3]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

                #plt.plot(plt_gaus[ind_block[2]:ind_block[3]], color = 'black')

                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                plt.subplot(4,4,5)
                plt.plot(plt_gaus[ind_block[3]:ind_block[4]][-min_ind:], color = 'black', label = 'Time B 5')
                rew = reward+np.max(plt_gaus[ind_block[3]:ind_block[4]][-min_ind:])
                plt.plot(rew[ind_block[3]:ind_block[4]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[3]:ind_block[4]][-min_ind:])+2
                plt.plot(ch[ind_block[3]:ind_block[4]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[3]:ind_block[4]][-min_ind:])+ 4
                if state[ind_block[3]:ind_block[4]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[3]:ind_block[4]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[3]:ind_block[4]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[3]:ind_block[4]][-min_ind:],color =  isl[5], label = 'B')             
               
                plt.legend()

                #plt.plot(plt_gaus[ind_block[3]:ind_block[4]], color = 'black')

                plt.ylabel('Firing Rate')
               # plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                plt.subplot(4,4,6)
                plt.plot(plt_gaus[ind_block[4]:ind_block[5]][-min_ind:], color = 'black', label = 'Time B 6')
                rew = reward+np.max(plt_gaus[ind_block[4]:ind_block[5]][-min_ind:])
                plt.plot(rew[ind_block[4]:ind_block[5]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[4]:ind_block[5]][-min_ind:])+2
                plt.plot(ch[ind_block[4]:ind_block[5]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[4]:ind_block[5]][-min_ind:])+ 4
                if state[ind_block[4]:ind_block[5]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[4]:ind_block[5]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[4]:ind_block[5]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[4]:ind_block[5]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

                #plt.plot(plt_gaus[ind_block[4]:ind_block[5]], color = 'black')
                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                plt.subplot(4,4,7)
                plt.plot(plt_gaus[ind_block[5]:ind_block[6]][-min_ind:], color = 'black', label = 'Time B 7')
                rew = reward+np.max(plt_gaus[ind_block[5]:ind_block[6]][-min_ind:])
                plt.plot(rew[ind_block[5]:ind_block[6]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[5]:ind_block[6]][-min_ind:])+2
                plt.plot(ch[ind_block[5]:ind_block[6]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[5]:ind_block[6]][-min_ind:])+ 4
                if state[ind_block[5]:ind_block[6]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[5]:ind_block[6]][-min_ind:],color = isl[4], label = 'A')             
                elif state[ind_block[5]:ind_block[6]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[5]:ind_block[6]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

              
                #plt.plot(plt_gaus[ind_block[5]:ind_block[6]], color = 'black')
                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                
                plt.subplot(4,4,8)
                plt.plot(plt_gaus[ind_block[6]:ind_block[7]][-min_ind:], color = 'black', label = 'Time B 8')
                rew = reward+np.max(plt_gaus[ind_block[6]:ind_block[7]][-min_ind:])
                plt.plot(rew[ind_block[6]:ind_block[7]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[6]:ind_block[7]][-min_ind:])+2
                plt.plot(ch[ind_block[6]:ind_block[7]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[6]:ind_block[7]][-min_ind:])+ 4
                if state[ind_block[6]:ind_block[7]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[6]:ind_block[7]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[6]:ind_block[7]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[6]:ind_block[7]][-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

                #plt.plot(plt_gaus[ind_block[6]:ind_block[7]], color = 'black')

                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                
                plt.subplot(4,4,9)
                plt.plot(plt_gaus[ind_block[7]:ind_block[8]][-min_ind:], color = 'black', label = 'Time B 9')
                rew = reward+np.max(plt_gaus[ind_block[7]:ind_block[8]][-min_ind:])
                plt.plot(rew[ind_block[7]:ind_block[8]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[7]:ind_block[8]][-min_ind:])+2
                plt.plot(ch[ind_block[7]:ind_block[8]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[7]:ind_block[8]][-min_ind:])+ 4
                if state[ind_block[7]:ind_block[8]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[7]:ind_block[8]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[7]:ind_block[8]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[7]:ind_block[8]][-min_ind:],color =  isl[5], label = 'B')             
               
                plt.legend()

                #plt.plot(plt_gaus[ind_block[7]:ind_block[8]], color = 'black')

                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                
                plt.subplot(4,4,10)
                plt.plot(plt_gaus[ind_block[8]:ind_block[9]][-min_ind:], color = 'black', label = 'Time B 10')
                rew = reward+np.max(plt_gaus[ind_block[8]:ind_block[9]][-min_ind:])
                plt.plot(rew[ind_block[8]:ind_block[9]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[8]:ind_block[9]][-min_ind:])+2
                plt.plot(ch[ind_block[8]:ind_block[9]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[8]:ind_block[9]][-min_ind:])+ 4
                if state[ind_block[8]:ind_block[9]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[8]:ind_block[9]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[8]:ind_block[9]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[8]:ind_block[9]][-min_ind:], color =  isl[5], label = 'B')             
                
                plt.legend()

                #plt.plot(plt_gaus[ind_block[8]:ind_block[9]], color = 'black')

                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                
                plt.subplot(4,4,11)
                plt.plot(plt_gaus[ind_block[9]:ind_block[10]][-min_ind:], color = 'black', label = 'Time B 11')
                rew = reward+np.max(plt_gaus[ind_block[9]:ind_block[10]][-min_ind:])
                plt.plot(rew[ind_block[9]:ind_block[10]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[9]:ind_block[10]][-min_ind:])+2
                plt.plot(ch[ind_block[9]:ind_block[10]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[9]:ind_block[10]][-min_ind:])+ 4
                if state[ind_block[9]:ind_block[10]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[9]:ind_block[10]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[9]:ind_block[10]][-min_ind:][0] == 0:
                    plt.plot(st[ind_block[9]:ind_block[10]][-min_ind:], color =  isl[5], label = 'B')             
                               
                plt.legend()

                #plt.plot(plt_gaus[ind_block[9]:ind_block[10]], color = 'black')

                plt.ylabel('Firing Rate')
                #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)

                
                plt.subplot(4,4,12)
                plt.plot(plt_gaus[ind_block[10]:ind_block[11]][-min_ind:], color = 'black', label = 'Time B 12')
                rew = reward+np.max(plt_gaus[ind_block[10]:ind_block[11]][-min_ind:])
                plt.plot(rew[ind_block[10]:ind_block[11]][-min_ind:], color = 'grey')
                ch = choice+np.max(plt_gaus[ind_block[10]:ind_block[11]][-min_ind:])+2
                plt.plot(ch[ind_block[10]:ind_block[11]][-min_ind:], color = 'green')
                st = state + np.max(plt_gaus[ind_block[10]:ind_block[11]][-min_ind:])+ 4
                if state[ind_block[10]:ind_block[11]][-min_ind:][0] == 1:
                    plt.plot(st[ind_block[10]:ind_block[11]][-min_ind:], color =  isl[4], label = 'A')             
                elif state[ind_block[10]:ind_block[11]][-min_ind:][0] == 0:
                    plt.plot((st[ind_block[10]:ind_block[11]])[-min_ind:], color =  isl[5], label = 'B')             
               
                plt.legend()

              
                #plt.plot(plt_gaus[ind_block[10]:ind_block[11]], color = 'black')

               # plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)
                plt.ylabel('Firing Rate')
                
                
                all_blocks = np.mean([plt_gaus[ind_block[10]:ind_block[11]][-min_ind:],plt_gaus[ind_block[9]:ind_block[10]][-min_ind:],\
                                     plt_gaus[ind_block[8]:ind_block[9]][-min_ind:], plt_gaus[ind_block[7]:ind_block[8]][-min_ind:],\
                                    plt_gaus[ind_block[6]:ind_block[7]][-min_ind:],plt_gaus[ind_block[5]:ind_block[6]][-min_ind:],\
                                    plt_gaus[ind_block[4]:ind_block[5]][-min_ind:],plt_gaus[ind_block[3]:ind_block[4]][-min_ind:],\
                                    plt_gaus[ind_block[2]:ind_block[3]][-min_ind:],plt_gaus[ind_block[1]:ind_block[2]][-min_ind:],\
                                    plt_gaus[ind_block[0]:ind_block[1]][-min_ind:],plt_gaus[:ind_block[0]][-min_ind:]],0)
               
                all_blocks_std = (np.std([plt_gaus[ind_block[10]:ind_block[11]][-min_ind:],plt_gaus[ind_block[9]:ind_block[10]][-min_ind:],\
                                     plt_gaus[ind_block[8]:ind_block[9]][-min_ind:], plt_gaus[ind_block[7]:ind_block[8]][-min_ind:],\
                                    plt_gaus[ind_block[6]:ind_block[7]][-min_ind:],plt_gaus[ind_block[5]:ind_block[6]][-min_ind:],\
                                    plt_gaus[ind_block[4]:ind_block[5]][-min_ind:],plt_gaus[ind_block[3]:ind_block[4]][-min_ind:],\
                                    plt_gaus[ind_block[2]:ind_block[3]][-min_ind:],plt_gaus[ind_block[1]:ind_block[2]][-min_ind:],\
                                    plt_gaus[ind_block[0]:ind_block[1]][-min_ind:],plt_gaus[:ind_block[0]][-min_ind:]],0))/np.sqrt(12)
                    
                all_choices = np.mean([choice[ind_block[10]:ind_block[11]][-min_ind:],choice[ind_block[9]:ind_block[10]][-min_ind:],\
                                     choice[ind_block[8]:ind_block[9]][-min_ind:], choice[ind_block[7]:ind_block[8]][-min_ind:],\
                                    choice[ind_block[6]:ind_block[7]][-min_ind:],choice[ind_block[5]:ind_block[6]][-min_ind:],\
                                    choice[ind_block[4]:ind_block[5]][-min_ind:],choice[ind_block[3]:ind_block[4]][-min_ind:],\
                                    choice[ind_block[2]:ind_block[3]][-min_ind:],choice[ind_block[1]:ind_block[2]][-min_ind:],\
                                    choice[ind_block[0]:ind_block[1]][-min_ind:],choice[:ind_block[0]][-min_ind:]],0)
              
                all_choices_std = (np.std([choice[ind_block[10]:ind_block[11]][-min_ind:],choice[ind_block[9]:ind_block[10]][-min_ind:],\
                                     choice[ind_block[8]:ind_block[9]][-min_ind:], choice[ind_block[7]:ind_block[8]][-min_ind:],\
                                    choice[ind_block[6]:ind_block[7]][-min_ind:],choice[ind_block[5]:ind_block[6]][-min_ind:],\
                                    choice[ind_block[4]:ind_block[5]][-min_ind:],choice[ind_block[3]:ind_block[4]][-min_ind:],\
                                    choice[ind_block[2]:ind_block[3]][-min_ind:],choice[ind_block[1]:ind_block[2]][-min_ind:],\
                                    choice[ind_block[0]:ind_block[1]][-min_ind:],choice[:ind_block[0]][-min_ind:]],0))/np.sqrt(12)
              
                all_rewards = np.mean([reward[ind_block[10]:ind_block[11]][-min_ind:],reward[ind_block[9]:ind_block[10]][-min_ind:],\
                                     reward[ind_block[8]:ind_block[9]][-min_ind:], reward[ind_block[7]:ind_block[8]][-min_ind:],\
                                    reward[ind_block[6]:ind_block[7]][-min_ind:],reward[ind_block[5]:ind_block[6]][-min_ind:],\
                                    reward[ind_block[4]:ind_block[5]][-min_ind:],reward[ind_block[3]:ind_block[4]][-min_ind:],\
                                    reward[ind_block[2]:ind_block[3]][-min_ind:],reward[ind_block[1]:ind_block[2]][-min_ind:],\
                                    reward[ind_block[0]:ind_block[1]][-min_ind:],reward[:ind_block[0]][-min_ind:]],0)
              
                all_rewards_std = (np.std([reward[ind_block[10]:ind_block[11]][-min_ind:],reward[ind_block[9]:ind_block[10]][-min_ind:],\
                                     reward[ind_block[8]:ind_block[9]][-min_ind:], reward[ind_block[7]:ind_block[8]][-min_ind:],\
                                    reward[ind_block[6]:ind_block[7]][-min_ind:],reward[ind_block[5]:ind_block[6]][-min_ind:],\
                                    reward[ind_block[4]:ind_block[5]][-min_ind:],reward[ind_block[3]:ind_block[4]][-min_ind:],\
                                    reward[ind_block[2]:ind_block[3]][-min_ind:],reward[ind_block[1]:ind_block[2]][-min_ind:],\
                                    reward[ind_block[0]:ind_block[1]][-min_ind:],reward[:ind_block[0]][-min_ind:]],0))/np.sqrt(12)
              
                all_blocks_array = np.vstack((plt_gaus[ind_block[10]:ind_block[11]][-min_ind:],plt_gaus[ind_block[9]:ind_block[10]][-min_ind:],\
                                     plt_gaus[ind_block[8]:ind_block[9]][-min_ind:], plt_gaus[ind_block[7]:ind_block[8]][-min_ind:],\
                                    plt_gaus[ind_block[6]:ind_block[7]][-min_ind:],plt_gaus[ind_block[5]:ind_block[6]][-min_ind:],\
                                    plt_gaus[ind_block[4]:ind_block[5]][-min_ind:],plt_gaus[ind_block[3]:ind_block[4]][-min_ind:],\
                                    plt_gaus[ind_block[2]:ind_block[3]][-min_ind:],plt_gaus[ind_block[1]:ind_block[2]][-min_ind:],\
                                    plt_gaus[ind_block[0]:ind_block[1]][-min_ind:],plt_gaus[:ind_block[0]][-min_ind:]))
               
                all_rewards_array = np.vstack((reward[ind_block[10]:ind_block[11]][-min_ind:],reward[ind_block[9]:ind_block[10]][-min_ind:],\
                                     reward[ind_block[8]:ind_block[9]][-min_ind:], reward[ind_block[7]:ind_block[8]][-min_ind:],\
                                    reward[ind_block[6]:ind_block[7]][-min_ind:],reward[ind_block[5]:ind_block[6]][-min_ind:],\
                                    reward[ind_block[4]:ind_block[5]][-min_ind:],reward[ind_block[3]:ind_block[4]][-min_ind:],\
                                    reward[ind_block[2]:ind_block[3]][-min_ind:],reward[ind_block[1]:ind_block[2]][-min_ind:],\
                                    reward[ind_block[0]:ind_block[1]][-min_ind:],reward[:ind_block[0]][-min_ind:]))
              
                all_choices_array = np.vstack((choice[ind_block[10]:ind_block[11]][-min_ind:],choice[ind_block[9]:ind_block[10]][-min_ind:],\
                                     choice[ind_block[8]:ind_block[9]][-min_ind:], choice[ind_block[7]:ind_block[8]][-min_ind:],\
                                    choice[ind_block[6]:ind_block[7]][-min_ind:],choice[ind_block[5]:ind_block[6]][-min_ind:],\
                                    choice[ind_block[4]:ind_block[5]][-min_ind:],choice[ind_block[3]:ind_block[4]][-min_ind:],\
                                    choice[ind_block[2]:ind_block[3]][-min_ind:],choice[ind_block[1]:ind_block[2]][-min_ind:],\
                                    choice[ind_block[0]:ind_block[1]][-min_ind:],choice[:ind_block[0]][-min_ind:]))
              
                all_states_array = np.vstack((state[ind_block[10]:ind_block[11]][-min_ind:],state[ind_block[9]:ind_block[10]][-min_ind:],\
                                     state[ind_block[8]:ind_block[9]][-min_ind:], state[ind_block[7]:ind_block[8]][-min_ind:],\
                                    state[ind_block[6]:ind_block[7]][-min_ind:],state[ind_block[5]:ind_block[6]][-min_ind:],\
                                    state[ind_block[4]:ind_block[5]][-min_ind:],state[ind_block[3]:ind_block[4]][-min_ind:],\
                                    state[ind_block[2]:ind_block[3]][-min_ind:],state[ind_block[1]:ind_block[2]][-min_ind:],\
                                    state[ind_block[0]:ind_block[1]][-min_ind:],state[:ind_block[0]][-min_ind:]))
            
               
                plt.subplot(4,4,13)
                plt.plot(all_blocks, color = 'red', label = 'All blocks average')
                plt.fill_between(np.arange(len(all_blocks)), all_blocks-all_blocks_std, all_blocks+all_blocks_std, alpha=0.2, color = 'red')

                plt.legend()
                plt.ylabel('Firing Rate')
                plt.xlabel('Trial Number')

                plt.subplot(4,4,14)
                correlation_time_choices = np.round(np.corrcoef(all_choices,all_blocks)[0,1],2)
                correlation_time_reward = np.round(np.corrcoef(all_choices,all_rewards)[0,1],2)
              
                plt.plot(all_choices, color = 'green', label = 'Choice' + ' ' +'r with time =' + str(correlation_time_choices))
                plt.fill_between(np.arange(len(all_choices)), all_choices-all_choices_std, all_choices+all_choices_std, alpha=0.3, color = 'green')

                plt.plot(all_rewards, color = 'grey', label = 'Reward'+ ' ' +'r with time =' + str(correlation_time_reward))
                plt.fill_between(np.arange(len(all_rewards)), all_rewards-all_rewards_std, all_rewards+all_rewards_std, alpha=0.3, color = 'grey')

                 #plt.ylim(np.min(plt_gaus),np.max(plt_gaus)+0.5)
                plt.ylabel('Av Reward/Choice on 12 blocks')
                plt.xlabel('Trial Number')
                plt.tight_layout()
                plt.legend()
                
                
                time = C[4,:,:]
                reward = C[1,:,:]
                choice = C[2,:,:]
                state = C[0,:,:]
                reward_choice_int = C[3,:,:]

                isl = wes.Moonrise5_6.mpl_colors
                plt.subplot(4,4,15)
                plt.plot(time[n_count], color = isl[0], label = 'Time')
                plt.plot(reward[n_count], color = isl[1], label = 'Reward')
                plt.plot(reward_choice_int[n_count], color = isl[2], label = 'Reward X Choice')
        
                plt.plot(choice[n_count], color = isl[3], label = 'Choice')
                plt.plot(state[n_count], color = isl[4], label = 'State')
                plt.xlabel('Time in Trial')
                plt.ylabel('Coefficient')
                plt.legend()

                # Block indicies for averaging across A and B blocks
                a_choices = []
                a_states = []
                a_reward = []
                a_firing = []
                b_choices = []
                b_states = []
                b_reward = []
                b_firing = []
                for c, choice_arr in enumerate(all_states_array):
                    if all_states_array[c][0]  == 1:
                        a_choices.append(all_choices_array[c])
                        a_states.append(all_states_array[c])
                        a_reward.append(all_rewards_array[c])
                        a_firing.append(all_blocks_array[c])
                    elif all_states_array[c][0]  == 0:
                        b_choices.append(all_choices_array[c])
                        b_states.append(all_states_array[c])
                        b_reward.append(all_rewards_array[c])
                        b_firing.append(all_blocks_array[c])


                a_av = np.mean(a_firing,0)
                a_std = (np.std(a_firing,0))/np.sqrt(len(a_av))

                b_av = np.mean(b_firing,0)
                b_std =  (np.std(b_firing,0))/np.sqrt(len(b_av))
                
                plt.subplot(4,4,16)

                plt.plot(a_av, color = isl[4], label = 'A block')
                plt.fill_between(np.arange(len(a_av)), a_av-a_std, a_av+a_std, alpha=0.3, color = isl[4])
                plt.plot(b_av, color = isl[5], label = 'B block')
                plt.fill_between(np.arange(len(b_av)), b_av-b_std, b_av+b_std, alpha=0.3, color = isl[5])

                plt.legend()
                sns.despine()
                pdf.savefig()
                plt.clf()
    pdf.close()


             
                

def plot(HP,PFC, experiment_aligned_PFC, data_HP,experiment_aligned_HP):
    
    n = 1
    
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(HP,experiment_aligned_HP,perm = n,trial_hypothesis = True, time_hypothesis = False, place_cells = False)
    cpd = cpd[:,:-1]
    cpd= cpd[:,4:]
    cpd_p_value = p[:,:-1]
    cpd_p_value = p[:,4:]
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
   #c = [*colors]
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors][4:]
    
    fig = plt.figure(1)
    
    fig.add_subplot(3,2,1)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.plot(cpd_p_value[:,i], label =p[i] +' '+'shuffled', color = c[i], alpha = 0.5, linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    plt.title('HP')
    
    
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(PFC,experiment_aligned_PFC,perm = n,trial_hypothesis = True, time_hypothesis = False, place_cells = False)
    cpd = cpd[:,:-1]
    cpd= cpd[:,4:]
    cpd_p_value = p[:,:-1]
    cpd_p_value = p[:,4:]
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    c = [*colors]
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors][4:]
    
    fig = plt.figure(1)
    
    fig.add_subplot(3,2,2)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.plot(cpd_p_value[:,i], label =p[i] +' '+'shuffled', color = c[i], alpha = 0.5, linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])

    plt.title('PFC')
    
   
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(data_HP,experiment_aligned_HP,perm = n,trial_hypothesis = False, time_hypothesis = True, place_cells = False)
    cpd = cpd[:,:-1]
    cpd= cpd[:,4:]
    cpd_p_value = p[:,:-1]
    cpd_p_value = p[:,4:]
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
   #c = [*colors]
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors][4:]
    
    
    fig.add_subplot(3,2,3)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.plot(cpd_p_value[:,i], label =p[i] +' '+'shuffled', color = c[i], alpha = 0.5, linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    
    
    
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(data_PFC,experiment_aligned_PFC,perm = n,trial_hypothesis = False, time_hypothesis = True, place_cells = False)
    cpd = cpd[:,:-1]
    cpd= cpd[:,4:]
    cpd_p_value = p[:,:-1]
    cpd_p_value = p[:,4:]
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    c = [*colors]
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors][4:]
    
    
    fig.add_subplot(3,2,4)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.plot(cpd_p_value[:,i], label =p[i] +' '+'shuffled', color = c[i], alpha = 0.5, linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])

    
   
   
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(data_HP,experiment_aligned_HP,perm = n,trial_hypothesis = False, time_hypothesis = False, place_cells = True)
    cpd = cpd[:,:-1]
    cpd= cpd[:,4:]
    cpd_p_value = p[:,:-1]
    cpd_p_value = p[:,4:]
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
   #c = [*colors]
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors][4:]
    
    
    fig.add_subplot(3,2,5)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.plot(cpd_p_value[:,i], label =p[i] +' '+'shuffled', color = c[i], alpha = 0.5, linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])
    
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(data_PFC,experiment_aligned_PFC,perm = n,trial_hypothesis = False, time_hypothesis = False, place_cells = True)
    cpd = cpd[:,:-1]
    cpd= cpd[:,4:]
    cpd_p_value = p[:,:-1]
    cpd_p_value = p[:,4:]
    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    c = [*colors]
    c =  ['violet', 'black', 'green', 'blue', 'turquoise', 'grey', 'yellow', 'pink',\
          'purple', 'darkblue', 'darkred', 'darkgreen','darkyellow','lightgreen']
    p = [*predictors][4:]
    
    
    fig.add_subplot(3,2,6)
    for i in np.arange(cpd.shape[1]):
        plt.plot(cpd[:,i], label =p[i], color = c[i])
        plt.plot(cpd_p_value[:,i], label =p[i] +' '+'shuffled', color = c[i], alpha = 0.5, linestyle = '--')

    plt.legend()
    plt.ylabel('CPD')
    plt.xlabel('Time in Trial')
    plt.xticks([25, 35, 42], ['I', 'C', 'R'])

    
    plt.tight_layout()
    sns.despine()
