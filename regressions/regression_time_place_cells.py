#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:22:08 2020

@author: veronikasamborska
"""
import sys

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/remapping')
sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/preprocessing')

import numpy as np
import matplotlib.pyplot as plt
import regression_function as reg_f
import regressions as re
from collections import OrderedDict
from matplotlib import colors as mcolors
from scipy.ndimage import gaussian_filter1d
from matplotlib.cbook import flatten


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

plt.rc('font', **font)

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
   


# Can you see if this explains more variance if it is literally time (expressed eg in seconds)#
#   vs block time (expressed as
# a fraction of the current block - so start of block =0 and end of block=1). 

#(2) In the last case (fraction), can you add another regressor (T-0.5)^2  (and its interaction with choice). 

def regression_time_of_trial(data,experiment_aligned_data,perm = False, trial_hypothesis = False, time_hypothesis = False, place_cells = False):
    runs = 1
    
    print(runs)
    C = []
    cpd = []
    
   
    dm = data['DM']
    firing = data['Data']
    
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
            
            #Time squared, interactions
            time_sq = (np.asarray(time_in_block)-0.5)**2
            choice_time_sq = choices*time_sq
            time_choice_int = time_in_block*choices
            
            #Trial squared, interactions

            trial_sq = (np.asarray(trials_since_block)-0.5)**2
            choice_trials_sq = choices*trial_sq
            interaction_trials_choice = trials_since_block*choices

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
                                              ('Reward Choice Int', reward_choice),
                                              ('Trials in Block', trials_since_block),
                                              ('Squared Time in Block', trial_sq),
                                              ('Trials x Choice', interaction_trials_choice),
                                              ('Trials x Choice Sq',choice_trials_sq),
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
                    # shuffle the start of all blocks
                    starts = []
                    for l in block_lengths:
                        start = np.random.randint(trials-l)
                        start_list = np.arange(start,start+l)
                        starts.append(start_list)
                        
                    ind = list(flatten(starts))
                    X_perm = X[ind,:]
                    tstats = reg_f.regression_code(y, X_perm)
        
                    C_perm[i].append(tstats.reshape(n_predictors,n_neurons,n_timepoints))  # Predictor loadings
                    cpd_perm[i].append(re._CPD(X_perm,y).reshape(n_neurons, n_timepoints, n_predictors))

     
    cpd = np.nanmean(np.concatenate(cpd,0), axis = 0)
    C = np.concatenate(C,1)
    
    if perm: # Evaluate P values.
        cpd_perm = np.stack([np.mean(np.concatenate(cpd_i,0),0) for cpd_i in cpd_perm],0)
        C_perm   = np.stack([np.concatenate(C_i,1) for C_i in C_perm],1)
        cpd_p_value  = np.mean(cpd_perm > cpd,0)
        p = np.percentile(cpd_perm,99, axis = 0)

    
    return C, cpd, predictors_all, cpd_perm,C_perm, cpd_p_value, p



def plot(data_PFC,experiment_aligned_PFC, data_HP,experiment_aligned_HP):
    
    n = 100
    
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(data_HP,experiment_aligned_HP,perm = n,trial_hypothesis = True, time_hypothesis = False, place_cells = False)
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
    
    
    C, cpd, predictors, cpd_perm,C_perm, cpd_p_value, p = regression_time_of_trial(data_PFC,experiment_aligned_PFC,perm = n,trial_hypothesis = True, time_hypothesis = False, place_cells = False)
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
    