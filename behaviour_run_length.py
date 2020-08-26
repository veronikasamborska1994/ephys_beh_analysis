#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:19:21 2020

@author: veronikasamborska
"""

from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import sys 
from matplotlib.cbook import flatten
#import utility as ut
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)
sys.path.append( '/Users/veronikasamborska/Desktop/Veronika Backup/2018-12-12-Reversal_learning/code/reversal_learning/')
import data_import as di

#exp = di.Experiment('/Users/veronikasamborska/Desktop/Veronika Backup/2018-12-12-Reversal_learning/data_pilot3')

def runs_length(experiment, subject_IDs ='all'):
    
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        
    run_length_list_task_subj = []
    run_length_list_correct_task_subj = []
    run_length_list_incorrect_task_subj = []
    
    
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
        run_length_list_task = []
        run_length_correct_list_task = []
        run_length_incorrect_list_task = []
        previous_session_config = 25
        run_length_list = []
        
        for j, session in enumerate(subject_sessions):
            choices = session.trial_data['choices']
            state = session.trial_data['state']
            correct = np.where(state == choices)[0]
            incorrect = np.where(state != choices)[0]
            configuration = session.trial_data['configuration_i'] 
            runs_list = []
            runs_list.append(0)
            runs_list_corr = []
            runs_list_incorr = []
            runs_list_incorr_after_correct = []
            block = session.trial_data['block']
            state_ch = np.where(np.diff(block)!=0)[0]+1
    
           
            run = 0
            for c, ch in enumerate(choices):
                if c > 0:
                    if choices[c] == choices[c-1]:
                        run += 1
                    elif choices[c] != choices[c-1]:
                        run = 0
                    runs_list.append(run)
            corr_run = 0
            run_ind_c =[]
            for c, ch in enumerate(choices):
                if c > 0  and c in correct:
                    if choices[c] == choices[c-1]:
                        if corr_run == 0:
                            run_ind_c.append(c)
                        corr_run +=1
                    elif choices[c] != choices[c-1]:
                        corr_run = 0
                else:
                    corr_run = 0
                runs_list_corr.append(corr_run)
             
            incorr_run = 0
            run_ind_inc = []
            for c, ch in enumerate(choices):
                if c > 0  and c in incorrect:
                    if choices[c] == choices[c-1]:
                        if incorr_run ==0:
                            run_ind_inc.append(c)
                        incorr_run +=1
                    elif choices[c] != choices[c-1]:
                        incorr_run = 0
                else:
                    incorr_run = 0
                    
                runs_list_incorr.append(incorr_run)
                
            inc = []
            co =[]
            for st in state_ch:
                inc_ind = [i for i in run_ind_inc if i > st]
                if len(inc_ind) > 0:
                    inc.append(min(inc_ind))
                co_ind = [i for i in run_ind_c if i > st]
                if len(co_ind) > 0:
                    co.append(min(co_ind))
                    
            if len(np.asarray(co)) == len(np.asarray(inc)):
                index_which_incorr_ignore = np.asarray(co) > np.asarray(inc)    
            elif len(np.asarray(co)) > len(np.asarray(inc)):
                index_which_incorr_ignore = np.asarray(co)[:len(np.asarray(inc)  )] > np.asarray(inc)    
              
            if len(inc)> len(index_which_incorr_ignore):
                inc = inc[:len(index_which_incorr_ignore)]
            elif len(index_which_incorr_ignore)> len(inc):
                index_which_incorr_ignore = index_which_incorr_ignore[:len(inc)]
       
            starts_to_ignore = np.asarray(inc)[index_which_incorr_ignore]
            all_ends = np.where(np.diff(runs_list_incorr) < 0)[0]
            ends_to_ignore =[]
            runs_list_incorr = np.asarray(runs_list_incorr)
            for st in starts_to_ignore:
                ends = [i for i in all_ends if i > st]
                if len(ends)>0:
                    ends_to_ignore.append(min(ends))
             
            for i, ii in enumerate(starts_to_ignore):
                if len(starts_to_ignore) == len(ends_to_ignore):
                    runs_list_incorr[starts_to_ignore[i]: ends_to_ignore[i]] = 0
                else:
                    runs_list_incorr[starts_to_ignore[i]:] = 0
                   
                
            
            if j == 0:
               previous_session_config = configuration[0]
       
            elif configuration[0]!= previous_session_config:
                # run_length_list_task.append(np.mean(list(flatten(run_length_list))))
                run_length_list_task.append((list(flatten(runs_list))))

                # run_length_correct_list_task.append(np.mean(list(flatten(corr_list))))
                run_length_correct_list_task.append((list(flatten(runs_list_corr))))

                # run_length_incorrect_list_task.append(np.mean(list(flatten(incorr_list))))
                run_length_incorrect_list_task.append((list(flatten(runs_list_incorr))))

                previous_session_config = configuration[0]  
               
                
        run_length_list_task_subj.append(run_length_list_task)
        run_length_list_correct_task_subj.append(run_length_correct_list_task)
        run_length_list_incorrect_task_subj.append(run_length_incorrect_list_task)
    
    _1 = [];  _2 = []; _3 = []; _4 = []; _5 = []; _6 = []; _7 = []; _8 = []; _9 = []
    _1_c = [];  _2_c = []; _3_c = []; _4_c = []; _5_c = []; _6_c = []; _7_c = []; _8_c = []; _9_c = []
    _1_i = [];  _2_i  = []; _3_i  = []; _4_i  = []; _5_i  = []; _6_i  = []; _7_i  = []; _8_i  = []; _9_i  = []

    for s,subj in enumerate(run_length_list_task_subj):
        _1.append(run_length_list_task_subj[s][0]);  _1_c.append(run_length_list_correct_task_subj[s][0]);  _1_i.append(run_length_list_incorrect_task_subj[s][0])
        _2.append(run_length_list_task_subj[s][1]);  _2_c.append(run_length_list_correct_task_subj[s][1]);  _2_i.append(run_length_list_incorrect_task_subj[s][1])
        _3.append(run_length_list_task_subj[s][2]);  _3_c.append(run_length_list_correct_task_subj[s][2]);  _3_i.append(run_length_list_incorrect_task_subj[s][2])
        _4.append(run_length_list_task_subj[s][3]);  _4_c.append(run_length_list_correct_task_subj[s][3]);  _4_i.append(run_length_list_incorrect_task_subj[s][3])
        _5.append(run_length_list_task_subj[s][4]);  _5_c.append(run_length_list_correct_task_subj[s][4]);  _5_i.append(run_length_list_incorrect_task_subj[s][4])
        _6.append(run_length_list_task_subj[s][5]);  _6_c.append(run_length_list_correct_task_subj[s][5]);  _6_i.append(run_length_list_incorrect_task_subj[s][5])
        _7.append(run_length_list_task_subj[s][6]);  _7_c.append(run_length_list_correct_task_subj[s][6]);  _7_i.append(run_length_list_incorrect_task_subj[s][6])
        _8.append(run_length_list_task_subj[s][7]);  _8_c.append(run_length_list_correct_task_subj[s][7]);  _8_i.append(run_length_list_incorrect_task_subj[s][7])
        _9.append(run_length_list_task_subj[s][8]);  _9_c.append(run_length_list_correct_task_subj[s][8]);  _9_i.append(run_length_list_incorrect_task_subj[s][8])
        # _10.append(run_length_list_task_subj[s][9]);  _10_c.append(run_length_list_correct_task_subj[s][9]);  _10_i.append(run_length_list_incorrect_task_subj[s][9])
   
    all_runs = np.vstack((_1, _2 , _3, _4, _5, _6,  _7, _8, _9))
    corr_runs = np.vstack((_1_c, _2_c , _3_c, _4_c, _5_c, _6_c,  _7_c, _8_c, _9_c))
    incorr_runs = np.vstack((_1_i, _2_i , _3_i, _4_i, _5_i, _6_i,  _7_i, _8_i, _9_i))

    plt.figure(figsize = (10,5))
    for i in range(9):
        plt.subplot(2,9,i+1)
        all_hist = np.asarray(list(flatten(all_runs[i])))
        all_hist = all_hist[np.nonzero(all_hist)]
        plt.hist(all_hist,10, color = 'grey', label = 'all')
        #plt.xlim(0,50)

        plt.subplot(2,9,i+1+9)
        inc_hist = np.asarray(list(flatten(incorr_runs[i])))
        incorr_all_hist = inc_hist[np.nonzero(inc_hist)]

        #plt.hist((list(flatten(corr_runs[i]))),10,  color = 'black', label = 'correct')
        plt.hist(incorr_all_hist,10, color = 'green', label = 'incorrect')
        #plt.xlim(0,50)
    #plt.tight_layout()
    plt.legend()
    sns.despine()

def runs_mean_bloc_training(experiment, subject_IDs ='all'):
    
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        
    run_length_list_correct_task_subj = []
    run_length_list_incorrect_task_subj = []
    
    #configuration_subj = []

    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
        run_length_correct_list_task = []
        run_length_incorrect_list_task = []
        run_length_correct_fraction_s = []
        run_length_incorrect_fraction_s = []
        for j, session in enumerate(subject_sessions):
            configuration = session.trial_data['configuration_i'] 

            if j == 0:
                previous_session_config = configuration[0] # Initiate previous configuration

            choices = session.trial_data['choices']
            state = session.trial_data['state']
            correct = np.where(state == choices)[0]
            incorrect = np.where(state != choices)[0]
            configuration = session.trial_data['configuration_i'] 
            runs_list = []
            runs_list.append(0)
            runs_list_corr = []
            runs_list_incorr = []
            runs_list_incorr_after_correct = []
            block = session.trial_data['block']
            state_ch = np.where(np.diff(block)!=0)[0]+1
    
           
            run = 0
            for c, ch in enumerate(choices):
                if c > 0:
                    if choices[c] == choices[c-1]:
                        run += 1
                    elif choices[c] != choices[c-1]:
                        run = 0
                    runs_list.append(run)
            corr_run = 0
            run_ind_c =[]
            for c, ch in enumerate(choices):
                if c > 0  and c in correct:
                    if choices[c] == choices[c-1]:
                        if corr_run == 0:
                            run_ind_c.append(c)
                        corr_run +=1
                    elif choices[c] != choices[c-1]:
                        corr_run = 0
                else:
                    corr_run = 0
                runs_list_corr.append(corr_run)
             
            incorr_run = 0
            run_ind_inc = []
            for c, ch in enumerate(choices):
                if c > 0  and c in incorrect:
                    if choices[c] == choices[c-1]:
                        if incorr_run ==0:
                            run_ind_inc.append(c)
                        incorr_run +=1
                    elif choices[c] != choices[c-1]:
                        incorr_run = 0
                else:
                    incorr_run = 0
                    
                runs_list_incorr.append(incorr_run)
                
            inc = []
            co =[]
            for st in state_ch:
                inc_ind = [i for i in run_ind_inc if i > st]
                if len(inc_ind) > 0:
                    inc.append(min(inc_ind))
                co_ind = [i for i in run_ind_c if i > st]
                if len(co_ind) > 0:
                    co.append(min(co_ind))
                    
            if len(np.asarray(co)) == len(np.asarray(inc)):
                index_which_incorr_ignore = np.asarray(co) > np.asarray(inc)    
            elif len(np.asarray(co)) > len(np.asarray(inc)):
                index_which_incorr_ignore = np.asarray(co)[:len(np.asarray(inc)  )] > np.asarray(inc)    
              
            if len(inc)> len(index_which_incorr_ignore):
                inc = inc[:len(index_which_incorr_ignore)]
            elif len(index_which_incorr_ignore)> len(inc):
                index_which_incorr_ignore = index_which_incorr_ignore[:len(inc)]
       
            starts_to_ignore = np.asarray(inc)[index_which_incorr_ignore]
            all_ends = np.where(np.diff(runs_list_incorr) < 0)[0]
            ends_to_ignore =[]
            runs_list_incorr = np.asarray(runs_list_incorr)
            for st in starts_to_ignore:
                ends = [i for i in all_ends if i > st]
                if len(ends)>0:
                    ends_to_ignore.append(min(ends))
             
            for i, ii in enumerate(starts_to_ignore):
                if len(starts_to_ignore) == len(ends_to_ignore):
                    runs_list_incorr[starts_to_ignore[i]: ends_to_ignore[i]] = 0
                else:
                    runs_list_incorr[starts_to_ignore[i]:] = 0
            runs_list_corr = np.asarray(runs_list_corr)     
            runs_list_corr = runs_list_corr[np.where(runs_list_corr!=0)]
            runs_list_incorr = runs_list_incorr[np.where(runs_list_incorr!=0)]

            run_length_correct_fraction_s.append(np.mean(runs_list_corr))
            run_length_incorrect_fraction_s.append(np.mean(runs_list_incorr))
         
            if configuration[0]!= previous_session_config:
                    
                
                  # run_length_list_task.append(np.mean(list(flatten(run_length_list))))
                   
                #print(len(run_length_incorrect_fraction))
                
                run_length_incorrect_list_task.append(np.nanmean(run_length_incorrect_fraction_s,0))
                #print(len(run_length_incorrect_fraction))

                    # run_length_correct_list_task.append(np.mean(list(flatten(corr_list))))
                run_length_correct_list_task.append(np.nanmean(run_length_correct_fraction_s,0))
    

                run_length_correct_fraction_s = []
                run_length_incorrect_fraction_s = []
                
                previous_session_config = configuration[0]  
 
        run_length_list_correct_task_subj.append(np.asarray(run_length_correct_list_task))
        run_length_list_incorrect_task_subj.append(np.asarray(run_length_incorrect_list_task))
    
    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    run_length_list_correct_task_mean_sub = np.nanmean(np.asarray(run_length_list_correct_task_subj),0)
    run_length_list_correct_task_std_sub = np.nanstd(np.asarray(run_length_list_correct_task_subj),0)/np.sqrt(9)
    plt.fill_between(np.arange(len(run_length_list_correct_task_mean_sub)),\
                         run_length_list_correct_task_mean_sub+run_length_list_correct_task_std_sub ,run_length_list_correct_task_mean_sub-run_length_list_correct_task_std_sub, alpha = 0.3,color = 'pink')
   
    plt.plot(run_length_list_correct_task_mean_sub, color = 'pink')
    plt.xlabel('Task #')
    plt.ylabel('Mean length of correct runs')

    plt.subplot(1,2,2)

    run_length_list_incorrect_task_mean_sub = np.mean(np.asarray(run_length_list_incorrect_task_subj),0)
    run_length_list_incorrect_task_std_sub = np.std(np.asarray(run_length_list_incorrect_task_subj),0)/np.sqrt(9)
    plt.fill_between(np.arange(len(run_length_list_incorrect_task_mean_sub)),\
                         run_length_list_incorrect_task_mean_sub+run_length_list_incorrect_task_std_sub ,run_length_list_incorrect_task_mean_sub-run_length_list_incorrect_task_std_sub, alpha = 0.3,color = 'cyan')
    plt.ylabel('Mean length of incorrect runs')
    plt.xlabel('Task #')

    plt.plot(run_length_list_incorrect_task_mean_sub, color = 'cyan')
    sns.despine()
    
  
     
def runs_function_bloc_training(experiment, subject_IDs ='all'):
    
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        
    run_length_list_correct_task_subj = []
    run_length_list_incorrect_task_subj = []
    
    #configuration_subj = []

    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
       # previous_session_config = 0 # Initiate previous configuration

        run_length_correct_list_task = []
        run_length_incorrect_list_task = []
        run_length_correct_fraction_s = []
        run_length_incorrect_fraction_s = []
        k  = 0

        for j, session in enumerate(subject_sessions):
            configuration = session.trial_data['configuration_i'] 

            if j == 0:
                previous_session_config = configuration[0]
  
            choices = session.trial_data['choices']
            state = session.trial_data['state']
            correct = choices&state
            incorrect = correct+1
            incorrect[np.where(incorrect == 2)] =0

            reward = session.trial_data['outcomes']
            block = session.trial_data['block'] 
            trials = session.trial_data['trials']
            state_ch = np.where(np.diff(block)!=0)[0]+1

            if len(np.unique(block))> 1:
                block = np.where(np.diff(block)!=0)[0]+1
                block_ind = np.insert(np.diff(block), 0,block[0])
                block_ind = np.insert(block_ind, len(block_ind), len(state)-block[-1])
        
                block_lengths= np.zeros(len(state))
                ind = 0
                for i in block_ind:
                   block_lengths[ind: ind+i] = (np.arange(1,i+1))
                   ind += i
                block_fraction_ind = np.insert(block,len(block),len(state))
                fractions = []
                block_l =  block_ind[0]
                j  = 0 
                for i,ii in enumerate(block_lengths):
                   if i in block_fraction_ind+1: 
                       j+=1
                       block_l = block_ind[j]
                       
                   fractions.append(ii/block_l)
                
            else:
                fractions = []

                for i,ii in enumerate(trials):
                   fractions.append(ii/len(trials))
            fractions = np.asarray(fractions)            
            #fractions = np.round(fractions,1)
            stay=choices[0:-1]==choices[1:]
            stay = stay*1
            stay = np.insert(stay,0,0)
            lastreward = reward[0:-1]
            lastreward = np.insert(lastreward,0,0)
    
    
            rl = np.zeros(len(stay))
            rl[0]=1
          
            rl_right = np.zeros(len(stay))
            rl_right[0]=choices[0]==state[0]
            choice_rr_start=-100
             
             
            rl_wrong=np.zeros(len(stay));
            rl_wrong[0]=choices[0]!=state[0];
            choice_rw_start=-100;
            
            for tr in range(len(stay)):
                if tr > 0: 
                    if stay[tr] == 1:
                        rl[tr] = rl[tr-1]+1
                    else:
                        rl[tr]=1
                    
                    
                    if ((choices[tr] == choice_rr_start) & (choices[tr]==state[tr])):
                        rl_right[tr]=rl_right[tr-1]+1
                        
                    elif (choices[tr]==state[tr]):
                        
                        rl_right[tr]=1;
                        choice_rr_start=choices[tr]
                    else:
                        rl_right[tr]=0;
                        choice_rr_start =-100; #If he made the wrong choice it can't be part of a correct run. 
                    
                    
                    if ((choices[tr]==choice_rw_start) & (choices[tr]!=state[tr])):
                        rl_wrong[tr]=rl_wrong[tr-1]+1
                        
                    elif choices[tr]!=state[tr]:
                        rl_wrong[tr]=1
                        choice_rw_start=choices[tr]
                    else:
                        rl_wrong[tr] = 0;
                        choice_rw_start=-100 #If he made the right choice it can't be part of a wrong run. 
            
            
           # uniq_fr =np.unique(fractions)
            run_length_incorrect_fraction = []
            run_length_correct_fraction = []
            #uniq_fr_end = [ 0.2, 0.4, 0.6, 0.8, 1]
            #uniq_fr_start = [0. , 0.2, 0.4, 0.6, 0.8]
            uniq_fr_end = [0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]
            uniq_fr_start = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
            for f,fr in enumerate(uniq_fr_end):
                ind_wr = np.where((fractions >= uniq_fr_start[f]) & (fractions < uniq_fr_end[f]))[0]
                #ind_wr = np.where((fractions == fr))[0]

                no_zeros_right = rl_right[ind_wr]
                #no_zeros_right = no_zeros_right[np.where(no_zeros_right!=0)]
                no_zeros_wrong = rl_wrong[ind_wr]
               #no_zeros_wrong = no_zeros_wrong[np.where(no_zeros_wrong!=0)]
              
                run_length_correct_fraction.append(np.nanmean(no_zeros_right))
                run_length_incorrect_fraction.append(np.nanmean(no_zeros_wrong))
                
            run_length_correct_fraction_s.append(run_length_correct_fraction)
            run_length_incorrect_fraction_s.append(run_length_incorrect_fraction)

            if configuration[0]!= previous_session_config:
                    
                
                  # run_length_list_task.append(np.mean(list(flatten(run_length_list))))
                   
                #print(len(run_length_incorrect_fraction))
                
                run_length_incorrect_list_task.append(np.nanmean(run_length_incorrect_fraction_s,0))
                #print(len(run_length_incorrect_fraction))

                    # run_length_correct_list_task.append(np.mean(list(flatten(corr_list))))
                run_length_correct_list_task.append(np.nanmean(run_length_correct_fraction_s,0))
    

                run_length_correct_fraction_s = []
                run_length_incorrect_fraction_s = []
                
                previous_session_config = configuration[0]  

        run_length_list_correct_task_subj.append(np.asarray(run_length_correct_list_task))
        run_length_list_incorrect_task_subj.append(np.asarray(run_length_incorrect_list_task))
        
    run_length_list_correct_task_mean_sub = np.nanmean(np.asarray(run_length_list_correct_task_subj),0)
    run_length_list_correct_task_std_sub = np.nanstd(np.asarray(run_length_list_correct_task_subj),0)/np.sqrt(9)
    
    plt.figure(figsize = (10,3))
    plt.subplot(1,2,1)
    for i,ii in enumerate(run_length_list_correct_task_mean_sub):
        plt.plot(np.arange(len(run_length_list_correct_task_mean_sub[0]))+i*(len(run_length_list_correct_task_mean_sub[0])),ii)
        plt.fill_between(np.arange(len(run_length_list_correct_task_mean_sub[0]))+i*(len(run_length_list_correct_task_mean_sub[0])),\
                         ii+run_length_list_correct_task_std_sub[i] ,ii-run_length_list_correct_task_std_sub[i], alpha = 0.3)

    run_length_list_incorrect_task_mean_sub = np.mean(np.asarray(run_length_list_incorrect_task_subj),0)
    run_length_list_incorrect_task_std_sub = np.std(np.asarray(run_length_list_incorrect_task_subj),0)/np.sqrt(9)

    plt.subplot(1,2,2)
    for i,ii in enumerate(run_length_list_incorrect_task_mean_sub):
        plt.plot(np.arange(len(run_length_list_incorrect_task_mean_sub[0]))+i*(len(run_length_list_incorrect_task_mean_sub[0])),ii)
        plt.fill_between(np.arange(len(run_length_list_incorrect_task_mean_sub[0]))+i*(len(run_length_list_incorrect_task_mean_sub[0])),\
                         ii+run_length_list_incorrect_task_std_sub[i] ,ii-run_length_list_incorrect_task_std_sub[i], alpha = 0.3)
    sns.despine()
  

def runs_as_function_of_block(data):
    
    dm = data['DM'][0]
    firing = data['Data'][0]
    neurons = 0
    session_run_incorrect = np.zeros((len(dm),10))
    session_run_correct  =  np.zeros((len(dm),10))
    for s in firing:
        neurons += s.shape[1]
    
    n_neurons_cum = 0
    for  s, sess in enumerate(dm):
        runs_list = []
        runs_list.append(0)
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons
        choices = DM[:,1]
        reward = DM[:,2]    
        state = DM[:,0]
        block = (np.where(np.diff(DM[:,4])!=0)[0])+1
        block_ind = np.insert(np.diff(block), 0,block[0])
        block_ind = np.insert(block_ind, len(block_ind), len(state)-block[-1])

        block_lengths= np.zeros(len(state))
        ind = 0
        for i in block_ind:
           block_lengths[ind: ind+i] = (np.arange(1,i+1))
           ind += i
        block_fraction_ind = np.insert(block,len(block),len(state))
        fractions = []
        block_l =  block_ind[0]
        j  = 0 
        for i,ii in enumerate(block_lengths):
           if i in block_fraction_ind+1: 
               j+=1
               block_l = block_ind[j]
               
           fractions.append(ii/block_l)
        
        
        #fractions = np.round(fractions,1)
        fractions = np.asarray(fractions)

        stay=choices[0:-1]==choices[1:]
        stay = stay*1
        stay = np.insert(stay,0,0)
        lastreward = reward[0:-1]
        lastreward = np.insert(lastreward,0,0)


        rl = np.zeros(len(stay))
        rl[0]=1
      
        rl_right = np.zeros(len(stay))
        rl_right[0]=choices[0]==state[0]
        choice_rr_start=-100
         
         
        rl_wrong=np.zeros(len(stay));
        rl_wrong[0]=choices[0]!=state[0];
        choice_rw_start=-100;
        
        for tr in range(len(stay)):
            if tr > 0: 
                if stay[tr] == 1:
                    rl[tr] = rl[tr-1]+1
                else:
                    rl[tr]=1
                
                
                if ((choices[tr] == choice_rr_start) & (choices[tr]==state[tr])):
                    rl_right[tr]=rl_right[tr-1]+1
                    
                elif (choices[tr]==state[tr]):
                    
                    rl_right[tr]=1;
                    choice_rr_start=choices[tr]
                else:
                    rl_right[tr]=0;
                    choice_rr_start =-100; #If he made the wrong choice it can't be part of a correct run. 
                
                
                if ((choices[tr]==choice_rw_start) & (choices[tr]!=state[tr])):
                    rl_wrong[tr]=rl_wrong[tr-1]+1
                    
                elif choices[tr]!=state[tr]:
                    rl_wrong[tr]=1
                    choice_rw_start=choices[tr]
                else:
                    rl_wrong[tr] = 0;
                    choice_rw_start=-100 #If he made the right choice it can't be part of a wrong run. 
        
        
        #uniq_fr =np.unique(fractions)
        run_length_incorrect_fraction = []
        run_length_correct_fraction = []

        uniq_fr_end = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
        uniq_fr_start = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
        #uniq_fr = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
        for f,fr in enumerate(uniq_fr_end):
            ind_wr = np.where((fractions >= uniq_fr_start[f]) & (fractions < uniq_fr_end[f]))[0]
            #ind_wr = np.where(fractions  == fr)[0]

            no_zeros_right = rl_right[ind_wr]
            #no_zeros_right = no_zeros_right[np.where(no_zeros_right!=0)]
            no_zeros_wrong = rl_wrong[ind_wr]
            #no_zeros_wrong = no_zeros_wrong[np.where(no_zeros_wrong!=0)]
          
            run_length_correct_fraction.append(np.mean(no_zeros_right))
            run_length_incorrect_fraction.append(np.mean(no_zeros_wrong))

        session_run_incorrect[s] = run_length_incorrect_fraction
        session_run_correct[s] = run_length_correct_fraction
    
    pal = sns.cubehelix_palette(8)
    pal_c = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95)

    plt.plot(np.mean(session_run_incorrect,0), color = pal[1],  label = 'Incorrect Run Length')
    plt.fill_between(np.arange(session_run_incorrect.shape[1]), \
                     np.mean(session_run_incorrect,0)+np.std(session_run_incorrect,0)/np.sqrt(session_run_incorrect.shape[0]),\
                     np.mean(session_run_incorrect,0)-np.std(session_run_incorrect,0)/np.sqrt(session_run_incorrect.shape[0]), color = pal[1], alpha = 0.7)
    plt.plot(np.mean(session_run_correct,0), color = pal_c[1], label = 'Correct Run Length')
    plt.fill_between(np.arange(session_run_correct.shape[1]), \
                     np.mean(session_run_correct,0)+np.std(session_run_correct,0)/np.sqrt(session_run_correct.shape[0]),\
                     np.mean(session_run_correct,0)-np.std(session_run_correct,0)/np.sqrt(session_run_correct.shape[0]), color = pal_c[1], alpha = 0.7)
  
    
    sns.despine()
    plt.xlabel('Fraciton of Block')
    plt.ylabel('Run Length')