#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:56:53 2020

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
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import geom
import scipy.stats as stt

def load_organised():
   data_HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
   data_PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
   design_matrix_HP = data_HP['DM'][0]
   firing_rates_HP = data_HP['Data'][0]
    
   design_matrix_PFC = data_PFC['DM'][0]
   firing_rates_PFC = data_PFC['Data'][0]
    
   time_ms = [0,   40,   80,  120,  160,  200,  240,  280,  320, 360,  400,  440,  480,  520,  560,  600,  640,  680,
               720,  760,  800,  840,  880,  920,  960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400,
               1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120,
               2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480]
    
   ind_init = 25
   ind_choice = 35
   ind_outcome = 42
    
   return design_matrix_HP, firing_rates_HP, design_matrix_PFC, firing_rates_PFC, time_ms, ind_init, ind_choice, ind_outcome



def separate_behaior_dat(dat,ix):

    #dat = design_matrix_PFC.copy()
    #Describe the design matrix
    desc = ['Latent_state',
           'Choices' ,
           'Outcomes',
           'forced_trials',
           'Block', 
           'Task',
           'A' ,
           'B' ,
           'I' ,
           'Chosen Q value' ,
           'Chosen cross-term Q value',
           'Value A' ,
           'Value B' ,
           #'Value A',
           'Constant']

    nTrials = dat[ix].shape[0]
    choices = dat[ix][:,desc.index('Choices')]  #choose A==1 choose B==0
    outcomes = dat[ix][:,desc.index('Outcomes')]
    block = dat[ix][:,desc.index('Block')]
    A = dat[ix][:,desc.index('A')]  #which port is A
    B = dat[ix][:,desc.index('B')]  #which port is B
    task = dat[ix][:,desc.index('Task')]  #which task is currently being performed
    I = dat[ix][:,desc.index('I')]  #Physical location of Initiation choice port
    forced_trials = dat[ix][:,desc.index('forced_trials')]  #Physical location of Initiation choice port
    lat_state = dat[ix][:,desc.index('Latent_state')] #1 means A is good 0 means B is good

    block_switches = np.array([i for i in np.concatenate([[0],1+np.where(block[1:] - block[:-1])[0]]) if ((nTrials-i)>20 and i>20)])  #first trial in each new block
    task_switches = np.concatenate([[0],1+np.where(task[1:] - task[:-1])[0]])

    mean_blockL = np.mean(block_switches[1:] - block_switches[:-1])
    min_blockL = np.mean(block_switches[1:] - block_switches[:-1])
    
    return choices,outcomes,block,task,forced_trials,lat_state,block_switches,task_switches,I


def Yves_code():
    design_matrix_HP, firing_rates_HP, design_matrix_PFC, firing_rates_PFC, time_ms, ind_init, ind_choice, ind_outcome = load_organised()
    ix = 1

    #%%timeit
  
    all_choices = []
    repP_store = [[] for _ in range(20)]  #repeat probability store
    repP_store_noR = [[] for _ in range(20)]  #repeat probability store
    changeP_store = [[] for _ in range(20)]
    changeP_store_noR = [[] for _ in range(20)]
    
    for dat in [design_matrix_HP,design_matrix_PFC]:
        for ix in range(len(dat)):
    
            #load data
            out_ = separate_behaior_dat(dat,ix)
            choices,outcomes,block,task,forced_trials,lat_state,block_switches,task_switches,I = out_
            all_choices.extend(choices)
            #data for individual session
            
            ctr = 0
            nRseq = 0  #number of sequential rewards on same side
            cSide = None
            for c,o in zip(choices[:-1],outcomes[:-1]):
                
                
                #if new task reset everything because then ports are 
                #different and who knows whats going on
                if ctr in task_switches:
                    #print(True)
                    nRseq = 0
                    cSide = None
                
                #if choose A and are rewarded
                if np.logical_and(c==1,o==1):
                    #if current side 
                    if cSide!=1:
                        if cSide==0:  #if previously went to B
                            changeP_store[nRseq].append(choices[ctr+1]==1) #what fraction of trials stay at A after nRseq rewards at B
                        nRseq = 0
                    cSide = 1
                    #repP_store[nRseq].append(choices[ctr+1]==1)
                    nRseq +=1
                    
                    
                elif np.logical_and(c==0,o==1):
                    if cSide!=0:
                        if cSide==1:
                            changeP_store[nRseq].append(choices[ctr+1]==0)
                        nRseq = 0
                    cSide = 0
                    repP_store[nRseq].append(choices[ctr+1]==0)
                    nRseq +=1
                
                else:
                    if cSide==1:
                        if c==1:
                            repP_store_noR[nRseq].append(choices[ctr+1]==1)
                        elif c==0:
                            changeP_store_noR[nRseq].append(choices[ctr+1]==1)
                    elif cSide==0:
                        if c==0:
                            repP_store_noR[nRseq].append(choices[ctr+1]==0)
                        elif c==1:
                            changeP_store_noR[nRseq].append(choices[ctr+1]==0)
                        
                    nRseq=0
                    cSide = None
    
                ctr +=1
                
    #This plot shows what happens, as a function of the number of rewards received on a given side (x1),
    #the probability of staying at x1 on trial n+1, if reward was received on trial n.
    #Prediction of a timing based learning stategy would be that after more rewards, this should go down
    MINN = 5
    ci = np.array([proportion_confint(np.sum(i),len(i),) for i in repP_store if len(i)>MINN])
    mu = np.array([np.mean(i) for i in repP_store if len(i)>MINN])
    x__ = np.arange(1,len(ci)+1)
    
    linreg = stt.linregress(x__,mu)
    plt.plot(x__,x__*linreg.slope + linreg.intercept,color='k',linewidth=3)
    plt.fill_between(x__,x__*(linreg.slope-linreg.stderr) + linreg.intercept,
                         x__*(linreg.slope+linreg.stderr) + linreg.intercept,
                     color='k',alpha=.1)
    
    plt.errorbar(x__,mu,yerr=np.abs(ci-mu[:,None]).T,marker='o',linewidth=0,elinewidth=4,markersize=13)
    plt.annotate('slope={:.3f} \np={:.3e}    r={:.3f}'.format(linreg.slope,linreg.pvalue,linreg.rvalue), (.1,.2),
                 xycoords='axes fraction',fontsize=18)
    plt.ylabel("Probability of repeating \nrew. choice on trial n+1")
    plt.xlabel("Number of successive rewards \non same side up to trial n")
    sns.despine()
    plt.locator_params(nbins=3)
    plt.ylim(0,1)
    print(stt.linregress(x__,mu))