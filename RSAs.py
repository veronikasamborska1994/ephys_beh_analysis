#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:19:56 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 4}

plt.rc('font', **font)
def RSA_physical_rdm():

    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:6] = True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:6] = True
    
    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0:6] = True
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[0:6] = True
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0:6] = True
    
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[0:6] = True
    
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_1[6:8] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_2[6:8] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    port_3_initiation_task_3[8:11] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[8:11] =  True
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[8:11] =  True
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[11:13] = True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[11:13] = True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[13:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[13:15] =  True
    
    
    physical_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
    plt.subplot(621)
    plt.imshow(physical_rsa)
    
    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
    
    plt.title('Physical Space')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    return physical_rsa

def RSA_a_b_initiation_rdm():

    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:6] = True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:6] = True
    
    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0:6] = True
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[0:6] = True
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0:6] = True
    
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[0:6] = True
    
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_1[6:9] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_2[6:9] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    port_3_initiation_task_3[6:9] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[9:15] =  True
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[9:15] =  True
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[9:15] = True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[9:15] = True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[9:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[9:15] =  True
    
    
    choice_ab_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
    
    plt.subplot(622)
    plt.imshow(choice_ab_rsa)
        
    plt.subplots_adjust(bottom=0.3)
    plt.title('Choice vs A/B')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
     
    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
   
    return choice_ab_rsa
   
def choice_initiation_no_space():        
    choice_ab_rsa = RSA_a_b_initiation_rdm()
    physical_rsa  =RSA_physical_rdm()
    # Choice/Initiation - Space 
    choice_init_nspace = choice_ab_rsa != physical_rsa 
    plt.subplot(623)

    plt.imshow(choice_init_nspace)
    
    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  
   
    plt.subplots_adjust(bottom=0.3)
    plt.title('Initiation/Choice Without Space')
    

def reward_rdm():  
    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0] = True
    port_a_choice_task_1_r[2] = True
    port_a_choice_task_1_r[4] = True
    
    port_a_choice_task_1_r[9] =  True
    port_a_choice_task_1_r[11] =  True
    port_a_choice_task_1_r[13] =  True
    

    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[1] = True
    port_a_choice_task_1_nr[3] = True
    port_a_choice_task_1_nr[5] = True
    
    port_a_choice_task_1_nr[10] =  True
    port_a_choice_task_1_nr[12] =  True
    port_a_choice_task_1_nr[14] =  True
       
    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0] = True
    port_a_choice_task_2_r[2] = True
    port_a_choice_task_2_r[4] = True   
     
    port_a_choice_task_2_r[9] =  True
    port_a_choice_task_2_r[11] =  True
    port_a_choice_task_2_r[13] =  True
    
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[1] = True
    port_a_choice_task_2_nr[3] = True
    port_a_choice_task_2_nr[5] = True
    
    port_a_choice_task_2_nr[10] =  True
    port_a_choice_task_2_nr[12] =  True
    port_a_choice_task_2_nr[14] =  True
      
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0] = True
    port_a_choice_task_3_r[2] = True
    port_a_choice_task_3_r[4] = True   
        
    port_a_choice_task_3_r[9] =  True
    port_a_choice_task_3_r[11] =  True
    port_a_choice_task_3_r[13] =  True
  
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[1] = True
    port_a_choice_task_3_nr[3] = True
    port_a_choice_task_3_nr[5] = True
        
    port_a_choice_task_3_nr[10] =  True
    port_a_choice_task_3_nr[12] =  True
    port_a_choice_task_3_nr[14] =  True
     
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[0] = True
    port_3_choice_task_2_r[2] = True
    port_3_choice_task_2_r[4] = True

    port_3_choice_task_2_r[9] =  True
    port_3_choice_task_2_r[11] =  True
    port_3_choice_task_2_r[13] =  True

    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_nr[1] = True
    port_3_choice_task_2_nr[3] = True
    port_3_choice_task_2_nr[5] = True

    port_3_choice_task_2_nr[10] =  True
    port_3_choice_task_2_nr[12] =  True
    port_3_choice_task_2_nr[14] =  True

    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[0] = True
    port_4_choice_task_1_r[2] = True
    port_4_choice_task_1_r[4] = True
    
    port_4_choice_task_1_r[9] =  True
    port_4_choice_task_1_r[11] =  True
    port_4_choice_task_1_r[13] =  True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[1] = True
    port_4_choice_task_1_nr[3] = True
    port_4_choice_task_1_nr[5] = True
    
    port_4_choice_task_1_nr[10] =  True
    port_4_choice_task_1_nr[12] =  True
    port_4_choice_task_1_nr[14] =  True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[0] = True
    port_5_choice_task_3_r[2] = True
    port_5_choice_task_3_r[4] = True
        
    port_5_choice_task_3_r[9] =  True
    port_5_choice_task_3_r[11] =  True
    port_5_choice_task_3_r[13] =  True
        
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[1] = True
    port_5_choice_task_3_nr[3] = True
    port_5_choice_task_3_nr[5] = True

    port_5_choice_task_3_nr[10] =  True
    port_5_choice_task_3_nr[12] =  True
    port_5_choice_task_3_nr[14] =  True
    
    reward_no_reward = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
    
    plt.subplot(624)

    plt.imshow(reward_no_reward)

    plt.subplots_adjust(bottom=0.1)
    plt.title('Reward & No Reward Space')
    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
   
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  
    return reward_no_reward

def reward_choice_space():
    reward_no_reward = reward_rdm()
    choice_ab_rsa = RSA_a_b_initiation_rdm()
    choice_initiation_no_space()
    reward_at_choices = reward_no_reward & choice_ab_rsa 
 
    plt.subplot(625)

    plt.imshow(reward_at_choices)
    
    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  
    plt.subplots_adjust(bottom=0.01)
    plt.title('Reward & No Reward Choice')
    
def remapping_a_to_b():
    
    reward_choice_space()
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:2] = True
    port_a_choice_task_1_r[11:15] =  True

    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:2] = True
    port_a_choice_task_1_nr[11:15] =  True

    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    #port_a_choice_task_2_r[0:6] = True
    
    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    #port_a_choice_task_2_nr[0:6] = True
    
    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    #port_a_choice_task_3_r[0:6] = True
    
    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    #port_a_choice_task_3_nr[0:6] = True
    
    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    #port_2_initiation_task_1[6:8] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    #port_2_initiation_task_2[6:8] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    #port_3_initiation_task_3[8:11] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[0:2] =  True
    port_4_choice_task_1_r[11:15] =  True

    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[0:2] =  True
    port_4_choice_task_1_nr[11:15] =  True

    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[0:2] =  True
    port_5_choice_task_3_r[11:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[0:2] =  True
    port_5_choice_task_3_nr[11:15] =  True
        
    remapping_a_to_b = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
   
    plt.subplot(626)

    plt.imshow(remapping_a_to_b)
    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
  
    plt.title('Remapping A to B')

def choice_vs_initiation():
    remapping_a_to_b()
    
    # RSA Physical Space coding
    
    port_a_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_r[0:6] = True
    port_a_choice_task_1_r[9:15] =  True
    
    port_a_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_1_nr[0:6] = True
    port_a_choice_task_1_nr[9:15] =  True

    port_a_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_r[0:6] = True
    port_a_choice_task_2_r[9:15] =  True

    port_a_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_2_nr[0:6] = True
    port_a_choice_task_2_nr[9:15] =  True

    port_a_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_r[0:6] = True
    port_a_choice_task_3_r[9:15] =  True

    port_a_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_a_choice_task_3_nr[0:6] = True
    port_a_choice_task_3_nr[9:15] =  True

    port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_1[6:9] = True
    
    port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(15)])
    port_2_initiation_task_2[6:9] = True
    
    port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(15)])
    port_3_initiation_task_3[6:9] = True
    
    port_3_choice_task_2_r = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_r[0:6] = True
    port_3_choice_task_2_r[9:15] =  True
    
    port_3_choice_task_2_nr = np.array([bool(i & 0) for i in range(15)])
    port_3_choice_task_2_nr[0:6] = True
    port_3_choice_task_2_nr[9:15] =  True
    
    port_4_choice_task_1_r = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_r[0:6] = True
    port_4_choice_task_1_r[9:15] = True
    
    port_4_choice_task_1_nr = np.array([bool(i & 0) for i in range(15)])
    port_4_choice_task_1_nr[0:6] = True
    port_4_choice_task_1_nr[9:15] = True
    
    port_5_choice_task_3_r = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_r[0:6] = True
    port_5_choice_task_3_r[9:15] =  True
    
    port_5_choice_task_3_nr = np.array([bool(i & 0) for i in range(15)])
    port_5_choice_task_3_nr[0:6] = True
    port_5_choice_task_3_nr[9:15] =  True
    
    
    choice_initiation_rsa = np.vstack([port_a_choice_task_1_r,port_a_choice_task_1_nr,port_a_choice_task_2_r,port_a_choice_task_2_nr,\
                              port_a_choice_task_3_r,port_a_choice_task_3_nr,port_2_initiation_task_1,port_2_initiation_task_2,\
                              port_3_initiation_task_3,port_3_choice_task_2_r,port_3_choice_task_2_nr,port_4_choice_task_1_r,\
                              port_4_choice_task_1_nr,port_5_choice_task_3_r,port_5_choice_task_3_nr])
    
    plt.subplot(627)
    plt.imshow(choice_initiation_rsa)
        
    plt.subplots_adjust(bottom=0.3)
    plt.title('Choice Initiation Space')
    plt.xticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'), rotation='vertical')

    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))

    physical_rsa  = RSA_physical_rdm()
    
    # Choice/Initiation - Space 
    choice_initiation_no_nspace = choice_initiation_rsa != physical_rsa 
    plt.subplot(628)

    plt.imshow(choice_initiation_no_nspace)
    plt.xticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'), rotation='vertical')

    plt.yticks(range(15), ('1 Choice T1 R', '1 Choice T1 NR','1 Choice T2 R', '1 Choice T2 NR',\
               '1 Choice T3 R','1 T3 Choice NR', '2 Init T1',\
               '2 Init T2', '3 Init T3', '3 Choice T1 R',\
               '3 Choice T1 NR','4 Choice T2 R', '4 Choice T2 NR', '5 Choice T3 R', '5 Choice T3 NR'))
   
   
    plt.title('Initiation/Choice Without Space')
    plt.subplots_adjust(bottom=0.001)
    plt.subplots_adjust(wspace = 0.3)  # the amount of width reserved for space between subplots,
    plt.subplots_adjust(hspace = 0.3)   # the amount of height reserved for space between subplots,
    
    return choice_initiation_rsa, choice_initiation_no_nspace
