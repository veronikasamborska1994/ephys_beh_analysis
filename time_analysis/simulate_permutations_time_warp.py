#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:20:14 2020

@author: veronikasamborska
"""
import pywt
from scipy.fftpack import rfft, irfft
import time_hierarchies as th
import numpy as np
from palettable import wesanderson as wes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def run():  
    
    ind_above_chance_HP,percentage_HP = plot_wavelet(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20,\
                        region = 'HP shuffle wavelet', plot = True, perm = 1000)
   
    ind_above_chance_PFC,percentage_PFC  = plot_wavelet(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20,\
                        region = 'PFC shuffle wavelet', plot = True, perm = 1000)
    
        
def perm_test_time_wavelet(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, perm = True):
    
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  th.hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
  
    distance_mean_neuron = []
    perm_mean = []
    neurons = 0
    for i,aa in enumerate(a_a_matrix_t_1_list):
        switch = int(a_a_matrix_t_1_list.shape[1]/2)

        neurons+=1
        
        a_1 = a_a_matrix_t_1_list[i][:switch] 
        a_2 = a_a_matrix_t_1_list[i][switch:]
        a_3 = a_a_matrix_t_2_list[i][:switch]
        a_4 = a_a_matrix_t_2_list[i][switch:]
        a_5 = a_a_matrix_t_3_list[i][:switch]
        a_6 = a_a_matrix_t_3_list[i][switch:]
      
        b_1 = b_b_matrix_t_1_list[i][:switch]
        b_2 = b_b_matrix_t_1_list[i][switch:] 
        b_3 = b_b_matrix_t_2_list[i][:switch]
        b_4 = a_a_matrix_t_2_list[i][switch:] 
        b_5 = b_b_matrix_t_3_list[i][:switch]
        b_6 = b_b_matrix_t_3_list[i][switch:] 
      
        if raw_data == True:
                
            a_1 = (a_1 - np.mean(a_1))
            a_2 = (a_2 - np.mean(a_2))
            a_3 = (a_3 - np.mean(a_3))
            a_4 = (a_4 - np.mean(a_4))
            a_5 = (a_5 - np.mean(a_5))
            a_6 = (a_6 - np.mean(a_6))

            b_1 = (b_1 - np.mean(b_1))
            b_2 = (b_2 - np.mean(b_2))
            b_3 = (b_3 - np.mean(b_3))
            b_4 = (b_4 - np.mean(b_4))
            b_5 = (b_5 - np.mean(b_5))
            b_6 = (b_6 - np.mean(b_6))

        switch = len(a_1)
        blocks =  [a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6]

        blocks_all_tasks =  np.mean([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)
        peak = np.max(blocks_all_tasks)
        troph = np.min(blocks_all_tasks)

        std_blocks_all_tasks =  np.std([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)/(np.sqrt(12))

        
        distance = abs(peak-troph)/np.max(std_blocks_all_tasks)

        distance_mean_neuron.append(distance)
    
        diff_perm = []
        
        grand_bud = wes.GrandBudapest1_4.mpl_colors
        grand_bud_1 = wes.GrandBudapest2_4.mpl_colors
        mend = wes.Mendl_4.mpl_colors
        cs =  [grand_bud,grand_bud_1,mend]
        if perm:
            for p in range(perm):
                shuffle_list = []
                for b in blocks:
                    cA, cD = pywt.dwt(b, 'db1')
                    np.random.shuffle(cD)
                    np.random.shuffle(cA)
                    shuffle = pywt.idwt(cA,cD, 'db1')
                    shuffle_list.append(shuffle)
                
                blocks_all_tasks_perm =  np.mean(shuffle_list,0)
                std_blocks_all_tasks_perm =  np.std(shuffle_list,0)/np.sqrt(12)


                peak_perm = np.max(blocks_all_tasks_perm)
                troph_perm = np.min(blocks_all_tasks_perm)
                
                distance_perm = abs(peak_perm-troph_perm)/np.max(std_blocks_all_tasks_perm)

                diff_perm.append(distance_perm)
                    
        perm_mean.append(np.percentile(diff_perm,95))
   
    # grand_bud = wes.GrandBudapest1_4.mpl_colors
    # plt.ion()
    # plt.figure()
    
    # his_perm, b_p = np.histogram(perm_mean,380)
    # hist_mean_neuron,b = np.histogram(distance_mean_neuron,380)
   
    # plt.bar(b[:-1], hist_mean_neuron, width = 0.05, color = grand_bud[0],alpha = 0.5, label = 'Neurons')
    # plt.bar(b_p[:-1], his_perm, width = 0.05, color = grand_bud[1], alpha = 0.5, label = 'Permutation')
    # plt.legend()
    ind_above_chance = np.where(np.array(distance_mean_neuron) > np.array(perm_mean))[0]
    percentage = (len(ind_above_chance)/neurons)*100
  
    # plt.legend()
    # if HP == True:
    #     plt.title('HP')
    # elif HP == False:
    #     plt.title('PFC')
    return ind_above_chance,percentage

def plot_wavelet(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True,start = 0, end = 20,\
                        region = 'HP', plot = True, perm = 5):

     a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  th.hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
   
    
     if HP == True:
         a_list,  b_list, rew_list,  no_rew_list = th.find_rewards_choices(data_HP, experiment_aligned_HP)
         ind_above_chance,percentage = perm_test_time_wavelet(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)
        
     elif HP == False:
        a_list,  b_list, rew_list,  no_rew_list = th.find_rewards_choices(data_PFC, experiment_aligned_PFC)
        ind_above_chance,percentage = perm_test_time_wavelet(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)

     switch = int(a_a_matrix_t_1_list.shape[1]/2)
     
     a_s_1_1 = a_a_matrix_t_1_list[:,:switch] 
     a_s_1_2 = a_a_matrix_t_1_list[:,switch:]
     b_s_1_1 = b_b_matrix_t_1_list[:,:switch]
     b_s_1_2 = b_b_matrix_t_1_list[:,switch:]
     
   
     a_s_2_1 = a_a_matrix_t_2_list[:,:switch]
     a_s_2_2 = a_a_matrix_t_2_list[:,switch:]
     b_s_2_1 = b_b_matrix_t_2_list[:,:switch]
     b_s_2_2 = b_b_matrix_t_2_list[:,switch:]
  
     a_s_3_1 = a_a_matrix_t_3_list[:,:switch]
     a_s_3_2 = a_a_matrix_t_3_list[:,switch:]
     b_s_3_1 = b_b_matrix_t_3_list[:,:switch]
     b_s_3_2 = b_b_matrix_t_3_list[:,switch:]
     
     isl_1 =  wes.Moonrise1_5.mpl_colors
     isl  = wes.Royal3_5.mpl_colors
     
     pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ region +'A vs B block 1 vs 2.pdf')
     plt.ioff()
     count = 0
     plot_new = True
     switch = int(a_a_matrix_t_1_list.shape[1]/2)
     neuron_count = 0
     plt.figure()
     for i,m in enumerate(a_a_matrix_t_1_list): 
        count +=1
        neuron_count += 1

        if count == 7:
            plot_new = True
            count = 1
        if plot_new == True:
            pdf.savefig()      
            plt.clf()
            plt.figure()
            plot_new = False
           
        plt.subplot(3,4,count)

        if raw_data == True:
                a_1 = (a_s_1_1[i]- np.mean(a_s_1_1[i]))
                a_2 = (a_s_1_2[i]- np.mean(a_s_1_2[i]))
                a_3 = (a_s_2_1[i]- np.mean(a_s_2_1[i]))
                a_4 = (a_s_2_2[i]- np.mean(a_s_2_2[i]))
                a_5 = (a_s_3_1[i]- np.mean(a_s_3_1[i]))
                a_6 = (a_s_3_2[i]- np.mean(a_s_3_2[i]))
   
                b_1 = (b_s_1_1[i]- np.mean(b_s_1_1[i]))
                b_2 = (b_s_1_2[i]- np.mean(b_s_1_2[i]))
                b_3 = (b_s_2_1[i]- np.mean(b_s_2_1[i]))
                b_4 = (b_s_2_2[i]- np.mean(b_s_2_2[i]))
                b_5 = (b_s_3_1[i]- np.mean(b_s_3_1[i]))
                b_6 = (b_s_3_2[i]- np.mean(b_s_3_2[i]))
               
            
        else:
               
            a_1 = a_s_1_1[i]#- np.mean(a_s_1_1[i])
            a_2 = a_s_1_2[i]#- np.mean(a_s_1_2[i])
            a_3 = a_s_2_1[i]#- np.mean(a_s_2_1[i])
            a_4 = a_s_2_2[i]#- np.mean(a_s_2_2[i])
            a_5 = a_s_3_1[i]#- np.mean(a_s_3_1[i])
            a_6 = a_s_3_2[i]#- np.mean(a_s_3_2[i])
   
            b_1 = b_s_1_1[i]#- np.mean(b_s_1_1[i])
            b_2 = b_s_1_2[i]#- np.mean(b_s_1_2[i])
            b_3 = b_s_2_1[i]#- np.mean(b_s_2_1[i])
            b_4 = b_s_2_2[i]#- np.mean(b_s_2_2[i])
            b_5 = b_s_3_1[i]#- np.mean(b_s_3_1[i])
            b_6 = b_s_3_2[i]#- np.mean(b_s_3_2[i])

          
          
        after = [a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6]
        blocks_all_tasls =  np.mean(after,0)
        std_blocks_all_tasls =  np.std([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)/(np.sqrt(12))
        plt.plot(blocks_all_tasls, color = isl_1[0], label = 'All Tasks Block Time')
        plt.fill_between(np.arange(len(blocks_all_tasls)), blocks_all_tasls-std_blocks_all_tasls, blocks_all_tasls+std_blocks_all_tasls, alpha=0.2, color = isl_1[0])
        plt.tight_layout()
       
        if count == 1:
            plt.legend()
        if  (neuron_count-1) in ind_above_chance:
            plt.title('Significant')
        else:
            plt.title(str(count))


        plt.subplot(3,4, count+6)
        plt.plot(a_list[i], color = isl[1], label = 'A')
        plt.plot(b_list[i], color = isl[2], label = 'B')
        plt.plot(rew_list[i], color = isl[3], label = 'Reward')
        plt.plot(no_rew_list[i], color = isl[4], label = 'No Rew')
        if  (neuron_count-1) in ind_above_chance:
            plt.title('Significant')
        else:
            plt.title(str(count))
        plt.vlines([25,36,43], np.min([np.min(a_list[i]),np.min(b_list[i]),np.min(rew_list[i]),np.min(no_rew_list[i])]),\
                                   np.max([np.max(a_list[i]),np.max(b_list[i]),np.max(rew_list[i]),np.max(no_rew_list[i])]),linestyle= '--', color = 'pink')

      
        if count == 1:
            plt.legend()
      
     pdf.savefig()      
     pdf.close()
         
     return ind_above_chance,percentage
         
                
     
                     
             
