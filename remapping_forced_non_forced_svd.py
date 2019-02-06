#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:48:33 2019

@author: veronikasamborska
"""

import svd_forced_trials as svf 
import svd_block_analysis as svnf
import numpy as np
import matplotlib.pyplot as plt

def remapping_forced_non_forced(experiment, HP = True):
    
    # Extracts firing rates based on blocks on forced trials 
    cluster_list_task_1_a_good_1, cluster_list_task_1_b_good_1,\
    cluster_list_task_2_a_good_1, cluster_list_task_2_b_good_1,\
    cluster_list_task_3_a_good_1, cluster_list_task_3_b_good_1, cluster_list_task_1_a_good_2,\
    cluster_list_task_1_b_good_2, cluster_list_task_2_a_good_2, cluster_list_task_2_b_good_2,\
    cluster_list_task_3_a_good_2, cluster_list_task_3_b_good_2  = svf.block_firings_rates_selection_forced_split_in_half(experiment_aligned_PFC)


    cluster_list_task_1_a_good_1_nf, cluster_list_task_1_b_good_1_nf,\
    cluster_list_task_2_a_good_1_nf, cluster_list_task_2_b_good_1_nf,\
    cluster_list_task_3_a_good_1_nf, cluster_list_task_3_b_good_1_nf, cluster_list_task_1_a_good_2_nf,\
    cluster_list_task_1_b_good_2_nf, cluster_list_task_2_a_good_2_nf, cluster_list_task_2_b_good_2_nf,\
    cluster_list_task_3_a_good_2_nf, cluster_list_task_3_b_good_2_nf  = svnf.block_firings_rates_selection(experiment_aligned_PFC)
    
    
    #A good task 1
    u_t1_a_good_1, s_t1_a_good_1, vh_t1_a_good_1 = np.linalg.svd(cluster_list_task_1_a_good_1, full_matrices = False)
    u_t1_a_good_2, s_t1_a_good_2, vh_t1_a_good_2 = np.linalg.svd(cluster_list_task_1_a_good_2, full_matrices = False)
    
    #B good task 1
    u_t1_b_good_1, s_t1_b_good_1, vh_t1_b_good_1 = np.linalg.svd(cluster_list_task_1_b_good_1, full_matrices = False)    
    u_t1_b_good_2, s_t1_b_good_2, vh_t1_b_good_2 = np.linalg.svd(cluster_list_task_1_b_good_2, full_matrices = False)
    
    #A good task 2
    u_t2_a_good_1, s_t2_a_good_1, vh_t2_a_good_1 = np.linalg.svd(cluster_list_task_2_a_good_1, full_matrices = False)
    u_t2_a_good_2, s_t2_a_good_2, vh_t2_a_good_2 = np.linalg.svd(cluster_list_task_2_a_good_2, full_matrices = False)
    
    #B good task 2
    u_t2_b_good_1, s_t2_b_good_1, vh_t2_b_good_1 = np.linalg.svd(cluster_list_task_2_b_good_1, full_matrices = False)    
    u_t2_b_good_2, s_t2_b_good_2, vh_t2_b_good_2 = np.linalg.svd(cluster_list_task_2_b_good_2, full_matrices = False)

    #A good task 3
    u_t3_a_good_1, s_t3_a_good_1, vh_t3_a_good_1 = np.linalg.svd(cluster_list_task_3_a_good_1, full_matrices = False)
    u_t3_a_good_2, s_t3_a_good_2, vh_t3_a_good_2 = np.linalg.svd(cluster_list_task_3_a_good_2, full_matrices = False)
    
    #B good task 3
    u_t3_b_good_1, s_t3_b_good_1, vh_t3_b_good_1 = np.linalg.svd(cluster_list_task_3_b_good_1, full_matrices = False)    
    u_t3_b_good_2, s_t3_b_good_2, vh_t3_b_good_2 = np.linalg.svd(cluster_list_task_3_b_good_2, full_matrices = False)
    
    
    
    #A good task 1 non-forced
    u_t1_a_good_1nf, s_t1_a_good_1nf, vh_t1_a_good_1nf = np.linalg.svd(cluster_list_task_1_a_good_1_nf, full_matrices = False)
    u_t1_a_good_2nf, s_t1_a_good_2nf, vh_t1_a_good_2nf = np.linalg.svd(cluster_list_task_1_a_good_2_nf, full_matrices = False)
    
    #B good task 1 non-forced
    u_t1_b_good_1nf, s_t1_b_good_1nf, vh_t1_b_good_1nf = np.linalg.svd(cluster_list_task_1_b_good_1_nf, full_matrices = False)    
    u_t1_b_good_2nf, s_t1_b_good_2nf, vh_t1_b_good_2nf = np.linalg.svd(cluster_list_task_1_b_good_2_nf, full_matrices = False)
    
    #A good task 2 non-forced
    u_t2_a_good_1nf, s_t2_a_good_1nf, vh_t2_a_good_1nf = np.linalg.svd(cluster_list_task_2_a_good_1_nf, full_matrices = False)
    u_t2_a_good_2nf, s_t2_a_good_2nf, vh_t2_a_good_2nf = np.linalg.svd(cluster_list_task_2_a_good_2_nf, full_matrices = False)
    
    #B good task 2 non-forced
    u_t2_b_good_1nf, s_t2_b_good_1nf, vh_t2_b_good_1nf = np.linalg.svd(cluster_list_task_2_b_good_1_nf, full_matrices = False)    
    u_t2_b_good_2nf, s_t2_b_good_2nf, vh_t2_b_good_2nf = np.linalg.svd(cluster_list_task_2_b_good_2_nf, full_matrices = False)

    #A good task 3 non-forced
    u_t3_a_good_1nf, s_t3_a_good_1nf, vh_t3_a_good_1nf = np.linalg.svd(cluster_list_task_3_a_good_1_nf, full_matrices = False)
    u_t3_a_good_2nf, s_t3_a_good_2nf, vh_t3_a_good_2nf = np.linalg.svd(cluster_list_task_3_a_good_2_nf, full_matrices = False)
    
    #B good task 3 non-forced
    u_t3_b_good_1nf, s_t3_b_good_1nf, vh_t3_b_good_1nf = np.linalg.svd(cluster_list_task_3_b_good_1_nf, full_matrices = False)    
    u_t3_b_good_2nf, s_t3_b_good_2nf, vh_t3_b_good_2nf = np.linalg.svd(cluster_list_task_3_b_good_2_nf, full_matrices = False)
    
    #Forced trials
    t_u_t1_a_good_1 = np.transpose(u_t1_a_good_1)
    t_vh_t1_a_good_1 = np.transpose(vh_t1_a_good_1)
      
    t_u_t2_a_good_1 = np.transpose(u_t2_a_good_1)
    t_vh_t2_a_good_1 = np.transpose(vh_t2_a_good_1)
    
    t_u_t3_a_good_1 = np.transpose(u_t3_a_good_1)
    t_vh_t3_a_good_1 = np.transpose(vh_t3_a_good_1)
    
    t_u_t1_b_good_1 = np.transpose(u_t1_b_good_1)
    t_vh_t1_b_good_1 = np.transpose(vh_t1_b_good_1)
      
    t_u_t2_b_good_1 = np.transpose(u_t2_b_good_1)
    t_vh_t2_b_good_1 = np.transpose(vh_t2_b_good_1)
   
    t_u_t3_b_good_1 = np.transpose(u_t3_b_good_1)
    t_vh_t3_b_good_1 = np.transpose(vh_t3_b_good_1)
    
    # Non-forced trials
    
    t_u_t1_a_good_1nf = np.transpose(u_t1_a_good_1nf)
    t_vh_t1_a_good_1nf = np.transpose(vh_t1_a_good_1nf)
      
    t_u_t2_a_good_1nf = np.transpose(u_t2_a_good_1nf)
    t_vh_t2_a_good_1nf = np.transpose(vh_t2_a_good_1nf)
    
    t_u_t3_a_good_1nf = np.transpose(u_t3_a_good_1nf)
    t_vh_t3_a_good_1nf = np.transpose(vh_t3_a_good_1nf)
    
    t_u_t1_b_good_1nf = np.transpose(u_t1_b_good_1nf)
    t_vh_t1_b_good_1nf = np.transpose(vh_t1_b_good_1nf)
      
    t_u_t2_b_good_1nf = np.transpose(u_t2_b_good_1nf)
    t_vh_t2_b_good_1nf = np.transpose(vh_t2_b_good_1nf)
   
    t_u_t3_b_good_1nf = np.transpose(u_t3_b_good_1nf)
    t_vh_t3_b_good_1nf = np.transpose(vh_t3_b_good_1nf)
    
    
    #Second half 
    t_u_t1_a_good_2nf = np.transpose(u_t1_a_good_2nf)
    t_vh_t1_a_good_2nf = np.transpose(vh_t1_a_good_2nf)
      
    t_u_t2_a_good_2nf = np.transpose(u_t2_a_good_2nf)
    t_vh_t2_a_good_2nf = np.transpose(vh_t2_a_good_2nf)
    
    t_u_t3_a_good_2nf = np.transpose(u_t3_a_good_2nf)
    t_vh_t3_a_good_2nf = np.transpose(vh_t3_a_good_2nf)
    
    t_u_t1_b_good_2nf = np.transpose(u_t1_b_good_2nf)
    t_vh_t1_b_good_2nf = np.transpose(vh_t1_b_good_2nf)
      
    t_u_t2_b_good_2nf = np.transpose(u_t2_b_good_2nf)
    t_vh_t2_b_good_2nf = np.transpose(vh_t2_b_good_2nf)
   
    t_u_t3_b_good_2nf = np.transpose(u_t3_b_good_2nf)
    t_vh_t3_b_good_2nf = np.transpose(vh_t3_b_good_2nf)
    
    
    
    # Predict within forced choice 
    s1_t1_a_from_a = np.linalg.multi_dot([t_u_t1_a_good_1, cluster_list_task_1_a_good_2, t_vh_t1_a_good_1])
    d_t1_a_from_a = s1_t1_a_from_a.diagonal()
    sum_s1_t1_a_from_a = np.cumsum(d_t1_a_from_a)/cluster_list_task_1_a_good_2.shape[0]
    
    s1_t2_a_from_a = np.linalg.multi_dot([t_u_t2_a_good_1, cluster_list_task_2_a_good_2, t_vh_t2_a_good_1])
    d_t2_a_from_a = s1_t2_a_from_a.diagonal()
    sum_s1_t2_a_from_a = np.cumsum(d_t2_a_from_a)/cluster_list_task_2_a_good_2.shape[0]
    
    s1_t3_a_from_a = np.linalg.multi_dot([t_u_t3_a_good_1, cluster_list_task_3_a_good_2, t_vh_t3_a_good_1])
    d_t3_a_from_a = s1_t3_a_from_a.diagonal()
    sum_s1_t3_a_from_a = np.cumsum(d_t3_a_from_a)/cluster_list_task_3_a_good_2.shape[0]
      
    s1_t1_b_from_b = np.linalg.multi_dot([t_u_t1_b_good_1, cluster_list_task_1_b_good_2, t_vh_t1_b_good_1])
    d_t1_b_from_b = s1_t1_b_from_b.diagonal()
    sum_s1_t1_b_from_b = np.cumsum(d_t1_b_from_b)/cluster_list_task_1_b_good_2.shape[0]
    
    s1_t2_b_from_b = np.linalg.multi_dot([t_u_t2_b_good_1, cluster_list_task_2_b_good_2, t_vh_t2_b_good_1])
    d_t2_b_from_b = s1_t2_b_from_b.diagonal()
    sum_s1_t2_b_from_b = np.cumsum(d_t2_b_from_b)/cluster_list_task_2_b_good_2.shape[0]
    
    s1_t3_a_from_a = np.linalg.multi_dot([t_u_t3_b_good_1, cluster_list_task_3_b_good_2, t_vh_t3_b_good_1])
    d_t3_b_from_b = s1_t3_a_from_a.diagonal()
    sum_s1_t3_b_from_b = np.cumsum(d_t3_b_from_b)/cluster_list_task_3_b_good_2.shape[0]
        
    # Predict within non-forced choice 
    
    s1_t1_a_from_a_nf = np.linalg.multi_dot([t_u_t1_a_good_1nf, cluster_list_task_1_a_good_2_nf, t_vh_t1_a_good_1nf])
    d_t1_a_from_a_nf = s1_t1_a_from_a_nf.diagonal()
    sum_s1_t1_a_from_a_nf = np.cumsum(d_t1_a_from_a_nf)/cluster_list_task_1_a_good_2_nf.shape[0]
    
    s1_t2_a_from_a_nf = np.linalg.multi_dot([t_u_t2_a_good_1nf, cluster_list_task_2_a_good_2_nf, t_vh_t2_a_good_1nf])
    d_t2_a_from_a_nf = s1_t2_a_from_a_nf.diagonal()
    sum_s1_t2_a_from_a_nf = np.cumsum(d_t2_a_from_a_nf)/cluster_list_task_2_a_good_2_nf.shape[0]
    
    s1_t3_a_from_a_nf = np.linalg.multi_dot([t_u_t3_a_good_1nf, cluster_list_task_3_a_good_2_nf, t_vh_t3_a_good_1nf])
    d_t3_a_from_a_nf = s1_t3_a_from_a_nf.diagonal()
    sum_s1_t3_a_from_a_nf = np.cumsum(d_t3_a_from_a_nf)/cluster_list_task_3_a_good_2_nf.shape[0]
    
    s1_t1_b_from_b_nf = np.linalg.multi_dot([t_u_t1_b_good_1nf, cluster_list_task_1_b_good_2_nf, t_vh_t1_b_good_1nf])
    d_t1_b_from_b_nf = s1_t1_b_from_b_nf.diagonal()
    sum_s1_t1_b_from_b_nf = np.cumsum(d_t1_b_from_b_nf)/cluster_list_task_1_b_good_2_nf.shape[0]
    
    s1_t2_b_from_b_nf = np.linalg.multi_dot([t_u_t2_b_good_1nf, cluster_list_task_2_b_good_2_nf, t_vh_t2_b_good_1nf])
    d_t2_b_from_b_nf = s1_t2_b_from_b_nf.diagonal()
    sum_s1_t2_b_from_b_nf = np.cumsum(d_t2_b_from_b_nf)/cluster_list_task_2_b_good_2_nf.shape[0]
    
    s1_t3_b_from_b_nf = np.linalg.multi_dot([t_u_t3_b_good_1nf, cluster_list_task_3_b_good_2_nf, t_vh_t3_b_good_1nf])
    d_t3_b_from_b_nf = s1_t3_b_from_b_nf.diagonal()
    sum_s1_t3_b_from_b_nf = np.cumsum(d_t3_b_from_b_nf)/cluster_list_task_3_b_good_2_nf.shape[0]
        
    average_within_forced_or_non_forced = np.mean([sum_s1_t1_a_from_a,sum_s1_t2_a_from_a, sum_s1_t3_a_from_a, sum_s1_t1_b_from_b,\
                                                   sum_s1_t2_b_from_b,sum_s1_t3_b_from_b,sum_s1_t1_a_from_a_nf,\
                                                   sum_s1_t2_a_from_a_nf,sum_s1_t3_a_from_a_nf, sum_s1_t1_b_from_b_nf,sum_s1_t2_b_from_b_nf,sum_s1_t3_b_from_b_nf], axis = 0)
    # Predict between non-forced and forced choice 
    
    s1_t1_a_from_a_nf_f_1 = np.linalg.multi_dot([t_u_t1_a_good_1nf, cluster_list_task_1_a_good_1, t_vh_t1_a_good_1nf])
    d_t1_a_from_a_nf_f_1 = s1_t1_a_from_a_nf_f_1.diagonal()
    sum_s1_t1_a_from_a_nf_f_1 = np.cumsum(d_t1_a_from_a_nf_f_1)/cluster_list_task_1_a_good_1.shape[0]
    
    s1_t2_a_from_a_nf_f_1 = np.linalg.multi_dot([t_u_t2_a_good_1nf, cluster_list_task_2_a_good_1, t_vh_t2_a_good_1nf])
    d_t2_a_from_a_nf_f_1 = s1_t2_a_from_a_nf_f_1.diagonal()
    sum_s1_t2_a_from_a_nf_f_1 = np.cumsum(d_t2_a_from_a_nf_f_1)/cluster_list_task_2_a_good_1.shape[0]
    
    s1_t3_a_from_a_nf_f_1 = np.linalg.multi_dot([t_u_t3_a_good_1nf, cluster_list_task_3_a_good_1, t_vh_t3_a_good_1nf])
    d_t3_a_from_a_nf_f_1 = s1_t3_a_from_a_nf_f_1.diagonal()
    sum_s1_t3_a_from_a_nf_f_1 = np.cumsum(d_t3_a_from_a_nf_f_1)/cluster_list_task_3_a_good_1.shape[0]
    
    s1_t1_b_from_b_nf_f_1 = np.linalg.multi_dot([t_u_t1_b_good_1nf, cluster_list_task_1_b_good_1, t_vh_t1_b_good_1nf])
    d_t1_b_from_b_nf_f_1 = s1_t1_b_from_b_nf_f_1.diagonal()
    sum_s1_t1_b_from_b_nf_f_1 = np.cumsum(d_t1_b_from_b_nf_f_1)/cluster_list_task_1_b_good_1.shape[0]
    
    s1_t2_b_from_b_nf_f_1 = np.linalg.multi_dot([t_u_t2_b_good_1nf, cluster_list_task_2_b_good_1, t_vh_t2_b_good_1nf])
    d_t2_b_from_b_nf_f_1 = s1_t2_b_from_b_nf_f_1.diagonal()
    sum_s1_t2_b_from_b_nf_f_1 = np.cumsum(d_t2_b_from_b_nf_f_1)/cluster_list_task_2_b_good_1.shape[0]
    
    s1_t3_a_from_a_nf_f_1 = np.linalg.multi_dot([t_u_t3_b_good_1nf, cluster_list_task_3_b_good_1, t_vh_t3_b_good_1nf])
    d_t3_b_from_b_nf_f_1 = s1_t3_a_from_a_nf_f_1.diagonal()
    sum_s1_t3_b_from_b_nf_f_1 = np.cumsum(d_t3_b_from_b_nf_f_1)/cluster_list_task_3_b_good_1.shape[0]
         
    # Second Half
    s1_t1_a_from_a_nf_f_2 = np.linalg.multi_dot([t_u_t1_a_good_2nf, cluster_list_task_1_a_good_2, t_vh_t1_a_good_2nf])
    d_t1_a_from_a_nf_f_2 = s1_t1_a_from_a_nf_f_2.diagonal()
    sum_s1_t1_a_from_a_nf_f_2 = np.cumsum(d_t1_a_from_a_nf_f_2)/cluster_list_task_1_a_good_2.shape[0]
    
    s1_t2_a_from_a_nf_f_2 = np.linalg.multi_dot([t_u_t2_a_good_2nf, cluster_list_task_2_a_good_2, t_vh_t2_a_good_2nf])
    d_t2_a_from_a_nf_f_2 = s1_t2_a_from_a_nf_f_2.diagonal()
    sum_s1_t2_a_from_a_nf_f_2 = np.cumsum(d_t2_a_from_a_nf_f_2)/cluster_list_task_2_a_good_2.shape[0]
    
    s1_t3_a_from_a_nf_f_2 = np.linalg.multi_dot([t_u_t3_a_good_2nf, cluster_list_task_3_a_good_2, t_vh_t3_a_good_2nf])
    d_t3_a_from_a_nf_f_2 = s1_t3_a_from_a_nf_f_2.diagonal()
    sum_s1_t3_a_from_a_nf_f_2 = np.cumsum(d_t3_a_from_a_nf_f_2)/cluster_list_task_3_a_good_2.shape[0]
    
    s1_t1_b_from_b_nf_f_2 = np.linalg.multi_dot([t_u_t1_b_good_2nf, cluster_list_task_1_b_good_2, t_vh_t1_b_good_2nf])
    d_t1_b_from_b_nf_f_2 = s1_t1_b_from_b_nf_f_2.diagonal()
    sum_s1_t1_b_from_b_nf_f_2 = np.cumsum(d_t1_b_from_b_nf_f_2)/cluster_list_task_1_b_good_2.shape[0]
    
    s1_t2_b_from_b_nf_f_2 = np.linalg.multi_dot([t_u_t2_b_good_2nf, cluster_list_task_2_b_good_2, t_vh_t2_b_good_2nf])
    d_t2_b_from_b_nf_f_2 = s1_t2_b_from_b_nf_f_2.diagonal()
    sum_s1_t2_b_from_b_nf_f_2 = np.cumsum(d_t2_b_from_b_nf_f_2)/cluster_list_task_2_b_good_2.shape[0]
    
    s1_t3_a_from_a_nf_f_2 = np.linalg.multi_dot([t_u_t3_b_good_2nf, cluster_list_task_3_b_good_2, t_vh_t3_b_good_2nf])
    d_t3_b_from_b_nf_f_2 = s1_t3_a_from_a_nf_f_2.diagonal()
    sum_s1_t3_b_from_b_nf_f_2 = np.cumsum(d_t3_b_from_b_nf_f_2)/cluster_list_task_3_b_good_2.shape[0]
    
    average_between_forced_non_forced = np.mean([sum_s1_t1_a_from_a_nf_f_1, sum_s1_t2_a_from_a_nf_f_1,\
                                                 sum_s1_t3_a_from_a_nf_f_1, sum_s1_t1_b_from_b_nf_f_1,\
                                                 sum_s1_t2_b_from_b_nf_f_1, sum_s1_t3_b_from_b_nf_f_1,\
                                                 sum_s1_t1_a_from_a_nf_f_2, sum_s1_t2_a_from_a_nf_f_2,\
                                                 sum_s1_t3_a_from_a_nf_f_2, sum_s1_t1_b_from_b_nf_f_2,\
                                                 sum_s1_t2_b_from_b_nf_f_2,sum_s1_t3_b_from_b_nf_f_2], axis = 0)
    
    if HP == True :
        plt.plot(average_within_forced_or_non_forced, label = 'Within Trial Type HP', color = 'grey')
        plt.plot(average_between_forced_non_forced, label = 'Between Trial Type HP', linestyle = '--', color='grey')
    elif HP == False:
        plt.plot(average_within_forced_or_non_forced, label = 'Within Trial Type PFC', color = 'pink')
        plt.plot(average_between_forced_non_forced, label = 'Between Trial Type PFC', linestyle = '--', color='pink')
        
    plt.legend()