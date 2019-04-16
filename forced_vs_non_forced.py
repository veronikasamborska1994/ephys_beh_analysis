#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:35:58 2019

@author: veronikasamborska
"""

import svd_block_analysis as svdb
import svd_forced_trials as svdf
import numpy as np
import matplotlib.pyplot as plt

#First half are As and second half are Bs 

cluster_list_task_1_a_good_1_nf, cluster_list_task_1_b_good_1_nf,\
cluster_list_task_2_a_good_1_nf, cluster_list_task_2_b_good_1_nf,\
cluster_list_task_3_a_good_1_nf, cluster_list_task_3_b_good_1_nf, cluster_list_task_1_a_good_2_nf,\
cluster_list_task_1_b_good_2_nf, cluster_list_task_2_a_good_2_nf, cluster_list_task_2_b_good_2_nf,\
cluster_list_task_3_a_good_2_nf, cluster_list_task_3_b_good_2_nf  = svdb.block_firings_rates_selection(experiment_aligned_PFC)

cluster_list_task_1_a_good_1, cluster_list_task_1_b_good_1,\
cluster_list_task_2_a_good_1, cluster_list_task_2_b_good_1,\
cluster_list_task_3_a_good_1, cluster_list_task_3_b_good_1, cluster_list_task_1_a_good_2,\
cluster_list_task_1_b_good_2, cluster_list_task_2_a_good_2, cluster_list_task_2_b_good_2,\
cluster_list_task_3_a_good_2, cluster_list_task_3_b_good_2  = svdf.block_firings_rates_selection_forced_split_in_half(PFC_forced)

#SVD to see if forced B choices in A good block are more different 
#than free B choices in A good block than forced B choices in B good block to free B choices in B good block.

trials_between_a_and_b = int(cluster_list_task_1_a_good_1_nf.shape[1]/2)

b_in_a_good_task_1_nf =cluster_list_task_1_a_good_1_nf[:,trials_between_a_and_b:]
b_in_b_good_task_1_nf =cluster_list_task_1_b_good_1_nf[:,trials_between_a_and_b:]

b_in_a_good_task_2_nf =cluster_list_task_2_a_good_1_nf[:,trials_between_a_and_b:]
b_in_b_good_task_2_nf =cluster_list_task_2_b_good_1_nf[:,trials_between_a_and_b:]

b_in_a_good_task_3_nf =cluster_list_task_3_a_good_1_nf[:,trials_between_a_and_b:]
b_in_b_good_task_3_nf =cluster_list_task_3_b_good_1_nf[:,trials_between_a_and_b:]

b_in_a_good_task_1_f =cluster_list_task_1_a_good_1[:,trials_between_a_and_b:]
b_in_b_good_task_1_f =cluster_list_task_1_b_good_1[:,trials_between_a_and_b:]

b_in_a_good_task_2_f =cluster_list_task_2_a_good_1[:,trials_between_a_and_b:]
b_in_b_good_task_2_f =cluster_list_task_2_b_good_1[:,trials_between_a_and_b:]

b_in_a_good_task_3_f =cluster_list_task_3_a_good_1[:,trials_between_a_and_b:]
b_in_b_good_task_3_f =cluster_list_task_3_b_good_1[:,trials_between_a_and_b:]

# =============================================================================
# # Task 1 B analysis 
# 
# =============================================================================
u_b_good_t1nf_b, s_t1_b_good_t1nf_b, vh_b_good_t1nf_b= np.linalg.svd(b_in_b_good_task_1_nf, full_matrices = False)
#u_b_good_t1f_a, s_t1_b_good_t1f_b, vh_b_good_t1f_b= np.linalg.svd(b_in_b_good_task_1_f, full_matrices = False)

u_a_good_t1nf_b, s_t1_a_good_t1nf_b, vh_a_good_t1nf_b= np.linalg.svd(b_in_a_good_task_1_nf, full_matrices = False)
#u_a_good_t1f_a, s_t1_a_good_t1f_b, vh_a_good_t1f_b= np.linalg.svd(b_in_a_good_task_1_f, full_matrices = False)
  

t_vh_b_good_t1nf_b= np.transpose(vh_b_good_t1nf_b)
t_u_b_good_t1nf_b = np.transpose(u_b_good_t1nf_b)
t_vh_a_good_t1nf_b = np.transpose(vh_a_good_t1nf_b)
t_u_a_good_t1nf_b = np.transpose(u_a_good_t1nf_b)

    
#Predict within blocks 
s_b_in_good = np.linalg.multi_dot([t_u_b_good_t1nf_b, b_in_b_good_task_1_f, t_vh_b_good_t1nf_b])

d_s_b_in_good = s_b_in_good.diagonal()

sum_s_b_in_good = np.cumsum(abs(d_s_b_in_good))/b_in_a_good_task_1_nf.shape[0]


#Predict between bocks
s_b_in_bad = np.linalg.multi_dot([t_u_a_good_t1nf_b, b_in_a_good_task_1_f, t_vh_a_good_t1nf_b])

d_s_b_in_bad = s_b_in_bad.diagonal()

sum_s_b_in_bad = np.cumsum(abs(d_s_b_in_bad))/b_in_a_good_task_1_nf.shape[0]
   
# =============================================================================
# # Task 2 B analysis
# 
# =============================================================================
u_b_good_t2nf_b, s_b_good_t2nf_b, vh_b_good_t2nf_b= np.linalg.svd(b_in_b_good_task_2_nf, full_matrices = False)
#u_b_good_t1f_a, s_t1_b_good_t1f_b, vh_b_good_t1f_b= np.linalg.svd(b_in_b_good_task_1_f, full_matrices = False)

u_a_good_t2nf_b, s_a_good_t2nf_b, vh_a_good_t2nf_b= np.linalg.svd(b_in_a_good_task_2_nf, full_matrices = False)
#u_a_good_t1f_a, s_t1_a_good_t1f_b, vh_a_good_t1f_b= np.linalg.svd(b_in_a_good_task_1_f, full_matrices = False)
  

t_vh_b_good_t2nf_b= np.transpose(vh_b_good_t2nf_b)
t_u_b_good_t2nf_b = np.transpose(u_b_good_t2nf_b)
t_vh_a_good_t2nf_b = np.transpose(vh_a_good_t2nf_b)
t_u_a_good_t2nf_b = np.transpose(u_a_good_t2nf_b)

    
#Predict within blocks 
s_b_in_good_t2 = np.linalg.multi_dot([t_u_b_good_t2nf_b, b_in_b_good_task_2_f, t_vh_b_good_t2nf_b])

d_s_b_in_good_t2= s_b_in_good_t2.diagonal()

sum_s_b_in_good_t2= np.cumsum(abs(d_s_b_in_good_t2))/b_in_a_good_task_2_nf.shape[0]


#Predict between bocks
s_b_in_bad_t2 = np.linalg.multi_dot([t_u_a_good_t2nf_b, b_in_a_good_task_2_f, t_vh_a_good_t2nf_b])

d_s_b_in_bad_t2 = s_b_in_bad_t2.diagonal()

sum_s_b_in_bad_t2 = np.cumsum(abs(d_s_b_in_bad_t2))/b_in_a_good_task_2_nf.shape[0]
   
# =============================================================================
# # Task 3 B analysis
# 
# =============================================================================

u_b_good_t3nf_b, s_b_good_t3nf_b, vh_b_good_t3nf_b= np.linalg.svd(b_in_b_good_task_3_nf, full_matrices = False)
#u_b_good_t1f_a, s_t1_b_good_t1f_b, vh_b_good_t1f_b= np.linalg.svd(b_in_b_good_task_1_f, full_matrices = False)

u_a_good_t3nf_b, s_a_good_t3nf_b, vh_a_good_t3nf_b= np.linalg.svd(b_in_a_good_task_3_nf, full_matrices = False)
#u_a_good_t1f_a, s_t1_a_good_t1f_b, vh_a_good_t1f_b= np.linalg.svd(b_in_a_good_task_1_f, full_matrices = False)
  

t_vh_b_good_t3nf_b= np.transpose(vh_b_good_t3nf_b)
t_u_b_good_t3nf_b = np.transpose(u_b_good_t3nf_b)
t_vh_a_good_t3nf_b = np.transpose(vh_a_good_t3nf_b)
t_u_a_good_t3nf_b = np.transpose(u_a_good_t3nf_b)

    
#Predict within blocks 
s_b_in_good_t3 = np.linalg.multi_dot([t_u_b_good_t3nf_b, b_in_b_good_task_3_f, t_vh_b_good_t3nf_b])

d_s_b_in_good_t3 = s_b_in_good_t3.diagonal()

sum_s_b_in_good_t3 = np.cumsum(abs(d_s_b_in_good_t3))/b_in_a_good_task_3_nf.shape[0]


#Predict between bocks
s_b_in_bad_t3 = np.linalg.multi_dot([t_u_a_good_t3nf_b, b_in_a_good_task_3_f, t_vh_a_good_t3nf_b])

d_s_b_in_bad_t3 = s_b_in_bad_t3.diagonal()

sum_s_b_in_bad_t3 = np.cumsum(abs(d_s_b_in_bad_t3))/b_in_a_good_task_3_nf.shape[0]
   

# =============================================================================
# # Task 1 A analysis 
# 
# =============================================================================

a_in_a_good_task_1_nf = cluster_list_task_1_a_good_1_nf[:,:trials_between_a_and_b]
a_in_b_good_task_1_nf = cluster_list_task_1_b_good_1_nf[:,:trials_between_a_and_b]

a_in_a_good_task_2_nf = cluster_list_task_2_a_good_1_nf[:,:trials_between_a_and_b]
a_in_b_good_task_2_nf = cluster_list_task_2_b_good_1_nf[:,:trials_between_a_and_b]

a_in_a_good_task_3_nf = cluster_list_task_3_a_good_1_nf[:,:trials_between_a_and_b]
a_in_b_good_task_3_nf = cluster_list_task_3_b_good_1_nf[:,:trials_between_a_and_b]

a_in_a_good_task_1_f = cluster_list_task_1_a_good_1[:,:trials_between_a_and_b]
a_in_b_good_task_1_f = cluster_list_task_1_b_good_1[:,:trials_between_a_and_b]

a_in_a_good_task_2_f = cluster_list_task_2_a_good_1[:,:trials_between_a_and_b]
a_in_b_good_task_2_f = cluster_list_task_2_b_good_1[:,:trials_between_a_and_b]

a_in_a_good_task_3_f = cluster_list_task_3_a_good_1[:,:trials_between_a_and_b]
a_in_b_good_task_3_f = cluster_list_task_3_b_good_1[:,:trials_between_a_and_b]



u_a_good_t1nf_a, s_t1_a_good_t1nf_a, vh_a_good_t1nf_a = np.linalg.svd(a_in_a_good_task_1_nf, full_matrices = False)
#u_b_good_t1f_a, s_t1_b_good_t1f_b, vh_b_good_t1f_b= np.linalg.svd(b_in_b_good_task_1_f, full_matrices = False)

u_b_good_t1nf_a, s_t1_b_good_t1nf_a, vh_b_good_t1nf_a = np.linalg.svd(a_in_b_good_task_1_nf, full_matrices = False)
#u_a_good_t1f_a, s_t1_a_good_t1f_b, vh_a_good_t1f_b= np.linalg.svd(b_in_a_good_task_1_f, full_matrices = False)
  

t_vh_a_good_t1nf_a = np.transpose(vh_a_good_t1nf_a)
t_u_a_good_t1nf_a = np.transpose(u_a_good_t1nf_a)
t_vh_b_good_t1nf_a = np.transpose(vh_b_good_t1nf_a)
t_u_b_good_t1nf_a = np.transpose(u_b_good_t1nf_a)

    
#Predict within blocks 
s_a_in_good = np.linalg.multi_dot([t_u_a_good_t1nf_a, a_in_a_good_task_1_f, t_vh_a_good_t1nf_a])

d_s_a_in_good = s_a_in_good.diagonal()

sum_s_a_in_good = np.cumsum(abs(d_s_a_in_good))/b_in_a_good_task_1_nf.shape[0]

# Predict between bocks
s_a_in_bad = np.linalg.multi_dot([t_u_b_good_t1nf_a, a_in_b_good_task_1_f, t_vh_b_good_t1nf_a])

d_s_a_in_bad = s_a_in_bad.diagonal()

sum_s_a_in_bad = np.cumsum(abs(d_s_a_in_bad))/b_in_a_good_task_1_nf.shape[0] 

# =============================================================================
# #Task 2 A analysis
# =============================================================================



u_a_good_t2nf_a, s_a_good_t2nf_a, vh_a_good_t2nf_a = np.linalg.svd(a_in_a_good_task_2_nf, full_matrices = False)

u_b_good_t2nf_a, s_b_good_t2nf_a, vh_b_good_t2nf_a = np.linalg.svd(a_in_b_good_task_2_nf, full_matrices = False)
  

t_vh_a_good_t2nf_a = np.transpose(vh_a_good_t2nf_a)
t_u_a_good_t2nf_a = np.transpose(u_a_good_t2nf_a)
t_vh_b_good_t2nf_a = np.transpose(vh_b_good_t2nf_a)
t_u_b_good_t2nf_a = np.transpose(u_b_good_t2nf_a)

    
#Predict within blocks 
s_a_in_good_t2 = np.linalg.multi_dot([t_u_a_good_t2nf_a, a_in_a_good_task_2_f, t_vh_a_good_t2nf_a])

d_s_a_in_good_t2 = s_a_in_good_t2.diagonal()

sum_s_a_in_good_t2 = np.cumsum(abs(d_s_a_in_good_t2))/b_in_a_good_task_2_nf.shape[0]

# Predict between bocks
s_a_in_bad_t2 = np.linalg.multi_dot([t_u_b_good_t2nf_a, a_in_b_good_task_2_f, t_vh_b_good_t2nf_a])

d_s_a_in_bad_t2 = s_a_in_bad_t2.diagonal()

sum_s_a_in_bad_t2 = np.cumsum(abs(d_s_a_in_bad_t2))/b_in_a_good_task_2_nf.shape[0] 



# =============================================================================
# #Task 3 A analysis
# =============================================================================


u_a_good_t3nf_a, s_a_good_t3nf_a, vh_a_good_t3nf_a = np.linalg.svd(a_in_a_good_task_3_nf, full_matrices = False)

u_b_good_t3nf_a, s_b_good_t3nf_a, vh_b_good_t3nf_a = np.linalg.svd(a_in_b_good_task_3_nf, full_matrices = False)
  

t_vh_a_good_t3nf_a = np.transpose(vh_a_good_t3nf_a)
t_u_a_good_t3nf_a = np.transpose(u_a_good_t3nf_a)
t_vh_b_good_t3nf_a = np.transpose(vh_b_good_t3nf_a)
t_u_b_good_t3nf_a = np.transpose(u_b_good_t3nf_a)

    
#Predict within blocks 
s_a_in_good_t3 = np.linalg.multi_dot([t_u_a_good_t3nf_a, a_in_a_good_task_3_f, t_vh_a_good_t3nf_a])

d_s_a_in_good_t3 = s_a_in_good_t3.diagonal()

sum_s_a_in_good_t3 = np.cumsum(abs(d_s_a_in_good_t3))/b_in_a_good_task_3_nf.shape[0]

# Predict between bocks
s_a_in_bad_t3 = np.linalg.multi_dot([t_u_b_good_t3nf_a, a_in_b_good_task_3_f, t_vh_b_good_t3nf_a])

d_s_a_in_bad_t3 = s_a_in_bad_t3.diagonal()

sum_s_a_in_bad_t3 = np.cumsum(abs(d_s_a_in_bad_t3))/b_in_a_good_task_3_nf.shape[0] 


mean_expected_choice_free_forced = np.mean([sum_s_a_in_good_t3,sum_s_a_in_good_t2,sum_s_a_in_good, sum_s_b_in_good_t3,sum_s_b_in_good_t2, sum_s_b_in_good], axis =0 )
mean_unexpected_choice_free_forced = np.mean([sum_s_a_in_bad_t3,sum_s_a_in_bad_t2,sum_s_a_in_bad,sum_s_b_in_bad_t3,sum_s_b_in_bad_t2,sum_s_b_in_bad], axis = 0)


plt.plot(mean_expected_choice_free_forced, label = 'Expected Choice Explaining Forced Choices from Free Choices PFC',color = 'black')

plt.plot(mean_unexpected_choice_free_forced, label = 'Unexpected Choice Explaining Forced Choices from Free Choices PFC', linestyle = '--',color = 'black')

plt.legend()

