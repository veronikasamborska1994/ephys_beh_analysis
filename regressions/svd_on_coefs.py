#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:37:33 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/veronikasamborska/Desktop/ephys_beh_analysis/regressions')

import regressions_general as reg_gen


def svd_on_coefs(data, title):
    
    C, cpd, C_1, cpd_1,C_2, cpd_2, C_3, cpd_3, predictors = reg_gen.regression_general(data)
    #C_1_PFC, cpd_1_PFC,C_2_PFC, cpd_2_PFC, C_3_PFC, cpd_3_PFC, predictors = reg_gen.regression_general(data_PFC)
    
    
    C_1_cut = np.transpose(np.transpose(C_1,[0,2,1]).reshape(6*C_1.shape[2], C_1.shape[1]))
    C_2_cut = np.transpose(np.transpose(C_2,[0,2,1]).reshape(6*C_2.shape[2], C_2.shape[1]))
    C_3_cut = np.transpose(np.transpose(C_3,[0,2,1]).reshape(6*C_3.shape[2], C_3.shape[1]))
    
    
    where_are_NaNs = np.isnan(C_1_cut)
    C_1_cut[where_are_NaNs] = 0
    
    where_are_inf = np.isinf(C_1_cut)
    C_1_cut[where_are_inf] = 0
    
    
    
    where_are_NaNs = np.isnan(C_3_cut)
    C_3_cut[where_are_NaNs] = 0
    
    where_are_inf = np.isinf(C_3_cut)
    C_3_cut[where_are_inf] = 0
    
    u_t1_1, s_t1_1, vh_t1_1 = np.linalg.svd(C_1_cut, full_matrices = True)
    
    t_u = np.transpose(u_t1_1)  
    t_v = np.transpose(vh_t1_1)  
    
    s_task_2 = np.linalg.multi_dot([t_u, C_3_cut, t_v])
    s_diag = np.cumsum(abs(s_task_2.diagonal()))/C_2_cut.shape[0]
    plt.plot(s_diag, label = title)
    plt.legend()
    
    return s_diag

s_diag = svd_on_coefs(HP, 'HP')
s_diag_PFC =  svd_on_coefs(data_PFC, 'PFC')
HP = np.trapz(s_diag)
PFC = np.trapz(s_diag_PFC)
