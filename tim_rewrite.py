#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:22:57 2019

@author: veronikasamborska
"""

flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = sv.flatten(experiment_aligned_PFC, tasks_unchanged = True, plot_a = False, plot_b = False, average_reward = False)

#% REORD.M  Try Spectral reordering on Oxford data.
#%
#% [r3]=reord(A,power)
#% Based on spec_cat.m
#%
#%
#% DJH, August 2003
# 
# 
#%%%%%% Spectral Ordering %%%%%%
#%A=sparse(A);
#G = (A + A')/2;             %forces symmetry
#G = G.^power;
#Q = -G;
#Q = triu(Q,1) + tril(Q,-1);
#Q = Q - diag(sum(Q));
# 
#t = 1./sqrt((sum(G)));
#Q =  diag(t)*Q*diag(t);    %Normalized Laplacian  
# 
#% get second eigenvalue
# 
#%Qs=sparse(Q);
#%[V,D]=eigs(Qs,2,'SM');
# 
#[V,D] = eig(Q);
# 
# 
#d = diag(D);
# 
#[a,b] = sort(d);
#index = b(2);
#%keyboard
#v2 = V(:,index);
# 
#v2scale = diag(t)*v2;
# 
#[y,r3] = sort(v2scale);
# 
