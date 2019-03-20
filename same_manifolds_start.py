#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:24:40 2019

@author: veronikasamborska
"""



d1 = np.linalg.multi_dot([u_t1_1, flattened_all_clusters_task_1_first_half])
d2 = np.linalg.multi_dot([u_t1_2, flattened_all_clusters_task_1_second_half])

d1_qr = np.linalg.qr(d1)
d2_qr = np.linalg.qr(d2)


d1_d2_qr = np.linalg.multi_dot([np.transpose(d1_qr[0]), d2_qr[0]])

u, s, v = np.linalg.svd(d1_d2_qr, full_matrices = False)
plt.plot(np.cumsum(s)/flattened_all_clusters_task_1_first_half.shape[0])

d1 = np.linalg.multi_dot([u_t1_1, flattened_all_clusters_task_1_first_half])
d3 = np.linalg.multi_dot([u_t2_2, flattened_all_clusters_task_2_first_half])

d1_qr = np.linalg.qr(d1)
d3_qr = np.linalg.qr(d3)


d1_d3_qr = np.linalg.multi_dot([np.transpose(d1_qr[0]), d3_qr[0]])

u1, s1, v1 = np.linalg.svd(d1_d3_qr, full_matrices = False)
plt.plot(np.cumsum(s1)/flattened_all_clusters_task_1_first_half.shape[0])