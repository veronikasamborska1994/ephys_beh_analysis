#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:19:56 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt

# RSA Physical Space coding

port_a_choice_task_1 = np.array([bool(i & 0) for i in range(10)])
port_a_choice_task_1[0:3] = True

port_a_choice_task_2 = np.array([bool(i & 0) for i in range(10)])
port_a_choice_task_2[0:3] = True

port_a_choice_task_3 = np.array([bool(i & 0) for i in range(10)])
port_a_choice_task_3[0:3] = True


port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(10)])
port_2_initiation_task_1[3:6] = True

port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(10)])
port_2_initiation_task_2[3:6] = True

port_2_choice_task_3 = np.array([bool(i & 0) for i in range(10)])
port_2_choice_task_3[3:6] = True

port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(10)])
port_3_initiation_task_3[6:8] = True

port_3_choice_task_2 = np.array([bool(i & 0) for i in range(10)])
port_3_choice_task_2[6:8] =  True

port_4_choice_task_1 = np.array([bool(i & 0) for i in range(10)])
port_4_choice_task_1[8] = True

port_5_choice_task_2 = np.array([bool(i & 0) for i in range(10)])
port_5_choice_task_2[9] =  True

physical_rsa = np.vstack([port_a_choice_task_1,port_a_choice_task_2,port_a_choice_task_3,\
                          port_2_initiation_task_1,port_2_initiation_task_2,port_2_choice_task_3,\
                          port_3_initiation_task_3,port_3_choice_task_2,port_4_choice_task_1,port_5_choice_task_2])

plt.imshow(physical_rsa)
plt.xticks([0,1,2,3,4,5,6,7,8,9], ('1 Choice T1', '1 Choice T2', '1 Choice T3', '2 Init T1',\
           '2 Init T2', '2 Choice T3', '3 Init T3', '3 Choice T2',\
           '4 Choice T1', '5 Choice T2'),rotation='vertical')

plt.yticks([0,1,2,3,4,5,6,7,8,9], ('A T1', 'A T2', 'A T3', '2 Init T1',\
           '2 Init T2', '2 Choice T3', '3 Init T3', '3 Choice T2',\
           '4 Choice T1', '5 Choice T2'))
plt.subplots_adjust(bottom=0.3)


# RSA Remapping Because of Meaning (Initiation vs Choice) Coding

port_a_choice_task_1 = np.array([bool(i & 0) for i in range(10)])
port_a_choice_task_1[0:3] = True

port_a_choice_task_2 = np.array([bool(i & 0) for i in range(10)])
port_a_choice_task_2[0:3] = True

port_a_choice_task_3 = np.array([bool(i & 0) for i in range(10)])
port_a_choice_task_3[0:3] = True


port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(10)])
port_2_initiation_task_1[3:5] = True

port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(10)])
port_2_initiation_task_2[3:5] = True

port_2_choice_task_3 = np.array([bool(i & 0) for i in range(10)])
port_2_choice_task_3[5] = True

port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(10)])
port_3_initiation_task_3[6] = True

port_3_choice_task_2 = np.array([bool(i & 0) for i in range(10)])
port_3_choice_task_2[7] =  True

port_4_choice_task_1 = np.array([bool(i & 0) for i in range(10)])
port_4_choice_task_1[8] = True

port_5_choice_task_2 = np.array([bool(i & 0) for i in range(10)])
port_5_choice_task_2[9] =  True

remapping_stop_firing = np.vstack([port_a_choice_task_1,port_a_choice_task_2,port_a_choice_task_3,\
                          port_2_initiation_task_1,port_2_initiation_task_2,port_2_choice_task_3,\
                          port_3_initiation_task_3,port_3_choice_task_2,port_4_choice_task_1,port_5_choice_task_2])

plt.imshow(remapping_stop_firing)
plt.xticks([0,1,2,3,4,5,6,7,8,9], ('1 Choice T1', '1 Choice T2', '1 Choice T3', '2 Init T1',\
           '2 Init T2', '2 Choice T3', '3 Init T3', '3 Choice T2',\
           '4 Choice T1', '5 Choice T2'),rotation='vertical')

plt.yticks([0,1,2,3,4,5,6,7,8,9], ('A T1', 'A T2', 'A T3', '2 Init T1',\
           '2 Init T2', '2 Choice T3', '3 Init T3', '3 Choice T2',\
           '4 Choice T1', '5 Choice T2'))
plt.subplots_adjust(bottom=0.3)



# RSA Initation  coding
port_a_choice_task_1 = np.array([bool(i & 0) for i in range(10)])

port_a_choice_task_2 = np.array([bool(i & 0) for i in range(10)])

port_a_choice_task_3 = np.array([bool(i & 0) for i in range(10)])


port_2_initiation_task_1 = np.array([bool(i & 0) for i in range(10)])
port_2_initiation_task_1[3] = True
port_2_initiation_task_1[4] = True
port_2_initiation_task_1[6] = True

port_2_initiation_task_2 = np.array([bool(i & 0) for i in range(10)])
port_2_initiation_task_2[3] = True
port_2_initiation_task_2[4] = True
port_2_initiation_task_2[6] = True


port_2_choice_task_3 = np.array([bool(i & 0) for i in range(10)])

port_3_initiation_task_3 = np.array([bool(i & 0) for i in range(10)])
port_3_initiation_task_3[3] = True
port_3_initiation_task_3[4] = True
port_3_initiation_task_3[6] = True

port_3_choice_task_2 = np.array([bool(i & 0) for i in range(10)])

port_4_choice_task_1 = np.array([bool(i & 0) for i in range(10)])

port_5_choice_task_2 = np.array([bool(i & 0) for i in range(10)])

initiation_rsa = np.vstack([port_a_choice_task_1,port_a_choice_task_2,port_a_choice_task_3,\
                          port_2_initiation_task_1,port_2_initiation_task_2,port_2_choice_task_3,\
                          port_3_initiation_task_3,port_3_choice_task_2,port_4_choice_task_1,port_5_choice_task_2])

plt.imshow(initiation_rsa)
plt.xticks([0,1,2,3,4,5,6,7,8,9], ('1 Choice T1', '1 Choice T2', '1 Choice T3', '2 Init T1',\
           '2 Init T2', '2 Choice T3', '3 Init T3', '3 Choice T2',\
           '4 Choice T1', '5 Choice T2'),rotation='vertical')

plt.yticks([0,1,2,3,4,5,6,7,8,9], ('A T1', 'A T2', 'A T3', '2 Init T1',\
           '2 Init T2', '2 Choice T3', '3 Init T3', '3 Choice T2',\
           '4 Choice T1', '5 Choice T2'))
plt.subplots_adjust(bottom=0.3)
