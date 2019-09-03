#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:57:22 2019

@author: veronikasamborska
"""
# =============================================================================
# State space of the generalisation task
# =============================================================================


import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# State Space of the task
states = 5
n = 40
DD = np.zeros((n,n))
ones =  np.ones(n)
indices_empty = [4,9,14,19,24,29,34,39]
ones[indices_empty] = 0 
np.fill_diagonal(DD[:,1:], ones)

com = 0 #change of mind - if zero the probability of startining in initiation state 1 and moving to choice state 2 is zero
rewprob = 0.75  #prob of moving to rewarded state
Stay_rew = 0.8
Stay_nr = 0.4
 

DD[states-1,2*states] = 1-com
DD[2*states-1,3*states] = 1-com

DD[3*states-1,4*states] = rewprob
DD[3*states-1,5*states] = 1-rewprob
DD[4*states-1,6*states] = rewprob
DD[4*states-1,7*states] = 1-rewprob;
 
DD[5*states-1,0] = Stay_rew
DD[5*states-1,states] = 1-Stay_rew

DD[6*states-1,0] = Stay_nr
DD[6*states-1,states] = 1-Stay_nr;
 
DD[7*states-1,0] = 1-Stay_rew
DD[7*states-1,states] = Stay_rew
DD[8*states-1,0] = 1-Stay_nr
DD[8*states-1,states] = Stay_nr


Areward_ind = np.concatenate([np.arange(0,states) ,np.arange(2*states,3*states),np.arange(4*states,5*states)], axis =0)

Anoreward_ind = np.concatenate([np.arange(0,states) ,np.arange(2*states,3*states),np.arange(5*states,6*states)], axis =0)

Breward_ind = np.concatenate([np.arange(states,2*states) ,np.arange(3*states,states*4),np.arange(6*states,7*states)], axis =0)

Bnoreward_ind = np.concatenate([np.arange(states,2*states) ,np.arange(3*states,4*states),np.arange(7*states,8*states)], axis =0)


w,v=np.linalg.eig(1-DD)
plt.figure(1)
plt.imshow(-np.log(expm(DD)))
plt.figure(2)
plt.imshow(v.real)
plt.figure(3)
plt.imshow(v.imag)

plt.figure(4)
plt.subplot(2,2,1)
plt.imshow(v.real[Areward_ind,:])
plt.subplot(2,2,2)
plt.imshow(v.real[Anoreward_ind,:])
plt.subplot(2,2,3)
plt.imshow(v.real[Breward_ind,:])
plt.subplot(2,2,4)
plt.imshow(v.real[Bnoreward_ind,:])

v.conjugate

plt.figure(5)
for i in np.arange(1,40):  
    plt.subplot(5,8,i)
    plt.plot(v.real[Areward_ind,i])
    plt.plot(v.real[Breward_ind,i],color = 'r')
    plt.plot(v.real[Anoreward_ind,i],'g')
    plt.plot(v.real[Bnoreward_ind,i],'k')
    plt.plot(v.imag[Areward_ind,i],linestyle = '--')
    plt.plot(v.imag[Breward_ind,i],color = 'r',linestyle = '--')
    plt.plot(v.imag[Anoreward_ind,i],'g',linestyle = '--')
    plt.plot(v.imag[Bnoreward_ind,i],'k',linestyle = '--')


