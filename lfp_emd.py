#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:26:07 2019

@author: veronikasamborska
"""


import sys
sys.path.append('/Users/veronikasamborska/Documents/source_code/emd')
import emd

#lfp = np.load('/Users/veronikasamborska/Desktop/neurons/m479/LFP/m479_2018-08-13.npy')
#plt.plot(lfp[0,:], lfp[1,:])

sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds

import numpy as np
time_vect = np.linspace(0, seconds, num_samples)

freq = 5
nonlinearity_deg = .25 # change extent of deformation from sinusoidal shape [-1 to 1]
nonlinearity_phi = -np.pi/4 # change left-right skew of deformation [-pi to pi]
x = emd.utils.abreu2010( freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds )
x += np.cos( 2*np.pi*1*time_vect )
imf = emd.sift.sift( x )
#plt.plot(x)

emd.plotting.plot_imfs(imf, cmap= True)

IP,IF,IA = emd.spectra.frequency_stats( imf, sample_rate, 'nht' )

freq_edges,freq_bins = emd.spectra.define_hist_bins(0,10,100)
hht = emd.spectra.hilberthuang( IF, IA, freq_edges )

import matplotlib.pyplot as plt
plt.figure( figsize=(16,8) )
plt.subplot(211,frameon=False)
plt.plot(time_vect,x,'k')
plt.plot(time_vect,imf[:,0]-4,'r')
plt.plot(time_vect,imf[:,1]-8,'g')
plt.plot(time_vect,imf[:,2]-12,'b')
plt.xlim(time_vect[0], time_vect[-1])
plt.grid(True)
plt.subplot(2,1,2)
plt.pcolormesh( time_vect, freq_bins, hht, cmap='ocean_r' )
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (secs)')
plt.grid(True)
plt.show()

sample_rate = 1000

sift_args = {'max_imfs': 9,
             'sd_thresh': 0.05,
             'ret_mask_freq': True,
             'interp_method': 'mono_pchip',
             'mask_freqs': np.array([350,200,70,40,30,7,1,0,0,0,0])/sample_rate,
             'mask_amp_mode': 'ratio_sig',
             'mask_amp': np.ones( (10,))}


imf_mask,mf = emd.sift.mask_sift_specified(lfp_ch_1_10s,**sift_args)
imf = emd.sift.sift( lfp_ch_1_10s )
#emd.plotting.plot_imfs(imf, cmap= True,  scale_y = True)
emd.plotting.plot_imfs(imf_mask, cmap= True, scale_y = True)

lfp_ch_1 = lfp[1,:]
lfp_ch_1_10s = lfp_ch_1[:100000]

plt.plot(imf_mask[:,2:6].sum(axis = 1))

