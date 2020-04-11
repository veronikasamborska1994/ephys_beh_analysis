#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:54:09 2020

@author: veronikasamborska
"""


from scipy.fftpack import rfft, irfft
import numpy as np
import time_hierarchies as th
from palettable import wesanderson as wes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter as gaus
from scipy.fftpack import fft, ifft
import itertools
from numba import jit
from tqdm import tqdm
import seaborn as sns

def simulate_plot():
    
    ind_above_chance_HP_fft,percentage_HP_fft = plot_fft(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20,\
                        region = 'HP_shuffle_fft_real.pdf', plot = True, perm = 1000)
   
    ind_above_chance_PFC_fft,percentage_PFC_fft  = plot_fft(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20,\
                        region = 'PFC_shuffle_fft_real.pdf', plot = True, perm = 1000)
        
        
    fft_hist(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20,index_above_chance = ind_above_chance_PFC_fft, title = 'PFC', subplot = 1)
   
    fft_hist(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20,index_above_chance = ind_above_chance_HP_fft, title = 'HP',subplot = 2)
        
    
# =============================================================================
#  Correlation Simulations with different smoothing critera (FFT vs shuffle)
# =============================================================================
      
    # Shuffle simulations
        
    # No smooth
        
    ind, percent_no_smooth_shuffle_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 0, \
                                    gaussian = False, l = 0, corr_permute = True)
       
    # ind, percent_no_smooth_half,mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = False,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_shuffle_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 1, \
                                    gaussian = True, l = 0, corr_permute = True)
        
    # ind, percent_1_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = False, std = 1, gaussian = True,\
    #                          l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_shuffle_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 2, \
                                    gaussian = True, l = 0, corr_permute = True)
       
    # ind, percent_2_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = False, std = 2, gaussian = True,\
    #                          l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_shuffle_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 3, \
                                    gaussian = True, l = 0, corr_permute = True)
       
    # ind, percent_3_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = False, std = 3, gaussian = True,\
    #                          l = 0, corr_permute = True)


     # FFT simulations
        
    # No smooth
        
    ind, percent_no_smooth_fft_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 0, \
                                    gaussian = False, l = 0, corr_permute = True)
       
    # ind, percent_no_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = True,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_fft_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 1, \
                                    gaussian = True, l = 0, corr_permute = True)
       
    # ind, percent_1_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = True, std = 1, gaussian = True,
    #                       l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_fft_corr, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 2, \
                                    gaussian = True, l = 0, corr_permute = True)
        
    # ind, percent_2_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = True, std = 2, gaussian = True,
    #                       l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_fft_corr, mean = simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 3, \
                                    gaussian = True, l = 0, corr_permute = True)
       
    # ind, percent_3_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = True, std = 3, gaussian = True,
    #                       l = 0, corr_permute = True)

   
    
# =============================================================================
#  Peak-Troph Simulations with different smoothing critera (FFT vs shuffle)
# =============================================================================
 
    # Shuffle simulations
        
    # No smooth
        
    ind, percent_no_smooth, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 0, \
                                    gaussian = False, l = 0, corr_permute = False)
       
    # ind, percent_no_smooth_half,mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = False,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 1, \
                                    gaussian = True, l = 0, corr_permute = False)
       
    # ind, percent_1_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = False, std = 1, gaussian = True,\
    #                          l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 2, \
                                    gaussian = True, l = 0, corr_permute = False)
        
    # ind, percent_2_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = False, std = 2, gaussian = True,\
    #                          l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 3, \
                                    gaussian = True, l = 0, corr_permute = False)
       
       
    # ind, percent_3_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = False, std = 3, gaussian = True,\
    #                          l = 0, corr_permute = True)



     # FFT simulations
        
    # No smooth
        
    ind, percent_no_smooth_fft, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 0, \
                                    gaussian = False, l = 0, corr_permute = False)
       
       
    # ind, percent_no_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = True,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_fft, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 1, \
                                    gaussian = True, l = 0, corr_permute = False)
       
       
    # ind, percent_1_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = True, std = 1, gaussian = True,
    #                       l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_fft, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 2, \
                                    gaussian = True, l = 0, corr_permute = False)
       
       
    # ind, percent_2_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = True, std = 2, gaussian = True,
    #                       l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_fft, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 3, \
                                    gaussian = True, l = 0, corr_permute = False)
       
       
    # ind, percent_3_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = True, std = 3, gaussian = True,
    #                       l = 0, corr_permute = True)
        

  # =============================================================================
#  Simulate longer arrays with different smoothing critera (FFT peak- troph)
# =============================================================================
   
    # Shuffle simulations
        
    # No smooth
        
    ind, percent_no_smooth_long, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 0, \
                                    gaussian = False, l = -10, corr_permute = False)
       
    # ind, percent_no_smooth_half,mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = False,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_long, mean = simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 1, \
                                    gaussian = True, l = -10, corr_permute = False)
    # ind, percent_1_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = False, std = 1, gaussian = True,\
    #                          l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_long, mean = simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 2, \
                                    gaussian = True, l = -10, corr_permute = False)
       
    # ind, percent_2_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = False, std = 2, gaussian = True,\
    #                          l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_long, mean =   simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = False, std = 3, \
                                    gaussian = True, l = -10, corr_permute = False)
       
    # ind, percent_3_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = False, std = 3, gaussian = True,\
    #                          l = 0, corr_permute = True)


    
    # FFT simulations
        
    # No smooth
        
    ind, percent_no_smooth_fft_long, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 0, \
                                    gaussian = False, l = -10, corr_permute = False)
       
    # ind, percent_no_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = True,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_fft_long, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 1, \
                                    gaussian = True, l = -10, corr_permute = False)
       
    # ind, percent_1_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = True, std = 1, gaussian = True,
    #                       l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_fft_long, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 2, \
                                    gaussian = True, l = -10, corr_permute = False)
       
    # ind, percent_2_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = True, std = 2, gaussian = True,
    #                       l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_fft_long, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 500, make_half_real = False,
                                    ffs_shuffle = True, std = 3, \
                                    gaussian = True, l = -10, corr_permute = False)
       
    # ind, percent_3_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = True, std = 3, gaussian = True,
    #                       l = 0, corr_permute = True)
        



# =============================================================================
#  Simulate shorter arrays with different smoothing critera (FFT peak- troph)
# =============================================================================
  
    # Shuffle simulations
        
    # No smooth
        
    ind, percent_no_smooth_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = False, std = 0, \
                                    gaussian = False, l = 10, corr_permute = False)
       
    # ind, percent_no_smooth_half,mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = False,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = False, std = 1, \
                                    gaussian = True, l = 10, corr_permute = False)
       
    # ind, percent_1_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = False, std = 1, gaussian = True,\
    #                          l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = False, std = 2, \
                                    gaussian = True, l = 10, corr_permute = False)
       
    # ind, percent_2_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = False, std = 2, gaussian = True,\
    #                          l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = False, std = 3, \
                                    gaussian = True, l = 10, corr_permute = False)
       
    # ind, percent_3_smooth_half, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = False, std = 3, gaussian = True,\
    #                          l = 0, corr_permute = True)


     # FFT simulations
        
    # No smooth
        
    ind, percent_no_smooth_fft_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = True, std = 3, \
                                    gaussian = False, l = 10, corr_permute = False)
       
    # ind, percent_no_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'No_smooth_half.pdf', ffs_shuffle = True,std = 2, gaussian = False,\
    #                          l = 0, corr_permute = True)

    # Smooth 1 
    
    ind, percent_1_smooth_fft_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = True, std = 2, \
                                    gaussian = True, l = 10, corr_permute = False)
       
    # ind, percent_1_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_1_half.pdf', ffs_shuffle = True, std = 1, gaussian = True,
    #                       l = 0, corr_permute = True)

    # Smooth 2
            
    ind, percent_2_smooth_fft_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = True, std = 2, \
                                    gaussian = True, l = 10, corr_permute = False)
       
    # ind, percent_2_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_2_half.pdf', ffs_shuffle = True, std = 2, gaussian = True,
    #                       l = 0, corr_permute = True)


    # Smooth 3
            
    ind, percent_3_smooth_fft_short, mean =  simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
                                    experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 20, perm = 1000, make_half_real = False,
                                    ffs_shuffle = True, std = 3, \
                                    gaussian = True, l = 10, corr_permute = False)
       
    # ind, percent_3_smooth_half_fft, mean =  plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
    #                     beh = True, block = False, raw_data = True, HP = False,start = 0, end = 20,\
    #                     region = 'PFC', plot = True, perm = 1000, make_half_real = True, title = 'Smooth_3_half.pdf', ffs_shuffle = True, std = 3, gaussian = True,
    #                       l = 0, corr_permute = True)


# =============================================================================
#     Plotting   
# =============================================================================
      
    plt.figure()
    grand_bud = wes.GrandBudapest1_4.mpl_colors
    plt.subplot(221)
    plt.bar(np.arange(8), [percent_no_smooth_shuffle_corr, percent_1_smooth_shuffle_corr,percent_2_smooth_shuffle_corr, percent_3_smooth_shuffle_corr,\
                           percent_no_smooth_fft_corr, percent_1_smooth_fft_corr,percent_2_smooth_fft_corr,percent_3_smooth_fft_corr],\
                            color = [grand_bud[2],grand_bud[2],grand_bud[2],grand_bud[2],\
                                     grand_bud[1],grand_bud[1],grand_bud[1],grand_bud[1]],
                            tick_label = ['Shuf No Smooth','Shuf Smooth 1', 'Shuf 2 Smooth', 'Shuf 3 Smooth',\
                                          'FFT No Smooth','FFT Smooth 1', 'FFT 2 Smooth', 'FFT 3 Smooth'], alpha = 0.5)
    plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
    plt.title('Correlations Simulations FFT and Shuffle')
    plt.ylabel('% Significant')
        
    
    
    plt.subplot(222)
    grand_bud = wes.GrandBudapest1_4.mpl_colors

    plt.bar(np.arange(8), [percent_no_smooth, percent_1_smooth,percent_2_smooth, percent_3_smooth,\
                           percent_no_smooth_fft, percent_2_smooth_fft,percent_2_smooth_fft,percent_3_smooth_fft],\
                            color = [grand_bud[2],grand_bud[2],grand_bud[2],grand_bud[2],\
                                     grand_bud[1],grand_bud[1],grand_bud[1],grand_bud[1]],
                            tick_label = ['Shuf No Smooth','Shuf Smooth 1', 'Shuf 2 Smooth', 'Shuf 3 Smooth',\
                                          'FFT No Smooth','FFT Smooth 1', 'FFT 2 Smooth', 'FFT 3 Smooth'], alpha = 0.5)
        
    plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
    
    plt.title('Peak-Troph Simulations FFT and Shuffle')
    plt.ylabel('% Significant')
        
        
    plt.subplot(223)

    grand_bud = wes.GrandBudapest1_4.mpl_colors

    plt.bar(np.arange(8), [percent_no_smooth_short, percent_1_smooth_short,percent_2_smooth_short, percent_3_smooth_short,\
                           percent_no_smooth_fft_short, percent_1_smooth_fft_short,percent_2_smooth_fft_short,percent_3_smooth_fft_short],\
                            color = [grand_bud[2],grand_bud[2],grand_bud[2],grand_bud[2],\
                                     grand_bud[1],grand_bud[1],grand_bud[1],grand_bud[1]],
                            tick_label = ['Shuf No Smooth','Shuf Smooth 1', 'Shuf 2 Smooth', 'Shuf 3 Smooth',\
                                          'FFT No Smooth','FFT Smooth 1', 'FFT 2 Smooth', 'FFT 3 Smooth'], alpha = 0.5)
    plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')    
    plt.title('Peak-Troph Simulations FFT and Shuffle Shorter Arrays')
    plt.ylabel('% Significant')
    
    
    plt.subplot(224)
    grand_bud = wes.GrandBudapest1_4.mpl_colors

    plt.bar(np.arange(8), [percent_no_smooth_long, percent_1_smooth_long,percent_2_smooth_long, percent_3_smooth_long,\
                           percent_no_smooth_fft_long, percent_1_smooth_fft_long,percent_2_smooth_fft_long,percent_3_smooth_fft_long],\
                            color = [grand_bud[2],grand_bud[2],grand_bud[2],grand_bud[2],\
                                     grand_bud[1],grand_bud[1],grand_bud[1],grand_bud[1]],
                            tick_label = ['Shuf No Smooth','Shuf Smooth 1', 'Shuf 2 Smooth', 'Shuf 3 Smooth',\
                                          'FFT No Smooth','FFT Smooth 1', 'FFT 2 Smooth', 'FFT 3 Smooth'], alpha = 0.5)
    plt.title('Peak-Troph Simulations FFT and Shuffle Longer Arrays')
    plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
    plt.ylabel('% Significant')
    sns.despine()
    plt.tight_layout()
    

    

def fft_hist(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, index_above_chance = [1,2,3], title = 'HP', subplot = 1):
    
    a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
     a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
     a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
     block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
     block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
     block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  th.hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
  
    neurons = 0
    peak_hist = []
    troph_hist = []
    for i,aa in tqdm(enumerate(a_a_matrix_t_1_list)):
        switch = int(a_a_matrix_t_1_list.shape[1]/2)

        neurons+=1
        if neurons in index_above_chance:
        
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
                
            blocks_all_tasks =  np.mean([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)
            peak = np.max(blocks_all_tasks)
            troph = np.min(blocks_all_tasks)

            peak_hist.append(np.where(blocks_all_tasks == peak)[0][0])
            troph_hist.append(np.where(blocks_all_tasks == troph)[0][0])

    plt.ion()
    grand_bud = wes.GrandBudapest1_4.mpl_colors
    plt.figure(1651)
    plt.subplot(2,2,subplot)
    plt.hist(troph_hist,color = grand_bud[3],alpha = 0.5, label = title + ' ' + 'Peaks')
    plt.legend()
    plt.subplot(2,2,subplot+2)
    plt.hist(peak_hist,color = grand_bud[2],alpha = 0.5, label = title + ' ' + 'Trophs')
    plt.legend()

   
    
def plot_simulation(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True,start = 0, end = 20,\
                        region = 'HP', plot = True, perm = 5, make_half_real = False, title = 'Simulate_FFT.pdf', ffs_shuffle = False, std = 2, gaussian = False):
     
     ind_above_chance,percentage,neuron_blocks,perm_mean = simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm, make_half_real = make_half_real, ffs_shuffle = ffs_shuffle, std = std, gaussian = gaussian)
    
     pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ title)
     plt.ioff()
     count = 0
     plot_new = True
     neuron_count = 0
     plt.figure()
     isl_1 =  wes.Moonrise1_5.mpl_colors
   
     for i,b in enumerate(neuron_blocks): 
        count +=1
        neuron_count += 1

        if count == 12:
            plot_new = True
            count = 1
        if plot_new == True:
            pdf.savefig()      
            plt.clf()
            plt.figure()
            plot_new = False
           
        plt.subplot(3,4,count)
       
        blocks_all_tasks =  np.mean(b,0)
        std_blocks_all_tasks =  np.std(b,0)/(np.sqrt(12))
        plt.plot(blocks_all_tasks, color = isl_1[0], label = 'All Tasks Block Time')
        plt.fill_between(np.arange(len(blocks_all_tasks)), blocks_all_tasks-std_blocks_all_tasks, blocks_all_tasks+std_blocks_all_tasks, alpha = 0.2, color = isl_1[0])
        plt.tight_layout()
       
        if count == 1:
            plt.legend()
        if  (neuron_count-1) in ind_above_chance:
            plt.title('Significant')
        else:
            plt.title(str(count))
    
        if count == 1:
            plt.legend()
      
     pdf.savefig()      
     pdf.close()
         
     return ind_above_chance,percentage,perm_mean
 
#@jit 
def simulate_data(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, perm = True, make_half_real = True, ffs_shuffle = True, std = 2, \
         gaussian = False, l = 0, corr_permute = False):
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
    
     for i,aa in tqdm(enumerate(a_a_matrix_t_1_list)):
         neurons+=1
         switch = int(a_a_matrix_t_1_list.shape[1]/2)
        
         size = a_a_matrix_t_1_list[i][:switch].shape
          
         a_1_mean, a1_st = [np.mean(a_a_matrix_t_1_list[i][:switch]), np.std(a_a_matrix_t_1_list[i][:switch])]
         a_1 = np.random.normal(a_1_mean,a1_st, size)

         a_2_mean, a2_st = [np.mean(a_a_matrix_t_1_list[i][switch:]), np.std(a_a_matrix_t_1_list[i][switch:])]
         a_2 = np.random.normal(a_2_mean,a2_st, size)

         a_3_mean, a3_st = [np.mean(a_a_matrix_t_2_list[i][:switch]), np.std(a_a_matrix_t_2_list[i][:switch])]
         a_3 = np.random.normal(a_3_mean,a3_st, size)

         a_4_mean, a4_st = [np.mean(a_a_matrix_t_2_list[i][switch:]), np.std(a_a_matrix_t_2_list[i][switch:])]
         a_4 = np.random.normal(a_4_mean,a4_st, size)

         a_5_mean, a5_st = [np.mean(a_a_matrix_t_3_list[i][:switch]), np.std(a_a_matrix_t_3_list[i][:switch])]
         a_5 = np.random.normal(a_5_mean,a5_st, size)

         a_6_mean, a6_st = [np.mean(a_a_matrix_t_3_list[i][switch:]), np.std(a_a_matrix_t_3_list[i][switch:])]
         a_6 = np.random.normal(a_6_mean,a6_st, size)

         
         b_1_mean, b1_st = [np.mean(b_b_matrix_t_1_list[i][:switch]), np.std(b_b_matrix_t_1_list[i][:switch])]
         b_1 = np.random.normal(b_1_mean,b1_st, size)

         b_2_mean, b2_st = [np.mean(b_b_matrix_t_1_list[i][switch:]), np.std(b_b_matrix_t_1_list[i][switch:])]
         b_2 = np.random.normal(b_2_mean,b2_st, size)

         b_3_mean, b3_st = [np.mean(b_b_matrix_t_2_list[i][:switch]), np.std(b_b_matrix_t_2_list[i][:switch])]
         b_3 = np.random.normal(b_3_mean,b3_st, size)

         b_4_mean, b4_st = [np.mean(b_b_matrix_t_2_list[i][switch:]), np.std(b_b_matrix_t_2_list[i][switch:])]
         b_4 = np.random.normal(b_4_mean,b4_st, size)

         b_5_mean, b5_st = [np.mean(b_b_matrix_t_3_list[i][:switch]), np.std(b_b_matrix_t_3_list[i][:switch])]
         b_5 = np.random.normal(b_5_mean,b5_st, size)

         b_6_mean, b6_st = [np.mean(b_b_matrix_t_3_list[i][switch:]), np.std(b_b_matrix_t_3_list[i][switch:])]
         b_6 = np.random.normal(b_6_mean,b6_st, size)
         
         if gaussian == True:
             a_1 = gaus(np.random.normal(a_1_mean,a1_st, size[0]-l),std)

             a_2 = gaus(np.random.normal(a_2_mean,a2_st, size[0]-l),std)

             a_3 = gaus(np.random.normal(a_3_mean,a3_st, size[0]-l),std)

             a_4 = gaus(np.random.normal(a_4_mean,a4_st, size[0]-l),std)
             
             a_5 = gaus(np.random.normal(a_5_mean, a5_st, size[0]-l),std)
           
             a_6 = gaus(np.random.normal(a_6_mean,a6_st, size[0]-l),std)

             b_1 = gaus(np.random.normal(b_1_mean,b1_st, size[0]-l),std)

             b_2 = gaus(np.random.normal(b_2_mean,b2_st, size[0]-l),std)

             b_3 = gaus(np.random.normal(b_3_mean,b3_st, size[0]-l),std)

             b_4 = gaus(np.random.normal(b_4_mean, b4_st, size[0]-l),std)
             
             b_5 = gaus(np.random.normal(b_5_mean, b5_st, size[0]-l),std)
           
             b_6 = gaus(np.random.normal(b_6_mean,b6_st, size[0]-l),std)
            # b_6 = b_6[5:-5]
             
             if l == -10:
                 a_1 = a_1[5:-5]
                 a_2 = a_2[5:-5]
                 a_3 = a_3[5:-5]
                 a_4 = a_4[5:-5]
                 a_5 = a_5[5:-5]
                 a_6 = a_6[5:-5]
                 b_1 = b_1[5:-5]
                 b_2 = b_2[5:-5]
                 b_3 = b_3[5:-5]
                 b_4 = b_4[5:-5]
                 b_5 = b_5[5:-5]
                 b_6 = b_6[5:-5]

                 
           
         if make_half_real == True:
            if i < (int(a_a_matrix_t_1_list.shape[0]/2)+100):
                 
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
            
         
         blocks =  [a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6]
         if corr_permute == True:
             index = itertools.permutations(np.arange(12), 2)
             correlatons = []
             for i in index:
                 correlation = np.corrcoef(blocks[i[0]],blocks[i[1]])[0,1]
                 if np.isnan(correlation):
                     correlation = 0        
                 correlatons.append(correlation)
             distance = np.mean(correlatons)
        
               
         else:   
             blocks_all_tasks =  np.mean([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)
             peak = np.max(blocks_all_tasks)
             troph = np.min(blocks_all_tasks)
    
             std_blocks_all_tasks =  np.std([a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6],0)/(np.sqrt(12))
    
            
             distance = abs(peak-troph)/np.max(std_blocks_all_tasks)
    
         distance_mean_neuron.append(distance)
        
         diff_perm = []
        
        
         if perm:
             for p in range(perm):
                 if ffs_shuffle == True:
                     shuffle_list = []
                     for b in blocks:                      
                         shuffle = phaseScrambleTS(b)                         
                         shuffle_list.append(shuffle) 
                          
                     blocks_all_tasks_perm =  np.mean(shuffle_list,0)
                     std_blocks_all_tasks_perm =  np.std(shuffle_list,0)/(np.sqrt(12))
                     
                     if corr_permute == True:
                        index = itertools.permutations(np.arange(12), 2)
                        correlatons = []
                        for i in index:
                            correlation = np.corrcoef(shuffle_list[i[0]],shuffle_list[i[1]])[0,1]
                            if np.isnan(correlation):
                                correlation = 0        
                            correlatons.append(correlation)
                        distance_perm = np.mean(correlatons)
                     else:
                        peak_perm = np.max(blocks_all_tasks_perm)
                        troph_perm = np.min(blocks_all_tasks_perm)
                        distance_perm = abs(peak_perm-troph_perm)/np.max(std_blocks_all_tasks_perm)
                     
                        
                    
                     
                 else:
                    switch = len(a_1)
                    perm_array_rolled = np.hstack((a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6))
                    np.random.shuffle(perm_array_rolled)
                    a_1_perm = perm_array_rolled[:switch]
                    a_2_perm = perm_array_rolled[switch:switch*2]
                    a_3_perm = perm_array_rolled[switch*2:switch*3]
                    a_4_perm = perm_array_rolled[switch*3:switch*4]
                    a_5_perm = perm_array_rolled[switch*4:switch*5]
                    a_6_perm = perm_array_rolled[switch*5:switch*6]
                  
                    b_1_perm = perm_array_rolled[switch*6:switch*7]
                    b_2_perm = perm_array_rolled[switch*7:switch*8]
                    b_3_perm = perm_array_rolled[switch*8:switch*9]
                    b_4_perm = perm_array_rolled[switch*9:switch*10]
                    b_5_perm = perm_array_rolled[switch*10:switch*11]
                    b_6_perm = perm_array_rolled[switch*11:]
                    blocks_shuffle_perm = [a_1_perm,a_2_perm,a_3_perm,a_4_perm,a_5_perm,a_6_perm,b_1_perm,b_2_perm,b_3_perm,b_4_perm,b_5_perm,b_6_perm]
                    blocks_all_tasks_perm =  np.mean([a_1_perm,a_2_perm,a_3_perm,a_4_perm,a_5_perm,a_6_perm,b_1_perm,b_2_perm,b_3_perm,b_4_perm,b_5_perm,b_6_perm],0)
                    std_blocks_all_tasks_perm =  np.std([a_1_perm,a_2_perm,a_3_perm,a_4_perm,a_5_perm,a_6_perm,b_1_perm,b_2_perm,b_3_perm,b_4_perm,b_5_perm,b_6_perm],0)/(np.sqrt(12))
                    
                    if corr_permute == True:
                        index = itertools.permutations(np.arange(12), 2)
                        correlatons = []
                        for i in index:
                            correlation = np.corrcoef(blocks_shuffle_perm[i[0]],blocks_shuffle_perm[i[1]])[0,1]
                            if np.isnan(correlation):
                                correlation = 0        
                            correlatons.append(correlation)
                        distance_perm = np.abs(np.mean(correlatons))
                    else:
                        peak_perm = np.max(blocks_all_tasks_perm)
                        troph_perm = np.min(blocks_all_tasks_perm)
                    
                        distance_perm = abs(peak_perm-troph_perm)/np.max(std_blocks_all_tasks_perm)
    
                 diff_perm.append(distance_perm)
                        
         perm_mean.append(np.percentile(diff_perm,95))
   
   
     ind_above_chance = np.where(np.abs(np.array(distance_mean_neuron)) > np.abs(np.array(perm_mean)))[0]
     percentage = (len(ind_above_chance)/neurons)*100
      
     return ind_above_chance,percentage,perm_mean
 

def simulate_all_neurons():
    
    ind_above_chance = simulate_data_optimise(data_HP, data_PFC,experiment_aligned_HP, \
      experiment_aligned_PFC, beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20, perm = 1000, ffs_shuffle = False, std = 2, \
          gaussian = True)
        
# =============================================================================
# @jit  Making Simulations faster??

def simulate_data_optimise(data_HP, data_PFC,experiment_aligned_HP, \
      experiment_aligned_PFC, beh = True, block = False, raw_data = False, HP = True, start = 0, end = 62, perm = True, ffs_shuffle = True, std = 2, \
          gaussian = False):
      a_a_matrix_t_1_list, b_b_matrix_t_1_list,\
      a_a_matrix_t_2_list, b_b_matrix_t_2_list,\
      a_a_matrix_t_3_list, b_b_matrix_t_3_list,\
      block_1_t1, block_2_t1, block_3_t1,block_4_t1,\
      block_1_t2, block_2_t2, block_3_t2, block_4_t2,\
      block_1_t3, block_2_t3, block_3_t3, block_4_t3 =  th.hieararchies_extract(data_HP, data_PFC,experiment_aligned_HP, \
      experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end)
    
      switch = int(a_a_matrix_t_1_list.shape[1]/2)
    
      size = a_a_matrix_t_1_list[:,:switch].shape
      
      a_1_mean, a1_st = [np.mean(a_a_matrix_t_1_list[:,:switch],1), np.std(a_a_matrix_t_1_list[:,:switch],1)]
      a_1 = np.transpose(np.random.normal(a_1_mean,a1_st, (size[1], size[0])))

      a_2_mean, a2_st = [np.mean(a_a_matrix_t_1_list[:,switch:]), np.std(a_a_matrix_t_1_list[:,switch:])]
      a_2 = np.transpose(np.random.normal(a_2_mean,a2_st, (size[1], size[0])))

      a_3_mean, a3_st = [np.mean(a_a_matrix_t_2_list[:,:switch]), np.std(a_a_matrix_t_2_list[:,:switch])]
      a_3 = np.transpose(np.random.normal(a_3_mean,a3_st, (size[1], size[0])))

      a_4_mean, a4_st = [np.mean(a_a_matrix_t_2_list[:,switch:]), np.std(a_a_matrix_t_2_list[:,switch:])]
      a_4 = np.transpose(np.random.normal(a_4_mean,a4_st, (size[1], size[0])))

      a_5_mean, a5_st = [np.mean(a_a_matrix_t_3_list[:,:switch]), np.std(a_a_matrix_t_3_list[:,:switch])]
      a_5 = np.transpose(np.random.normal(a_5_mean,a5_st, (size[1], size[0])))

      a_6_mean, a6_st = [np.mean(a_a_matrix_t_3_list[:,switch:]), np.std(a_a_matrix_t_3_list[:,switch:])]
      a_6 = np.transpose(np.random.normal(a_6_mean,a6_st, (size[1], size[0])))

    
      b_1_mean, b1_st = [np.mean(b_b_matrix_t_1_list[:,:switch]), np.std(b_b_matrix_t_1_list[:,:switch])]
      b_1 = np.transpose(np.random.normal(b_1_mean,b1_st, (size[1], size[0])))
  
      b_2_mean, b2_st = [np.mean(b_b_matrix_t_1_list[:,switch:]), np.std(b_b_matrix_t_1_list[:,switch:])]
      b_2 = np.transpose(np.random.normal(b_2_mean,b2_st, (size[1], size[0])))
  
      b_3_mean, b3_st = [np.mean(b_b_matrix_t_2_list[:,:switch]), np.std(b_b_matrix_t_2_list[:,:switch])]
      b_3 = np.transpose(np.random.normal(b_3_mean,b3_st, (size[1], size[0])))

      b_4_mean, b4_st = [np.mean(b_b_matrix_t_2_list[:,switch:]), np.std(b_b_matrix_t_2_list[:,switch:])]
      b_4 = np.transpose(np.random.normal(b_4_mean,b4_st, (size[1], size[0])))
  
      b_5_mean, b5_st = [np.mean(b_b_matrix_t_3_list[:,:switch]), np.std(b_b_matrix_t_3_list[:,:switch])]
      b_5 = np.transpose(np.random.normal(b_5_mean,b5_st, (size[1], size[0])))

      b_6_mean, b6_st = [np.mean(b_b_matrix_t_3_list[:,switch:]), np.std(b_b_matrix_t_3_list[:,switch:])]
      b_6 = np.transpose(np.random.normal(b_6_mean,b6_st, (size[1], size[0])))
    
      if gaussian == True:
        a_1 = gaus(a_1,std)

        a_2 = gaus(a_2,std)

        a_3 = gaus(a_3,std)

        a_4 = gaus(a_4,std)
        
        a_5 = gaus(a_5,std)
      
        a_6 = gaus(a_6,std)

        b_1 = gaus(b_1,std)

        b_2 = gaus(b_2,std)

        b_3 = gaus(b_3,std)

        b_4 = gaus(b_4,std)
        
        b_5 = gaus(b_5,std)
      
        b_6 = gaus(b_6,std)
      
    
      if raw_data == True:
        
        a_1 = (a_1.T - np.mean(a_1,1)).T
        a_2 = (a_2.T - np.mean(a_2,1)).T
        a_3 = (a_3.T- np.mean(a_3,1)).T
        a_4 = (a_4.T - np.mean(a_4,1)).T
        a_5 = (a_5.T - np.mean(a_5,1)).T
        a_6 = (a_6.T - np.mean(a_6,1)).T

        b_1 = (b_1.T - np.mean(b_1,1)).T
        b_2 = (b_2.T - np.mean(b_2,1)).T
        b_3 = (b_3.T - np.mean(b_3,1)).T
        b_4 = (b_4.T - np.mean(b_4,1)).T
        b_5 = (b_5.T - np.mean(b_5,1)).T
        b_6 = (b_6.T - np.mean(b_6,1)).T
        
    
      blocks =  [a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6]
      
      index = itertools.permutations(np.arange(12), 2)
      correlation = []
      for i in index:    
          correlation.append(np.diag((np.corrcoef(blocks[i[0]].T,blocks[i[1]].T)),switch))
      distance = np.mean(correlation,0)
        
      diff_perm = []
      perm_mean = []

      if perm:
          for p in range(perm):
              if ffs_shuffle == True:
                  neuron_block_shuffle = []
                  for b in blocks:  
                      shuffle_list = []
                      for n in b: 
                          shuffle = phaseScrambleTS(n)                         
                          shuffle_list.append(shuffle[:-1]) 
                      neuron_block_shuffle.append(np.asarray(shuffle_list)) 
                      
                  neuron_block_shuffle = np.asarray(neuron_block_shuffle)
                  index = itertools.permutations(np.arange(12), 2)
                  correlation = []
                  for i in index:
                      correlation.append(np.diag((np.corrcoef(neuron_block_shuffle[i[0]].T,neuron_block_shuffle[i[1]].T)),switch))
                  distance_perm = np.mean(correlation,0)
                
                      
              else:
                  switch = a_1.shape[1]
                  perm_array_rolled = np.hstack((a_1,a_2,a_3,a_4,a_5,a_6,b_1,b_2,b_3,b_4,b_5,b_6))
                  np.random.shuffle(perm_array_rolled)
                  a_1_perm = perm_array_rolled[:,:switch]
                  a_2_perm = perm_array_rolled[:,switch:switch*2]
                  a_3_perm = perm_array_rolled[:,switch*2:switch*3]
                  a_4_perm = perm_array_rolled[:,switch*3:switch*4]
                  a_5_perm = perm_array_rolled[:,switch*4:switch*5]
                  a_6_perm = perm_array_rolled[:,switch*5:switch*6]
                  
                  b_1_perm = perm_array_rolled[:,switch*6:switch*7]
                  b_2_perm = perm_array_rolled[:,switch*7:switch*8]
                  b_3_perm = perm_array_rolled[:,switch*8:switch*9]
                  b_4_perm = perm_array_rolled[:,switch*9:switch*10]
                  b_5_perm = perm_array_rolled[:,switch*10:switch*11]
                  b_6_perm = perm_array_rolled[:,switch*11:]
              
                  blocks_all_tasks_perm =  [a_1_perm,a_2_perm,a_3_perm,a_4_perm,a_5_perm,a_6_perm,b_1_perm,b_2_perm,b_3_perm,b_4_perm,b_5_perm,b_6_perm]
                    
                  index = itertools.permutations(np.arange(12), 2)
                  correlation = []
                  for i in index:    
                      correlation.append(np.diag((np.corrcoef(blocks_all_tasks_perm[i[0]].T,blocks_all_tasks_perm[i[1]].T)),switch))
                  distance_perm = np.mean(correlation,0)
                   
              diff_perm.append(distance_perm)
                        
          perm_mean.append(np.percentile(diff_perm,95,0))
    
    
      ind_above_chance = np.abs(np.array(distance)) > np.abs(np.array(perm_mean))[0]*1
      ind_above_chance  = np.where(ind_above_chance*1 == 1)[0]
      return ind_above_chance
 

def phaseScrambleTS_2(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = rfft(ts)
    # rfft returns real and imaginary components in adjacent elements of a real array
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    # use broadcasting and ravel to interleave the real and imaginary components. 
    # The first and last elements in the fourier array don't have any phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr

def phaseScrambleTS(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = fft(ts)
    pow_fs = np.abs(fs) ** 2.
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = ifft(fsrp)
    return tsr

def perm_test_time_fft(data_HP, data_PFC,experiment_aligned_HP, \
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
                    shuffle = phaseScrambleTS(b)
                    shuffle_list.append(shuffle)
                
                if perm == 1 and neurons ==1: 
                    plt.subplot(1,2,1)
                    for i,b in blocks:
                        plt.figure(b,color = cs[i])
                    plt.subplot(1,2,2)
                    for i,sh in shuffle_list:
                        plt.figure(sh, color = cs[i])
                        
                blocks_all_tasks_perm =  np.mean(shuffle_list,0)
                std_blocks_all_tasks_perm =  np.std(shuffle_list,0)/(np.sqrt(12))


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


def plot():
    
    ind_above_chance_HP_fft,percentage_HP_fft = plot_fft(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = True, start = 0, end = 20,\
                        region = 'HP shuffle fft', plot = True, perm = 1000)
   
    ind_above_chance_PFC_fft,percentage_PFC_fft  = plot_fft(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
                        beh = True, block = False, raw_data = True, HP = False, start = 0, end = 20,\
                        region = 'PFC shuffle fft', plot = True, perm = 1000)
        
        
   
          
def plot_fft(data_HP, data_PFC,experiment_aligned_HP, experiment_aligned_PFC,\
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
         ind_above_chance,percentage = perm_test_time_fft(data_HP, data_PFC,experiment_aligned_HP, \
     experiment_aligned_PFC, beh = beh, block = block, raw_data = raw_data, HP = HP, start = start, end = end, perm = perm)
        
     elif HP == False:
        a_list,  b_list, rew_list,  no_rew_list = th.find_rewards_choices(data_PFC, experiment_aligned_PFC)
        ind_above_chance,percentage = perm_test_time_fft(data_HP, data_PFC,experiment_aligned_HP, \
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
     
     pdf = PdfPages('/Users/veronikasamborska/Desktop/time/'+ region)
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
               
            a_1 = a_s_1_1[i]
            a_2 = a_s_1_2[i]
            a_3 = a_s_2_1[i]
            a_4 = a_s_2_2[i]
            a_5 = a_s_3_1[i]
            a_6 = a_s_3_2[i]
   
            b_1 = b_s_1_1[i]
            b_2 = b_s_1_2[i]
            b_3 = b_s_2_1[i]
            b_4 = b_s_2_2[i]
            b_5 = b_s_3_1[i]
            b_6 = b_s_3_2[i]
          
    
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
         
                
     
                     
             