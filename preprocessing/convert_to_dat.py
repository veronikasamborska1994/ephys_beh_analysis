#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:56:18 2018

@author: behrenslab
"""

# =============================================================================
# Script to covert openephys files to .dat files 
# =============================================================================

channels_m483 = [7,11,6,12,10,5,13,4,8,14,3,15,1,9,2,16,26,22,27,21,23,28,20,29,25,19,30,18,32,24,31,17]

channels_m483_second_box = [39, 43, 38, 44, 42, 37, 45, 36, 40, 46, 35, 47, 33, 41, 34, 48, 58, 54, 59, 53, 55, 60, 52, 61, 57, 51, 62, 50, 64, 56, 63, 49]

channels_m486_m482_m480_m478_second_box = [35, 39, 33, 38, 34, 42, 48, 45, 41, 40, 47, 46, 36, 37, 44, 43, 62, 
                      58, 64, 59, 63, 55, 49, 52, 56, 57, 50, 51, 61, 60, 53, 54]

channels_m486_m482_m480_m478 = [3,7,1,6,2,10,16,13,9,8,15,14,4,5,12,11,30,26,32,27,31,23,17,20,24,25,18,19,29,28,21,22]

channels_m484_m479 = [1,13,2,10,16,6,9,7,3,8,5,27,4,23,14,20,15,25,12,26,11,17,29,31,28,32,21,30,22,24,19,18]

channels__m484_m479_second_box = [33, 45, 34, 42, 48, 38, 41, 39, 35, 40, 37, 59, 36, 55, 46, 52, 47,
       57, 44, 58, 43, 49, 61, 63, 60, 64, 53, 62, 54, 56, 51, 50]

import OpenEphys as op
import os

files_folder = '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/Multiple_Animals/'
files = os.listdir(files_folder)
#
#for file in files:
#    ephys_folder = os.path.join(files_folder, file)
#    print(ephys_folder)
#    if ephys_folder == '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/Multiple_Animals/m480_m479_2018-08-24_11-02-58':
#        continue
#    elif ephys_folder == '/media/behrenslab/My Book/Ephys_Reversal_Learning/data/Ephys 3 Tasks Reversal Learning/Multiple_Animals/m480_m484_2018-09-10_17-27-43':
#        continue
#    elif ephys_folder.find('m480') == 104:
#        if 'm480.dat' not in os.listdir(ephys_folder):
#            op.pack_2(ephys_folder,filename =ephys_folder +'/'+'m480'+'.dat', channels =channels_m486_m482_m480_m478, chprefix = 'CH', dref = 'ave')
#    elif ephys_folder.find('m478') == 104:
#        if 'm478.dat' not in os.listdir(ephys_folder):
#            op.pack_2(ephys_folder, filename =ephys_folder +'/'+'m478'+'.dat', channels =channels_m486_m482_m480_m478, chprefix = 'CH', dref = 'ave')
#    elif ephys_folder.find('m486') == 104:
#        if 'm486.dat' not in os.listdir(ephys_folder):
#            op.pack_2(ephys_folder, filename =ephys_folder +'/'+'m486'+'.dat', channels =channels_m486_m482_m480_m478, chprefix ='CH', dref = 'ave')
#    elif ephys_folder.find('m482') == 104:
#        if 'm482.dat' not in os.listdir(ephys_folder):
#            op.pack_2(ephys_folder, filename =ephys_folder +'/'+'m482'+'.dat', channels =channels_m486_m482_m480_m478, chprefix ='CH', dref = 'ave')
#    elif ephys_folder.find('m484') == 109:
#        if 'm484.dat' not in os.listdir(ephys_folder):
#            op.pack_2(ephys_folder, filename =ephys_folder +'/'+'m484'+'.dat', channels =channels_m484_m479, chprefix ='CH', dref = 'ave')
#    elif ephys_folder.find('m479') == 109:
#        if 'm479.dat' not in os.listdir(ephys_folder):
#            op.pack_2(ephys_folder, filename =ephys_folder +'/'+'m479'+'.dat', channels =channels_m484_m479, chprefix ='CH', dref = 'ave')
#      
        
for file in files:
    ephys_folder = os.path.join(files_folder, file)
    if ephys_folder.find('m480') == 104:
        if 'm480.dat' not in os.listdir(ephys_folder):
            op.pack_2(ephys_folder,filename =ephys_folder +'/'+'m480'+'.dat', channels =channels_m486_m482_m480_m478_second_box, chprefix ='CH', dref = 'ave')


