#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:49:37 2018

@author: veronikasamborska
"""

def debugging(HP,experiment_aligned_HP):
    

    for s,session in zip(HP_to_compare, experiment_aligned_HP_to_compare):     
        if session.file_name == 'm483-2018-06-14-172430.txt':
            raw_spikes = s.ephys
            print(raw_spikes)