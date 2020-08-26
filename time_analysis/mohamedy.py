#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:38:25 2020

@author: veronikasamborska
"""
import pickle 
import os
import numpy as np
import data_import as di
import re
import datetime
import copy 
from datetime import datetime
                                         
ephys_path = '/Users/veronikasamborska/Desktop/neurons_mohamady'
beh_path = '/Users/veronikasamborska/Desktop/data_3_tasks_ephys'

HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions = import_code(ephys_path,beh_path, lfp_analyse = 'True')
def import_code(ephys_path,beh_path, lfp_analyse = 'True'):
    subjects_ephys = os.listdir(ephys_path)
    subjects_ephys = [subject for subject in subjects_ephys if not subject.startswith('.')] #Exclude .DS store from subject list
    subjects_beh = os.listdir(beh_path)
    
    m480 = []
    m483 = []
    m479 = []
    m486 = []
    m478 = []
    m481 = []
    m484 = []
    
    for subject_ephys in subjects_ephys: 
        # Go through each animal
        subject_subfolder = ephys_path + '/' + subject_ephys
        subject_sessions = os.listdir(subject_subfolder)
        # List all ephys_sessions
        subject_sessions = [session for session in subject_sessions if not session.startswith('.')] #Exclude .DS store from subject list
        subject_sessions = [session for session in subject_sessions if not session.startswith('LFP')] #Exclude LFP from subject list

        # List all ephys sessions in LFP folder
        if lfp_analyse == 'True':
            lfp_folder = subject_subfolder + '/'  + 'LFP'
            lfp_sessions = os.listdir(lfp_folder)
            lfp_sessions = [s for s in lfp_sessions if not s.startswith('.')] #Exclude .DS store from subject list
    

        for session in subject_sessions:
            match_ephys = re.search(r'\d{4}-\d{2}-\d{2}', session)
            date_ephys = datetime.strptime(match_ephys.group(), '%Y-%m-%d').date()
            date_ephys = match_ephys.group()
            
            for subject in subjects_beh:
                if subject == subject_ephys:
                    subject_beh_subfolder = beh_path + '/' + subject
                    subject_beh_sessions = os.listdir(subject_beh_subfolder)
                    subject_beh_sessions = [session for session in subject_beh_sessions if not session.startswith('.')] #Exclude .DS store from subject list
                    for beh_session in subject_beh_sessions:
                        match_behaviour = re.search(r'\d{4}-\d{2}-\d{2}', beh_session)
                        date_behaviour = datetime.strptime(match_behaviour.group(), '%Y-%m-%d').date()
                        date_behaviour = match_behaviour.group()
                        if date_ephys == date_behaviour:
                            behaviour_path = subject_beh_subfolder +'/'+beh_session
                            behaviour_session = di.Session(behaviour_path)
                            neurons_path = subject_subfolder+'/'+session 
                            neurons = np.load(neurons_path)
                            neurons = neurons[:,~np.isnan(neurons[1,:])]
                            behaviour_session.ephys = neurons
                            
                            # Exclude sessions where ephys software stopped working in the middle of a session or no neurons for some reason 
                            
                            if lfp_analyse == 'True':
                                if (subject_ephys == 'm484'): #or (subject_ephys == 'm484') :#r (subject_ephys == 'm484'):
                                    for s in lfp_sessions: 
                                         match_lfp = re.search(r'\d{4}-\d{2}-\d{2}', s)
                                         date_lfp = datetime.strptime(match_lfp.group(), '%Y-%m-%d').date()
                                         date_lfp = match_lfp.group()
                                         if date_lfp == date_behaviour: 
                                            lfp_path = subject_subfolder+'/'+'LFP''/'+subject_ephys+'_'+ date_lfp + '_1_channel'+ '.npy'
                                            lfp = np.load(lfp_path)
                                            lfp_time = lfp[0,:]
                                            lfp_signal = lfp[1,:]
                                            behaviour_session.lfp = lfp_signal
                                            behaviour_session.lfp_time = lfp_time     
                                            
                                
                            if subject_ephys == 'm480':
                                m480.append(behaviour_session)
                            elif subject_ephys == 'm483':
                                m483.append(behaviour_session)
                            elif subject_ephys == 'm479':
                                m479.append(behaviour_session)
                            elif subject_ephys == 'm486':
                                m486.append(behaviour_session)
                            elif subject_ephys == 'm478':
                                m478.append(behaviour_session)
                            elif subject_ephys == 'm481':
                                m481.append(behaviour_session)
                            elif subject_ephys == 'm484':
                                m484.append(behaviour_session)
                       
                                
    HP = m484 + m479 + m483
    PFC = m478 + m486 + m480 + m481
    all_sessions = m484  + m479 + m483 + m478 + m486 + m480 + m481
    return HP,PFC, m484, m479, m483, m478, m486, m480, m481, all_sessions
