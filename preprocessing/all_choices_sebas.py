#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:19:03 2019

@author: veronikasamborska
"""
import data_import as di
import numpy as np
import pickle


## Exract all choices the animal has ever made

all_sesssions_ever = '/Users/veronikasamborska/Desktop/all_sessions_all_animals'


behaviour_sessions_obj = di.Experiment(all_sesssions_ever)

sessions = behaviour_sessions_obj.sessions

# Get subject IDs
ids = []
for session in sessions:
    ids.append(session.subject_ID)
unique_ids = np.unique(ids)

dict_ids_ch = {i:[] for i in unique_ids}    
dict_ids_conf = {i:[] for i in unique_ids}    

for session in sessions:
    choices = session.trial_data['choices']
    subj = session.subject_ID
    dict_ids_ch[subj].append(choices)
    trial_lines = [line[1] for line in session.print_lines if line[1][:2] == 'T#'] # Lines with trial data.
                   
    if 'Task' in trial_lines[0]: 

        task_ids = session.trial_data['task']
        dict_ids_conf[subj].append(task_ids)
    else:  
        ch_l = np.arrange(len(session.trial_data['choices']))
        dict_ids_conf[subj].append(ch_l)

dict_ids_conf_conf = {i:[] for i in unique_ids}    

for key in dict_ids_conf.keys():
    count = 0
    subj = dict_ids_conf[key]
    subj_id = key
    for i,ii in enumerate(subj):
        if len(np.unique(subj[i-1])) == 1:
            count+=1
        elif len(np.unique(subj[i-1])) ==2:
            count+=2
        elif len(np.unique(subj[i-1])) ==3:
            count+=3
        ii+=count
        dict_ids_conf_conf[key].append(ii)
        

dict_ids_ch_arr = {i:[] for i in unique_ids}    
      
for k in dict_ids_ch.keys():
    subj = dict_ids_ch[k]
    subj_id = k
    dict_ids_ch_arr[subj_id] = np.concatenate(subj, 0)


dict_ids_conf_id = {i:[] for i in unique_ids}    
      
for k in dict_ids_conf_conf.keys():
    subj = dict_ids_conf_conf[k]
    subj_id = k
    dict_ids_conf_id[subj_id] = np.concatenate(subj, 0)


for kk in dict_ids_conf_id.keys():
    subj  = dict_ids_conf_id[kk]
    subj -= subj[0]
    
    

f = open('configs.pkl','wb')
pickle.dump(dict_ids_conf_id,f)
f.close()
