#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:31:32 2019

@author: veronikasamborska
"""


# Script to find latencies to exit pokes

all_sessions = HP + PFC

def reward_latencies(all_sessions):
    
    for s,session in enumerate(all_sessions):
        events = session.events
        

        times_poke_out = []
        times_poke_in = []
        for e,event in enumerate(events):
            if e < len(events):
                if event.name == 'sound_a_no_reward' or event.name == 'sound_b_no_reward':
                    event_before = events[e-1]
                    event_after = events[e+1]
                    if event_before.name + '_out' == event_after.name:
                        times_poke_out.append(event_after)
                        times_poke_in.append(event)
        latency_nr = np.asarray(times_poke_out)-np.asarray(times_poke_in)

            
            
        times_poke_out_r = []
        times_poke_in_r = []
        for e,event in enumerate(events):
            if e < len(events):
                if event.name == 'sound_a_reward' or event.name == 'sound_b_reward':
                    event_before = events[e-1]
                    event_after = events[e+1]
                    if event_before.name + '_out' == event_after.name:
                        times_poke_out_r.append(event_after)
                        times_poke_in_r.append(event)


        latency_r = np.asarray(times_poke_out_r)-np.asarray(times_poke_in_r)
        
        return latency_nr,latency_r