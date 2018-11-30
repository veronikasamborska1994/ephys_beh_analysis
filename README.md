# Ephys and Behaviour Analyses

**Esync** script includes functions to synchronise behaviour, camera and ephys 

**OpenEphys.py** are functions provided by the openephys system (use pack2dat for coverting openephys files to .dat format; mean/median referencing)

**beh_and_ephys_plots.py** is an ugly scipt for initial analyses for the grant write-up; might have useful lines of code for behavioural analyses 

**align_activity.py** is a master function for syncronising behaviour and ephys 

**channel_maps** has channels maps for packing the openephys data to .dat format; can be used for simultaneous recordings from multiple animals

**convert_to_dat.py** is the master script used for packing openephys data to .dat

**data_import.py** functions for importing behaviour from pyControl 

**funcs.py** more generic functions provided by OpenEphys
**ephys_beh_import.py** imports spike data after Kilosort and LFP data from .dat files and combines it with behaviour

**good_sessions.py** is a list of sessions with good behaviour

**heatmap_aligned.py** plots for things like trial aligned heatmaps, regressions and angles between firing rate vectors in different tasks 

**heatplot.py** old script for plotting heatplots for the grant application (might have useful lines)

**neuron_firing_all_pokes.py** is for making cool single neuron firing plots with a box set-up plotted on the same figure 

**plotting.py** behavioural analyses for making learning curves plots

**position_code.py** first script for making place-like plots for openfield data (needs work)

**preprocess.py** is the preprocessing script that finds the correct behavioural and ephys sessions; aligns times and makes .npy arrays for spikes and LFP

**regression.py** projecting regression coefficients from one task to another (needs work)

**remapping.py** is a scipt for controlling that remapping between tasks is larger than within tasks (using firing rate vectors as similarity index)

**remapping_surprise.py** is a scipt for calculating surprise between tasks and within a task for each neuron (needs work)

**remapping_time_course.py** is a script for calculating timecourse of neuronal activity to see how abrupt remapping is (has a selection bias problem so don't use; but might have useful lines)
