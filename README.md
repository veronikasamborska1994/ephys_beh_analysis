# Ephys and Behaviour Analyses

**Esync** _ _script includes functions to synchronise behaviour, camera and ephys_ _ 

**OpenEphys.py** _ _are functions provided by the openephys system (use pack2dat for coverting openephys files to .dat format; mean/median referencing)_ _

**beh_and_ephys_plots.py** _ _is an ugly scipt for initial analyses for the grant write-up; might have useful lines of code for behavioural analyses 

**align_activity.py** _ _is a master function for syncronising behaviour and ephys 

**channel_maps** _ _has channels maps for packing the openephys data to .dat format; can be used for simultaneous recordings from multiple animals

**convert_to_dat.py** _ _is the master script used for packing openephys data to .dat

**data_import.py** _ _functions for importing behaviour from pyControl 

**funcs.py** _ _more generic functions provided by OpenEphys

**ephys_beh_import.py** _ _imports spike data after Kilosort and LFP data from .dat files and combines it with behaviour

**good_sessions.py** _ _is a list of sessions with good behaviour

**heatmap_aligned.py** _ _plots for things like trial aligned heatmaps, regressions and angles between firing rate vectors in different tasks 

**heatplot.py** _ _old script for plotting heatplots for the grant application (might have useful lines)

**neuron_firing_all_pokes.py** _ _is for making cool single neuron firing plots with a box set-up plotted on the same figure 

**plotting.py** _ _behavioural analyses for making learning curves plots

**position_code.py** _ _first script for making place-like plots for openfield data (needs work)

**preprocess.py** _ _is the preprocessing script that finds the correct behavioural and ephys sessions; aligns times and makes .npy arrays for spikes and LFP

**regression.py** _ _projecting regression coefficients from one task to another (needs work)

**remapping.py** _ _is a scipt for controlling that remapping between tasks is larger than within tasks (using firing rate vectors as similarity index)

**remapping_surprise.py** _ _is a scipt for calculating surprise between tasks and within a task for each neuron (needs work)

**remapping_time_course.py** _ _is a script for calculating timecourse of neuronal activity to see how abrupt remapping is (has a selection bias problem so don't use; but might have useful lines)
