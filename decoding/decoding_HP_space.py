#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:33:57 2019

@author: veronikasamborska
"""
# =============================================================================
# Script for testing Bayes decoder on someone else's data set with place cells (position decoding)
# =============================================================================

import numpy as np
from scipy import io
import sys

###Import functions for binning data for preprocessing###
from Neural_Decoding.preprocessing_funcs import bin_spikes
from Neural_Decoding.preprocessing_funcs import bin_output

from Neural_Decoding.decoders import NaiveBayesDecoder

###Load Data###

folder='/Users/veronikasamborska/Desktop/' #ENTER THE FOLDER THAT YOUR DATA IS IN
# folder='/home/jglaser/Data/DecData/' 
data=io.loadmat(folder+'hc_data_raw.mat')
spike_times=data['spike_times'] #Load spike times of all neurons
pos=data['pos'] #Load x and y positions
pos_times=data['pos_times'][0] #Load times at which positions were recorded


dt=.2 #Size of time bins (in seconds)
t_start=pos_times[0] #Time to start extracting data - here the first time position was recorded
t_end=5608 #pos_times[-1] #Time to finish extracting data - when looking through the dataset, the final position was recorded around t=5609, but the final spikes were recorded around t=5608
downsample_factor=1 #Downsampling of output (to make binning go faster). 1 means no downsampling.


#When loading the Matlab cell "spike_times", Python puts it in a format with an extra unnecessary dimension
#First, we will put spike_times in a cleaner format: an array of arrays
spike_times=np.squeeze(spike_times)
for i in range(spike_times.shape[0]):
    spike_times[i]=np.squeeze(spike_times[i])
    
    
##Preprocessing to put spikes and output in bins###

#Bin neural data using "bin_spikes" function
neural_data=bin_spikes(spike_times,dt,t_start,t_end)

#Bin output (position) data using "bin_output" function
pos_binned=bin_output(pos,pos_times,dt,t_start,t_end,downsample_factor)


import pickle

data_folder='/Users/veronikasamborska/Desktop/' 

with open(data_folder+'example_data_hc.pickle','wb') as f:
    pickle.dump([neural_data,pos_binned],f)
    


import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import sys
import pickle


#Import decoder functions

folder='/Users/veronikasamborska/Desktop/'  #ENTER THE FOLDER THAT YOUR DATA IS IN

with open(folder+'example_data_hc.pickle','rb') as f:
    neural_data,pos_binned=pickle.load(f,encoding='latin1') #If using python 3

bins_before = 4 #How many bins of neural data prior to the output are used for decoding
bins_current = 1 #Whether to use concurrent time bin of neural data
bins_after = 5 #How many bins of neural data after the output are used for decoding


#Remove neurons with too few spikes in HC dataset
nd_sum=np.nansum(neural_data,axis=0) #Total number of spikes of each neuron
rmv_nrn=np.where(nd_sum<100) #Find neurons who have less than 100 spikes total
neural_data=np.delete(neural_data,rmv_nrn,1) #Remove those neurons
X=neural_data

#Set decoding output
y=pos_binned

N=bins_before+bins_current+bins_after


#Remove time bins with no output (y value)
rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
X=np.delete(X,rmv_time,0)
y=np.delete(y,rmv_time,0)

#Set what part of data should be part of the training/testing/validation sets

training_range=[0, 0.5]
valid_range=[0.5,0.65]
testing_range=[0.65, 0.8]


#Number of examples after taking into account bins removed for lag alignment
num_examples=X.shape[0]

#Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
#This makes it so that the different sets don't include overlapping neural data
training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

#Get training data
X_train=X[training_set,:]
y_train=y[training_set,:]

#Get testing data
X_test=X[testing_set,:]
y_test=y[testing_set,:]

#Get validation data
X_valid=X[valid_set,:]
y_valid=y[valid_set,:]


#Initialize matrices for neural data in Naive bayes format
num_nrns=X_train.shape[1]
X_b_train=np.empty([X_train.shape[0]-N+1,num_nrns])
X_b_valid=np.empty([X_valid.shape[0]-N+1,num_nrns])
X_b_test=np.empty([X_test.shape[0]-N+1,num_nrns])

#Below assumes that bins_current=1 (otherwise alignment will be off by 1 between the spikes and outputs)

#For all neurons, within all the bins being used, get the total number of spikes (sum across all those bins)
#Do this for the training/validation/testing sets
for i in range(num_nrns):
    X_b_train[:,i]=N*np.convolve(X_train[:,i], np.ones((N,))/N, mode='valid') #Convolving w/ ones is a sum across those N bins
    X_b_valid[:,i]=N*np.convolve(X_valid[:,i], np.ones((N,))/N, mode='valid')
    X_b_test[:,i]=N*np.convolve(X_test[:,i], np.ones((N,))/N, mode='valid')

#Make integer format
X_b_train=X_b_train.astype(int)
X_b_valid=X_b_valid.astype(int)
X_b_test=X_b_test.astype(int)

#Make y's aligned w/ X's
#e.g. we have to remove the first y if we are using 1 bin before, and have to remove the last y if we are using 1 bin after
if bins_before>0 and bins_after>0:
    y_train=y_train[bins_before:-bins_after,:]
    y_valid=y_valid[bins_before:-bins_after,:]
    y_test=y_test[bins_before:-bins_after,:]
    
if bins_before>0 and bins_after==0:
    y_train=y_train[bins_before:,:]
    y_valid=y_valid[bins_before:,:]
    y_test=y_test[bins_before:,:]

#Declare model

#The parameter "encoding_model" can either be linear or quadratic, although additional encoding models could later be added.

#The parameter "res" is the number of bins used (resolution) for decoding predictions
#So if res=100, we create a 100 x 100 grid going from the minimum to maximum of the output variables (x and y positions)
#The prediction the decoder makes will be a value on that grid 

model_nb=NaiveBayesDecoder(encoding_model='quadratic',res=100)

#Fit model
model_nb.fit(X_b_train,y_train)

y_valid_predicted=model_nb.predict(X_b_valid,y_valid)


#Get metric of fit
R2_nb=get_R2(y_valid,y_valid_predicted)
print(R2_nb)

plt.plot(y_valid[2000:2500,1])
plt.plot(y_valid_predicted[2000:2500,1])