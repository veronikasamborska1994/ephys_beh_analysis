#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:06:33 2019

@author: veronikasamborska
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def regression_code(data, design_matrix): 
    
    tc = np.identity(design_matrix.shape[1])
    
    pdes = np.linalg.pinv(design_matrix)
    tc_pdes = np.matmul(tc,pdes)
    pdes_tc = np.matmul(np.transpose(pdes),np.transpose(tc))
    
    prevar = np.diag(np.matmul(tc_pdes, pdes_tc))
    
    R = np.identity(design_matrix.shape[0]) - np.matmul(design_matrix, pdes)
    tR = np.trace(R)
    
    pe = np.matmul(pdes,data)
    cope = np.matmul(tc,pe)
    
    res = data - np.matmul(design_matrix,pe)
    sigsq = np.sum((res*res)/tR, axis = 0)
    sigsq = np.reshape(sigsq,(1,res.shape[1]))
    prevar = np.reshape(prevar,(tc.shape[0],1))
    varcope = prevar*sigsq
    
    tstats = cope/np.sqrt(varcope)
    
    return tstats
    
    


def regression_code_fstats(Y,X,c_f): 
    EVs = X.shape[1]
    T = X.shape[0]
    N = Y.shape[1]
    # Fit GLM
    # Get beta weights by multiplying data by pseudoinverse of design matrix 
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)
    # Get residuals
    e = Y - np.matmul(X, beta)
    # Get residual standard deviation
    s = np.std(e, axis=0)
    
    # Calculate t-stats
    # Create a set of contrasts with a single one, so taking each EV individually
    #c = np.eye(EVs)
    # Initialise empty array of t-stats
    #t = np.zeros(beta.shape)
    # Pre-calculate (X^T * X)^-1 because we'll need it for each contrast
    inv_cov_X = np.linalg.inv(np.matmul(np.transpose(X), X))
    # Run through all EVs, calculating t-stats
    # for i in range(EVs):
    #     # Get current contrast
    #     c_i = c[i,:]
    #     # Calculate current t-stat
    #     t[i,:] = np.matmul(c_i, beta) / (s * np.sqrt(np.matmul(np.matmul(c_i, inv_cov_X),np.transpose(c_i))))
    # Do some plotting for inspection
    # plt.figure()
    # # First subplot: histogram of t-stats
    # plt.subplot(1, 2, 1)
    # # Make histogram of t-stats across all contrasts across all neurons
    # plt.hist(np.ravel(t), density=True)
    # # Also plot expected t-distribution. Degrees of freedom: timepoinst - number of EVs
    # plt.plot(np.linspace(-4,4,50), scipy.stats.t.pdf(np.linspace(-4,4,50), T-EVs)) 
    # # Second subplot: data and full-model fit for highest t-stat
    # best_neuron = np.unravel_index(t.argmax(), t.shape)[1]
    # plt.subplot(1,2,2)
    # # Get data from neuron where the highest t-stat was found
    # plt.plot(Y[:,best_neuron])
    # # Calculate prediction for that neuron using all betas
    # plt.plot(np.matmul(X, beta[:,best_neuron]))
    # plt.show()
    
    # Now go on and calculate f-stats
    # Assume the contrast we want to test is whether a neuron is active for any of the first 5 EVs
    #_f = c[contrast,:]
    # Create empty f-stats array: one value per neuron
    f_1 = np.zeros(N)
    # Run through all neurons and calculate f-stat
    for i in range(N):
        # Calculate f-stat for this neuron individually
        f_1[i] = 1/c_f.shape[0] * np.matmul(np.matmul(np.matmul(np.transpose(beta[:,i]), np.transpose(c_f)), np.linalg.inv(s[i]*s[i] * np.matmul(np.matmul(c_f, inv_cov_X), np.transpose(c_f)))), np.matmul(c_f,beta[:,i]))
    # Now do it again, but calculate all f-stats simulateously. Put sigma in front and only keep diagonal of NxN matrix
    f_2 = 1/c_f.shape[0] * 1/(s*s) * np.matmul(np.matmul(np.matmul(np.transpose(beta), np.transpose(c_f)), np.linalg.inv(np.matmul(np.matmul(c_f, inv_cov_X), np.transpose(c_f)))), np.matmul(c_f,beta)).diagonal()
    # Plot distribution of F-stats
    # plt.figure()
    # # Make histogram of f-stats across all neurons
    # plt.hist(f_1, density=True)
    # # Also plot expected f-statistic pdf. Note that histogram doesn't match PDF very well; not sure if degrees of freedom are set correctly
    # plt.plot(np.linspace(0.1,5,50), scipy.stats.f.pdf(np.linspace(0.1,10,50), T-EVs, c_f.shape[0]))
    # plt.show()  
    
    return f_1
    