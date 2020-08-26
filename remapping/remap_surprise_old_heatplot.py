#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:44:37 2020

@author: veronikasamborska
"""



def remap_surprise(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):
    
    y = data['DM'][0]
    x = data['Data'][0]
    #x,y = rtf.remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)

    ind = 10
    surprise_list_neurons_a_a = []
    surprise_list_neurons_b_b = []
    
    for  s, sess in enumerate(x):
        DM = y[s]
       
        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
                
        if task_1_2 == True:
            
            taskid_1 = 1
            taskid_2 = 2
            
        elif task_2_3 == True:
            
            taskid_1 = 2
            taskid_2 = 3
        
        elif task_1_3 == True:
            
            taskid_1 = 1
            taskid_2 = 3
        
        task_1_a = np.where((taskid == taskid_1) & (choices == 1))[0] # Find indicies for task 1 A
        task_1_a_last = task_1_a[-ind:] # Find indicies for task 1 A last 10 
        task_1_a_first = task_1_a[:ind] # Find indicies for task 1 A last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_first = task_2_a[:ind] # First 10 indicies for task 2 A 
        task_2_a_last = task_2_a[-ind:] # Find indicies for task 1 B last 10 

        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_last = task_1_b[-ind:] # Find indicies for task 1 B last 10 
        task_1_b_first = task_1_b[:ind] # Find indicies for task 1 B last 10 

        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_first = task_2_b[:ind] # First indicies for task 2 B
        task_2_b_last = task_2_b[-ind:] # Find indicies for task 1 B last 10 

       
        firing_rates_mean_time = x[s]
       
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        for neuron in range(n_neurons):
            
            n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
                     
            # Task 1 Mean rates on the first 20 A trials
            task_1_mean_a = np.mean(n_firing[task_1_a_first], axis = 0)
            task_1_std_a = np.std(n_firing[task_1_a_first], axis = 0)   
           
            # Task 1 Mean rates on the first 20 B trials
            task_1_mean_b = np.mean(n_firing[task_1_b_first], axis = 0)
            task_1_std_b = np.std(n_firing[task_1_b_first], axis = 0)
              
            # Task 1 Mean rates on the last 20 A trials
            task_1_mean_a_l = np.mean(n_firing[task_1_a_last], axis = 0)
            task_1_std_a_l = np.std(n_firing[task_1_a_last], axis = 0)   
           
            # Task 1 Mean rates on the last 20 B trials
            task_1_mean_b_l = np.mean(n_firing[task_1_b_last], axis = 0)
            task_1_std_b_l = np.std(n_firing[task_1_b_last], axis = 0)
              
            # Task 2 Mean rates on the first 20 A trials
            task_2_mean_a = np.mean(n_firing[task_2_a_first], axis = 0)
            task_2_std_a = np.std(n_firing[task_2_a_first], axis = 0)   
           
            # Task 2 Mean rates on the first 20 B trials
            task_2_mean_b = np.mean(n_firing[task_2_b_first], axis = 0)
            task_2_std_b = np.std(n_firing[task_2_b_first], axis = 0)
             
            
            a1_fr_last = n_firing[task_1_a_last]
            a2_fr_first =  n_firing[task_2_a_first]
            
            b1_fr_last = n_firing[task_1_b_last]            
            b2_fr_first = n_firing[task_2_b_first]
            
            a2_fr_last = n_firing[task_2_a_last]     
            b2_fr_last = n_firing[task_2_b_last] 
            
            min_std = 1
                     
            a_within_1 = -norm.logpdf(a1_fr_last, task_1_mean_a, task_1_std_a + min_std)
            a_within_2 = -norm.logpdf(a2_fr_last,task_2_mean_a, task_2_std_a + min_std)
            b_within_1 = -norm.logpdf(b1_fr_last, task_1_mean_b, task_1_std_b + min_std)
            b_within_2 = -norm.logpdf(b2_fr_last, task_2_mean_b, task_2_std_b + min_std)
             
            a_between = -norm.logpdf(a2_fr_first, task_1_mean_a_l, task_1_std_a_l + min_std)
            b_between = -norm.logpdf(b2_fr_first, task_1_mean_b_l, task_1_std_b_l+ min_std)
            
            a_between_re = -norm.logpdf(a1_fr_last, task_2_mean_a, task_2_std_a + min_std)
            b_between_rev = -norm.logpdf(b1_fr_last, task_2_mean_b, task_2_std_b + min_std)
        
            
            surprise_av_a_within = (a_within_1+a_within_2)/2 
            surprise_av_b_within = (b_within_1 + b_within_2)/2
      
            surprise_av_a_between = (a_between+a_between_re)/2 
            surprise_av_b_between = (b_between + b_between_rev)/2
      
            surprise_array_a = np.concatenate([surprise_av_a_within, surprise_av_a_between])                   
            surprise_array_b = np.concatenate([surprise_av_b_within,surprise_av_b_between])         
            
            surprise_list_neurons_a_a.append(surprise_array_a)
            surprise_list_neurons_b_b.append(surprise_array_b)
            
    surprise_list_neurons_a_a = np.transpose(np.mean(np.asarray(surprise_list_neurons_a_a), axis = 0))
    surprise_list_neurons_b_b = np.transpose(np.mean(np.asarray(surprise_list_neurons_b_b), axis = 0))
    
    surprise_list_neurons_a_a_std =np.transpose(np.std(np.asarray(surprise_list_neurons_a_a), axis = 0))/(np.sqrt(len(np.asarray(surprise_list_neurons_a_a))))
    surprise_list_neurons_b_b_std  =np.transpose(np.std(np.asarray(surprise_list_neurons_b_b), axis = 0))/(np.sqrt(len(np.asarray(surprise_list_neurons_b_b))))

    return surprise_list_neurons_b_b, surprise_list_neurons_a_a,surprise_list_neurons_a_a_std, surprise_list_neurons_b_b_std


def remap_surprise_shuffle(data, task_1_2 = False, task_2_3 = False, task_1_3 = False, n_perms = 5):
    
    #y = data['DM'][0]
    #x = data['Data'][0]
    x,y = rtf.remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)

    ind = 10
    surprise_list_neurons_a_a_p = []
    surprise_list_neurons_b_b_p = []
   
    for  s, sess in enumerate(x):
        surprise_list_neurons_a_a = []
        surprise_list_neurons_b_b = []
   
        DM = y[s]
       
        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
                
        if task_1_2 == True:
            
            taskid_1 = 1
            taskid_2 = 2
            
        elif task_2_3 == True:
            
            taskid_1 = 2
            taskid_2 = 3
        
        elif task_1_3 == True:
            
            taskid_1 = 1
            taskid_2 = 3
        
        task_1_a = np.where((taskid == taskid_1) & (choices == 1))[0] # Find indicies for task 1 A
        task_1_a_last = task_1_a[-ind:] # Find indicies for task 1 A last 10 
        task_1_a_first = task_1_a[:ind] # Find indicies for task 1 A last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_first = task_2_a[:ind] # First 10 indicies for task 2 A 
        task_2_a_last = task_2_a[-ind:] # Find indicies for task 1 B last 10 

        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_last = task_1_b[-ind:] # Find indicies for task 1 B last 10 
        task_1_b_first = task_1_b[:ind] # Find indicies for task 1 B last 10 

        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_first = task_2_b[:ind] # First indicies for task 2 B
        task_2_b_last = task_2_b[-ind:] # Find indicies for task 1 B last 10 

      
        firing_rates_mean_time = x[s]
       
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        surprise_list_neurons_a_a_perm = []
        surprise_list_neurons_b_b_perm = []
               
        for perm in range(n_perms):

            firing_perm  = np.roll(firing_rates_mean_time,np.random.randint(low = 41, high = len(taskid)), axis=0)
     
            for neuron in range(n_neurons):
                
                n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
                         
                # Task 1 Mean rates on the first 20 A trials
                task_1_mean_a = np.mean(n_firing[task_1_a_first], axis = 0)
                task_1_std_a = np.std(n_firing[task_1_a_first], axis = 0)   
               
                # Task 1 Mean rates on the first 20 B trials
                task_1_mean_b = np.mean(n_firing[task_1_b_first], axis = 0)
                task_1_std_b = np.std(n_firing[task_1_b_first], axis = 0)
                  
                # Task 1 Mean rates on the last 20 A trials
                task_1_mean_a_l = np.mean(n_firing[task_1_a_last], axis = 0)
                task_1_std_a_l = np.std(n_firing[task_1_a_last], axis = 0)   
               
                # Task 1 Mean rates on the last 20 B trials
                task_1_mean_b_l = np.mean(n_firing[task_1_b_last], axis = 0)
                task_1_std_b_l = np.std(n_firing[task_1_b_last], axis = 0)
                  
                # Task 2 Mean rates on the first 20 A trials
                task_2_mean_a = np.mean(n_firing[task_2_a_first], axis = 0)
                task_2_std_a = np.std(n_firing[task_2_a_first], axis = 0)   
               
                # Task 2 Mean rates on the first 20 B trials
                task_2_mean_b = np.mean(n_firing[task_2_b_first], axis = 0)
                task_2_std_b = np.std(n_firing[task_2_b_first], axis = 0)
                 
                
                a1_fr_last = n_firing[task_1_a_last]
                a2_fr_first =  n_firing[task_2_a_first]
                
                b1_fr_last = n_firing[task_1_b_last]            
                b2_fr_first = n_firing[task_2_b_first]
                
                a2_fr_last = n_firing[task_2_a_last]     
                b2_fr_last = n_firing[task_2_b_last] 
                
                min_std = 2
                         
                a_within_1 = -norm.logpdf(a1_fr_last, task_1_mean_a, task_1_std_a + min_std)
                a_within_2 = -norm.logpdf(a2_fr_last,task_2_mean_a, task_2_std_a + min_std)
                b_within_1 = -norm.logpdf(b1_fr_last, task_1_mean_b, task_1_std_b + min_std)
                b_within_2 = -norm.logpdf(b2_fr_last, task_2_mean_b, task_2_std_b + min_std)
                 
                a_between = -norm.logpdf(a2_fr_first, task_1_mean_a_l, task_1_std_a_l + min_std)
                b_between = -norm.logpdf(b2_fr_first, task_1_mean_b_l, task_1_std_b_l+ min_std)
                
                a_between_re = -norm.logpdf(a1_fr_last, task_2_mean_a, task_2_std_a + min_std)
                b_between_rev = -norm.logpdf(b1_fr_last, task_2_mean_b, task_2_std_b + min_std)
            
                
                surprise_av_a_within = (a_within_1+a_within_2)/2 
                surprise_av_b_within = (b_within_1 + b_within_2)/2
          
                surprise_av_a_between = (a_between+a_between_re)/2 
                surprise_av_b_between = (b_between + b_between_rev)/2
          
                if task_2_3 == True:

                    surprise_array_a = np.concatenate([a_within_1, a_between], axis = 0)                   
                    surprise_array_b = np.concatenate([b_within_1, b_between], axis = 0)         
                else:
                    surprise_array_a = np.concatenate([surprise_av_a_within, surprise_av_a_between], axis = 0)                   
                    surprise_array_b = np.concatenate([surprise_av_b_within,surprise_av_b_between], axis = 0)         
               
           
                surprise_list_neurons_a_a.append(surprise_array_a)
                surprise_list_neurons_b_b.append(surprise_array_b)
                
            surprise_list_neurons_a_a_mean = np.transpose(np.nanmean(np.asarray(surprise_list_neurons_a_a), axis = 0))
            surprise_list_neurons_b_b_mean = np.transpose(np.nanmean(np.asarray(surprise_list_neurons_b_b), axis = 0))
            
            surprise_list_neurons_a_a_perm.append(abs(surprise_list_neurons_a_a_mean[:,:10] - surprise_list_neurons_a_a_mean[:,10:]))
            surprise_list_neurons_b_b_perm.append(abs(surprise_list_neurons_b_b_mean[:,:10] - surprise_list_neurons_b_b_mean[:,10:]))
    
        surprise_list_neurons_a_a_p.append(np.percentile(np.asarray(surprise_list_neurons_a_a_perm),95, axis = 0))
        surprise_list_neurons_b_b_p.append(np.percentile(np.asarray(surprise_list_neurons_b_b_perm),95, axis = 0))
        
    surprise_list_neurons_a_a_p = np.mean(surprise_list_neurons_a_a_p,0)
    surprise_list_neurons_b_b_p = np.mean(surprise_list_neurons_b_b_p,0)
    
    return surprise_list_neurons_a_a_p, surprise_list_neurons_b_b_p
                
            