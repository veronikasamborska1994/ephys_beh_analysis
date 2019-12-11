
# Script for running surprise measure analysis through time or through trials 

import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm

font = {'weight' : 'normal',
        'size'   : 5}

plt.rc('font', **font)



          
def remap_surprise_block(data):
    data = data_HP
    y = data['DM']
    x = data['Data']
    
    for  s, sess in enumerate(x):
        DM = y[s]
        firing = x[s]
        n_trials, n_neurons, n_time = firing.shape
       # state = DM[:,0]
        block = DM[:,4]
       # b_pokes = DM[:,7]
       # a_pokes = DM[:,6]
        
        block_change = np.where(np.diff(block) != 0)[0]

        ind_40_around = []
        ind_baseline = []
        
        for i in block_change:
            start_end = np.arange(i-5,i+5)
            trials_before = np.arange(i-10,i-5)
            
            ind_40_around.append(start_end)
            ind_baseline.append(trials_before)
            
        firing_rates_mean_time = np.mean(firing, axis = 2)
        
        for neuron in range(n_neurons):
            
            n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
                     
            block_1_baseline = np.mean(n_firing[ind_baseline[0]], axis = 0)
           
            

    
def remap_surprise(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):
    
    y = data['DM'][0]
    x = data['Data'][0]

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
        task_1_a_last = task_1_a[-20:] # Find indicies for task 1 A last 10 

        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_last = task_1_b[-20:] # Find indicies for task 1 B last 10 

        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_first = task_2_b[:20] # First indicies for task 2 B
        task_2_b_last = task_2_b[-20:] # Find indicies for task 1 B last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_first = task_2_a[:20] # First 10 indicies for task 2 A 
        task_2_a_last = task_2_a[-20:] # Find indicies for task 1 B last 10 

        firing_rates_mean_time = x[s]
       
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        for neuron in range(n_neurons):
            
            n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
                     
            # Task 1 Mean rates on the first 20 A trials
            task_1_mean_a = np.mean(n_firing[task_1_a[:20]], axis = 0)
            task_1_std_a = np.std(n_firing[task_1_a[:20]], axis = 0)   
           
            # Task 1 Mean rates on the first 20 B trials
            task_1_mean_b = np.mean(n_firing[task_1_b[:20]], axis = 0)
            task_1_std_b = np.std(n_firing[task_1_b[:20]], axis = 0)
              
            # Task 1 Mean rates on the last 20 A trials
            task_1_mean_a_l = np.mean(n_firing[task_1_a_last], axis = 0)
            task_1_std_a_l = np.std(n_firing[task_1_a_last], axis = 0)   
           
            # Task 1 Mean rates on the last 20 B trials
            task_1_mean_b_l = np.mean(n_firing[task_1_b_last], axis = 0)
            task_1_std_b_l = np.std(n_firing[task_1_b_last], axis = 0)
              
            # Task 1 Mean rates on the first 20 A trials
            task_2_mean_a = np.mean(n_firing[task_2_a[:20]], axis = 0)
            task_2_std_a = np.std(n_firing[task_2_a[:20]], axis = 0)   
           
            # Task 1 Mean rates on the first 20 B trials
            task_2_mean_b = np.mean(n_firing[task_2_b[:20]], axis = 0)
            task_2_std_b = np.std(n_firing[task_2_b[:20]], axis = 0)
             
            
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
      
            surprise_array_a = np.concatenate([surprise_av_a_within, surprise_av_a_between])                   
            surprise_array_b = np.concatenate([surprise_av_b_within,surprise_av_b_between])         
            
            surprise_list_neurons_a_a.append(surprise_array_a)
            surprise_list_neurons_b_b.append(surprise_array_b)
            
    surprise_list_neurons_a_a = np.transpose(np.mean(np.asarray(surprise_list_neurons_a_a), axis = 0))
    surprise_list_neurons_b_b = np.transpose(np.mean(np.asarray(surprise_list_neurons_b_b), axis = 0))
    
    surprise_list_neurons_a_a_std =np.transpose(np.std(np.asarray(surprise_list_neurons_a_a), axis = 0))/(np.sqrt(len(np.asarray(surprise_list_neurons_a_a))))
    surprise_list_neurons_b_b_std  =np.transpose(np.std(np.asarray(surprise_list_neurons_b_b), axis = 0))/(np.sqrt(len(np.asarray(surprise_list_neurons_b_b))))

    return surprise_list_neurons_b_b, surprise_list_neurons_a_a,surprise_list_neurons_a_a_std, surprise_list_neurons_b_b_std


   
            
def plot_heat_surprise(data_HP, data_PFC):
    
    mean_b_b_t1_2_HP, mean_a_a_t1_t2_HP, surprise_list_neurons_a_a_std_1_2_HP, surprise_list_neurons_b_b_std_1_2_HP = remap_surprise(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_t2_3_HP, mean_a_a_t2_t3_HP, surprise_list_neurons_a_a_std_2_3_HP, surprise_list_neurons_b_b_std_2_3_HP = remap_surprise(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)  
    mean_b_b_t1_3_HP, mean_a_a_t1_3_HP, surprise_list_neurons_a_a_std_1_3_HP, surprise_list_neurons_b_b_std_1_3_HP = remap_surprise(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)

    mean_b_b_t1_2_PFC, mean_a_a_t1_t2_PFC, surprise_list_neurons_a_a_std_1_2_PFC, surprise_list_neurons_b_b_std_1_2_PFC = remap_surprise(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_t2_3_PFC, mean_a_a_t2_t3_PFC,  surprise_list_neurons_a_a_std_2_3_PFC, surprise_list_neurons_b_b_std_2_3_PFC = remap_surprise(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)  
    mean_b_b_t1_3_PFC, mean_a_a_t1_3_PFC, surprise_list_neurons_a_a_std_1_3_PFC, surprise_list_neurons_b_b_std_1_3_PFC = remap_surprise(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)


    fig1, axes1 = plt.subplots(nrows=2, ncols=2)
    im = axes1[0,0].imshow(mean_b_b_t2_3_HP, aspect = 'auto')

    clim=im.properties()['clim']
    axes1[0,1].imshow(mean_b_b_t2_3_PFC, clim=clim, aspect = 'auto')
    axes1[1,0].imshow(mean_a_a_t2_t3_HP,clim=clim, aspect = 'auto')
    axes1[1,1].imshow(mean_a_a_t2_t3_PFC,clim=clim, aspect = 'auto')
    fig1.colorbar(im, ax=axes1.ravel().tolist(), shrink=0.5)

    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0,0].imshow(mean_b_b_t1_2_HP, aspect = 'auto')
    axes[0,1].imshow(mean_a_a_t1_t2_PFC, clim=clim, aspect = 'auto')
    axes[1,0].imshow(mean_a_a_t1_t2_HP,clim=clim, aspect = 'auto')
    axes[1,1].imshow(mean_a_a_t1_t2_PFC,clim=clim, aspect = 'auto')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

   
    fig2, axes2 = plt.subplots(nrows=2, ncols=2)
    axes2[0,0].imshow(mean_b_b_t1_3_HP, aspect = 'auto')
    axes2[0,1].imshow(mean_b_b_t1_3_PFC, clim=clim, aspect = 'auto')
    axes2[1,0].imshow(mean_a_a_t1_3_HP,clim=clim, aspect = 'auto')
    axes2[1,1].imshow(mean_a_a_t1_3_PFC,clim=clim, aspect = 'auto')
    fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.5)
     
    
    x_pos = np.arange(len(np.mean(mean_a_a_t1_3_HP[25:36,:], axis = 0)))

    t_1_2_HP = np.mean(mean_a_a_t1_t2_HP[36:42,:], axis = 0)
    t_2_3_HP = np.mean(mean_a_a_t2_t3_HP[36:42,:], axis = 0)
    t_1_3_HP = np.mean(mean_a_a_t1_3_HP[36:42,:], axis = 0)

    t_1_2_PFC = np.mean(mean_a_a_t1_t2_PFC[36:42,:], axis = 0)
    t_2_3_PFC = np.mean(mean_a_a_t2_t3_PFC[36:42,:], axis = 0)
    t_1_3_PFC = np.mean(mean_a_a_t1_3_PFC[36:42,:], axis = 0)
 
    plt.figure()
    plt.plot(t_1_2_HP, color = 'green',  label = 'HP 1 2')
    plt.fill_between(x_pos, t_1_2_HP -surprise_list_neurons_a_a_std_1_2_HP, t_1_2_HP + surprise_list_neurons_a_a_std_1_2_HP, alpha=0.2, color = 'green')

# 
    plt.plot(t_2_3_HP, color = 'seagreen',  label = 'HP 2 3')
    plt.fill_between(x_pos, t_2_3_HP -surprise_list_neurons_a_a_std_2_3_HP, t_2_3_HP + surprise_list_neurons_a_a_std_2_3_HP, alpha=0.2, color = 'seagreen')
#
#
    plt.plot(t_1_3_HP, color = 'olive',  label = 'HP 1 3')
    plt.fill_between(x_pos, t_1_3_HP -surprise_list_neurons_a_a_std_1_3_HP, t_1_3_HP + surprise_list_neurons_a_a_std_1_3_HP, alpha=0.2, color = 'olive')
#
    plt.plot(t_1_2_PFC, color = 'lightblue', label = 'PFC 1 2')
    plt.fill_between(x_pos, t_1_2_PFC -surprise_list_neurons_a_a_std_1_2_PFC, t_1_2_PFC + surprise_list_neurons_a_a_std_1_2_PFC, alpha=0.2, color = 'lightblue')
#
# 
    plt.plot(t_2_3_PFC, color = 'darkblue',  label = 'PFC 2 3')
    plt.fill_between(x_pos, t_2_3_PFC -surprise_list_neurons_a_a_std_2_3_PFC, t_2_3_PFC + surprise_list_neurons_a_a_std_2_3_PFC, alpha=0.2, color = 'darkblue')


    plt.plot(t_1_3_PFC, color = 'blue', label = 'PFC 1 3')
    plt.fill_between(x_pos, t_1_3_PFC -surprise_list_neurons_a_a_std_1_3_PFC, t_1_3_PFC + surprise_list_neurons_a_a_std_1_3_PFC, alpha=0.2, color = 'blue')

    plt.legend()
    
    
def through_time_plot(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):    
    
    y = data['DM'][0]
    x = data['Data'][0]

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
        task_1_a_last = task_1_a[-20:] # Find indicies for task 1 A last 10 

        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_last = task_1_b[-20:] # Find indicies for task 1 B last 10 

        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_first = task_2_b[:20] # First indicies for task 2 B
        task_2_b_last = task_2_b[-20:] # Find indicies for task 1 B last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_first = task_2_a[:20] # First 10 indicies for task 2 A 
        task_2_a_last = task_2_a[-20:] # Find indicies for task 1 B last 10 

        firing_rates_mean_time = x[s]
       
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        for neuron in range(n_neurons):
            
            n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
                     
            # Task 1 Mean rates on the first 20 A trials
            task_1_mean_a = np.mean(n_firing[task_1_a[:20]], axis = 0)
            task_1_std_a = np.std(n_firing[task_1_a[:20]], axis = 0)   
           
            # Task 1 Mean rates on the first 20 B trials
            task_1_mean_b = np.mean(n_firing[task_1_b[:20]], axis = 0)
            task_1_std_b = np.std(n_firing[task_1_b[:20]], axis = 0)
              
            # Task 1 Mean rates on the last 20 A trials
            task_1_mean_a_l = np.mean(n_firing[task_1_a_last], axis = 0)
            task_1_std_a_l = np.std(n_firing[task_1_a_last], axis = 0)   
           
            # Task 1 Mean rates on the last 20 B trials
            task_1_mean_b_l = np.mean(n_firing[task_1_b_last], axis = 0)
            task_1_std_b_l = np.std(n_firing[task_1_b_last], axis = 0)
              
            # Task 1 Mean rates on the first 20 A trials
            task_2_mean_a = np.mean(n_firing[task_2_a[:20]], axis = 0)
            task_2_std_a = np.std(n_firing[task_2_a[:20]], axis = 0)   
           
            # Task 1 Mean rates on the first 20 B trials
            task_2_mean_b = np.mean(n_firing[task_2_b[:20]], axis = 0)
            task_2_std_b = np.std(n_firing[task_2_b[:20]], axis = 0)
             
            
            a1_fr_last = np.mean(n_firing[task_1_a_last], axis = 0)           
            a2_fr_first =  np.mean(n_firing[task_2_a_first], axis = 0)
            
            b1_fr_last = np.mean(n_firing[task_1_b_last], axis = 0)            
            b2_fr_first = np.mean(n_firing[task_2_b_first], axis = 0)
            
            a2_fr_last = np.mean(n_firing[task_2_a_last], axis = 0)           
            b2_fr_last = np.mean(n_firing[task_2_b_last], axis = 0)

            min_std = 2
            
            task_1_mean_a_rep = np.tile(task_1_mean_a,[task_1_mean_a.shape[0],1])            
            task_1_std_a_rep = np.tile(task_1_std_a,[task_1_std_a.shape[0],1])
            
            task_1_mean_b_rep = np.tile(task_1_mean_b,[task_1_mean_b.shape[0],1])
            task_1_std_b_rep = np.tile(task_1_std_b,[task_1_std_b.shape[0],1])
            
            task_1_mean_a_rep_l = np.tile(task_1_mean_a_l,[task_1_mean_a_l.shape[0],1])            
            task_1_std_a_rep_l = np.tile(task_1_std_a_l,[task_1_std_a_l.shape[0],1])
            
            task_1_mean_b_rep_l = np.tile(task_1_mean_b_l,[task_1_mean_b_l.shape[0],1])
            task_1_std_b_rep_l = np.tile(task_1_std_b_l,[task_1_std_b_l.shape[0],1])
            
            task_2_mean_a_rep = np.tile(task_2_mean_a,[task_2_mean_a.shape[0],1])
            task_2_std_a_rep = np.tile(task_2_std_a,[task_2_std_a.shape[0],1]) 
               
            task_2_mean_b_rep = np.tile(task_2_mean_b,[task_2_mean_b.shape[0],1])
            task_2_std_b_rep = np.tile(task_2_std_b,[task_2_std_b.shape[0],1])

            a_within_1 = -norm.logpdf(a1_fr_last,np.transpose(task_1_mean_a_rep, (1,0)),task_1_std_a_rep+min_std)
            a_within_2 = -norm.logpdf(a2_fr_last, np.transpose(task_2_mean_a_rep, (1,0)), task_2_std_a_rep+min_std)
            b_within_1 = -norm.logpdf(b1_fr_last, np.transpose(task_1_mean_b_rep, (1,0)), task_1_std_b_rep+min_std)
            b_within_2 = -norm.logpdf(b2_fr_last, np.transpose(task_2_mean_b_rep, (1,0)), task_2_std_b_rep+min_std)
             
            a_between = -norm.logpdf(a2_fr_first, np.transpose(task_1_mean_a_rep_l, (1,0)), task_1_std_a_rep_l+min_std)
            b_between = -norm.logpdf(b2_fr_first, np.transpose(task_1_mean_b_rep_l, (1,0)), task_1_std_b_rep_l+min_std)
            
            a_between_re = -norm.logpdf(a1_fr_last, np.transpose(task_2_mean_a_rep, (1,0)), task_2_std_a_rep+min_std)
            b_between_rev = -norm.logpdf(b1_fr_last, np.transpose(task_2_mean_b_rep, (1,0)), task_2_std_b_rep+min_std)
        
            
            surprise_av_a_within = (a_within_1+a_within_2)/2 
            surprise_av_b_within = (b_within_1 + b_within_2)/2
      
            surprise_av_a_between = (a_between+a_between_re)/2 
            surprise_av_b_between = (b_between + b_between_rev)/2
      
            surprise_array_a = np.concatenate([surprise_av_a_within, surprise_av_a_between], axis = 1)                   
            surprise_array_b = np.concatenate([surprise_av_b_within,surprise_av_b_between], axis = 1)         
            
            surprise_list_neurons_a_a.append(surprise_array_a)
            surprise_list_neurons_b_b.append(surprise_array_b)
            
    surprise_list_neurons_a_a = -np.sqrt(np.mean(np.asarray(surprise_list_neurons_a_a), axis = 0))
    surprise_list_neurons_b_b = -np.sqrt(np.mean(np.asarray(surprise_list_neurons_b_b), axis = 0))
    
    return surprise_list_neurons_b_b,surprise_list_neurons_a_a




def plot_through_time(data_HP,data_PFC):
    
    mean_b_b_HP_1_2, mean_a_a_HP_1_2 = through_time_plot(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_HP_2_3, mean_a_a_HP_2_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    mean_b_b_HP_1_3, mean_a_a_HP_1_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    
    mean_b_b_PFC_1_2, mean_a_a_PFC_1_2 = through_time_plot(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_PFC_2_3, mean_a_a_PFC_2_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    mean_b_b_PFC_1_3, mean_a_a_PFC_1_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)


    fig, axes = plt.subplots(nrows = 2, ncols = 2)
    
    axes[0,0].imshow(mean_b_b_PFC_1_2, aspect = 'auto')
    axes[0,1].imshow(mean_b_b_PFC_1_2, aspect = 'auto')
    axes[1,0].imshow(mean_a_a_PFC_1_2, aspect = 'auto')
    axes[1,1].imshow(mean_a_a_PFC_1_2, aspect = 'auto')

    fig1, axes1 = plt.subplots(nrows = 2, ncols = 2)
    im = axes1[0,0].imshow(mean_b_b_HP_2_3, aspect = 'auto')

    axes1[0,1].imshow(mean_b_b_PFC_2_3, aspect = 'auto')
    axes1[1,0].imshow(mean_a_a_HP_2_3, aspect = 'auto')
    axes1[1,1].imshow(mean_a_a_PFC_2_3, aspect = 'auto')

    
    fig2, axes2 = plt.subplots(nrows = 2, ncols = 2)
    im = axes2[0,0].imshow(mean_b_b_HP_1_3, aspect = 'auto')

    axes2[0,1].imshow(mean_b_b_PFC_1_3,  aspect = 'auto')
    axes2[1,0].imshow(mean_a_a_HP_1_3,aspect = 'auto')
    axes2[1,1].imshow(mean_a_a_PFC_1_3,aspect = 'auto')
     
#    fig, axes = plt.subplots(nrows=2, ncols=2)
#    im = axes[0,0].imshow(mean_b_b_HP_1_2, aspect = 'auto')
#    clim=im.properties()['clim']
#    axes[0,1].imshow(mean_b_b_PFC_1_2, clim=clim, aspect = 'auto')
#    axes[1,0].imshow(mean_a_a_HP_1_2,clim=clim, aspect = 'auto')
#    axes[1,1].imshow(mean_a_a_PFC_1_2,clim=clim, aspect = 'auto')
#    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
#
#    fig1, axes1 = plt.subplots(nrows=2, ncols=2)
#    clim=im.properties()['clim']
#    axes1[0,1].imshow(mean_b_b_PFC_2_3, clim=clim, aspect = 'auto')
#    axes1[1,0].imshow(mean_a_a_HP_2_3,clim=clim, aspect = 'auto')
#    axes1[1,1].imshow(mean_a_a_PFC_2_3,clim=clim, aspect = 'auto')
#    fig1.colorbar(im, ax=axes1.ravel().tolist(), shrink=0.5)
#
#    
#    fig2, axes2 = plt.subplots(nrows=2, ncols=2)
#    im = axes2[0,0].imshow(mean_b_b_HP_1_3, aspect = 'auto')
#    clim=im.properties()['clim']
#    axes2[0,1].imshow(mean_b_b_PFC_1_3, clim=clim, aspect = 'auto')
#    axes2[1,0].imshow(mean_a_a_HP_1_3,clim=clim, aspect = 'auto')
#    axes2[1,1].imshow(mean_a_a_PFC_1_3,clim=clim, aspect = 'auto')
#    fig2.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.5)
#     
    



