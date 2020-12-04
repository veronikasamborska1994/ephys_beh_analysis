
# Script for running surprise measure analysis through time or through trials 

import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm
from scipy import io
import palettable
from palettable import wesanderson as wes
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

font = {'weight' : 'normal',
        'size'   : 2}


    

def run():
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')



def through_time_plot(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):    
    fr,dm = remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)
    surprise_list_neurons_a_a_session_mean = []; surprise_list_neurons_b_b_session_mean = []
    
    surprise_list_neurons_a_a = []
    surprise_list_neurons_b_b = []
    ind_pre = 31
    ind_post = 20
    n_count = 0

    for  s, sess in enumerate(fr):
        DM = dm[s]
       
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
        task_1_a_pre_baseline = task_1_a[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
        task_1_a_pre  = task_1_a[-ind_pre+10:] # Find indicies for task 1 A last 10 
        
        # Reverse
        
        task_1_a_pre_baseline_rev = task_1_a[-10:] # Find indicies for task 1 A last 10 
        task_1_a_pre_rev  = task_1_a[-ind_pre:-ind_pre+20] # Find indicies for task 1 A last 10 
       
        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_pre_baseline = task_1_b[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
        task_1_b_pre  = task_1_b[-ind_pre+10:] # Find indicies for task 1 A last 10 
     
        task_1_b_pre_baseline_rev = task_1_b[-10:] # Find indicies for task 1 A last 10 
        task_1_b_pre_rev  = task_1_b[-ind_pre:-ind_pre+20]# Find indicies for task 1 A last 10 
     
        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_post = task_2_b[:ind_post] # Find indicies for task 1 A last 10 

        task_2_b_post_rev_baseline = task_2_b[-10:] # Find indicies for task 1 A last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_post = task_2_a[:ind_post] # Find indicies for task 1 A last 10 

        task_2_a_post_rev_baseline = task_2_a[-10:] # Find indicies for task 1 A last 10 

     
        firing_rates_mean_time = fr[s]
       
        n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        
        for neuron in range(n_neurons):
            n_count +=1
            
            n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
            n_firing =  gaussian_filter1d(n_firing.astype(float),2,1)

            # Task 1 Mean rates on the first 20 A trials
            task_1_mean_a = np.tile(np.mean(n_firing[task_1_a_pre_baseline], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1])   
            task_1_std_a = np.tile(np.std(n_firing[task_1_a_pre_baseline], axis = 0),[np.std(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
           
            task_1_mean_a_rev = np.tile(np.mean(n_firing[task_1_a_pre_baseline_rev], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline_rev],0).shape[0],1] )
            task_1_std_a_rev = np.tile(np.std(n_firing[task_1_a_pre_baseline_rev], axis = 0),[np.std(n_firing[task_1_a_pre_baseline_rev],0).shape[0],1] ) 
           
            # Task 1 Mean rates on the first 20 B trials
            task_1_mean_b = np.tile(np.mean(n_firing[task_1_b_pre_baseline], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
            task_1_std_b = np.tile(np.std(n_firing[task_1_b_pre_baseline], axis = 0),[np.std(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
             
            task_1_mean_b_rev = np.tile(np.mean(n_firing[task_1_b_pre_baseline_rev], axis = 0), [np.mean(n_firing[task_1_b_pre_baseline_rev],0).shape[0],1] ) 
            task_1_std_b_rev = np.tile(np.std(n_firing[task_1_b_pre_baseline_rev], axis = 0), [np.std(n_firing[task_1_b_pre_baseline_rev],0).shape[0],1] ) 
           
            # Task 1 Mean rates on the last 20 A trials
            task_1_mean_a_l = np.tile(np.mean(n_firing[task_1_a_pre], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
            task_1_std_a_l = np.tile(np.std(n_firing[task_1_a_pre], axis = 0),[np.std(n_firing[task_1_a_pre_baseline],0).shape[0],1] )
           
            task_1_mean_a_l_rev = np.tile(np.mean(n_firing[task_1_a_pre_rev], axis = 0),[np.mean(n_firing[task_1_a_pre_rev],0).shape[0],1] )

            # Task 1 Mean rates on the last 20 B trials
            task_1_mean_b_l = np.tile(np.mean(n_firing[task_1_b_pre], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
            #task_1_std_b_l = np.std(n_firing[task_1_b_pre], axis = 0)
            
            task_1_mean_b_l_rev = np.tile(np.mean(n_firing[task_1_b_pre_rev], axis = 0),[np.mean(n_firing[task_1_b_pre_rev],0).shape[0],1] ) 

            # Task 1 Mean rates on the first 20 A trials
            task_2_mean_a = np.tile(np.mean(n_firing[task_2_a_post], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
            #task_2_std_a = np.std(n_firing[task_2_a_post], axis = 0)   
            
            task_2_mean_a_rev = np.tile(np.mean(n_firing[task_2_a_post_rev_baseline], axis = 0),[np.mean(n_firing[task_2_a_post_rev_baseline],0).shape[0],1] ) 

            task_2_std_a_rev = np.tile(np.std(n_firing[task_2_a_post_rev_baseline], axis = 0),[np.std(n_firing[task_2_a_post_rev_baseline],0).shape[0],1] ) 
            task_2_std_a_rev = np.tile(np.std(n_firing[task_2_a_post_rev_baseline], axis = 0),[np.std(n_firing[task_2_a_post_rev_baseline],0).shape[0],1] ) 


            # Task 1 Mean rates on the first 20 B trials
            task_2_mean_b = np.tile(np.mean(n_firing[task_2_b_post], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
            #task_2_std_b = np.std(n_firing[task_2_b_post], axis = 0)
            task_2_mean_b_rev = np.tile(np.mean(n_firing[task_2_b_post_rev_baseline], axis = 0),[np.mean(n_firing[task_2_b_post_rev_baseline],0).shape[0],1] ) 
            task_2_std_b_rev = np.tile(np.std(n_firing[task_2_b_post_rev_baseline], axis = 0),[np.std(n_firing[task_2_b_post_rev_baseline],0).shape[0],1] ) 
           
           
           
              
            min_std = 2
            
            if (len(np.where(task_1_mean_a_l == 0)[0]) == 0) and (len(np.where(task_1_mean_a == 0)[0]) == 0)\
                    and (len(np.where(task_1_mean_a_rev == 0)[0]) == 0) and (len(np.where(task_1_mean_b_l == 0)[0]) == 0)\
                    and (len(np.where(task_1_mean_b == 0)[0]) == 0) and (len(np.where(task_1_mean_b_rev == 0)[0]) == 0)\
                    and (len(np.where(task_2_mean_a == 0)[0]) == 0) and (len(np.where(task_2_mean_a_rev == 0)[0]) == 0)\
                    and (len(np.where(task_2_mean_b == 0)[0]) == 0) and (len(np.where(task_2_mean_b_rev == 0)[0]) == 0):

                a_within_1 = -norm.logpdf(task_1_mean_a_l, np.transpose(task_1_mean_a, (1,0)), np.transpose(task_1_std_a+min_std))
                a_within_1_rev = -norm.logpdf(task_1_mean_a_l_rev, np.transpose(task_1_mean_a_rev, (1,0)), np.transpose(task_1_std_a_rev+min_std))
    
                b_within_1 = -norm.logpdf(task_1_mean_b_l, np.transpose(task_1_mean_b, (1,0)), np.transpose(task_1_std_b+min_std))
                b_within_1_rev = -norm.logpdf(task_1_mean_b_l_rev, np.transpose(task_1_mean_b_rev, (1,0)), np.transpose(task_1_std_b_rev+min_std))
    
                a_between = -norm.logpdf(task_2_mean_a, np.transpose(task_1_mean_a, (1,0)), np.transpose(task_1_std_a+min_std))
                a_between_rev = -norm.logpdf(task_1_mean_a_l, np.transpose(task_2_mean_a_rev, (1,0)), np.transpose(task_2_std_a_rev+min_std))
    
                b_between = -norm.logpdf(task_2_mean_b, np.transpose(task_1_mean_b, (1,0)), np.transpose(task_1_std_b+min_std))
                b_between_rev = -norm.logpdf(task_1_mean_b_l, np.transpose(task_2_mean_b_rev, (1,0)), np.transpose(task_2_std_b_rev+min_std))
            else:
               
                a_within_1 = np.zeros(task_1_mean_a_l.shape); a_within_1[:] = np.NaN
                a_within_1_rev = np.zeros(task_1_mean_a_l.shape); a_within_1_rev[:] = np.NaN
                b_within_1 = np.zeros(task_1_mean_a_l.shape); b_within_1[:] = np.NaN
                b_within_1_rev = np.zeros(task_1_mean_a_l.shape); b_within_1_rev[:] = np.NaN
                 
                a_between = np.zeros(task_1_mean_a_l.shape); a_between[:] = np.NaN
                a_between_rev = np.zeros(task_1_mean_a_l.shape); a_between_rev[:] = np.NaN
                b_between = np.zeros(task_1_mean_a_l.shape); b_between[:] = np.NaN
                b_between_rev = np.zeros(task_1_mean_a_l.shape); b_between_rev[:] = np.NaN

             
            within_a = np.mean([a_within_1,a_within_1_rev],0)
            within_b = np.mean([b_within_1,b_within_1_rev],0)

            between_a = np.mean([a_between,a_between_rev],0)
            between_b = np.mean([b_between,b_between_rev],0)
            
            if task_2_3 == True:

                surprise_array_a = np.concatenate([a_within_1, a_between], axis = 0)                   
                surprise_array_b = np.concatenate([b_within_1,b_between], axis = 0)         
            else:
                surprise_array_a = np.concatenate([within_a, between_a], axis = 0)                   
                surprise_array_b = np.concatenate([within_b,between_b], axis = 0)         
               
            surprise_list_neurons_a_a.append(surprise_array_a)
            surprise_list_neurons_b_b.append(surprise_array_b)
    
    surprise_list_neurons_a_a_session_mean = (-np.sqrt(np.nanmean(np.asarray(surprise_list_neurons_a_a), axis = 0)))
    surprise_list_neurons_b_b_session_mean = (-np.sqrt(np.nanmean(np.asarray(surprise_list_neurons_b_b), axis = 0)))
      
    
    return surprise_list_neurons_b_b_session_mean,surprise_list_neurons_a_a_session_mean


      

def plot_through_time(HP,PFC):
    
    n_perms = 1000
    
    _s_mean_b_b_HP_1_2, _s_mean_a_a_HP_1_2, _s_mean_b_i_HP_1_2 = shuffle_block_start(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_HP_2_3, _s_mean_a_a_HP_2_3, _s_mean_b_i_HP_2_3 = shuffle_block_start(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_HP_1_3, _s_mean_a_a_HP_1_3, _s_mean_b_i_HP_1_3 = shuffle_block_start(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True, n_perms = n_perms)
    
    _s_mean_b_b_PFC_1_2, _s_mean_a_a_PFC_1_2, _s_mean_b_i_PFC_1_2 = shuffle_block_start(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_PFC_2_3, _s_mean_a_a_PFC_2_3, _s_mean_b_i_PFC_2_3 = shuffle_block_start(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_PFC_1_3, _s_mean_a_a_PFC_1_3, _s_mean_b_i_PFC_1_3= shuffle_block_start(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True, n_perms = n_perms)
   


    mean_b_b_HP_1_2, mean_a_a_HP_1_2 = through_time_plot(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_HP_2_3, mean_a_a_HP_2_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    mean_b_b_HP_1_3, mean_a_a_HP_1_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    
    mean_b_b_PFC_1_2, mean_a_a_PFC_1_2 = through_time_plot(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_PFC_2_3, mean_a_a_PFC_2_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    mean_b_b_PFC_1_3, mean_a_a_PFC_1_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
     
     
    _1_3_a_HP = abs(np.diag(mean_a_a_HP_1_3[:63])-np.diag( mean_a_a_HP_1_3[63:]))
    _1_3_b_HP = abs(np.diag(mean_b_b_HP_1_3[:63])- np.diag(mean_b_b_HP_1_3[63:]))
   
    _1_2_a_HP =  abs(np.diag(mean_a_a_HP_1_2[:63])-  np.diag(mean_a_a_HP_1_2[63:]))
    _1_2_b_HP =  abs(np.diag(mean_b_b_HP_1_2[:63]) -  np.diag(mean_b_b_HP_1_2[63:]))
 
    _2_3_a_HP =  abs(np.diag(mean_a_a_HP_2_3[:63]) -  np.diag(mean_a_a_HP_2_3[63:]))
    _2_3_b_HP =  abs(np.diag(mean_b_b_HP_2_3[:63]) - np.diag(mean_b_b_HP_2_3[63:]))
    
    _1_3_a_PFC = abs(np.diag(mean_a_a_PFC_1_3[:63])- np.diag(mean_a_a_PFC_1_3[63:]))
    _1_3_b_PFC = abs(np.diag(mean_b_b_PFC_1_3[:63])- np.diag(mean_b_b_PFC_1_3[63:]))
   
    _1_2_a_PFC =  abs(np.diag(mean_a_a_PFC_1_2[:63])-  np.diag(mean_a_a_PFC_1_2[63:]))
    _1_2_b_PFC =  abs(np.diag(mean_b_b_PFC_1_2[:63]) -  np.diag(mean_b_b_PFC_1_2[63:]))
 
    _2_3_a_PFC =  abs(np.diag(mean_a_a_PFC_2_3[:63]) -  np.diag(mean_a_a_PFC_2_3[63:]))
    _2_3_b_PFC =  abs(np.diag(mean_b_b_PFC_2_3[:63]) - np.diag(mean_b_b_PFC_2_3[63:]))
    
    
    _2_3_init_b_HP_within =  mean_b_b_HP_2_3.T[36,:63]
    _2_3_init_b_HP_between =  mean_b_b_HP_2_3.T[36,63:]
 
    
    _2_3_init_b_PFC_within =  mean_b_b_PFC_2_3.T[36,:63]
    _2_3_init_b_PFC_between =  mean_b_b_PFC_2_3.T[36,63:]
    
   
    _1_3_init_b_HP_within =  mean_b_b_HP_1_3.T[36,:63]
    _1_3_init_b_HP_between =  mean_b_b_HP_1_3.T[36,63:]
    
    _1_3_init_b_PFC_within =  mean_b_b_PFC_1_3.T[36,:63]
    _1_3_init_b_PFC_between =  mean_b_b_PFC_1_3.T[36,63:]
   
   
    _1_2_init_b_HP_within =  mean_b_b_HP_1_2.T[36,:63]
    _1_2_init_b_HP_between =  mean_b_b_HP_1_2.T[36,63:]
   
    _1_2_init_b_PFC_within =  mean_b_b_PFC_1_2.T[36,:63]
    _1_2_init_b_PFC_between =  mean_b_b_PFC_1_2.T[36,63:]

    
    p_1_2_a_hp = np.where(_1_2_a_HP > np.max(_s_mean_a_a_HP_1_2))
    p_1_3_a_hp = np.where(_1_3_a_HP > np.max(_s_mean_a_a_HP_1_3))
    p_2_3_a_hp = np.where(_2_3_a_HP > np.max(_s_mean_a_a_HP_2_3))
    
    p_1_2_b_hp = np.where(_1_2_b_HP > np.max(_s_mean_b_b_HP_1_2))
    p_1_3_b_hp = np.where(_1_3_b_HP > np.max(_s_mean_b_b_HP_1_3))
    p_2_3_b_hp = np.where(_2_3_b_HP > np.max(_s_mean_b_b_HP_2_3))
 
    
    p_1_2_a_pfc = np.where(_1_2_a_PFC > np.max(_s_mean_a_a_PFC_1_2))
    p_1_3_a_pfc = np.where(_1_3_a_PFC > np.max(_s_mean_a_a_PFC_1_3))
    p_2_3_a_pfc = np.where(_2_3_a_PFC > np.max(_s_mean_a_a_PFC_2_3))
    
    p_1_2_b_pfc = np.where(_1_2_b_PFC > np.max(_s_mean_b_b_PFC_1_2))
    p_1_3_b_pfc = np.where(_1_3_b_PFC > np.max(_s_mean_b_b_PFC_1_3))
    p_2_3_b_pfc = np.where(_2_3_b_PFC > np.max(_s_mean_b_b_PFC_2_3))
 
    
 ## B to Init
    p_1_2_b_i_hp = abs(_1_2_init_b_HP_within-_1_2_init_b_HP_between)[26]> _s_mean_b_i_HP_1_2[26]
    p_1_3_b_i_hp = abs(_1_3_init_b_HP_within-_1_3_init_b_HP_between)[26]> _s_mean_b_i_HP_1_3[26]
    p_2_3_b_i_hp = abs(_2_3_init_b_HP_within-_2_3_init_b_HP_between)[26]> _s_mean_b_i_HP_2_3[26]
   
    p_1_2_b_i_pfc = abs(_1_2_init_b_PFC_within-_1_2_init_b_PFC_between)[26]> _s_mean_b_i_PFC_1_2[26]
    p_1_3_b_i_pfc = abs(_1_3_init_b_PFC_within-_1_3_init_b_PFC_between)[26]> _s_mean_b_i_PFC_1_3[26]
    p_2_3_b_i_pfc = abs(_2_3_init_b_PFC_within-_2_3_init_b_PFC_between)[26]> _s_mean_b_i_PFC_2_3[26]
    
  
        
    mmin  = np.min([_2_3_init_b_HP_within,_2_3_init_b_HP_between,_2_3_init_b_PFC_within,_2_3_init_b_PFC_between])
    mmax  = np.max([_2_3_init_b_HP_within,_2_3_init_b_HP_between,_2_3_init_b_PFC_within,_2_3_init_b_PFC_between ])+0.04
    
 
    isl = wes.Royal2_5.mpl_colors
    It = 26
    Ct = 36
    Re = 42

    plt.figure(figsize = (15,3))

    plt.subplot(1,3,1)

    plt.plot(_1_2_init_b_HP_within, color = isl[2], label = 'HP Within Task')

    plt.plot(_1_2_init_b_HP_between, color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(_1_2_init_b_PFC_within, color = isl[3], label = 'PFC Within Task')

    plt.plot(_1_2_init_b_PFC_between, color = isl[3], linestyle = '--', label = 'PFC Between Task')


    plt.subplot(1,3,2)

    plt.plot(_1_3_init_b_HP_within, color = isl[2], label = 'HP Within Task')

    plt.plot(_1_3_init_b_HP_between, color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(_1_3_init_b_PFC_within, color = isl[3], label = 'PFC Within Task')

    plt.plot(_1_3_init_b_PFC_between, color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    plt.subplot(1,3,3)

    plt.plot(_2_3_init_b_HP_within, color = isl[2], label = 'HP Within Task')

    plt.plot(_2_3_init_b_HP_between, color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(_2_3_init_b_PFC_within, color = isl[3], label = 'PFC Within Task')

    plt.plot(_2_3_init_b_PFC_between, color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)
    sns.despine()

 

        
    # 1 2 A

    isl = wes.Royal2_5.mpl_colors
    It = 25
    Ct = 25
    Re = 42

    plt.figure(figsize = (15,3))
    
      
    mmin  =np.min([mean_a_a_HP_1_2, mean_a_a_HP_1_3,mean_a_a_HP_2_3,\
                   mean_b_b_HP_1_2, mean_b_b_HP_1_3,mean_b_b_HP_2_3,\
                   mean_a_a_PFC_1_2, mean_a_a_PFC_1_3,mean_a_a_PFC_2_3,\
                   mean_b_b_PFC_1_2, mean_b_b_PFC_1_3,mean_b_b_PFC_2_3])
    mmax  =np.max([mean_a_a_HP_1_2, mean_a_a_HP_1_3,mean_a_a_HP_2_3,\
                   mean_b_b_HP_1_2, mean_b_b_HP_1_3,mean_b_b_HP_2_3,\
                   mean_a_a_PFC_1_2, mean_a_a_PFC_1_3,mean_a_a_PFC_2_3,\
                   mean_b_b_PFC_1_2, mean_b_b_PFC_1_3,mean_b_b_PFC_2_3 ])+0.04
    
   
    plt.subplot(2,3,1)
    
    plt.plot(np.diag(mean_a_a_HP_1_2[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(mean_a_a_HP_1_2[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(mean_a_a_PFC_1_2[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(mean_a_a_PFC_1_2[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
   
    
    plt.legend()
    plt.title('1 2')


    # 1 3 A
    
    plt.subplot(2,3,2)
    
    plt.plot(np.diag(mean_a_a_HP_1_3[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(mean_a_a_HP_1_3[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(mean_a_a_PFC_1_3[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(mean_a_a_PFC_1_3[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.title('1 3')

    
   #  2 3 A
    
    plt.subplot(2,3,3)
    
    plt.plot(np.diag(mean_a_a_HP_2_3[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(mean_a_a_HP_2_3[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(mean_a_a_PFC_2_3[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(mean_a_a_PFC_2_3[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.title('2 3')


    plt.subplot(2,3,4)
    
    plt.plot(np.diag(mean_b_b_HP_1_2[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(mean_b_b_HP_1_2[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(mean_b_b_PFC_1_2[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(mean_b_b_PFC_1_2[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.legend()


    # 1 3 A
    
    plt.subplot(2,3,5)
    
    plt.plot(np.diag(mean_b_b_HP_1_3[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(mean_b_b_HP_1_3[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(mean_b_b_PFC_1_3[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(mean_b_b_PFC_1_3[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    #  2 3 A
    
    plt.subplot(2,3,6)
    
    plt.plot(np.diag(mean_b_b_HP_2_3[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(mean_b_b_HP_2_3[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(mean_b_b_PFC_2_3[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(mean_b_b_PFC_2_3[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    sns.despine()

  

 

def remap_surprise_time(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):
    
    y = data['DM'][0]
    x = data['Data'][0]
    task_time_confound_data = []
    task_time_confound_dm = []
    
    for  s, sess in enumerate(x):
        DM = y[s]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
                
        if task_1_2  == True:
            
            taskid_1 = 1
            taskid_2 = 2
            
        elif task_2_3 == True:
            
            taskid_1 = 2
            taskid_2 = 3
        
        elif task_1_3 == True:
            
            taskid_1 = 1
            taskid_2 = 3
        
        task_1 = np.where(taskid == taskid_1)[0][-1]
        task_2 = np.where(taskid == taskid_2)[0][0]
        if task_1+1 == task_2: #or task_1+1== task_2:
            task_time_confound_data.append(sess)
            task_time_confound_dm.append(y[s])
        if task_2_3 == False:
            task_1_rev = np.where(taskid == taskid_1)[0][0]
            task_2_rev = np.where(taskid == taskid_2)[0][-1]
            
            if task_2_rev+1 == task_1_rev:
    
                task_time_confound_data.append(sess)
                task_time_confound_dm.append(y[s])
          
    return task_time_confound_data,task_time_confound_dm



      


def plot_surprise(HP, PFC):
    b_b_hp_1_2,a_a_hp_1_2 = through_time_plot(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    b_b_hp_1_3,a_a_hp_1_3  = through_time_plot(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    b_b_hp_2_3,a_a_hp_2_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    
    b_b_pfc_1_2,a_a_pfc_1_2 = through_time_plot(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    b_b_pfc_1_3,a_a_pfc_1_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    b_b_pfc_2_3,a_a_pfc_2_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    
    cmax = np.max([b_b_hp_1_2,a_a_hp_1_2,b_b_hp_1_3,a_a_hp_1_3,b_b_hp_2_3,a_a_hp_2_3,\
                  b_b_pfc_1_2,a_a_pfc_1_2,b_b_pfc_1_3,a_a_pfc_1_3 , b_b_pfc_2_3,a_a_pfc_2_3 ])
    cmin = np.min([b_b_hp_1_2,a_a_hp_1_2,b_b_hp_1_3,a_a_hp_1_3,b_b_hp_2_3,a_a_hp_2_3,\
                  b_b_pfc_1_2,a_a_pfc_1_2,b_b_pfc_1_3,a_a_pfc_1_3 , b_b_pfc_2_3,a_a_pfc_2_3 ])
    
    #cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    cmap = plt.cm.viridis
    hp_1_2 = np.vstack([b_b_hp_1_2,a_a_hp_1_2])
    hp_1_3 = np.vstack([b_b_hp_1_3,a_a_hp_1_3])
    hp_2_3 = np.vstack([b_b_hp_2_3,a_a_hp_2_3 ])

    pfc_1_2 = np.vstack([b_b_pfc_1_2,a_a_pfc_1_2])
    pfc_1_3 = np.vstack([b_b_pfc_1_3,a_a_pfc_1_3])
    pfc_2_3 = np.vstack([b_b_pfc_2_3,a_a_pfc_2_3])

    yt = [10,25,35,42,50,60]
    yl = ['-0.6','Init', 'Ch','R', '+0.32', '+0.72']
    
    xt = [10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3]
    xl = ['-0.6','Init', 'Ch','R', '+0.32', '+0.72', '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
          '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
              '-0.6','Init', 'Ch','R', '+0.32', '+0.72']
        
    plt.figure(figsize=(10,10))
    plt.subplot(6,1,1)
    plt.imshow(hp_1_2.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks(yt,yl)
    plt.xticks(xt,xl)
 
    plt.subplot(6,1,2)
    plt.imshow(hp_1_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks(yt,yl)
    plt.xticks(xt,xl)
 
    plt.subplot(6,1,3)
    plt.imshow(hp_2_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks(yt,yl)
    plt.xticks(xt,xl)
 
    plt.subplot(6,1,4)
    plt.imshow(pfc_1_2.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks(yt,yl)
    plt.xticks(xt,xl)
 
    plt.subplot(6,1,5)
    plt.imshow(pfc_1_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks(yt,yl)
    plt.xticks(xt,xl)
   
    plt.subplot(6,1,6)
    plt.imshow(pfc_2_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks(yt,yl)
    plt.xticks(xt,xl)
                                                                
    plt.tight_layout()
    plt.colorbar()


 
def shuffle_block_start(data, task_1_2 = False, task_2_3 = False, task_1_3 = False, n_perms = 5):
    
    fr,dm = remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)
    
   
    ind_pre = 31
    ind_post = 20
    n_count = 0
  
    surprise_list_neurons_a_a_p = []
    surprise_list_neurons_b_b_p = []
    surprise_list_neurons_b_init_p = []

    for  s, sess in tqdm(enumerate(fr)):
        
        DM = dm[s]
        task = DM[:,5]
        surprise_list_neurons_a_a_perm = []
        surprise_list_neurons_b_b_perm = []
        surprise_list_neurons_b_init_perm = []
                   

        for perm in range(n_perms):

            choices = DM[:,1]
            b_pokes = DM[:,7]
            a_pokes = DM[:,6]
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
            task_1_a_pre_baseline = task_1_a[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
            task_1_a_pre  = task_1_a[-ind_pre+10:] # Find indicies for task 1 A last 10 
            
            # Reverse
            
            task_1_a_pre_baseline_rev = task_1_a[-10:] # Find indicies for task 1 A last 10 
            task_1_a_pre_rev  = task_1_a[-ind_pre:-ind_pre+20] # Find indicies for task 1 A last 10 
           
            task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
            task_1_b_pre_baseline = task_1_b[-ind_pre:-ind_pre+10] # Find indicies for task 1 A last 10 
            task_1_b_pre  = task_1_b[-ind_pre+10:] # Find indicies for task 1 A last 10 
         
            task_1_b_pre_baseline_rev = task_1_b[-10:] # Find indicies for task 1 A last 10 
            task_1_b_pre_rev  = task_1_b[-ind_pre:-ind_pre+20]# Find indicies for task 1 A last 10 
         
            task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
            task_2_b_post = task_2_b[:ind_post] # Find indicies for task 1 A last 10 
    
            task_2_b_post_rev_baseline = task_2_b[-10:] # Find indicies for task 1 A last 10 
    
            task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
            task_2_a_post = task_2_a[:ind_post] # Find indicies for task 1 A last 10 
    
            task_2_a_post_rev_baseline = task_2_a[-10:] # Find indicies for task 1 A last 10 
    
         
            firing_rates_mean_time = fr[s]
           
            n_trials, n_neurons, n_time = firing_rates_mean_time.shape
            
            surprise_list_neurons_a_a = []
            surprise_list_neurons_b_b = []
            for neuron in range(n_neurons):
                n_count +=1
                
                n_firing = firing_rates_mean_time[:,neuron, :]  # Firing rate of each neuron
                n_firing =  gaussian_filter1d(n_firing.astype(float),2,1)
    
                # Task 1 Mean rates on the first 20 A trials
                task_1_mean_a = np.tile(np.mean(n_firing[task_1_a_pre_baseline], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1])   
                task_1_std_a = np.tile(np.std(n_firing[task_1_a_pre_baseline], axis = 0),[np.std(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
               
                task_1_mean_a_rev = np.tile(np.mean(n_firing[task_1_a_pre_baseline_rev], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline_rev],0).shape[0],1] )
                task_1_std_a_rev = np.tile(np.std(n_firing[task_1_a_pre_baseline_rev], axis = 0),[np.std(n_firing[task_1_a_pre_baseline_rev],0).shape[0],1] ) 
               
                # Task 1 Mean rates on the first 20 B trials
                task_1_mean_b = np.tile(np.mean(n_firing[task_1_b_pre_baseline], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
                task_1_std_b = np.tile(np.std(n_firing[task_1_b_pre_baseline], axis = 0),[np.std(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
                 
                task_1_mean_b_rev = np.tile(np.mean(n_firing[task_1_b_pre_baseline_rev], axis = 0), [np.mean(n_firing[task_1_b_pre_baseline_rev],0).shape[0],1] ) 
                task_1_std_b_rev = np.tile(np.std(n_firing[task_1_b_pre_baseline_rev], axis = 0), [np.std(n_firing[task_1_b_pre_baseline_rev],0).shape[0],1] ) 
               
                # Task 1 Mean rates on the last 20 A trials
                task_1_mean_a_l = np.tile(np.mean(n_firing[task_1_a_pre], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
                task_1_std_a_l = np.tile(np.std(n_firing[task_1_a_pre], axis = 0),[np.std(n_firing[task_1_a_pre_baseline],0).shape[0],1] )
               
                task_1_mean_a_l_rev = np.tile(np.mean(n_firing[task_1_a_pre_rev], axis = 0),[np.mean(n_firing[task_1_a_pre_rev],0).shape[0],1] )
    
                # Task 1 Mean rates on the last 20 B trials
                task_1_mean_b_l = np.tile(np.mean(n_firing[task_1_b_pre], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
                #task_1_std_b_l = np.std(n_firing[task_1_b_pre], axis = 0)
                
                task_1_mean_b_l_rev = np.tile(np.mean(n_firing[task_1_b_pre_rev], axis = 0),[np.mean(n_firing[task_1_b_pre_rev],0).shape[0],1] ) 
    
                # Task 1 Mean rates on the first 20 A trials
                task_2_mean_a = np.tile(np.mean(n_firing[task_2_a_post], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
                #task_2_std_a = np.std(n_firing[task_2_a_post], axis = 0)   
                
                task_2_mean_a_rev = np.tile(np.mean(n_firing[task_2_a_post_rev_baseline], axis = 0),[np.mean(n_firing[task_2_a_post_rev_baseline],0).shape[0],1] ) 
    
                task_2_std_a_rev = np.tile(np.std(n_firing[task_2_a_post_rev_baseline], axis = 0),[np.std(n_firing[task_2_a_post_rev_baseline],0).shape[0],1] ) 
                task_2_std_a_rev = np.tile(np.std(n_firing[task_2_a_post_rev_baseline], axis = 0),[np.std(n_firing[task_2_a_post_rev_baseline],0).shape[0],1] ) 
    
    
                # Task 1 Mean rates on the first 20 B trials
                task_2_mean_b = np.tile(np.mean(n_firing[task_2_b_post], axis = 0),[np.mean(n_firing[task_1_a_pre_baseline],0).shape[0],1] ) 
                #task_2_std_b = np.std(n_firing[task_2_b_post], axis = 0)
                task_2_mean_b_rev = np.tile(np.mean(n_firing[task_2_b_post_rev_baseline], axis = 0),[np.mean(n_firing[task_2_b_post_rev_baseline],0).shape[0],1] ) 
                task_2_std_b_rev = np.tile(np.std(n_firing[task_2_b_post_rev_baseline], axis = 0),[np.std(n_firing[task_2_b_post_rev_baseline],0).shape[0],1] ) 
               
               
               
                  
                min_std = 2
                
                if (len(np.where(task_1_mean_a_l == 0)[0]) == 0) and (len(np.where(task_1_mean_a == 0)[0]) == 0)\
                        and (len(np.where(task_1_mean_a_rev == 0)[0]) == 0) and (len(np.where(task_1_mean_b_l == 0)[0]) == 0)\
                        and (len(np.where(task_1_mean_b == 0)[0]) == 0) and (len(np.where(task_1_mean_b_rev == 0)[0]) == 0)\
                        and (len(np.where(task_2_mean_a == 0)[0]) == 0) and (len(np.where(task_2_mean_a_rev == 0)[0]) == 0)\
                        and (len(np.where(task_2_mean_b == 0)[0]) == 0) and (len(np.where(task_2_mean_b_rev == 0)[0]) == 0):
    
                    a_within_1 = -norm.logpdf(task_1_mean_a_l, np.transpose(task_1_mean_a, (1,0)), np.transpose(task_1_std_a+min_std))
                    a_within_1_rev = -norm.logpdf(task_1_mean_a_l_rev, np.transpose(task_1_mean_a_rev, (1,0)), np.transpose(task_1_std_a_rev+min_std))
        
                    b_within_1 = -norm.logpdf(task_1_mean_b_l, np.transpose(task_1_mean_b, (1,0)), np.transpose(task_1_std_b+min_std))
                    b_within_1_rev = -norm.logpdf(task_1_mean_b_l_rev, np.transpose(task_1_mean_b_rev, (1,0)), np.transpose(task_1_std_b_rev+min_std))
        
                    a_between = -norm.logpdf(task_2_mean_a, np.transpose(task_1_mean_a, (1,0)), np.transpose(task_1_std_a+min_std))
                    a_between_rev = -norm.logpdf(task_1_mean_a_l, np.transpose(task_2_mean_a_rev, (1,0)), np.transpose(task_2_std_a_rev+min_std))
        
                    b_between = -norm.logpdf(task_2_mean_b, np.transpose(task_1_mean_b, (1,0)), np.transpose(task_1_std_b+min_std))
                    b_between_rev = -norm.logpdf(task_1_mean_b_l, np.transpose(task_2_mean_b_rev, (1,0)), np.transpose(task_2_std_b_rev+min_std))
                else:
                   
                    a_within_1 = np.zeros(task_1_mean_a_l.shape); a_within_1[:] = np.NaN
                    a_within_1_rev = np.zeros(task_1_mean_a_l.shape); a_within_1_rev[:] = np.NaN
                    b_within_1 = np.zeros(task_1_mean_a_l.shape); b_within_1[:] = np.NaN
                    b_within_1_rev = np.zeros(task_1_mean_a_l.shape); b_within_1_rev[:] = np.NaN
                     
                    a_between = np.zeros(task_1_mean_a_l.shape); a_between[:] = np.NaN
                    a_between_rev = np.zeros(task_1_mean_a_l.shape); a_between_rev[:] = np.NaN
                    b_between = np.zeros(task_1_mean_a_l.shape); b_between[:] = np.NaN
                    b_between_rev = np.zeros(task_1_mean_a_l.shape); b_between_rev[:] = np.NaN
    
                 
                within_a = np.mean([a_within_1,a_within_1_rev],0)
                within_b = np.mean([b_within_1,b_within_1_rev],0)
    
                between_a = np.mean([a_between,a_between_rev],0)
                between_b = np.mean([b_between,b_between_rev],0)
                
                if task_2_3 == True:
    
                    surprise_array_a = np.concatenate([a_within_1, a_between], axis = 0)                   
                    surprise_array_b = np.concatenate([b_within_1,b_between], axis = 0)         
                else:
                    surprise_array_a = np.concatenate([within_a, between_a], axis = 0)                   
                    surprise_array_b = np.concatenate([within_b,between_b], axis = 0)         
                   
                surprise_list_neurons_a_a.append(surprise_array_a)
                surprise_list_neurons_b_b.append(surprise_array_b)

                
            surprise_list_neurons_a_a_mean = (-np.sqrt(np.nanmean(surprise_list_neurons_a_a,0)))
            surprise_list_neurons_b_b_mean = (-np.sqrt(np.nanmean(surprise_list_neurons_b_b,0)))
            
            surprise_list_neurons_a_a_perm.append(abs(np.diag(surprise_list_neurons_a_a_mean.T[:,:63]) - np.diag(surprise_list_neurons_b_b_mean.T[:,63:])))
            surprise_list_neurons_b_b_perm.append(abs(np.diag(surprise_list_neurons_b_b_mean.T[:,:63]) - np.diag(surprise_list_neurons_b_b_mean.T[:,63:])))
            surprise_list_neurons_b_init_perm.append(abs(surprise_list_neurons_b_b_mean.T[36,:63] - surprise_list_neurons_b_b_mean.T[36,63:]))
             
              
        surprise_list_neurons_a_a_p.append(np.percentile(np.asarray(surprise_list_neurons_a_a_perm),95, axis = 0))
        surprise_list_neurons_b_b_p.append(np.percentile(np.asarray(surprise_list_neurons_b_b_perm),95, axis = 0))
        surprise_list_neurons_b_init_p.append(np.percentile(np.asarray(surprise_list_neurons_b_init_perm),95, axis = 0))
    surprise_list_neurons_a_a_p = np.nanmean(surprise_list_neurons_a_a_p,0)
    surprise_list_neurons_b_b_p = np.nanmean(surprise_list_neurons_b_b_p,0)
    surprise_list_neurons_b_init_p  = np.nanmean(surprise_list_neurons_b_init_p,0)
  
    return surprise_list_neurons_a_a_p, surprise_list_neurons_b_b_p,surprise_list_neurons_b_init_p
         

 