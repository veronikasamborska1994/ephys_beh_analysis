
# Script for running surprise measure analysis through time or through trials 

import numpy as np
import matplotlib.pyplot as plt
import remapping_count as rc 
from scipy.stats import norm
from scipy import io
from palettable import wesanderson as wes
import seaborn as sns
import remap_time_fix as rtf
from  statsmodels.stats.anova import AnovaRM
import pingouin as pg
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel as ttest_rel
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import palettable

font = {'weight' : 'normal',
        'size'   : 2}


    

def run():
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    data_plot = io.loadmat('/Users/veronikasamborska/Desktop/plotDat.mat')


def tim_matlab():
    data_plot = io.loadmat('/Users/veronikasamborska/Desktop/plotDat.mat')
    isl = wes.Royal2_5.mpl_colors
    It = 25
    Ct = 36
    Re = 42
 
    data = data_plot['plotDat']
    data_t_1_2 = data[0,0][0]
    plt.figure()
    for i,d in enumerate(data_t_1_2):
        plt.subplot(2,2,i+1)
        plt.imshow(d[:,10:])
    plt.figure()

    data_t_1_3 = data[1,0][0]
    for i,d in enumerate(data_t_1_3):
        plt.subplot(2,2,i+1)
        plt.imshow(d[:,10:])
 

    plt.figure()

    data_t_2_3 = data[2,0][0]
    for i,d in enumerate(data_t_2_3):
        plt.subplot(2,2,i+1)
        plt.imshow(d[:,10:])
       
   #  init_b_HP_within = data_t_2_3[0][:,24]
   #  init_b_HP_between = data_t_2_3[0][:,84]

   #  init_b_PFC_within = data_t_2_3[1][:,24]
   #  init_b_PFC_between = data_t_2_3[1][:,87]
  
   #  plt.figure(figsize = (7,2))
   #  plt.subplot(1,4,1)
   #  plt.plot(init_b_HP_within, color = isl[0], label = 'HP Within Task')
   #  plt.plot(init_b_HP_between, color = isl[0], linestyle = '--', label = 'HP Between Tasks')

   #  plt.plot(init_b_PFC_within, color = isl[1], label = 'PFC Within Task')
   #  plt.plot(init_b_PFC_between, color = isl[1], linestyle = '--', label = 'PFC Between Task')
   #  mmin  =np.min([init_b_HP_within,init_b_HP_between,init_b_PFC_within,init_b_PFC_between])
   #  mmax  =np.max([init_b_HP_within,init_b_HP_between,init_b_PFC_within,init_b_PFC_between])
           
   #  plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

   #  plt.legend()
   #  plt.title('Init to B')

   #  # CA1
    
   #  # Different Space 
    
   #  within_diff_space_HP = np.diag(data_t_1_2[0][:,:63])
   #  between_diff_space_HP = np.diag(data_t_1_2[0][:,63:])
    
   #  # Same Space 

   #  within_same_space_a_HP =  np.diag(data_t_1_3[2][:,:63])
   #  between_same_space_a_HP = np.diag(data_t_1_3[2][:,63:])
    
   #  # Initiation moves to B
   #  within_b_to_init_HP =  np.diag(data_t_2_3[0][:,:63])
   #  between_b_to_init_HP = np.diag(data_t_2_3[0][:,63:])
   
    
   # # PFC
   # # Different Space 
    
   #  within_diff_space_PFC = np.diag(data_t_1_2[1][:,:63])
   #  between_diff_space_PFC = np.diag(data_t_1_2[1][:,63:])
    
   #  # Same Space 

   #  within_same_space_a_PFC =  np.diag(data_t_1_3[3][:,:63])
   #  between_same_space_a_PFC = np.diag(data_t_1_3[3][:,63:])

   #  # Initiation moves to B
   #  within_b_to_init_PFC =  np.diag(data_t_2_3[1][:,:63])
   #  between_b_to_init_PFC = np.diag(data_t_2_3[1][:,63:])
    
    
   #  # Diff Space plots

   #  isl = wes.Royal2_5.mpl_colors
   #  It = 25
   #  Ct = 36
   #  Re = 42

   #  plt.subplot(1,4,2)
   #  plt.plot(within_diff_space_HP, color = isl[0], label = 'HP Within Task')
   #  plt.plot(between_diff_space_HP, color = isl[0], linestyle = '--', label = 'HP Between Tasks')

   #  plt.plot(within_diff_space_PFC, color = isl[1], label = 'PFC Within Task')
   #  plt.plot(between_diff_space_PFC, color = isl[1], linestyle = '--', label = 'PFC Between Task')
   #  mmin  =np.min([within_diff_space_HP,between_diff_space_HP,within_diff_space_PFC,between_diff_space_PFC])
   #  mmax  =np.max([within_diff_space_HP,between_diff_space_HP,within_diff_space_PFC,between_diff_space_PFC])
           
   #  plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

   #  plt.legend()
   #  plt.title('Different Space')



   #  # Same Space plots
   #  plt.subplot(1,4,3)
   #  plt.plot(within_same_space_a_HP, color = isl[0],label = 'HP Within Task')
   #  plt.plot(between_same_space_a_HP, color = isl[0], linestyle = '--',label = 'HP Between Tasks')

   #  plt.plot(within_same_space_a_PFC, color = isl[1], label = 'PFC Within Task')
   #  plt.plot(between_same_space_a_PFC, color = isl[1], linestyle = '--', label = 'PFC Between Task')
    
   #  mmin  =np.min([within_same_space_a_HP,between_same_space_a_HP,within_same_space_a_PFC,between_same_space_a_PFC])
   #  mmax  =np.max([within_same_space_a_HP,between_same_space_a_HP,within_same_space_a_PFC,between_same_space_a_PFC])
           
   #  plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

   #  plt.legend()
   #  plt.title('Same Space')

   #  # Init becomes B
    
   #  plt.subplot(1,4,4)
   #  plt.plot(within_b_to_init_HP, color = isl[0],label = 'HP Within Task')
   #  plt.plot(between_b_to_init_HP, color = isl[0], linestyle = '--', label = 'HP Between Tasks')

   #  plt.plot(within_b_to_init_PFC, color = isl[1], label = 'PFC Within Task')
   #  plt.plot(between_b_to_init_PFC, color = isl[1], linestyle = '--', label = 'PFC Between Task')
   #  mmin  =np.min([within_b_to_init_HP,between_b_to_init_HP])
   #  mmax  =np.max([within_b_to_init_HP,between_b_to_init_HP])
           
   #  plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
   #  plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

   
   #  plt.legend()   
   #  plt.title('Init Becomes B')
   #  sns.despine()
    




def through_time_plot(data, task_1_2 = False, task_2_3 = False, task_1_3 = False):    
    x,y = rtf.remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)
    surprise_list_neurons_a_a_session_mean = []; surprise_list_neurons_b_b_session_mean = []
    # y = data['DM'][0]
    # x = data['Data'][0]

    surprise_list_neurons_a_a = []
    surprise_list_neurons_b_b = []
    ind = 10
    n_count = 0

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
        task_1_a_pre_baseline = task_1_a[:10] # Find indicies for task 1 A last 10 
        task_1_a_pre  = task_1_a[10:10+ind] # Find indicies for task 1 A last 10 
        
        # Reverse
        task_1_a_pre_baseline_rev = task_1_a[-10:] # Find indicies for task 1 A last 10 
        task_1_a_pre_rev  = task_1_a[-10-ind:-10] # Find indicies for task 1 A last 10 
       
        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_1_b_pre_baseline = task_1_b[:10] # Find indicies for task 1 A last 10 
        task_1_b_pre  = task_1_b[10:10+ind] # Find indicies for task 1 A last 10 
     
        task_1_b_pre_baseline_rev = task_1_b[-10:] # Find indicies for task 1 A last 10 
        task_1_b_pre_rev  = task_1_b[-10-ind:-10]# Find indicies for task 1 A last 10 
     
        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        task_2_b_post = task_2_b[:10] # Find indicies for task 1 A last 10 
        task_2_b_post_rev_baseline = task_2_b[-10-ind:-10] # Find indicies for task 1 A last 10 

        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_a_post = task_2_a[:10] # Find indicies for task 1 A last 10 
        task_2_a_post_rev_baseline = task_2_a[-10-ind:-10] # Find indicies for task 1 A last 10 

        firing_rates_mean_time = x[s]
       
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

def plot_surprise():
    surprise_list_neurons_b_b_session_mean_hp_1_2,surprise_list_neurons_a_a_session_mean_hp_1_2 = through_time_plot(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    surprise_list_neurons_b_b_session_mean_hp_1_3,surprise_list_neurons_a_a_session_mean_hp_1_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    surprise_list_neurons_b_b_session_mean_hp_2_3,surprise_list_neurons_a_a_session_mean_hp_2_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    
    
    surprise_list_neurons_b_b_session_mean_pfc_1_2,surprise_list_neurons_a_a_session_mean_pfc_1_2 = through_time_plot(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    surprise_list_neurons_b_b_session_mean_pfc_1_3,surprise_list_neurons_a_a_session_mean_pfc_1_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    surprise_list_neurons_b_b_session_mean_pfc_2_3,surprise_list_neurons_a_a_session_mean_pfc_2_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    
    cmax = np.max([surprise_list_neurons_b_b_session_mean_hp_1_2,surprise_list_neurons_a_a_session_mean_hp_1_2,\
                   surprise_list_neurons_b_b_session_mean_hp_1_3,surprise_list_neurons_a_a_session_mean_hp_1_3,\
                   surprise_list_neurons_b_b_session_mean_hp_2_3,surprise_list_neurons_a_a_session_mean_hp_2_3,\
                   surprise_list_neurons_b_b_session_mean_pfc_1_2,surprise_list_neurons_a_a_session_mean_pfc_1_2,\
                   surprise_list_neurons_b_b_session_mean_pfc_1_3,surprise_list_neurons_a_a_session_mean_pfc_1_3,\
                   surprise_list_neurons_b_b_session_mean_pfc_2_3,surprise_list_neurons_a_a_session_mean_pfc_2_3])
    cmin = np.min([surprise_list_neurons_b_b_session_mean_hp_1_2,surprise_list_neurons_a_a_session_mean_hp_1_2,\
                   surprise_list_neurons_b_b_session_mean_hp_1_3,surprise_list_neurons_a_a_session_mean_hp_1_3,\
                   surprise_list_neurons_b_b_session_mean_hp_2_3,surprise_list_neurons_a_a_session_mean_hp_2_3,\
                   surprise_list_neurons_b_b_session_mean_pfc_1_2,surprise_list_neurons_a_a_session_mean_pfc_1_2,\
                   surprise_list_neurons_b_b_session_mean_pfc_1_3,surprise_list_neurons_a_a_session_mean_pfc_1_3,\
                   surprise_list_neurons_b_b_session_mean_pfc_2_3,surprise_list_neurons_a_a_session_mean_pfc_2_3])
   
    
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    hp_1_2 = np.vstack([surprise_list_neurons_b_b_session_mean_hp_1_2,surprise_list_neurons_a_a_session_mean_hp_1_2])
    hp_1_3 = np.vstack([surprise_list_neurons_b_b_session_mean_hp_1_3,surprise_list_neurons_a_a_session_mean_hp_1_3])
    hp_2_3 = np.vstack([surprise_list_neurons_b_b_session_mean_hp_2_3,surprise_list_neurons_a_a_session_mean_hp_2_3])

    pfc_1_2 = np.vstack([surprise_list_neurons_b_b_session_mean_pfc_1_2,surprise_list_neurons_a_a_session_mean_pfc_1_2])
    pfc_1_3 = np.vstack([surprise_list_neurons_b_b_session_mean_pfc_1_3,surprise_list_neurons_a_a_session_mean_pfc_1_3])
    pfc_2_3 = np.vstack([surprise_list_neurons_b_b_session_mean_pfc_2_3,surprise_list_neurons_a_a_session_mean_pfc_2_3])

    plt.figure(figsize=(10,10))
    plt.subplot(6,1,1)
    plt.imshow(hp_1_2.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks([10,25,35,42,50,60], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
    plt.xticks([10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                   '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
 
    plt.subplot(6,1,2)
    plt.imshow(hp_1_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks([10,25,35,42,50,60], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
    plt.xticks([10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                   '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
 
    plt.subplot(6,1,3)
    plt.imshow(hp_2_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks([10,25,35,42,50,60], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
    plt.xticks([10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                   '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
 
    plt.subplot(6,1,4)
    plt.imshow(pfc_1_2.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks([10,25,35,42,50,60], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
    plt.xticks([10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                   '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
 
    plt.subplot(6,1,5)
    plt.imshow(pfc_1_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks([10,25,35,42,50,60], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
    plt.xticks([10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                   '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
 
    plt.subplot(6,1,6)
    plt.imshow(pfc_2_3.T, cmap =cmap, vmin =cmin,  vmax =cmax)
    plt.yticks([10,25,35,42,50,60], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
    plt.xticks([10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3], ['-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                   '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
                                                                                                                    '-0.6','Init', 'Ch','R', '+0.32', '+0.72'])
                                                                 
    plt.tight_layout()
    plt.colorbar()
    

def shuffle_block_start(data, task_1_2 = False, task_2_3 = False, task_1_3 = False, n_perms = 5):
    
    x,y = rtf.remap_surprise_time(data, task_1_2 = task_1_2, task_2_3 = task_2_3, task_1_3 = task_1_3)
    # y = data['DM'][0]
    # x = data['Data'][0]

    ind = 10

    n_count = 0
    
    surprise_list_neurons_a_a_p = []
    surprise_list_neurons_b_b_p = []
    
    for  s, sess in tqdm(enumerate(x)):
        
        DM = y[s]
        task = DM[:,5]
        surprise_list_neurons_a_a_perm = []
        surprise_list_neurons_b_b_perm = []
                   

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
                
           
            surprise_list_neurons_a_a = []
            surprise_list_neurons_b_b = []
           
            taskid  = np.roll(task,np.random.randint(len(task)), axis=0)
           # np.random.shuffle(taskid)
            
            
            task_1_a = np.where((taskid == taskid_1) & (choices == 1))[0] # Find indicies for task 1 A
            task_1_a_pre_baseline = task_1_a[:10] # Find indicies for task 1 A last 10 
            task_1_a_pre  = task_1_a[10:10+ind] # Find indicies for task 1 A last 10 
            
            # Reverse
            task_1_a_pre_baseline_rev = task_1_a[-10:] # Find indicies for task 1 A last 10 
            task_1_a_pre_rev  = task_1_a[-10-ind:-10] # Find indicies for task 1 A last 10 
           
            task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
            task_1_b_pre_baseline = task_1_b[:10] # Find indicies for task 1 A last 10 
            task_1_b_pre  = task_1_b[10:10+ind] # Find indicies for task 1 A last 10 
         
            task_1_b_pre_baseline_rev = task_1_b[-10:] # Find indicies for task 1 A last 10 
            task_1_b_pre_rev  = task_1_b[-10-ind:-10]# Find indicies for task 1 A last 10 
         
            task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
            task_2_b_post = task_2_b[:10] # Find indicies for task 1 A last 10 
            task_2_b_post_rev_baseline = task_2_b[-10-ind:-10] # Find indicies for task 1 A last 10 
    
            task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
            task_2_a_post = task_2_a[:10] # Find indicies for task 1 A last 10 
            task_2_a_post_rev_baseline = task_2_a[-10-ind:-10] # Find indicies for task 1 A last 10 

         
            firing_rates_mean_time = x[s]
           
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
                if len(np.where(task_1_mean_a_l == 0)[0]) == 0 and len(np.where(task_1_mean_a == 0)[0]) == 0\
                    and len(np.where(task_1_mean_a_rev == 0)[0]) == 0 and len(np.where(task_1_mean_b_l == 0)[0]) == 0\
                    and len(np.where(task_1_mean_b == 0)[0]) == 0 and len(np.where(task_1_mean_b_rev == 0)[0]) == 0\
                    and len(np.where(task_2_mean_a == 0)[0]) == 0 and len(np.where(task_2_mean_a_rev == 0)[0]) == 0\
                    and len(np.where(task_2_mean_b == 0)[0]) == 0 and len(np.where(task_2_mean_b_rev == 0)[0]) == 0:

            
                
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
    
                    surprise_array_a = np.hstack([a_within_1, a_between])                   
                    surprise_array_b = np.hstack([b_within_1,b_between])         
                else:
                    surprise_array_a = np.hstack([within_a, between_a])                   
                    surprise_array_b = np.hstack([within_b,between_b])     
                
                surprise_list_neurons_a_a.append(surprise_array_a)
                surprise_list_neurons_b_b.append(surprise_array_b)
    
            surprise_list_neurons_a_a_mean = (-np.sqrt(np.nanmean(surprise_list_neurons_a_a,0)))
            surprise_list_neurons_b_b_mean = (-np.sqrt(np.nanmean(surprise_list_neurons_b_b,0)))
            
            surprise_list_neurons_a_a_perm.append(abs(surprise_list_neurons_a_a_mean[:,:63] - surprise_list_neurons_b_b_mean[:,63:]))
            surprise_list_neurons_b_b_perm.append(abs(surprise_list_neurons_b_b_mean[:,:63] - surprise_list_neurons_b_b_mean[:,63:]))
               
              
        surprise_list_neurons_a_a_p.append(np.percentile(np.asarray(surprise_list_neurons_a_a_perm),95, axis = 0))
        surprise_list_neurons_b_b_p.append(np.percentile(np.asarray(surprise_list_neurons_b_b_perm),95, axis = 0))
        
    surprise_list_neurons_a_a_p = np.nanmean(surprise_list_neurons_a_a_p,0)
    surprise_list_neurons_b_b_p = np.nanmean(surprise_list_neurons_b_b_p,0)
    
  
    return surprise_list_neurons_a_a_p, surprise_list_neurons_b_b_p
         


def plot_through_time(HP,PFC):
    
    
    mean_b_b_HP_1_2, mean_a_a_HP_1_2 = through_time_plot(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_HP_2_3, mean_a_a_HP_2_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    mean_b_b_HP_1_3, mean_a_a_HP_1_3 = through_time_plot(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True)
    
    mean_b_b_PFC_1_2, mean_a_a_PFC_1_2 = through_time_plot(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False)
    mean_b_b_PFC_2_3, mean_a_a_PFC_2_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False)
    mean_b_b_PFC_1_3, mean_a_a_PFC_1_3 = through_time_plot(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True)
     
   
     
    within_same_space_a_HP = np.diag(mean_a_a_HP_1_3[:63])
    between_same_space_a_HP = np.diag(mean_a_a_HP_1_3[63:])
     
    within_diff_space_HP = np.diag(mean_b_b_HP_1_2[:63])
    between_diff_space_HP = np.diag(mean_b_b_HP_1_2[63:])
    
    within_b_to_init_HP =  np.diag(mean_b_b_HP_2_3[:63])
    between_b_to_init_HP =np.diag( mean_b_b_HP_2_3[63:])
    
    within_diff_space_PFC = np.diag(mean_b_b_PFC_1_2[:63])
    between_diff_space_PFC = np.diag(mean_b_b_PFC_1_2[63:])
    
    within_same_space_a_PFC =  np.diag(mean_a_a_PFC_1_3[:63])
    between_same_space_a_PFC = np.diag(mean_a_a_PFC_1_3[63:])
   
    within_b_to_init_PFC =  np.diag(mean_b_b_PFC_2_3[:63])
    between_b_to_init_PFC = np.diag(mean_b_b_PFC_2_3[63:])


    n_perms = 500
    _s_mean_b_b_HP_1_2, _s_mean_a_a_HP_1_2 = shuffle_block_start(HP, task_1_2 = True, task_2_3 = False, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_HP_2_3, _s_mean_a_a_HP_2_3 = shuffle_block_start(HP, task_1_2 = False, task_2_3 = True, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_HP_1_3, _s_mean_a_a_HP_1_3 = shuffle_block_start(HP, task_1_2 = False, task_2_3 = False, task_1_3 = True, n_perms = n_perms)
    
    _s_mean_b_b_PFC_1_2, _s_mean_a_a_PFC_1_2 = shuffle_block_start(PFC, task_1_2 = True, task_2_3 = False, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_PFC_2_3, _s_mean_a_a_PFC_2_3 = shuffle_block_start(PFC, task_1_2 = False, task_2_3 = True, task_1_3 = False, n_perms = n_perms)
    _s_mean_b_b_PFC_1_3, _s_mean_a_a_PFC_1_3 = shuffle_block_start(PFC, task_1_2 = False, task_2_3 = False, task_1_3 = True, n_perms = n_perms)
 

    diff_same_space_HP_perm =  np.diag(_s_mean_a_a_HP_1_3)
     
    diff_dif_space_HP_perm = np.diag(_s_mean_b_b_HP_1_2)
    
    diff_b_to_init_HP_perm =  np.diag(_s_mean_b_b_HP_2_3)
    
    diff_same_space_PFC_perm = np.diag(_s_mean_a_a_PFC_1_3)
    
    diff_dif_space_PFC_perm =  np.diag(_s_mean_b_b_PFC_1_2)
   
    diff_b_to_init_PFC_perm =  np.diag(_s_mean_b_b_PFC_2_3)
   
    
    diff_same_space_HP = abs(within_same_space_a_HP - between_same_space_a_HP)
     
    diff_dif_space_HP = abs(within_diff_space_HP -between_diff_space_HP)

    diff_b_to_init_HP =   abs(within_b_to_init_HP- between_b_to_init_HP)
    
    diff_same_space_PFC = abs(within_same_space_a_PFC - between_same_space_a_PFC)
     
    diff_dif_space_PFC = abs(within_diff_space_PFC -between_diff_space_PFC)

    diff_b_to_init_PFC =   abs(within_b_to_init_PFC- between_b_to_init_PFC)
  
    id_same_HP = np.where(diff_same_space_HP > np.max(diff_same_space_HP_perm))
    id_diff_HP = np.where(diff_dif_space_HP >np.max(diff_dif_space_HP_perm))
    id_init_b_HP = np.where(diff_b_to_init_HP >np.max(diff_b_to_init_HP_perm))

    
    id_same_PFC = np.where(diff_same_space_PFC > np.max(diff_same_space_PFC_perm))
    id_diff_PFC = np.where(diff_dif_space_PFC > np.max(diff_dif_space_PFC_perm))
    id_init_b_PFC = np.where(diff_b_to_init_PFC >np.max(diff_b_to_init_PFC_perm))
    
        
    # Diff Space plots

    isl = wes.Royal2_5.mpl_colors
    It = 25
    Ct = 25
    Re = 42

    plt.figure(figsize = (15,3))
    
      
    mmin  =np.min([within_same_space_a_HP, between_same_space_a_HP,  within_diff_space_HP, between_diff_space_HP,\
    within_b_to_init_HP, between_b_to_init_HP, within_diff_space_PFC, between_diff_space_PFC, within_same_space_a_PFC,\
    between_same_space_a_PFC, within_b_to_init_PFC ])
    mmax  =np.max([within_same_space_a_HP, between_same_space_a_HP,  within_diff_space_HP, between_diff_space_HP,\
    within_b_to_init_HP, between_b_to_init_HP, within_diff_space_PFC, between_diff_space_PFC, within_same_space_a_PFC,\
    between_same_space_a_PFC, within_b_to_init_PFC])+0.04
    
   
    plt.subplot(1,3,1)
    
    plt.plot(within_diff_space_HP, color = isl[2], label = 'HP Within Task')
   # plt.fill_between(np.arange(len(within_diff_space_HP)), within_diff_space_HP -_std_within_diff_space_HP, within_diff_space_HP + _std_within_diff_space_HP, alpha=0.2, color = isl[2])

    plt.plot(between_diff_space_HP, color = isl[2], linestyle = '--', label = 'HP Between Tasks')
   # plt.fill_between(np.arange(len(between_diff_space_HP)), between_diff_space_HP -_std_between_diff_space_HP, between_diff_space_HP + _std_between_diff_space_HP, alpha=0.2, color = isl[2])

    plt.plot(within_diff_space_PFC, color = isl[3], label = 'PFC Within Task')
   # plt.fill_between(np.arange(len(within_diff_space_PFC)), within_diff_space_PFC -_std_within_diff_space_PFC, within_diff_space_PFC + _std_within_diff_space_PFC, alpha=0.2, color = isl[3])

    plt.plot(between_diff_space_PFC, color = isl[3], linestyle = '--', label = 'PFC Between Task')
   # plt.fill_between(np.arange(len(between_diff_space_PFC)), between_diff_space_PFC -_std_between_diff_space_PFC, between_diff_space_PFC + _std_between_diff_space_PFC, alpha=0.2, color = isl[3])

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.plot(id_diff_PFC[0], np.ones(id_diff_PFC[0].shape)*y, '.', markersize=3, color= isl[3])
    
    plt.plot(id_diff_HP[0], np.ones(id_diff_HP[0].shape)*(y + 0.02), '.', markersize=3, color= isl[2])

   
    
    plt.legend()
    plt.title('Different Space')


    # Same Space plots
    plt.subplot(1,3,2)
    
    plt.plot(within_same_space_a_HP, color = isl[2], label = 'HP Within Task')
  #  plt.fill_between(np.arange(len(within_same_space_a_HP)), within_same_space_a_HP -_std_within_same_space_a_HP, within_same_space_a_HP + _std_within_same_space_a_HP, alpha=0.2, color = isl[2])

    plt.plot(between_same_space_a_HP, color = isl[2], linestyle = '--', label = 'HP Between Tasks')
  #  plt.fill_between(np.arange(len(between_same_space_a_HP)), between_same_space_a_HP -_std_between_same_space_a_HP, between_same_space_a_HP + _std_between_same_space_a_HP, alpha=0.2, color = isl[2])

    plt.plot(within_same_space_a_PFC, color = isl[3], label = 'PFC Within Task')
  #  plt.fill_between(np.arange(len(within_same_space_a_PFC)), within_same_space_a_PFC -_std_within_same_space_a_PFC, within_same_space_a_PFC + _std_within_same_space_a_PFC, alpha=0.2, color = isl[3])

    plt.plot(between_same_space_a_PFC, color = isl[3], linestyle = '--', label = 'PFC Between Task')
  #  plt.fill_between(np.arange(len(between_same_space_a_PFC)), between_same_space_a_PFC -_std_between_same_space_a_PFC, between_same_space_a_PFC + _std_between_same_space_a_PFC, alpha=0.2, color = isl[3])
  
           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)
    
    y = mmax + 0.02
    plt.plot(id_same_PFC[0], np.ones(id_same_PFC[0].shape)*y, '.', markersize=3, color= isl[3])
    plt.plot(id_same_HP[0], np.ones(id_same_HP[0].shape)*(y + 0.02), '.', markersize=3, color= isl[2])

    plt.legend()
    plt.title('Same Space')

    # Init becomes B
    
    plt.subplot(1,3,3)
     
    plt.plot(within_b_to_init_HP, color = isl[2], label = 'HP Within Task')
  #  plt.fill_between(np.arange(len(within_b_to_init_HP)), within_same_space_a_HP -_std_within_b_to_init_HP, within_same_space_a_HP + _std_within_b_to_init_HP, alpha=0.2, color = isl[2])

    plt.plot(between_b_to_init_HP, color = isl[2], linestyle = '--', label = 'HP Between Tasks')
  #  plt.fill_between(np.arange(len(between_b_to_init_HP)), between_b_to_init_HP -_std_between_b_to_init_HP, between_b_to_init_HP + _std_between_b_to_init_HP, alpha=0.2, color = isl[2])

    plt.plot(within_b_to_init_PFC, color = isl[3], label = 'PFC Within Task')
  #  plt.fill_between(np.arange(len(within_b_to_init_PFC)), within_same_space_a_PFC -_std_within_b_to_init_PFC, within_same_space_a_PFC + _std_within_b_to_init_PFC, alpha=0.2, color = isl[3])

    plt.plot(between_b_to_init_PFC, color = isl[3], linestyle = '--', label = 'PFC Between Task')
  #  plt.fill_between(np.arange(len(between_b_to_init_PFC)), between_b_to_init_PFC -_std_between_b_to_init_PFC, between_b_to_init_PFC + _std_between_b_to_init_PFC, alpha=0.2, color = isl[3])

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)
    y = mmax + 0.04
    plt.plot(id_init_b_PFC[0], np.ones(id_init_b_PFC[0].shape)*y, '.', markersize=3, color=isl[3])
    plt.plot(id_init_b_HP[0], np.ones(id_init_b_HP[0].shape)*(y + 0.04), '.', markersize=3, color=isl[2])
   
    plt.legend()   
    plt.title('Init Becomes B')
    sns.despine()

  