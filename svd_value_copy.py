#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:04:01 2020

@author: veronikasamborska
"""



def value_reg_svd(data, n = 10, plot_a = False, plot_b = False,  first_half = 1, a ='PFC'):
  
   # dm = data['DM'][0]
   # firing = data['Data'][0]

    average = rew_prev_behaviour(data, n = n)
    if a == 'PFC':
        all_subjects = [data['DM'][0][:9], data['DM'][0][9:25],data['DM'][0][25:39],data['DM'][0][39:]]
        all_firing = [data['Data'][0][:9], data['Data'][0][9:25],data['Data'][0][25:39],data['Data'][0][39:]]
    else:   
        all_subjects = [data['DM'][0][:16], data['DM'][0][16:24],data['DM'][0][24:]]
        all_firing = [data['Data'][0][:16], data['Data'][0][16:24],data['Data'][0][24:]]
        
    C_1_all = []; C_2_all = []; C_3_all = []
    kf = KFold(n_splits= 20)
    
    for d,dd in enumerate(all_subjects):
        C_1 = []; C_2 = []; C_3 = []

        dm = all_subjects[d]
        firing = all_firing[d]

    
        for  s, sess in enumerate(dm):
            
           
            DM = dm[s]
            block = DM[:,4]
            block_df = np.diff(block)
            ind_block = np.where(block_df != 0)[0]
        
            firing_rates = firing[s]#[:ind_block[11]]
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            choices = DM[:,1]#[:ind_block[11]]
            reward = DM[:,2]#[:ind_block[11]]  
            state = DM[:,0]
            task =  DM[:,5]#[:ind_block[11]]
           
            a_pokes = DM[:,6]#[:ind_block[11]]
            b_pokes = DM[:,7]#[:ind_block[11]]
            
            taskid = task_ind(task, a_pokes, b_pokes)
          
            
            task_1 = np.where(taskid == 1)[0]
            task_2 = np.where(taskid == 2)[0]
            task_3 = np.where(taskid == 3)[0]
            # plt.figure()
            # plt.imshow(np.mean(firing_rates,2).T, aspect ='auto')
    
    
            reward_current = reward
            choices_current = choices-0.5
    
           
            rewards_1 = reward_current[task_1]
            choices_1 = choices_current[task_1]
            
            previous_rewards_1 = scipy.linalg.toeplitz(rewards_1, np.zeros((1,n)))[n-1:-1]         
            previous_choices_1 = scipy.linalg.toeplitz(0.5-choices_1, np.zeros((1,n)))[n-1:-1]       
            interactions_1 = scipy.linalg.toeplitz((((0.5-choices_1)*(rewards_1-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_1)).reshape(len(interactions_1),1)
             
            X_1 = np.hstack([previous_rewards_1,previous_choices_1,interactions_1,ones])
            value_1 =np.matmul(X_1, average)
    
            rewards_1 = rewards_1[n:]
            choices_1 = choices_1[n:]
            state_1 = state[task_1]
            state_1 = state_1[n:]
 
            
            ones_1 = np.ones(len(choices_1))
            trials_1 = len(choices_1)
            
 
          
            firing_rates_1 = firing_rates[task_1][n:]
            
            a_1 = np.where(choices_1 == 0.5)[0]
            b_1 = np.where(choices_1 == -0.5)[0]
            
            if plot_a == True:
                
                # rewards_1 = rewards_1[a_1] 
                # choices_1 = choices_1[a_1]    
                # value_1 = value_1[a_1]
                # ones_1  = ones_1[a_1]
                # firing_rates_1 = firing_rates_1[a_1]
                
                ind_st_b = np.intersect1d(np.where(state_1==0)[0],a_1)
                ind_st_a = np.intersect1d(np.where(state_1==1)[0],a_1)
 
                ind_1_b = ind_st_b[::2]
                ind_2_b =  ind_st_b[1::2]
               
                ind_1_a = ind_st_a[::2]
                ind_2_a =  ind_st_a[1::2]
          
             
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))

                if first_half == 1:
                    rewards_1 = rewards_1[ind_1] 
                    choices_1 = choices_1[ind_1]    
                    value_1 = value_1[ind_1]
                    ones_1  = ones_1[ind_1]
                    firing_rates_1 = firing_rates_1[ind_1]
                    
                elif first_half == 2:
                  
                    rewards_1 = rewards_1[ind_2] 
                    choices_1 = choices_1[ind_2]    
                    value_1 = value_1[ind_2]
                    ones_1  = ones_1[ind_2]
                    firing_rates_1 = firing_rates_1[ind_2]
                    
             
                   
                
              
            elif plot_b == True:
                
                # rewards_1 = rewards_1[b_1] 
                # choices_1 = choices_1[b_1]    
                # value_1 = value_1[b_1]
                # ones_1  = ones_1[b_1]
                # firing_rates_1 = firing_rates_1[b_1]
                ind_st_b = np.intersect1d(np.where(state_1==0)[0],b_1)
                ind_st_a = np.intersect1d(np.where(state_1==1)[0],b_1)
 
                ind_1_b = ind_st_b[::2]
                ind_2_b =  ind_st_b[1::2]
                
                ind_1_a = ind_st_a[::2]
                ind_2_a =  ind_st_a[1::2]
          
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))

                if first_half == 1:
                    rewards_1 = rewards_1[ind_1] 
                    choices_1 = choices_1[ind_1]    
                    value_1 = value_1[ind_1]
                    ones_1  = ones_1[ind_1]
                    firing_rates_1 = firing_rates_1[ind_1]
                    
                elif first_half == 2:
                  
                    rewards_1 = rewards_1[ind_2] 
                    choices_1 = choices_1[ind_2]    
                    value_1 = value_1[ind_2]
                    ones_1  = ones_1[ind_2]
                    firing_rates_1 = firing_rates_1[ind_2]
              
                
            # ind = np.arange(len(ones_1))
            # ind_1 = ind[::2]
            # ind_2 =  ind[1::2]
           
        
            # if first_half == 1:
            #     rewards_1 = rewards_1[ind_1] 
            #     choices_1 = choices_1[ind_1]    
            #     value_1 = value_1[ind_1]
            #     ones_1  = ones_1[ind_1]
            #     firing_rates_1 = firing_rates_1[ind_1]
                
            # elif first_half == 2:
              
            #     rewards_1 = rewards_1[ind_2] 
            #     choices_1 = choices_1[ind_2]    
            #     value_1 = value_1[ind_2]
            #     ones_1  = ones_1[ind_2]
            #     firing_rates_1 = firing_rates_1[ind_2]
                
                
                
            predictors_all = OrderedDict([
                                        ('Reward', rewards_1),
                                        ('Value',value_1), 
                                      
                                        ('ones', ones_1)])
            
            X_1 = np.vstack(predictors_all.values()).T[:trials_1,:].astype(float)
            
            n_predictors = X_1.shape[1]
            y_1 = firing_rates_1.reshape([len(firing_rates_1),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats,x = regression_code_session(y_1, X_1)
           # tstats =  reg_f.regression_code(y_1, X_1)
    
            C_1.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
            
            
            rewards_2 = reward_current[task_2]
            choices_2 = choices_current[task_2]
            
            previous_rewards_2 = scipy.linalg.toeplitz(rewards_2, np.zeros((1,n)))[n-1:-1]      
            previous_choices_2 = scipy.linalg.toeplitz(0.5-choices_2, np.zeros((1,n)))[n-1:-1]        
            interactions_2 = scipy.linalg.toeplitz((((0.5-choices_2)*(rewards_2-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_2)).reshape(len(interactions_2),1)
             
            X_2 = np.hstack([previous_rewards_2,previous_choices_2,interactions_2,ones])
            value_2 =np.matmul(X_2, average)
    
            rewards_2 = rewards_2[n:]
            choices_2 = choices_2[n:]
            state_2 = state[task_2]
            state_2 = state_2[n:]

            
            ones_2 = np.ones(len(choices_2))
            trials_2 = len(choices_2)
    
            firing_rates_2 = firing_rates[task_2][n:]
            
     
            a_2 = np.where(choices_2 == 0.5)[0]
            b_2 = np.where(choices_2 == -0.5)[0]
            
            if plot_a == True:
                
                # rewards_2 = rewards_2[a_2] 
                # choices_2 = choices_2[a_2]    
                # value_2 = value_2[a_2]
                # ones_2  = ones_2[a_2]
                # firing_rates_2 = firing_rates_2[a_2]
                   
                 
                ind_st_b = np.intersect1d(np.where(state_2==0)[0],a_2)
                ind_st_a = np.intersect1d(np.where(state_2==1)[0],a_2)
 
                ind_1_b = ind_st_b[::2]
                ind_2_b =  ind_st_b[1::2]
               
                ind_1_a = ind_st_a[::2]
                ind_2_a =  ind_st_a[1::2]
               
            
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))

                if first_half == 1:
                    rewards_2 = rewards_2[ind_1] 
                    choices_2 = choices_2[ind_1]    
                    value_2 = value_2[ind_1]
                    ones_2  = ones_2[ind_1]
                    firing_rates_2 = firing_rates_2[ind_1]
                    
                elif first_half == 2:
                    rewards_2 = rewards_2[ind_2] 
                    choices_2 = choices_2[ind_2]    
                    value_2 = value_2[ind_2]
                    ones_2  = ones_2[ind_2]
                    firing_rates_2 = firing_rates_2[ind_2]
                     
             
              
            elif plot_b == True:
                # rewards_2 = rewards_2[b_2] 
                # choices_2 = choices_2[b_2]    
                # value_2 = value_2[b_2]
                # ones_2  = ones_2[b_2]
                # firing_rates_2 = firing_rates_2[b_2]
                ind_st_b = np.intersect1d(np.where(state_2==0)[0],b_2)
                ind_st_a = np.intersect1d(np.where(state_2==1)[0],b_2)
 
                ind_1_b = ind_st_b[::2]
                ind_2_b =  ind_st_b[1::2]
               
                ind_1_a = ind_st_a[::2]
                ind_2_a =  ind_st_a[1::2]
          
             
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))

                if first_half == 1:
                    rewards_2 = rewards_2[ind_1] 
                    choices_2 = choices_2[ind_1]    
                    value_2 = value_2[ind_1]
                    ones_2  = ones_2[ind_1]
                    firing_rates_2 = firing_rates_2[ind_1]
                    
                elif first_half == 2:
                    rewards_2 = rewards_2[ind_2] 
                    choices_2 = choices_2[ind_2]    
                    value_2 = value_2[ind_2]
                    ones_2  = ones_2[ind_2]
                    firing_rates_2 = firing_rates_2[ind_2]
               
            
             
            # ind = np.arange(len(ones_2))
            # ind_1 = ind[::2]
            # ind_2 =  ind[1::2]
           
            
            # if first_half == 1:
               
            #     rewards_2 = rewards_2[ind_1] 
            #     choices_2 = choices_2[ind_1]    
            #     value_2 = value_2[ind_1]
            #     ones_2  = ones_2[ind_1]
            #     firing_rates_2 = firing_rates_2[ind_1]
                 
            # elif first_half == 2:
      
            #     rewards_2 = rewards_2[ind_2] 
            #     choices_2 = choices_2[ind_2]    
            #     value_2 = value_2[ind_2]
            #     ones_2  = ones_2[ind_2]
            #     firing_rates_2 = firing_rates_2[ind_2]
                  
                        
            predictors_all = OrderedDict([
                                        ('Reward', rewards_2),
                                        ('Value',value_2), 
                                     
                                        ('ones', ones_2)])
            
            X_2 = np.vstack(predictors_all.values()).T[:trials_2,:].astype(float)
            
            n_predictors = X_2.shape[1]
            y_2 = firing_rates_2.reshape([len(firing_rates_2),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats,x = regression_code_session(y_2, X_2)
            #tstats =  reg_f.regression_code(y_2, X_2)

            C_2.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
      
        
            
            rewards_3 = reward_current[task_3]
            choices_3 = choices_current[task_3]
            
            previous_rewards_3 = scipy.linalg.toeplitz(rewards_3, np.zeros((1,n)))[n-1:-1]
             
            previous_choices_3 = scipy.linalg.toeplitz(0.5-choices_3, np.zeros((1,n)))[n-1:-1]
             
            interactions_3 = scipy.linalg.toeplitz((((0.5-choices_3)*(rewards_3-0.5))*2),np.zeros((1,n)))[n-1:-1]
             
    
            ones = np.ones(len(interactions_3)).reshape(len(interactions_3),1)
             
            X_3 = np.hstack([previous_rewards_3,previous_choices_3,interactions_3,ones])
            value_3 =np.matmul(X_3, average)
    
            rewards_3 = rewards_3[n:]
            choices_3 = choices_3[n:]
            state_3 = state[task_3]
            state_3 = state_3[n:]

            
            ones_3 = np.ones(len(choices_3))
            trials_3 = len(choices_3)
    
            firing_rates_3 = firing_rates[task_3][n:]
            
        
            a_3 = np.where(choices_3 == 0.5)[0]
            b_3 = np.where(choices_3 == -0.5)[0]
            if plot_a == True:
                
                # rewards_3 = rewards_3[a_3] 
                # choices_3 = choices_3[a_3]    
                # value_3 = value_3[a_3]
                # ones_3  = ones_3[a_3]
                # firing_rates_3 = firing_rates_3[a_3]
                ind_st_b = np.intersect1d(np.where(state_3==0)[0],a_3)
                ind_st_a = np.intersect1d(np.where(state_3==1)[0],a_3)
 
                ind_1_b = ind_st_b[::2]
                ind_2_b =  ind_st_b[1::2]
               
                ind_1_a = ind_st_a[::2]
                ind_2_a =  ind_st_a[1::2]
           
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))

                if first_half == 1:
                    rewards_3 = rewards_3[ind_1] 
                    choices_3 = choices_3[ind_1]    
                    value_3 = value_3[ind_1]
                    ones_3  = ones_3[ind_1]
                    firing_rates_3 = firing_rates_3[ind_1]   
                    
                    
                    
                elif first_half == 2:
                  
                    rewards_3 = rewards_3[ind_2] 
                    choices_3 = choices_3[ind_2]    
                    value_3 = value_3[ind_2]
                    ones_3  = ones_3[ind_2]
                    firing_rates_3 = firing_rates_3[ind_2]   
   
              
            elif plot_b == True:
                # rewards_3 = rewards_3[b_3] 
                # choices_3 = choices_3[b_3]    
                # value_3 = value_3[b_3]
                # ones_3  = ones_3[b_3]
                # firing_rates_3 = firing_rates_3[b_3]
                np.where(value_3 < np.median(value_3))
                ind_st_b = np.intersect1d(np.where(state_3==0)[0],b_3)
                ind_st_a = np.intersect1d(np.where(state_3==1)[0],b_3)
                
                # ind_1_b = ind_st_b[::2]
                # ind_2_b =  ind_st_b[1::2]
                
                # ind_1_a = ind_st_a[::2]
                # ind_2_a =  ind_st_a[1::2]
          
                ind_1 = np.concatenate((ind_1_b,ind_1_a))
                ind_2 = np.concatenate((ind_2_b,ind_2_a))

                if first_half == 1:
                    rewards_3 = rewards_3[ind_1] 
                    choices_3 = choices_3[ind_1]    
                    value_3 = value_3[ind_1]
                    ones_3  = ones_3[ind_1]
                    firing_rates_3 = firing_rates_3[ind_1]   
                    
                    
                    
                elif first_half == 2:
                  
                    rewards_3 = rewards_3[ind_2] 
                    choices_3 = choices_3[ind_2]    
                    value_3 = value_3[ind_2]
                    ones_3  = ones_3[ind_2]
                    firing_rates_3 = firing_rates_3[ind_2]   

             
                   
            ind = np.arange(len(ones_3)) 
            ind_1 = ind[::2]
            ind_2 =  ind[1::2]
           
        
            # if first_half == 1:
                
            #     rewards_3 = rewards_3[ind_1] 
            #     choices_3 = choices_3[ind_1]    
            #     value_3 = value_3[ind_1]
            #     ones_3  = ones_3[ind_1]
            #     firing_rates_3 = firing_rates_3[ind_1]
         
            # elif first_half == 2:
               
            #     rewards_3 = rewards_3[ind_2] 
            #     choices_3 = choices_3[ind_2]    
            #     value_3 = value_3[ind_2]
            #     ones_3  = ones_3[ind_2]
            #     firing_rates_3 = firing_rates_3[ind_2]
           
  
            predictors_all = OrderedDict([
                                        ('Rew', rewards_3),
                                        ('Value',value_3),                
                                        ('ones', ones_3)])
            
            X_3 = np.vstack(predictors_all.values()).T[:trials_3,:].astype(float)
            y_3 = firing_rates_3.reshape([len(firing_rates_3),-1]) # Activity matrix [n_trials, n_neurons*n_timepoints]
            tstats,x = regression_code_session(y_3, X_3)
           # tstats =  reg_f.regression_code(y_3, X_3)
    
            C_3.append(tstats.reshape(n_predictors,n_neurons,n_timepoints)) # Predictor loadings
           
        C_1 = np.concatenate(C_1,1)
        C_2 = np.concatenate(C_2,1)
        C_3 = np.concatenate(C_3,1)
    
      
        C_1_all.append(C_1); C_2_all.append(C_2); C_3_all.append(C_3)

        
    # C_1 = np.concatenate(C_1,1)
    
    # C_2 = np.concatenate(C_2,1)
    
    # C_3 = np.concatenate(C_3,1)
   
     

      
    return C_1_all,C_2_all,C_3_all
