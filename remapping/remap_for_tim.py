import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from itertools import combinations 
import scipy
import palettable
import remapping_count as rc 
from scipy import io
import random
font = {'weight' : 'normal',
        'size'   : 2}

from palettable import wesanderson as wes
from scipy.ndimage import gaussian_filter1d

    
def tim_rewrite(area = 1):
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')

    ntrials=20;
    baselength=10;
    #A{s,ch,stype}=[]
    n = 3
    A = [[[ [] for _ in range(n)] for _ in range(n)] for _ in range(n)]
               
    if area==1:
        Data = HP
    else:
        Data = PFC
    neuron_num=0
    
    for  i, ii in enumerate(Data['DM'][0]):
         
        
        DD = Data['Data'][0][i]
        DM = Data['DM'][0][i]
  
        choices = DM[:,1]
        b_pokes = DM[:,7]
        a_pokes = DM[:,6]
        task = DM[:,5]
        taskid = rc.task_ind(task,a_pokes,b_pokes)
        
           
        sw_point=np.where(abs(np.diff(task)>0))[0]+1
        for stype in [1,2,3]: #1 is 1-2; 2 is 1-3; 3 is 2-3
            for s in range(2):
           
                #figure out type of switch. 
                prepost=[taskid[sw_point[s]-2], taskid[sw_point[s]+2]]
                         
                if(sum(prepost)==stype+2):
               
                    #FIND LAST ntrials A before switch and first ntrials as after switch 
                    for ch in [1,2]:
                        Aind=np.where(choices==(ch-1))[0]
               
                        Aind_pre_sw = Aind[Aind<=sw_point[s]]
                        
                        Aind_pre_sw = Aind_pre_sw[-ntrials-baselength-1:]
               
                        Aind_post_sw = Aind[Aind>sw_point[s]]
                        
                        Aind_post_sw = Aind_post_sw[:ntrials]
                       
                        Atrials= np.hstack([Aind_pre_sw,Aind_post_sw])
                
                        A[s][ch-1][stype-1].append((DD[Atrials]))
    return A

def plot_surprise(HP, PFC):
    
    A_HP =  tim_rewrite(area = 1)
    pre_post_b_1_hp,pre_post_a_1_hp,pre_post_b_2_hp,pre_post_a_2_hp,pre_post_b_3_hp,pre_post_a_3_hp = surprise(A_HP)
    A_PFC =  tim_rewrite(area = 2)
    pre_post_b_1_pfc,pre_post_a_1_pfc,pre_post_b_2_pfc,pre_post_a_2_pfc,pre_post_b_3_pfc,pre_post_a_3_pfc = surprise(A_PFC)

      
    cmax = np.max([ pre_post_b_1_hp,pre_post_a_1_hp,pre_post_b_2_hp,pre_post_a_2_hp,pre_post_b_3_hp,pre_post_a_3_hp ,\
                    pre_post_b_1_pfc,pre_post_a_1_pfc,pre_post_b_2_pfc,pre_post_a_2_pfc,pre_post_b_3_pfc,pre_post_a_3_pfc])
    cmin = np.min([pre_post_b_1_hp,pre_post_a_1_hp,pre_post_b_2_hp,pre_post_a_2_hp,pre_post_b_3_hp,pre_post_a_3_hp ,\
                    pre_post_b_1_pfc,pre_post_a_1_pfc,pre_post_b_2_pfc,pre_post_a_2_pfc,pre_post_b_3_pfc,pre_post_a_3_pfc])
        
    #cmap =  palettable.scientific.sequential.GrayC_6.mpl_colormap
    cmap =  palettable.scientific.sequential.Acton_20.mpl_colormap
    #cmap ="bone"
    yt = [10,25,35,42,50,60]
    yl = ['-0.6','Init', 'Ch','R', '+0.32', '+0.72']
    label_size = 5
    plt.rcParams['xtick.labelsize'] = label_size 
    plt.rcParams['ytick.labelsize'] = label_size 
    
    xt = [10,25,35,42,50,60,
                10+63,25+63,35+63,42+63,50+63,60+63,\
                10+63*2,25+63*2,35+63*2,42+63*2,50+63*2,60+63*2,\
                10+63*3,25+63*3,35+63*3,42+63*3,50+63*3,60+63*3]
    xl = ['-0.6','Init', 'Ch','R', '+0.32', '+0.72', '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
          '-0.6','Init', 'Ch','R', '+0.32', '+0.72',\
              '-0.6','Init', 'Ch','R', '+0.32', '+0.72']
    
    hp_1_2 = np.vstack([pre_post_b_1_hp,pre_post_a_1_hp])
    hp_1_3 = np.vstack([pre_post_b_2_hp,pre_post_a_2_hp])
    hp_2_3 = np.vstack([pre_post_b_3_hp,pre_post_a_3_hp ])

    pfc_1_2 = np.vstack([pre_post_b_1_pfc,pre_post_a_1_pfc])
    pfc_1_3 = np.vstack([pre_post_b_2_pfc,pre_post_a_2_pfc])
    pfc_2_3 = np.vstack([pre_post_b_3_pfc,pre_post_a_3_pfc])
         
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
                                                                
    #plt.tight_layout()
    #plt.colorbar()
    
    isl = wes.Royal2_5.mpl_colors
    It = 25
    Ct = 36
    Re = 42
    mmin  =np.min([np.diag(pre_post_a_1_hp[:63]),\
                  np.diag(pre_post_a_1_hp[63:]),\
                 np.diag( pre_post_a_1_pfc[:63]),\
                  np.diag(pre_post_a_1_pfc[63:]),\
                      np.diag(pre_post_a_2_hp[:63]),\
                    np.diag(pre_post_a_2_hp[63:]),\
                 np.diag( pre_post_a_2_pfc[:63]),\
                  np.diag(pre_post_a_2_pfc[63:]),\
                np.diag(pre_post_a_3_hp[:63]),\
                  np.diag(pre_post_a_3_hp[63:]),\
                 np.diag( pre_post_a_3_pfc[:63]),\
                  np.diag(pre_post_a_3_pfc[63:]),
                  
                  np.diag(pre_post_b_1_hp[:63]),\
                  np.diag(pre_post_b_1_hp[63:]),\
                 np.diag( pre_post_b_1_pfc[:63]),\
                  np.diag(pre_post_b_1_pfc[63:]),\
                      np.diag(pre_post_b_2_hp[:63]),\
                    np.diag(pre_post_b_2_hp[63:]),\
                 np.diag( pre_post_b_2_pfc[:63]),\
                  np.diag(pre_post_b_2_pfc[63:]),\
                np.diag(pre_post_b_3_hp[:63]),\
                  np.diag(pre_post_b_3_hp[63:]),\
                 np.diag( pre_post_b_3_pfc[:63]),\
                  np.diag(pre_post_b_3_pfc[63:])])
        
        
    mmax  =np.max([np.diag(pre_post_a_1_hp[:63]),\
                  np.diag(pre_post_a_1_hp[63:]),\
                 np.diag( pre_post_a_1_pfc[:63]),\
                  np.diag(pre_post_a_1_pfc[63:]),\
                      np.diag(pre_post_a_2_hp[:63]),\
                    np.diag(pre_post_a_2_hp[63:]),\
                 np.diag( pre_post_a_2_pfc[:63]),\
                  np.diag(pre_post_a_2_pfc[63:]),\
                np.diag(pre_post_a_3_hp[:63]),\
                  np.diag(pre_post_a_3_hp[63:]),\
                 np.diag( pre_post_a_3_pfc[:63]),\
                  np.diag(pre_post_a_3_pfc[63:]),
                  
                  np.diag(pre_post_b_1_hp[:63]),\
                  np.diag(pre_post_b_1_hp[63:]),\
                 np.diag( pre_post_b_1_pfc[:63]),\
                  np.diag(pre_post_b_1_pfc[63:]),\
                      np.diag(pre_post_b_2_hp[:63]),\
                    np.diag(pre_post_b_2_hp[63:]),\
                 np.diag( pre_post_b_2_pfc[:63]),\
                  np.diag(pre_post_b_2_pfc[63:]),\
                np.diag(pre_post_b_3_hp[:63]),\
                  np.diag(pre_post_b_3_hp[63:]),\
                 np.diag( pre_post_b_3_pfc[:63]),\
                  np.diag(pre_post_b_3_pfc[63:])])
    plt.subplot(2,3,1)
    
    plt.plot(np.diag(pre_post_a_1_hp[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(pre_post_a_1_hp[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(pre_post_a_1_pfc[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(pre_post_a_1_pfc[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
   
    
    plt.legend()
    plt.title('1 2')


    # 1 3 A
    
    plt.subplot(2,3,2)
    
    plt.plot(np.diag(pre_post_a_2_hp[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(pre_post_a_2_hp[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(pre_post_a_2_pfc[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(pre_post_a_2_pfc[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.title('1 3')

    
   #  2 3 A
    
    plt.subplot(2,3,3)
    
    
    plt.plot(np.diag(pre_post_a_3_hp[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(pre_post_a_3_hp[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(pre_post_a_3_pfc[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(pre_post_a_3_pfc[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.title('2 3')


    plt.subplot(2,3,4)
    
    plt.plot(np.diag(pre_post_b_1_hp[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(pre_post_b_1_hp[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(pre_post_b_1_pfc[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(pre_post_b_1_pfc[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')
        
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    plt.legend()


    # 1 3 A
    
    plt.subplot(2,3,5)
    
    plt.plot(np.diag(pre_post_b_2_hp[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(pre_post_b_2_hp[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(pre_post_b_2_pfc[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(pre_post_b_2_pfc[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    #  2 3 A
    
    plt.subplot(2,3,6)
    
    plt.plot(np.diag(pre_post_b_3_hp[:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(np.diag(pre_post_b_3_hp[63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(np.diag(pre_post_b_3_pfc[:63]), color = isl[3], label = 'PFC Within Task')

    plt.plot(np.diag(pre_post_b_3_pfc[63:]), color = isl[3], linestyle = '--', label = 'PFC Between Task')

           
    plt.vlines(It,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Ct,mmin,mmax, color = 'grey', alpha = 0.5)
    plt.vlines(Re,mmin,mmax, color = 'grey', alpha = 0.5)

    y = mmax + 0.02
    sns.despine()
    
    
    plt.figure()
    
    plt.plot(-np.sqrt(pre_post_b_3_hp.T[36,:63]), color = isl[2], label = 'HP Within Task')

    plt.plot(-np.sqrt(pre_post_b_3_hp.T[36,63:]), color = isl[2], linestyle = '--', label = 'HP Between Tasks')

    plt.plot(-np.sqrt(pre_post_b_3_pfc.T[36,:63]),  color = isl[3], label = 'PFC Within Task')

    plt.plot(-np.sqrt(pre_post_b_3_pfc.T[36,63:]),  color = isl[3], linestyle = '--', label = 'PFC Between Task')

      
 


 


def surprise(A):          
    stdmin = 2 
    ntrials = 20;
    baselength = 10;

    for stype in [1,2,3]:
        #plt.figure()

        for ch in [1,2]:
            
            #plt.subplot(1,2,(ch))
            for s in range(2):
                if s == 0:
                    Adim = np.concatenate(A[s][ch-1][stype-1],1).shape
      
                    A_temp = np.concatenate(A[s][ch-1][stype-1],1)
                   # A_temp =  gaussian_filter1d(A_temp.astype(float),2,2)
            
                    mm_base = np.tile(np.mean(A_temp[:baselength,:,:],0).reshape(Adim[1],Adim[2],1),(1, 1, Adim[2]))
                    std_base = np.tile(np.nanstd(A_temp[:baselength,:,:],0, ddof=1).reshape(Adim[1],Adim[2],1),(1, 1, Adim[2]))+stdmin
                   
                    mm_pre = np.tile(np.mean(A_temp[baselength:baselength+int(ntrials/2),:,:],0).reshape(Adim[1],Adim[2],1),(1, 1, Adim[2]))
                    mm_post = np.tile(np.mean(A_temp[baselength+int(ntrials/2):,:,:],0).reshape(Adim[1],Adim[2],1),(1, 1, Adim[2]))
    
        
                    p_pre = np.nanmean(((np.transpose(mm_pre,[0, 2, 1])-mm_base)**2)/(std_base**2),0)
                    p_post = np.nanmean(((np.transpose(mm_post,[0, 2, 1])-mm_base)**2)/(std_base**2),0)
                    pre_post = np.vstack((p_pre.T,p_post.T))
                    
            if stype == 1:
                if ch == 1:
                    pre_post_b_1 = pre_post
                elif ch == 2:
                    pre_post_a_1 = pre_post
            elif stype == 2:
                if ch == 1:
                    pre_post_b_2 = pre_post
                elif ch == 2:
                    pre_post_a_2 = pre_post
                    
            elif stype == 3:
                if ch == 1:
                    pre_post_b_3 = pre_post
                elif ch == 2:
                    pre_post_a_3 = pre_post
    return  pre_post_b_1,pre_post_a_1,pre_post_b_2,pre_post_a_2,pre_post_b_3,pre_post_a_3
                # scipy.io.savemat('/Users/veronikasamborska/Desktop/'+ 'try' + '.mat',{'Data': pre_post})
     
                #plt.imshow(-np.sqrt(pre_post.T))
            
def perm_run():
    
    A_perms_HP = perm_A(n_perms = 500, area = 1)
    
    pre_post_b_1_p_hp = []
    pre_post_a_1_p_hp = []
    pre_post_b_2_p_hp = []
    pre_post_a_2_p_hp = []
    pre_post_b_3_p_hp = []
    pre_post_a_3_p_hp = []
    for A_HP in A_perms_HP:
        pre_post_b_1,pre_post_a_1,pre_post_b_2,pre_post_a_2,pre_post_b_3,pre_post_a_3 = surprise(A_HP)
        pre_post_b_1_p_hp.append(abs(np.diag(pre_post_b_1[:63,:])-np.diag(pre_post_b_1[63:,:])))
        pre_post_a_1_p_hp.append(abs(np.diag(pre_post_a_1[:63,:])-pre_post_a_1[63:,:]))
        pre_post_b_2_p_hp.append(abs(np.diag(pre_post_b_2[:63,:])-pre_post_b_2[63:,:]))
        pre_post_a_2_p_hp.append(abs(np.diag(pre_post_a_2[:63,:])-pre_post_a_2[63:,:]))
        pre_post_b_3_p_hp.append(abs(np.diag(pre_post_b_3[:63,:])-pre_post_b_3[63:,:]))
        pre_post_a_3_p_hp.append(abs(np.diag(pre_post_a_3[:63,:])-pre_post_a_3[63:,:]))
        
    b_1_hp = np.percentile(np.asarray(pre_post_b_1_p_hp),95, axis = 0)
    b_2_hp = np.percentile(np.asarray(pre_post_b_2_p_hp),95, axis = 0)
    b_3_hp = np.percentile(np.asarray(pre_post_b_3_p_hp),95, axis = 0)
     
    a_1_hp = np.percentile(np.asarray(pre_post_a_1_p_hp),95, axis = 0)
    a_2_hp = np.percentile(np.asarray(pre_post_a_2_p_hp),95, axis = 0)
    a_3_hp = np.percentile(np.asarray(pre_post_a_3_p_hp),95, axis = 0)

    A_perms_PFC = perm_A(n_perms = 500, area = 2)

    pre_post_b_1_p_pfc = []
    pre_post_a_1_p_pfc = []
    pre_post_b_2_p_pfc = []
    pre_post_a_2_p_pfc = []
    pre_post_b_3_p_pfc = []
    pre_post_a_3_p_pfc = []
  
    for A_PFC in A_perms_PFC:
        pre_post_b_1,pre_post_a_1,pre_post_b_2,pre_post_a_2,pre_post_b_3,pre_post_a_3 = surprise(A_PFC)
        pre_post_b_1_p_pfc.append(abs(pre_post_b_1[:63,:]-pre_post_b_1[63:,:]))
        pre_post_a_1_p_pfc.append(abs(pre_post_a_1[:63,:]-pre_post_a_1[63:,:]))
        pre_post_b_2_p_pfc.append(abs(pre_post_b_2[:63,:]-pre_post_b_2[63:,:]))
        pre_post_a_2_p_pfc.append(abs(pre_post_a_2[:63,:]-pre_post_a_2[63:,:]))
        pre_post_b_3_p_pfc.append(abs(pre_post_b_3[:63,:]-pre_post_b_3[63:,:]))
        pre_post_a_3_p_pfc.append(abs(pre_post_a_3[:63,:]-pre_post_a_3[63:,:]))

    b_1_pfc = np.percentile(np.asarray(pre_post_b_1_p_pfc),95, axis = 0)
    b_2_pfc = np.percentile(np.asarray(pre_post_b_2_p_pfc),95, axis = 0)
    b_3_pfc = np.percentile(np.asarray(pre_post_b_3_p_pfc),95, axis = 0)
     
    a_1_pfc = np.percentile(np.asarray(pre_post_a_1_p_pfc),95, axis = 0)
    a_2_pfc = np.percentile(np.asarray(pre_post_a_2_p_pfc),95, axis = 0)
    a_3_pfc = np.percentile(np.asarray(pre_post_a_3_p_pfc),95, axis = 0)

    
  
def perm_A(n_perms = 1000, area = 1):
    HP = io.loadmat('/Users/veronikasamborska/Desktop/HP.mat')
    PFC = io.loadmat('/Users/veronikasamborska/Desktop/PFC.mat')
    
    A_perms = []
    for perm in range(n_perms):

      
         
        ntrials=20;
        baselength=10;
        n = 3
        A = [[[ [] for _ in range(n)] for _ in range(n)] for _ in range(n)]
                   
        if area==1:
            Data = HP
        else:
            Data = PFC
        neuron_num=0
        
        for  i, ii in enumerate(Data['DM'][0]):
             
            
            DD = Data['Data'][0][i]
            DM = Data['DM'][0][i]
      
            choices = DM[:,1]
            b_pokes = DM[:,7]
            a_pokes = DM[:,6]
            task = DM[:,5]
            taskid = rc.task_ind(task,a_pokes,b_pokes)
            sw_point=np.where(abs(np.diff(task)>0))[0]+1
            b_s =np.where(choices==(0))[0]
            a_s =np.where(choices==(1))[0]
        
                
                          
            for stype in [1,2,3]: #1 is 1-2; 2 is 1-3; 3 is 2-3
            
                for s in range(2):
               
                    #figure out type of switch. 

                    prepost=[taskid[sw_point[s]-2], taskid[sw_point[s]+2]]
                             
                    if(sum(prepost)==stype+2):
                
                                   
                        #FIND LAST ntrials A before switch and first ntrials as after switch 
                        for ch in [1,2]:
                            
                            while  len(sw_point)<2:
                                task = np.roll(task,np.random.randint(len(task)), axis=0)
                                sw_point = np.where(abs(np.diff(task)>0))[0]+1
                            while len(np.where(b_s > sw_point[0])[0]) < 31 or len(np.where(b_s > sw_point[0])[0])< 31  or\
                                 len(np.where(a_s > sw_point[0])[0])< 31 or len(np.where(a_s > sw_point[0])[0])< 31 or\
                                     len(np.where(b_s > sw_point[1])[0]) < 31 or len(np.where(b_s > sw_point[1])[0])< 31  or\
                                 len(np.where(a_s > sw_point[1])[0])< 31 or len(np.where(a_s > sw_point[1])[0])< 31 or\
                                     len(np.where(b_s <= sw_point[0])[0]) < 31 or len(np.where(b_s <= sw_point[0])[0])< 31  or\
                                 len(np.where(a_s <= sw_point[0])[0])< 31 or len(np.where(a_s <= sw_point[0])[0])< 31 or\
                                     len(np.where(b_s <= sw_point[1])[0]) < 31 or len(np.where(b_s <= sw_point[1])[0])< 31  or\
                                 len(np.where(a_s <= sw_point[1])[0])< 31 or len(np.where(a_s <= sw_point[1])[0])< 31  :
                                    task = np.roll(task,np.random.randint(len(task)), axis=0)
                                    sw_point = np.where(abs(np.diff(task)>0))[0]+1
                                    while  len(sw_point)<2:
                                        task = np.roll(task,np.random.randint(len(task)), axis=0)
                                        sw_point = np.where(abs(np.diff(task)>0))[0]+1
                          
                        
                                                            
                            
                            Aind=np.where(choices==(ch-1))[0]
                   
                            Aind_pre_sw = Aind[Aind<=sw_point[s]]
                            
                            Aind_pre_sw = Aind_pre_sw[-ntrials-baselength-1:]
                   
                            Aind_post_sw = Aind[Aind>sw_point[s]]
                            
                            Aind_post_sw = Aind_post_sw[:ntrials]
                           
                            Atrials= np.hstack([Aind_pre_sw,Aind_post_sw])
                    
                            A[s][ch-1][stype-1].append((DD[Atrials]))
                            if DD[Atrials].shape[0] !=51:
                                print(sw_point, len(Aind_post_sw),len(Aind_pre_sw))
                                
                            
        A_perms.append(A)
    return A_perms

   