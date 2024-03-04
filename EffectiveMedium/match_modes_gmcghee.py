#!/usr/bin/env python
# coding: utf-8

# In[3]:
import warnings
warnings.filterwarnings('ignore')
import numpy as np 

from pylab import *
from matplotlib.colors import LogNorm
import os
import time
from scipy.signal import *
import nds2
import pickle
import sys
from tqdm.notebook import tqdm
import pandas as pd

import sys
# Adjust the path to the directory containing your modules
sys.path.append('/mnt/data/CRIME/results/CSU_TiGemania/EffectiveMedium')

import EffectiveMediuim as eff

#test update comment ignore




def match_modes_gmcghee(uncoated_ave, coated_ave, uncoated_std, coated_std, bad_mode_IDS,debugging):
    """
    An edited version of a previous code by g vajente.  
    This one also matched the STDEVS for cropping when comparing uncoated and coated measurements
    It also allows for modes that were "bad" measurements (too low Q/ too high loss/ only 1 measurement etcetc) 
    to be removed even if the mode was measured in both datasets - the previous code only got rid of modes that
    didn't appear in uncoated but still appeared in coated and visa versa. 
    This allows for more user control of what data gets analysed, without having to delete or edit big sets of raw data
    
    
    
    Find the matching modes in the two lists. Returns the index vectors for the two lists.
    
    
    Parameters
    ----------
    uncoated_ave: numpy.array
        uncoated disk results
    coated_ave: numpy.array
        coated disk results

    Returns
    -------
    f_exp_uncoated: numpy.array
        experimental frequencies matched to each other (uncoated disk)
    f_exp_coated: numpy.array
        experimental frequencies matched to each other (coated disk)
    modes: numpy.array
        list of corresponding modes
    idx_uncoated: numpy.array
        list of matching indexes from the uncoated results
    idx_coated: numpy.array
        list of matching indexes from the coated results
    """   
    
    
    c_av = coated_ave        #this should be input with a "coated_ave[T,d][s]" into function
    c_std = coated_std       #this should be input with a "coated_std[T,d][s]" into function
    u_av = uncoated_ave      #this should be input with a "uncoated_ave[s]" into function
    u_std = uncoated_std     #this should be input with a "uncoated_std[s]" into function
    
    
    if debugging:   
            # Print information about c_av
            
            eff.debug_var(c_av,'c_av')
            eff.debug_var(u_av,'u_av')
            eff.debug_var(u_std,'u_std')


    
    

    C_AV = np.array((c_av[:,0],c_av[:,1],c_av[:,2],c_std[:,0],c_std[:,1],c_av[:,3],c_av[:,4])).T
    U_AV = np.array((u_av[:,0],u_av[:,1],u_av[:,2],u_std[:,0],u_std[:,1],u_av[:,3],u_av[:,4])).T
    if debugging: 
        print('*****Mode Matching ****')
        print('Inputs')
        print()
        eff.debug_var(C_AV,'C_AV')
        eff.debug_var(U_AV,'U_AV')
    


    def find_matches_and_return_smaller(C_AV, U_AV):
        C_AV_last_two_rounded = np.round(C_AV[:, -2:])
        U_AV_last_two_rounded = np.round(U_AV[:, -2:])

        # Initialize lists for matching indices
        idx_c_to_u = []  # Indices in C_AV that have matches in U_AV
        idx_u_to_c = []  # Indices in U_AV that have matches in C_AV

        # Find matches from C_AV to U_AV
        for i, c_row in enumerate(C_AV_last_two_rounded):
            for j, u_row in enumerate(U_AV_last_two_rounded):
                if np.array_equal(c_row, u_row):
                    idx_c_to_u.append(i)


        # Find matches from U_AV to C_AV
        for i, u_row in enumerate(U_AV_last_two_rounded):
            for j, c_row in enumerate(C_AV_last_two_rounded):
                if np.array_equal(u_row, c_row):
                    idx_u_to_c.append(i)


        # Compare sizes and return the smaller set of indices
        if len(C_AV) < len(U_AV):
            print('hello')
            U_AV_updated = np.array((U_AV[idx_c_to_u,0],U_AV[idx_c_to_u,1],U_AV[idx_c_to_u,2],U_AV[idx_c_to_u,3],U_AV[idx_c_to_u,4],U_AV[idx_c_to_u,5],U_AV[idx_c_to_u,6])).T
            C_AV_updated = np.array((C_AV[idx_c_to_u,0],C_AV[idx_c_to_u,1],C_AV[idx_c_to_u,2],C_AV[idx_c_to_u,3],C_AV[idx_c_to_u,4],C_AV[idx_c_to_u,5],C_AV[idx_c_to_u,6])).T
    
            return np.array(idx_c_to_u), 'C_AV_to_U_AV', U_AV_updated,C_AV_updated
        else:
            print('wassup')
            U_AV_updated = np.array((U_AV[idx_u_to_c,0],U_AV[idx_u_to_c,1],U_AV[idx_u_to_c,2],U_AV[idx_u_to_c,3],U_AV[idx_u_to_c,4],U_AV[idx_u_to_c,5],U_AV[idx_u_to_c,6])).T
            C_AV_updated = np.array((C_AV[idx_u_to_c,0],C_AV[idx_u_to_c,1],C_AV[idx_u_to_c,2],C_AV[idx_u_to_c,3],C_AV[idx_u_to_c,4],C_AV[idx_u_to_c,5],C_AV[idx_u_to_c,6])).T
            return np.array(idx_u_to_c), 'U_AV_to_C_AV', U_AV_updated,C_AV_updated


    matching_indices, match_direction,U_AV_updated,C_AV_updated = find_matches_and_return_smaller(C_AV, U_AV)
    
    if debugging:
        
        print('After Mode Matching')
        print()
        print(f"Smaller matching indices: {matching_indices}")
        print(f"Direction of match: {match_direction}")

        eff.debug_var(U_AV_updated,'U_AV_updated')
        eff.debug_var(C_AV_updated,'C_AV_updated')
        
    
    raise
    if debugging:
        # Print matching indices
        
        print("Matching Indicies ")
        print()
        eff.debug_var(idx_uncoated,'idx_uncoated')
        eff.debug_var(idx_coated,'idx_coated')
        eff.debug_var(idx_coated,'idx_coated')
        eff.debug_var(U_AV,'U_AV')
        eff.debug_var(C_AV,'C_AV')
        
        
    """
    # find matching modes in the experimental data
    idx_coated   = []
    idx_uncoated = []
    for i,c in enumerate(C_AV):

        idx = np.where(np.logical_and(U_AV[:,-2] == c[-2], U_AV[:,-1] == c[-1]))[0]
        if len(idx):
            idx = idx[0]
            idx_coated.append(i)
            idx_uncoated.append(idx)
    

  
    
    temp_result = ismember(idx_coated, idx_uncoated)
    idx_coated, idx_uncoated = temp_result, temp_result
    
    
    idx_coated   = np.array(idx_coated)
    idx_uncoated = np.array(idx_uncoated)
    
    if debugging:
        print(f"idx_coated  {idx_coated}")
        print(f"idx_uncoated  {idx_uncoated}")
        print()
        print(np.shape(C_AV))
            
        print(np.shape(idx_uncoated))
        
        
    """
    
    U_AV_updated = np.array((U_AV[idx_uncoated,0],U_AV[idx_uncoated,1],U_AV[idx_uncoated,2],U_AV[idx_uncoated,3],U_AV[idx_uncoated,4],U_AV[idx_uncoated,5],U_AV[idx_uncoated,6])).T
    C_AV_updated = np.array((C_AV[idx_uncoated,0],C_AV[idx_uncoated,1],C_AV[idx_uncoated,2],C_AV[idx_uncoated,3],C_AV[idx_uncoated,4],C_AV[idx_uncoated,5],C_AV[idx_uncoated,6])).T
    

    print(f"U_AV_updated {pd.DataFrame(U_AV_updated)}")   
    print(f"C_AV_updated {pd.DataFrame(C_AV_updated)}")

    
    #bad_modes_to_remove = [3,5]
    #bad_modes_to_remove = ['none']
    
    print(bad_mode_IDS)
    bad_modes_to_remove = bad_mode_IDS
    
    
    
    
    if bad_modes_to_remove != ['none']:
        
        U_AV_updated_CUT     = np.delete(U_AV_updated, bad_modes_to_remove, axis = 0)
        C_AV_updated_CUT     = np.delete(C_AV_updated, bad_modes_to_remove, axis = 0)

        u_CUTindexlength     = len(idx_uncoated)-len(bad_modes_to_remove)
        c_CUTindexlength     = len(idx_uncoated)-len(bad_modes_to_remove)

        m_exp                = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength], -2:]   # experimental mode ids, both
        m1_exp               = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],5]      # experimental mode ids, 1st, both
        m2_exp               = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],6]      # experimental mode ids, 2nd, both

        f_exp_uncoated       = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],0]      # experimental frequencies, uncoated
        Q1_exp_uncoated      = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],1]      # experimental Q1s, uncoated
        Q2_exp_uncoated      = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],2]      # experimental Q2s, uncoated
        Q1err_exp_uncoated   = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],3]      # experimental Q1 errors, uncoated
        Q2err_exp_uncoated   = U_AV_updated_CUT[idx_uncoated[:u_CUTindexlength],4]      # experimental Q2 errors, uncoated


        f_exp_coated         = C_AV_updated_CUT[idx_coated[:c_CUTindexlength],0]        # experimental frequencies, coated
        Q1_exp_coated        = C_AV_updated_CUT[idx_coated[:c_CUTindexlength],1]        # experimental Q1s , coated
        Q2_exp_coated        = C_AV_updated_CUT[idx_coated[:c_CUTindexlength],2]        # experimental Q2s, coated
        Q1err_exp_coated     = C_AV_updated_CUT[idx_coated[:c_CUTindexlength],3]        # experimental Q1 errors, coated
        Q2err_exp_coated     = C_AV_updated_CUT[idx_coated[:c_CUTindexlength],4]        # experimental Q2 errors, coated
        
        idx_uncoated         = idx_uncoated[:u_CUTindexlength]
        idx_coated           = idx_coated[:c_CUTindexlength]


        uncoated_ave_final   = np.array((f_exp_uncoated, Q1_exp_uncoated, Q2_exp_uncoated, m1_exp, m2_exp)).T
        uncoated_std_final   = np.array((Q1err_exp_uncoated, Q2err_exp_uncoated)).T

        coated_ave_final     = np.array((f_exp_coated, Q1_exp_coated, Q2_exp_coated, m1_exp, m2_exp)).T
        coated_std_final     = np.array((Q1err_exp_coated, Q2err_exp_coated)).T




    else:
        m_exp                = U_AV_updated[:, -2:]       # experimental mode ids, both
        m1_exp               = U_AV_updated[:,5]          # experimental mode ids, 1st, both
        m2_exp               = U_AV_updated[:,6]          # experimental mode ids, 2nd, both

        f_exp_uncoated       = U_AV_updated[:,0]          # experimental frequencies, uncoated
        Q1_exp_uncoated      = U_AV_updated[:,1]          # experimental Q1s, uncoated
        Q2_exp_uncoated      = U_AV_updated[:,2]          # experimental Q2s, uncoated
        Q1err_exp_uncoated   = U_AV_updated[:,3]          # experimental Q1 errors, uncoated
        Q2err_exp_uncoated   = U_AV_updated[:,4]          # experimental Q2 errors, uncoated

        f_exp_coated         = C_AV_updated[:,0]            # experimental frequencies, coated
        Q1_exp_coated        = C_AV_updated[:,1]            # experimental Q1s , coated
        Q2_exp_coated        = C_AV_updated[:,2]            # experimental Q2s, coated
        Q1err_exp_coated     = C_AV_updated[:,3]            # experimental Q1 errors, coated
        Q2err_exp_coated     = C_AV_updated[:,4]            # experimental Q2 errors, coated


        uncoated_ave_final   = np.array((f_exp_uncoated, Q1_exp_uncoated, Q2_exp_uncoated, m1_exp, m2_exp)).T
        uncoated_std_final   = np.array((Q1err_exp_uncoated, Q2err_exp_uncoated)).T

        coated_ave_final     = np.array((f_exp_coated, Q1_exp_coated, Q2_exp_coated, m1_exp, m2_exp)).T
        coated_std_final     = np.array((Q1err_exp_coated, Q2err_exp_coated)).T
        
        if debugging: 
            
            print(f"Experimental Mode IDs (both): {m_exp}")
            print(f"Experimental Mode ID 1 (both): {m1_exp}")
            print(f"Experimental Mode ID 2 (both): {m2_exp}\n")

            print(f"Experimental Frequencies (Uncoated): {f_exp_uncoated}")
            print(f"Experimental Q1s (Uncoated): {Q1_exp_uncoated}")
            print(f"Experimental Q2s (Uncoated): {Q2_exp_uncoated}")
            print(f"Experimental Q1 Errors (Uncoated): {Q1err_exp_uncoated}")
            print(f"Experimental Q2 Errors (Uncoated): {Q2err_exp_uncoated}\n")

            print(f"Experimental Frequencies (Coated): {f_exp_coated}")
            print(f"Experimental Q1s (Coated): {Q1_exp_coated}")
            print(f"Experimental Q2s (Coated): {Q2_exp_coated}")
            print(f"Experimental Q1 Errors (Coated): {Q1err_exp_coated}")
            print(f"Experimental Q2 Errors (Coated): {Q2err_exp_coated}\n")

            print(f"Uncoated Average Final: {uncoated_ave_final}")
            print(f"Uncoated Standard Deviation Final: {uncoated_std_final}\n")

            print(f"Coated Average Final: {coated_ave_final}")
            print(f"Coated Standard Deviation Final: {coated_std_final}")

    
    
    return f_exp_uncoated, f_exp_coated, m_exp, idx_uncoated, idx_coated, uncoated_ave_final, coated_ave_final, uncoated_std_final, coated_std_final



#functions to filter out duplicate modes measured on one suspension blank or coated before matching
def coated_duplicate_modes_remover_gmcghee(coated_ave, coated_std, coated_duplicate_mode_IDS,debugging=False):
    """
    Remove duplicate modes from coated average and standard deviation arrays.
    @param coated_ave - Coated average array
    @param coated_std - Coated standard deviation array
    @param coated_duplicate_mode_IDS - Array of duplicate mode IDs
    @param debugging - Flag to enable debugging mode
    
    @return  blank_av_dup, blank_std_dup
    """
    c_av  = coated_ave       #this should be input with a "coated_ave[T,d][s]" into function
    c_std = coated_std      #this should be input with a "coated_std[T,d][s]" into function
    
    if debugging:
        print(f"Function Name: coated_duplicate_modes_remover_gmcghee")
        
        eff.debug_var(c_av,"c_av")
        eff.debug_var(c_std ,"c_std")
        eff.debug_var(coated_duplicate_mode_IDS,"coated_duplicate_mode_IDS")
    
    
    if coated_duplicate_mode_IDS != ['none']:
       coated_av_dup              = np.delete(c_av, coated_duplicate_mode_IDS, axis = 0)
       coated_std_dup             = np.delete(c_std, coated_duplicate_mode_IDS, axis = 0)
    
    else:
        coated_av_dup = c_av
        coated_std_dup =c_std
        
        if debugging: 
            eff.debug_var(coated_av_dup,'coated_av_dup')
            eff.debug_var(coated_std_dup,'coated_std_dup')
            
            
    return coated_av_dup, coated_std_dup


def blank_duplicate_modes_remover_gmcghee(uncoated_ave, uncoated_std, blank_duplicate_mode_IDS,debugging=False):
    u_av  = uncoated_ave       #this should be input with a "uncoated_ave[T,d][s]" into function
    u_std = uncoated_std      #this should be input with a "uncoated_std[T,d][s]" into function
    
    if debugging:
        print(f"Function Name: blank_duplicate_modes_remover_gmcghee")
        eff.debug_var(blank_duplicate_mode_IDS,'blank_duplicate_mode_IDS')
        
    
    
    if blank_duplicate_mode_IDS != ['none']:
       blank_av_dup              = np.delete(u_av, blank_duplicate_mode_IDS, axis = 0)
       blank_std_dup             = np.delete(u_std, blank_duplicate_mode_IDS, axis = 0)
    
    else:
        blank_av_dup = u_av
        blank_std_dup =u_std
    
    return blank_av_dup, blank_std_dup
    


# In[ ]:




