#!/usr/bin/env python
# coding: utf-8


import json
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time
from matplotlib.colors import LogNorm
from scipy.signal import *
from tqdm.notebook import tqdm
import warnings
import json

# If you are using a Jupyter Notebook, uncomment the following line:
# %matplotlib inline

# Local application/library specific imports
from pycrime.data_analysis import average_results
from pycrime.data_analysis.comsol_model_strains_75mm_1mm import *
sys.path.append('/mnt/data/CRIME/results/CSU_TiGemania/')
from match_modes_gmcghee import *

# Settings and configurations
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 14, 'figure.figsize': (10, 6)})


def save_fig(fig_id, tight_layout=True):
    path = fig_id + '.png'
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=100)

def save_fig_pdf(fig_id, tight_layout=True):
    path = fig_id + '.pdf'
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='pdf', dpi=100)

def pcolormesh_logscale(T, F, S):
    plt.pcolormesh(T, F, S, norm=LogNorm(vmin=S.min(), vmax=S.max()))



def stack_values_extractor_replacingvalues(s, base_dir, out_dir, material_dict=None, suspension_info=None, flags=None, textfiles=None, MAX_FREQ_DIFF=None,debugging=False):
    
    
    """
    Extracts values from provided dictionaries and processes them according to
    specified logic. Initializes dictionaries as empty if None and unpacks
    variables for further processing.
    
    
    :  base_dir                 : Directory containing measurement files 
    :  out_dir                  : Output directory path
    :  material_dict            : Dictionary containing material properties
    :  suspension_info          : Dictionary containing suspension information
    :  flags                    : Dictionary of boolean flags controlling function behavior
    :  textfiles                : Dictionary containing paths of text files
    :  max_freq_diff            : Maximum frequency speration used for frequency mode matching to COMSOL 
    :  debugging                : Boolean to show additional print statements
    :  return                   : 
    
    :  f_exp_coated[T,d,s]      : Experimental Frequencies consistent in uncoated, coated and COMSOL data 
    :  m_exp                    : Experimential Mode identification (m , n)
    :  uncoated_ave[s]          : np.array : Experimental uncoated data |Frequency|Q1|Q2|m|n| 
    :  uncoated_std[s]          : np.array : Experimental uncoated data |std(Q1)|std(Q2)| 
    :  coated_ave[T,d][s]       : np.array : Experimental coated data   |Frequency|Q1|Q2|m|n| 
    :  coated_std[T,d][s]       : np.array : Experimental uncoated data |std(Q1)|std(Q2)| 
    :  Dbulk                    : 
    :  Dshear                   : 
    :  comsol_modes             : 
    :  s2ds                     : 
    :  s2delplus6s              : 
    :  elasticProps             :  
    
    
    
    
    """
    
    ################################################################################
                            ### Variable Parsing #### 
    
    
     # Ensure dictionaries are not None and initialize them as empty if they are
    material_dict = material_dict if material_dict is not None else {}
    suspension_info = suspension_info if suspension_info is not None else {}
    flags = flags if flags is not None else {}
    textfiles = textfiles if textfiles is not None else {}

    # Unpack variables from material_dict
    material                        = material_dict.get("material")
    thickness                       = material_dict.get("thickness")
    density                         = material_dict.get("density")
    Y_substrate                     = material_dict.get("Y_substrate")
    nu_substrate                    = material_dict.get("nu_substrate")
    density_substrate               = material_dict.get("density_substrate")
    thickness_substrate             = material_dict.get("thickness_substrate")
    
    
    
    # Unpack variables from suspension_info
    temperature                     = suspension_info.get("temperature")
    duration                        = suspension_info.get("duration")
    date_uncoated                   = suspension_info.get("date_uncoated")
    date_coated                     = suspension_info.get("date_coated")
    suspension_number               = suspension_info.get("suspension_number")
    suspension_label                = suspension_info.get("sus_label")
    
    TF_uncoated                     = flags.get("TF_uncoated") 
    TF_coated                       = flags.get("TF_coated"  ) 
    coated_duplicate_mode_IDS       = flags.get("coated_duplicate_mode_IDS") 
    blank_duplicate_mode_IDS        = flags.get("blank_duplicate_mode_IDS") 
    bad_mode_IDS                    = flags.get("bad_mode_IDS" ) 
    YLIMT                           = flags.get("YLIMT") 
    coatedreplace                   = flags.get("coatedreplace")
    blankreplace                    = flags.get("blankreplace") 
    
    
    
    Q_SUB_txt   = textfiles.get("Q_SUB_txt" )
    ERR_SUB_txt = textfiles.get("ERR_SUB_txt")
    Q_ED_txt    = textfiles.get("Q_ED_txt"  )
    ERR_ED_txt  = textfiles.get("ERR_ED_txt")



    ##Setup secondary variables
    serials             = [s]
    temperatures        = [temperature]         #need list type for later
    durations           = [duration]            #need list type for later
    thickness           = dict(zip(serials, thickness))
    material            = dict(zip(serials, material))
    density             = dict(zip(serials, density))
    Y_substrate         = dict(zip(serials, Y_substrate))
    nu_substrate        = dict(zip(serials, nu_substrate))
    density_substrate   = dict(zip(serials, density_substrate))
    thickness_substrate = dict(zip(serials, thickness_substrate))

    ############################################################################
     

    #defined here instead to make cutting parts work in this location in code
    uncoated_ave = {}
    uncoated_std = {}
   
    coated_ave = {}
    coated_std = {}
    
    
    T = temperature
    d = duration
    

    if blankreplace == ['no']:

        Qsub          = np.loadtxt(Q_SUB_txt, delimiter='\t')
        stdsub        = np.loadtxt(ERR_SUB_txt, delimiter='\t')
    
        
        if s not in uncoated_ave:
            uncoated_ave[(T, d)] = {}

        # Assign Qed to the inner dictionary using key s
        uncoated_ave[s] = Qsub

        # Repeat the process for coated_std
        if s not in uncoated_std:
            uncoated_std[(T, d)] = {}
        uncoated_std[s] = stdsub
    
    
    if coatedreplace == ['no']:
        
        Qed = np.loadtxt(Q_ED_txt, delimiter='\t')
        stded = np.loadtxt(ERR_ED_txt, delimiter='\t')
        
        if (T, d) not in coated_ave:
            coated_ave[(T, d)] = {}

        # Assign Qed to the inner dictionary using key s
        coated_ave[(T, d)][s] = Qed

        # Repeat the process for coated_std
        if (T, d) not in coated_std:
            coated_std[(T, d)] = {}
        coated_std[(T, d)][s] = stded
            
    
    #functions to filter out duplicate modes measured on one suspension blank or coated
    if (T, d) in coated_ave and (T, d) in coated_std and s in coated_ave[(T, d)] and s in coated_std[(T, d)]:
        coated_av_dup, coated_std_dup = coated_duplicate_modes_remover_gmcghee(coated_ave[T,d][s], coated_std[T,d][s], coated_duplicate_mode_IDS, debugging)
    else:
        print(f"Key error with T={T}, d={d}, s={s}")
        
    
    coated_ave[T,d][s] = coated_av_dup
    coated_std[T,d][s] = coated_std_dup
    
    
    if s in uncoated_ave and s in uncoated_std:
       
        blank_av_dup, blank_std_dup = blank_duplicate_modes_remover_gmcghee(uncoated_ave[s], uncoated_std[s], blank_duplicate_mode_IDS, debugging
        )
    else:
        # If `s` is not found in either dictionary, print an error message or handle the case appropriately

        print(f"\nKey '{s}' not found in one or both dictionaries.")    
        
    uncoated_ave[s] = blank_av_dup
    uncoated_std[s] = blank_std_dup
    
    #function to match uncoated and coated mode data AND crop bad measurements
    if debugging:  
        print("Inputs:")
        print("uncoated_ave[s]:", uncoated_ave[s])
        print("coated_ave[T,d][s]:", coated_ave[T,d][s])
        print("uncoated_std[s]:", uncoated_std[s])
        print("coated_std[T,d][s]:", coated_std[T,d][s])
        print("bad_mode_IDS:", bad_mode_IDS)
    
    

    #############################
    
    
    
    
    matched_data_dict, idx_uncoated, idx_coated= eff.preprocess_and_match(uncoated_ave[s], coated_ave[T,d][s], uncoated_std[s], coated_std[T,d][s], bad_mode_IDS,debugging)
    
    #unpack variables 
    m_exp              = matched_data_dict["m_exp"]
    m1_exp             = matched_data_dict["m1_exp"]
    m2_exp             = matched_data_dict["m2_exp"]

    f_exp_uncoated     = matched_data_dict["f_exp_uncoated"]
    Q1_exp_uncoated    = matched_data_dict["Q1_exp_uncoated"]
    Q2_exp_uncoated    = matched_data_dict["Q2_exp_uncoated"]
    Q1err_exp_uncoated = matched_data_dict["Q1err_exp_uncoated"]
    Q2err_exp_uncoated = matched_data_dict["Q2err_exp_uncoated"]

    uncoated_ave_final = matched_data_dict["uncoated_ave_final"]
    uncoated_std_final = matched_data_dict["uncoated_std_final"]
    coated_ave_final   = matched_data_dict["coated_ave_final"]
    coated_std_final   = matched_data_dict["coated_std_final"]
    

    #f_exp_uncoated, f_exp_coated, m_exp, idx_uncoated, idx_coated, uncoated_ave_final, coated_ave_final, uncoated_std_final, coated_std_final = match_modes_gmcghee(uncoated_ave[s], coated_ave[T,d][s], uncoated_std[s], coated_std[T,d][s], bad_mode_IDS,debugging)
    
    
    
    
    
    if debugging:
        print("\nOutputs:")
        print("f_exp_uncoated:", f_exp_uncoated)
        print("f_exp_coated:", f_exp_coated)
        print("m_exp:", m_exp)
        print("idx_uncoated:", idx_uncoated)
        print("idx_coated:", idx_coated)
        print("uncoated_ave_final:", uncoated_ave_final)
        print("coated_ave_final:", coated_ave_final)
        print("uncoated_std_final:", uncoated_std_final)
        print("coated_std_final:", coated_std_final)
        
    uncoated_ave[s]    = uncoated_ave_final
    uncoated_std[s]    = uncoated_std_final
    coated_ave[T,d][s] = coated_ave_final
    coated_std[T,d][s] = coated_std_final


    ##Single loss angle analysis

    Y  = {}
    nu = {}
    e  = {}
    dfreqs_model = {}
    f_exp_uncoated = {}
    f_exp_coated = {}
    modes = {}

    # Calculating elastic properties using simple interpolation fit 
    
    for s in serials:
        for T in temperatures:
            for d in durations:
                sys.stdout.write('%s %3d C %3d h' % (s,T, d))
                Y[T,d,s], nu[T,d,s], e[T,d,s], dfreqs_model[T,d,s], \
                       f_exp_uncoated[T,d,s], f_exp_coated[T,d,s], modes[T,d,s] = fit_coated_disk(uncoated_ave[s], coated_ave[T,d][s], 1e9*thickness[s], \
                                                                                                           density[s], verbose=False, p0=[50, 0.28], edge_bounds=[0, 1.0],\
                                                                                                           Y_bounds=[20, 300], nu_bounds=[0, 0.5], return_modes=True, \
                                                                                                           th_sub=thickness_substrate[s], nu_sub=nu_substrate[s],max_freq_diff=MAX_FREQ_DIFF)
                sys.stdout.write('\tY = %5.1f GPa    nu = %5.3f    edge = %5.3f mm\n' % (Y[T,d,s], nu[T,d,s], e[T,d,s]));





    # If you are using a Jupyter Notebook, uncomment the following line:
    # %matplotlib inline

    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8, 6), sharex=True, sharey=True)

    for i, s in enumerate(serials):
        plt.plot(f_exp_uncoated[T, d, s], (f_exp_coated[T, d, s] - f_exp_uncoated[T, d, s]) * 7.5e-3, 'o', markerfacecolor='none', markeredgewidth=3, markersize=7, label='Experimental')
        plt.plot(f_exp_uncoated[T, d, s], dfreqs_model[T, d, s] * 7.5e-3, 'x', markerfacecolor='none', markeredgewidth=3, markersize=7, label='Best fit', zorder=10)
        plt.grid()
        plt.legend(fontsize=10, loc='upper left')
        plt.title(f"{s} {T}°C {d} h", fontsize=10)
        plt.ylabel('Frequency shift [Hz]')
        plt.xlabel('Mode frequency [Hz]')

    fig.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{s}", f"{s}{suspension_label}FrequencyShift_Vs_COMSOL.png"), facecolor='white')


#duplicate and mode match used to be here



    Yb  = {}
    nub = {}
    eb  = {}
    samples = {}
    for s in serials:
        for T in temperatures:
            for d in durations:
                sys.stdout.write('%s %3d C %3d h' % (s,T,d))
                Yb[T,d,s], nub[T,d,s], eb[T,d,s], samples[T,d,s] =  bayesian_comsol(uncoated_ave[s], coated_ave[T,d][s], 
                                                                                    1e9*thickness[s], density[s], Y[T,d,s], nu[T,d,s], e[T,d,s],
                                                                                    edge_bounds=[0, 1.5], Y_bounds=[20, 300], nu_bounds=[-0.5, 0.5], 
                                                                                    th_sub=thickness_substrate[s], nu_sub=nu_substrate[s])
                sys.stdout.write('\tY = %5.1f ± %3.1f GPa, nu = %6.3f ± %5.3f, edge = %6.3f ± %5.3f mm \n' % (Yb[T,d,s][1], 0.5*(Yb[T,d,s][2]-Yb[T,d,s][0]), \
                                                                                      nub[T,d,s][1], 0.5*(nub[T,d,s][2]-nub[T,d,s][0]), \
                                                                                      eb[T,d,s][1], 0.5*(eb[T,d,s][2]-eb[T,d,s][0])));








    # compute the coating loss angle using measurements and dilution from COMSOL simulations

    frequency_ = {}
    loss_angle_ = {}
    loss_angle_err_ = {}

    for s in serials:
        for T in temperatures:
            for d in durations:
                frequency_[T,d,s], loss_angle_[T,d,s], loss_angle_err_[T,d,s], Dbulk, Dshear, comsol_modes, s2ds, s2delplus6s = compute_coating_lossangle_gmcghee(\
               coated_ave[T,d][s],     # averaged coated Q values
               uncoated_ave[s],   # averaged uncoated Q values
               coated_std[T,d][s],     # error on coated Q values
               uncoated_std[s],   # error on uncoated Q values
               Yb[T, d, s][1], nub[T,d,s][1], 1e9*thickness[s], density[s], eb[T,d,s][1], 
               th_sub=thickness_substrate[s], nu_sub=nu_substrate[s]) # coating properties




    #PLOT_GRAPH AND SAVE EXCEL OUTPUT
    matplotlib.rcParams.update({'font.size': 10})
    k = list(frequency_.keys())
    k.sort()

    for i, m in enumerate(k):
        if m[0] == 30:
            label = ' as depos.'
        else:
            label = f' ann. {m[0]}$^\circ$C ({m[1]} h)'

        phi     = numpy.array([loss_angle_[m][i,j] for i,j in enumerate(numpy.argmin(loss_angle_[m], axis=1))])
        phi_err = numpy.array([loss_angle_err_[m][i,j] for i,j in enumerate(numpy.argmin(loss_angle_[m], axis=1))])

        freqm = frequency_[m]
        loss_array = np.array([freqm, phi, phi_err]).T
        samp_name = k[i][2]
        samp_temp = f"{k[i][0]}C_"
        samp_hrs = f"{k[i][1]}hr_"
        suspension = f"S{suspension_number}"
        df = pd.DataFrame(loss_array)
        
        file_dir = os.path.join(out_dir, f"{s}")
        os.makedirs(file_dir, exist_ok=True)
        filepath = os.path.join(file_dir, f"{s}_coating loss{suspension_label}.xlsx")
        df.to_excel(filepath, index=False)

        fig, ax = plt.subplots()
        ax.errorbar(1e-3 * freqm, 1e3 * phi, yerr=1e3 * phi_err, color='green', markerfacecolor='none', marker='s', linewidth=0, elinewidth=2, capsize=5, label=m[-1] + label)
        ax.legend(fontsize=9, ncol=2, loc='lower left')
        ax.set_ylabel('Loss angle [$\\times 10^{-3}$]')
        ax.set_xlabel('Frequency [kHz]')
        ax.set_yscale('log')
        ax.grid(axis='y')

        fig_save_path = os.path.join(file_dir, f"{s}{suspension_label}CoatingLoss.png")
        plt.savefig(fig_save_path, facecolor='white')
        # time.sleep(5) # Uncomment if needed for debugging
        plt.show()
        
        #matplotlib.pyplot.gca().yaxis.grid(True) #input to try to make only y grid but ended up making 2nd figure
    
        
    key = f"{temperature}_{duration}"

    # Prepare the elastic properties dictionary
    elasticProps = {
        'temperature': temperature,
        'duration': duration,
        'Y': Y,
        'nu': nu,
        'e': e,
        'Yb': [Yb[T,d,s][1], 0.5 * (Yb[T,d,s][2] - Yb[T,d,s][0])],
        'nub': [nub[T,d,s][1], 0.5 * (nub[T,d,s][2] - nub[T,d,s][0])],
        'eb': [eb[T,d,s][1], 0.5 * (eb[T,d,s][2] - eb[T,d,s][0])]
    }

    json_path = os.path.join(out_dir, s, f'{s}_MaterialProperties.json')

    # Initialize or load existing data
    try:
        with open(json_path, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    # Convert and merge new data
    converted_data = eff.convert_tuple_keys(elasticProps)
    existing_data[key] = converted_data

    # Serialize the updated dictionary to JSON
    json_str = json.dumps(existing_data, indent=4)

    # Optional debugging printout
    if debugging:
        print('Info Written to Materials JSON\n')
        print(json_str)

    # Write the updated JSON string to the file
    with open(json_path, 'w') as file:
        file.write(json_str)
    
    
       

    return f_exp_coated[T,d,s], m_exp, uncoated_ave[s], uncoated_std[s], coated_ave[T,d][s], coated_std[T,d][s], Dbulk, Dshear, comsol_modes, s2ds, s2delplus6s ,elasticProps 




















def compute_coating_lossangle_gmcghee(c_ave, u_ave, c_std, u_std, coa_Y, coa_nu, coa_thickness, coa_density, coa_edge,\
                              Y_sub=73.2, nu_sub=0.164, th_sub=1.0, rho_sub=2220.):
    """
    SAME AS PREVIOUS FUNCTION MADE BY G VAJENTE BUT NOW USES mode_match_gmcghee function and has additional input bad_mode_IDS
    
    Compute the coating loss angles, from meaured Q values before and after coating, and 
    computing the dilution factors using COMSOL modle polynomial fit.
    
    Parameters
    ----------
    c_ave: numpy.array        averaged Q values for the coated sample
    u_ave: numpy.array        averaged Q values for the uncoated sample
    c_std: numpy.array        uncertainties of Q values for the coated sample
    u_std: numpy.array        uncertainties of  Q values for the uncoated sample   
    coa_Y: float              Young's modulus of the film [GPa]
    coa_nu: float             Poisson ratio of the film
    coa_thickness: float      film thickness [nm]
    coa_density: float        film density [kg/m^3]
    coa_edge: float           edge size [mm]
    Y_sub: float              substrate Young's modulus [GPa]
    nu_sub: float             substrate Poisson ratio
    th_sub: float             substrate thickness [mm]
    rho_sub: float            substrate density [kg/m^3]
        
    Returns
    -------
    fr: numpy.array                 mode frequencies
    phi: numpy.array                loss angle values
    phi_err: numpy.array            loss angle uncertainties
    Dbulk: numpy.array              bulk dilution factors
    Dshear: numpy.array             shear dilution factors
    modes: numpy.array              modes id
    S2_Ds: numpy.array              S_{D,s}^2 strain, integrated over the surface of the substrate, 
                                    as needed in equation 3.3.3 of M. Fejer T2100186
                                    Returned only if return_strains==True
    S2_Delta6s: numpy.array         (S_{\Delta,s}^2 + S_{6,s}^2) strain, integrated over the surface of the substrate, 
                                    as needed in equation 3.3.3 of M. Fejer T2100186
                                    Returned only if return_strains==True
    
    """

   
   
    # match modes and compute dilution factors
    f_exp_uncoated, f_exp_coated, m_exp, idx_uncoated, idx_coated = match_modes(u_ave, c_ave)
    dfreqs, Dbulk, Dshear, comsol_modes, s2ds, s2delplus6s = fitted_model_comsol_strains(coa_Y, coa_nu, coa_thickness, coa_density, coa_edge, \
                                                                      Y_sub=Y_sub, nu_sub=nu_sub, \
                                                                      th_sub=th_sub, rho_sub=rho_sub, return_strains = True)
    dilution_comsol = Dshear + Dbulk
    
    idx_comsol = [where( (m[0] == comsol_modes[:,0])*(m[1] == comsol_modes[:,1]))[0][0] for m in m_exp]
    dilution = dilution_comsol[idx_comsol]
    
    # compute loss angle and error
    phi_c = 1/c_ave[idx_coated,1:3]
    phi_c_err = (c_std[idx_coated,1:3]/c_ave[idx_coated,1:3]) * phi_c
    phi_u = 1/u_ave[idx_uncoated,1:3]
    phi_u_err = (u_std[idx_uncoated,1:3]/u_ave[idx_uncoated,1:3]) * phi_u
    dilution = dilution.reshape(-1,1)
    phi = 1/dilution * phi_c - (1-dilution)/dilution * phi_u
    phi_err = sqrt( (1/dilution * phi_c_err)**2 + ((1-dilution)/dilution * phi_u_err)**2 )

    return f_exp_coated, phi, phi_err, Dbulk, Dshear, comsol_modes, s2ds, s2delplus6s

# In[ ]:




