#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8



import os
import pickle
import sys
import time
import warnings


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.signal import detrend  # or any specific functions you use from scipy.signal
from tqdm.notebook import tqdm
import json 

# Local application/library specific imports
from pycrime.data_analysis import average_results
from pycrime.data_analysis.comsol_model_strains_75mm_1mm import *
from pycrime.data_analysis.bayesian_loss_angle_75mm_1mm import *
import operator  # Used for identifying best model and Bayesian split of outputs
from match_modes_gmcghee import *

# Configuration and settings
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

def validate_not_none(variable, variable_name):
    """Raise an error if the variable is None."""
    if variable is None:
        raise ValueError(f"{variable_name} must not be None")
    
def check_arguments(**kwargs):
    # Process each keyword argument
    for key, value in kwargs.items():
        if value is None:
            kwargs[key] = {}  # Assign an empty dictionary if the value is None
        elif not isinstance(value, dict):
            raise TypeError(f"Expected '{key}' to be a dictionary or None, got {type(value).__name__} instead.")
    # Return the modified keyword arguments
    return kwargs




def singlelayer_values_extractor_replacingvalues(s, base_dir, out_dir, material_dict, flags, suspension_info, textfiles,MCMC_Params, MAX_FREQ_DIFF,debugging=False):
    

  ################################################################################

    # Ensure dictionaries are not None and initialize them as empty if they are
                            ### Variable Parsing #### 
    
    # Unpack the returned values directly if function is used at the start of another function or process
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
    
    # unpack variables from MCMC_Params 
    model_slope         = MCMC_Params.get('model_slope')
    model_split         = MCMC_Params.get('model_split')
    walkers             = MCMC_Params.get('walkers')
    iterations          = MCMC_Params.get('iterations')
    warmups             = MCMC_Params.get('warmups')
    Y_range             = MCMC_Params.get('Y_range')
    nu_range            = MCMC_Params.get('nu_range')
    edge_range          = MCMC_Params.get('edge_range')
    phi_range           = MCMC_Params.get('phi_range')
    PL_exp_val          = MCMC_Params.get('PL_exp_val')
    LIN_slope_val       = MCMC_Params.get('LIN_slope_val')

    # unpack varaibles from flags 
    TF_uncoated                     = flags.get("TF_uncoated") 
    TF_coated                       = flags.get("TF_coated"  ) 
    coated_duplicate_mode_IDS       = flags.get("coated_duplicate_mode_IDS") 
    blank_duplicate_mode_IDS        = flags.get("blank_duplicate_mode_IDS") 
    bad_mode_IDS                    = flags.get("bad_mode_IDS" ) 
    YLIMT                           = flags.get("YLIMT") 
    coatedreplace                   = flags.get("coatedreplace")
    blankreplace                    = flags.get("blankreplace") 
    
    #unpack variables from textfiles 
    Q_SUB_txt   = textfiles.get('Q_SUB_txt')
    ERR_SUB_txt = textfiles.get('ERR_SUB_txt')
    Q_ED_txt    = textfiles.get('Q_ED_txt')
    ERR_ED_txt  = textfiles.get('ERR_ED_txt')

  ################################################################################
       

    #You can mostly ignore these 2 a green coloured plot will also be output displaying one of the phi bulk or phi shear fit of the best fitting model, these are the y_axis bounds, ignor this graph and focus on the other six 
    low = 1e-6
    up = 4e-4


    ##Setup secondary variables
    serials             = [s]
    temperatures        = [temperature]                                     #need list type for later
    durations           = [duration]                                        #need list type for later    
    thickness           = dict(zip(serials, [thickness]))
    material            = dict(zip(serials, material))
    density             = dict(zip(serials, density))
    Y_substrate         = dict(zip(serials, Y_substrate))
    nu_substrate        = dict(zip(serials, nu_substrate))
    density_substrate   = dict(zip(serials, density_substrate))
    thickness_substrate = dict(zip(serials, thickness_substrate))


    ##Process deltaF and Q from measurements
    uncoated_ave, uncoated_std  = average_results(serials, base_dir, date_uncoated, min_num_meas=1, bayesian=TF_uncoated)
    idx = numpy.ones(uncoated_ave[s].shape[0], dtype=bool)
    #idx[[12]] = False
    uncoated_ave[s] = uncoated_ave[s][idx,:]
    uncoated_std[s] = uncoated_std[s][idx,:]

    coated_ave = {}
    coated_std = {}
    # temperature [deg C], duration [h]
    coated_ave[temperature,duration], coated_std[temperature,duration]  = average_results(serials, base_dir, date_coated, min_num_meas=1, bayesian=TF_coated)

    
    #defined here instead to make cutting parts work in this location in code
    T = temperature
    d = duration
    
    
    
    if coatedreplace == ['yes']:
        Qed = np.loadtxt(Q_ED_txt, delimiter='\t')
        stded = np.loadtxt(ERR_ED_txt, delimiter='\t')
        
        coated_ave[T,d][s]=Qed
        coated_std[T,d][s] =stded
        
        
        
    if blankreplace == ['yes']:       
        Qsub = np.loadtxt(Q_SUB_txt, delimiter='\t')
        stdsub = np.loadtxt(ERR_SUB_txt, delimiter='\t')
        
        uncoated_ave[s] =Qsub
        uncoated_std[s] =stdsub
    
    
    
    
    #functions to filter out duplicate modes measured on one suspension blank or coated
    coated_av_dup, coated_std_dup = coated_duplicate_modes_remover_gmcghee(coated_ave[T,d][s], coated_std[T,d][s], coated_duplicate_mode_IDS)
    coated_ave[T,d][s] = coated_av_dup
    coated_std[T,d][s] = coated_std_dup
    
    if debugging: 
        print(f" uncoated_ave[s]{uncoated_ave[s]}")
        print(f"  uncoated_std[s]{ uncoated_std[s]}")
        print(f" blank_duplicate_mode_IDS{blank_duplicate_mode_IDS}")
        print(f" coated_duplicate_mode_IDS{coated_duplicate_mode_IDS}")
    
   
    blank_av_dup, blank_std_dup = blank_duplicate_modes_remover_gmcghee(uncoated_ave[s], uncoated_std[s], blank_duplicate_mode_IDS)
    uncoated_ave[s] = blank_av_dup
    uncoated_std[s] = blank_std_dup







    #function to match uncoated and coated mode data AND crop bad measurements
    f_exp_uncoated, f_exp_coated, m_exp, idx_uncoated, idx_coated, uncoated_ave_final, coated_ave_final, uncoated_std_final, coated_std_final = match_modes_gmcghee(uncoated_ave[s], coated_ave[T,d][s], uncoated_std[s], coated_std[T,d][s], bad_mode_IDS)
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




    for s in serials:
        for T in temperatures:
            for d in durations:
                sys.stdout.write('%s %3d C %3d h' % (s,T, d))
                Y[T,d,s], nu[T,d,s], e[T,d,s], dfreqs_model[T,d,s], \
                       f_exp_uncoated[T,d,s], f_exp_coated[T,d,s], modes[T,d,s] = fit_coated_disk(uncoated_ave[s], coated_ave[T,d][s], 1e9*thickness[s], \
                                                                                                           density[s], verbose=False, p0=[50, 0.28], edge_bounds=[0, 1.0],\
                                                                                                           Y_bounds=[20, 300], nu_bounds=[0, 0.5], return_modes=True, \
                                                                                                           th_sub=thickness_substrate[s], nu_sub=nu_substrate[s])
                sys.stdout.write('\tY = %5.1f GPa    nu = %5.3f    edge = %5.3f mm\n' % (Y[T,d,s], nu[T,d,s], e[T,d,s]));





#fshiftplot commented out for now
    get_ipython().run_line_magic('matplotlib', 'inline')
    matplotlib.rcParams.update({'font.size': 10})
#    #fig, ax = pylab.subplots(2, 1, figsize=(8,6), sharex=True, sharey=True)
#    for i,s in enumerate(serials):
#         plot(f_exp_uncoated[T,d,s], (f_exp_coated[T,d,s] - f_exp_uncoated[T,d,s])*7.5e-3, 'o', markerfacecolor='none', markeredgewidth=3, markersize=7, label='Experimental')
#         plot(f_exp_uncoated[T,d,s], dfreqs_model[T,d,s]*7.5e-3, 'x', markerfacecolor='none', markeredgewidth=3, markersize=7, label='Best fit', zorder=10)
#         grid()
#         legend(fontsize=10, loc='upper left')
#         title(s + ' %d$^\circ$C %d h' % (T,d), fontsize=10)
#         ylabel('Frequency shift [Hz]')
#         xlabel('Mode frequency [Hz]')
#         #label_outer()
#     #fig.tight_layout()





    
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

    elasticProps= { 
    'Y  ' :  Y,    
    'nu ' :  nu,   
    'e  ' :  e,    
    'Yb ' :  [Yb[T,d,s][1] ,0.5*(Yb[T,d,s][2] -Yb[T,d,s][0])],
    'nub' :  [nub[T,d,s][1],0.5*(nub[T,d,s][2]-nub[T,d,s][0])],  
    'eb ' :  [eb[T,d,s][1] ,0.5*(eb[T,d,s][2] -eb[T,d,s][0])]
    }        
    
    import json
    # Function to convert tuple keys in nested dictionaries to string keys
    def convert_tuple_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_key = str(k) if isinstance(k, tuple) else k
                new_dict[new_key] = convert_tuple_keys(v) if isinstance(v, dict) else v
            return new_dict
        else:
            return obj

    # Convert tuple keys to string keys in the results_dict
    elasticProps_serializable = convert_tuple_keys(elasticProps)

    # Serialize the dictionary to JSON
    json_str = json.dumps(elasticProps_serializable, indent=4)
    
    if debugging:
        print('Info Written to Materials JSON')
        print()
        print(json_str)

    # If you want to save this JSON string to a file
    
    file_path = os.path.join(out_dir, str(s), f'MaterialProperties_{s}.json')

    # Use the constructed file_path in the open function
    with open(file_path, 'w') as file:
        file.write(json_str)


    # compute the coating loss angle using measurements and dilution from COMSOL simulations

    frequency_ = {}
    loss_angle_ = {}
    loss_angle_err_ = {}

    for s in serials:
        for T in temperatures:
            for d in durations:
                frequency_[T,d,s], loss_angle_[T,d,s], loss_angle_err_[T,d,s] = compute_coating_lossangle(\
               coated_ave[T,d][s],     # averaged coated Q values
               uncoated_ave[s],   # averaged uncoated Q values
               coated_std[T,d][s],     # error on coated Q values
               uncoated_std[s],   # error on uncoated Q values
               Y[T,d,s], nu[T,d,s], 1e9*thickness[s], density[s], e[T,d,s], 
               th_sub=thickness_substrate[s], nu_sub=nu_substrate[s]) # coating properties




    #PLOT_GRAPH AND SAVE EXCEL OUTPUT
    #matplotlib.rcParams.update({'font.size': 10})
    k = list(frequency_.keys())
    k.sort()
    m_labels = [] 
    for i,m in enumerate(k):

        if m[0] == 30:
            label = ' as depos.'
        else:
            label = ' ann. %d$^\circ$C (%d h)' % (m[0], m[1])
        m_labels.append((m, label)) 

        phi     = numpy.array([loss_angle_[m][i,j] for i,j in enumerate(numpy.argmin(loss_angle_[m], axis=1))])
        phi_err = numpy.array([loss_angle_err_[m][i,j] for i,j in enumerate(numpy.argmin(loss_angle_[m], axis=1))])

        freqm =frequency_[m]
        loss_array = np.array([freqm, phi, phi_err]).T
        samp_name = k[i][2]
        samp_temp = str(k[i][0])+'C_'
        samp_hrs = str(k[i][1]) + 'hr_'
        suspension = 'S' +str(suspension_number)
        df = pd.DataFrame (loss_array)
        
        filepath = samp_name + '_coatingloss_' + samp_temp + samp_hrs + suspension + '.xlsx'
        full_path = os.path.join(out_dir,s, filepath)
        df.to_excel(os.path.join(out_dir,os.sep,full_path), index=False)


        #plt.figure()
        #errorbar(1e-3*frequency_[m], 1e3*phi, yerr=1e3*phi_err, color ='green', markerfacecolor='none', \
                     #marker='s', linewidth=0, elinewidth=2, capsize=5, label=m[-1] + label)



        #legend(fontsize=9, ncol=2, loc='lower left')
        #ylim([0.0005, 1])
        #xlim([0, 30])
        #ylabel('Loss angle [$\\times 10^{-3}$]')
       # xlabel('Frequency [kHz]')
        #outer()
        #ax[i].set_xscale('log')
        #yscale('log')
        #print(k[i][2])
        #print(freqm)
        #print(phi)
        #print(phi_err)
        
        #grid(axis='y')
        #ylim(YLIMT)
        #plt.show()
        #matplotlib.pyplot.gca().yaxis.grid(True) #input to try to make only y grid but ended up making 2nd figure
        
        
    #BAYSIAN FITS    
    #from pycrime.data_analysis.bayesian_loss_angle_75mm_1mm import *        
    
    quantiles     = {s:{} for s in serials}
    samples       = {s:{} for s in serials}
    model_loglike = {s:{} for s in serials}
    phi_bulk      = {s:{} for s in serials}
    phi_shear     = {s:{} for s in serials}
    labels        = {s:{} for s in serials}
    phi_bulk_samples  = {s:{} for s in serials}
    phi_shear_samples = {s:{} for s in serials}


    loss_fr = np.logspace(2,4,100)




    #make plot directories 
    corner_plots_dir = os.path.join(out_dir, s, "CornerPlots")
    os.makedirs(corner_plots_dir, exist_ok=True)
    

    powerlaw_exp = [2,2]
    linear_slope = [2,2]
    powerlaw_exp = dict(zip(serials, powerlaw_exp))
    linear_slope = dict(zip(serials, linear_slope))
    for s in serials:
        for loss_model in model_slope: #in ['powerlaw', 'linear', 'constant']:
            for loss_angles in model_split: #in ['bulk_shear', 'equal']:
                print()
                quantiles[s][loss_model, loss_angles], \
                samples[s][loss_model, loss_angles], \
                labels[s][loss_model, loss_angles], \
                model_loglike[s][loss_model, loss_angles], \
                phi_bulk[s][loss_model, loss_angles], \
                phi_shear[s][loss_model, loss_angles], \
                phi_bulk_samples[s][loss_model, loss_angles], \
                phi_shear_samples[s][loss_model, loss_angles] = bayesian_loss_angle(uncoated_ave[s], uncoated_std[s], 
                                                                                coated_ave[T,d][s], coated_std[T,d][s], 
                                                        1e9*thickness[s], density[s], 
                                                        loss_model=loss_model, loss_angles=loss_angles,
                                                        f0=10e3, fmax=30e3, df=0.1,
                                                        fit_bounds={
                                                                        'Y': Y_range,
                                                                        'nu': nu_range, 
                                                                        'edge': edge_range,
                                                                        'phi': phi_range,
                                                                        'powerlaw_exp': PL_exp_val,
                                                                        'linear_slope': LIN_slope_val,
                                                                    },
                         #'powerlaw_exp': powerlaw_exp[s], 'linear_slope': linear_slope[s],
                                                        Y_sub=Y_substrate[s], nu_sub=nu_substrate[s], 
                                                        rho_sub=density_substrate[s], th_sub=thickness_substrate[s],
                                                        n_walkers=walkers, n_iter=iterations, n_warmup=warmups, progress=True,
                                                        loss_fr=loss_fr)
                

                corner_filename = os.path.join(corner_plots_dir, s+ '_' +loss_model +'_' +loss_angles + '_corner.pdf')
                phi_filename    = os.path.join(corner_plots_dir, s+ '_' +loss_model + '_'+loss_angles + '_phi.pdf')
                
                plot_corner(samples[s][loss_model, loss_angles], labels[s][loss_model, loss_angles],s,corner_filename)
                plot_loss_angles(loss_fr, phi_bulk[s][loss_model, loss_angles], phi_shear[s][loss_model, loss_angles],s, phi_filename, logx=True, logy=True)
        
        
    for s in serials:
        print()
        print('------ ' + s + ' ' + ('%-9s' % material[s].replace('$','')) + ' ------')
        print('--------------------------------')
        print('Model      Angles      Log.Prob.')
        print('--------------------------------')

        for k in model_loglike[s].keys():
            print('%-10s %-10s %8.1f' % (k[0], k[1], model_loglike[s][k] - np.array(list(model_loglike[s].values())).max()))
    
        print('--------------------------------')    
        
    
    
    best_slope, best_split =max(model_loglike[s].items(), key=operator.itemgetter(1))[0]
    print('The best fitting model to this data is a' + ' '+ best_slope, best_split + ' model.')
    
    
    bulk_data = phi_bulk[s][best_slope, best_split][1,:]
    # shear_data = phi_shear[s][best_slope, best_split][1,:]  # Uncomment if using shear data

    # Combine bulk and shear data for min/max calculations
    combined_data = np.concatenate([bulk_data])  # , shear_data])  # Add shear_data back if needed

    # Calculate the min and max, then apply a 20% buffer
    data_min = np.min(combined_data)
    data_max = np.max(combined_data)
    buffer = 0.20  # 20%

    lower_limit = data_min - (data_min * buffer)
    upper_limit = data_max + (data_max * buffer)

    # Now apply these limits to your plot's y-axis
    fig, ax = plt.subplots()
    ax.loglog(loss_fr, bulk_data, 'tab:green', linewidth=3, label='Bulk & Shear')
    ax.fill_between(loss_fr, y1=phi_bulk[s][best_slope, best_split][2,:], y2=phi_bulk[s][best_slope, best_split][0,:], color='tab:green', alpha=0.1)
    # ax.loglog(loss_fr, phi_shear[s][best_slope, best_split][1,:], 'tab:orange', linewidth=3, label='Shear')
    # ax.fill_between(loss_fr, y1=phi_shear[s][best_slope, best_split][2,:], y2=phi_shear[s][best_slope, best_split][0,:], color='tab:orange', alpha=0.1)
    ax.set_title(s + ' ' + material[s])
    ax.set_xlabel('Frequency (Hz)')  # Example, adjust as needed
    ax.set_ylabel('Loss')  # Example, adjust as needed
    ax.set_ylim(bottom=lower_limit, top=upper_limit)

    plot_path =os.path.join(corner_plots_dir, s+ '_' +loss_model +'_' +loss_angles + '_Best.pdf')
    plt.savefig(plot_path)
        
#     A=samples[s][(best_slope, best_split)]
#     Y_p = A[:,0];
#     nu_p = A[:,1];
#     e_p = A[:,2];
#     aB_p = A[:,3];
#     bB_p = A[:,4];
#     aS_p = A[:,5];
#     bS_p = A[:,6];
        

    f_stack=loss_fr;
    
    result = {}
    phi_values = {}
    
    result["best_slope" ]   = best_slope
    result["best_split" ]   = best_split
    result["model_loglike"] = model_loglike
    result['labels']        = labels

    if debugging:
        print(f"best_slope {best_slope}")
        print(f"best_split  {best_split}")
        print(f"model_loglike{model_loglike}")
        print(f"labels       {labels}")
        print(f" samples[s][(best_slope, best_split)]{samples[s][(best_slope, best_split)]}")
        print(f"shape: {np.shape(samples[s][(best_slope, best_split)])}")
        
    
    
    if best_slope == 'powerlaw' and best_split == 'bulk_shear':            
        #POWER LAW NON EQUAL
        A    =samples[s][(best_slope, best_split)]
        Y_p  = A[:,0];
        nu_p = A[:,1];
        e_p  = A[:,2];
        aB_p = A[:,3];
        bB_p = A[:,4];
        aS_p = A[:,5];
        bS_p = A[:,6];

        #calculate phi_bulk and phi_shear
        phi_bulk_p = np.zeros((len(A),len(f_stack)))
        phi_shear_p = np.zeros((len(A),len(f_stack)))
        #f=1114.9
        for F in range(len(f_stack)):
            f=f_stack[F]
            for L in range(len(A)):
                phi_bulk_p[L,F] = (1e-5)*aB_p[L]*(f/10000)**bB_p[L]
            B=A
            for M in range(len(B)):
                phi_shear_p[M,F] = (1e-5)*aS_p[M]*((f/10000)**bS_p[M])
        
        result["phi_bulk_p"]  = phi_bulk_p
        result["phi_shear_p"] = phi_shear_p
        phi_values["bulk"]    = phi_bulk_p.tolist()  # Convert numpy arrays to lists for JSON serialization
        phi_values["shear"]   = phi_shear_p.tolist()

            
            
            
            
    elif best_slope == 'linear' and best_split == 'bulk_shear':
        #LINEAR NON EQUAL model
        A    = samples[s][(best_slope, best_split)]
        Y_p  = A[:,0];
        nu_p = A[:,1];
        e_p  = A[:,2];
        aB_p = A[:,3];
        bB_p = A[:,4];
        aS_p = A[:,5];
        bS_p = A[:,6];
        #calculate phi_bulk and phi_shear
        phi_bulk_p = np.zeros((len(A),len(f_stack)))
        phi_shear_p = np.zeros((len(A),len(f_stack)))

        for F in range(len(f_stack)):
            f=f_stack[F]


            for L in range(len(A)):
                phi_bulk_p[L,F] = (1e-5)*aB_p[L]*(1+bB_p[L]*((f-10000)/10000))
            B=A
            for M in range(len(B)):
                phi_shear_p[M,F] = (1e-5)*aS_p[M]*(1+bS_p[M]*((f-10000)/10000))
        
        result["phi_bulk_p"]  = phi_bulk_p
        result["phi_shear_p"] = phi_shear_p
        phi_values["bulk"]    = phi_bulk_p.tolist()  
        phi_values["shear"]   = phi_shear_p.tolist()
            
            
            
            
    elif best_slope == 'constant' and best_split == 'bulk_shear':
        #constant NON EQUAL model
        A=samples[s][(best_slope, best_split)]
        Y_p    = A[:,0];
        nu_p   = A[:,1];
        e_p    = A[:,2];
        phiB_p = A[:,3];
        phiS_p = A[:,4];
        #print(mean(phiB_p))

        #calculate phi_bulk and phi_shear
        phi_bulk_p = np.zeros((len(A),len(f_stack)))
        phi_shear_p = np.zeros((len(A),len(f_stack)))


        for F in range(len(f_stack)):
            f=f_stack[F]


            for L in range(len(A)):
                phi_bulk_p[L,F] = (1e-5)*phiB_p[L]
            B=A
            for M in range(len(B)):
                phi_shear_p[M,F] = (1e-5)*phiS_p[M]
            
        result["phi_bulk_p"]  = phi_bulk_p
        result["phi_shear_p"] = phi_shear_p
        phi_values["bulk"]    = phi_bulk_p.tolist()  
        phi_values["shear"]   = phi_shear_p.tolist()   
            
            
    elif best_slope == 'powerlaw' and best_split == 'equal':            
        #POWER LAW AND EQUAL
        A    =samples[s][(best_slope, best_split)]
        Y_p  = A[:,0];
        nu_p = A[:,1];
        e_p  = A[:,2];
        a_p  = A[:,3];
        b_p  = A[:,4];

        #calculate phi
        phi_equal_p = np.zeros((len(A),len(f_stack)))
        #f=1114.9
        for F in range(len(f_stack)):
            f=f_stack[F]
            for L in range(len(A)):
                phi_equal_p[L,F] = (1e-5)*a_p[L]*(f/10000)**b_p[L]
                
        result["phi_equal_p"]     = phi_equal_p
        phi_values["phi_equal_p"] = phi_equal_p.tolist()
            
            
            
            

    elif best_slope == 'linear' and best_split == 'equal':            
        #LINEAR AND EQUAL
        A    = samples[s][(best_slope, best_split)]
        Y_p  = A[:,0];
        nu_p = A[:,1];
        e_p  = A[:,2];
        a_p  = A[:,3];
        b_p  = A[:,4];

        #calculate phi
        phi_equal_p = np.zeros((len(A),len(f_stack)))

        for F in range(len(f_stack)):
            f=f_stack[F]


            for L in range(len(A)):
                phi_equal_p[L,F] = (1e-5)*a_p[L]*(1+b_p[L]*((f-10000)/10000))
            
        result["phi_equal_p"]     = phi_equal_p
        phi_values["phi_equal_p"] = phi_equal_p.tolist()
            
            

    elif best_slope == 'constant' and best_split == 'equal':
        #constant AND EQUAL model
        A     = samples[s][(best_slope, best_split)]
        Y_p   = A[:,0];
        nu_p  = A[:,1];
        e_p   = A[:,2];
        phi_p = A[:,3];

        #calculate phi_bulk and phi_shear
        phi_equal_p = np.zeros((len(A),len(f_stack)))

        for F in range(len(f_stack)):
            f=f_stack[F]


            for L in range(len(A)):
                phi_equal_p[L,F] = (1e-5)*phi_p[L]
        
        
        result["phi_equal_p"]     = phi_equal_p
        phi_values["phi_equal_p"] = phi_equal_p.tolist()
            
            
     # Writing results to a JSON file
    json_filename = os.path.join(out_dir, f"{s}", f"{s}_{T}C_{d}hrs_BulkShear_Summary.json")
     
    with open(json_filename, 'w') as json_file:
        json.dump(phi_values, json_file, indent=4)
    
    #CALCULATE G_A, G_B, G_C, G_D for an unequal bulk and shear model
    
    # Want to define <> terms of the u eqn we later will want to thickness average 
    # u=0.25*[<G_A>*SDs^2 + <G_B>*(Sdels^2 + S6s^2)]
    # G_A = Y/(1-nu) 
    # G_B = Y/(1+nu)

    G_A = np.zeros((len(A)))
    G_B = np.zeros((len(A)))

    for i in range(len(A)):
        G_A[i] = Y_p[i]/(1-nu_p[i])
        G_B[i] = Y_p[i]/(1+nu_p[i])   

    
    #now G_C and G_D which depend on phi
    if best_split == 'bulk_shear':         
        # Want to define <> terms of the Pdis eqn we later will want to thickness average 
        # Pdis=(omega/2)*[<G_C>*SDs^2 + <G_D>*(Sdels^2 + S6s^2)]
        # G_C = [Y/[6*(1-nu)^2]]*[2*(1-2*nu)*phiB + (1+nu)*phiS]; 
        # G_D = Y/(2*(1+nu))*phiS

        #G_C broken into 3 parts for calculation
        c1 = np.zeros((len(A)))
        c2 = np.zeros((len(A),len(f_stack)))
        c3 = np.zeros((len(A),len(f_stack)))


        G_C = np.zeros((len(A),len(f_stack)))
        G_D = np.zeros((len(A),len(f_stack)))

        for F in range(len(f_stack)):
            for i in range(len(A)):
                c1[i] = ((1/6)*Y_p[i])/((1-nu_p[i])**2)
                c2[i,F] = 2*(1-2*nu_p[i])*phi_bulk_p[i,F]
                c3[i,F] = (1+nu_p[i])*phi_shear_p[i,F]

                G_D[i,F] = phi_shear_p[i,F]*0.5*Y_p[i]/(1+nu_p[i])

                G_C[i,F] = c1[i]*(c2[i,F]+c3[i,F])
    
    
    
    
    elif best_split == 'equal':
        #if bulk and shear loss are equal G_C eqn becomes much simpler
        # Want to define <> terms of the Pdis eqn we later will want to thickness average 
        # Pdis=(omega/2)*[<G_C>*SDs^2 + <G_D>*(Sdels^2 + S6s^2)]
        # G_C = Y/(2*(1-nu))*phi_equal;  
        # G_D = Y/(2*(1+nu))*phi_equal


        G_C = np.zeros((len(A),len(f_stack)))
        G_D = np.zeros((len(A),len(f_stack)))

        for F in range(len(f_stack)):
            for i in range(len(A)):

                G_D[i,F] = phi_equal_p[i,F]*0.5*Y_p[i]/(1+nu_p[i])

                G_C[i,F] = phi_equal_p[i,F]*0.5*Y_p[i]/(1-nu_p[i])
        
        print('for G_C and G_D terms note bulk and shear are equal')        
        
    return G_A, G_B, G_C, G_D, A, fig, best_slope, best_split























