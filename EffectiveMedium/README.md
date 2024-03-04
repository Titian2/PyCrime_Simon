Effective Medium Analysis 

These analysis scripts build on the previous work by G. Vajente and G.McGhee to analyse the mechanical loss data  

Overview
This project includes a set of Python scripts designed to perform data analysis and processing tasks, specifically focusing on handling nested dictionaries, serializing and deserializing data to and from JSON, and generating plots with Matplotlib. It is tailored for scenarios involving measurements categorized by serial numbers, temperatures, and durations, providing tools for efficient data management and visualization.pytho

EffectiveMedium.py 
functions: 

save_fig': None,
 'save_fig_pdf': None,

 'pcolormesh_logscale': None,

 'summarise': 'Generate a summary of the data provided, including information about the data structure, type, and shape.\n@param data - the input data to be summarized\n@param all - a boolean flag to indicate whether detailed information for all elements should be included\n@return a list of summaries containing key, type, and size of the data, and print the total number of items in the data.\nAuthor S.Tait 2024 ',

 'calculate_density': 'Calculate the  average density of a material based on its refractive index, number of layers, layer density, and wavelength.\n@param refractive_indx - The refractive index of the material\n@param num_layers - The number of layers in the material\n@param layer_density - The density of each layer\n@param wavelength - The wavelength of the light\n@return density - The calculated density\n@return denominator - The denominator value used in the calculation\n@return thicknesses_by_layer - The thicknesses of each layer multiplied by the number of layers\nAuthor S.Tait 2024 ',

 'fig_to_base64': 'Convert a matplotlib figure to a base64 encoded image.\n@param fig - the matplotlib figure to convert\n@return a base64 encoded image of the figure',

 'calculate_group_stats_and_plot': None,

 'merge_on_frequency_and_plot': 'Merge multiple dataframes based on frequency and plot the data with error bars.\n@param *dataframes - variable number of dataframes to merge\n@param debugging - boolean flag to print the combined DataFrame before frequency matching (default is False)\n@param tolerance - tolerance level for frequency matching (default is 1)\n@return None\nAuthor S.Tait 2024 ',

 'best_results': 'This function processes the best results for a given set of serials, base path, output directory, measurement dates, and a label. It also has optional parameters for verbosity and plotting.\n@param serials - The serial numbers of the devices.\n@param base_path - The base path where the data is stored.\n@param out_dir - The output directory where results will be saved.\n@param measurement_dates - The dates of the measurements.\n@param label - The label for the data.\n@param verbose - Whether to display verbose output (default is False).\n@param plots - Whether to display plots (default is False).\nAuthor S.Tait 2024 ',
 
 'save_dataframe_slice_to_csv': 'Saves specified columns of a DataFrame to a CSV file\ndf              : DataFrame to slice and save.\ncolumns         : List of column indices to include in the slice.\nfilename        : Filename for the output CSV.\ndirectory       : Directory where the CSV file will be saved.\nAuthor S.Tait 2024 ',
 
 'check_lengths': 'Check if the lengths of the first three arrays are the same.\n@param f_stack (numpy.ndarray): First array.\n@param comsolmod (numpy.ndarray): Second array.\n@param dbulk (numpy.ndarray): Third array.\n@param dshear (numpy.ndarray): Fourth array.\n@return bool: True if the lengths are the same, False otherwise.\nAuthor S.Tait 2024 ',
 
 'check_ordering': 'Check if the last two arrays have the same ordering.\n\nParameters               : \ncomsolmod (numpy.ndarray): COMSOL outputs\nm_exp (numpy.ndarray)    : Measured Frequencies\n\nReturns:\n    bool: True if the ordering is the same, False otherwise.\nAuthor S.Tait 2024 ',
 
 'ismember': 'Check if elements in array a are present in array b and return their indices.\n@param a - The first array to compare.\n@param b - The second array to compare.\n@return A list of indices where elements in array a are found in array b.\nAuthor S.Tait 2024 ',
 
 'extract_info_for_serial': 'Extract information related to a specific serial number from a file.\n@param serial_number - The serial number to extract information for.
 \n@param file_path - The path to the file containing the information.\n@return A dictionary containing the serial number and corresponding measurements.\nAuthor S.Tait 2024 ',
 
 'debug_var': 'Debug a variable by printing its name, type, and length (if applicable).\n@param var - the variable to debug\n@param var_name - the name of the variable (default is "unknown")\n@return None\nAuthor S.Tait 2024 ',
 
 'preprocess_and_match': 'Preprocess and match the input data for uncoated and coated samples.\n@param uncoated_ave - average values for uncoated samples\n@param coated_ave - average values for coated samples\n@param uncoated_std - standard deviation for uncoated samples\n@param coated_std - standard deviation for coated samples\n@param bad_mode_IDS - list of bad mode IDs\n@param debugging - flag for debugging mode\n@return None\nAuthor S.Tait 2024 ',
 
 'prepareMeasuredData': 'Prepare measured data for analysis and visualization.\n@param idx - index of the data\n@param table_data - data in tabular format\n@param base_dir - base directory for data\n@param out_dir - output directory for results\n@param temperature - temperature of the measurement\n@param duration - duration of the measurement\n@param material_dict - dictionary of material properties\n@param suspension_info - information about the suspension\n@param Booleans - boolean values for analysis\n@param textfiles - text files for additional information\n@param debugging - flag for debugging mode (default False)\n@param plots - flag for generating plots (default False)\nAuthor S.Tait 2024 ',
 
 'convert_tuple_keys': 'Converts tuple keys to strings and tuple values to lists in a dictionary for JSON serialization.\n@param d - A dictionary possibly containing tuple keys and values.\n@return A JSON-serializable dictionary.',
 
 'read_measurements_for_serial': 'Read measurements from a CSV file for a specific serial number.\n@param csv_file_path - the path to the CSV file\n@param serial_number - the serial number to retrieve measurements for\n@return A dictionary of measurements for the specified serial number, or None if not found.\nAuthor S.Tait 2024 ',
 
 'read_measurements': 'Read measurements from a CSV file, returning data based on specified flags.\n@param csv_file_path: The path to the CSV file.\n@param flags: Keyword arguments controlling the output.\n@return: Depending on flags, either a list of serials or a dictionary keyed by serials with specified measurement dates.\nAuthor S.Tait 2024 ',


 'plot_material_properties': 'Plot the material properties for a list of serials and display the plots if specified.\n@param serials - list of serial numbers\n@param out_dir - directory where the output will be saved\n@param plots - boolean flag to display the plots\n@return None',
 
 'create_dataframe_from_json': None,
 
 'normalize_for_json': None,
 
 'save_to_json': None,
 
 'load_from_json': None,



 Function: match_modes_gmcghee, Docstring: An edited version of a previous code by g vajente.  
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
Function: coated_duplicate_modes_remover_gmcghee, Docstring: Remove duplicate modes from coated average and standard deviation arrays.
@param coated_ave - Coated average array
@param coated_std - Coated standard deviation array
@param coated_duplicate_mode_IDS - Array of duplicate mode IDs
@param debugging - Flag to enable debugging mode

@return  blank_av_dup, blank_std_dup
Function: blank_duplicate_modes_remover_gmcghee, Docstring: None
Function: find_matches_and_return_smaller, Docstring: None


Function: save_fig, Docstring: None
Function: save_fig_pdf, Docstring: None
Function: pcolormesh_logscale, Docstring: None
Function: stack_values_extractor_replacingvalues, Docstring: Extracts values from provided dictionaries and processes them according to
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
Function: compute_coating_lossangle_gmcghee, Docstring: SAME AS PREVIOUS FUNCTION MADE BY G VAJENTE BUT NOW USES mode_match_gmcghee function and has additional input bad_mode_IDS

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
