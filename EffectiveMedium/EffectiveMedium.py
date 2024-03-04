

import base64
import os
import pickle
import random
import sys
import time
from io import BytesIO
import warnings

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from matplotlib.colors import LogNorm
from scipy.signal import *
from tabulate import tabulate
from tqdm.notebook import tqdm
import numpy_groupies as npg
from sklearn.cluster import DBSCAN
import json
from tabulate import tabulate
import numpy as np
import csv 

sys.path.append('/mnt/data/CRIME/results/CSU_TiGemania/EffectiveMedium')
from   stack_values_extractor_replacingvalues import stack_values_extractor_replacingvalues
from   singlelayer_values_extractor_replacingvalues import singlelayer_values_extractor_replacingvalues
from   match_modes_gmcghee  import match_modes_gmcghee


# Configuration and settings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10, 6]  # or any other default size you want
plt.rcParams.update({'font.size': 10})

# If you are using a Jupyter Notebook, uncomment the following line:
# %matplotlib inline



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
    
def summarise(data, all=False):
    summaries = []

    # Function to generate summary for the current data
    def generate_summary(data, parent_key='', level=0):
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                summaries.append((level, new_key, type(value).__name__, np.shape(value) if isinstance(value, float) else (len(value) if hasattr(value, '__len__') else 'N/A')))
                if all and isinstance(value, (dict, list, np.ndarray)):
                    generate_summary(value, new_key, level + 1)
        elif isinstance(data, (list, np.ndarray)):
            for index, value in enumerate(data):
                new_key = f"{parent_key}[{index}]" if parent_key else str(index)
                summaries.append((level, new_key, type(value).__name__, np.shape(value) if isinstance(value, float) else (len(value) if hasattr(value, '__len__') else 'N/A')))
                if all and isinstance(value, (dict, list, np.ndarray)):
                    generate_summary(value, new_key, level + 1)
    
    generate_summary(data)

    # Sorting summaries by level to ensure high-level summaries are printed first
    summaries.sort(key=lambda x: x[0])

    # Removing the level information now that sorting is done
    formatted_summaries = [(key, type_, size) for _, key, type_, size in summaries]

    # Printing the high-level summary followed by all sub-key entries
    print("Total items:", len(data))
    print(tabulate(formatted_summaries, headers=["Key", "Type", "Size/Length"], tablefmt="grid"))
    
    
def calculate_density(refractive_indx, num_layers, layer_density, wavelength):
    """
    Calculate the  average density of a material based on its refractive index, number of layers, layer density, and wavelength.
    @param refractive_indx - The refractive index of the material
    @param num_layers - The number of layers in the material
    @param layer_density - The density of each layer
    @param wavelength - The wavelength of the light
    @return density - The calculated density
    @return denominator - The denominator value used in the calculation
    @return thicknesses_by_layer - The thicknesses of each layer multiplied by the number of layers
    Author S.Tait 2024 
    """
    layer_thickness = [wavelength / (4 * r_indx) for r_indx in refractive_indx]

    densities = []
    thicknesses_by_layer = []
    for i in range(len(layer_thickness)):
        density_i = (layer_thickness[i] * num_layers[i] * layer_density[i])
        densities.append(density_i)
        thicknesses_by_layer.append(layer_thickness[i] * num_layers[i])   

    numerator = sum(densities)

    denominator = 0
    for i in range(len(layer_thickness)):
        denominator += layer_thickness[i] * num_layers[i]

    density = numerator / denominator

    return density ,denominator,thicknesses_by_layer


def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded image.
    @param fig - the matplotlib figure to convert
    @return a base64 encoded image of the figure
    """
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def calculate_group_stats_and_plot(filtered_group_data,sample_name,date_string,base_dir,verbose,confidence_level=0.95):
        # Get unique cluster labels
        filtered_group_data = pd.read_csv(filtered_group_data,sep='\t')
        
        group_idx     = filtered_group_data[filtered_group_data['FitCheck']==True]['Group']
        Frequencies   = filtered_group_data[filtered_group_data['FitCheck']==True]['Freq']
        Q1Values      = filtered_group_data[filtered_group_data['FitCheck']==True]['Q1']
        Q2Values      = filtered_group_data[filtered_group_data['FitCheck']==True]['Q2']
        mode_m_values = filtered_group_data[filtered_group_data['FitCheck']==True]['mode(m)']
        mode_n_values = filtered_group_data[filtered_group_data['FitCheck']==True]['mode(n)']
        
        freq    = npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Frequencies), func='mean')
        Q1means =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='mean')
        Q2means =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='mean')
        Q1max   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='max')
        Q2max   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='max')
        Q1min   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='min')
        Q2min   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='min')
        Q1std   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='std')
        Q2std   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='std')
        m       =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(mode_m_values), func='first')
        n       =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(mode_n_values), func='first')

        results_df = pd.DataFrame({
            'Frequency': freq,
            'Q1 Mean': Q1means,
            'Q2 Mean': Q2means,
            'Q1 Max': Q1max,
            'Q2 Max': Q2max,
            'Q1 Min': Q1min,
            'Q2 Min': Q2min,
            'Q1 Std': Q1std,
            'Q2 Std': Q2std,
            'm': m,
            'n': n
            })
        
        if verbose: 
            print(results_df)
        
        return results_df 
    


def summarise(data, all=False):
    """
    Generate a summary of the data provided, including information about the data structure, type, and shape.
    @param data - the input data to be summarized
    @param all - a boolean flag to indicate whether detailed information for all elements should be included
    @return a list of summaries containing key, type, and size of the data, and print the total number of items in the data.
    Author S.Tait 2024 
    """
    summaries = []

    # Function to generate summary for the current data
    def generate_summary(data, parent_key='', level=0):
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                summaries.append((level, new_key, type(value).__name__, np.shape(value) if isinstance(value, float) else (len(value) if hasattr(value, '__len__') else 'N/A')))
                if all and isinstance(value, (dict, list, np.ndarray)):
                    generate_summary(value, new_key, level + 1)
        elif isinstance(data, (list, np.ndarray)):
            for index, value in enumerate(data):
                new_key = f"{parent_key}[{index}]" if parent_key else str(index)
                summaries.append((level, new_key, type(value).__name__, np.shape(value) if isinstance(value, float) else (len(value) if hasattr(value, '__len__') else 'N/A')))
                if all and isinstance(value, (dict, list, np.ndarray)):
                    generate_summary(value, new_key, level + 1)
    
    generate_summary(data)

    # Sorting summaries by level to ensure high-level summaries are printed first
    summaries.sort(key=lambda x: x[0])

    # Removing the level information now that sorting is done
    formatted_summaries = [(key, type_, size) for _, key, type_, size in summaries]

    # Printing the high-level summary followed by all sub-key entries
    print("Total items:", len(data))
    print(tabulate(formatted_summaries, headers=["Key", "Type", "Size/Length"], tablefmt="grid"))
    
    
def merge_on_frequency_and_plot(*dataframes,debugging=False ,  tolerance=1):
    """
    Merge multiple dataframes based on frequency and plot the data with error bars.
    @param *dataframes - variable number of dataframes to merge
    @param debugging - boolean flag to print the combined DataFrame before frequency matching (default is False)
    @param tolerance - tolerance level for frequency matching (default is 1)
    @return None
    Author S.Tait 2024 
    """
    num_dataframes = len(dataframes)
    
    if num_dataframes >= 2:
        combined_df = pd.concat([df.assign(source_df=i) for i, df in enumerate(dataframes, 1)], ignore_index=True)

        if debugging: 
            # Diagnostic print to verify the merge operation
            print("Combined DataFrame before frequency matching:")
            print(combined_df)

        combined_df['matched_freq'] = combined_df.apply(
            lambda x: combined_df.loc[(abs(combined_df['Frequency'] - x['Frequency']) <= tolerance) & (combined_df['source_df'] != x['source_df']), 'Frequency'].min(), axis=1)

        if debugging:
            # Diagnostic print to check frequency matching
            print("\nFrequency matching results:")
            print(combined_df[['Frequency', 'matched_freq']].dropna())

        def calculate_error(row):
            std_error = row['Q1 Std']
            max_mean_diff = abs(row['Q1 Max'] - row['Q1 Mean'])
            return std_error + max_mean_diff

        combined_df['error'] = combined_df.apply(calculate_error, axis=1)


        best_matches = combined_df.sort_values(by=['matched_freq', 'error']).groupby('matched_freq', as_index=False).first()


        if debugging:
            # Diagnostic print to verify error calculation and selection
            print("\nRows with calculated error and their selection:")
            print(best_matches[['Frequency', 'error']])

        output_columns = ['Frequency', 'Q1 Mean', 'Q2 Mean', 'Q1 Max', 'Q2 Max', 'Q1 Min', 'Q2 Min', 'Q1 Std', 'Q2 Std', 'm', 'n']
        output_df = best_matches[output_columns]
        ## from pycrime.data_analysis average results
        #idx = where(logical_and(d[:,7] == d[i,7], d[:,8] == d[i,8]))[0]



        remove_idx = []  # To store indices of rows to be removed


        for i in range(len(output_df)):
            idx = np.where([(output_df['m'] == output_df.iloc[i]['m']) & (output_df['n'] == output_df.iloc[i]['n'])])

            if len(idx[1]) == 2:  # Proceed if more than two matching rows found
                # Calculate errors and sum of 'Q1 Mean' and 'Q2 Mean' for each matching row
                idx = idx[1]


                errors = output_df.iloc[idx].apply(calculate_error, axis=1)
                mean_sums = output_df.iloc[idx]['Q1 Mean'] + output_df.iloc[idx]['Q2 Mean']


                # Find the row with the highest error or highest mean sum
                max_error_idx = errors.idxmax()
                max_mean_sum_idx = mean_sums.idxmax()

                # Decide which index to remove (prioritizing max_error_idx)
                if errors[max_error_idx] > errors[max_mean_sum_idx] or mean_sums[max_error_idx] >= mean_sums[max_mean_sum_idx]:
                    to_remove = max_error_idx
                else:
                    to_remove = max_mean_sum_idx

                remove_idx.append(to_remove)
                if debugging: 
                    print(f"WARNING: duplicate modes : {output_df.iloc[to_remove]['Frequency']:.2f} Hz")

        # Remove duplicate warnings and indices to remove
        remove_idx = list(set(remove_idx))

        output_df = output_df.drop(remove_idx).reset_index(drop=True)
    else:
               # Convert tuple to numpy array
        array_data = np.array(dataframes)

        # Flatten the 3D array to 2D
        flattened_array = array_data.reshape(-1, array_data.shape[-1])

        # Convert flattened array to DataFrame
        output_df = pd.DataFrame(flattened_array)
        output_df.columns = ['Frequency', 'Q1 Mean', 'Q2 Mean', 'Q1 Max', 'Q2 Max', 'Q1 Min', 'Q2 Min', 'Q1 Std', 'Q2 Std', 'm', 'n']


        
    
    # Plotting function for DataFrames
    def plot_with_error_bars(ax, df, marker='o', label_prefix='', color=None):
        for i, q in enumerate(['Q1', 'Q2']):
            ax.errorbar(df['Frequency'], df[f'{q} Mean'], yerr=df[f'{q} Std'], fmt=marker, color=color, label=f'{label_prefix}{q} Mean and Std')
        return ax

    viridis = plt.get_cmap('viridis')
    num_dfs = len(dataframes) + 1  # +1 for the output DataFrame


    fig, ax = plt.subplots()
    # Plot for each input DataFrame
    for i, df in enumerate(dataframes, 1):
        color = viridis(i / num_dfs)  # Compute the color for this dataset
        ax = plot_with_error_bars(ax, df, marker='o', label_prefix=f'DF{i} ')
    
    # Plot for the output DataFrame
    output_color = viridis(num_dfs / num_dfs)  # Ensuring the output DataFrame has a unique color
    ax = plot_with_error_bars(ax, output_df, marker='*', label_prefix='Output ', color=output_color)
    
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Mechanical Loss')
    ax.legend()
    ax.grid()

    return output_df, fig


def best_results(serials,base_path,out_dir,measurement_dates,label,debugging=False,plots=False): 
    """
    This function processes the best results for a given set of serials, base path, output directory, measurement dates, and a label. It also has optional parameters for verbosity and plotting.
    @param serials - The serial numbers of the devices.
    @param base_path - The base path where the data is stored.
    @param out_dir - The output directory where results will be saved.
    @param measurement_dates - The dates of the measurements.
    @param label - The label for the data.
    @param verbose - Whether to display verbose output (default is False).
    @param plots - Whether to display plots (default is False).
    Author S.Tait 2024 
    """
    
    df_dict = {}  # Initialize an empty dictionary

    for date in measurement_dates :

        #ave, std = average_results(serials, base_path, [date], min_num_meas=1, bayesian=False)

        filename = os.path.join(base_path,serials,f"Grouped_Output_{serials}_{date}.txt")

        temp_df = calculate_group_stats_and_plot(filename, serials, [date], base_path, verbose=False, confidence_level=0.95)
        

        # Assuming the function returns a DataFrame, sort and clean it
        temp_df_sorted = temp_df.sort_values(by=['Frequency'])
        temp_df_cleaned = temp_df_sorted.loc[(temp_df_sorted != 0).any(axis=1)]

        # Store the cleaned DataFrame in the dictionary, keyed by the date
        df_dict[date] = temp_df_cleaned
    if debugging:
        summarise(df_dict)
    output_df, figure  = merge_on_frequency_and_plot(*df_dict.values(),debugging =debugging ,tolerance=1)
    
        
    
    
    filename = serials+'_Measurement_Summary' + label
    figure_path = os.path.join(out_dir,serials, filename)
    plt.savefig(figure_path,facecolor='white')

    # Displaying DataFrame and Plot side by side
    display_html = f"""
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div style='width: 30%;'> {output_df.to_html()} </div>
        <div style='width: 60%;'> <img src='data:image/png;base64,{fig_to_base64(figure)}'/> </div>
    </div>
    """

    
    best_data     = {serials: np.array(output_df.iloc[:,[0,1,2,9,10]])}
    
    best_data_std = {serials: np.array(output_df.iloc[:, [0] + list(range(2, len(output_df.columns)))])}
    if debugging: 
        summarise(best_data)
    # Display the HTML content
    if plots:
        display(HTML(display_html.strip()))    
    return output_df , best_data,best_data_std

def save_dataframe_slice_to_csv(df, columns, filename, directory):
    """
    Saves specified columns of a DataFrame to a CSV file
    df              : DataFrame to slice and save.
    columns         : List of column indices to include in the slice.
    filename        : Filename for the output CSV.
    directory       : Directory where the CSV file will be saved.
    Author S.Tait 2024 
    """
    # Construct the full path for the output file
    file_path = os.path.join(directory, filename)
    
    # Slice the DataFrame and save to CSV
    df.iloc[:, columns].to_csv(file_path, sep='\t', index=False,header=False)
    
    if isinstance(file_path, tuple):
        file_path = file_path[0]
        
    # Check if the file was created successfully
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} was not created successfully!")
    
    return file_path
    
    
def check_lengths(f_stack, comsolmod, dbulk, dshear):
    """
    Check if the lengths of the first three arrays are the same.
    @param f_stack (numpy.ndarray): First array.
    @param comsolmod (numpy.ndarray): Second array.
    @param dbulk (numpy.ndarray): Third array.
    @param dshear (numpy.ndarray): Fourth array.
    @return bool: True if the lengths are the same, False otherwise.
    Author S.Tait 2024 
    """
    """
    Check if the lengths of the first three arrays are the same.
    
    Parameters:
        f_stack (numpy.ndarray): First array.
        comsolmod (numpy.ndarray): Second array.
        dbulk (numpy.ndarray): Third array.
        dshear (numpy.ndarray): Fourth array.
    
    Returns:
        bool: True if the lengths are the same, False otherwise.
    """
    lengths = [len(arr) for arr in [f_stack, comsolmod, dbulk]]
    
    return len(set(lengths)) == 1

def check_ordering(comsolmod, m_exp):
    """
    Check if the last two arrays have the same ordering.
    
    Parameters               : 
    comsolmod (numpy.ndarray): COMSOL outputs
    m_exp (numpy.ndarray)    : Measured Frequencies
    
    Returns:
        bool: True if the ordering is the same, False otherwise.
    Author S.Tait 2024 
    """
    return np.array_equal(comsolmod, m_exp)

def ismember(a, b):
    """
    Check if elements in array a are present in array b and return their indices.
    @param a - The first array to compare.
    @param b - The second array to compare.
    @return A list of indices where elements in array a are found in array b.
    Author S.Tait 2024 
    """
    # Ensure a and b are NumPy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Initialize an empty list to store indices of matching rows
    matches = []
    
    # Iterate over each row in array a
    for row in a:
        # Check if the row exists in array b
        match = np.where(np.all(b == row, axis=1))[0]
        if len(match) > 0:
            matches.append(match[0])
        else:
            matches.append(None)
    
    return matches

# extracting informatin from lookup table 
def extract_info_for_serial(serial_number, file_path):
    """
    Extract information related to a specific serial number from a file.
    @param serial_number - The serial number to extract information for.
    @param file_path - The path to the file containing the information.
    @return A dictionary containing the serial number and corresponding measurements.
    Author S.Tait 2024 
    """
    # Define the structure of the output based on expected data
    info = {
        "Serial Number": serial_number,
        "Uncoated Measurements": [],
        "Coated Measurements": [],
        "HT1 Measurements": [],
        "HT2 Measurements": []
    }
    
    # Open and read the file
    with open(file_path, 'r') as file:
        # Skip the headers
        next(file)
        
        # Iterate over each line in the file
        for line in file:
            parts = line.strip().split(", ")
            # Check if the current line's serial number matches the one we're looking for
            if parts[0] == serial_number:
                # Assuming 2 dates for each measurement type, adjust indices accordingly
                info["Uncoated Measurements"] = parts[1:3]
                info["Coated Measurements"] = parts[3:5]
                info["HT1 Measurements"] = parts[5:7]
                info["HT2 Measurements"] = parts[7:9]
                break  # Exit the loop once the matching serial number is found
    
    return info

def debug_var(var, var_name="unknown"):
    """
    Debug a variable by printing its name, type, and length (if applicable).
    @param var - the variable to debug
    @param var_name - the name of the variable (default is "unknown")
    @return None
    Author S.Tait 2024 
    """
    print()
    print()
    print(f"Variable Name: {var_name}")
    print()
    
    try:
        # Attempt to directly convert the input to a DataFrame
        df = pd.DataFrame(var)
        print(f"Converted '{var_name}' to DataFrame:")
        print(df)
    except Exception as e:
        # If conversion fails, handle depending on type
        print(f"Could not convert '{var_name}' to DataFrame due to: {e}")
    if isinstance(var, np.ndarray):
        print("Type: NumPy Array")
        print("Shape:", var.shape)
        
    elif isinstance(var, (list, tuple)):
        print("Type:", type(var).__name__)
        print("Length:", len(var))
        
    elif isinstance(var, dict):
        print("Type: Dictionary")
        print("Keys:", var.keys())
        
    else:
        print("Type:", type(var))
        
    print()
    
    

#bad_mode_IDS,debugging):

def preprocess_and_match(uncoated_ave, coated_ave,uncoated_std, coated_std, bad_mode_IDS,debugging):
    """
    Preprocess and match the input data for uncoated and coated samples.
    @param uncoated_ave - average values for uncoated samples
    @param coated_ave - average values for coated samples
    @param uncoated_std - standard deviation for uncoated samples
    @param coated_std - standard deviation for coated samples
    @param bad_mode_IDS - list of bad mode IDs
    @param debugging - flag for debugging mode
    @return None
    Author S.Tait 2024 
    """
    
    if bad_mode_IDS ==['none']:
        idx_uncoated = ['none']
        idx_coated = ['none']
    
    if debugging:
        debug_var(uncoated_ave,"uncoated_ave")
        debug_var(coated_ave,"uncoated_var")
        debug_var(uncoated_std,"uncoated_var")
        debug_var(coated_std,"coated_std")
    
    
    input_av  = np.concatenate([uncoated_ave, coated_ave], axis=0)
    input_std  = np.concatenate([uncoated_std, coated_std], axis=0)
    
    if debugging: 
        debug_var(input_av,"input_av ")
        debug_var(input_std,"input_std ")


    RoundTo = 100
    # Preprocess by rounding the first column to the nearest 100
    rounded1 = np.round(uncoated_ave[:, 0] / RoundTo ) * RoundTo 
    rounded2 = np.round(coated_ave[:, 0] / RoundTo ) * RoundTo 
    if debugging:
        debug_var(rounded1,"rounded1")
        debug_var(rounded2,"rounded2")
    
    
    # Keep track of used indices to ensure unique matches
    used_indices_uncoated_ave = set()
    used_indices_coated_ave = set()
    
    matches = []  # Store unique matches
    scaling_factors = []  # Store scaling factors for DBSCAN
    
    for i, val1 in enumerate(rounded1):
        if i in used_indices_uncoated_ave:
            continue  # Skip if already matched
        diffs = np.abs(val1 - rounded2)
        min_diff_idx = np.argmin(diffs)
        if diffs[min_diff_idx] <= 10 and min_diff_idx not in used_indices_coated_ave:  # Check threshold and uniqueness
            # Record match and scaling factor
            matches.append((i, min_diff_idx))
            scaling_factors.append(uncoated_ave[i, 0] / coated_ave[min_diff_idx, 0])
            used_indices_uncoated_ave.add(i)
            used_indices_coated_ave.add(min_diff_idx)
            
    
    
    scaled_uncoated_ave = uncoated_ave[:,0]
    raw_coated_ave    = coated_ave[:,0]
    scaled_coated_ave = coated_ave[:,0]*scaling_factors[0]
    
    uncoated_ave_2d = scaled_uncoated_ave.reshape(-1, 1)
    raw_coated_ave_2d = raw_coated_ave.reshape(-1, 1)
    scaled_coated_ave_2d = scaled_coated_ave.reshape(-1, 1)

    uncoated_indices = np.arange(len(uncoated_ave)).reshape(-1, 1)  # Row indices for uncoated_ave
    coated_indices = np.arange(len(coated_ave)).reshape(-1, 1)  # Row indices for coated_ave

   
    indices_for_dbscan = np.vstack((uncoated_indices, coated_indices))  # Adjust as needed

    labels_uncoated_ave = np.ones((len(uncoated_ave_2d), 1))  # All ones, with shape matching uncoated_ave_2d
    labels_coated_ave = np.ones((len(scaled_coated_ave_2d), 1)) * 2  # All twos, shape matching scaled_coated_ave_2d
    labels_combined = np.vstack((labels_uncoated_ave, labels_coated_ave))
    
    raw_frequencies = np.array(np.vstack((uncoated_ave_2d, raw_coated_ave_2d)))
    
    data_for_dbscan = np.array(np.vstack((uncoated_ave_2d, scaled_coated_ave_2d)))
    
    
    data_with_indices = np.hstack((data_for_dbscan, indices_for_dbscan,labels_combined))
    
    #debug_var(data_with_indices)
    
    last_two_uncoated_ave = uncoated_ave[:, -2:]
    last_two_coated_ave = coated_ave[:, -2:]

    # Vertically stack the last two columns from uncoated_ave and coated_ave
    combined_last_two = np.vstack((last_two_uncoated_ave, last_two_coated_ave))
    
    if not matches:
        print("No unique matching values within threshold.")
        return False, None, None
    
    if debugging: 
        debug_var(data_for_dbscan,"data_for_dbscan")
    
    # Run DBSCAN on the prepared data
    db = DBSCAN(eps=5, min_samples=2).fit(data_for_dbscan)  # Adjust eps and min_samples as needed
    labels = db.labels_  # Extract cluster labels


    # Identify indices of points forming valid clusters (excluding noise points)
    valid_clusters_indices = labels != -1
    if debugging:
        debug_var(valid_clusters_indices,"valid_clusters_indices")
    # Optionally, print out the number of valid clusters found (excluding noise)
    
    num_valid_clusters = len(set(labels[valid_clusters_indices])) - (1 if -1 in labels else 0)
    
    print(f"Number of matching modes found: {num_valid_clusters}")
    
    # Identify indices of points forming valid clusters (excluding noise points)
    valid_clusters_indices = (labels != -1).astype(int)  # Convert boolean array to int for concatenation
    # Concatenate data with indices, labels, and valid cluster indicators
    if debugging:
        debug_var(data_for_dbscan ,"data_for_dbscan")
        debug_var(indices_for_dbscan ,"indices_for_dbscan")
        debug_var(labels_combined ,"labels_combined")
        debug_var(np.array(valid_clusters_indices), "valid_clusters_indices")
    
    data_with_indices = np.hstack((input_av,input_std, indices_for_dbscan, labels_combined, valid_clusters_indices.reshape(-1,1)))    
    
    filtered_data =  data_with_indices[data_with_indices[:, -1] != 0]
    filtered_data = filtered_data[:, :-1 ]
    
    filtered_uncoated  = filtered_data[filtered_data[:, -1] == 1]
    filtered_coated    = filtered_data[filtered_data[:, -1] ==2]
    
    filtered_uncoated = filtered_uncoated[:,:-1]
    filtered_coated   = filtered_coated[:,:-1]
    if debugging: 
        debug_var(filtered_data ,"filtered_data")
        debug_var(filtered_uncoated ,"filtered_uncoated")
        debug_var(filtered_coated ,"filtered_coated")
    
    

    # Print cluster labels to inspect them
    unique_labels = np.unique(labels)
    
    
    for label in unique_labels:
        # Select data points belonging to the current label
        cluster_data = raw_frequencies[labels == label]
        
        if debugging: 
            # Print the cluster data
            print(f"Cluster {label} (total points: {len(cluster_data)}):")
            print(cluster_data)
            

        indices = np.where(labels == label)[0]
        cluster_data = combined_last_two[indices]
    
        try:
          
            indices = np.where(labels == label)[0]
            cluster_data = combined_last_two[indices]
            # Initialize a flag to indicate if all points are identical
            all_identical = True
        
            # Check each point in the cluster against the first point
            for i, point in enumerate(cluster_data):
                if not np.array_equal(point, cluster_data[0]):
                    if debugging:
                        print(f"Point not identical in cluster {label} at index {indices[i]}: {point}")
                    all_identical = False
            
            if not all_identical:
                if debugging: 
                    print(f"Mode Mismatching identified: {label} are identical.\n")
        except ValueError as e:
            print(e)
        
        if debugging:        
            debug_var(cluster_data)

        
    
        U_AV_updated  = (filtered_uncoated[:,[0,1,2,3,4]])
        U_STD_updated = (filtered_uncoated[:,[5,6]])
        
        C_AV_updated  = (filtered_coated[:,[0,1,2,3,4]])
        C_STD_updated = (filtered_coated[:,[5,6]])

        if debugging:
            debug_var(U_AV_updated,"U_AV_updated")
            debug_var(U_STD_updated,"U_STD_updated")
            debug_var(C_AV_updated,"C_AV_updated")
            debug_var(C_STD_updated,"C_STD_updated")
        
        
    
    matched_data_dict = {
        "m_exp"             : U_AV_updated[:, -2:],                         # experimental mode ids,    both
        "m1_exp"            : U_AV_updated[:, 3],                           # experimental mode ids,    1st, both
        "m2_exp"            : U_AV_updated[:, 4],                           # experimental mode ids,    2nd, both
        "f_exp_uncoated"    : U_AV_updated[:, 0],                           # experimental frequencies, uncoated
        "Q1_exp_uncoated"   : U_AV_updated[:, 1],                           # experimental Q1s,         uncoated
        "Q2_exp_uncoated"   : U_AV_updated[:, 2],                           # experimental Q2s,         uncoated
        "Q1err_exp_uncoated": U_STD_updated[:, 0],                          # experimental Q1 errors,   uncoated
        "Q2err_exp_uncoated": U_STD_updated[:, 1],                          # experimental Q2 errors,   uncoated
        "f_exp_coated"      : C_AV_updated[:, 0],                           # experimental frequencies, coated
        "Q1_exp_coated"     : C_AV_updated[:, 1],                           # experimental Q1s,         coated
        "Q2_exp_coated"     : C_AV_updated[:, 2],                           # experimental Q2s,         coated
        "Q1err_exp_coated"  : C_STD_updated[:, 0],                          # experimental Q1 errors,   coated
        "Q2err_exp_coated"  : C_STD_updated[:, 1],                          # experimental Q2 errors,   coated
        
        "uncoated_ave_final": np.array((U_AV_updated[:, 0], U_AV_updated[:, 1], U_AV_updated[:, 2], U_AV_updated[:, 3], U_AV_updated[:, 4])).T,
        "uncoated_std_final": np.array((U_STD_updated[:, 0], U_STD_updated[:, 1])).T,
        "coated_ave_final"  : np.array((C_AV_updated[:, 0], C_AV_updated[:, 1], C_AV_updated[:, 2], C_AV_updated[:, 3], C_AV_updated[:, 4])).T,
        "coated_std_final"  : np.array((C_STD_updated[:, 0], C_STD_updated[:, 1])).T
    }

    
    
    return matched_data_dict, idx_uncoated, idx_coated


    
    





def prepareMeasuredData(idx,table_data,base_dir,out_dir,temperature,duration,material_dict,suspension_info,Booleans,textfiles,debugging=False,plots=False):
    """
    Prepare measured data for analysis and visualization.
    @param idx - index of the data
    @param table_data - data in tabular format
    @param base_dir - base directory for data
    @param out_dir - output directory for results
    @param temperature - temperature of the measurement
    @param duration - duration of the measurement
    @param material_dict - dictionary of material properties
    @param suspension_info - information about the suspension
    @param Booleans - boolean values for analysis
    @param textfiles - text files for additional information
    @param debugging - flag for debugging mode (default False)
    @param plots - flag for generating plots (default False)
    Author S.Tait 2024 
    """
    
    if debugging: 
        plots = True          
    
    
    if temperature ==30: 
        label = str('_AsDeposited')
        measurement_key = 'Coated Measurements'
    else:
        label = "_".join(["_{}hrs".format(temperature), "{}h".format(duration)])
        if temperature == 600 and duration ==10:
            measurement_key = 'HT1 Measurements'
        elif temperature == 600 and duration ==114: 
             measurement_key = 'HT2 Measurements'
    if measurement_key is None:
        print("Error: 'measurement_key' not assigned.")
        raise 
        
    lookup_table_file = os.path.join(f"{out_dir}measurements_lookup_table.csv")

    s                  = table_data[idx]["Serial"]
    measurement_lookup = read_measurements_for_serial(lookup_table_file,s)
    
    print()
    print(f"Information for serial number {s}:")
    for key, value in measurement_lookup.items():
        # Check if the value is a list; if so, filter out 'NaN', otherwise, leave it as is
        if isinstance(value, list):
            cleaned_value = [item for item in value if item != 'NaN']
        else:
            cleaned_value = value

        # Update the dictionary with the cleaned or original value
        measurement_lookup[key] = cleaned_value
        print(f"{key}: {cleaned_value}")

    material_dict["material"]            = [table_data[idx]["Material"]]                     # material or stack name
    material_dict["thickness"]           = [table_data[idx]["Total Thickness"]]                    # film thickness in [m]
    material_dict["density"]             = [table_data[idx]["Average Stack Density"]]
    material_dict["thickness_substrate"] = [table_data[idx]["Sub Thickness"]]                # substrate thickness in [mm], INPUT AS LIST.   If using any pycrime data from DCC look at the "mode_id" output graph for thickness info FOR THE UNCOATED disk
    material_dict["Y_substrate"]         = [73.2]                                            # substrate Young's modulus in [GPa], INPUT AS LIST
    material_dict["nu_substrate"]        = [0.167]                                           # substrate Poisson Ratio,     INPUT AS LIST
    material_dict["density_substrate"]   = [2220.]                                           # substrate density in kg/m^3, INPUT AS LIST
   
    suspension_info["date_uncoated"]     = measurement_lookup["Uncoated Measurements"]            # date of uncoated suspension you wish to analyse
    suspension_info["date_coated"]       = measurement_lookup[measurement_key]                   # date of coated suspension you wish to analyse
    suspension_info["suspension_number"] = ''                                                    # either put in a number,           i.e 2 (doesn't have to be string) or 'best' as a string for the best value set
    suspension_info["temperature"]       = temperature                                            # heat treatment temperature, input as integer
    suspension_info["duration"]          = duration                                               # heat treatment duration,    input as integer
    suspension_info["sus_label"]         = "_".join([label, suspension_info["suspension_number"]])
    suspension_info['MAX_FREQ_DIFF']    = 400                                     # this defines the max frequency shift analysis cutoff, will want it around 500Hz for a stack (can keep it as this for single layyer too but probably only needs to be  ~50Hz)

                     
    Booleans['coated_duplicate_mode_IDS' ] = ['none']                             # e, g [3,9] removes the data of the 4th and 10th coated modes because of bad measurement or for matching with uncoated data
    Booleans['blank_duplicate_mode_IDS'  ] = ['none']                             # e, g [3,9] removes the data of the 4th and 10th uncoated modes because of bad measurement or for matching with coated data
    Booleans['bad_mode_IDS'              ] = ['none']                             # user input to remove bad modes by giving the mode ID [2,5,8,9,etc] or ['none'].
    Booleans['bad_mode_IDS_SL'           ] = ['none']   
    Booleans['TF_uncoated'               ] = False                              # True / False = Bayesian fits to the already analysed data or not
    Booleans['TF_coated'                 ] = False                              # True / False = Bayesian fits to the already analysed data or not
    Booleans['blankreplace'              ] = ['no']                             # ['yes']/['no'] this is here to rewrite the data it's currently working
    Booleans['coatedreplace'             ] = ['no']           
                                                                                # You can also use bad_mode_IDS to remove things as well, 
                                                                                # it's slightly more automated in that it will remove the same
                                                                                # mode from both datasets however
                                                                                # it was built before the two above and I prefer to use these
                                                                                # and set bad_mode_IDS to ['none']

    
    
    
    outlabel = "_".join([label,"uncoated"])
    uncoated_data , best_uncoated_data, best_uncoated_std =  best_results(serials=s,base_path=base_dir,out_dir=out_dir,measurement_dates=suspension_info["date_uncoated"],label=outlabel,debugging=debugging,plots=plots)
    outlabel = "_".join([label,"coated"])
    coated_df,      best_coated_data   ,best_coated_std   =  best_results(serials=s,base_path=base_dir,out_dir=out_dir,measurement_dates=suspension_info["date_coated"],label=outlabel,debugging=debugging,plots=plots)

    ## write to text files so that they can be in the same format as Graemes Code 

    Q_SUB_txt   = save_dataframe_slice_to_csv(uncoated_data, [0, 1, 2, 9, 10], f"{s}_Best_uncoated.txt", os.path.join(out_dir, f"{s}"))
    ERR_SUB_txt = save_dataframe_slice_to_csv(uncoated_data, [7, 8], f"{s}_Best_uncoated_Errors.txt", os.path.join(out_dir, f"{s}"))
    Q_ED_txt    = save_dataframe_slice_to_csv(coated_df, [0, 1, 2, 9, 10], f"{s}_Best_AsDeposited.txt", os.path.join(out_dir, f"{s}"))
    ERR_ED_txt  = save_dataframe_slice_to_csv(coated_df, [7, 8], f"{s}_Best_AsDeposited_Errors.txt", os.path.join(out_dir, f"{s}"))

    textfiles["Q_SUB_txt"  ] =Q_SUB_txt 
    textfiles["ERR_SUB_txt"] =ERR_SUB_txt
    textfiles["Q_ED_txt"   ] =Q_ED_txt  
    textfiles["ERR_ED_txt" ] =ERR_ED_txt



    # Check if all files exist
    files_to_check = [Q_SUB_txt, ERR_SUB_txt, Q_ED_txt, ERR_ED_txt]

    for file in files_to_check:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} was not created successfully!")
    print("All files created successfully.")

    # Match Uncoated nd Coated modes and run quick coating loss analysis on stack, pulling out measured COATED Qs at the corresponding frequencies

    set_ylimit = ['no']          # manually choose if you want to set y axis limits for coating loss graph ['yes'] for manual, ['no'] for automatic
    YLIMT= [0, 0.012]            # manual specifications for y axis limits for output coating loss graph units of x10^{-3}. 

    # calculates coating loss of the disk you just input details for, mostly used in this anaysis for stacks, but no reason it cant be done for single layers
 
        
        
    f_stack, m_exp, uncoated_ave_stack, uncoated_std_stack, coated_ave_stack,\
        coated_std_stack, Dbulk, Dshear, comsol_modes, s2ds, s2delplus6s,elasticProps=stack_values_extractor_replacingvalues(s, base_dir, out_dir, material_dict=material_dict, suspension_info=suspension_info, flags=Booleans, textfiles=textfiles, MAX_FREQ_DIFF=MAX_FREQ_DIFF,debugging=debugging)


 
    
    # Find membership of rows in m_exp within comsol_modes
    matches = ismember(np.array(m_exp), np.array(comsol_modes))


    mask = np.zeros(len(comsol_modes), dtype=bool)
    mask[matches] = True

    # Use the mask to select elements to keep

    comsolmod       = comsol_modes[mask]
    dbulk           = Dbulk[mask]
    dshear          = Dshear[mask]
    s2dscut         = s2ds[mask]
    s2delplus6scut  = s2delplus6s[mask]


    print("Performing sanity checks... ")
    if debugging:
        # if the first 3 here all have the same length that is a good first check
        print("Check Lengths:", check_lengths(f_stack, comsolmod, dbulk, dshear))

        # if the last 2 have the same ordering on visual inspection that is your second check cutting has been done correctly
        print("Check Ordering:", check_ordering(comsolmod, m_exp))

    if check_lengths(f_stack, comsolmod, dbulk, dshear) and check_ordering(comsolmod, m_exp):
        print('Success: All Checks Passed ')
        if debugging: 
            print()
            print(f"identified modes:\n {comsol_modes[matches]}")
        
    else:
        raise ValueError('Checks failed!')

    
    
    data_dict = {} 
    temp_dur_key = (temperature, duration)
    
    if baysian:  
        
        #BAYSIAN FITS    
        
        #from pycrime.data_analysis.bayesian_loss_angle_75mm_1mm import *        
        
        if len(s) == 1:
            quantiles = {s: {}}
            samples = {s: {}}
            model_loglike = {s: {}}
            phi_bulk = {s: {}}
            phi_shear = {s: {}}
            labels = {s: {}}
            phi_bulk_samples = {s: {}}
            phi_shear_samples = {s: {}}
        else: 
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
            
            
        MCMC_dict = {
            "G_A": G_A,
            "G_B": G_B,
            "G_C": G_C,
            "G_D": G_D,
            "A": A,
            "fig": fig,
            "best_slope": best_slope,
            "best_split": best_split
        }
            
    
    data_dict = {}
    
    data_dict[s] = data_dict.get(s, {})  # Ensure this serial number has a dictionary
    data_dict[s][temp_dur_key] = {
        "f_stack"           : f_stack,
        "m_exp"             : m_exp,
        "uncoated_ave_stack": uncoated_ave_stack,
        "uncoated_std_stack": uncoated_std_stack,
        "coated_ave_stack"  : coated_ave_stack,
        "coated_std_stack"  : coated_std_stack,
        "Dbulk"             : Dbulk,
        "Dshear"            : Dshear,
        "Eratio"            : Dbulk/Dshear,
        "comsol_modes"      : comsol_modes,
        "s2ds"              : s2ds,
        "s2delplus6s"       : s2delplus6s,
        "s2dscut"           : s2ds[mask],
        "s2delplus6scut"    : s2delplus6s[mask],
        "elasticProps"      : elasticProps,
        "material_dict "    : material_dict,
        "suspension_info "  : suspension_info,
        "Booleans "         : Booleans,
        "textfiles"         : textfiles,
    }
    
    data_dict[s] = data_dict.get(s, {})  # Ensure this serial number has a dictionary
    data_dict[s][temp_dur_key] = {
        "f_stack"           : f_stack,
        "m_exp"             : m_exp,
        "uncoated_ave_stack": uncoated_ave_stack,
        "uncoated_std_stack": uncoated_std_stack,
        "coated_ave_stack"  : coated_ave_stack,
        "coated_std_stack"  : coated_std_stack,
        "Dbulk"             : Dbulk,
        "Dshear"            : Dshear,
        "Eratio"            : Dbulk/Dshear,
        "comsol_modes"      : comsol_modes,
        "s2ds"              : s2ds,
        "s2delplus6s"       : s2delplus6s,
        "s2dscut"           : s2ds[mask],
        "s2delplus6scut"    : s2delplus6s[mask],
        "elasticProps"      : elasticProps,
        "MCMC_dict"         : MCMC_dict,
        "material_dict "    : material_dict,
        "suspension_info "  : suspension_info,
        "Booleans "         : Booleans,
        "textfiles"         : textfiles,
    }

    return data_dict


def convert_tuple_keys(d):
    """
    Converts tuple keys to strings and tuple values to lists in a dictionary for JSON serialization.
    @param d - A dictionary possibly containing tuple keys and values.
    @return A JSON-serializable dictionary.
    """
    """
    Recursively converts tuple keys to strings and tuple values to lists in a dictionary, 
    making it JSON-serializable. Nested dictionaries are also processed. 
    Tuple keys are converted to strings by joining their string representations with underscores. 
    This approach ensures the dictionary can be serialized using JSON without losing the 
    structure of tuple keys or values.
    
    Parameters:
    - d (dict): A dictionary possibly containing tuple keys and values.
    
    Returns:
    - dict: A dictionary with all tuple keys and values converted to strings and lists, respectively.
    Author S.Tait 2024 
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary.")
    
    serializable_dict = {}
    
    
    for key, value in d.items():
        # Convert tuple keys to string by joining with underscore to prevent key ambiguity.
        new_key = "_".join(str(item) for item in key) if isinstance(key, tuple) else key
        
        # Recursively convert nested dictionaries or convert tuple values to lists.
        if isinstance(value, dict):
            new_value = convert_tuple_keys(value)
        elif isinstance(value, tuple):
            new_value = list(value)
        else:
            new_value = value
        
        serializable_dict[new_key] = new_value
    
    return serializable_dict


def read_measurements_for_serial(csv_file_path, serial_number):
    """
    Read measurements from a CSV file for a specific serial number.
    @param csv_file_path - the path to the CSV file
    @param serial_number - the serial number to retrieve measurements for
    @return A dictionary of measurements for the specified serial number, or None if not found.
    Author S.Tait 2024 
    """

    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            current_serial, measurements_str = row
            if current_serial == serial_number:
                # Split the measurements string into individual measurement types
                measurement_types = measurements_str.split(' | ')
                measurements_dict = {}
                for measurement in measurement_types:
                    # Extract the measurement name and its dates
                    name, dates_str = measurement.split(': ')
                    # Parse the dates, removing quotes and splitting by comma
                    dates = [date.strip("'") for date in dates_str.split(', ')]
                    measurements_dict[name] = dates
                return measurements_dict
    # Return None or an empty dictionary if the serial number is not found
    return None


def read_measurements(csv_file_path, **flags):
    """
    Read measurements from a CSV file, returning data based on specified flags.
    @param csv_file_path: The path to the CSV file.
    @param flags: Keyword arguments controlling the output.
    @return: Depending on flags, either a list of serials or a dictionary keyed by serials with specified measurement dates.
    Author S.Tait 2024 
    
    """
    
    data = {}
    
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            serial_number, measurements_str = row
            measurement_types = measurements_str.split(' | ')
            for measurement in measurement_types:
                if ': ' in measurement:
                    name, dates_str = measurement.split(': ')
                    dates = [date.strip("'") for date in dates_str.split(', ')]
                    if serial_number not in data:
                        data[serial_number] = {}
                    data[serial_number][name.strip()] = dates
    
    if 'get' in flags:
        get_type = flags['get'].capitalize()  # Adjusting for case sensitivity
        get_type += " Measurements"  # Adjusting name to match expected format
        if get_type == 'Serials Measurements':  # Handling 'serials' flag separately
            return list(data.keys())
        else:
            specific_data = {serial: measurements.get(get_type, []) for serial, measurements in data.items()}
            return specific_data if any(specific_data.values()) else f"No data found for '{get_type}'."
    
    return "Invalid or missing flags."



def plot_material_properties(serials, out_dir,plots=False):
    """
    Plot the material properties for a list of serials and display the plots if specified.
    @param serials - list of serial numbers
    @param out_dir - directory where the output will be saved
    @param plots - boolean flag to display the plots
    @return None
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    fig.set_facecolor('white')

    xshift = 50


    
    for serial in serials:
        json_file_path = os.path.join(out_dir, serial, f'{serial}_MaterialProperties.json')
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {json_file_path}")
            continue
        
        # Initialize lists to store plot data
        temperatures, durations = [], []
        Yb_values, Yb_errors = [], []
        nub_values, nub_errors = [], []
        eb_values, eb_errors = [], []

        for key, value in data.items():
            if all(k in value for k in ['temperature', 'duration', 'Yb', 'nub', 'eb']):
                temperature = value['temperature']
                duration = value['duration']
                Yb, nub, eb = value['Yb'], value['nub'], value['eb']
                
                # Adjust temperature based on duration condition
                if duration > 100:
                    temperature += xshift 
                
                temperatures.append(temperature)
                durations.append(duration)  # This line is optional if duration is only used for the condition
                Yb_values.append(Yb[0])
                Yb_errors.append(Yb[1])
                nub_values.append(nub[0])
                nub_errors.append(nub[1])
                eb_values.append(eb[0])
                eb_errors.append(eb[1])
            else:
                print(f"Missing key in data entry {key} for serial {serial}.")
                
        custom_labels = {30: 'As Deposited ', 600: '10 hrs', 600+xshift:'114 hrs ' }
        # Plotting
        if temperatures:
            axs[0].errorbar(temperatures, Yb_values, yerr=Yb_errors, label=serial, fmt='o')
            axs[1].errorbar(temperatures, nub_values, yerr=nub_errors, label=serial, fmt='o')
            axs[2].errorbar(temperatures, eb_values, yerr=eb_errors, label=serial, fmt='o')
    
        # Setting titles for each subplot
        axs[0].set_title('Young\'s Modulus', fontsize=16) 
        axs[1].set_title('Poisson Ratio', fontsize=16) 
        axs[2].set_title('Edge', fontsize=16) 
        
        # axs[0].set_xscale('log')
        # axs[1].set_xscale('log')
        # axs[2].set_xscale('log')
        
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        
        axs[0].set_xlabel('Heat Treatment Tempeature', fontsize=12) 
        axs[1].set_xlabel('Heat Treatment Tempeature', fontsize=12) 
        axs[2].set_xlabel('Heat Treatment Tempeature', fontsize=12) 
        
        axs[0].set_ylabel('Youngs ', fontsize=12) 
        axs[1].set_ylabel('Poisson', fontsize=12) 
        axs[2].set_ylabel('edge', fontsize=12) 

        axs[0].legend(loc='center')
        axs[1].legend(loc='center')
        axs[2].legend(loc='center')
        
        for ax in axs:
            # Get current ticks and labels
            current_ticks = ax.get_xticks()
            new_labels = [custom_labels.get(temp, str(temp)) for temp in current_ticks]
            ax.set_xticks(current_ticks)
            ax.set_xticklabels(new_labels, rotation=45)
        
        
        
        
        plt.tight_layout()
        
        plt.show()
        plt.savefig(os.path.join(out_dir,'MaterialProperties_withTempeature.png'),facecolor='white')
        
        
    if plots:     
        display_html = f"""
        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
            <div style='width: 60%;'> <img src='data:image/png;base64,{fig_to_base64(fig)}'/> </div>
        </div>
        """
        
        display(HTML(display_html.strip()))    
        
    return fig
        
        

def create_dataframe_from_json(serials, out_dir):
    rows_list = []

    for serial in serials:
        file_path = os.path.join(out_dir, serial, f'{serial}_MaterialProperties.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        
        serial_data = {'Serial': serial}

        for key, value in data.items():
            temp_dur_label = f"{value['temperature']}_deg_{value['duration']}_hrs"

            # Concatenating value and error for each property in the form "Value +/- Error"
            serial_data[f'Y_{temp_dur_label}'] = f"{value['Yb'][0]:.2f} +/- {value['Yb'][1]:.2f}"
            serial_data[f'nu_{temp_dur_label}'] = f"{value['nub'][0]:.2f} +/- {value['nub'][1]:.2f}"
            serial_data[f'e_{temp_dur_label}'] = f"{value['eb'][0]:.2f} +/- {value['eb'][1]:.2f}"

        rows_list.append(serial_data)

    df = pd.DataFrame(rows_list)
    return df

def normalize_for_json(data):
    if isinstance(data, dict):
        return {str(key): normalize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [normalize_for_json(item) for item in data]
    elif isinstance(data, type({}.keys())):
        # Correctly identify dict_keys objects
        return list(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        raise TypeError(f"Type {type(data)} not serializable")


    
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # Optionally, handle additional types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add more custom type handling here
        return json.JSONEncoder.default(self, obj)

def save_to_json(data, file_path):
    normalized_data = normalize_for_json(data)
    with open(file_path, 'w') as f:
        json.dump(normalized_data, f, cls=CustomEncoder, indent=4)

def load_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
