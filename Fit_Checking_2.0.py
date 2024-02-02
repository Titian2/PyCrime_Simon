import argparse
import os
import re
import pandas as pd
import time 
import logging
import configparser
import glob 
from PIL import Image
import numpy as np 
from pathlib import Path
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_config(config_file: str) -> configparser.ConfigParser:
    """Reads configuration from a file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def get_sample_name(sample_name: str) -> str:
    """Returns the sample name."""
    return sample_name

def get_folders(base_dir: Path) -> List[str]:
    """Retrieves folders in the base directory."""
    try:
        directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        folders = [d for d in directories if re.match(r'\d{4}_\d{2}_\d{2}', d)]
        return folders
    except Exception as e:
        logging.error(f"Error getting folders: {e}")
        return []

def process_folders(folders: List[str], date_string: str, verbose: bool) -> Tuple[List[str], List[str], List[str]]:
    """Processes folder names to extract dates and letters based on provided date string."""
    dates, letters, matched_folders = [], [], []
    for folder in folders:
        try:
            if re.match(rf'{date_string}[a-zA-Z]*_CR', folder):
                matched_folders.append(folder)
                logging.info(f"Matched folder: {folder}")

                # Extracting the letter part
                letter_match = re.search(rf'{date_string}([a-zA-Z]*)_CR', folder)
                if letter_match:
                    letter = letter_match.group(1)
                    dates.append(date_string)
                    letters.append(letter)
                elif verbose:
                    logging.warning(f"Letter pattern not found in folder '{folder}'")
            elif verbose:
                logging.warning(f"Folder '{folder}' does not match the provided date '{date_string}'")
        except Exception as e:
            logging.error(f"Error processing folder '{folder}': {e}")

    if matched_folders:
        print()
        logging.info(f"Total matched folders: {len(matched_folders)}")
    else:
        logging.warning("No folders matched the given criteria.")
    print()
    return dates, letters, matched_folders

def find_matching_image(freq, fit_images):
    """
    Attempts to find a matching image for the given frequency by checking both rounded up and down.
    Returns the first matching image found, or None if no match is found.
    """
    # Round down if the decimal part is exactly .5, otherwise use standard rounding
    freq_round_down = math.floor(freq) if (freq - math.floor(freq)) == 0.5 else round(freq)
    # Always round up if the decimal part is .5
    freq_round_up = math.ceil(freq)
    
    # First, try matching with freq_round_down
    match_down = [img for img in fit_images if re.search(f"fit_{freq_round_down}\D", os.path.basename(img))]
    if match_down:
        return match_down[0]  # Assuming only one match is expected, return the first match
    
    # If no match found, try with freq_round_up
    match_up = [img for img in fit_images if re.search(f"fit_{freq_round_up}\D", os.path.basename(img))]
    if match_up:
        return match_up[0]  # Assuming only one match is expected, return the first match
    
    # If no match is found with either, return None
    return None

def read_and_process_files(folders, base_dir,verbose):
    
    all_data = pd.DataFrame()
    matched_fit_images = []
    matched_results_files = []
    
    if verbose: 
        print("{:<40} {:<15} {:<15}".format("Fit Image", "Image Freq", "Results Freq"))
        print("-" * 70)


    for folder in folders:
        fit_images = glob.glob(os.path.join(base_dir, folder, 'fit*.png'))
        folder_result_files = glob.glob(os.path.join(base_dir, folder, 'results*.txt'))
    
        for result_file in folder_result_files:
            temp_df = pd.read_csv(result_file, sep='\t', skiprows=[0], header=None, engine='python')
            temp_df = temp_df.dropna(axis=1, how='all')
            temp_df.columns = ['Freq', 'Q1', 'Q2', 'Q1_CI', 'Q1_CI.1', 'Q2_CI', 'Q2_CI.1', 'mode(m)', 'mode(n)']


            for _, row in temp_df.iterrows():
                freq = row['Freq']
                matching_image = find_matching_image(freq, fit_images)
                if matching_image:
                    matched_fit_images.append(matching_image)
                    matched_results_files.append(result_file)
                    img_freq_match = re.search(r"fit_(\d+)", os.path.basename(matching_image))
                    img_freq = img_freq_match.group(1) if img_freq_match else "N/A"
                    if verbose:
                        print("{:<40} {:<15} {:<15}".format(os.path.basename(matching_image), img_freq, str(freq)))
                else:
                    
                    print(f"No matching fit image found for frequency {freq} in {result_file}")
            
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
      
    return  all_data, matched_fit_images, matched_results_files
"""
def read_and_process_files(folders, base_dir):
    all_data = pd.DataFrame()
    matched_fit_images = []
    matched_results_files = []

    #print("{:<40} {:<15} {:<15}".format("Fit Image", "Image Freq", "Results Freq"))
    #print("-" * 70)

    for folder in folders:
        # Collecting fit image paths
        fit_images = glob.glob(os.path.join(base_dir, folder, 'fit*.png'))
        
        # Collecting result file paths
        folder_result_files = glob.glob(os.path.join(base_dir, folder, 'results*.txt'))
        
        # Process each result file and match with fit images
        for result_file in folder_result_files:
            temp_df = pd.read_csv(result_file, sep='\t', skiprows=1, engine='python')
            temp_df = temp_df.dropna(axis=1, how='all')
            temp_df.columns = ['Freq', 'Q1', 'Q2', 'Q1_CI', 'Q1_CI.1', 'Q2_CI', 'Q2_CI.1', 'mode(m)', 'mode(n)']
            
            for _, row in temp_df.iterrows():
                freq = custom_round(row['Freq'])
                print(freq)
                match = [img for img in fit_images if re.search(f"fit_{freq}\D", os.path.basename(img))]
                if match:
                    matched_fit_images.append(match[0])
                    matched_results_files.append(result_file)
                    img_freq_match = re.search(r"fit_(\d+)", os.path.basename(match[0]))
                    if img_freq_match:
                        img_freq = img_freq_match.group(1)
                       # print("{:<40} {:<15} {:<15}".format(os.path.basename(match[0]), img_freq, str(freq)))
                else:
                    print(f"No matching fit image found for frequency {freq} in {result_file}")
            
            all_data = pd.concat([all_data, temp_df], ignore_index=True)

    return all_data, matched_fit_images, matched_results_files
""" 

def perform_kmeans_analysis(data,unique_counts):
    # Assuming 'Freq' is a column in your DataFrame
    if 'Freq' in data.columns:
        # Perform k-means clustering
        freq = data['Freq'].values

        kmeans = KMeans(n_clusters=unique_counts)
        data['Cluster'] = kmeans.fit_predict(data[['Freq']])
        labels = kmeans.fit_predict(freq.reshape(-1, 1))
        groups = np.unique(labels)

         # Calculate indices for each group
        group_indices = {}
        group_indices = {group: np.where(labels == group)[0] for group in groups}

        
    else:
        print("Column 'Freq' not found in data.")
        # Handle the case where 'Freq' is not present
    return data, group_indices

def perform_dbscan_analysis(data, column_name,verbose,eps=0.5, min_samples=5):
    # Extracting the column to cluster
    values = data[column_name].values.reshape(-1, 1)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(values)
    labels = clustering.labels_

     # Assigning the cluster labels to the DataFrame
    data['Cluster'] = labels

    # Handling noise points (labelled as -1)
    unique_labels = set(labels) - {-1}

    # Creating a dictionary to hold indices for each cluster
    group_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    return group_indices, data , labels 

    
def plot_data(data):
    plt.figure()

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Freq'], 1/cluster_data['Q1'])  # Replace with your actual columns
    plt.grid('on')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mechanical Loss')
    plt.title('Number of Clusters identified:' + str(np.size(data['Cluster'].unique())))
    plt.show()


def open_image(image_path):
    try:
        os.system(f'open "{image_path}"')  # For macOS
        # For Windows, use os.system(f'start "{image_path}"')
        # For Linux, use os.system(f'xdg-open "{image_path}"')
    except Exception as e:
        print(f"Error opening file {image_path}: {e}")

def user_fit_check(image_paths, max_figsize=(10, 8)):
    fit_checks = []
    for path in image_paths:
        try:
            # Load the image
            img = Image.open(path)
            img_width, img_height = img.size

            # Calculate aspect ratio and adjust figure size
            aspect_ratio = img_width / img_height
            fig_width, fig_height = max_figsize
            if aspect_ratio >= 1:
                fig_height = fig_width / aspect_ratio
            else:
                fig_width = fig_height * aspect_ratio

            # Create a new figure with a dynamically adjusted size
            plt.figure(figsize=(fig_width, fig_height))
            
            # Display the image without blocking
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show(block=False)
            
            # Prompt user for input while the image is displayed
            user_input = input("Are you happy with this fit [Y/N]? ").strip().lower()
            fit_checks.append(user_input == 'y' or user_input == '')
            
            # Close the image window programmatically
            plt.close()
            
        except Exception as e:
            print(f"Error loading or displaying image {path}: {e}")
            fit_checks.append(False)
        
    return fit_checks

def process_data_and_calculate_averages(data, fit_checks, group_label):
    data['FitCheck'] = fit_checks
    data['Group'] = group_label
    filtered_data = data[data['FitCheck']]
    
    # Calculate averages and standard deviations
    averages = filtered_data.mean()
    std_devs = filtered_data.std()

    return averages, std_devs

def get_user_decision_on_cluster_count(actual_count,clustered_data,expected_count=17):
    if actual_count != expected_count:
        print(f"Only {actual_count} frequencies were identified in the clustering, expected {expected_count}.")
        plot_data(clustered_data)
        user_decision = input("Do you want to continue with these clusters? [Y/N]: ").strip().lower()
         
        if user_decision == 'y':
            return True
        else:
            new_count = int(input("Enter the new number of clusters: "))
            return new_count
    else:
        return True



def image_paths_for_group(group_label, group_indices, fit_images,verbose):
    """
    Retrieve fit image paths for a specific group.
    
    :param group_label: The label of the group.
    :param group_indices: A dictionary mapping group labels to lists of indices.
    :param fit_images: A list of all fit image paths.
    :return: A list of image paths for the given group.
    """
    
    
    indices = group_indices.get(group_label, [])
    if verbose: 
        print(f"Group {group_label} Indices: {indices}")  # Debugging print
        print(f"Image Paths for Group {group_label}: {img_paths}")  # Debugging print
    img_paths = [fit_images[i] for i in indices if i < len(fit_images)]
    
       
    return img_paths
    
def plot_group_error_bars(all_averages, all_std_devs, group_label,sample_name,date_string):
    """
    Plot error bars for the specified group and save the figure with a name that includes the date string.
    
    :param all_averages: Dictionary of averages indexed by group_label.
    :param all_std_devs: Dictionary of standard deviations indexed by group_label.
    :param group_label: The label of the group being plotted.
    :param date_string: String representing the date, used in the filename.
    """
    if group_label not in all_averages or group_label not in all_std_devs:
        print(f"No data available for group {group_label}.")
        return

    averages = all_averages[group_label]
    std_devs = all_std_devs[group_label]

    # Extracting necessary values for plotting
    avg_freq = averages['Freq']
    avg_inv_Q1 = 1 / averages['Q1']
    std_inv_Q1 = 1 / std_devs['Q1']
    avg_inv_Q2 = 1 / averages['Q2']
    std_inv_Q2 = 1 / std_devs['Q2']
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(avg_freq, avg_inv_Q1, yerr=std_inv_Q1, fmt='o', label=f'1/Q1 for Group {group_label}', capsize=5)
    plt.errorbar(avg_freq, avg_inv_Q2, yerr=std_inv_Q2, fmt='o', label=f'1/Q2 for Group {group_label}', capsize=5, color='red')
    
    plt.xlabel('Average Frequency (Hz)')
    plt.ylabel('Inverse Q')
    plt.title(f'Error Bar Plot of Inverse Qs for Group {group_label}')
    plt.legend()
    plt.grid(True)
    
    # Saving the figure
    filename = f'AverageLoss_{sample_name}_{date_string}.png'
    plt.savefig(filename)
    print(f"Figure saved as {filename}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('sample_name', type=str, help='Name of the sample')
    parser.add_argument('date_string', type=str, help='Date string in YYYY_MM_DD format')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')

    args = parser.parse_args()

    config = read_config('config.ini')
    base_dir = Path(config.get('Paths', 'BaseDir', fallback='/Volumes/UNTITLED/')) / get_sample_name(args.sample_name)

    logging.info(f'Sample Name: {base_dir.name}')

    folders = get_folders(base_dir)
    dates, letters, matched_folders = process_folders(folders, args.date_string, args.verbose)

    all_data, fit_images, results_files = read_and_process_files(matched_folders, base_dir,args.verbose)
    
    
    
    # Initial cluster count
    unique_counts = len(np.unique(np.ceil(all_data['Freq'] / 100) * 100))

    print("Clustering...")
    # Perform initial clustering
    
    
    
    
    group_indices, data , group_labels  = perform_dbscan_analysis(all_data, 'Freq',args.verbose, eps=0.5, min_samples=5)
    
    labels = group_labels

    # Adding the 'Group' column to 'data'
    data['Group'] = labels
    
    
    # Check cluster count and get user decision
    if np.size(np.unique(group_labels)) !=18:
    
        decision = get_user_decision_on_cluster_count(len(group_labels),data)

        # If user provides a new cluster count
        if isinstance(decision, int):
            unique_counts = decision
            group_indices = perform_kmeans_analysis(all_data,unique_counts)
            group_labels = list(group_indices.keys())
    else:
        decision = True
        print('All Frequencies Identified!')

    if decision:
        
        # Assuming group_labels contains the labels of each group you want to process
        group_labels = sorted(group_indices.keys())  # Sorting the group labels for consistent processing order
        fit_checks_by_group = {}  # Initialize a dictionary to store fit checks by group

        all_averages = {}
        all_std_devs = {}

        for group_label in group_labels:
            # Get the relevant image paths for this group
            img_paths = image_paths_for_group(group_label, group_indices, fit_images,args.verbose)
            
            # Perform user fit check for the current group
            fit_checks = user_fit_check(img_paths)
            
            # Filter the data for the current group
            group_data = data[data['Group'] == group_label]
            
            # Ensure group_data has the same number of rows as there are fit_checks
            if len(group_data) != len(fit_checks):
                print(f"Warning: Mismatch in number of fit checks ({len(fit_checks)}) and data rows ({len(group_data)}) for group {group_label}")
                continue  # Skip to the next group or handle this case as needed
            
            # Add fit_checks to group_data
            group_data = group_data.assign(FitCheck=pd.Series(fit_checks).values)
            
            # Process data and calculate averages
            averages, std_devs = process_data_and_calculate_averages(group_data, fit_checks, group_label)
            
           # Store the results in dictionaries indexed by group_label
            all_averages[group_label] = averages
            all_std_devs[group_label] = std_devs
            
        # Now plot the error bars using the averages and std_devs
        plot_group_error_bars(all_averages, all_std_devs, group_label,args.sample_name,args.date_string)

    else:
         print(f"Correct Number of clusers could not be found. Exciting...")
         raise 
    


   

if __name__ == "__main__":
    main()
