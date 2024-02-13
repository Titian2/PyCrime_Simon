import argparse
import os
import re
import pandas as pd
import numpy as np
import time 
import logging
import configparser
import glob 
from PIL import Image
from pathlib import Path
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
import warnings
from scipy import stats
import numpy_groupies as npg 


import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter-based backend
import matplotlib.pyplot as plt

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
    Freq_round_down = math.floor(freq) if (freq - math.floor(freq)) == 0.5 else round(freq)
    # Always round up if the decimal part is .5
    Freq_round_up = math.ceil(freq)
    
    # First, try matching with Freq_round_down
    match_down = [img for img in fit_images if re.search(f"fit_{Freq_round_down}\D", os.path.basename(img))]
    if match_down:
        return match_down[0]  # Assuming only one match is expected, return the first match
    
    # If no match found, try with Freq_round_up
    match_up = [img for img in fit_images if re.search(f"fit_{Freq_round_up}\D", os.path.basename(img))]
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
            temp_df.columns = ['Freq', 'Q1', 'Q2', 'Q1_UpperCI', 'Q1_LowerCI', 'Q2_UpperCI', 'Q2_LowerCI', 'mode(m)', 'mode(n)']


            for _, row in temp_df.iterrows():
                freq = row['Freq']
                matching_image = find_matching_image(freq, fit_images)
                if matching_image:
                    matched_fit_images.append(matching_image)
                    matched_results_files.append(result_file)
                    img_Freq_match = re.search(r"fit_(\d+)", os.path.basename(matching_image))
                    img_freq = img_Freq_match.group(1) if img_Freq_match else "N/A"
                    if verbose:
                        print("{:<40} {:<15} {:<15}".format(os.path.basename(matching_image), img_freq, str(freq)))
                else:
                    
                    print(f"No matching fit image found for frequency {freq} in {result_file}")
            
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
      
    return  all_data, matched_fit_images, matched_results_files

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

def perform_dbscan_analysis(data, column_name,verbose,eps=0.5, min_samples=2):
    if verbose:
        print(data)
    # Extracting the column to cluster
    values = data[column_name].values.reshape(-1, 1)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(values)
    labels = clustering.labels_
    labels  = labels 
     # Assigning the cluster labels to the DataFrame
    data['Cluster'] = labels

    # Handling noise points (labelled as -1)
    #unique_labels = set(labels) - {-1}
    unique_labels =np.unique(labels)

    # Creating a dictionary to hold indices for each cluster
    group_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    
    
    
    return group_indices, data , labels 

def set_window_position(fig, x, y):
    """
    Set the position of the window associated with the given figure.

    Parameters:
        fig: matplotlib.figure.Figure
            The figure object for which to set the window position.
        x: int
            The x-coordinate of the window position.
        y: int
            The y-coordinate of the window position.
    """
    # Get the Tkinter window associated with the figure
    tk_window = fig.canvas.manager.window
    
    # Set the position of the window
    tk_window.geometry(f'+{x}+{y}')


def plot_data(data):
    
    markers =  np.tile(['o','^','p'],int(np.round(17/2)+1))
    counter = 0 
    fig,ax = plt.subplots()

    for cluster in data['Cluster'].unique():
        
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(cluster_data['Freq'], 1/cluster_data['Q1'],marker= markers[counter])  # Replace with your actual columns
        counter = counter +1
    plt.grid('on')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mechanical Loss')
    plt.title('Number of Clusters identified:' + str(np.size(data['Cluster'].unique())))
    # Plot some data
# Set the window position
    fig.set_size_inches(10, 8)
    # Set the window size and position
    fig.canvas.manager.window.geometry("1000x800+100+100")
    plt.show(block=False)


def open_image(image_path,verbose):
    try:
        if verbose:
            print(image_path)
        os.system(f'open "{image_path}"')  # For macOS
        # For Windows, use os.system(f'start "{image_path}"')
        # For Linux, use os.system(f'xdg-open "{image_path}"')
    except Exception as e:
        print(f"Error opening file {image_path}: {e}")
        
        

def user_fit_check(image_paths,verbose, max_figsize=(10, 8)):
    
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
            if verbose:
                print(f" Function: user_fit_check - img_path = {path}")
           
            
            # Display the image without blocking
            fig, ax = plt.subplots()  # Use subplots to correctly create fig and ax
            ax.imshow(img)
            ax.axis('off')  # Turn off axis numbers and ticks
            fig.set_size_inches(fig_height, fig_width)

            # Set the window size and position
            fig.canvas.manager.window.geometry("1000x800+100+100")

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


def process_data(filtered_data,group_label):
    
    filtered_data['Group'] = group_label  
    

    
    for label in filtered_data['Group'].unique():
        group_data = filtered_data[filtered_data['Group'] == label]
        
        labels = group_data['Cluster'].unique()
        rmidx = np.array([], dtype=int)
        
        for cluster_label in labels:
            cluster_data = group_data[group_data['Cluster'] == cluster_label]
            
             #Compute the median and MAD for 'Q1'
            q1_median = cluster_data['Q1'].median()
            q1_mad = np.median(np.abs(cluster_data['Q1'] - q1_median))
            
              #Compute the median and MAD for 'Q2'
            q2_median = cluster_data['Q2'].median()
            q2_mad = np.median(np.abs(cluster_data['Q2'] - q1_median))

            if not cluster_data['Q1'].empty:
                # Compute the outlier mask using the modified Z-score method with 1.4826 * MAD for conversion to standard deviation
                outlier_mask = cluster_data['Q1'].apply(lambda x: np.abs(x - q1_median) > 1.4826 * q1_mad)
        
                # Update 'FitCheck' for outliers
                filtered_data.loc[cluster_data[outlier_mask].index, 'FitCheck'] = False
                
            if not cluster_data['Q2'].empty:
                # Compute the outlier mask using the modified Z-score method with 1.4826 * MAD for conversion to standard deviation
                outlier_mask = cluster_data['Q2'].apply(lambda x: np.abs(x - q2_median) > 1.4826 * q2_mad)
        
                # Update 'FitCheck' for outliers
                filtered_data.loc[cluster_data[outlier_mask].index, 'FitCheck'] = False
            
        # Remove outliers and append the cleaned group data to the resulting DataFrame
        
     
        
    return filtered_data


def get_user_decision_on_cluster_count(actual_count,clustered_data,expected_count=18):
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
    
       
       
    img_paths = [fit_images[i] for i in indices if i < len(fit_images)]
    
    if verbose: 
        print(f"Group {group_label} Indices: {indices}")  # Debugging print
        print(f"Image Paths for Group {group_label}: {img_paths}")  # Debugging print
       
    return img_paths
    
    
    #Plot error bars for the specified group and save the figure with a name that includes the date string.

def calculate_group_stats_and_plot(filtered_group_data,sample_name,date_string,base_dir,verbose,confidence_level=0.95):
        # Get unique cluster labels

        group_idx = filtered_group_data[filtered_group_data['FitCheck']==True]['Group']
        Frequencies  = filtered_group_data[filtered_group_data['FitCheck']==True]['Freq']
        Q1Values = filtered_group_data[filtered_group_data['FitCheck']==True]['Q1']
        Q2Values = filtered_group_data[filtered_group_data['FitCheck']==True]['Q2']
        
        freq = npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Frequencies), func='mean')
        Q1means =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='mean')
        Q2means =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='mean')
        Q1max   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='max')
        Q2max   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='max')
        Q1min   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q1Values), func='min')
        Q2min   =  npg.aggregate(np.array(group_idx).astype(int)+1, np.array(Q2Values), func='min')

        fig, ax = plt.subplots()
        plt.errorbar(freq,1/Q1means,[np.abs(1/Q1max-1/Q1means),np.abs(1/Q1means-1/Q1min)],marker= 'o',color ='b', linestyle='None')
        plt.errorbar(freq,1/Q2means,[np.abs(1/Q2max-1/Q2means),np.abs(1/Q2means-1/Q2min)],marker= 'o',color ='r' ,linestyle='None')
        plt.title(f"{sample_name} {date_string}")
        plt.legend()
        plt.xlabel('Frequency [Hz')
        fig.canvas.manager.window.geometry("1000x800+100+100")
        plt.ylabel(r'Average Mechanical Loss [$phi_{mech}$]')
        plt.show()
        
              
        # Saving the figure
        filename = f'AverageLoss_{sample_name}_{date_string}.png'
        plt.savefig(os.path.join(base_dir,filename))
        print(f"Figure saved as {filename}")


        results_df = pd.DataFrame({
            'Frequency': freq,
            'Q1 Mean': Q1means,
            'Q2 Mean': Q2means,
            'Q1 Max': Q1max,
            'Q2 Max': Q2max,
            'Q1 Min': Q1min,
            'Q2 Min': Q2min,
            })
        
   
    
        if verbose: 
            print(results_df)
        
        return results_df 

class Args:
    def __init__(self, date_string, sample_name, verbose=False):
        self.date_string = date_string
        self.sample_name = sample_name
        self.verbose = verbose

## Example usage, mimicking command-line input:
#args = Args(date_string="2024_01_10", sample_name="S1600962", verbose=False)



def main():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('sample_name', type=str, help='Name of the sample')
    parser.add_argument('date_string', type=str, help='Date string in YYYY_MM_DD format')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity' )
    args = parser.parse_args()

    config = read_config('config.ini')
    base_dir = Path(config.get('Paths', 'BaseDir', fallback='/Volumes/UNTITLED/')) / get_sample_name(args.sample_name)

    logging.info(f'Sample Name: {base_dir.name}')

    folders = get_folders(base_dir)
    dates, letters, matched_folders = process_folders(folders, args.date_string, args.verbose)

    time.sleep(2)
    all_data, fit_images, results_files = read_and_process_files(matched_folders, base_dir,args.verbose)

    # Initial cluster count
    unique_counts = len(np.unique(np.ceil(all_data['Freq'] / 100) * 100))


    print("Clustering...")
    # Perform initial clustering
    group_indices, data , group_labels  = perform_dbscan_analysis(all_data, 'Freq',args.verbose, eps=0.5, min_samples=5)



    # Adding the 'Group' column to 'data'
    data['Group'] = group_labels


    # Check cluster count and get user 
    if np.size(np.unique(group_labels)) !=18:

        decision = get_user_decision_on_cluster_count(np.size(np.unique(group_labels)),data)        
        # If user provides a new cluster count
        if decision == False:
            unique_counts = decision
            group_indices = perform_kmeans_analysis(all_data,unique_counts)
            group_labels = list(group_indices.keys())
            
    else:
        decision = True
        print('All Frequencies Identified!')

    if args.verbose:
        print(f"Decision:\t\t{decision}")
        print(f"Number of clusters\t{np.size(np.unique(group_labels))}")
        print()
        unique_labels, counts = np.unique(group_labels, return_counts=True)

        # Print formatted table header
        print(f"{'Frequency':<10} |{'Group Label':<20} | {'Count':<10}")
        print('-' * 40)

        # Iterate over each unique label and its count to print them
        for group_label, count in zip(unique_labels, counts):
            meanf = np.round(np.mean(data[data['Group'] == group_label]['Freq']))
            print(f"{meanf:<10} | {group_label:<20} | {count:<10}")
    else: 
        print(f"Number of clusters\t{np.size(np.unique(group_labels))}")
    
    
    if decision:
        all_averages = {}
        all_std_devs = {}
        all_group_data = {}
        column_names = data.columns.tolist()
        filtered_group_data = pd.DataFrame(columns = column_names)

        unique_group_labels = np.unique(group_labels)
        
        
        for group_label in unique_group_labels:
            
            # Get the relevant image paths for this group
            img_paths = image_paths_for_group(group_label, group_indices, fit_images, args.verbose)
            
            print(f"Group Label: {group_label}")
            if group_label<0: 
                group_data = data[data['Group']<0]
            else:
                group_data = data[data['Group']==group_label]
            
            print(f'Group Label: {group_label}')
            print(f'Group Data: {group_data}')
            
            
                # Perform user fit check for the current group
            fit_checks = user_fit_check(img_paths,args.verbose)
            if len(fit_checks) ==0: 
                print(f"Error No Images could be found which match group_index {group_label}") 
                raise 
            
            # Filter the data for the current group
            group_data = data[data['Group'] == group_label]
            print("group data")
            print(group_data )
            print()
            
            
            group_data = group_data.assign(FitCheck=pd.Series(fit_checks).values)
            # Process data and calculate averages and std_devs for the filtered data
            # Print the number of rows in group_data
            print(len(group_data))

            # Check if the number of rows in group_data is greater than 5
            if len(group_data) > 5:  # Corrected to use len() for clarity
                processed_data = process_data(group_data,group_label)
            else:
                processed_data = group_data

            # Assign the filtered_group_data to the specific column based on group_label
            filtered_group_data = filtered_group_data._append(processed_data, ignore_index=True)
        
        filtered_group_data = filtered_group_data.sort_values(by=["Freq"], ascending=True)
        filtered_group_data.to_csv(os.path.join(base_dir, f'Grouped_Output_{args.sample_name}_{args.date_string}.txt'), sep='\t', index=False)

        
        # Now plot the error bars using the averages and std_devs
        
        Final_Dataframe=  calculate_group_stats_and_plot(filtered_group_data,args.sample_name,args.date_string,base_dir,args.verbose,confidence_level=0.95)
        
        Final_Dataframe.to_csv(os.path.join(base_dir, f'Suspension_Summary_{args.sample_name}_{args.date_string}.txt'), sep='\t', index=False)
        
            
        if args.verbose: 
            print(Final_Dataframe)
        
    else:
            print(f"Correct Number of clusers could not be found. Exciting...")
            raise 





if __name__ == "__main__":
    main()
