import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import os
import time
import tempfile 
import subprocess 

from PIL import Image
import pandas as pd
from IPython.display import display, clear_output

import nds2
import pickle
import sys
from io import StringIO  
import glob 
import openpyxl 
import string
#import dcc 
#import subprocess
import matplotlib.pyplot as plt 

from tqdm.notebook import tqdm
from pylab import *
from scipy.signal import *


matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'figure.figsize': (10,6)})

def save_fig(fig_id, tight_layout=True):
    pathxx = fig_id + '.png'
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
    pcolormesh(T, F, S, norm=LogNorm(vmin=S.min(), vmax=S.max()))
    
    # get information about each sample : - add in  dcc functionallity later 

def find_matching_folders(serials, directory_path, date=None):
    # Initialize lists and dictionaries
    matching_folders = []
    date_matching_subdirectories = {}  # Dictionary to store matching subdirectories for each matching folder

    # Iterate through the items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path) and item in serials:
            matching_folders.append(item)

            # Find and match subdirectories based on the supplied date
            if date:
                subdirectories = [subdir for subdir in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, subdir))]
                matching_subdirectories = [subdir for subdir in subdirectories if subdir.startswith(date)]
                date_matching_subdirectories[item] = matching_subdirectories
                
    # Calculate non-matching folders (items in serials that were not found)
    non_matching_folders = [serial for serial in serials if serial not in matching_folders]

    # Check if all serials have been found
    all_serials_found = set(matching_folders) == set(serials)

    return matching_folders, all_serials_found, date_matching_subdirectories

def create_subdirectory_paths(matching_folders, date_matching_subdirectories, directory_path):
    # Initialize a list to store paths of matching subdirectories
    subdirectory_paths = []

    # Create paths for matching subdirectories and sort them
    for folder in matching_folders:
        item_path = os.path.join(directory_path, folder)
        matching_subdirectories = date_matching_subdirectories.get(folder, [])
        
        # Sort matching subdirectories based on the last 5 characters
        sorted_subdirectories = sorted(matching_subdirectories, key=lambda x: x[-5:])
        
        subdirectory_paths.extend([os.path.join(item_path, subdir) for subdir in sorted_subdirectories])

    return subdirectory_paths

def find_fit_images_and_results_files(subdirectory_paths):
    # Initialize dictionaries to store fit_images and results_files
    fit_images_dict = {}
    results_files_dict = {}

    # Iterate through subdirectories
    for subdirectory in subdirectory_paths:
        # Find fit_images (files matching 'fit*.png')
        fit_images = glob.glob(os.path.join(subdirectory, 'fit*.png'))
        if fit_images:
            serial = os.path.basename(os.path.dirname(subdirectory))
            fit_images_dict[serial] = fit_images

        # Find results_files (files matching 'results*.txt')
        results_files = glob.glob(os.path.join(subdirectory, 'results*.txt'))
        if results_files:
            serial = os.path.basename(os.path.dirname(subdirectory))
            results_files_dict[serial] = results_files

    return fit_images_dict, results_files_dict

def print_verbose_outputs(matching_folders, all_serials_found, date_matching_subdirectories, subdirectory_paths, fit_images_dict, results_files_dict):
    print("Matching Folders: {}".format(matching_folders))
    print("All Serials Found: {}".format(all_serials_found))
    print("Date Matching Subdirectories:")
    for folder, subdirectories in date_matching_subdirectories.items():
        print("{}: {}".format(folder, subdirectories))
    print("Sorted Subdirectory Paths:")
    for path in subdirectory_paths:
        print("{}".format(path))
    print("Fit Images Dictionary:")
    for serial, fit_images in fit_images_dict.items():
        print("Serial {}: {}".format(serial, fit_images))
    print("Results Files Dictionary:")
    for serial, results_files in results_files_dict.items():
        print("Serial {}: {}".format(serial, results_files))


def print_summary_outputs(matching_folders, all_serials_found, date_matching_subdirectories, subdirectory_paths, fit_images_dict, results_files_dict):
    print("Number of Matching Folders: {}".format(len(matching_folders)))
    print("All Serials Found: {}".format(all_serials_found))
    print()
    
    for folder, subdirectories in date_matching_subdirectories.items():
        print("{}: {} measurements".format(folder, len(subdirectories)))
    
    print()
    print("Number of Results Files Dictionary Entries: {}".format(len(results_files_dict)))

    
    
def prompt_user_to_keep_fit(image_path):
    with Image.open(image_path) as img:
        subprocess.Popen(['feh',image_path])
        
        #with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            #img.save(f.name)
            #print(f.name)
            # Use subprocess to open the image with the default viewer
            #subprocess.Popen(['feh' '"' , f.name, '"'])

            
    # Prompt the user for input
    while True:
        user_input = input("Press Enter to keep this fit, otherwise press 'n': ").strip().lower()
        if user_input == "":
            return True
        elif user_input == "n":
            return False

def process_fit_images(serial, fit_images):
    fit_responses = []

    # Iterate through fit images
    for image_path in fit_images:
        # Prompt user to keep the fit
        keep_fit = prompt_user_to_keep_fit(image_path)

        # Store the result
        fit_responses.append({
            'Serial': serial,
            'Image Filename': os.path.basename(image_path),
            'Keep Fit': keep_fit
        })
    return fit_responses


    
    

def main(serials, directory_path, date=None, verbose=True):
    """
    Main function to find matching folders, subdirectories, fit images, and results files
    within a specified directory.

    Parameters:
    - serials (list):             List of sample serial numbers to match against folder names.
    - directory_path (str):       Path to the directory where matching folders and subdirectories are located.
    - date (str, optional):       Date string to match subdirectories (default is None). should have format YYYY_MM_DD i.e. 2024_01_12
    - verbose (bool, optional):   If True, print detailed outputs; if False, print summarized outputs (default is True).

    Returns:
    - None: Outputs are printed based on the 'verbose' parameter.
    """
    
    
    matching_folders, all_serials_found, date_matching_subdirectories = find_matching_folders(serials, directory_path, date)
    subdirectory_paths = create_subdirectory_paths(matching_folders, date_matching_subdirectories, directory_path)
    fit_images_dict, results_files_dict = find_fit_images_and_results_files(subdirectory_paths)

    if verbose:
        print_verbose_outputs(matching_folders, all_serials_found, date_matching_subdirectories, subdirectory_paths, fit_images_dict, results_files_dict)
    else:
        print_summary_outputs(matching_folders, all_serials_found, date_matching_subdirectories, subdirectory_paths, fit_images_dict, results_files_dict)
        
     # Process fit images for each serial
    fit_responses_list = []
    
    for serial in serials:
        if serial in fit_images_dict:
            fit_responses = process_fit_images(serial, fit_images_dict[serial])
            fit_responses_list.append(fit_responses)

    # Concatenate the results into a single DataFrame
    fit_responses_df = pd.concat(fit_responses_list, ignore_index=True)

    # Print or return the DataFrame
    if verbose:
        print("Fit Responses:")
        display(fit_responses_df)
    else:
        return fit_responses_df
        
        
        
        
serials = ['S1600962']#, 'S1600963', 'S1600964', 'S1600965']
date = '2024_01_22'  # Replace with the date you want to match
directory_path =   "/mnt/data/Notebooks/CRIME/results" # Replace with the actual directory path
verbose = False   # Set to True for detailed outputs or False for summarized outputs

main(serials, directory_path, date, verbose)
