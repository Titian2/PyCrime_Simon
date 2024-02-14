import os
import re
from collections import Counter, defaultdict
import subprocess 

SampleName = 'S1600966'

# Define the base directory
base_directory = os.path.join('/Users/Simon/Desktop/CRIME_BACKUP/',SampleName)

# List all subdirectories (which are assumed to be date directories)
date_directories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Initialize lists for dates and letters
dates = []
letters = []

# Extract dates and letters using regular expression
date_pattern = r'\d{4}_\d{2}_\d{2}'
letter_pattern = r'([a-zA-Z]+)_CR'

for directory in date_directories:
    date_match = re.search(date_pattern, directory)
    letter_match = re.search(letter_pattern, directory)
    
    if date_match and letter_match:
        dates.append(date_match.group())
        letters.append(letter_match.group(1))

# Count occurrences of each unique date
date_counts = Counter(dates)
unique_dates = date_counts.keys()

# Create a dictionary to group date_directories by unique date
date_directory_dict = defaultdict(list)

for date, directory in zip(dates, date_directories):
    date_directory_dict[date].append(directory)

# Print the results


#for date, directories in date_directory_dict.items():
#        print(f'Date: {date}')

    
print(f'\nSample Name:\t{SampleName}')


verbose = '' 
for date in (list(unique_dates))[2:]:
    print(f"Analysis Date:\t{date:<18}\n")
    
    command = f"python Fit_Checking_2.0.py {SampleName} {date} {verbose}"
    #print(f"{date:<18} {date_counts[date]}")
    subprocess.call(command, shell=True)

    
    
    
    
    
