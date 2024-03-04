import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from datetime import datetime

def round_nearest(x, base=50):
    return base * round(x/base)

def nearest_point(main_freq, compare_freq):
    tree = cKDTree(compare_freq[:, None])
    distances, indices = tree.query(main_freq[:, None], k=1)
    return indices

def plot_best_measurements(sample_name,prefix):
    sampleData = {}
    FieldName = []

    # Loading and preprocessing data
    for file in glob.glob('*Summary*.txt'):
        tmp = pd.read_csv(file, sep='\t')  # Adjust sep according to your file format
        tmp = tmp[tmp.Frequency != 0]
        field_name = 'x' + file.split('Suspension_Summary_{sample_name}_')[1].split('.txt')[0]
        FieldName.append(field_name)
        sampleData[field_name] = tmp

    # Converting and adjusting data
    for field_name in FieldName:
        X = sampleData[field_name].to_numpy()
        X[:, 1:] = 1.0 / X[:, 1:]
        sampleData[field_name] = pd.DataFrame(X, columns=['Frequency', 'phi_1Mean', 'phi_2Mean', 'phi_1Max', 'phi_2Max', 'phi_1Min', 'phi_2Min'])

    # Extracting dates and clustering
    dates = np.array([datetime.strptime(name[1:], '%Y_%m_%d') for name in FieldName])
    dates.sort()
    clusterIDs = np.zeros(len(dates), dtype=int)
    currentClusterID = 0

    for i, current_date in enumerate(dates):
        if clusterIDs[i] == 0:
            currentClusterID += 1
            clusterIDs[i] = currentClusterID
            for j, compare_date in enumerate(dates[i+1:], start=i+1):
                if abs((current_date - compare_date).days) <= 5 and clusterIDs[j] == 0:
                    clusterIDs[j] = currentClusterID

    # Plotting setup
    plt.figure()
    plt.yscale('log')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Mechanical Loss phi_{mech}')
    plt.grid(True, which="both", ls="--")

    # Processing clusters
    for cid in np.unique(clusterIDs):
        print(f'Cluster ID: {cid}')
        relevant_fields = [FieldName[i] for i, cluster_id in enumerate(clusterIDs) if cluster_id == cid]
        print(relevant_fields)
        
        if len(relevant_fields) < 2:
            continue  # Skip if less than 2 tables for this cluster
        
        table1 = sampleData[relevant_fields[0]]
        table2 = sampleData[relevant_fields[1]]
        
        table1 = table1.sort_values(by='Frequency')
        table2 = table2.sort_values(by='Frequency')

        A = table1['Frequency'].apply(round_nearest).to_numpy()
        B = table2['Frequency'].apply(round_nearest).to_numpy()
        
        if len(table1) != len(table2):
            maxidx = 1 if len(table1) > len(table2) else 2
            
            if maxidx == 1:
                matching_idx = nearest_point(A, B)
                table2 = table2.iloc[matching_idx].reset_index(drop=True)
            else:  # maxidx == 2
                matching_idx = nearest_point(B, A)
                table1 = table1.iloc[matching_idx].reset_index(drop=True)
                
   

plot_best_measurements(sample_name='S1600962',prefix=r'/Users/simon/Desktop/CRIME_BACKUP/S1600962')
