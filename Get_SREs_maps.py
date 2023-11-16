'''
A piece of code, which generates maps that quantify the SREs occurrence 
for different percentiles and durations, based on CMIP6 piControl data.

Outcomes of this script can be seen in the SREs_Global_Analysis.ipynb notebook.

Copyright (c) 2023, ETH Zurich, Guillaume Senger
'''
import my_functions
from cmip6_basic_tools.functions_raw_data import RawData, cbt
import netCDF4
import numpy as np
import os
import pickle
from tqdm import tqdm
import sys

# Choose model of interest   
model_name = 'GFDL-ESM4'

print(f'{model_name = }')

print('Download rsds_all.......')

rsds_all = my_functions.return_data_all('rsds', model_name) # Get rsds time series for all grid point for the chosen model 

print(f'{rsds_all.shape = }')
print('Download rsds_all completed')

percentiles = [5, 95] # Choose the percentiles of interest for the maps
min_length  = 3       # Choose the min SREs length we want to consider
max_length  = 10      # Choose the min SREs length we want to consider

# Get the dimensions of the maps
number_lat = len(rsds_all[0,:,0]) 
number_lon = len(rsds_all[0,0,:])

# Define an empty array to store the results
number_of_extreme_events = np.zeros((len(percentiles), number_lat, number_lon, max_length + 1))

# To keep track of the loop
count     = 0
count_tot = len(percentiles)*number_lat*number_lon

# For each grid point, perform the SREs analysis and store the number of SREs detected for each percentiles and each duration L
for lat_index in range(number_lat):
    for lon_index in range(number_lon):

        rsds = rsds_all[:, lat_index, lon_index] # Get the time series for the current grid box

        output_SREs = my_functions.compute_SREs(time_series=rsds, percentiles=percentiles, min_length=min_length, max_length=max_length) # Compute SREs for the current grid box

        for p, percentile in enumerate(percentiles): # For each percentile, get the overall occurrence for every event length
            
            extreme_events_per_day = np.sum(output_SREs[percentile]['extreme_events_per_day'], axis = 0)
            number_of_extreme_events[p, lat_index, lon_index, :] = extreme_events_per_day

        # To keep track of the loop
        count +=1
        print(f'step {count}/{count_tot}')

with open(f'Data/maps/{model_name}_TEST.pickle', 'wb') as f: pickle.dump(number_of_extreme_events, f) # Store the maps