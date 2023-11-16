'''
A piece of code, which computes the daily PV power generation at a given location.
This computation is based on daily mean SSR time series extracted from a CMIP6 piControl run from a specific model.

Copyright (c) 2023, ETH Zurich, Guillaume Senger
'''
import my_functions
import numpy as np
from  gsee import climatedata_interface
import picklexw
from tqdm import tqdm
import xarray as xr
import pandas as pd

location      = 'Zurich'    # Choose location of interest   
model_name    = 'GFDL-ESM4' # Choose model of interest

print(f'Loading rsds data from {model_name} model for the grid box containing {location}....')

rsds_models   = pickle.load(open(f'Data/rsds/{location}.pickle', 'rb')) # Load data created using the script Get_Data_Local.py
rsds          = rsds_models[model_name] # Get only data for the chosen model
reshaped_rsds = my_functions.reshape(rsds)
n_years       = len(rsds)//365

print('Loading and reshaping completed')

# Set panel technical parameters for PV system
tilt = 55    # Tilt angle in degrees
azim = 180   # Azimuth angle in degrees
tracking = 0 # Tracking configuration (0: no tracking, 1: 1-axis tracking, 2: 2-axis tracking)


# Display the panel configuration details
print(f'Panel configuration: \n     location: {location}\n     tilt: {tilt} degree \n     azim: {azim} degree \n     tracking: {tracking}')

pv_power = np.zeros_like(rsds) # Initialize array to store PV power

for year in tqdm(np.arange(n_years), total=n_years, desc='Processing years'):

    data = reshaped_rsds[year,:] # Get data for the current year

    pv_power[year*365:(year+1)*365] = my_functions.get_PV_one_year(data, location, tilt, azim, tracking)

with open(f'Data/PV/{model_name}_{location}_tilt_{tilt}_azim_{azim}_tracking_{tracking}.pickle', 'wb') as f: pickle.dump(pv_power, f)