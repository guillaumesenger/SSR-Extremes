'''
A piece of code, designed to store CMIP6 piControl time series data in a dictionnary for a selected grid box and models."

Copyright (c) 2023, ETH Zurich, Guillaume Senger
'''
import my_functions
import numpy as np
import pickle

variable_name = 'rsds'   # Choose CMIP6 variable of interest 
location      = 'Zurich' # Choose location of interest   

print(f'Loading {variable_name} variable for {location}')

# Model names from which we want to load the data
model_names    = [ 
    'CanESM5',
    'CESM2',
    'CMCC-CM2-SR5',
    'CMCC-ESM2',
    'GFDL-CM4',
    'GFDL-ESM4',
    'NorESM2-LM',
    'NorESM2-MM',
    'TaiESM1',
    # 'INM-CM4-8',
    # 'INM-CM5-0',
    # 'IPSL-CM5A2-INCA',
    # 'KIOST-ESM'
]

output = {} # Initialize dictionary to store output

for m, model_name in enumerate(model_names):

    time_series = my_functions.get_data(variable_name, model_name, location, output = 'return') # Get time series data 

    output[model_name] = time_series # Store time series data in the output dictionary

with open(f'Data/' + variable_name + '/' + location  + '.pickle', 'wb') as f: pickle.dump(output, f) # Save output dictionary to a pickle file