'''
This script contains all functions used in this repository. Please see the my_functions_examples.ipynb notebook for examples of use.

Copyright (c) 2023, ETH Zurich, Guillaume Senger
'''
import numpy as np
from geopy.geocoders import Nominatim 
from datetime import datetime
import re
from cmip6_basic_tools.functions_raw_data import RawData, cbt
import netCDF4
import pickle
import gsee
import xarray as xr
import pandas as pd

def get_lat_lon(location):
    """
    Function that returns the latitude and longitude of a given location.
    
    Parameters:
        location (str): The location to geocode.
        
    Returns:
        tuple: A tuple containing the latitude and longitude of the geocoded location.
        
    Raises:
        ValueError: If the location cannot be geocoded.
    """
    geolocator = Nominatim(user_agent="MyApp")
    geocoded_location = geolocator.geocode(location)

    if geocoded_location is None:
        raise ValueError(f"Could not geocode location: {location}")

    lat_location = geocoded_location.latitude  # type: ignore
    lon_location = geocoded_location.longitude # type: ignore

    return lat_location, lon_location


def get_indices_lat_lon(location, lat, lon):
    """
    Function that returns the closest indices of the latitude and longitude corresponding to a given location.
    
    Parameters:
        location (str): The location to geocode.
        lat (array-like): Array or list of latitude values.
        lon (array-like): Array or list of longitude values.
        
    Returns:
        tuple: A tuple containing the indices of the latitude and longitude corresponding to the geocoded location.
        
    Raises:
        ValueError: If the location cannot be geocoded.
    """
    
    lat_location, lon_location = get_lat_lon(location)

    if lon_location < 0:
        lon_location = 360 + lon_location

    index_lon_location = np.absolute(lon[:] - lon_location).argmin()
    index_lat_location = np.absolute(lat[:] - lat_location).argmin()

    return index_lat_location, index_lon_location


def get_day_from_date(date):
    """
    Function that returns the day number given a date in the format 'dd-mm'.
    
    Parameters:
        date (str): The date string in the format 'dd-mm'.
        
    Returns:
        int: The day number corresponding to the given date.
        
    Raises:
        ValueError: If the input is not a string in the format 'dd-mm'.
    """
    if not isinstance(date, str) or not re.match(r"\d{2}-\d{2}$", date):
        raise ValueError("The input is not a string in the format 'dd-mm'")
        
    date_object = datetime.strptime(date, "%d-%m")
    day_of_year = date_object.timetuple().tm_yday - 1
    
    return day_of_year


def reshape(array_to_reshape):
    """
    Function that reshapes a time series into a 365 x n_year array.
    
    Parameters:
        array_to_reshape (array-like): The input array to reshape.
        
    Returns:
        array-like: The reshaped array of shape (n_year, 365).
        
    Raises:
        ValueError: If the length of the array is not a multiple of 365.
    """
    if len(array_to_reshape) % 365 != 0:
        raise ValueError("The array length is not a multiple of 365.")
    
    n_years = len(array_to_reshape) // 365
    reshaped_array = np.reshape(array_to_reshape, (n_years, 365))
    
    return reshaped_array


def count_and_replace(arr):
    """
    Function that counts the number of consecutive ones in an array and give the total at the position of the first consecutive one with zeros for the others.
    
    Parameters:
        arr (array-like): The input array.
        
    Returns:
        array-like: The array with the counts of consecutive ones
    """
    arr = arr.astype(int)
    
    # Count consecutive ones
    result = np.zeros_like(arr)
    
    for i in range(len(arr)):
        if arr[i] == 1:
            count = 1
            j = i + 1
            while j < len(arr) and arr[j] == 1:
                count += 1
                j += 1
            result[i] = count

    # Get rid of ones that have been counted
    i = 0
    while i < len(arr):
        if arr[i] == 1 and result[i] > 0:
            result[i+1:i+result[i]] = 0
            i += result[i]
        else:
            i += 1

    return result


def count_occurrences(arr, min_length = 3, max_length = 16):
    """
    Function that counts the occurrences of values within a given range in an array.
    
    Parameters:
        arr (array-like): The input array.
        min_length (int): The minimum value to consider for counting occurrences (default: 3).
        max_length (int): The maximum value to consider for counting occurrences (default: 16).
        
    Returns:
        array-like: An array where each value is the occurence of the index in the input array.
    """
    arr    = arr.astype(int)
    result = np.zeros((max_length + 1))
    
    for val in arr:   
        if min_length <= val <= max_length:
            result[val] += 1

    # To remove false zeros := days that are above the threshold but set to zeros by count_and_replace
    if min_length == 0:
        for i in range(2, max_length + 1):
            result[0] = result[0] - result[i] * (i - 1)

    return result


def compute_group_means_and_replace(arr):
    """
    Function that compute the mean of each group of non-zero elements, replacing each group with its mean at the first position and zeros after. Zero elements are preserved in the resulting array.

    Parameters:
        arr (array_like): The input array.

    Returns:
        array_like: The array with the group means
    """

    result = []
    current_sum = 0
    count = 0

    for num in arr:
        if num != 0:
            current_sum += num
            count += 1
        else:
            if count > 0:
                mean = current_sum / count
                result.extend([mean] + [0] * (count - 1))
                current_sum = 0
                count = 0
            result.append(0)

    if count > 0:
        mean = current_sum / count
        result.extend([mean] + [0] * (count - 1))

    return np.transpose(result)


def get_seasons_masks(n_years):
    """
    Generate n_years long masks for the seasons: 
        Spring: March, April and May 
        Summer: June, July and August 
        Autumn: September, October and November 
        Winter: December, January and February

    Parameters:
        n_years (int): The number of years for which the seasonal masks are to be generated.

    Returns:
        tuple: A tuple containing a list of season names and a dictionary of corresponding masks for each season.
    """

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn'] 

    # Create a dictionary to store the masks for each season
    masks = {
        'Winter': np.zeros(365, dtype=int),
        'Spring': np.zeros(365, dtype=int),
        'Summer': np.zeros(365, dtype=int),
        'Autumn': np.zeros(365, dtype=int)
    }

    # Set the appropriate values in each mask to indicate the duration of each season
    masks['Winter'][0:59]    = 1  # Winter: January + February
    masks['Winter'][334:365] = 1  # Winter: December
    masks['Spring'][59:152]  = 1  # Spring: March to May
    masks['Summer'][152:244] = 1  # Summer: June to August
    masks['Autumn'][244:334] = 1  # Autumn: September to November

    # Repeat the masks n_years times to match the length of the time series
    masks = {season: np.tile(mask, n_years) for season, mask in masks.items()}

    return seasons, masks


def compute_SREs(time_series, percentiles = 'all', min_length = 3, max_length = 16):
    """
    Compute Sustained Radiation Events (SREs) based on the given time series.

    Parameters:
        time_series (array-like): The input time series.
        percentiles (str or array-like): The percentiles to consider. If 'all', considers all percentiles from 1 to 99.
        min_length (int): The minimum SREs lengths considered (default: 3).
        max_length (int): The maximum SREs lengths considered (default: 16).

    Returns:
        dict: A dictionary containing computed SREs and related information.
    """
    print('SUSTAINED RADIATION EVENTS')
    print('--------------------------')
    
    output = {} # Create an empty dictionnary for the outputs
    
    rsds                 = time_series
    reshaped_rsds        = reshape(rsds)
    n_years              = reshaped_rsds.shape[0]

    if len(rsds) % 365 != 0: 
        raise ValueError(f"The data should be an integer multiple of 365.")

    print(f'Time series duration: {n_years} years ({len(time_series)} days)\n')

    if percentiles == 'all':

        percentiles = np.arange(1, 100, 1)
        print('Percentiles considered: 1th to 99th')
         
    else: print(f'Percentiles considered: {percentiles}')

    print('\nSREs lengths considered:')
    print(f'      mininum  {min_length} consecutive days')
    print(f'      maxinum {max_length} consecutive days')

    output['n_years']     = n_years
    output['percentiles'] = percentiles
    output['min_length']  = min_length
    output['max_length']  = max_length  
    
    for i, percentile in enumerate(percentiles):

        rsds_all_years_above   = np.zeros_like(rsds)
        extreme_events_per_day = np.zeros((365, max_length + 1))

        thresholds_one_year    = np.nanpercentile(reshaped_rsds, percentile, axis=0) # type: ignore # Compute the daily threshold
        thresholds_all_years   = np.tile(thresholds_one_year, n_years) # Create an array that repeat the daily threshold for then entire time series length

        if percentile > 50: # type: ignore # Create rsds_all_years_above = 1 if rsds > threshold, 0 otherwise
            rsds_all_years_above[rsds > thresholds_all_years] = 1
        else:
            rsds_all_years_above[rsds < thresholds_all_years] = 1

        SREs_all_years = count_and_replace(rsds_all_years_above)
        SREs_all_years[SREs_all_years < 3] = 0 # Get rid if less than 3 consecutive days

        for day in range(365): # Store SREs depending on their calendar starting day 

            indices_one_day = [day + i*365 for i in range(n_years)]
            radiation_wave_for_one_day = SREs_all_years[indices_one_day]
            
            extreme_events_per_day[day, :] = count_occurrences(radiation_wave_for_one_day, min_length, max_length)

        output[percentile] = {}  # Initialize the dictionary for the specific percentile
        output[percentile]['thresholds_all_years'] = thresholds_all_years
        output[percentile]['rsds_all_years_above'] = rsds_all_years_above
        output[percentile]['SREs_all_years']       = SREs_all_years
        output[percentile]['extreme_events_per_day'] = extreme_events_per_day 

        if i == 0:
            print('\nFor each percentile, available outputs are:')
            print(f"          - thresholds_all_years {output[percentile]['thresholds_all_years'].shape}: give the daily thresholds for all simulation days")
            print(f"          - rsds_all_years_above {output[percentile]['rsds_all_years_above'].shape}: binary array with 1: exceed daily threshold, 0: do not exceed daily threshold")
            print(f"          - SREs_all_years {output[percentile]['SREs_all_years'].shape}: give the SRE length at the first day of all SREs, 0 otherwise")
            print(f"          - extreme_events_per_day {output[percentile]['extreme_events_per_day'].shape}: give how many events of length L are observed for each calendar day")

    print("\nSREs computed successfully.")

    return output


def get_data(variable_name, model_name, location):
    """
    Load the time series form a chosen climate model for the grid box containing a given location.

    Parameters:
        variable_name (str): the variable we want to extract (e.g., 'rsds', 'psl',...).
        model_name (str): the CMIP6 model from which we want to extract the time series
        location (str): the output will be for the grid box containing this location
    """

    working_dir      = '/net/o3/echam/bchtirkova/tests/' 
    model_type       = 'day'
    model_generation = 'cmip6'
    experiment       = 'piControl'

    var_pi          = RawData(working_dir = working_dir, variable_name = variable_name, experiment = experiment, model_type = model_type, model_generation = model_generation)
    model_ensembles = var_pi.get_ensemble_members(model_name)  
    file_list       = var_pi.get_netcdf_file_list(model_name, model_ensembles[0])
  
    print(f'Loading {variable_name} data from {model_name} (grid box containing {location}):')
  

    try: # Method 1 : All files together

        print('     Attempt with Method 1 (MFDataset)...')
        file = netCDF4.MFDataset(file_list) # type: ignore

        lon  = file.variables['lon'][:]
        lat  = file.variables['lat'][:]

        index_lat_location, index_lon_location = get_indices_lat_lon(location, lat, lon)

        time_series  = file.variables[variable_name][:, index_lat_location, index_lon_location]

        print('     Method 1 suceeded')


    except ValueError:


        try: # Method 2 : One file at the time
            print('     Method 1 failed')
            print('     Attempt with Method 2 (Dataset)...')

            time_series = []  # Initialize an empty list to store all the rsds_one_file arrays

            for i, file_name in enumerate(file_list):
                
                file = netCDF4.Dataset(file_name) # type: ignore

                if i == 0: # Load lon and lat only for the first array

                    lon  = file.variables['lon'][:]
                    lat  = file.variables['lat'][:]
                                            
                    index_lat_location, index_lon_location = get_indices_lat_lon(location, lat, lon)
        
                time_series_one_file  = file.variables[variable_name][:, index_lat_location, index_lon_location]
                time_series.append(time_series_one_file) 

            time_series = np.concatenate(time_series)

            print('     Method 2 suceeded')

        except:

            print(f'Problem with {model_name}: None of the two methods suceeded... Pass')

            return False

    
    print('     Checking format...') # Check if the time series is a multiple of 365

    if len(time_series) % 365 == 0:

        print('     Correct format')

        print(f'Loading completed')

        if output == 'load':
            with open(f'Data/' + variable_name + '/' + location + '_' +  model_name + '.pickle', 'wb') as f: pickle.dump(time_series, f)
            return True
    else:

        print(f'Incorrect format... Pass')
        return False


def return_data_all(variable_name, model_name):
    """ 
    Return the time series form a chosen climate model for all grid boxes.

    Parameters:
        variable_name (str): the variable we want to extract (e.g., 'rsds', 'psl',...).
        model_name (str): the CMIP6 model from which we want to extract the time series

    Return:
        array_like: Time series data for all grid boxes if successful, False otherwise.
    """

    
    working_dir      = '/net/o3/echam/bchtirkova/tests/' 
    model_type       = 'day'
    model_generation = 'cmip6'
    experiment       = 'piControl'

    var_pi          = RawData(working_dir = working_dir, variable_name = variable_name, experiment = experiment, model_type = model_type, model_generation = model_generation)
    model_ensembles = var_pi.get_ensemble_members(model_name)  
    file_list       = var_pi.get_netcdf_file_list(model_name, model_ensembles[0])

    print(f'Returning {variable_name} data from {model_name} (all grid boxes):')

    try: # Method 1 : All files together


        print('     Attempt with Method 1 (MFDataset)...')
        file = netCDF4.MFDataset(file_list) # type: ignore

        time_series  = file.variables[variable_name][:]

        print('     Method 1 suceeded')

    except ValueError:

        try: # Method 2 : One file at the time

            print('     Method 1 failed')
            print('     Attempt with Method 2 (Dataset)...')
            time_series = None  # Initialize an empty list to store all the rsds_one_file arrays

            for i, file_name in enumerate(file_list):
                
                file = netCDF4.Dataset(file_name) # type: ignore
                     
                time_series_one_file  = file.variables[variable_name][:]


                print(f'{time_series_one_file.shape =}')

                if time_series is None:
                    
                    time_series = time_series_one_file
                else:
        
                    time_series = np.concatenate((time_series, time_series_one_file), axis=0)

                    print(f'{time_series.shape =}')

            print('     Method 2 suceeded')

        except:

            print(f'Problem with {model_name}: None of the two methods suceeded... Pass')

            return False

    
    print('     Checking format...') # Check if the time series is a multiple of 365

    if len(time_series[:,0,0]) % 365 == 0:

        print('     Correct format')

        return time_series
    
    else:

        print(f'Incorrect format... Pass')

        return False


def get_mesh(model_name):
    """
    Retrieve longitude and latitude mesh information for a specified climate model's data.

    Parameters:
        model_name (str): The name of the climate model.

    Note:
        The function retrieves mesh information (longitude and latitude) from netCDF files associated with the model.
        It stores this information in a dictionary and saves it into a pickle file named 'mesh_{model_name}.pickle'.
    """
    
    working_dir      = '/net/o3/echam/bchtirkova/tests/' 
    model_type       = 'day'
    model_generation = 'cmip6'
    variable_name    = 'rsds'
    experiment       = 'piControl' 

    var_pi          = RawData(working_dir = working_dir, variable_name = variable_name, experiment = experiment, model_type = model_type, model_generation = model_generation)
    model_ensembles = var_pi.get_ensemble_members(model_name)  
    file_list       = var_pi.get_netcdf_file_list(model_name, model_ensembles[0])

    try: # Method 1 : All files together

        file = netCDF4.MFDataset(file_list) # type: ignore

        lon  = file.variables['lon'][:]
        lat  = file.variables['lat'][:]

    except:

        # Method 2 : One file at the time
        file = netCDF4.Dataset(file_list[0]) # type: ignore

        lon  = file.variables['lon'][:]
        lat  = file.variables['lat'][:]

    mesh_dict = {'lon': lon[:], 'lat': lat[:]} #  Dictionary containing mesh information

    with open(f'Data/mesh_' +  model_name + '.pickle', 'wb') as f: pickle.dump(mesh_dict, f)



def get_PV_one_year(data, location, tilt=35, azim=180, tracking=0):
    """
    Calculate PV power generation for a given location and dataset for one year.

    Parameters:
        data (array-like): daily mean SSR data for 365 days.
        location (str): The location for which PV power is calculated.
        tilt (float, optional): Tilt angle of the PV panels (default: 35).
        azim (float, optional): Azimuth angle of the PV panels (default: 180).
        tracking (int, optional): Tracking configuration of the PV panels (default: 0) with 0: fixe, 1: 1-axis tracking and 2: 2-axis tracking.

    Returns:
        numpy.ndarray: Array containing the PV power generation for each day of the year in Wh/day.

    Raises:
        ValueError: If the data provided is not of length 365.

    """

    if len(data) != 365:
        raise ValueError("The data should be of length 365.")

    # Get latitude and longitude for the provided location
    lat_location = get_lat_lon(location)[0] 
    lon_location = get_lat_lon(location)[1] 

    PV_power = gsee.climatedata_interface.interface.run_interface_from_dataset( # type: ignore
            data = xr.Dataset(
                    data_vars={"global_horizontal": (("time", "lat", "lon"), np.reshape([data], (365, 1,1)))},
                    coords={
                        "time": pd.date_range(
                            start=f'1850-01-01', periods=365, freq='D'
                        ), 
                        "lat": [lat_location], 
                        "lon": [lon_location]}),  
            params = {
                "tilt": tilt, 
                "azim": azim, 
                "tracking": tracking, 
                "capacity": 1000, 
                "use_inverter": False, 
                "technology":"csi"}, 
            pdfs_file = None
            )
    
    return PV_power['pv'][:,0,0]
