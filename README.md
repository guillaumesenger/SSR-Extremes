# SSR-Extremes

Guillaume Senger, ETH Zürich, guillaume.senger@usys.ethz.ch

## Overview
This repository contains scripts and notebooks related to the analysis of Surface Solar Radiation Extreme (SSR) events.

## Usage

Each notebook corresponds to a specific analysis and uses functions from the `my_functions` folder. All the data can be generated using functions from the `my_functions` folder and the following scripts: `Get_Data_Local.py`, `Get_PV_Power.py`, and `Get_SREs_maps.py`.

## Data

Data from the Coupled Model Intercomparison Project - Phase 6 were used in the manuscript: [Doi: 10.5194/gmd-9-1937-2016](https://doi.org/10.5194/gmd-9-1937-2016)

The observational data can be obtained from the World Radiation Data Center (WRDC): [Doi: 10.17616/R3N30Q ](http://doi.org/10.17616/R3N30Q)

The estimated PV power production is calculated using GSEE: Global Solar Energy Estimator: [Doi: 10.1016/j.energy.2016.08.060](https://doi.org/10.1016/j.energy.2016.08.060)

## Requirements

The Python package versions utilized are as follows:

`numpy`      version 1.22.3

`matplotlib` version 3.5.1

`netcdf4`    version 1.5.7

`pandas`     version 1.4.1

`xarray`     version 2022.3.0

`scipy`      version 1.8.0

`cartopy`    version 0.20.0

`seaborn`    version 0.11.2

For all requirements related to GSEE, visit: [gsee.readthedocs.io/](https://gsee.readthedocs.io/en/latest/)

Regarding `cmip6_basic_tools` package, see : https://gitlab.com/mjade/cmip6_basic_tools
    
## Scripts

### `my_functions.py`
Folder containing custom functions used across multiple notebooks and scripts.

### `my_functions_examples.ipynb`
Notebook demonstrating the usage and functionality of functions in the `my_functions` folder. 
The `SSR_GFDL_ESM4.pickle` is used as an example time series to illustrate the `Compute_SREs` function.

### `Get_Data_Local.py`
Script loading CMIP6 time series for a grid box containing a specific location.

### `Get_PV_Power.py`
Script that computes the daily PV power generation of a specific PV installation from a daily mean SSR time series.

### `Get_SREs_maps.py`
Script that generates maps that quantify the SREs occurrence for different SREs parameters based on CMIP6 piControl data.

## Notebooks

### `SSR_Overview.ipynb`
Notebook providing an overview of SSR statistics for different data sets.

### `SREs_Local_Analysis.ipynb`
Notebook analyzing Sustained Radiation Events (SREs) for a particular grid box.

### `SREs_Global_Analysis.ipynb`
Notebook analyzing SREs on a global scale.

### `SREs_PV_Power_Analysis.ipynb`
Notebook analyzing the relationship between SSR and PV power during SREs.

### `SREs_SLP_Analysis.ipynb`
Notebook analyzing sea-level pressure in relation to SREs.

### `PV_extreme_days.ipynb`
Notebook focusing on PV power production during extreme SSR days.

### `CREs_Local_Analysis.ipynb`
Notebook analyzing Cumulative Radiation Events (CREs) for a particular grid box.
