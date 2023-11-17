# SSR-Extremes

## Overview
This repository contains scripts and notebooks related to the analysis of Surface Solar Radiation Extreme (SSR) extreme events.

## Usage

Each notebook corresponds to a specific analysis and uses functions from the `my_functions` folder. All the data can be generated using functions from the `my_functions` folder and the following scripts: `Get_Data_Local.py`, `Get_PV_Power.py`, and `Get_SREs_maps.py`.

## Requirements


## Scripts

### `my_functions`
Folder containing custom functions used across multiple notebooks and scripts.

### `my_functions_examples.ipynb`
Notebook demonstrating the usage and functionality of functions in the `my_functions` folder. 
The `SSR_GFDL_ESM4.pickle` is used as an example time series to illustrate the `def Compute_SREs():` function.

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

## Contact
Guillaume Senger, ETH Zürich, guillaume.senger@usys.ethz.ch

