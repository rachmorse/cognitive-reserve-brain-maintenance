## Cognitive Reserve and Brain Maintenance Research Study

This repository contains the analysis scripts for our study on Cognitive Reserve (CR) and Brain Maintenance (BM), which explores their longitudinal roles in cognitive aging. This document provides an overview of the project and the available scripts.

## Repository Structure

#### How to create the CR/BM variables

- `making_crbm_variables.Rmd`: This script provides code for generating the CR and BM variables, including code to simulate example data and visualize the resulting measures. If you are interested in alternative formulations of the variables, please see `sensitivity_analyses.py` below. 

#### Analysis Scripts

This section includes several scripts used to conduct this study:

- `data_cleaning.Rmd`: Cleans and merges data from the Lifebrain cohorts, removes outliers.
- `calculating_annual_change.Rmd`: Calculates adjusted hippocampal volume, average functional connectivity, and annual change variables. Also, takes the residuals of GAMs with hippocampal and memory change and age. 
- `crbm_vars.py`: Creates the CR/BM variables and visualizes the moderation effect. 
- `analysis_and_results.Rmd`: Contains primary and supplementary analyses from the study.
- `sensitivity_analyses.py`: Generates CR and BM measure variations (options: for BM change scaling method, distance calculation method, distance weighting, from theoretical to empirial method; for CR change k, θ). Compares the variations to original measures, and plots each variation.

## Usage

This repository provides details on the analyses and the creation of study variables for transparency. The dataset used is not included. For access - with appropriate ethical approval - you can request the data from each Lifebrain study’s primary investigator ([information here](https://www.lifebrain.uio.no/about/lifebrain-researchers/)).