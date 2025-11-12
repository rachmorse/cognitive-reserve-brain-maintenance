## Cognitive Reserve and Brain Maintenance Research Study

This repository contains the analysis scripts for our study on Cognitive Reserve (CR) and Brain Maintenance (BM), which explores their longitudinal quantification and roles in cognitive aging. This document provides an overview of the project and the available scripts.

## Repository Structure

#### How to create the CR/BM variables

- `making_crbm_variables.py`: This script provides code for generating the CR and BM variables, including code to simulate example data and visualize the resulting measures. If you are interested in alternative formulations of the variables, please see `sensitivity_analyses.py` below. 

#### Analysis Scripts

This section includes several scripts used to conduct this study:

- `data_cleaning.Rmd`: Cleans and merges data from the Lifebrain cohorts.
- `calculating_annual_change.Rmd`: Calculates adjusted hippocampal volume, average functional connectivity, and annual change variables. Also, takes the residuals of age regressed on hippocampal and memory change. 
- `crbm_measures.py`: Creates the CR and BM measures and visualizes the CR moderation effect. 
- `analysis_and_results.Rmd`: Contains primary and supplementary analyses from the study.
- `sensitivity_analyses.py`: Generates CR and BM measure variations. Compares the variations to original measures and plots each variation.

## Usage

This repository provides details on the analyses for transparency. The dataset used is not included. For access - with appropriate ethical approval - you can request the data from each Lifebrain study’s primary investigator ([more information here](https://www.lifebrain.uio.no/about/lifebrain-researchers/)).

## License

The code is available under the [MIT License](https://opensource.org/licenses/MIT), allowing others to reuse and adapt it with appropriate credit.