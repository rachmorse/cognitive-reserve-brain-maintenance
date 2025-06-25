## Cognitive Reserve and Brain Maintenance Research Study

This repository contains the analysis scripts for our study on Cognitive Reserve (CR) and Brain Maintenance (BM), which explores their longitudinal roles in cognitive aging. This document provides an overview of the project and the available scripts.

## Repository Structure

#### How to create the CR/BM variables

- `Making CR_BM Variables.Rmd`: This script provides instructions and code for generating the CR and BM variables. It includes data requirements and guidance for generating variables with additional cohort data.

#### Analysis Scripts

This section includes several scripts used to conduct this study:

- `01_Data Cleaning.Rmd`: Cleans and merges data from the Lifebrain cohorts, removes outliers.
- `02_Calculating Annual Change.Rmd`: Calculates adjusted hippocampal volume, average functional connectivity, and annual change variables.
- `03_Descriptive Stats_CRBM Variables.Rmd`: Visualizes and conducts descriptive statistics for study variables and creates the CR/BM variables.
- `04_Analysis and Results.Rmd`: Contains all primary and supplementary analyses from the study.
- `05_calculate_avg_sn_by_group.R`: Creates a CSV with average connectivity for each ROI in the Salience Network for low CR and high CR groups across all timepoints.
- `06_sn_by_cr_group_figure.py`: Generates multi-view brain surface plots from Salience Network connectivity comparing low CR and high CR groups.

## Usage

This repository provides details on the analyses and the creation of study variables for transparency. The dataset used is not included. For access - with appropriate ethical approval - you can request the data from each Lifebrain study’s primary investigator ([information here](https://www.lifebrain.uio.no/about/lifebrain-researchers/)).