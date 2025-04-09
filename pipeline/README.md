# Here is a step by step process to run the ALEXIS pipeline. 
The processing pipeline has been split up into different componenets that must be run in order. 
This way, you can run the same pipeline in chuncks, making it easier to debug and to communicate with the team for trouble shooting. 
## Please feel free to submit an issue. 


# ALEXIS by Parts:
There are 4 main scripts that make up the ALEXIS pipeline. The four scripts are:
### 1. ALEXIS_download_data.py
### 2. ALEXIS_02_create_initialize_files.py
### 3. ALEXIS_03_create_wd_w_init_files_before_hyperspectral.py
### 4. create_peakfinder_scale_02.py


# 1. Download data:
## Features
- Downloads solar flare data based on user-specified flare classes and timestamps.
- Queries data availability from AIA and SXI databases.
- Organizes downloaded data into a structured directory format.
- Ensures data integrity using hashed URLs and caching mechanisms.
- Supports parallel processing for faster downloads using the Ruffus pipeline.

The script will:

Create working directories for each flare in the DATA_DIR_FLARE_CANDIDATES directory (defined in dataconfig.py).
Download and save data files in the DATA_DIR_IMG_RUFFUS_OUTPUT directory.
Generate a flowchart of the pipeline in the data_products directory.
Pipeline Steps
Create Working Directories: Generates directories for each flare based on its class and timestamp.
Query Data Availability: Checks the availability of AIA and SXI data for the specified flares.
Download Data: Downloads the data files and saves them in a structured format.
Save Metadata: Stores metadata about the downloaded files, including URLs, timestamps, and file paths.
Error Handling
The script includes mechanisms to handle download errors and retry failed downloads.
If a file cannot be parsed or downloaded, it will log the issue and allow the pipeline to be restarted.
Notes
Ensure that the dataconfig.py file is correctly configured with paths to the data directories.
The script uses parallel processing (multiprocess=10) to speed up downloads. Adjust this value based on your system's capabilities.
Flowchart
The script generates a flowchart of the pipeline in the data_products directory as download_data_flowchart.png.

## Usage
To download data for a single flare:

Set input_time_list = ['2011-02-08 21:11:00.08'].
Set input_class_list = ['C4.0'].
Run the script. The data will be downloaded and organized in the specified directories.





