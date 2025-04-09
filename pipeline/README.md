# Here is a step by step process to run the ALEXIS pipeline. 
### Please feel free to submit an issue. 
## Warning
Ensure that the dataconfig.py file is correctly configured with paths to the data directories.
Ensure that you have followed the instructions in the on-boarding process. 
The processing pipeline has been split up into different componenets that must be run in order. 
This way, you can run the same pipeline in chuncks, making it easier to debug and to communicate with the team for trouble shooting. 

# Running ALEXIS by parts:
There are 4 main scripts that make up the ALEXIS pipeline. The four scripts are:
1. **ALEXIS_download_data.py**
    - *quick run*: change lines 84 and 85 to include a list of flare classes and flare datetimes you are interested in running. 
2. **ALEXIS_02_create_initialize_files.py**
    - *quick run*: once data for your flares is downloaded, ALEXIS_02_create_initialize_files.py can  be run without further modification. 
3. **ALEXIS_03_create_wd_w_init_files_before_hyperspectral.py**
    - *quick run*: once initialization files have been created, ALEXIS_03_create_wd_w_init_files_before_hyperspectral.py can be run without further modification. 
4. **create_peakfinder_scale_02.py**
    - *quick run*: once hyperspectral candidates are created, create_peakfinder_scale_02.py can be run without further modification.

#
Note: 
The script uses parallel processing (multiprocess=10) to speed up downloads. Adjust this value based on your system's capabilities. 
The script includes mechanisms to handle download errors and retry failed downloads; if a file cannot be parsed or downloaded, it will log the issue and allow the pipeline to continue.
If the pipeline is re-run, it will re-try all files that you were not able to download. 

## Usage
The user needs to set what time-ranges and flare magnitude they want ALEXIS to create. 
For example, if you are interested in the C4.0 class flare that occured on 2011-02-08 21:11:00.08, the pipeline will create all files related to this flare in the directory: "<path_to_your_data_directory>/flare_candidates/flarecandidate_C4.0_at_2011-02-08T21_11_00_08.working".

On line 84 and 85 of the ALEXIS_download_data.py script you will see the following:

``` 
Set input_time_list = ['2011-02-08 21:11:00.08']
Set input_class_list = ['C4.0']
```

Currently, the pipeline will run data for that combination of input_class and input_time. The data will be downloaded and organized in the specified directories.

If you want to do multiple events, change those lines to reflect the working_directories you want to create. 

For example, if you are also interested in the C1.8 flare that occured on 2011-11-14 20:12:30.25 UTC you would modify the above code as follows to create two working_directories: 

```
# Make flare directories:
# 1st flare directory to create: flarecandidate_C4.0_at_2011-02-08T21_11_00_08.working
# 2nd flare directory to create: flarecandidate_C8.2_at_2012-07-01T15_45_00_47.working

# flare class list in the format of : '<letter><number>.<number>'
input_class_list = ['C4.0', 'C8.2']

# flare date-time list in UTC with format: 'YYYY-MM-DD HH:MM:SS.ss'
input_time_list = ['2011-02-08 21:11:00.08', '2012-07-01 15:45:00.47']

```





