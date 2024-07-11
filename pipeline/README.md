# Here is a step by step process to run the ALEXIS pipeline. 
There are two ways in which to run this pipeline. The first is to do it by parts. The second is a stand alone pipleline that one can give any date from May 2010-March2020 and ALEXIS will try to find anything within an hour of the target date-time. The first way will run the same pipeline but in chuncks, making it easier to debug and to communicate with the team for trouble shooting. Please feel free to submit an issue. 

# ALEXIS by Parts:
There are 4 main scripts that make up the ALEXIS pipeline. The four scripts are:
# 1. ALEXIS_download_data.py
# 2. ALEXIS_02_create_initialize_files.py
# 3. ALEXIS_02_create_workdir_w_init_file_before_main_ALEXIS02.py
# 4. create_peakfinder_scale_02.py