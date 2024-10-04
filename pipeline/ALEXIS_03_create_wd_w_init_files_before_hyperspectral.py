import re 
from glob import glob
import numpy as np
import re
import os
import pickle
import dataconfig_argonne as dataconfig
import pandas as pd
import helio_reg_exp_module
import query_the_data
import check_data_qual_module_02


# some functions for pipeline

def convert_int_class_to_float(flare_class):

    """ 
        make sure flare class is not C1 or M1 but C1.0 or M1.0 etc.
    """
    
    if len(flare_class) == 2:
        flare_letter = flare_class[:1]
        flare_number = np.float(flare_class[1:])
        flare_class = f'{flare_letter}{flare_number}'

    return(flare_class)

def create_working_dir(flare_class, flare_time):
    this_file_time = f'{flare_time:%Y-%m-%dT%H_%M_%S_%f}'[:-4] # cut the milisecond precision to seconds
    name_of_directory = f'flarecandidate_{flare_class}_at_{this_file_time}.working'
    return(name_of_directory)

tw = lambda x: os.path.join(WORKING_DIR, x)

## end of some functions

# workdir variable
WORKING_DIR = dataconfig.DATA_DIR_FLARE_CANDIDATES


# load flare_list from differential analysis of XRS data and aggregate to SolarSoft and SWPC

flare_list = pickle.load(open(f'{dataconfig.DATA_DIR_PRODUCTS}/agg_flare_df.pickle', 'rb'))

# all_examples_for_argonne = flare_list[(flare_list.id_tuple == (1,1,1)) | (flare_list.id_tuple == (1,0,1)) | (flare_list.id_tuple == (1,1,0))] # uncomment for full argonne sample
all_examples_for_argonne = flare_list # place holder for when you uncomment above line. 

# make sure that we are making a letter-float and not letter-integer
all_examples_for_argonne['merged_class'] = [convert_int_class_to_float(this_class) for this_class in all_examples_for_argonne.merged_class]

# make working directory paths
all_examples_for_argonne['working_dir'] = [tw(create_working_dir(this_class, this_time)) for this_class, this_time in zip(all_examples_for_argonne['merged_class'], all_examples_for_argonne['merged_datetime'])]

##### limit this run to WD we have on lift. Comment up to here for full argonne sample. 

# load the working_dir for lift3 proof of concept
lift_flares = pickle.load(open('/home/padialjr/jorge-helio/argonne_files/movie_wd_file_list.pickle', 'rb'))

# parse lift path to return just the working directory
test_dir_list = [helio_reg_exp_module.work_dir_from_flare_candidate_input_string(this_file) for this_file in lift_flares]

do_these_directories = pd.concat([all_examples_for_argonne[all_examples_for_argonne.working_dir == tw(this_dir)] for this_dir in test_dir_list])

##### limit this run to WD we have on lift. Comment section above to create all WD for full argonne sample

# do_these_directories = all_examples_for_argonne # uncomment when running on full argonne sample


### check good data quality for elements that we initialized to download data

def decide_good_qual_wavelength_instr_pairs_per_work_dir(row):

    work_dir = row['working_dir']

    CHECK_WORK_DIR_EXISTS = os.path.isdir(work_dir)

    CHECK_INIT_FILE_EXISTS = os.path.isfile(f'{work_dir}/initialize_hyperspectral_with_these_files_df.pickle')

    if not CHECK_INIT_FILE_EXISTS:
        # os.makedirs(f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}')

        this_date_time = row['merged_datetime']

        clean_df = query_the_data.query_downloaded_image_availability(this_date_time)

        # we want to verify singile files, peakfinder can return multiples entries
        # lets drop duplicates
        duplicates_url_drop_bc_of_peakfinder = clean_df.drop_duplicates(subset = ['url'])

        duplicates_url_drop_bc_of_peakfinder['working_dir'] = [work_dir for _ in duplicates_url_drop_bc_of_peakfinder.QUALITY]

        clean_df['working_dir'] = [work_dir for _ in clean_df.QUALITY]

        data_qual_df = (check_data_qual_module_02.return_good_data_quantities_per_wl_and_inst(duplicates_url_drop_bc_of_peakfinder))

        data_group_better_than_90_percent = check_data_qual_module_02.return_better_than_90_percent_of_data(data_qual_df)

        do_these_wl_only = check_data_qual_module_02.return_enough_length_data(data_group_better_than_90_percent)

        if len(do_these_wl_only) == 0:

            print('-------------------')

            print(f'no data for {work_dir}')

            print('-------------------')

        else:

            print(f'making init for {work_dir}')

            good_wl_inst_pairs = pd.concat([clean_df[(clean_df.wavelength == this_wl) & (clean_df.instrument == this_inst) & (clean_df.telescope == this_tel)] for this_wl, this_inst, this_tel in zip(do_these_wl_only.wavelength, do_these_wl_only.instrument, do_these_wl_only.telescope)])

            masked_qual = good_wl_inst_pairs[(good_wl_inst_pairs.data_level == 'lev2') & (good_wl_inst_pairs.QUALITY == 0) ]
            
            good_wl_inst_pairs_single_files = pd.concat([duplicates_url_drop_bc_of_peakfinder[(duplicates_url_drop_bc_of_peakfinder.wavelength == this_wl) & (duplicates_url_drop_bc_of_peakfinder.instrument == this_inst) & (duplicates_url_drop_bc_of_peakfinder.telescope == this_tel)] for this_wl, this_inst, this_tel in zip(do_these_wl_only.wavelength, do_these_wl_only.instrument, do_these_wl_only.telescope)])

            masked_qual_single_files = good_wl_inst_pairs_single_files[(good_wl_inst_pairs_single_files.data_level == 'lev2') & (good_wl_inst_pairs_single_files.QUALITY == 0) ]

            single_output_file_name = f'{work_dir}/initialize_with_these_files_df.pickle'

            hyperspectral_output_file_name = f'{work_dir}/initialize_hyperspectral_with_these_files_df.pickle'

#             print(masked_qual.wavelength.unique(), masked_qual.QUALITY.unique())
            
            single_drop_list = ['peakfinder_time_1', 'dbscan_time_1', 'cleaning_time_1', 'peaks_found','dbscan_1_x_hpc', 'dbscan_1_y_hpc', 'dbscan_1_x_pix', 'dbscan_1_y_pix','dbscan_1_num_members', 'dbscan_1_label']

            output_masked_single_files = masked_qual_single_files.drop(single_drop_list, axis = 1).reset_index(drop = True)
            
            # return(masked_qual,output_masked_single_files)

            if not CHECK_WORK_DIR_EXISTS:

                os.makedirs(work_dir)

            pickle.dump(masked_qual, open(hyperspectral_output_file_name, 'wb'))

            pickle.dump(output_masked_single_files, open(single_output_file_name, 'wb'))

    else:
        pass
        #print(f'init files already made: {work_dir}')


# # parse which working directories have data that are usable and begin 
# # analysis
for _, row in do_these_flares.iterrows():

    decide_good_qual_wavelength_instr_pairs_per_work_dir(row)


# Get the current time in the CST timezone
current_time_end = datetime.now(cst)

# Print the current time in the CST timezone
print("Current time end in CST timezone:", current_time_end.strftime("%Y-%m-%d %H:%M:%S %Z"))
