# print('start loading libraries')
import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import pytz
from datetime import datetime
# Get the CST timezone
cst = pytz.timezone('US/Central')

load_data_start = datetime.now(cst)
import re

import pandas as pd
# from time import sleep
# from random import randint
import time

import sys
import os
import hashlib

from modules import download_module_w_new_file_syst

# import query_the_data_argonne as query_the_data

import dataconfig

# import convert_datetime

from ruffus import *

# import sqlalchemy as sa

# from astropy.io import fits

# import joblibs
# import requests

import warnings
warnings.filterwarnings("ignore")

from modules import clean_img_data_02

# import matplotlib.pyplot as plt

# from download_module_argonne 

# import check_data_qual_module_02

# from glob import glob

# import numpy as np

import pickle
import argparse

load_data_end = datetime.now(cst)

print(f'loaded libraries MAIN2: {load_data_end - load_data_start}')


# print('libraries loaded')

#########

########################## start prepping test files for lift vs argonne comparison ###################################

# we want to run the full alexis pipeline on test_dir on debug node to compare with 
# the speed at which lift vs argonne can run through this pipeline

# test_dir = 'flarecandidate_C4.0_at_2011-09-22T20_20_20_37.working'
# test_dir = 'flarecandidate_C1.1_at_2011-11-29T01_16_00_16.working'
# test_dir = 'flarecandidate_M1.0_at_2014-02-14T16_39_06_19.working'

# test_dir_list = ['flarecandidate_C4.0_at_2011-09-22T20_20_20_37.working', 'flarecandidate_C1.1_at_2011-11-29T01_16_00_16.working',
#                  'flarecandidate_M1.0_at_2014-02-14T16_39_06_19.working']

# import pytz
# from datetime import datetime

# # Get the CST timezone
# cst = pytz.timezone('US/Central')

# # Get the current time in the CST timezone
# current_time_start = datetime.now(cst)

# # Print the current time in the CST timezone
# print("Current time start in CST timezone:", current_time_start.strftime("%Y-%m-%d %H:%M:%S %Z"))

# all_start = time.time()
# # print(test_dir)

# #########

# def convert_int_class_to_float(flare_class):

#     """ 
#         make sure flare class is not C1 or M1 but C1.0 or M1.0 etc.
#     """
    
#     if len(flare_class) == 2:
#         flare_letter = flare_class[:1]
#         flare_number = np.float(flare_class[1:])
#         flare_class = f'{flare_letter}{flare_number}'

#     return(flare_class)

# #########

# # load flare_list from differential analysis of XRS data and aggregate to SolarSoft and SWPC

# flare_list = pickle.load(open(f'{dataconfig.DATA_DIR_PRODUCTS}/agg_flare_df.pickle', 'rb'))

# # make sure that we are making a letter-float and not letter-integer
# flare_list['merged_class'] = [convert_int_class_to_float(this_class) for this_class in flare_list.merged_class]

# # create initialization of data used in ALEXIS demo

# def create_working_dir(flare_class, flare_time):
#     this_file_time = f'{flare_time:%Y-%m-%dT%H_%M_%S_%f}'[:-4] # cut the milisecond precision to seconds
#     name_of_directory = f'flarecandidate_{flare_class}_at_{this_file_time}.working'
#     return(name_of_directory)

# # #### tell program to download the following example for the proposal
# # WORKING_DIR = dataconfig.DATA_DIR_FLARE_CANDIDATES
# # # WORKING_DIR

# # tw = lambda x: os.path.join(WORKING_DIR, x)

# flare_list['working_dir'] = [(create_working_dir(this_class, this_time)) for this_class, this_time in zip(flare_list['merged_class'], flare_list['merged_datetime'])]


# many_dir = pd.concat([flare_list[flare_list.working_dir == this_dir] for this_dir in test_dir_list])

# DL_DATASETS = []
# for flare_candidate in many_dir.itertuples():
#     data_directory = f'{dataconfig.MAIN_DIR}/image_data'

#     the_timestamp = flare_candidate.merged_datetime

#     aia_avail_df = query_the_data.aia_availability(input_datetime = the_timestamp, query_time = 20) #query_time must be given in minutes

#     # save aia_sql result with filename instead of file_name. Change in order to continue
#     aia_avail_df.rename(columns = {'filename':'file_name'}, inplace = True)

#     sxi_avail_df = query_the_data.sxi_availability_sql_db(input_datetime = the_timestamp, query_time = 40)

#     # #sci qual SXI is data_level == 'BA'
#     sci_qual_sxi_df = sxi_avail_df[sxi_avail_df.data_level == 'BA']

#     sci_qual_sxi_df['url'] = [re.search(r'ftp://satdat.ngdc.noaa.gov/sxi/archive/fits/goes\d{2}/\d{4}/\d{2}/\d{2}/SXI_\d{8}_\d{9}_BA_\d{2}.FTS', this_download_string).group(0) for this_download_string in sci_qual_sxi_df.download_string]

#     #drop columns to make aia and SXI the same dictionaries

#     PASS_THIS_SXI = sci_qual_sxi_df.drop(['download_string', 'data_level', 'instrument'], axis = 1)

#     PASS_THIS_AIA = aia_avail_df.drop(['EXPTIME', 'QUALITY', 'WAVELNTH'], axis = 1)


#     lst_concat_df = pd.concat([PASS_THIS_SXI, PASS_THIS_AIA])

#     # lst_concat_df['entry_num'] = [k for _ in lst_concat_df.url]

#     # # lst_concat_df = aia_avail_df

#     # k = k + 1

#     DL_DATASETS.append(lst_concat_df)


# # all data has been found, concat into dataframe of every single file
# THESE_DF = pd.concat(DL_DATASETS)


# THESE_DF['hashed_url'] = [hashlib.sha256(url.encode('utf-8')).hexdigest() for url in THESE_DF.url]

# THESE_DF['file_syst_sub_dir'] = [hashed_url[:2] for hashed_url in THESE_DF['hashed_url']]
# THESE_DF

# def make_outfile(row):

#     # print(row['file_syst_sub_dir'])

#     sub = row[1].file_syst_sub_dir

#     ruff = row[1]['hashed_url']

#     outfile = f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}/{sub}/{ruff}.downloaded'
#     return(outfile)

# THESE_DF['ruffus_download_file'] = [make_outfile(row) for row in THESE_DF.iterrows()]

# THESE_DF['file_exists'] = [os.path.exists(this_file) for this_file in THESE_DF.ruffus_download_file]

# download_data = THESE_DF[THESE_DF.file_exists == True].ruffus_download_file.to_list()

# download_data



########################## done prepping test files for lift vs argonne comparison ###################################


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# download_data = glob(f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}/*/*.downloaded')[:1000]


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


parser = argparse.ArgumentParser()

parser.add_argument("-s", "--start", type=int)
parser.add_argument("-e", "--end", type=int)
# parser.add_argument("-a", "--all", type=bool)

args = parser.parse_args()

start, end = args.start, args.end


load_data_start = datetime.now(cst)


# need to add glob function or something. Maybe based on working dir?
download_data = pickle.load(open(os.path.join(dataconfig.DATA_DIR_PRODUCTS, 'downloaded_data_list.pickle'), 'rb'))[start:end]


load_data_end = datetime.now(cst)

print(f'list of data added: {load_data_end - load_data_start}')

# os.system(f'echo file ran')
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# print(len(download_data), start, end)


@transform(download_data, suffix('.downloaded'), '.parsed_downloader_w_single_peakfinder.pickle') # function 2
def file_size_and_speed_w_single_image_peakfinder(infile, outfile):

    print('--------------------')

    # Get the current time in the CST timezone
    current_time_start = datetime.now(cst)

    load_data_start = datetime.now(cst)

    downloaded_dict = pickle.load(open(infile, 'rb'))

    load_data_end = datetime.now(cst)

    print(f'infile loaded: {load_data_end - load_data_start}, {infile}')

    # we included the parser in the download script. Need only pass the ruffus download outfile
    parsed_dict = downloaded_dict

    clean_time_start = time.time()

    new_config, data, raw_header, dbscan_cluster_eps = download_module_w_new_file_syst_argonne.clean_imgs_for_peakfinder(parsed_dict)

    clean_time_end = time.time()

    clean_time = clean_time_end - clean_time_start

    single_peakfinder = download_module_w_new_file_syst_argonne.single_image_peakfinder(new_config, data, raw_header, dbscan_cluster_eps, clean_time)

    # Get the current time in the CST timezone
    current_time_end = datetime.now(cst)

    instrument = single_peakfinder['instrument'].iloc[0]

    print(f'full analysis {instrument}: {current_time_end - current_time_start}')

    print('--------------------')

    # output_df = join_all_peaks_found_into_coord_list(single_peakfinder)

    pickle.dump(single_peakfinder, open(outfile, 'wb'))

# all_end = time.time()



pipeline_run([file_size_and_speed_w_single_image_peakfinder], multiprocess = 64, verbose = 1)

# print((all_end-all_start)/60)

# # Get the current time in the CST timezone
# current_time_end = datetime.now(cst)

# # Print the current time in the CST timezone
# print("Current time end in CST timezone:", current_time_end.strftime("%Y-%m-%d %H:%M:%S %Z"))

# pipeline_run([insert_image_downloaded_availability_into_sqlite], multiprocess = 20, verbose = 1)

# ##################################################
# ### data has been downloaded, parsed, and cleaned
# ### insert into SQLITE database
# ##################################################

# import json
# import sqlalchemy as sa

# engine = sa.create_engine(f'sqlite:////{dataconfig.DATA_DIR_PRODUCTS}/image_data_availability.db', echo=False)
# metadata = sa.MetaData()

# image_data_availability = sa.Table(
#     'image_data_availability', metadata,
#     sa.Column('download_string', sa.types.String()),
#     sa.Column('time_stamp', sa.types.Float(), index=True),
#     sa.Column('data_level', sa.types.String()),
#     sa.Column('file_name', sa.types.String()),  # Added index on file_name
#     sa.Column('exp_time', sa.types.Float()),
#     sa.Column('file_size_MB', sa.types.Float()),
#     sa.Column('QUALITY', sa.types.Float()),
#     sa.Column('wavelength', sa.types.Integer()),
#     sa.Column('url', sa.types.String()),
#     sa.Column('instrument', sa.types.String(), index=True),
#     sa.Column('file_path', sa.types.String()),
#     sa.Column('wget_log_file', sa.types.String()),
#     sa.Column('download_speed', sa.types.String()),
#     sa.Column('cached_dict_path', sa.types.String()),
#     sa.Column('file_size_MB', sa.types.Float()),
#     sa.Column('telescope', sa.types.String()),
#     sa.Column('img_data_max', sa.types.Float()),
#     sa.Column('img_data_min', sa.types.Float()),
#     sa.Column('img_flux', sa.types.Float()),
#     sa.Column('peakfinder_time_1', sa.types.Float()),
#     sa.Column('dbscan_time_1', sa.types.Float()),
#     sa.Column('cleaning_time_1', sa.types.Float()),
#     sa.Column('peaks_found', sa.types.Boolean()),
#     sa.Column('dbscan_1_x_hpc', sa.types.Float()),
#     sa.Column('dbscan_1_y_hpc', sa.types.Float()),
#     sa.Column('dbscan_1_x_pix', sa.types.Float()),
#     sa.Column('dbscan_1_y_pix', sa.types.Float()),
#     sa.Column('dbscan_1_num_members', sa.types.Float()),
#     sa.Column('dbscan_time_1', sa.types.Float()),
#     sa.Column('dbscan_1_label', sa.types.Float())
# )



# @jobs_limit(1)
# @transform(file_size_and_speed_w_single_image_peakfinder, suffix('.parsed_downloader_w_single_peakfinder.pickle'), '.sqlite_inserted.pickle')
# def insert_image_downloaded_availability_into_sqlite(infile, outfile):

#     infile_df = pickle.load(open(infile, 'rb'))

#     infile_to_dict = infile_df.to_dict('records')

#     metadata.create_all(engine)

#     conn = engine.connect()

#     conn.execute(image_data_availability.insert(), infile_to_dict)

#     pickle.dump(infile_to_dict, open(outfile, 'wb'))



# pipeline_run([insert_image_downloaded_availability_into_sqlite], multiprocess = 20, verbose = 1)



# ### check good data quality for elements that we initialized to download data

# def decide_good_qual_wavelength_instr_pairs_per_work_dir(row):

#     work_dir = row['working_dir']

#     CHECK_WORK_DIR_EXISTS = os.path.isdir(work_dir)

#     CHECK_INIT_FILE_EXISTS = os.path.isdir(f'{work_dir}/initialize_with_these_files_df.pickle')

#     if not CHECK_INIT_FILE_EXISTS:
#         # os.makedirs(f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}')

#         this_date_time = row['merged_datetime']

#         clean_df = query_the_data.query_downloaded_image_availability(this_date_time)

#         # we want to verify singile files, peakfinder can return multiples entries
#         # lets drop duplicates

#         duplicates_url_drop_bc_of_peakfinder['working_dir'] = [work_dir for _ in duplicates_url_drop_bc_of_peakfinder.QUALITY]

#         data_qual_df = (check_data_qual_module_02.return_good_data_quantities_per_wl_and_inst(duplicates_url_drop_bc_of_peakfinder))

#         data_group_better_than_90_percent = check_data_qual_module_02.return_better_than_90_percent_of_data(data_qual_df)

#         do_these_wl_only = check_data_qual_module_02.return_enough_length_data(data_group_better_than_90_percent)

#         if len(do_these_wl_only) == 0:

#             print(f'no data for {work_dir}')

#         else:

#             print(f'making init for {work_dir}')

#             good_wl_inst_pairs = pd.concat([clean_df[(clean_df.wavelength == this_wl) & (clean_df.instrument == this_inst) & (clean_df.telescope == this_tel)] for this_wl, this_inst, this_tel in zip(do_these_wl_only.wavelength, do_these_wl_only.instrument, do_these_wl_only.telescope)])

#             masked_qual = good_wl_inst_pairs[(good_wl_inst_pairs.data_level == 'lev2') & (good_wl_inst_pairs.QUALITY == 0) ]

#             output_file_name = f'{work_dir}/initialize_with_these_files_df.pickle'

#             os.makedirs(work_dir)

#             pickle.dump(masked_qual, open(output_file_name, 'wb'))

#     else:
#         pass
#         # print(f'directory made: {work_dir}')

# # parse which working directories have data that are usable and begin 
# # analysis
# for _, row in do_these_flares.iterrows():

#     decide_good_qual_wavelength_instr_pairs_per_work_dir(row)