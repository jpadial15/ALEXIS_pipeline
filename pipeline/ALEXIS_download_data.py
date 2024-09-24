import sys  
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import re
import pickle
import pandas as pd
from time import sleep
from random import randint
import time
import sys
import os
import hashlib
from ruffus import *
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# argonne libraries

# from download_module_argonne import *

from modules import download_module_w_new_file_syst
from modules import query_the_data
import dataconfig
from modules import convert_datetime



#########

def convert_int_class_to_float(flare_class):

    """ 
        make sure flare class is not C1 or M1 but C1.0 or M1.0 etc.
    """
    
    if len(flare_class) == 2:

        flare_letter = flare_class[:1]
        flare_number = np.float(flare_class[1:])
        flare_class = f'{flare_letter}{flare_number}'

    return(flare_class)

#########

# load flare_list from differential analysis of XRS data and aggregate to SolarSoft and SWPC

flare_list = pickle.load(open(f'{dataconfig.DATA_DIR_PRODUCTS}/agg_flare_df.pickle', 'rb'))

# make sure that we are making a letter-float and not letter-integer
flare_list['merged_class'] = [convert_int_class_to_float(this_class) for this_class in flare_list.merged_class]

# mask for flares everyone agrees on
flares_111 = flare_list[flare_list.id_tuple == (1,1,1)].reset_index(drop = True).sort_values(by = 'merged_datetime')

# mask for flares everyone agrees on
flares_101 = flare_list[flare_list.id_tuple == (1,0,1)].reset_index(drop = True).sort_values(by = 'merged_datetime')

flares_110 = flare_list[flare_list.id_tuple == (1,1,0)].reset_index(drop = True).sort_values(by = 'merged_datetime')

ALL_FLARES = pd.concat([flares_111, flares_101, flares_110])

THESE_FLARES = ALL_FLARES.drop_duplicates(subset=['merged_datetime'])

# create dictionaries for downloading data

DL_DATASETS = []
# keep track of flare number w/ k
k  = 0
for flare_candidate in THESE_FLARES.itertuples():

    data_directory = f'{dataconfig.MAIN_DIR}/image_data'

    the_timestamp = flare_candidate.merged_datetime

    aia_avail_df = query_the_data.aia_availability(input_datetime = the_timestamp, query_time = 20) #query_time must be given in minutes

    # save aia_sql result with filename instead of file_name. Change in order to continue
    aia_avail_df.rename(columns = {'filename':'file_name'}, inplace = True)

    sxi_avail_df = query_the_data.sxi_availability_sql_db(input_datetime = the_timestamp, query_time = 40) #query_time must be given in minutes

    # #sci qual SXI is data_level == 'BA'
    sci_qual_sxi_df = sxi_avail_df[sxi_avail_df.data_level == 'BA']

    sci_qual_sxi_df['url'] = [re.search(r'ftp://satdat.ngdc.noaa.gov/sxi/archive/fits/goes\d{2}/\d{4}/\d{2}/\d{2}/SXI_\d{8}_\d{9}_BA_\d{2}.FTS', this_download_string).group(0) for this_download_string in sci_qual_sxi_df.download_string]
    # [re.search(r'https:\/\/www.ncei.noaa.gov\/data\/goes-solar-xray-imager\/access\/fits\/goes\d{2}\/\d{4}\/\d{2}\/\d{2}\/SXI_\d{8}_\d{9}_[A-B][A-B]_\d{2}.FTS', this_download_string).group(0) for this_download_string in available_data_df.download_string]

    #drop columns to make aia and SXI the same dictionaries

    PASS_THIS_SXI = sci_qual_sxi_df.drop(['download_string', 'data_level', 'instrument'], axis = 1)

    PASS_THIS_AIA = aia_avail_df.drop(['EXPTIME', 'QUALITY', 'WAVELNTH'], axis = 1)


    lst_concat_df = pd.concat([PASS_THIS_SXI, PASS_THIS_AIA])

    lst_concat_df['entry_num'] = [k for _ in lst_concat_df.url]

    # lst_concat_df = aia_avail_df

    k = k + 1

    DL_DATASETS.append(lst_concat_df)


# all data has been found, concat into dataframe of every single file
THESE_DF = pd.concat(DL_DATASETS)

# drop any duplicate urls

not_duplicated_urls = THESE_DF.drop_duplicates(subset=['url']).reset_index(drop = True)



# take out file that is failing to see if that helps w the errno28: no space left on device

#not_duplicated_urls = not_duplicated_urls[not_duplicated_urls.url !='http://jsoc.stanford.edu/SUM35/D516428691/S00000/image_lev1.fits']

# ######
# # download_these are the dictionaries which are equal to the config files
# # for download for unique url's

download_these = not_duplicated_urls.to_dict('records')

print('done finding files')

# #####
def dl_params():
    for config in download_these:

        infile = None

        url = config['url']

        hashed_url = hashlib.sha256(url.encode('utf-8')).hexdigest()

        file_syst_sub_dir = hashed_url[:2]

        config['file_syst_sub_dir'] = file_syst_sub_dir

        outfile = f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}/{file_syst_sub_dir}/{hashed_url}.downloaded'

        config['ruffus_download_file'] = outfile
    
        config['hashed_url'] = hashed_url

        yield(infile, outfile, config)


# #####

import errno
import datetime

@files(dl_params)
@jobs_limit(10)
def download_data(infile, outfile, config): #function 1

    
    CHECK_RUFFUS_OUTPUT_DIRECTORY_EXISTS = os.path.isdir(f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}')

    if not CHECK_RUFFUS_OUTPUT_DIRECTORY_EXISTS:
        os.makedirs(f'{dataconfig.DATA_DIR_IMG_RUFFUS_OUTPUT}')
    # load URL class
    # url_downloader = UrlDownloader_parse_wget(MAIN_DL_DIRECTORY=f'{dataconfig.DATA_DIR_IMG_DATA}')

    all_path_info_w_config = download_module_w_new_file_syst.all_paths_config_and_hashed_url(dataconfig.DATA_DIR_IMG_DATA, config)

    # url = config['url']

    # cached_dict = url_downloader.download_url(url)

    cached_dict = download_module_w_new_file_syst.download_url(all_path_info_w_config)

    # print(cached_dict)
    if cached_dict != None:

        old_file_name = config['file_name']

        if old_file_name.startswith('SXI'):
            cached_dict['instrument'] = 'SXI'

        if old_file_name.startswith('aia'):
            cached_dict['instrument'] = 'AIA'

        # save ruffus outfile

        # print(cached_dict)

        pickle.dump(cached_dict, open(outfile, 'wb'))

        #---------------------------------------------------------# 
        # ----- this is an error handler ----- #
#         try:
#             pickle.dump(cached_dict, open(outfile, 'wb'))
#         except OSError as e:
#             if e.errno == errno.ENOSPC:
# #                print(cached_dict)
#                 match = re.search(r'ruffus_output_files/(.*?)(\.downloaded)?$', outfile)
                
#                 cached_file = match.group(1)

#                 output = f'{cached_file}.no_space_error.pickle'
#                 this_file = f'~/jorge-helio/argonne_files/no_space_error_cache/{output}.txt'
                
# #                time_now = (datetime.datetime.now())

#                 from datetime import datetime, timedelta, timezone

#                 # Define the CST timezone offset
#                 cst_offset = timedelta(hours=-5)

#                 # Get the current UTC time

#                 current_utc_time = datetime.now(timezone.utc)

#                 # Calculate the current time in CST
#                 current_time_cst = current_utc_time + cst_offset
#                 time_now = current_time_cst.strftime('%Y-%m-%d %H:%M:%S')
#                 print(f'no space error: {cached_file} {time_now}')
#                 os.system(f'touch {this_file}')
                
#        #         print(this_file)
# #                pickle.dump(cached_dict, open(this_file, 'wb'))


#             else:
#                 print(f'an oserror {e} occurred')

            # ----- end of error handler ------ #
            #--------------------------------------------------------#

    else: 

        old_config = all_path_info_w_config.CONFIG_FILE

        old_file_name, this_url = old_config['file_name'], old_config['url']

        print(f'data for {old_file_name}, {this_url} not parsable. Will re-run when ruffus is restarted')

        pass

#CHECKSUM_REGENERATE = 2

#pipeline_run([download_data], multiprocess = 2, verbose = 1, touch_files_only = CHECKSUM_REGENERATE)

pipeline_run([download_data], multiprocess = 10, verbose = 3)



# flares_111 = flare_list[flare_list.id_tuple == (1,1,1)].reset_index(drop = True).sort_values(by = 'merged_datetime')

# do_these_flares_and = flares_111[150:].sample(n = 10, random_state = 14)

# do_these_flares_too = flares_111[flares_111.merged_class > 'C2.2'][50:].sample(n = 16, random_state = 27)
# # flares_111

# do_these_flares_also = flares_111[(flares_111.merged_datetime.dt.year == 2016) & (flares_111.merged_datetime.dt.month == 8)].reset_index(drop = True)[16:20]


# do_these_flares_and = flares_111[10:].sample(n = 10, random_state = 14)

# m_class_flares = flares_111[(flares_111.merged_class > 'M1') & (flares_111.merged_class < 'M9.9')].sample(n = 15, random_state = 14)

# x_class_flares = flares_111[(flares_111.merged_class > 'X1') & (flares_111.merged_class < 'X9.9')].sample(n = 15, random_state = 14)

# do_these_flares_also['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in do_these_flares_also.merged_class]

# do_these_flares_and['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in do_these_flares_and.merged_class]



# x_class_flares['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in x_class_flares.merged_class]


# m_class_flares2 = flares_111[(flares_111.merged_class > 'M1') & (flares_111.merged_class < 'M9.9')].sample(n = 20, random_state = 52)

# x_class_flares2 = flares_111[(flares_111.merged_class > 'X1') & (flares_111.merged_class < 'X9.9')].sample(n = 23, random_state = 52)

# c_class_flares = flares_111[(flares_111.merged_class > 'C1') & (flares_111.merged_class < 'C9.9')].sample(n = 10, random_state = 4)

# m_class_flares['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in m_class_flares.merged_class]

# m_class_flares2['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in m_class_flares2.merged_class]

# x_class_flares2['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in x_class_flares2.merged_class]

# c_class_flares['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in c_class_flares.merged_class]

# do_these_flares_too['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in do_these_flares_too.merged_class]

# do_these_flares = pd.concat([do_these_flares_and, do_these_flares_also, do_these_flares_too,m_class_flares,x_class_flares,m_class_flares2,x_class_flares2, c_class_flares])


# do_these_flares = do_these_flares.drop_duplicates().sort_values(by = 'merged_class').reset_index(drop = True)

# # # Above are 100 flares that we have proof of concept working. 

# # # Lets expand this to 500 examples

# all_x_class = flares_111[(flares_111.merged_class > 'X0')] # there are only 35 X-Class

# next_m_class = flares_111[(flares_111.merged_class > 'M2') & (flares_111.merged_class < 'M9.9')].sample(n = 150, random_state = 52)

# next_c_class = flares_111[(flares_111.merged_class > 'C1') & (flares_111.merged_class < 'C9.9')].sample(n = 150, random_state = 52)


# next_c_class3 = flares_111[(flares_111.merged_class > 'C3') & (flares_111.merged_class < 'C9.9')].sample(n = 150, random_state = 35)

# next_m_class3 = flares_111[(flares_111.merged_class > 'M1') & (flares_111.merged_class < 'M9.9')].sample(n = 40, random_state = 52)




# #add 500 more

# add_cs = flares_111[(flares_111.merged_class > "C") & (flares_111.merged_class < 'M')].sample(n = 300, random_state = 100)

# add_ms = flares_111[(flares_111.merged_class > "M") & (flares_111.merged_class < 'X')].sample(n = 300, random_state = 100)


# list_of_500_duplicated = pd.concat([do_these_flares, all_x_class, next_m_class, next_c_class, next_c_class3, add_cs, add_ms])


# list_of_500 = list_of_500_duplicated.drop_duplicates()

# # make sure the directory name includes flare + float
# # for Example, C1 will be changed to C1.0
# # need this for working directory regular expression.

# list_of_500['merged_class'] = [f'{this_merged_class[0]}{float(this_merged_class[1:])}' for this_merged_class in list_of_500.merged_class]


# list_of_500 = list_of_500.reset_index(drop = True)


# ##### ERASED ALL AIA DATA AND REDOWNLOADING###
# # big at C2.4 @ 2014-02-27T05_58_08_18 at index = 270
# # bug at C3.3 @ 2014-08-30T04_56_10_62 at index = 464
# # bug at C3.9 2015-10-19T17_27_00_55 at index = 495
# # bug at M2.1 @ 2014-02-12T15_51_06_19 index = 188

# list_of_500 = list_of_500.drop([270, 464, 495, 188, 433, 479,821,799, 759, 989, 991, 549, 946]) # cleaning bugs


# do_these_flares = list_of_500.drop_duplicates().sort_values(by = 'merged_class').reset_index(drop = True)

# ### taking too long for cluster filtering. undrop later oct172022
# # see ALEXIS_cluster_bug.ipynb

# ## 354 is taking a long time:=> has 5 candidates ... 4^5 cartesian product of zoom in :(
# ## 761's data is interpolating missing values

# do_these_flares['working_dir'] = [tw(create_working_dir(this_class, this_time)) for this_class, this_time in zip(do_these_flares['merged_class'], do_these_flares['merged_datetime'])]

# do_these_flares = do_these_flares.drop([354, 761])