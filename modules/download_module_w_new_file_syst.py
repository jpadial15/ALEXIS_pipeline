import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

import pytz
from datetime import datetime
# Get the CST timezone
cst = pytz.timezone('US/Central')

load_data_start = datetime.now(cst)

import os
import pickle
import re
from time import sleep
import dataconfig
from random import randint
# import errno
# import datetime
from modules import clean_img_data_02
import time
import convert_datetime
from skimage.transform import rotate
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
import pandas as pd
import numpy as np

load_data_end = datetime.now(cst)

print(f'loaded libraries download module w argonne file syst: {load_data_end - load_data_start}')


class all_paths_config_and_hashed_url:
    
    """
    Checks if directories are made. If not made, then make them. 
    
    Return directory object with possible directories. 
    
    --- > possible MAIN DIR are:
    
    check_return_if_dir_made.data_directory = <MAIN_DL_PATH>/fits_data
    
    check_return_if_dir_made.wget_logs_directory = <MAIN_DL_PATH>/wget_logs
    
    check_return_if_dir_made.cached_directory = <MAIN_DL_PATH>/cached_directory
    
    check_return_if_dir_made.ruffus_directory = <MAIN_DL_PATH>/ruffus_output_files
    
    --- > sub directories depends on the first two elements of the hashed file
    
    check_return_if_dir_made.data_sub_directory = <MAIN_DL_PATH>/fits_data/<first_two_elements_of_hash>
    
    check_return_if_dir_made.wget_log_sub_directory = <MAIN_DL_PATH>/wget_logs/<first_two_elements_of_hash>
    
    check_return_if_dir_made.cached_sub_directory = <MAIN_DL_PATH>/cached_directory/<first_two_elements_of_hash>
    
    check_return_if_dir_made.ruffus_sub_directory = <MAIN_DL_PATH>/ruffus_output_files/<first_two_elements_of_hash>
    
    
    """
    
    def __init__(self, MAIN_DL_DIRECTORY, config):
        
        # Make sure the main directories for each of our elements are made

        self.MAIN_DL_DIRECTORY = MAIN_DL_DIRECTORY
        
        self.CONFIG_FILE = config

        self.data_directory = os.path.join(self.MAIN_DL_DIRECTORY, 'fits_data')

        self.wget_logs_directory = os.path.join(self.MAIN_DL_DIRECTORY, 'wget_logs')

        self.cached_directory = os.path.join(self.MAIN_DL_DIRECTORY, 'cached_directory')

        self.ruffus_directory = os.path.join(self.MAIN_DL_DIRECTORY, 'ruffus_output_files')

        if not os.path.exists(self.MAIN_DL_DIRECTORY):

            os.makedirs(self.MAIN_DL_DIRECTORY)

        if not os.path.exists(self.data_directory):

            os.makedirs(self.data_directory)

        if not os.path.exists(self.wget_logs_directory):

            os.makedirs(self.wget_logs_directory)

        if not os.path.exists(self.cached_directory):

            os.makedirs(self.cached_directory)
            
        if not os.path.exists(self.ruffus_directory):

            os.makedirs(self.ruffus_directory)


        # make sure the sub directories are made

        file_syst_sub_dir = self.CONFIG_FILE['file_syst_sub_dir']

        self.data_sub_directory = os.path.join(self.data_directory , file_syst_sub_dir)

        self.wget_logs_sub_directory = os.path.join(self.wget_logs_directory, file_syst_sub_dir)

        self.cached_sub_directory = os.path.join(self.cached_directory, file_syst_sub_dir)

        self.ruffus_sub_directory = os.path.join(self.ruffus_directory, file_syst_sub_dir)


        if not os.path.exists(self.ruffus_sub_directory):

            os.makedirs(self.ruffus_sub_directory)

        if not os.path.exists(self.cached_sub_directory):

            os.makedirs(self.cached_sub_directory)

        if not os.path.exists(self.wget_logs_sub_directory):

            os.makedirs(self.wget_logs_sub_directory)

        if not os.path.exists(self.data_sub_directory):

            os.makedirs(self.data_sub_directory)

        # return file paths with subdir

        self.hashed_url = self.CONFIG_FILE['hashed_url']

        self.fits_path_w_sub = os.path.join(self.data_sub_directory , self.hashed_url)

        self.wget_log_path_w_sub = os.path.join(self.wget_logs_sub_directory, f'{self.hashed_url}.log')

        self.cached_path_w_sub = os.path.join(self.cached_sub_directory, f'{self.hashed_url}.cached.pickle')

        self.ruffus_download_path_w_sub = os.path.join(self.ruffus_sub_directory, f'{self.hashed_url}.downloaded')


        # return file paths w/o subdir

        self.fits_path_wo_sub = os.path.join(self.data_directory , self.hashed_url)

        self.wget_log_path_wo_sub = os.path.join(self.wget_logs_directory, f'{self.hashed_url}.log')

        self.cached_path_wo_sub = os.path.join(self.cached_directory, f'{self.hashed_url}.cached.pickle')

        self.ruffus_download_path_wo_sub = os.path.join(self.ruffus_directory, f'{self.hashed_url}.downloaded')

        # add a glob return

        self.glob_all_files = f'{self.MAIN_DL_DIRECTORY}/*/{file_syst_sub_dir}/{self.hashed_url}*'


        


def check_if_file_exists(check_exists_of_this_file):

    """
    verify if passed file exists
    """

    exists_bool = os.path.exists(check_exists_of_this_file)

    return(exists_bool)

def create_wget_link(all_path_info_w_config):

    # url_hash = self._hash_url(url)

    download_hashed_file_path = all_path_info_w_config.fits_path_w_sub

    wget_log_hashed_file_path = all_path_info_w_config.wget_log_path_w_sub

    url = all_path_info_w_config.CONFIG_FILE['url']

    wget_link = f'wget -o {wget_log_hashed_file_path} --output-document {download_hashed_file_path} {url}'

    return(wget_link)


def update_old_filesyt_files(all_path_info_w_config):

    # load dicts saved before filesys created
    old_cache = pickle.load(open(all_path_info_w_config.cached_path_wo_sub, 'rb'))

    old_ruffus = pickle.load(open(all_path_info_w_config.ruffus_download_path_wo_sub, 'rb'))

    # update values in the dicts for the filesys created

    old_cache['download_string'], old_ruffus['download_string'] = create_wget_link(all_path_info_w_config), create_wget_link(all_path_info_w_config)

    old_cache['wget_log_file'], old_ruffus['wget_log_file'] = all_path_info_w_config.wget_log_path_w_sub, all_path_info_w_config.wget_log_path_w_sub

    old_cache['cached_dict_path'], old_ruffus['cached_dict_path'] = all_path_info_w_config.cached_path_w_sub, all_path_info_w_config.cached_path_w_sub

    old_cache['file_path'], old_ruffus['file_path'] = all_path_info_w_config.fits_path_w_sub, all_path_info_w_config.fits_path_w_sub

    old_cache['file_syst_sub_dir'], old_ruffus['file_syst_sub_dir'] = all_path_info_w_config.CONFIG_FILE['file_syst_sub_dir'], all_path_info_w_config.CONFIG_FILE['file_syst_sub_dir']

    old_cache['time_stamp'], old_ruffus['time_stamp'] = all_path_info_w_config.CONFIG_FILE['time_stamp'], all_path_info_w_config.CONFIG_FILE['time_stamp']

    old_cache['date_time'], old_ruffus['date_time'] = all_path_info_w_config.CONFIG_FILE['date_time'], all_path_info_w_config.CONFIG_FILE['date_time']

    old_cache['hashed_url'], old_ruffus['hashed_url'] = all_path_info_w_config.CONFIG_FILE['hashed_url'], all_path_info_w_config.CONFIG_FILE['hashed_url']

    old_cache['file_name'], old_ruffus['file_name'] = all_path_info_w_config.CONFIG_FILE['file_name'], all_path_info_w_config.CONFIG_FILE['file_name']

    old_cache['entry_num'], old_ruffus['entry_num'] = all_path_info_w_config.CONFIG_FILE['entry_num'], all_path_info_w_config.CONFIG_FILE['entry_num']

    old_cache['ruffus_download_file'], old_ruffus['ruffus_download_file'] = all_path_info_w_config.ruffus_download_path_w_sub, all_path_info_w_config.ruffus_download_path_w_sub

    # move fits files and wget logs to new directories

    move_wget_log_str = f'mv {all_path_info_w_config.wget_log_path_wo_sub} {all_path_info_w_config.wget_log_path_w_sub}'

    move_fits_str = f'mv {all_path_info_w_config.fits_path_wo_sub} {all_path_info_w_config.fits_path_w_sub}'

    os.system(move_wget_log_str)

    os.system(move_fits_str)

    return(old_cache, old_ruffus)


def find_file_size_and_speed_from_wget_log(path_to_wget_log):

    with open(path_to_wget_log, 'r') as f:
        log_content = f.read()

    # Extract download speed and file size from log content
    match_speed = re.search(r'\(([\d.]+ [A-Za-z]+/s)\)', log_content)
    match_size = re.search(r'Length: (\d+)', log_content)

    if match_speed and match_size:
        download_speed = match_speed.group(1)
        file_size = int(match_size.group(1))/(1024*1024)

    return(download_speed, file_size)



def download_url(all_path_info_w_config):

    if check_if_file_exists(all_path_info_w_config.fits_path_wo_sub): # verify if this is an download made before creating binary filesystem organization

        new_cache, new_ruffus = update_old_filesyt_files(all_path_info_w_config)

        # print('need to transfer old files')

        pickle.dump(new_cache, open(all_path_info_w_config.cached_path_w_sub, 'wb'))

        # print('need to delete old cache')

        os.system(f'rm -r {all_path_info_w_config.cached_path_wo_sub}')

        os.system(f'rm -r {all_path_info_w_config.ruffus_download_path_wo_sub}')

        # print('done w/ old files')

        return(new_ruffus)

    else: # check if filesystem cache exists

        if check_if_file_exists(all_path_info_w_config.fits_path_w_sub):

            # print('data downloaded w subsys, returning cached dict')

            cached_dict = pickle.load(open(all_path_info_w_config.cached_path_w_sub, 'rb'))

            return(cached_dict)
        
        else:
            while True:

                try:

                    this_wget_link = create_wget_link(all_path_info_w_config)

                    # print(this_wget_link)

                    os.system(this_wget_link)

                    break

                except:
                    print(f'problems w/ {this_wget_link}. Sleeping now')
                    sleep(randint(1,11))

        ##### download complete. verify parsability

        try:

            download_speed, file_size = find_file_size_and_speed_from_wget_log(all_path_info_w_config.wget_log_path_w_sub)

            this_dict = all_path_info_w_config.CONFIG_FILE

            this_dict['wget_log_file'] = all_path_info_w_config.wget_log_path_w_sub

            this_dict['cached_dict_path'] = all_path_info_w_config.cached_path_w_sub

            this_dict['file_path'] = all_path_info_w_config.fits_path_w_sub

            this_dict['file_size_MB'] = file_size

            this_dict['download_speed'] = download_speed

            this_dict['download_string'] = this_wget_link

            # save dictionary in cache directory

            pickle.dump(this_dict, open(all_path_info_w_config.cached_path_w_sub, 'wb'))

            return(this_dict)
        
        except:

            sub_dir = all_path_info_w_config.CONFIG_FILE['file_syst_sub_dir']
            
            ERASE_THESE_FILES = os.path.join(dataconfig.DATA_DIR_IMG_DATA, f'*/{sub_dir}/{all_path_info_w_config.hashed_url}*')

            os.system(f'rm -r {ERASE_THESE_FILES}')

            output = all_path_info_w_config.CONFIG_FILE['file_name']

            # this_file = f'~/jorge-helio/argonne_files/no_space_error_cache/{output}.txt'

            this_file = f'{dataconfig.image}/bad_download_cache/{output}'

            os.system(f'touch {this_file}')

            return(None)
        
def find_closest_valid_datetime(datetime_str):
    # Use regular expression to find the seconds value in the datetime string
    pattern = r'(\d\d)\.\d\dZ'
    match = re.search(pattern, datetime_str)
    
    if match:
        # Check if the seconds value is '60'
        seconds = int(datetime_str[match.start(1):match.end(1)])
        if seconds == 60:
            # Replace the seconds and milliseconds values with '59'
            updated_datetime_str = datetime_str[:match.start(1)] + '59.99Z'
            return updated_datetime_str
        else:
            return datetime_str
    else:
        # If the seconds value is not found, return the original datetime string
        return datetime_str
        
def clean_imgs_for_peakfinder(parsed_dict):

    instrument = parsed_dict['instrument']

    # if config['QUALITY'] == 0:

    if instrument == 'SXI':

        clean_start = time.time()

        load_data_start = datetime.now(cst)

        new_config, raw_data, raw_header = clean_img_data_02.clean_sxi_data(parsed_dict) # open fits sxi chooses the first elelem on the fits format. We can use it here for clean_aia_data
        
        load_data_end = datetime.now(cst)

        print(f'download module w argonne file syst data clean SXI: {load_data_end - load_data_start}')

        clean_end = time.time()

        if new_config['QUALITY'] == 0:

            rotated_data = rotate(np.float32(raw_data), -1*raw_header['CROTA1'])

            data = rotated_data/raw_header['EXPTIME'] # DN/s

        if new_config['QUALITY'] == 10: # data file header corrupt

            
            copy_dict = new_config.copy()

            copy_dict['telescope'] =  'goesXX'

            timestamp = convert_datetime.convert_datetime_to_timestamp(pd.Timestamp('1970-01-01T00:00:00', tz = 'utc'))

            copy_dict['time_stamp'] = timestamp

            dbscan_cluster_eps = 10

            return(copy_dict, raw_data, raw_header, dbscan_cluster_eps)


        else:
            data = raw_data

        dbscan_cluster_eps = 10 # 50 arcseconds

        # r_sun = raw_header['RSUN']

        # arcsec_per_pixel = raw_header['CDELT1']

        T_obs = raw_header['DATEOBS']

        T_OBS_date_time_UTC = pd.Timestamp(f'{T_obs}', tz = 'utc')

        # print(T_OBS_date_time_UTC)

        timestamp = convert_datetime.convert_datetime_to_timestamp(T_OBS_date_time_UTC)

        telescope_dict = {'GOES-15': 'goes15', 'GOES-14': 'goes14', 'GOES-13': 'goes13' }

        copy_dict = new_config.copy()

        copy_dict['telescope'] =  telescope_dict[raw_header['TELESCOP']]

        copy_dict['time_stamp'] = timestamp


    if instrument == 'AIA':

        clean_start = time.time()

        load_data_start = datetime.now(cst)

        new_config, raw_data, raw_header = clean_img_data_02.clean_aia_data(parsed_dict)

        load_data_end = datetime.now(cst)

        print(f'download module w argonne file syst data clean AIA: {load_data_end - load_data_start}')

        clean_end = time.time()

        data = raw_data

        # r_sun = raw_header['R_SUN']

        dbscan_cluster_eps = 83 # 50 arcseconds

        # arcsec_per_pixel = raw_header['CDELT1']

        copy_dict = new_config.copy()

        copy_dict['telescope'] =  'SDO'


        # check if there is a 60 in the seconds of the date_time_string
        date_time_string = find_closest_valid_datetime(raw_header['T_obs'])

        T_OBS_date_time_UTC = pd.Timestamp(date_time_string)

        timestamp = convert_datetime.convert_datetime_to_timestamp(T_OBS_date_time_UTC)

        copy_dict['time_stamp'] = timestamp

    
    return(copy_dict, data, raw_header, dbscan_cluster_eps)
    
def single_image_peakfinder(new_config, data, raw_header, dbscan_cluster_eps, clean_time):

    centroid_array = []

    copy_dict = new_config.copy()

    if new_config['QUALITY'] == 0:

        arcsec_per_pixel = raw_header['CDELT1']

        peak_start = time.time()

        load_data_start = datetime.now(cst)

        coordinates = peak_local_max(data, threshold_rel = .9)

        load_data_end = datetime.now(cst)

        instrument = copy_dict['instrument']

        print(f'peakfind download module w/ argonne filesys {instrument}: {load_data_end - load_data_start}')

        peak_end = time.time()

        peakfinder_time = peak_end - peak_start

        if len(coordinates) != 0:

            # analogous to matrixes, [rows, columns] == [y,x] ==> coordinates[:,1] will give the x values
            # coordinates[:,0] will give the y values

            peakfinder_array = np.array( [ [x,y] for x,y in zip(coordinates[:,1],coordinates[:,0]) ] )

            dbscan_start = time.time()

            # define the dbscan model
            # all the peaks from each image will be clustered to return n-number of peaks
            # peaks need be more than apart to be considered "outside" the dbscan radius
            # 50 arcseconds (or 83 pixels for AIA ; 10 pixels for SXI)

            load_data_start = datetime.now(cst)

            dbscan_model = DBSCAN(eps = dbscan_cluster_eps, min_samples=1)

            # train the model
            dbscan_model.fit(peakfinder_array)

            dbscan_result = dbscan_model.fit_predict(peakfinder_array)

            load_data_end = datetime.now(cst)

            print(f'dbscan download module w/ argonne filesys {instrument}: {load_data_end - load_data_start}')

            dbscan_end = time.time()
            
            dbscan_time = dbscan_end - dbscan_start

            # find unique labels in dbscan_result

            all_labels = np.unique(dbscan_result)

            # drop outliers if part of all_labels. outliers are returned by
            # the DBSCAN  defined as == float(-1)

            good_labels = all_labels[all_labels != -1]

            for label in good_labels:

                copy_dict = new_config.copy()

                which_pixels = np.where(dbscan_result == label)

                num_of_pix_members_in_cluster = len(peakfinder_array[which_pixels][:,0])

                x_mean = np.mean(peakfinder_array[which_pixels][:,0])
                y_mean = np.mean(peakfinder_array[which_pixels][:,1])

                # data_map = sunpy.map.Map(data, raw_header)

                # these_pixels_work = coordinate_conversion_module.algorithm_to_project_pixel_onto_surface(x_mean, y_mean, outer_perimiter, data_map)

                # print(these_pixels_work, x_mean, y_mean)

                # sky = WCS(data_map.fits_header).pixel_to_world(these_pixels_work[0], these_pixels_work[1])

                # # sky = WCS(data_map.fits_header).pixel_to_world(x_mean, y_mean)


                # sky_x_arc, sky_y_arc = sky.Tx, sky.Ty

                # x_hgs, y_hgs = coordinate_conversion_module.hpc_to_hgs([sky_x_arc.value, sky_y_arc.value], data_map)

                x_hpc = (x_mean - raw_header['CRPIX1']) * arcsec_per_pixel

                y_hpc = (y_mean - raw_header['CRPIX2']) * arcsec_per_pixel


                # print(sky_x_arc.value, sky_y_arc.value, these_pixels_work[0], these_pixels_work[1], x_hgs, y_hgs)

                # sky_skycoord_obj = SkyCoord(sky_x_arc.Tx.value, sky_y_arc.Tx.value, frame = data_map.coordinate_frame)

                # hpc_x, hpc_y = sky_skycoord_obj.Tx.value, sky_skycoord_obj.Ty.value


                # , sky_x_arc.value, these_pixels_work[0], these_pixels_work[1], x_hgs, y_hgs

                

                copy_dict.update({
                                'dbscan_1_x_hpc': x_hpc,
                                'dbscan_1_y_hpc': y_hpc,
                                'dbscan_1_x_pix': x_mean,
                                'dbscan_1_y_pix': y_mean,
                                'dbscan_1_num_members': num_of_pix_members_in_cluster,
                                'dbscan_1_label': label, 
                                'peakfinder_time_1': peakfinder_time,
                                'dbscan_time_1': dbscan_time, 
                                'cleaning_time_1': clean_time,
                                'peaks_found': True})

                centroid_array.append(copy_dict)
        else:
            #returns this dictionary if there are no peaks in this image
            copy_dict.update({
            'dbscan_1_x_hpc': np.nan,
            'dbscan_1_y_hpc': np.nan,
            'dbscan_1_x_pix': np.nan,
            'dbscan_1_y_pix': np.nan,
            'dbscan_1_num_members': np.nan,
            'dbscan_1_label': np.nan,
            'peakfinder_time_1': np.nan, 
            'dbscan_time_1': np.nan, 
            'cleaning_time_1': clean_time, 
            'peaks_found': False})

            centroid_array.append(copy_dict)

    else:
        # returns this dictionary if QUALITY of passed image not good
        # this, in theory, should never happen because we only pass 
        # QUALITY == 0 and data_level = lev2 data into the ALEXIS pipeline
        copy_dict.update({
        'dbscan_1_x_hpc': np.nan,
        'dbscan_1_y_hpc': np.nan,
        'dbscan_1_x_pix': np.nan,
        'dbscan_1_y_pix': np.nan,
        'dbscan_1_num_members': np.nan,
        'dbscan_1_label': np.nan, 
        'peakfinder_time_1': np.nan, 
        'dbscan_time_1': np.nan, 
        'cleaning_time_1': clean_time,
        'peaks_found': False})

        centroid_array.append(copy_dict)

    return(pd.DataFrame(centroid_array))