import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
from glob import glob
import re
import pickle
import numpy as np
import os
import pandas as pd
# from ruffus.task import fill_queue_with_job_parameters
from sunpy.time import TimeRange
import time
from datetime import datetime, timedelta
import sqlalchemy as sql
import re
import warnings

# from create_aia_12s_availability import DATA_PRODUCT_DIR
warnings.filterwarnings("ignore")
from astropy.io import fits
from skimage.feature import peak_local_max
from ruffus import *
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from scipy.stats import linregress

from skimage.transform import rotate



import matplotlib.pyplot as plt
from astropy.wcs import WCS
import sunpy.map
from astropy import units as u
from scipy.signal import find_peaks

import sys

# import associate_HARP_to_flare_region_module

from astropy.coordinates import SkyCoord

import scipy.stats
from scipy.stats import mstats
from sklearn.metrics import mean_squared_error
from modules import query_the_data

# import aia_module

import dataconfig

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from time import sleep
from random import randint

from aiapy.calibrate.util import get_correction_table, get_pointing_table

from modules import helio_reg_exp_module

import itertools
from scipy.optimize import nnls

import cvxpy as cp

from sunpy.coordinates import frames


from sklearn.metrics import mean_squared_error 
import ast
import itertools


from scipy.spatial import KDTree

import itertools
import scipy.stats
from scipy.stats import mstats
from sklearn.metrics import mean_squared_error


from modules import sxi_module
from modules import clean_img_data_02
from modules import convex_fits_and_filtering_module
from modules import convert_datetime
from modules import associate_candidates_to_flare_meta_module
from modules import LASSO_metrics_module
from modules import ALEXIS_03_create_hek_report_module
from modules import ALEXIS_02_define_ALEXIS_flares_module
from modules import ALEXIS_02_associate_flare_to_harp_module
from modules import ALEXIS_02_define_goes_class_module
from modules import coordinate_conversion_module






# import associate_HARP_to_flare_region_module


# import check_data_qual_module_02
# need to bring to github












# import ALEXIS_02_plotting_module_movie


###################################################################################################
###################################################################################################

print('hello')

# files with good data have been created. Start Ruffus Pipeline

initialization_files = glob(f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/*/initialize_hyperspectral_with_these_files_df.pickle')

print(f'doing {len(initialization_files)}')


# print(initialization_files)
# pipeline_run([peakfinder_clustering_per_frame], multithread = 15)


# # @collate(peakfinder_clustering_per_frame, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working*'), output = '{subpath[0][0]}/interesting_pixels.pickle')
# @collate(peakfinder_clustering_per_frame, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working*'), output = '{subpath[0][0]}/v3_interesting_pixels.pickle')
@transform(initialization_files, suffix('initialize_hyperspectral_with_these_files_df.pickle'), 'v3_interesting_pixels.pickle' )
def time_avg_and_hyperspectral_clustering(infile, outfile):

    """
    Collates all the dictionaries created from peakfinder and single image dbscan.

    Applies temporal and hyperspectral clustering. 
    """

    # print(infile, outfile)

    # peakfinder_df = pd.concat([pd.DataFrame(pickle.load(open(peakfinder_dict,'rb'))) for peakfinder_dict in infiles])
    peakfinder_df = pickle.load(open(infile, 'rb'))

    df = peakfinder_df.dropna().sort_values(by = 'date_time').reset_index(drop = True)

    avg_dbscan_time = np.mean(df['dbscan_time_1'])

    avg_peakfinder_time = np.mean(df['peakfinder_time_1'])

    avg_clean_time = np.mean(df['cleaning_time_1'])

    start_time = time.time()

    ##############################################################################
    ##################### temporal dbscan start ##################################

    # keep track of temporal dbscan list
    timeseries_clusters_dict = []

    for temporal_label, temporal_group in df.groupby(['wavelength', 'instrument', 'telescope']):

        # a = time.time()

        fit_this_df = temporal_group

        # temporal_group

        if len(fit_this_df) > 5: #only use wl that have more than 5 data points to work from

            peakfinder_array = fit_this_df[['dbscan_1_x_hpc', 'dbscan_1_y_hpc']].values

            # the coordinates of a cluster must have survived for at least 15% of the images
            # aka: if 200 AIA images are used, the cluster must have survived at least 
            # AIA: (200*.15) * 12 seconds = 6 min
            # SXI: (20*.15) * 5 min = 20 min

            num_of_data_entries = int(len(fit_this_df.date_time.unique())*.15)

            # define the model
            # we are now working in hpc space
            # dbscan will look for object 50 arcseconds apart
            dbscan_model = DBSCAN(eps=50, min_samples=num_of_data_entries)
            # print(num_of_data_entries, wavelength)

            # train the model
            dbscan_model.fit(peakfinder_array)

            dbscan_result = dbscan_model.fit_predict(peakfinder_array)

            # find unique labels in dbscan_result

            all_labels = np.unique(dbscan_result)

            # drop outliers if part of all_labels. outliers are returned by
            # the DBSCAN  defined as == float(-1)

            good_labels = all_labels[all_labels != -1]

            for label in good_labels:
            
                which_pixels = np.where(dbscan_result == label)

                cluster_df = fit_this_df.iloc[np.where(dbscan_result == label)]

                x_mean = np.mean(peakfinder_array[which_pixels][:,0])
                y_mean = np.mean(peakfinder_array[which_pixels][:,1])

                timeseries_clusters_dict.append({'wavelength': temporal_label[0], 
                                                'instrument': temporal_label[1],
                                                'telescope': temporal_label[2],
                                                    'hpc_x': x_mean,
                                                    'hpc_y': y_mean, 
                                                    'label': label, 
                                                    'passing_vote': num_of_data_entries, 
                                                    'num_votes': len(cluster_df), 
                                                    'num_of_members': len(fit_this_df)})

    #################temporal dbscan end ####################
    ########################################################

    ########### hyperspectral dbscan start ####################
    ########################################################

    # create df with results from all WL temporal dbscan
    hyper_spectral_df = pd.DataFrame(timeseries_clusters_dict)
    hyper_spectral_df['vote_ratio'] = hyper_spectral_df.num_votes/hyper_spectral_df.num_of_members

    # create array of hpc coords for the possible clusters
    peakfinder_array_hyper = hyper_spectral_df[['hpc_x', 'hpc_y']].values


    # 66% of available wavelength are needed such that a cluster 
    # is defined hyperspectrally
    max_wl_avail = int( len(hyper_spectral_df.wavelength.unique()) * (2/3))

    # NOTE: patch for when when there is only 1 available WL to do clustering, 
    # change the max allowed from zero  to 1

    if max_wl_avail == 0:
        max_wl_avail = 1

    # End patch for min number of voting WL

    ## define the hyperspectral model
    dbscan_model2 = DBSCAN(eps=50, min_samples=max_wl_avail)

    ## train the model
    dbscan_model2.fit(peakfinder_array_hyper)

    dbscan_result2 = dbscan_model2.fit_predict(peakfinder_array_hyper)

    # # find unique labels in dbscan_result

    all_labels2 = np.unique(dbscan_result2)

    # # drop outliers if part of all_labels. outliers are returned by
    # # the DBSCAN  defined as == float(-1)

    good_labels2 = all_labels2[all_labels2 != -1]

    hyper_spectral_result = []

    for hyper_label in good_labels2:
        

        hyper_cluster_df = hyper_spectral_df.iloc[np.where(dbscan_result2 == hyper_label)]


        avail_df = hyper_spectral_df[hyper_spectral_df.label == 0].reset_index(drop = True)

        x_hpc, y_hpc = np.mean(hyper_cluster_df.hpc_x), np.mean(hyper_cluster_df.hpc_y)

        hyper_cluster_df['x_hpc'] = x_hpc
        hyper_cluster_df['y_hpc'] = y_hpc

        hyper_cluster_df['cluster_label'] = hyper_label

        end_time = time.time()

        hyper_clustering_time = end_time - start_time

        hyper_spectral_result.append({'x_hpc':x_hpc, 
                                    'y_hpc': y_hpc, 
                                    'label': hyper_label,
                                    'avail_wls': avail_df['wavelength'].to_list(),
                                    'avail_tel' : avail_df['telescope'].to_list(),
                                    'avail_inst' : avail_df['instrument'].to_list(),
                                    'voting_wl': hyper_cluster_df['wavelength'].to_list(),
                                    'voting_tel' : hyper_cluster_df['telescope'].to_list(),
                                    'voting_inst' : hyper_cluster_df['instrument'].to_list(),
                                    'working_dir': df.iloc[0].working_dir,
                                    'passing_votes': max_wl_avail, 
                                    'avail_votes': len(avail_df),
                                    'temporal_vote_ratio': hyper_cluster_df.vote_ratio.to_list(),
                                    'temporal_vote_number': hyper_cluster_df.num_votes.to_list(), 
                                    'temporal_vote_members': hyper_cluster_df.num_of_members.to_list(),
                                    'temporal_passing_vote': hyper_cluster_df.passing_vote.to_list(),
                                    'num_votes': len(hyper_cluster_df), 
                                    'avg_dbscan_time_1': avg_dbscan_time, 
                                    'avg_peakfinder_time_1': avg_peakfinder_time, 
                                    'dbscan_time_2': hyper_clustering_time,
                                    'avg_clean_time_peakfinder': avg_clean_time})
        # outputs a list of dictionaries for all the interesting clusters

    pickle.dump(hyper_spectral_result, open(outfile, 'wb'))


# pipeline_run([time_avg_and_hyperspectral_clustering], multithread = 15)

# #######################################################################


@subdivide(time_avg_and_hyperspectral_clustering, formatter(),
            # Output parameter: Glob matches any number of output file names
            "{path[0]}/{basename[0]}.*.v3_initialize_cluster_flux.pickle",
            # Extra parameter:  Append to this for output file names
            "{path[0]}/{basename[0]}")
def initialize_flux_per_cluster_found(infile, outfiles, output_file_name_root):

    # start_time = time.time()

    investigate_these_pixels = pd.DataFrame(pickle.load(open(infile, 'rb')))

    working_dir = investigate_these_pixels.iloc[0].working_dir

    good_quality_data_df = pickle.load(open(f'{working_dir}/initialize_with_these_files_df.pickle', 'rb'))

    all_dfs = []

    for cluster, cluster_group in investigate_these_pixels.groupby(['label']):

        good_qual_copy = good_quality_data_df.copy()

        good_qual_copy['x_hpc'] = [cluster_group['x_hpc'].iloc[0] for _ in good_qual_copy.file_path]

        good_qual_copy['y_hpc'] = [cluster_group['y_hpc'].iloc[0] for _ in good_qual_copy.file_path]

        good_qual_copy['final_cluster_label'] = [cluster for _ in good_qual_copy.file_path]

        all_dfs.append(good_qual_copy)

    concat_df = pd.concat(all_dfs).sort_values(by = 'date_time')

    for labels, config_df in concat_df.groupby(['telescope','instrument', 'wavelength']):

        telescope, instrument, wl = labels[0],labels[1],labels[2]

        # initialize_start = time.time()

        if instrument == 'AIA':
            wl = int(wl)

        ordered_df = config_df.sort_values(by = 'date_time').reset_index(drop = True)

        for num, date_time in enumerate(ordered_df.date_time.unique()):

            masked_df = ordered_df[ordered_df.date_time == date_time]

            output_file_name = f'{output_file_name_root}.{telescope}.{instrument}.{wl}.cluster.{num}.v3_initialize_cluster_flux.pickle'

            # output a dataframe with number of rows == number of clusters
            # each dataframe is for 1 file only. 

            pickle.dump(masked_df, open(output_file_name, 'wb'))

# pipeline_run([initialize_flux_per_cluster_found], multithread = 15)
 

    
@transform(initialize_flux_per_cluster_found, suffix('.v3_initialize_cluster_flux.pickle'), '.v3_cluster_flux_timeseries.pickle')
def find_flux_timeseries(infile, outfile):

    clean_start = time.time()

    config_df = pickle.load(open(infile, 'rb'))

    good_qual = config_df[config_df.QUALITY == 0]

    #choose first element in to load data 

    config = good_qual.to_dict('records')[0]

    instrument = config['instrument']

    #### OPEN IMAGE AND LOAD CLEAN IMG

    if instrument == ('SXI'):

        _, raw_data, raw_header = clean_img_data_02.clean_sxi_data(config) # open fits sxi chooses the first elelem on the fits format. We can use it here for clean_aia_data

        rotated_data = rotate(np.float32(raw_data), -1*raw_header['CROTA1'])

        data = rotated_data/config['exp_time']

        # data_map = sunpy.map.Map(data, raw_header)

        arcsec_per_pixel = raw_header['CDELT1']

        mask_pixel_distance_list = [15,10,8,5]

        bbox_linear_size_dict = {15: 150, 10: 100, 8: 80, 5: 50}

    ##############################################################################
        # mask_pixel_distance = 15 * 5arcs/pix = 75 arcsec for 1/2 linear size ==> 150x150 bbox
        # mask_pixel_distance = 10 # 100x100 bbox
        # mask_pixel_distance = 8 # 80 x 80 bbox
        #mask_pixel_distance = 5 # 50 x 50 bbox
    #############################################################################

        r_sun = raw_header['RSUN']

        # print(r_sun)

    if instrument == 'AIA':

        # if i dont want to ping jsoc many times i can 
        # query the clean up files outside the loop and then not 
        # load it again for each different WL

        _, raw_data, raw_header = clean_img_data_02.clean_aia_data(config)

        data = raw_data

        arcsec_per_pixel = raw_header['CDELT1']

        # data_map = sunpy.map.Map(data, raw_header)


        ##############################################################################
        # mask_pixel_distance = 85 # 85 pix*.6arcsec/pixel = 50arcs 1/2 linear size ==> ( 100x 100 bbox)
        # mask_pixel_distance = 67 # 80 x 80 bbox
        #mask_pixel_distance = 41 # 50 x 50 bbox
        # mask_pixel_distance = 125 # 125 pix * .6 arcsec/pix = 75 arcseconds 1/2 linear size ==> (150x150 bbox)

        mask_pixel_distance_list = [125,85,67,41]

        bbox_linear_size_dict = {125: 150, 85: 100, 67: 80, 41: 50}

        # mask_pixel_distance = 85 # 85 pix*.6arcsec/pixel = 

        #############################################################################


        r_sun = raw_header['R_SUN']

        ####  analysis function
    ####################################################################

    pixel_list = []

    for cluster_label, group in good_qual.groupby(['final_cluster_label']):

        x_hpc, y_hpc = group['x_hpc'].iloc[0], group['y_hpc'].iloc[0]

        # if we use the coord conversion from WCS it takes 0.04 seconds per coord pair
        # original_cluster_hpc_skycoord_obj = SkyCoord(x_hpc*u.arcsec, y_hpc*u.arcsec, frame = data_map.coordinate_frame)
        # original_pixels2 = WCS(data_map.fits_header).world_to_pixel(original_cluster_hpc_skycoord_obj)

        # print(original_pixels2)

        # if we use the coord conversion from us it takes 0.0005 seconds per coord pair
        # we are off the WCS conversion by +1 pixel
        original_pixels = (np.array((x_hpc * (arcsec_per_pixel)**-1) + raw_header['CRPIX1']), np.array((y_hpc * (arcsec_per_pixel)**-1) + raw_header['CRPIX2']))

        # print(original_pixels)

        pixel_list.append({'cluster_label': cluster_label, 'pixel_array': original_pixels})


    pixels_df = pd.DataFrame(pixel_list)

    def return_flux_dict_list(good_qual, mask_pixel_distance):

        flux_dict_list = [] 

        
        for candidate_label, group in good_qual.groupby(['final_cluster_label']):


            # check to see if coords returned are w/n solar boundaries

            # these_pixels_work = coordinate_conversion_module.algorithm_to_project_pixel_onto_surface(original_pixels[0], original_pixels[1], outer_perimiter,  data_map)



            pixel_mask = pixels_df[pixels_df.cluster_label == candidate_label]

            original_pixels = pixel_mask.pixel_array.iloc[0]

            square_borders_for_mask = sxi_module.make_square_region_v2(original_pixels, mask_pixel_distance) # we make a square with the pixel distances defined above

            array = np.array(square_borders_for_mask)

            x_array = array[:,0]
            y_array = array[:,1]

            x_max, x_min = x_array.max(), x_array.min()

            y_max, y_min = y_array.max(), y_array.min()

            # this_mask = sxi_module.create_region_mask_shapeless(data, square_borders_for_mask) # we can also feed in any polynomial and it will create a mask with the inputs as borders

            flux = np.sum(data[int(y_min):int(y_max),int(x_min):int(x_max)])


            copy_config = group.copy()

            copy_config['region_flux'] = flux

            copy_config['x_pix'] = original_pixels[0]
            copy_config['y_pix'] = original_pixels[1]

            # copy_config['x_hgs'] = these_hgs_work[0]
            # copy_config['y_hgs'] = these_hgs_work[1]

            copy_config['x_hpc'] = copy_config.x_hpc
            copy_config['y_hpc'] = copy_config.y_hpc

            copy_config['integration_pixel_bbox'] = [square_borders_for_mask]

            copy_config['integration_hpc_bbox'] = [coordinate_conversion_module.pix_to_hpc(square_borders_for_mask, raw_header)]

            copy_config['bbox_linear_size_arcsec'] = bbox_linear_size_dict[mask_pixel_distance]

            copy_config['bbox_half_linear_size_pix'] = mask_pixel_distance

            # pass_these_dict = copy_config.to_dict('records')

            flux_dict_list.append(copy_config)


        return(flux_dict_list)


    # end of analiysis function

    # list comprehension

    final_df = pd.concat([pd.concat(return_flux_dict_list(good_qual, pixel_distance)) for pixel_distance in mask_pixel_distance_list])

    b = time.time()

    full_time = b - clean_start

    final_df['find_flux_timeseries_run_time'] = [full_time for _ in final_df.region_flux]

    pickle.dump(final_df, open(outfile, 'wb'))



# pipeline_run([find_flux_timeseries], multithread = 15)


#     #######################################################################


# # @collate(find_flux_timeseries, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working\/interesting_pixels.[A-Z0-9]{1,}.[A-Za-z0-9]{1,}.[A-Z0-9]{1,}.cluster.[0-9]{1,}.step2'), output = '{subpath[0][0]}/clusters_flux_df.pickle')
@collate(find_flux_timeseries, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working\/v3_interesting_pixels.[A-Za-z0-9]{1,}.[A-Za-z0-9]{1,}.[A-Z0-9]{1,}.cluster.[0-9]{1,}.v3_cluster_flux_timeseries.pickle'), output = '{subpath[0][0]}/clusters_flux_df_v3.pickle')
def join_flux_per_wl_and_cluster(infiles, outfile):

    """
    Here we agregate all the flux from all the images available in this particular working directory.

    Input: list of dataframes w/ config info and region flux as well as x_pix and y_pix

    input columns == > ['download_string', 'time_stamp', 'data_level', 'file_name', 'exp_time',
       'file_size_MB', 'QUALITY', 'wavelength', 'url', 'instrument',
       'file_path', 'wget_log_file', 'download_speed', 'cached_dict_path',
       'telescope', 'img_data_max', 'img_data_min', 'img_flux',
       'cleaning_time', 'date_time', 'working_dir', 'x_hpc', 'y_hpc',
       'final_cluster_label', 'region_flux', 'x_pix', 'y_pix',
       'integration_pixel_bbox', 'integration_hpc_bbox',
       'bbox_linear_size_arcsec', 'bbox_half_linear_size_pix',
       'find_flux_timeseries_run_time'],

    ### After we concat, we will be left with a dataframe with calculated flux
    ### for all clusters specific to a working_dir.. 

    OUTPUT: DataFrame

    ##### NOTE ######

    # "{working_dir}/clusters_flux_df.pickle"
    # contains all the timeseries for the imgs and clusters
    
    """
    # print(len(infiles), outfile)

    df = pd.concat([pickle.load(open(data,'rb')) for data in infiles])

    out_put_df_list = []
    for _, config_df in df.groupby(['telescope','instrument', 'wavelength', 'final_cluster_label', 'bbox_linear_size_arcsec']):

        copy_df = config_df.copy()

        if len(copy_df) >= 10:

            out_put_df_list.append(copy_df)

        else:
            #quality flag 9 == less than 10 datapoints
            copy_df['QUALITY'] = 9
            out_put_df_list.append(copy_df)

    outfile_df = pd.concat(out_put_df_list)
            
    pickle.dump(outfile_df, open(outfile, 'wb'))
            
# pipeline_run([join_flux_per_wl_and_cluster], multithread = 15)


@jobs_limit(1)
# @subdivide(join_flux_per_wl_and_cluster, formatter(),
#             # Output parameter: Glob matches any number of output file names
#             "{path[0]}/{basename[0]}.*.xrs_resampled_df",
#             # Extra parameter:  Append to this for output file names
#             "{path[0]}/{basename[0]}")
# @transform(join_flux_per_wl_and_cluster, suffix('clusters_flux_df.pickle'), 'resampled_XRS_df.pickle')
@transform(join_flux_per_wl_and_cluster, suffix('clusters_flux_df_v3.pickle'), 'resampled_XRS_df_v3.pickle')
# output_file_name = f"{output_file_name_root}.{telescope}.{instrument}.{wl}.cluster{final_cluster_label}.step3"
def resample_XRS_data(infile, outfile):
    """
     This function queries the XRS database for all available data that covers the span
     of time available from cluster_flux.pickle. 

     SQLite doesn't play well with multi-threading. @jobs_limit(1) decorator used. ==> Possible bottleneck.

    ######
    input == df with flux of all available clusters for each available img

    outfile == df w/ resampled_xray @ 1 min cadence for each XRS wl/inst pair

    ouput columns:
        ['resampled_date_time', 'resampled_time_stamp', 'xrs_resampled_data',
       'xrs_wavelength', 'xrs_instrument', 'xrs_telescope', 'working_dir',
       'cluster_flux_file', 'xrs_resample_file', 'xrs_resample_run_time']
    
    """

    start_time = time.time()

    previous_df = pickle.load(open(infile, 'rb')).sort_values(by = 'date_time') # order all the cluster flux for this working_dir by date_time

    good_qual_data = previous_df[(previous_df.QUALITY == 0)].sort_values(by = 'date_time')

    # # What are the limits of ALL the data that we have available in terms of images?

    df = good_qual_data.copy()


    avail_clusters_datetime_min, avail_clusters_datetime_max = good_qual_data.date_time.iloc[0], good_qual_data.date_time.iloc[-1]

    # # lets make a list of these limits resampled at specified cadence
    # freq = '12s'

    # ticks_to_use = pd.date_range(avail_clusters_datetime_min, end =  avail_clusters_datetime_max, freq = freq, tz = 'utc' ).tolist()



    # query XRS SQLite DB for all XRS data accross the available time. 

    xrs_flux = query_the_data.xray_sql_db_timea_to_timeb(avail_clusters_datetime_min, avail_clusters_datetime_max)

    # find how many data entries we have a keep track of inst, wl, and number of entries
    num_of_data_entries = []

    for label2, group2 in xrs_flux.groupby(['instrument', 'wavelength']): # XRS db has instrument == 'goes##', yelp. Change later?
                    
            telescope, instrument, wl,  = label2[0], 'XRS', label2[1]

            example2 = group2.sort_values(by = 'date_time').copy() #copy the original df

            num_of_data_entries.append({'tel': telescope, 'inst': instrument, 'wl': wl, 'len': len(example2)})

    #create a dataframe for the results of previous loop
    lengths_of_data = pd.DataFrame(num_of_data_entries)

    #find max of entries
    max_len = np.max(lengths_of_data.len)

    # find how many data points each wl/inst is missing
    xrs_data_differentiator = [max_len - this_len for this_len in lengths_of_data.len]

    # filter the xrs data to only return good-long quality data
    truth_array_length = [val < 150 for val in xrs_data_differentiator] # will only allow 6 minutes worth of gaps in data

    # boolean mask of original xrs data with inst/wl of good data available 

    surviving_xrs = (lengths_of_data[truth_array_length])


    #create the output dataframe masking only for inst/wl that are good
    good_xrs_list_of_df = []
    for label, _ in surviving_xrs.groupby(['tel', 'wl']): # we only need the label to mask the original xrs_flux dataframe

        tel, wl = label[0], label[1]

        this_good_xrs_flux = xrs_flux[(xrs_flux.instrument == tel) & (xrs_flux.wavelength == wl)]

        sorted_good_xrs = this_good_xrs_flux.sort_values(by = 'date_time').reset_index(drop = True)

        good_xrs_list_of_df.append(sorted_good_xrs)

    # concat the resulting DF from above into one DF    
    good_xrs_raw_flux_data = pd.concat(good_xrs_list_of_df)


    # now we have good data, lets resample. 

    resampled_xrs_list = []

    for resample_label, group2 in good_xrs_raw_flux_data.groupby(['instrument', 'wavelength']):

            instrument, telescope, wl,  = 'XRS',resample_label[0],resample_label[1]

            this_mask  = xrs_flux[(xrs_flux.instrument == telescope) & (xrs_flux.wavelength == wl)].sort_values(by = 'date_time').reset_index(drop = True)

            fit_raw_data = sxi_module.fit_the_data(this_mask.time_stamp.to_list(), this_mask.value.to_list())

            resample_time_start, resample_time_end = this_mask.date_time.iloc[0], this_mask.date_time.iloc[-1]

            # resampling_rate_df returns df w/ two columns: "resampled_time_stamp" & "resampled_date_time"
            resampling_rate_df = sxi_module.create_resampling_rate_df(resample_time_start, resample_time_end ) #default resampling rate is 1 min. If change specify the "resample_freq" argument


            resampled_xrs= sxi_module.resample_fitted_data(resampling_rate_df.resampled_time_stamp.to_list(), fit_raw_data)
            # resampled_full_disk = sxi_module.resample_fitted_data(resampling_rate_df.resampled_time_stamp.to_list(), fit_raw_data)

            resampling_rate_df['xrs_resampled_data'] = resampled_xrs

            resampling_rate_df['xrs_wavelength'] = wl 
            resampling_rate_df['xrs_instrument'] = instrument
            resampling_rate_df['xrs_telescope'] = telescope

            #keep track of some metadata from cluster flux and output file path
            resampling_rate_df['working_dir'] = df.working_dir.iloc[0]

            resampling_rate_df['cluster_flux_file'] = infile

            resampling_rate_df['xrs_resample_file'] = outfile


            ordered_df = resampling_rate_df.sort_values(by = 'resampled_date_time')

            resampled_xrs_list.append(ordered_df)

    #concat the resampled good quality xrs data
    output_df = pd.concat(resampled_xrs_list)

    end_time = time.time()

    xrs_resample_run_time = end_time - start_time

    output_df['xrs_resample_run_time'] = [xrs_resample_run_time for _ in output_df.working_dir]


    # save outputfile
    pickle.dump(output_df, open(outfile, 'wb'))

# pipeline_run([resample_XRS_data], multithread = 15)


# @transform(resample_XRS_data, suffix('resampled_XRS_df.pickle'), 'resampled_img_df.pickle')
@transform(resample_XRS_data, suffix('resampled_XRS_df_v3.pickle'), 'resampled_img_df_v3.pickle')
def resample_imgs(infile, outfile):

    """
    Takes in resampled XRS and finds what img data we have 
    """

    start_time = time.time()


    #load the resampled_xrs_data
    xrs_resampling_df = pickle.load(open(infile, 'rb')).sort_values(by = 'resampled_date_time')

    # load cluster fluxs

    these_cluster_flux = pd.DataFrame(pickle.load(open(xrs_resampling_df.cluster_flux_file.iloc[0], 'rb'))).sort_values(by = 'date_time')


    # mask for quality of data. This mask returns a df where img data contains more than 10 data points

    good_qual_data = these_cluster_flux[(these_cluster_flux.QUALITY == 0)].sort_values(by = 'date_time')


    #keep track of resampled_imgs
    all_resampled_df_list = []

    for img_label, img_data in good_qual_data.groupby(['telescope', 'instrument', 'wavelength', 'final_cluster_label',  'bbox_linear_size_arcsec']):

        tel, inst, wl, cluster_label, bbox_linear_size_arcsec = img_label[0], img_label[1], img_label[2], img_label[3], img_label[4]

        copy_img_data = img_data.sort_values(by = 'date_time')

        for _, xrs_data in xrs_resampling_df.groupby(['xrs_instrument', 'xrs_wavelength', 'xrs_telescope']):

            copy_xrs_data = xrs_data

            # copy_xrs_data

            xrs_resampling_df_mask = copy_xrs_data[(copy_xrs_data.resampled_date_time >= copy_img_data.date_time.iloc[0])
                                                                        & (copy_xrs_data.resampled_date_time <= copy_img_data.date_time.iloc[-1])]

            xrs_timestamp_list = xrs_resampling_df_mask.resampled_time_stamp.to_list()


            # fit the data
            fit_raw_data_cluster = sxi_module.fit_the_data(copy_img_data.time_stamp.to_list(), copy_img_data.region_flux.to_list())

            fit_raw_data_full_disk = sxi_module.fit_the_data(copy_img_data.time_stamp.to_list(), copy_img_data.img_flux.to_list())

            # resample the raw data with the xrs_resampled_time_stamp columns

            resampled_data_cluster = sxi_module.resample_fitted_data(xrs_timestamp_list, fit_raw_data_cluster) #cluster specific flux

            resampled_data_full_disk = sxi_module.resample_fitted_data(xrs_timestamp_list, fit_raw_data_full_disk)# full disk flux

            #keep track of xrs_inst/xrs_wl - img_wl, img_inst, img_tel. candidate_label

            # print(len(resampled_data_full_disk), len(xrs_resampling_df_mask), len(xrs_timestamp_list),xrs_label, img_label)
        

            xrs_resampling_df_mask['img_resampled_value_cluster'] = resampled_data_cluster

            xrs_resampling_df_mask['img_resampled_value_full_disk'] = resampled_data_full_disk

            # make note of some metadata

            xrs_resampling_df_mask['img_instrument'] = inst

            xrs_resampling_df_mask['img_wavelength'] = wl

            xrs_resampling_df_mask['img_telescope'] = tel

            xrs_resampling_df_mask['final_cluster_label'] = cluster_label


            # keep track of flare candidate metadata

            xrs_resampling_df_mask['x_hpc'] = copy_img_data.x_hpc.iloc[0]

            xrs_resampling_df_mask['y_hpc'] = copy_img_data.y_hpc.iloc[0]

            xrs_resampling_df_mask['x_pix'] = copy_img_data.x_pix.iloc[0]

            xrs_resampling_df_mask['y_pix'] = copy_img_data.y_pix.iloc[0]

            xrs_resampling_df_mask['y_pix'] = copy_img_data.y_pix.iloc[0]

            xrs_resampling_df_mask['integration_pixel_bbox'] = [copy_img_data.integration_pixel_bbox.iloc[0] for _ in xrs_resampling_df_mask.resampled_date_time]

            xrs_resampling_df_mask['integration_hpc_bbox'] = [copy_img_data.integration_hpc_bbox.iloc[0] for _ in xrs_resampling_df_mask.resampled_date_time]

            # xrs_resampling_df_mask['x_hgs'] = copy_img_data.x_hgs.iloc[0]

            # xrs_resampling_df_mask['y_hgs'] = copy_img_data.y_hgs.iloc[0]


            xrs_resampling_df_mask['working_dir'] = xrs_resampling_df.working_dir.iloc[0]

            xrs_resampling_df_mask['all_resampled_data_file'] = outfile

            xrs_resampling_df_mask['bbox_linear_size_arcsec'] = bbox_linear_size_arcsec

            xrs_resampling_df_mask['bbox_half_linear_size_pix'] = copy_img_data.bbox_half_linear_size_pix.iloc[0]


            all_resampled_df_list.append(xrs_resampling_df_mask)


    # concat all resampled dataframes into a single dataframe

    output_df = pd.concat(all_resampled_df_list)

    end_time = time.time()

    resample_imgs_run_time = end_time - start_time

    output_df['resample_imgs_run_time'] = [resample_imgs_run_time for _ in output_df.working_dir]

    pickle.dump(output_df, open(outfile, 'wb'))

# pipeline_run([resample_imgs], multithread = 15)


# pipeline_run([resample_imgs], multithread = 15)
@subdivide(resample_imgs, formatter(),
            # Output parameter: Glob matches any number of output file names
            "{path[0]}/{basename[0]}.*.v3_normalized_and_lowpass_resampled_data.pickle",
            # Extra parameter:  Append to this for output file names
            "{path[0]}/{basename[0]}")
def create_df_w_normed_and_lowpass_columns(infile, outfiles, output_file_name_root):
# @transform(resample_imgs, suffix('resampled_img_df_v2.pickle'), 'normalized_and_lowpass_resampled_data.pickle')
# def create_df_w_normed_and_lowpass_columns(infile, outfile):

    start_time = time.time()
    
    lin_reg_df = pickle.load(open(infile, 'rb'))

    lst = lin_reg_df.final_cluster_label.unique()
    # lst = [0]
    combs = []

    for i in range(1, len(lst)+1):
        els = [list(x) for x in itertools.combinations(lst, i)]
        combs.extend(els)

    # combs is the gridsearch values made on the final_cluster_labels

    for output_number, these_clusters_from_gridsearch in enumerate(combs):

        df = pd.concat([lin_reg_df[(lin_reg_df.final_cluster_label == this_cluster)] for this_cluster in these_clusters_from_gridsearch]).sort_values(by = 'final_cluster_label')

        # print()

        cluster_labels = (df.final_cluster_label.unique())

        s = df.bbox_linear_size_arcsec.unique()

        list_to_permute = [s.tolist()] * len(df.final_cluster_label.unique())

        all_possible_zoom_in_combos = list(itertools.product(*list_to_permute))

        all_data_df_list = []

        for label, this_xrs_w_this_img_df in df.groupby(['img_wavelength', 'img_telescope', 'img_instrument', 'xrs_instrument', 'xrs_telescope', 'xrs_wavelength']):

            start_time = time.time()

            for do_this_permutation in all_possible_zoom_in_combos:

                # print([do_this_permutation])

                gridsearch_integration_pixel_list = []
                gridsearch_integration_hpc_list = []
                gridsearch_coord_pairs_pix_list = []
                gridsearch_coord_pairs_hpc_list = []



                A_cvx, x_cvx, timeseries, this_cluster_label_list = convex_fits_and_filtering_module.return_matrices_for_this_zoom_in_combo(do_this_permutation, this_xrs_w_this_img_df)

                A_cvx_lowpass, x_cvx_lowpass, timeseries, this_cluster_label_list = convex_fits_and_filtering_module.return_matrices_for_this_zoom_in_combo_low_pass_filter(do_this_permutation, this_xrs_w_this_img_df)

                cluster_data_list = []

                for order_number, cluster_label in enumerate(cluster_labels):
                        
                    this_zoom_in_data = this_xrs_w_this_img_df[(this_xrs_w_this_img_df['bbox_linear_size_arcsec'] == do_this_permutation[order_number]) & (this_xrs_w_this_img_df['final_cluster_label'] == cluster_label)].sort_values(by = 'resampled_date_time')

                    this_zoom_in_data['low_pass_normed_xrs'] = x_cvx_lowpass

                    this_zoom_in_data['low_pass_normed_cluster'] = A_cvx_lowpass[:,order_number]

                    this_zoom_in_data['normed_cluster_flux'] = A_cvx[:, order_number]

                    this_zoom_in_data['normed_xrs_flux'] = x_cvx

                    this_zoom_in_data['zoom_in_type'] = [do_this_permutation for _ in this_zoom_in_data.low_pass_normed_cluster]

                    this_zoom_in_data['gridsearch_clusters'] = [tuple(these_clusters_from_gridsearch) for _ in this_zoom_in_data.low_pass_normed_cluster]

                    gridsearch_coord_pairs_pix_list.append([this_zoom_in_data.x_pix.iloc[0], this_zoom_in_data.y_pix.iloc[0]])

                    gridsearch_coord_pairs_hpc_list.append([this_zoom_in_data.x_hpc.iloc[0], this_zoom_in_data.y_hpc.iloc[0]])

                    gridsearch_integration_hpc_list.append(this_zoom_in_data.integration_hpc_bbox.iloc[0])

                    gridsearch_integration_pixel_list.append(this_zoom_in_data.integration_pixel_bbox.iloc[0])
                                                        
                    cluster_data_list.append(this_zoom_in_data)

                # add gridsearch metadata to cluster_data_list and append to all_data_list

                cluster_data = pd.concat(cluster_data_list)

                cluster_data['gridsearch_integration_pixel_bbox'] = [gridsearch_integration_pixel_list for _ in cluster_data.zoom_in_type]

                cluster_data['gridsearch_integration_hpc_bbox'] = [gridsearch_integration_hpc_list for _ in cluster_data.zoom_in_type]

                cluster_data['gridsearch_coord_pairs_pix_list'] = [gridsearch_coord_pairs_pix_list for _ in cluster_data.zoom_in_type]

                cluster_data['gridsearch_coord_pairs_hpc_list'] = [gridsearch_coord_pairs_hpc_list for _ in cluster_data.zoom_in_type]

                all_data_df_list.append(cluster_data)

        output_df = pd.concat(all_data_df_list)

        output_file_name = f'{output_file_name_root}.gridsearch_{output_number}.v3_normalized_and_lowpass_resampled_data.pickle'

        # print(output_file_name)

        pickle.dump(output_df, open(output_file_name, 'wb'))

# pipeline_run([create_df_w_normed_and_lowpass_columns], multithread = 15)


@transform(create_df_w_normed_and_lowpass_columns, suffix('.v3_normalized_and_lowpass_resampled_data.pickle'), '.zoom_in_lasso_results_v3.pickle')
# @transform(resample_imgs, suffix('resampled_img_df.pickle'), 'zoom_in_lasso_results_w_lowpass_filter.pickle')
def fit_all_zoom_LASSO(infile, outfile):

    all_data_normed = pickle.load(open(infile, 'rb'))

    lasso_df_list = []

    for label, group in all_data_normed.groupby(['xrs_instrument', 'xrs_wavelength', 'img_wavelength', 'img_telescope', 'zoom_in_type', 'img_instrument', 'xrs_telescope' ]):

        all_df_list = []

        # start_time = time.time()

        ordered_group = group.sort_values('resampled_date_time')

        resampled_img_list = []

        cluster_label_list = []

        coord_hpc_list = []

        coord_pix_list = []

        gridsearch_cluster = ordered_group.iloc[0].gridsearch_clusters

        gridsearch_pix_integration_bbox = ordered_group.iloc[0].gridsearch_integration_pixel_bbox

        gridsearch_hpc_integration_bbox = ordered_group.iloc[0].gridsearch_integration_hpc_bbox

        working_dir = group.working_dir.iloc[0]

        # print(group.gridsearch_clusters.value_counts(), label)

        # print('---')

        for cluster_label in gridsearch_cluster:

    #         cluster_label_list.append(cluster_label)

            ordered_mask = ordered_group[ordered_group.final_cluster_label == cluster_label].sort_values(by = 'resampled_date_time')

            resampled_img_list.append(ordered_mask['low_pass_normed_cluster'])

            coord_hpc_list.append((ordered_mask.x_hpc.iloc[0],ordered_mask.y_hpc.iloc[0]))

            coord_pix_list.append((ordered_mask.x_pix.iloc[0],ordered_mask.y_pix.iloc[0]))


        # create optimization matrixes    

        A_cvx = np.vstack(resampled_img_list).T

        x_cvx = np.array(ordered_mask.low_pass_normed_xrs)

            # A_cvx, x_cvx, timeseries, this_cluster_label_list = convex_fits_and_filtering_module.return_matrices_for_this_zoom_in_combo_low_pass_filter(do_this_permutation, this_xrs_w_this_img_df)



        X_train = A_cvx
        Y_train = x_cvx

        beta = cp.Variable(A_cvx.shape[1])

        constraints_model1 = [beta>=0, beta <= 1]

        # constraints_model2 = [beta>=0, cp.sum(beta) == 1]

        lambd = cp.Parameter(nonneg=True)

        problem = cp.Problem(cp.Minimize(convex_fits_and_filtering_module.objective_fn(X_train, Y_train, beta, lambd)), constraints_model1)

        lambd_values = np.logspace(-3,1, 50)


        for this_lambda_value in lambd_values:

            start_time = time.time()

            lambd.value = this_lambda_value

            problem.solve()

            # train_errors.append(mse(X_train, Y_train, beta))

            # beta_values.append(beta.value)

            test_these_beta_values = beta.value



            this_df = pd.DataFrame(
                                    {'lamda': this_lambda_value, 
                                    'coeff': [test_these_beta_values],
                                    'img_wavelength': label[2] , 
                                    'img_telescope': label[3] ,
                                    'img_instrument' : label[5],
                                    'xrs_instrument': label[0] , 
                                    'xrs_wavelength': label[1] ,
                                    'xrs_telescope': label[6] ,
                                    'zoom_in_type':  [label[4]] , 
                                    'resampled_file': infile, 
                                    'gridsearch_clusters': [gridsearch_cluster],
                                    'hpc_coord_tuple': [tuple(coord_hpc_list)],
                                    'pix_coord_tuple': [tuple(coord_pix_list)],
                                    'integration_pixels_bbox': [gridsearch_pix_integration_bbox],
                                    'integration_hpc_bbox': [gridsearch_hpc_integration_bbox],
                                    'working_dir': working_dir
                                    })
        

            #####

            fit_linear_combo_vector = np.dot(A_cvx, test_these_beta_values)

            # RMSE_array = np.sqrt(mean_squared_error(x_cvx, fit_linear_combo_vector))

            this_df['RMSE'] = np.sqrt(mean_squared_error(x_cvx, fit_linear_combo_vector))

            pears, pval = mstats.pearsonr(x_cvx, fit_linear_combo_vector)

            this_df['pears_corr'] = pears

            this_df['p_val'] = pval

            this_df['MSE'] = mean_squared_error(x_cvx, fit_linear_combo_vector)

            E_tot_img = np.sum(fit_linear_combo_vector, axis = 0)

            E_tot_xray = [np.sum(x_cvx)]

            this_df['vector_fit'] = LASSO_metrics_module.fit_least_sq_single_vector(fit_linear_combo_vector, x_cvx)

            this_df['E_tot'] = LASSO_metrics_module.fit_least_sq_E_tot(E_tot_img,E_tot_xray)

            this_df['xray_data'] = [x_cvx]

            this_df['linear_combo_fit'] = [fit_linear_combo_vector]

            this_df['resampled_time_stamp'] = [ordered_mask.resampled_time_stamp.to_list()]

            #####

            this_df['best_fit'] = LASSO_metrics_module.best_fit(this_df)

            ### save clusters fitted matrix

            ### 
            output_fitted_matrix = []
            for number, coeff in enumerate(this_df.iloc[0].coeff):

                output_fitted_matrix.append(A_cvx[:, number]* coeff)

            this_df['cluster_matrix'] = [np.array(output_fitted_matrix)]

            end_time = time.time()

            zoom_lasso_run_time = end_time - start_time

            this_df['zoom_lasso_run_time'] = [zoom_lasso_run_time]

            all_df_list.append(this_df)

        output_df = pd.concat(all_df_list)

        # output_df['zoom_lasso_run_time'] = [zoom_lasso_run_time for _ in output_df.coeff]

        lasso_df_list.append(output_df)   

    output_df = pd.concat(lasso_df_list)
        
    pickle.dump(output_df, open(outfile, 'wb'))


# pipeline_run([fit_all_zoom_LASSO], multithread = 15, verbose = 1)



@collate(fit_all_zoom_LASSO, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working\/resampled_img_df_v3.gridsearch_\d{1,3}.zoom_in_lasso_results_v3.pickle'), output = '{subpath[0][0]}/v3_LASSO_df.pickle')
def collate_gridsearched_LASSO_fits_and_metrics(infiles, outfile):

    all_lasso_fits = pd.concat([pickle.load(open(this_file, 'rb')) for this_file in infiles]).sort_values(by = 'best_fit')


    pickle.dump(all_lasso_fits, open(outfile, 'wb'))


# pipeline_run([collate_gridsearched_LASSO_fits_and_metrics], multithread = 15, verbose = 1)
   
@transform(collate_gridsearched_LASSO_fits_and_metrics, suffix('v3_LASSO_df.pickle'), 'all_best_fits_w_lowpass_filter_v3.pickle')
def candidates_w_coeff_greater_than_10_percent_for_each_xrs_img_combo_filter(infile, outfile):

    ######## analysis function #############

######## analysis function #############

    def filter_for_coeff_greater_than_10_percent(all_lasso_data_df):

        best_df = all_lasso_data_df.sort_values('best_fit').iloc[0]

        coef_array = best_df.coeff

        gridsearch_cluster_array = np.array(best_df.gridsearch_clusters)

        surviving_clusters_tuple = tuple(gridsearch_cluster_array[np.where(coef_array > .1)])

        filtered_for_good_clusters = all_lasso_data_df[all_lasso_data_df.gridsearch_clusters == surviving_clusters_tuple].sort_values('best_fit')

        return(filtered_for_good_clusters.iloc[0])


    ######## analysis function #############

    collated_lasso_metrics = pickle.load(open(infile, 'rb'))

    filtered_clusters_coeff_by_10_percent_df_list = []

    for label, group in collated_lasso_metrics.groupby(['xrs_wavelength', 'xrs_telescope', 'xrs_instrument', 'img_wavelength', 'img_telescope', 'img_instrument']):
        
    #     print(label)
        
        try:

            surviving = filter_for_coeff_greater_than_10_percent(group)

            filtered_clusters_coeff_by_10_percent_df_list.append(surviving)

        # find how many xrs/img combos agree to a particular gridsearch_cluster tuple

            no_conf_df = pd.DataFrame(filtered_clusters_coeff_by_10_percent_df_list).sort_values(['best_fit']) # ouput_df without gridsearch confidence calculation

            conf_df = pd.DataFrame(no_conf_df.gridsearch_clusters.value_counts()).reset_index()

            output_list = []

            for _, row in conf_df.iterrows():

                cluster_tuples = row['index']

                mask = no_conf_df[no_conf_df.gridsearch_clusters == cluster_tuples]

                mask['gridsearch_cluster_conf'] = [len(mask)/ len((no_conf_df)) for _ in mask.gridsearch_clusters]

                output_list.append(mask)

            output_df = pd.concat(output_list)
        
        # sometimes, there will only be 1 coeff and that coef will be < .10. 
        # pass that wl/xrs fit
        # happened in : ['flarecandidate_M1.1_at_2017-10-20T23_27_40_23.working','flarecandidate_C8.2_at_2017-09-07T06_28_40_36.working','flarecandidate_C3.1_at_2011-09-24T07_23_20_37.working']
        except IndexError:
    #         print('index error')
            pass

    output_df = pd.concat(output_list)


    pickle.dump(output_df, open(outfile, 'wb'))

# pipeline_run([candidates_w_coeff_greater_than_10_percent_for_each_xrs_img_combo_filter], multithread = 15, verbose = 1)

@transform(candidates_w_coeff_greater_than_10_percent_for_each_xrs_img_combo_filter, suffix('all_best_fits_w_lowpass_filter_v3.pickle'), 'best_of_the_rest_fit_vote_w_lowpass_filter_v3.pickle')
def choose_best_of_the_rest_fit(infile, outfile):

    voting_df = pickle.load(open(infile, 'rb')).sort_values(['best_fit']).iloc[0]

    pickle.dump(voting_df, open(outfile, 'wb'))

# pipeline_run([choose_best_of_the_rest_fit], multithread = 35, verbose = 1)

@transform(choose_best_of_the_rest_fit, suffix('best_of_the_rest_fit_vote_w_lowpass_filter_v3.pickle'), 'best_fit_lowpass_w_known_flare_meta_path.pickle')
def find_known_flares_metadata(infile, outfile):

    best_fit_file = pickle.load(open(infile,'rb'))

    hek_flares = pickle.load(open(f'{dataconfig.DATA_DIR_PRODUCTS}/SS_and_SWPC_hek_flares.pickle', 'rb'))

    flare_datetime = helio_reg_exp_module.date_time_from_flare_candidate_working_dir(infile)

    work_dir = best_fit_file.working_dir

    initialization_files = pickle.load(open(f'{work_dir}/initialize_with_these_files_df.pickle', 'rb'))

    # clean_mask = clean_data_meta_data[(clean_data_meta_data.instrument == defined_flares.iloc[0].img_instrument) & (clean_data_meta_data.wavelength == defined_flares.iloc[0].img_wavelength) & (clean_data_meta_data.QUALITY == 0)]

    clean_mask = initialization_files[(initialization_files.instrument == best_fit_file.img_instrument) & (initialization_files.wavelength == best_fit_file.img_wavelength) & (initialization_files.instrument == best_fit_file.img_instrument) ]

    ordered_clean_mask = clean_mask.sort_values(by = 'date_time')

    all_known_flares = hek_flares[(hek_flares.peak_time > ordered_clean_mask.date_time.iloc[0]) & (hek_flares.peak_time < ordered_clean_mask.date_time.iloc[-1])]

    single_flares_to_associate = associate_candidates_to_flare_meta_module.filter_the_multiple_entries_from_hek_flares(all_known_flares)

    known_flare_meta_path = os.path.join(work_dir, 'known_flares.pickle')

    pickle.dump(single_flares_to_associate, open(known_flare_meta_path, 'wb'))

    best_fit_file['known_flares_file_path'] = known_flare_meta_path

    pickle.dump(best_fit_file, open(outfile, 'wb'))



# pipeline_run([find_known_flares_metadata], multithread = 15, verbose = 1)

@jobs_limit(1)
@transform(find_known_flares_metadata, suffix('best_fit_lowpass_w_known_flare_meta_path.pickle'),'best_fit_lowpass_w_harp_and_flare_meta_paths.pickle')
def query_HARP_drms_database_and_pass_AR_available(infile, outfile):

# save the drms harp pickle file path within the define flare data frame
# save the drms dataframe as a singular python pickle

    infile_df = pickle.load(open(infile, 'rb'))

    copy_infile = infile_df.copy()

    query_window_list = [30,60,120,300,500,1000,1400] # 1hour, 2 hours, 4 hours, 10 hours, 16 hours, 33 hours, 46 hours

    flare_datetime = helio_reg_exp_module.date_time_from_flare_candidate_working_dir(infile)

    query_list_decision = [len(query_the_data.drms_AR_hmi(flare_datetime, query_time = time)) != 0 for time in query_window_list]

    best_query_time = query_window_list[np.where(np.array(query_list_decision) == True)[0][0]]

    output_df = query_the_data.drms_AR_hmi(flare_datetime, query_time = best_query_time)

    work_dir = copy_infile.working_dir

    drms_file_name = os.path.join(dataconfig.DATA_DIR_FLARE_CANDIDATES, work_dir, 'drms_harp_availability.pickle')

    pickle.dump(output_df, open(drms_file_name, 'wb'))

    copy_infile['drms_harp_file'] = drms_file_name 

    copy_infile['drms_query_time'] = best_query_time 

    pickle.dump(copy_infile, open(outfile, 'wb'))

# pipeline_run([query_HARP_drms_database_and_pass_AR_available], multithread = 30, verbose = 3)


@transform(query_HARP_drms_database_and_pass_AR_available, suffix('best_fit_lowpass_w_harp_and_flare_meta_paths.pickle'), 'defined_flare_peaks_w_best_fit_180s_window.pickle')
def define_ALEXIS_flare_peaktime(infile, outfile):

    input_df = pickle.load(open(infile, 'rb'))

    output_df = ALEXIS_02_define_ALEXIS_flares_module.find_alexis_peaks(input_df)

    pickle.dump(output_df, open(outfile, 'wb'))

# pipeline_run([define_ALEXIS_flare_peaktime], multithread = 35, verbose = 1)


@transform(define_ALEXIS_flare_peaktime, suffix('defined_flare_peaks_w_best_fit_180s_window.pickle'), 'defined_flares_w_harp.pickle')
def associate_candidates_to_HARP(infile, outfile):


    test_flare_df = pickle.load(open(infile, 'rb'))
    #find span of harps and the fits_header

    span_of_harps_df, data_map = ALEXIS_02_associate_flare_to_harp_module.find_harp_span(test_flare_df)
    # add path to span of harps file
    span_file_name = 'span_of_available_harps.pickle'

    test_flare_df['span_of_harp_file_path'] = [f'{test_flare_df.iloc[0].working_dir}/{span_file_name}' for _ in test_flare_df.working_dir]

    hmi_drms_df = pickle.load(open(test_flare_df.iloc[0].drms_harp_file, 'rb'))

    output_df_list = []

    for tracking_number, cluster_label in enumerate(test_flare_df.iloc[0].gridsearch_clusters):

        masked_df = test_flare_df[test_flare_df.final_cluster_label == cluster_label]

        pix_coords = test_flare_df.iloc[0].pix_coord_tuple[tracking_number]

        hpc_coords = test_flare_df.iloc[0].hpc_coord_tuple[tracking_number]

        x_pix, y_pix = pix_coords[0], pix_coords[1]

        x_hpc, y_hpc = hpc_coords[0], hpc_coords[1]

        masked_df['final_cluster_x_pix'], masked_df['final_cluster_y_pix'] = x_pix, y_pix

        masked_df['final_cluster_x_hpc'], masked_df['final_cluster_y_hpc'] = x_hpc, y_hpc
        
        pix_bbox_integration = test_flare_df.iloc[0].integration_pixels_bbox[tracking_number]
        
        hpc_bbox_integration = test_flare_df.iloc[0].integration_hpc_bbox[tracking_number]
        
        masked_df['final_cluster_pix_integration_bbox'] = [pix_bbox_integration]
        
        masked_df['final_cluster_hpc_integration_bbox'] = [hpc_bbox_integration]

        masked_df['outside_solar_border'] = ALEXIS_02_associate_flare_to_harp_module.is_coords_outside_of_solar_borders(x_pix, y_pix, data_map.fits_header)

        associated_harps_df = ALEXIS_02_associate_flare_to_harp_module.find_boolean_list_of_harps_that_bound_pix_coords(x_pix,y_pix, masked_df, span_of_harps_df, hmi_drms_df)

        include_harps_found_w_meta = ALEXIS_02_associate_flare_to_harp_module.pass_all_harps_found_metadata(associated_harps_df, hmi_drms_df,span_of_harps_df, data_map)

        output_df_list.append(include_harps_found_w_meta)

    output_df = pd.concat(output_df_list)

    pickle.dump(output_df, open(outfile, 'wb'))

# pipeline_run([associate_candidates_to_HARP], multithread = 35, verbose = 1)


@jobs_limit(1)
@transform(associate_candidates_to_HARP, suffix('defined_flares_w_harp.pickle'), 'defined_flares_w_harp_meta_and_goes_flare_class.pickle')
def assign_xray_class_to_alexis_peak(infile, outfile):

    test_infile = pickle.load(open(infile, 'rb'))
    
    test_xrs_b = ALEXIS_02_define_goes_class_module.find_xrs_data(test_infile)
    
    full_cadence_xrs_peaks = ALEXIS_02_define_goes_class_module.find_closest_peak(test_xrs_b)

    xrs_peaks_dict ={'xrs_full_peak_datetime':full_cadence_xrs_peaks[0], 'xrs_full_peak_val':full_cadence_xrs_peaks[1] }

    xrs_peaks_df = pd.DataFrame(xrs_peaks_dict).sort_values(by = 'xrs_full_peak_datetime')
    
    output_dict_list = []

    for this_dict in test_infile.to_dict('records'):


        if this_dict['ALEXIS_found'] == True:
        
            alexis_peak_datetime_list = this_dict['alexis_peaktime']
            

            for this_peak in alexis_peak_datetime_list:

                copy_dict = this_dict.copy()
                
                try:
                    fwd_bwk_time = '3m'

                    masked_xrs_peaks_df = xrs_peaks_df[(xrs_peaks_df.xrs_full_peak_datetime >= this_peak - pd.Timedelta(fwd_bwk_time)) &  (xrs_peaks_df.xrs_full_peak_datetime <= this_peak + pd.Timedelta(fwd_bwk_time))]

                    max_values = masked_xrs_peaks_df.sort_values(by = 'xrs_full_peak_val', ascending = False)

                    final_peak_time, final_peak_value = max_values.iloc[0].xrs_full_peak_datetime, max_values.iloc[0].xrs_full_peak_val
                    
                    flare_class = ALEXIS_02_define_goes_class_module.mag_to_class(final_peak_value/0.7 * u.watt/u.m**2)
                
                
                except IndexError:
            
                    fwd_bwk_time = '5m'
                    
                    masked_xrs_peaks_df = xrs_peaks_df[(xrs_peaks_df.xrs_full_peak_datetime >= this_peak - pd.Timedelta(fwd_bwk_time)) &  (xrs_peaks_df.xrs_full_peak_datetime <= this_peak + pd.Timedelta(fwd_bwk_time))]

                    max_values = masked_xrs_peaks_df.sort_values(by = 'xrs_full_peak_val', ascending = False)

                    final_peak_time, final_peak_value = max_values.iloc[0].xrs_full_peak_datetime, max_values.iloc[0].xrs_full_peak_val
                    
                    flare_class = ALEXIS_02_define_goes_class_module.mag_to_class(final_peak_value/0.7 * u.watt/u.m**2)
                    
                #### create output_dict
                
                copy_dict['final_ALEXIS_entry'] = True
                
                copy_dict['final_ALEXIS_peaktime'] = final_peak_time

                copy_dict['ALEXIS_vector_peak'] = this_peak
                
                copy_dict['final_ALEXIS_value'] = final_peak_value
                
                copy_dict['final_ALEXIS_goes_class'] = flare_class
                
                copy_dict['final_fwd_bkg_time'] = fwd_bwk_time
                
                output_dict_list.append(copy_dict)
                
                

        if this_dict['ALEXIS_found'] == False:
            
            copy_dict = this_dict.copy()
            
            copy_dict['final_ALEXIS_entry'] = False

            copy_dict['ALEXIS_vector_peak'] = pd.Timestamp('01-01-1970T00:00:00', tz = 'utc')

            copy_dict['final_ALEXIS_peaktime'] = pd.Timestamp('01-01-1970T00:00:00', tz = 'utc')

            copy_dict['final_ALEXIS_value'] = 0

            copy_dict['final_ALEXIS_goes_class'] = '0'
            
            copy_dict['final_fwd_bkg_time'] = 0
            
            output_dict_list.append(copy_dict)

    output_df = pd.DataFrame(output_dict_list)

    pickle.dump(output_df, open(outfile, 'wb'))

# pipeline_run([assign_xray_class_to_alexis_peak], multithread = 35, verbose = 1)

@transform(assign_xray_class_to_alexis_peak, suffix('defined_flares_w_harp_meta_and_goes_flare_class.pickle'), 'first_hek_report.pickle')
def create_first_hek_report(infile, outfile):
    
    defined_flares = pickle.load(open(infile,'rb'))

    known_flares = pickle.load(open(defined_flares.iloc[0].known_flares_file_path, 'rb'))

    known_flares['peak_date_time'] = [convert_datetime.astropytime_to_pythondatetime(this_time) for this_time in known_flares.peak_time]

    #load span of harps df with HARP ARS and HARP AR
    span_of_harps_df = ALEXIS_03_create_hek_report_module.create_span_of_harps(infile)


    canonical_w_harp_df = pd.DataFrame([ALEXIS_03_create_hek_report_module.find_canonical_harp_by_coords_per_flare_entry(this_row, span_of_harps_df) for _,this_row in known_flares.iterrows()])

    master_known_flares = pd.DataFrame([ALEXIS_03_create_hek_report_module.find_canonical_harp_by_hekAR_per_flare_entry(row, span_of_harps_df) for _,row in canonical_w_harp_df.iterrows()])


    for_hek_alexis = ALEXIS_03_create_hek_report_module.standardize_alexis_for_comparison(defined_flares)

    for_hek_others = pd.DataFrame([(ALEXIS_03_create_hek_report_module.standardize_others_for_comparison(row)) for _, row in master_known_flares.iterrows()])

    hek_report = pd.concat([for_hek_others,for_hek_alexis]) 

    hek_report['working_dir'] = [defined_flares.iloc[0].working_dir for _ in hek_report.event_date]

    pickle.dump(hek_report, open(outfile,'wb'))

pipeline_run([create_first_hek_report], multiprocess = 35, verbose = 4)

############ CONTINUATION OF PIPELINE PRIOR TO APRIL 11 2024 #####################

# @transform(assign_xray_class_to_alexis_peak, suffix('defined_flares_w_harp_meta_and_goes_flare_class.pickle'), 'ALEXIS_flares_w_harp_goes_class_and_known_flare_meta.pickle')
# def associate_candidates_to_flare_meta(infile, outfile):

#     defined_flares = pickle.load(open(infile,'rb'))

#     known_flares = pickle.load(open(defined_flares.iloc[0].known_flares_file_path, 'rb'))
    
#     known_flares['peak_date_time'] = [convert_datetime.astropytime_to_pythondatetime(this_time) for this_time in known_flares.peak_time]

#     master_subset = known_flares[['hpc_x', 'hpc_y', 'id_team', 'peak_date_time', 'goes_class']]

#     master_hek_flares = master_subset.rename(columns = {'hpc_x': 'x_hpc', 'hpc_y': 'y_hpc'})

#     output_dict_list = []

#     for this_dict in defined_flares.to_dict('records'):

#         if this_dict['ALEXIS_found'] == True:
            
#             selected_keys = ['final_ALEXIS_goes_class','ALEXIS_vector_peak', 'final_cluster_x_hpc', 'final_cluster_y_hpc']

#             # Create a new DataFrame from the selected keys
#             master_alexis_subset = pd.DataFrame([{k: this_dict[k] for k in selected_keys}])
        
#             master_alexis_subset['id_team'] = 'ALEXIS'

#             master_alexis_flare = master_alexis_subset.rename(columns = {'final_ALEXIS_goes_class': 'goes_class', 'final_cluster_x_hpc': 'x_hpc', 'final_cluster_y_hpc': 'y_hpc', 'ALEXIS_vector_peak': 'peak_date_time'})
# #             master_alexis_flare

#             hek_flares = pd.concat([master_hek_flares, master_alexis_flare])
    
# #             print(hek_flares)

#             integration_bbox = this_dict['final_cluster_hpc_integration_bbox']

#             x_max, x_min = np.array(integration_bbox)[:,0].max(), np.array(integration_bbox)[:,0].min()
#             # x_max, x_min

#             y_max, y_min = np.array(integration_bbox)[:,1].max(), np.array(integration_bbox)[:,1].min()
#             # y_max, y_min

#             wn_boundaries_hek = hek_flares[(hek_flares.x_hpc <= x_max) & (hek_flares.x_hpc >= x_min) & (hek_flares.y_hpc <= y_max)& (hek_flares.y_hpc >= y_min)]
            
# #             print(wn_boundaries_hek)

# #             wn_boundaries_hek['peak_date_time'] = [convert_datetime.astropytime_to_pythondatetime(this_time) for this_time in wn_boundaries_hek.peak_time]

#             test_alexis_peak = this_dict['ALEXIS_vector_peak']

#             # test_alexis_peak
#             this_time_max, this_time_min = test_alexis_peak + pd.Timedelta('3m'), test_alexis_peak - pd.Timedelta('3m')

#             wn_time_hek = hek_flares[ (hek_flares.peak_date_time >= this_time_min) & (hek_flares.peak_date_time <= this_time_max) ]


#             teams_detected_in_time_and_space = wn_boundaries_hek[ (wn_boundaries_hek.peak_date_time >= this_time_min) & (wn_boundaries_hek.peak_date_time <= this_time_max) ]
            
#             this_dict['teams_detected_this_flare'] = tuple(teams_detected_in_time_and_space.id_team)
            
#             this_dict['teams_detected_this_flare_class'] = tuple(teams_detected_in_time_and_space.goes_class)
            
#             this_dict['teams_detected_this_flare_coords'] = tuple([(this_x, this_y) for this_x, this_y in zip(teams_detected_in_time_and_space.x_hpc,teams_detected_in_time_and_space.y_hpc)])
            
#             this_dict['teams_detected_this_flare_time'] = tuple([(peaktime) for peaktime in teams_detected_in_time_and_space.peak_date_time])
            
#             this_dict['teams_detected_this_flare_index'] = tuple(teams_detected_in_time_and_space.index.to_list())
            
#             this_dict['teams_wn_space'] = tuple(wn_boundaries_hek.id_team)

#             this_dict['teams_wn_time'] = tuple(wn_time_hek.id_team)
            
# #             this_dict['hek_flare_time_meta'] = tuple(master_hek_flares.peak_time)

#             output_dict_list.append(this_dict)

#         else: 

#             this_dict['teams_detected_this_flare'] = ()
            
#             this_dict['teams_detected_this_flare_class'] = ()

#             this_dict['teams_detected_this_flare_coords'] = ()
            
#             this_dict['teams_detected_this_flare_time'] = ()
            
#             this_dict['teams_detected_this_flare_index'] = ()

#             this_dict['teams_wn_space'] = ()

#             this_dict['teams_wn_time'] = ()

#             output_dict_list.append(this_dict)

#     catalog_df = pd.DataFrame(output_dict_list)

    

# @transform(associate_candidates_to_flare_meta, suffix('ALEXIS_flares_w_harp_goes_class_and_known_flare_meta.pickle'), 'ALEXIS_flares_w_harp_goes_class_and_known_flare_meta_harp.pickle')
# def associate_flare_meta_to_harp_w_dbscan_metrics_and_alexis_catalog(infile, outfile):

#     alexis_cat_df = pickle.load(open(infile,'rb'))

#         #prep known metadata info

#     defined_flares = pickle.load(open(infile,'rb'))

#     known_flares = pickle.load(open(defined_flares.iloc[0].known_flares_file_path, 'rb'))

#     known_flares['peak_date_time'] = [convert_datetime.astropytime_to_pythondatetime(this_time) for this_time in known_flares.peak_time]

#     master_subset = known_flares[['hpc_x', 'hpc_y', 'id_team', 'peak_date_time', 'goes_class', 'AR_num', 'hgs_x', 'hgs_y']]

#     master_hek_flares = master_subset.rename(columns = {'hpc_x': 'x_hpc', 'hpc_y': 'y_hpc', 'AR_num': 'hek_ar_num', 'hgs_x': 'x_hgs', 'hgs_y': 'y_hgs'})


#     # open span of harps file

#     harp_span_file_path = alexis_cat_df.iloc[0].span_of_harp_file_path

#     span_of_harps_df = pickle.load(open(harp_span_file_path,'rb'))

#     #### some needed functions. Garden this out to a module later

#     def check_known_flare_wn_alexis_integration_area(AR_hpc_bbox, candidate_flare_coords):

#         xvals = np.array(AR_hpc_bbox)[:,0]
#         yvals = np.array(AR_hpc_bbox)[:,1]

#         # print(xvals,yvals)
#         # print(candidate_flare_coords)

#         # #create array of coordinates 
#         x_check = np.array([xvals.max(), xvals.min(), candidate_flare_coords[0]])
#         y_check = np.array([yvals.max(), yvals.min(), candidate_flare_coords[1]])

#         # print(x_check, y_check)

#         # #sort the values
#         # #candidate coord should lie in the middle with index 1
#         sorted_xcheck = (np.sort(x_check))
#         sorted_ycheck = (np.sort(y_check))
#         8302813425219152
#         # print(sorted_xcheck, sorted_ycheck)
#         # # find the location of the candidate coords. 
#         # # They should be in position 1 of the array
        
#         position_of_x = (np.where( sorted_xcheck ==  candidate_flare_coords[0]))[0]
#         position_of_y = (np.where( sorted_ycheck ==  candidate_flare_coords[1]))[0]

#         # print(position_of_x, position_of_y)

#         # ### if coords are within the bounding box and we define r = position_of_x * position_of_y == 1
        
#         r = position_of_x*position_of_y == 1

#         # print(r)
        
#         return(r[0])

#     # check_known_flare_wn_alexis_integration_area(span,coordinates) for span in zip(AR_hpc_bbox_row)

#     def find_canonical_harp_per_flare_entry(canonical_data_row):

#         coordinates_hpc = [canonical_data_row['x_hpc'],canonical_data_row['y_hpc']]

#         coordinates_hgs = [canonical_data_row['x_hgs'],canonical_data_row['y_hgs']]

#         canonical_wn_hpc_span = [check_known_flare_wn_alexis_integration_area(span, coordinates_hpc) for span in span_of_harps_df.span_hpc_bbox]

#         canonical_wn_hgs_span = [check_known_flare_wn_alexis_integration_area(span, coordinates_hgs) for span in span_of_harps_df.span_hgs_bbox]

#         canonical_to_harp_df = pd.DataFrame({'HARP': span_of_harps_df.HARPNUM.to_list(), 'found_harp_in_hgs': canonical_wn_hgs_span, 'found_harp_in_hpc': canonical_wn_hpc_span, 'id_team':canonical_data_row['id_team'] })

        

#         canonical_row_dict = canonical_data_row.to_dict()
        
#         output_dict_list = []
#         for _, row in canonical_to_harp_df.iterrows():

#             copy_row = row.copy()

#             copy_dict = copy_row.to_dict()

#             combined_dict = {**copy_dict, **canonical_row_dict}

#             output_dict_list.append(combined_dict)

#         output_df = pd.DataFrame(output_dict_list)

#         return(output_df)


#         # end of needed functions

#     canonical_w_harp_df = pd.concat([find_canonical_harp_per_flare_entry(this_row) for _,this_row in master_hek_flares.iterrows()])

#     # check if they are associated to HARP, how many HARPS, if no HARP, then still include row but with the HARP num, HARP found, HEK ar num, and others a generic value

#     meta_flare_catalog_w_harp_list = []

#     for id_team, id_group in canonical_w_harp_df.groupby(['id_team', 'goes_class']):

#         # for_, row in id_group.iterrows():

#         mask_for_truth_in_one_harp = id_group[id_group.found_harp_in_hgs == True]

#         if len(mask_for_truth_in_one_harp) >= 1:

#             new_df = mask_for_truth_in_one_harp.drop(['found_harp_in_hgs', 'found_harp_in_hpc', 'HARP'], axis=1).iloc[0].to_dict()

#             new_df['num_of_HARPS_found'] = len(mask_for_truth_in_one_harp)

#             new_df['HARP_found'] = True

#             new_df['HARPNUM_list'] = [mask_for_truth_in_one_harp.HARP.to_list()]

#             new_df['HARPS_NOAA_ARS_list'] =[ mask_for_truth_in_one_harp.hek_ar_num.to_list()]

#             meta_flare_catalog_w_harp_list.append(pd.DataFrame(new_df))

#         if len(mask_for_truth_in_one_harp) == 0:

#             no_harp_found_row = id_group.drop(['found_harp_in_hgs', 'found_harp_in_hpc', 'HARP'], axis=1).iloc[0]

#             no_harp_found_row['num_of_HARPS_found'] = len(mask_for_truth_in_one_harp)

#             no_harp_found_row['HARP_found'] = False

#             no_harp_found_row['HARPNUM_list'] = [[]]

#             no_harp_found_row['HARPS_NOAA_ARS_list'] = [[]]

#             no_harp_df = pd.DataFrame([no_harp_found_row.to_dict()])

#             meta_flare_catalog_w_harp_list.append(no_harp_df)

#     # make df w harp associated canonical flares
#     hek_flares = pd.concat(meta_flare_catalog_w_harp_list)
#     # lets find ALEXIS in flare meta. but not find teams that detected... just the information to answer that question 


#     output_dict_list = []
#     # NOTE: defined_flares has all clusters, regardless if they have a flare associated to it
#     for _, row in defined_flares.iterrows():

#         selected_keys = ['final_ALEXIS_goes_class','ALEXIS_vector_peak', 'final_cluster_x_hpc', 'final_cluster_y_hpc','HARP_found','num_of_HARPS_found','HARPS_NOAA_ARS_list', 'HARPNUM_list']

#         # Create a new DataFrame from the selected keys
#         master_alexis_subset = pd.DataFrame([{k: row[k] for k in selected_keys}])

#         master_alexis_subset['id_team'] = 'ALEXIS'

#         master_alexis_flare = master_alexis_subset.rename(columns = {'final_ALEXIS_goes_class': 'goes_class', 'final_cluster_x_hpc': 'x_hpc', 'final_cluster_y_hpc': 'y_hpc', 'ALEXIS_vector_peak': 'peak_date_time'})

#         #master hek dataframe

#         selected_keys = ['id_team', 'x_hpc', 'y_hpc', 'peak_date_time', 'goes_class', 'num_of_HARPS_found', 'HARP_found', 'HARPNUM_list', 'HARPS_NOAA_ARS_list']

#         # Create a new DataFrame from the selected keys
#         master_hek_subset = hek_flares[selected_keys]

#         # join hek master and alexis master

#         all_flares_df = pd.concat([master_hek_subset, master_alexis_flare]).sort_values(by = 'id_team')

#         #integration area

#         integration_bbox_hpc = row['final_cluster_hpc_integration_bbox']

#         x_max, x_min = np.array(integration_bbox_hpc)[:,0].max(), np.array(integration_bbox_hpc)[:,0].min()
#         # print( x_max, x_min)

#         y_max, y_min = np.array(integration_bbox_hpc)[:,1].max(), np.array(integration_bbox_hpc)[:,1].min()
#         # print(y_max, y_min)
#         # are catalog compilations bound in space?
#         wn_boundaries_hek = all_flares_df[(all_flares_df.x_hpc <= x_max) & (all_flares_df.x_hpc >= x_min) & (all_flares_df.y_hpc <= y_max)& (all_flares_df.y_hpc >= y_min)]#.sort_values('id_team')

#         # find peak-time of ALEXIS candidate
#         test_alexis_peak = row['ALEXIS_vector_peak']

#         # are catalog compilations bound in time?
#         this_time_max, this_time_min = test_alexis_peak + pd.Timedelta('3m'), test_alexis_peak - pd.Timedelta('3m')

#         wn_time_hek = all_flares_df[ (all_flares_df.peak_date_time >= this_time_min) & (all_flares_df.peak_date_time <= this_time_max) ]#.sort_values('id_team')

#         # now we have wn_time and wn_space
#         this_dict = row.copy()
        
#         this_dict['teams_wn_time'] = tuple(wn_time_hek.id_team)
#         this_dict['teams_wn_time_flare_peak'] = tuple(wn_time_hek.peak_date_time)
#         this_dict['teams_wn_time_goes_class'] = tuple(wn_time_hek.goes_class)
#         this_dict['teams_wn_time_hpc_coords'] = tuple([(this_x, this_y) for this_x, this_y in zip(wn_time_hek.x_hpc,wn_time_hek.y_hpc)])
#         this_dict['teams_wn_time_harps_found'] = tuple(wn_time_hek.HARP_found)
#         this_dict['teams_wn_time_num_harps_found'] = tuple(wn_time_hek.num_of_HARPS_found)
#         this_dict['teams_wn_time_harp_num'] = tuple(wn_time_hek.HARPNUM_list)
#         this_dict['teams_wn_time_NOAA_AR_num'] = tuple(wn_time_hek.HARPS_NOAA_ARS_list)

#         this_dict['teams_wn_space'] = tuple(wn_boundaries_hek.id_team)
#         this_dict['teams_wn_space_flare_peak'] = tuple(wn_boundaries_hek.peak_date_time)
#         this_dict['teams_wn_space_goes_class'] = tuple(wn_boundaries_hek.goes_class)
#         this_dict['teams_wn_space_hpc_coords'] = tuple([(this_x, this_y) for this_x, this_y in zip(wn_boundaries_hek.x_hpc,wn_boundaries_hek.y_hpc)])
#         this_dict['teams_wn_space_harps_found'] = tuple(wn_boundaries_hek.HARP_found)
#         this_dict['teams_wn_space_num_harps_found'] = tuple(wn_boundaries_hek.num_of_HARPS_found)
#         this_dict['teams_wn_space_harp_num'] = tuple(wn_boundaries_hek.HARPNUM_list)
#         this_dict['teams_wn_space_NOAA_AR_num'] = tuple(wn_boundaries_hek.HARPS_NOAA_ARS_list)


#         output_dict_list.append(this_dict)
    
    
#     ALEXIS_catalog_df = pd.DataFrame(output_dict_list)

#     # lets add the metadata for the clustering 
#     clustering_df_path = f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/{helio_reg_exp_module.work_dir_from_flare_candidate_input_string(infile)}/v3_interesting_pixels.pickle'

#     cluster_meta_df = pd.DataFrame(pickle.load(open(clustering_df_path, 'rb')))

#     hyper_cluster_dict = cluster_meta_df[['voting_wl', 'voting_tel', 'voting_inst','label','num_votes','passing_votes','avail_votes','temporal_vote_number','temporal_passing_vote','temporal_vote_members']].to_dict('records')
    
#     # Create lists for the 'label' column for each entry
#     # avail_df_dict
#     avail_dict_list = []
#     for row in hyper_cluster_dict:

#         [row['label']]*len(row['voting_inst'])

#         row['label_list'] = [row['label']]*len(row['voting_inst'])

#         avail_dict_list.append(row)
#     # avail_dict_list
#     full_avail = pd.DataFrame(avail_dict_list)
   

#     modified_by_avail_spectral_df_list = []

#     for final_cluster_label, final_cluster_group in cluster_meta_df.groupby('label'):

#         mask = full_avail[full_avail.label == final_cluster_label]

#         cluster_mask = final_cluster_group.copy()

#         cluster_mask['spectral_avail_dict'] = [mask.to_dict('records')]

#         modified_by_avail_spectral_df_list.append(cluster_mask)

#     spectral_w_avail_dict_df =pd.concat((modified_by_avail_spectral_df_list))


#     test = []
#     for element in spectral_w_avail_dict_df.spectral_avail_dict.to_list():

#         test.append(pd.DataFrame(element[0]))

#     voting_df = pd.concat(test)

#     # lets add the metadata for the avail images; we need to fix temporal_vote_members
#     avail_img_path = f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/{helio_reg_exp_module.work_dir_from_flare_candidate_input_string(infile)}/initialize_with_these_files_df.pickle'
#     avail_img_df = pickle.load(open(avail_img_path, 'rb'))

#     output_list = []

#     for label, group in voting_df.groupby(['voting_wl', 'voting_tel', 'voting_inst', 'label']):

#         # print(label)

#         mask = avail_img_df[(avail_img_df.wavelength == label[0]) & (avail_img_df.instrument == label[2]) &  (avail_img_df.telescope == label[1])]

#         output_df = group.copy()

#         good_mask = mask[mask.QUALITY == 0]

#         # print(len(good_mask))

#         output_df['temporal_vote_members'] = [len(good_mask) for _ in group.temporal_vote_members]

#         output_list.append(output_df)

#     df = pd.concat(output_list)

#     df['hyper_percent'] = df['num_votes']/df['avail_votes']

#     df['temporal_percent'] = df['temporal_vote_number']/df['temporal_vote_members']

#     row_list = []

#     for label, row in df.groupby(['voting_wl', 'voting_tel', 'voting_inst']):

#         if label[2] == 'AIA':
#             row['temporal_time_min'] =  row['temporal_vote_number'] * 12/60
#         if label[2] == 'SXI':
#             row['temporal_time_min'] =  row['temporal_vote_number'] * (5*60)/60

#         row_list.append(row)

#     dbscan_meta_df = pd.concat(row_list).sort_values(by ='label')

#     # insert dbscan_meta_dict in ALEXIS_flare_catalog

#     final_output_df_list = []

#     for cluster, cluster_group in ALEXIS_catalog_df.groupby(['final_cluster_label']):
        
#         for _, row in cluster_group.iterrows():
        
#             dbscan_mask = dbscan_meta_df[dbscan_meta_df.label == cluster]

#             dbscan_meta_list_of_dicts = dbscan_mask.to_dict('records')

#             output_copy = row.copy()

#             output_copy['dbscan_meta_dict_list'] = dbscan_meta_list_of_dicts

#             final_output_df_list.append(output_copy)
        
#     alexis_df_w_dbscan_meta = pd.DataFrame(final_output_df_list)
            
#     pickle.dump(alexis_df_w_dbscan_meta, open(outfile, 'wb'))

# pipeline_run([associate_flare_meta_to_harp_w_dbscan_metrics_and_alexis_catalog], multiprocess = 30, verbose = 1)


########## END OF PIPELINE APRIL 11 2024 ########################


# @subdivide(associate_candidates_to_flare_meta, formatter(),
#             # Output parameter: Glob matches any number of output file names
#             "{path[0]}/{basename[0]}.*.v3_initialize_summary_jpegs_for_movie.pickle",
#             # Extra parameter:  Append to this for output file names
#             "{path[0]}/{basename[0]}")
# def initialize_summary_jpeg_for_movie(infile, outfiles, output_file_name_root):

#     # start_time = time.time()

#     these_flares = pd.DataFrame(pickle.load(open(infile, 'rb')))

#     working_dir = these_flares.iloc[0].working_dir

#     good_quality_data_df = pickle.load(open(f'{working_dir}/initialize_with_these_files_df.pickle', 'rb'))

#     mask_properties = these_flares.iloc[0]

#     this_wl, this_inst, this_tel = mask_properties.img_wavelength, mask_properties.img_instrument, mask_properties.img_telescope

#     these_images = good_quality_data_df[(good_quality_data_df.wavelength == this_wl) & (good_quality_data_df.telescope == this_tel) & (good_quality_data_df.instrument == this_inst) & (good_quality_data_df.QUALITY == 0)]

#     these_images['catalog_file'] = [infile for _ in these_images.file_path]

#     # load and save xrs data

#     xrs_b_data = ALEXIS_02_define_goes_class_module.find_xrs_data(these_flares)

#     xrs_real_data_file_path = os.path.join(working_dir, 'real_xrs_data.pickle')

#     pickle.dump(xrs_b_data, open(xrs_real_data_file_path, 'wb'))

#     these_images['real_xrs_data_path'] = [xrs_real_data_file_path for _ in these_images.file_path]

#     for output_number, output_row in enumerate(these_images.to_dict('records')):

#         output_file_name = f'{output_file_name_root}.movie_making_{output_number}.v3_initialize_summary_jpegs_for_movie.pickle'

#         pickle.dump(output_row, open(output_file_name, 'wb'))

# # pipeline_run([initialize_summary_jpeg_for_movie], multithread = 30, verbose = 1)



# @transform(initialize_summary_jpeg_for_movie, suffix('.v3_initialize_summary_jpegs_for_movie.pickle'), '.img_for_movie_made.pickle')
# def make_movie_plots(infile, outfile):

#     input_dict = pickle.load(open(infile, 'rb'))

#     image_df = pd.DataFrame([input_dict])

#     xrs_df = pickle.load(open(image_df.iloc[0].real_xrs_data_path, 'rb'))

#     flares_df = pickle.load(open(image_df.iloc[0].catalog_file, 'rb'))

#     figure_name = ALEXIS_02_plotting_module_movie.plot_results_for_movie(image_df, flares_df, xrs_df)

#     image_df['figure_name'] = [figure_name for _ in image_df.file_path]

#     pickle.dump(image_df, open(outfile, 'wb'))


# # pipeline_run([make_movie_plots], multithread = 30, verbose = 3)

# @collate(make_movie_plots, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working\/ALEXIS_flares_w_harp_goes_class_and_known_flare_meta.movie_making_\d{1,3}.img_for_movie_made.pickle'), output = '{subpath[0][0]}/collate_jpeg_df.pickle')
# def join_movie_files(infiles, outfile):

#     output_df = pd.concat([pickle.load(open(infile, 'rb')) for infile in infiles])

#     pickle.dump(output_df, open(outfile, 'wb'))

#     # print(len(infiles), outfile)

# # pipeline_run([join_movie_files], multithread = 30, verbose = 1)

# @transform(join_movie_files, suffix('collate_jpeg_df.pickle'), 'made_movie.pickle')
# def ffmpeg_movie_maker(infile, outfile):

#     input_df = pickle.load(open(infile, 'rb'))

#     working_dir = input_df.iloc[0].working_dir

#     wd = helio_reg_exp_module.work_dir_from_flare_candidate_input_string(working_dir)

#     ffmpeg_string = f"ffmpeg -framerate 10 -pattern_type glob -i '{working_dir}/movie_file_*.jpeg' -c:v libx264 -pix_fmt yuv420p {working_dir}/{wd}_summary_vid.mp4"

#     os.system(ffmpeg_string)    # $ ffmpeg -framerate 1 -pattern_type glob -i '*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4

#     input_df['movie_path'] = [f'{working_dir}/{wd}_summary_vid.mp4' for _ in input_df.figure_name]

#     pickle.dump(input_df, open(outfile, 'wb'))

# pipeline_run([ffmpeg_movie_maker], multithread = 30, verbose = 1)
    # os.system(ffmpeg_string)    # $ ffmpeg -framerate 1 -pattern_type glob -i '*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4


# ALEXIS_flares_w_harp_goes_class_and_known_flare_meta.movie_making_*.v3_initialize_summary_jpegs_for_movie.pickle

# @transform(join_movie_files)
# def use_ffmpeg_for_movie():


    ##### query xrs data and save for plotting

    

    # all_dfs = []

    # for cluster, cluster_group in investigate_these_pixels.groupby(['label']):

    #     good_qual_copy = good_quality_data_df.copy()

    #     good_qual_copy['x_hpc'] = [cluster_group['x_hpc'].iloc[0] for _ in good_qual_copy.file_path]

    #     good_qual_copy['y_hpc'] = [cluster_group['y_hpc'].iloc[0] for _ in good_qual_copy.file_path]

    #     good_qual_copy['final_cluster_label'] = [cluster for _ in good_qual_copy.file_path]

    #     all_dfs.append(good_qual_copy)

    # concat_df = pd.concat(all_dfs).sort_values(by = 'date_time')

    # for labels, config_df in concat_df.groupby(['telescope','instrument', 'wavelength']):

    #     telescope, instrument, wl = labels[0],labels[1],labels[2]

    #     # initialize_start = time.time()

    #     if instrument == 'AIA':
    #         wl = int(wl)

    #     ordered_df = config_df.sort_values(by = 'date_time').reset_index(drop = True)

    #     for num, date_time in enumerate(ordered_df.date_time.unique()):

    #         masked_df = ordered_df[ordered_df.date_time == date_time]

    #         output_file_name = f'{output_file_name_root}.{telescope}.{instrument}.{wl}.cluster.{num}.v3_initialize_cluster_flux.pickle'

    #         # output a dataframe with number of rows == number of clusters
    #         # each dataframe is for 1 file only. 

    #         pickle.dump(masked_df, open(output_file_name, 'wb'))

# pipeline_printout_graph (f'{dataconfig.DATA_DIR_PRODUCTS}/feb_12_2023_ALEXIS_flowchart.png', "png", [assign_xray_class_to_alexis_peak])

################################################################################

# END OF PIPELINE
# NEED MAKE MOVIE
# NEED INSERT INTO SQLITE ALEXIS CATALOG

################################################################################
# print(initialization_files)




### GRAVE YARD ####

# @transform(associate_candidates_to_HARP, suffix('defined_flares_w_harp.pickle'), 'defined_flares_w_harp_and_known_flare_meta.pickle')
# def associate_candidates_to_flare_meta(infile, outfile):



#     ###############################################################


#     defined_flares = pickle.load(open(infile,'rb'))

#     hek_flares = pickle.load(open('/data/padialjr/jorge-helio/data_products/SS_and_SWPC_hek_flares.pickle', 'rb'))

#     flare_datetime = helio_reg_exp_module.date_time_from_flare_candidate_working_dir(infile)


#     # known_solarsoft = solarsoft[(solarsoft.peak_time > time_start) & (solarsoft.peak_time < time_end)]
#     # known_swpc = swpc[(swpc.peak_time > time_start) & (swpc.peak_time < time_end)]
#     clean_data_meta_data = pd.DataFrame([pickle.load(open(infile, 'rb')) for infile in glob(os.path.join(dataconfig.DATA_DIR_FLARE_CANDIDATES, defined_flares.iloc[0].work_dir, '*_clean.pickle'))])

#     # clean_mask = clean_data_meta_data[(clean_data_meta_data.instrument == defined_flares.iloc[0].img_instrument) & (clean_data_meta_data.wavelength == defined_flares.iloc[0].img_wavelength) & (clean_data_meta_data.QUALITY == 0)]

#     clean_mask = clean_data_meta_data[(clean_data_meta_data.instrument == defined_flares.iloc[0].img_instrument) & (clean_data_meta_data.wavelength == defined_flares.iloc[0].img_wavelength) ]

#     ordered_clean_mask = clean_mask.sort_values(by = 'date_time')

#     all_known_flares = hek_flares[(hek_flares.peak_time > ordered_clean_mask.date_time.iloc[0]) & (hek_flares.peak_time < ordered_clean_mask.date_time.iloc[-1])]

#     # all_known_flares = hek_flares[(hek_flares.peak_time > flare_datetime - pd.Timedelta('40min')) & (hek_flares.peak_time < flare_datetime+ pd.Timedelta('40min'))]

#         #########################

#     # make sure each team has only one entry per peak time
#     # might need to change this in the future bc
#     # of multiple entries that a team has that is seperated by minutes, seconds, or miliseconds



#     # droped_duplicate_peak_time_df = drop_duplicate_peak_times(all_known_flares)

#     # single_flares_to_associate = what_if_peak_time_not_duplicated(droped_duplicate_peak_time_df)


#     single_flares_to_associate = associate_candidates_to_flare_meta_module.filter_the_multiple_entries_from_hek_flares(all_known_flares)


#     these_alexis_df_list = []

#     for cluster_label, cluster_group in defined_flares.groupby(['final_cluster_label']):


#         # are each of the known flare location within this clusters integration boundary
        
#         known_wn_alexis_integration_df = associate_candidates_to_flare_meta_module.find_if_known_flares_wn_alexis_integration_area(cluster_group, single_flares_to_associate)

#         #load clean data and mask the img wavelength

#         # clean_data_meta_data = pd.DataFrame([pickle.load(open(infile, 'rb')) for infile in glob(os.path.join(dataconfig.DATA_DIR_FLARE_CANDIDATES, cluster_group.iloc[0].work_dir, '*clean*pickle'))])

#         # clean_mask = clean_data_meta_data[(clean_data_meta_data.instrument == cluster_group.iloc[0].img_instrument) & (clean_data_meta_data.wavelength == cluster_group.iloc[0].img_wavelength)]


#         good_data_length = len((ordered_clean_mask[ordered_clean_mask.QUALITY == 0]))

#         bad_data_length = len(ordered_clean_mask)


#         if cluster_group.iloc[0].ALEXIS_found == True:

            

#             for this_alexis_time in cluster_group.alexis_peaktime.iloc[0]:

#                 this_flare_df = associate_candidates_to_flare_meta_module.find_if_known_flares_wn_same_alexis_peak_time(this_alexis_time, known_wn_alexis_integration_df, cluster_group)

#                 this_flare_df['ALEXIS_flare_time'] = this_alexis_time

#                 this_flare_df['good_data_percent'] = good_data_length/bad_data_length

#                 these_alexis_df_list.append(this_flare_df)


#         if cluster_group.iloc[0].ALEXIS_found == False:

#             cluster_group['found_teams_in_time'] = [0]

#             cluster_group['ALEXIS_flare_time'] = [0]

#             cluster_group['found_teams_in_time'] = [0] # +- 3 min

#             cluster_group['all_teams_order_wn_time'] = [[]]

#             cluster_group['all_team_datetimes'] = [[]]

#             cluster_group['all_team_wn_alexis_bbox'] = [[]]

#             cluster_group['ALEXIS_hpc_bbox'] = [known_wn_alexis_integration_df.alexis_hpc_bbox.iloc[0]]

#             cluster_group['all_team_hpc_coords'] =[[]]

#             cluster_group['all_team_goes_class'] =[[]]

#             cluster_group['good_data_percent'] = good_data_length/bad_data_length

#             # take care of the meta
#             cluster_group['all_teams_order_meta'] = [[this_team for this_team in known_wn_alexis_integration_df.id_team]]

#             cluster_group['all_team_datetimes_meta'] = [[this_datetime for this_datetime in known_wn_alexis_integration_df.peak_time]]

#             cluster_group['all_team_wn_alexis_bbox_meta'] = [[wn_alexis_int for wn_alexis_int in known_wn_alexis_integration_df.within_alexis_integration]]

#             cluster_group['ALEXIS_hpc_bbox_meta'] = [known_wn_alexis_integration_df.alexis_hpc_bbox.iloc[0]]

#             cluster_group['all_team_hpc_coords_meta'] =[[[this_x, this_y] for this_x, this_y in zip(known_wn_alexis_integration_df.hpc_x.to_list(), known_wn_alexis_integration_df.hpc_y.to_list())]]

#             cluster_group['all_team_goes_class_meta'] =[[this_class for this_class in known_wn_alexis_integration_df.goes_class]]
            
#             # cluster_group['all_team_hpc_coords']  = [[]]

#             these_alexis_df_list.append(cluster_group)

#     output_df = pd.concat(these_alexis_df_list)

#     pickle.dump(output_df, open(outfile, 'wb'))


# @subdivide(initialization_files, formatter(),
#             # Output parameter: Glob matches any number of output file names
#             "{path[0]}/{basename[0]}.*.v3_good_clean.pickle",
#             # Extra parameter:  Append to this for output file names
#             "{path[0]}/{basename[0]}")
# def initialize_peakfinder(infile, outfiles, output_file_name_root):

#     these_wl_df = pickle.load(open(infile, 'rb'))


#     ####


#     for label, group in these_wl_df.groupby(['wavelength', 'instrument', 'telescope']):

#         wl, inst, tel = label[0], label[1], label[2]

#         if inst == 'AIA':
#             wl = int(wl)

#         ordered_group = group.sort_values(by = 'date_time')

#         for number, row in enumerate(ordered_group.to_dict('records')):

#             output_file_name = f'{output_file_name_root}.{wl}.{inst}.{tel}.{number}.v3_good_clean.pickle'

#             # print(output_file_name, len(row))

#             # output file is a config dictionary for each file to apply peakfinder and 
#             # initial dbscan
#             pickle.dump(row, open(output_file_name, 'wb'))


# # pipeline_run([initialize_peakfinder], multithread = 20)

# @transform(initialize_peakfinder, suffix('.v3_good_clean.pickle'), '.v3_peakfinder.df.pickle') #function 3
# def peakfinder_clustering_per_frame(infile, outfile):

#     """
    
#     Runs peakfinder, clusters to N-number of groups per single image, and converts
#     pixel values into hpc coords.

#     TO-DO: Might be good to reproject coordinates outside the solar border to the interior

#     """

    
#     config = pickle.load(open(infile, 'rb'))

#     instrument = config['instrument']

#     copy_dict = config.copy()
#     centroid_array = []

    

#     if config['QUALITY'] == 0:

#         if instrument == 'SXI':

#             clean_start = time.time()

#             _, raw_data, raw_header = clean_img_data_02.clean_sxi_data(config) # open fits sxi chooses the first elelem on the fits format. We can use it here for clean_aia_data

#             clean_end = time.time()

#             rotated_data = rotate(np.float32(raw_data), -1*raw_header['CROTA1'])

#             data = rotated_data/config['exp_time'] # DN/s

#             dbscan_cluster_eps = 10 # 50 arcseconds

#             r_sun = raw_header['RSUN']

#             arcsec_per_pixel = raw_header['CDELT1']

#         if instrument == 'AIA':

#             clean_start = time.time()

#             _, raw_data, raw_header = clean_img_data_02.clean_aia_data(config)

#             clean_end = time.time()

#             data = raw_data

#             r_sun = raw_header['R_SUN']

#             dbscan_cluster_eps = 83 # 50 arcseconds

#             arcsec_per_pixel = raw_header['CDELT1']

#         # make outerperimiter

        
#         # rad_list = np.linspace(0,2, 10000)*np.pi

#         # x2 = [raw_header['CRPIX1'] + r_sun * np.cos(rad) for rad in rad_list]
#         # y2 = [raw_header['CRPIX2'] + r_sun * np.sin(rad) for rad in rad_list]

#         # outer_perimiter = pd.DataFrame({'x_circle': x2, 'y_circle': y2})

#         # run the peakfinder in pixel space

#         clean_time = clean_end - clean_start

#         peak_start = time.time()

#         coordinates = peak_local_max(data, threshold_rel = .9)

#         peak_end = time.time()

#         peakfinder_time = peak_end - peak_start

#         if len(coordinates) != 0:

#             # analogous to matrixes, [rows, columns] == [y,x] ==> coordinates[:,1] will give the x values
#             # coordinates[:,0] will give the y values

#             peakfinder_array = np.array( [ [x,y] for x,y in zip(coordinates[:,1],coordinates[:,0]) ] )

#             dbscan_start = time.time()

#             # define the dbscan model
#             # all the peaks from each image will be clustered to return n-number of peaks
#             # peaks need be more than apart to be considered "outside" the dbscan radius
#             # 50 arcseconds (or 83 pixels for AIA ; 10 pixels for SXI)

#             dbscan_model = DBSCAN(eps = dbscan_cluster_eps, min_samples=1)

#             # train the model
#             dbscan_model.fit(peakfinder_array)

#             dbscan_result = dbscan_model.fit_predict(peakfinder_array)

#             dbscan_end = time.time()
            
#             dbscan_time = dbscan_end - dbscan_start

#             # find unique labels in dbscan_result

#             all_labels = np.unique(dbscan_result)

#             # drop outliers if part of all_labels. outliers are returned by
#             # the DBSCAN  defined as == float(-1)

#             good_labels = all_labels[all_labels != -1]

#             for label in good_labels:

#                 copy_dict = config.copy()

#                 which_pixels = np.where(dbscan_result == label)

#                 num_of_pix_members_in_cluster = len(peakfinder_array[which_pixels][:,0])

#                 x_mean = np.mean(peakfinder_array[which_pixels][:,0])
#                 y_mean = np.mean(peakfinder_array[which_pixels][:,1])

#                 # data_map = sunpy.map.Map(data, raw_header)

#                 # these_pixels_work = coordinate_conversion_module.algorithm_to_project_pixel_onto_surface(x_mean, y_mean, outer_perimiter, data_map)

#                 # print(these_pixels_work, x_mean, y_mean)

#                 # sky = WCS(data_map.fits_header).pixel_to_world(these_pixels_work[0], these_pixels_work[1])

#                 # # sky = WCS(data_map.fits_header).pixel_to_world(x_mean, y_mean)


#                 # sky_x_arc, sky_y_arc = sky.Tx, sky.Ty

#                 # x_hgs, y_hgs = coordinate_conversion_module.hpc_to_hgs([sky_x_arc.value, sky_y_arc.value], data_map)

#                 x_hpc = (x_mean - raw_header['CRPIX1']) * arcsec_per_pixel

#                 y_hpc = (y_mean - raw_header['CRPIX2']) * arcsec_per_pixel


#                 # print(sky_x_arc.value, sky_y_arc.value, these_pixels_work[0], these_pixels_work[1], x_hgs, y_hgs)

#                 # sky_skycoord_obj = SkyCoord(sky_x_arc.Tx.value, sky_y_arc.Tx.value, frame = data_map.coordinate_frame)

#                 # hpc_x, hpc_y = sky_skycoord_obj.Tx.value, sky_skycoord_obj.Ty.value


#                 # , sky_x_arc.value, these_pixels_work[0], these_pixels_work[1], x_hgs, y_hgs

                

#                 copy_dict.update({
#                                 'dbscan_1_x_hpc': x_hpc,
#                                 'dbscan_1_y_hpc': y_hpc,
#                                 'dbscan_1_x_pix': x_mean,
#                                 'dbscan_1_y_pix': y_mean,
#                                 'dbscan_1_num_members': num_of_pix_members_in_cluster,
#                                 'dbscan_1_label': label, 
#                                 'peakfinder_time_1': peakfinder_time,
#                                 'dbscan_time_1': dbscan_time, 
#                                 'cleaning_time_1': clean_time})

#                 centroid_array.append(copy_dict)
#         else:
#             #returns this dictionary if there are no peaks in this image
#             copy_dict.update({
#             'dbscan_1_x_hpc': np.nan,
#             'dbscan_1_y_hpc': np.nan,
#             'dbscan_1_x_pix': np.nan,
#             'dbscan_1_y_pix': np.nan,
#             'dbscan_1_num_members': np.nan,
#             'dbscan_1_label': np.nan,
#             'peakfinder_time_1': np.nan, 
#             'dbscan_time_1': np.nan, 
#             'cleaning_time_1': np.nan})

#             centroid_array.append(copy_dict)

#     else:
#         # returns this dictionary if QUALITY of passed image not good
#         # this, in theory, should never happen because we only pass 
#         # QUALITY == 0 and data_level = lev2 data into the ALEXIS pipeline
#         copy_dict.update({
#         'dbscan_1_x_hpc': np.nan,
#         'dbscan_1_y_hpc': np.nan,
#         'dbscan_1_x_pix': np.nan,
#         'dbscan_1_y_pix': np.nan,
#         'dbscan_1_num_members': np.nan,
#         'dbscan_1_label': np.nan, 
#         'peakfinder_time_1': np.nan, 
#         'dbscan_time_1': np.nan, 
#         'cleaning_time_1': np.nan})

#         centroid_array.append(copy_dict)

#     # export an array of config dictionaries for each cluster found by the peakfinder/dbscan. 
#     # Config output coordinates in arcseconds helioprojective carrington

#     # end_time = time.time()

    

#     pickle.dump(centroid_array, open(outfile, 'wb'))
