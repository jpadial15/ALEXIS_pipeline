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


from astropy.coordinates import SkyCoord

import scipy.stats
from scipy.stats import mstats
from sklearn.metrics import mean_squared_error


import convert_datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import clean_img_data_02

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import sxi_module
from time import sleep
from random import randint

from aiapy.calibrate.util import get_correction_table, get_pointing_table

import helio_reg_exp_module

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

from modules import coordinate_conversion_module
from modules import associate_HARP_to_flare_region_module












def find_harp_span(df):
    
    init_file = f'{df.iloc[0].working_dir}/initialize_with_these_files_df.pickle'

    init_files = pickle.load(open(init_file, 'rb'))

    masked_img = init_files[(init_files.wavelength == df.iloc[0].img_wavelength ) ]

    instrument = df.iloc[0].img_instrument

    if instrument == 'AIA':

        _, data,header = clean_img_data_02.clean_aia_data(masked_img.to_dict('records')[0])
        
        data_map = sunpy.map.Map(data, header)
        
        r_sun = header['R_SUN']

    if instrument == 'SXI':

        _, raw_data, header = clean_img_data_02.clean_sxi_data(masked_img.to_dict('records')[0])

        rotated_data = rotate(np.float32(raw_data), -1*header['CROTA1'])

        data = rotated_data/masked_img.to_dict('records')[0]['exp_time']
        
        data_map = sunpy.map.Map(data, header)
        
        r_sun = header['RSUN']
        
        
        # plot the HARP areas
    
    harp_file = f'{df.iloc[0].working_dir}/drms_harp_availability.pickle'

    harp_file_df = pickle.load(open(harp_file, 'rb'))
    
    span_of_harps_coords_dict_list = []
    for harp_num, element in harp_file_df.groupby('HARPNUM'):

        span_of_harp_over_time_hgs = associate_HARP_to_flare_region_module.return_harp_lifetime_span_hgs_bbox(element) # list_ of harp perimiter over time in hgs

        span_of_harp_pix_bbox = associate_HARP_to_flare_region_module.return_pixel_ar_drms_bbox(span_of_harp_over_time_hgs,data_map) # list of harp perimiter over time in pix
        
        span_of_harp_hpc_bbox = coordinate_conversion_module.hpc_to_pix(span_of_harp_pix_bbox, data_map.fits_header)

        span_of_harps_coords_dict_list.append({'HARPNUM': harp_num, 'span_hgs_bbox': span_of_harp_over_time_hgs, 'span_pix_bbox': span_of_harp_pix_bbox, 'span_hpc_bbox': span_of_harp_hpc_bbox})


    #create span of AR dataframe
    span_of_harps_df = pd.DataFrame(span_of_harps_coords_dict_list)
    
    span_file_name = 'span_of_available_harps.pickle'
    
#     print(f'{df.iloc[0].working_dir}/{span_file_name}')
    
    pickle.dump(span_of_harps_df, open(f'{df.iloc[0].working_dir}/{span_file_name}', 'wb'))
    
    return(span_of_harps_df, data_map)
        
    
def is_coords_outside_of_solar_borders( x_pix, y_pix, fits_header):

    z_squared = (x_pix - fits_header['CRPIX1'])**2 + (y_pix -fits_header['CRPIX2'])**2


    try: # check AIA
        check_if_wn = z_squared - fits_header['R_SUN']**2
    except: # check SXI
        check_if_wn = z_squared - fits_header['RSUN']**2

    if check_if_wn < 0:

        return False

    else:
        return True



def find_boolean_list_of_harps_that_bound_pix_coords(x_pix,y_pix, group, span_of_harps_df, hmi_drms_df):
    
    boolean_list = ([associate_HARP_to_flare_region_module.check_flare_coord_wn_AR(this_span_of_harp_pix_bbox, [x_pix,y_pix])[0] for this_span_of_harp_pix_bbox in span_of_harps_df.span_pix_bbox])


    # are coords related to any harp

    are_coords_linked_to_harp = np.sum(boolean_list) # will be zero if harp not found. Any other number if harp is found

    if are_coords_linked_to_harp != 0:

        # print(work_dir) 

        masked_active_regions_for_candidate_coords = span_of_harps_df[boolean_list]

        # go back to full ar drms df and mask all the available harps where our candidate is at
        # we do this to find the closest one in time

        list_of_spanned_harps_w_our_candidate_coord = masked_active_regions_for_candidate_coords.HARPNUM.unique()

        only_harps = pd.concat([hmi_drms_df[hmi_drms_df.HARPNUM == this_harp] for this_harp in list_of_spanned_harps_w_our_candidate_coord])

        group['HARPNUM_list'] = [only_harps.HARPNUM.unique()]

        group['num_of_HARPS_found'] = len(only_harps.HARPNUM.unique())

        group['HARP_found'] = True

#         work_group_associated_w_harps_list.append(group)

    else:

        # print(work_dir)
        group['HARPNUM_list'] = [[]]

        group['num_of_HARPS_found'] = 0

        group['HARP_found'] = False
        
    return(group)

#         work_group_associated_w_harps_list.append(group)


def pass_all_harps_found_metadata(coordinate_bbox_group, hmi_drms_df, span_of_harps_df, fits_header):
    
    if coordinate_bbox_group.iloc[0].HARP_found == True:
        
        work_dir = coordinate_bbox_group.iloc[0].working_dir

        only_these_harps = pd.concat([hmi_drms_df[hmi_drms_df.HARPNUM == this_harp] for this_harp in coordinate_bbox_group.HARPNUM_list.iloc[0]])

        masked_only_harps_list = []

        for harp, choose_wc_harp_closest_in_time in only_these_harps.groupby(['HARPNUM']):

            work_dir_date_time = helio_reg_exp_module.date_time_from_flare_candidate_working_dir(work_dir)

            choose_wc_harp_closest_in_time['time_delta'] = [np.abs((this_obs_time - work_dir_date_time).total_seconds()) for this_obs_time in choose_wc_harp_closest_in_time.obs_date_time]

            closest_harp_in_time = choose_wc_harp_closest_in_time.sort_values(by ='time_delta').iloc[0]

            masked_only_harps_list.append(closest_harp_in_time.to_dict())

        masked_only_harps = pd.DataFrame(masked_only_harps_list)

        test_df = masked_only_harps.sort_values(by = 'HARPNUM')
        
#         mask_span_harp = span_of_harps_df[span_of_harps_df.this_harp]

        # test_df

        harps_num_list = [this_harp_num for this_harp_num in test_df.HARPNUM]
        coordinate_bbox_group['HARPNUM_list'] = [harps_num_list]


        harps_hgs_list = [this_harp_hgs for this_harp_hgs in test_df.hgs_bbox]
        coordinate_bbox_group['HARP_hgs_bbox_list'] = [harps_hgs_list]


        harps_pix_bbox_list = [associate_HARP_to_flare_region_module.return_pixel_ar_drms_bbox(this_harp_hgs, fits_header) for this_harp_hgs in test_df.hgs_bbox]
        coordinate_bbox_group['HARP_pix_bbox_list'] = [harps_pix_bbox_list]            

        harps_QUALITY_list = [this_harp_QUALITY for this_harp_QUALITY in test_df.QUALITY]
        coordinate_bbox_group['HARPS_QUALITY_list'] = [harps_QUALITY_list]


        harps_NOAA_ARS_list = [this_harp_NOAA_ARS for this_harp_NOAA_ARS in test_df.NOAA_ARS]
        coordinate_bbox_group['HARPS_NOAA_ARS_list'] = [harps_NOAA_ARS_list]


        harps_AREA_list = [this_harp_AREA for this_harp_AREA in test_df.AREA]
        coordinate_bbox_group['HARPS_AREA_list'] = [harps_AREA_list]


        harps_NOAA_AR_list = [this_harp_NOAA_AR for this_harp_NOAA_AR in test_df.NOAA_AR]
        coordinate_bbox_group['HARPS_NOAA_AR_list'] = [harps_NOAA_AR_list]


        harps_NOAA_NUM_list = [this_harp_NOAA_NUM for this_harp_NOAA_NUM in test_df.NOAA_NUM]
        coordinate_bbox_group['HARPS_NOAA_NUM_list'] = [harps_NOAA_NUM_list]


        harps_obs_date_time_list = [this_harp_obs_date_time for this_harp_obs_date_time in test_df.obs_date_time]
        coordinate_bbox_group['HARPS_obs_date_time_list'] = [harps_obs_date_time_list]

        harps_obs_time_stamp_list = [this_harp_obs_time_stamp for this_harp_obs_time_stamp in test_df.obs_time_stamp]
        coordinate_bbox_group['HARPS_obs_time_stamp_list'] = [harps_obs_time_stamp_list]

        # coordinate_bbox_group['HARPS_hpc_bbox']  = [ for this_harp_num in coordinate_bbox_group.HARPNUM]

#         bbox_appended_df_list.append(coordinate_bbox_group)
    else:
        # harps_num_list = [this_harp_num for this_harp_num in test_df.HARPNUM]
        coordinate_bbox_group['HARPNUM_list'] = [[]]


        # harps_hgs_list = [this_harp_hgs for this_harp_hgs in test_df.hgs_bbox]
        coordinate_bbox_group['HARP_hgs_bbox_list'] = [[]]

        # harps_QUALITY_list = [this_harp_QUALITY for this_harp_QUALITY in test_df.QUALITY]
        coordinate_bbox_group['HARPS_QUALITY_list'] = [[]]


        # harps_NOAA_ARS_list = [this_harp_NOAA_ARS for this_harp_NOAA_ARS in test_df.NOAA_ARS]
        coordinate_bbox_group['HARPS_NOAA_ARS_list'] = [[]]


        # harps_AREA_list = [this_harp_AREA for this_harp_AREA in test_df.AREA]
        coordinate_bbox_group['HARPS_AREA_list'] = [[]]


        # harps_NOAA_AR_list = [this_harp_NOAA_AR for this_harp_NOAA_AR in test_df.NOAA_AR]
        coordinate_bbox_group['HARPS_NOAA_AR_list'] = [[]]


        # harps_NOAA_NUM_list = [this_harp_NOAA_NUM for this_harp_NOAA_NUM in test_df.NOAA_NUM]
        coordinate_bbox_group['HARPS_NOAA_NUM_list'] = [[]]


        # harps_obs_date_time_list = [this_harp_obs_date_time for this_harp_obs_date_time in test_df.obs_date_time]
        coordinate_bbox_group['HARPS_obs_date_time_list'] = [[]]

        # harps_obs_time_stamp_list = [this_harp_obs_time_stamp for this_harp_obs_time_stamp in test_df.obs_time_stamp]
        coordinate_bbox_group['HARPS_obs_time_stamp_list'] = [[]]
        
    return(coordinate_bbox_group)