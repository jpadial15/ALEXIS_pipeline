import pytz
from datetime import datetime
# Get the CST timezone
cst = pytz.timezone('US/Central')

load_data_start = datetime.now(cst)
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle
# from glob import glob
import os


# from ruffus import *
# from skimage.util.dtype import convert
# import sqlalchemy as sa

# from scipy.spatial import ConvexHull
# from scipy import ndimage

# from skimage.transform import rotate

# commented something to test vscode github2

import warnings
warnings.filterwarnings("ignore")

# import convert_datetime
# import query_the_data
import dataconfig
import sxi_module
# import helio_reg_exp_module


# from astropy.wcs import WCS
# from reproject import reproject_interp
from astropy.io import fits
from astropy import units as u
import sunpy.map

from aiapy.calibrate import correct_degradation, normalize_exposure, register, update_pointing
from aiapy.calibrate.util import get_correction_table, get_pointing_table

# from time import time, sleep
# from random import randint
# import sunpy

import convert_datetime
import sys
import helio_reg_exp_module


load_data_end = datetime.now(cst)

print(f'loaded libraries clean_img_02 module: {load_data_end - load_data_start}')

def clean_sxi_data(config):

    path_to_file = config['file_path']

    raw_file_name = config['file_name']

    # print(raw_file_name)

    
    load_data_start = datetime.now(cst)

    try:


        raw_sxi_data, raw_sxi_header = sxi_module.open_fits_sxi(path_to_file)

    except TypeError as e:
    # Handle the specific TypeError with the message
        if "buffer is too small for requested array" in str(e):
            copy_dict = config.copy()

            copy_dict.update({
                    'QUALITY': 10 ,
                    'data_level': 'lev1',
                    'img_data_max': np.nan,
                    'img_data_min': np.nan,
                    'img_flux': np.nan,
                    'exp_time': np.nan})
            return(copy_dict, np.nan, np.nan)

    load_data_end = datetime.now(cst)

    print(f'Open SXI data for clean img: {load_data_end - load_data_start}')

    copy_dict = config.copy()

    # del copy_dict['WAVELNTH']

    allowed_wavelength = raw_sxi_header['WAVELNTH']

    # print(allowed_wavelength)

    if allowed_wavelength  in ['TM', 'PTHK', 'PTHNA']:

        # print('succes')

        if raw_sxi_header['IMG_CODE'] == 'FL':

            # copy_dict = config.copy()

            load_data_start = datetime.now(cst)

            pixel_x, pixel_y, radii = sxi_module.find_solar_borders(raw_sxi_data)

            load_data_end = datetime.now(cst)

            print(f'find SXI solar border from clean_img_02 module: {load_data_end - load_data_start}')

            
            # We encountered an error where we couldnt find the sun with current algorithm ... 
            # Check if we return any values for the sun before proceding.

            # print(expanded_cent_x, expanded_cent_y, expanded_radii)
            # print(raw_sxi_header['CRPIX1'], raw_sxi_header['CRPIX2'])

            if (len(pixel_x), len(pixel_y), len(radii)) == (0,0,0):
                # print(f'Data no Good: Didnt find a solar border for {raw_file_name}')
                copy_dict.update({
                                    'QUALITY': 2,
                                    'data_level': 'lev1',
                                    'img_data_max': raw_sxi_data.max(),
                                    'img_data_min': raw_sxi_data.min(),
                                    'img_flux': np.sum(raw_sxi_data),
                                    'exp_time': raw_sxi_header['EXPTIME']})
                

                return(copy_dict, raw_sxi_data, raw_sxi_header)

            if sxi_module.test_if_sun_borders_ok(pixel_x[0], pixel_y[0], raw_sxi_header) == False:
                # print(f'Data no Good: Center pixels found are more than 20x20 pixels away from raw sxi center pix for {raw_file_name}.')
                copy_dict.update({
                                    'QUALITY': 3,
                                    'data_level':'lev1',
                                    'img_data_max': raw_sxi_data.max(),
                                    'img_data_min': raw_sxi_data.min(),
                                    'img_flux': np.sum(raw_sxi_data),
                                    'exp_time': raw_sxi_header['EXPTIME'],
                                    'wavelength': raw_sxi_header['WAVELNTH']})
                
                return(copy_dict, raw_sxi_data, raw_sxi_header)


            else: # solar borders found, check for CCD bad readout

                no_borders_sxi_data = raw_sxi_data[1:511,1:579]

                axis_expanded_sxi_data = sxi_module.resize_sxi_data(no_borders_sxi_data)

                sxi_data = sxi_module.crop_sxi_data(axis_expanded_sxi_data, np.mean(pixel_x)+100, np.mean(pixel_y)+100)

                # pixel_x, pixel_y, radii = sxi_module.find_solar_borders(sxi_data)

                # rotated_data = rotate(np.float32(sxi_data), -1*raw_header['CROTA1'])

                sxi_header = sxi_module.fix_sxi_header(sxi_data, raw_sxi_header)

                

                if sxi_header['BAD_CCD_READOUT'] == 0:

                    test_hdu = fits.PrimaryHDU(sxi_data,sxi_header)

                    # WL = raw_sxi_header['WAVELNTH']

                    # clean_fits_full_path = re.sub('.FTS', f'_{WL}_clean.fits', path_to_file)

                    # test_hdu.writeto(clean_fits_full_path)

                    # del copy_dict['WAVELNTH']

                    copy_dict.update({'QUALITY': 0,
                                        'img_data_max': raw_sxi_data.max(),
                                        'img_data_min': raw_sxi_data.min(), 
                                        'data_level': 'lev2', 
                                    'img_flux': np.sum(sxi_data),
                                    'exp_time': raw_sxi_header['EXPTIME'],
                                    'wavelength': raw_sxi_header['WAVELNTH']})
                    
                    # del copy_dict['WAVELNTH']
                    # del copy_dict['url']
                    # test_hdu.close()
                    # print(copy_dict)
                    return(copy_dict, sxi_data, sxi_header)

                if sxi_header['BAD_CCD_READOUT'] == 1:

                        copy_dict.update({'QUALITY': 5,
                                            'img_data_max': raw_sxi_data.max(),
                                            'img_data_min': raw_sxi_data.min(), 
                                            'data_level': 'lev1', 
                                    'img_flux': np.sum(sxi_data),
                                    'exp_time': raw_sxi_header['EXPTIME'],
                                    'wavelength': raw_sxi_header['WAVELNTH']})
                        
                        # del copy_dict['WAVELNTH']
                        # del copy_dict['url']
                        # test_hdu.close()
                        # print(copy_dict)
                        return(copy_dict, raw_sxi_data, raw_sxi_header)
        else:
            copy_dict.update({
                                'QUALITY': 6,
                                'img_data_max': raw_sxi_data.max(),
                                'img_data_min': raw_sxi_data.min(),
                                'img_flux': np.sum(raw_sxi_data),
                                'data_level': 'lev1',
                                'exp_time': raw_sxi_header['EXPTIME'],
                                'wavelength': raw_sxi_header['WAVELNTH']})
            # del copy_dict['WAVELNTH']
            # del copy_dict['url']

            return(copy_dict, raw_sxi_data, raw_sxi_header)
    else:
        copy_dict.update({
                            'QUALITY': 1,
                            'img_data_max': raw_sxi_data.max(),
                            'img_data_min': raw_sxi_data.min(),
                            'img_flux': np.sum(raw_sxi_data),
                            'data_level': 'lev1',
                            'exp_time': raw_sxi_header['EXPTIME'],
                            'wavelength': raw_sxi_header['WAVELNTH']})
        # del copy_dict['WAVELNTH']
        # del copy_dict['url']

        return(copy_dict, raw_sxi_data, raw_sxi_header)


def clean_aia_data(config):

    # i = 0

    # print(config['file_path'])

    # return(config,config['file_path'],config['file_path'] )

    try:
        file_path = config['file_path']

        
        load_data_start = datetime.now(cst)

        raw_aia_data, raw_aia_header = sxi_module.open_fits_aia(file_path)

        if raw_aia_header['QUALITY'] != 0:

            copy_dict = config.copy()

            copy_dict.update({
                'QUALITY': raw_aia_header['QUALITY'], 
                'img_data_max': np.nan,
                'img_data_min': np.nan,
                'instrument': 'AIA',
                'telescope': 'SDO',
                'data_level': 'lev0', 
                'img_flux': np.nan,
                'exp_time': raw_aia_header['EXPTIME'],
                'wavelength': raw_aia_header['WAVELNTH']})

            return(copy_dict, raw_aia_data, raw_aia_header)
        else:
            pass


        load_data_end = datetime.now(cst)

        print(f'Open AIA imag in clean_img_02: {load_data_end - load_data_start}')

        load_data_start = datetime.now(cst)

        level_1_maps = sunpy.map.Map(raw_aia_data, raw_aia_header)

        load_data_end = datetime.now(cst)

        print(f'Make AIA map in clean_img_02: {load_data_end - load_data_start}')

        # date_time = level_1_maps.date
    except:
        print(f'{file_path} data corrupt')
        sys.exit()

    # download one 
    def make_aia_calibration_files(this_dir):

        pointing_table_file_path = f'{this_dir}/aia_pointing_table.pickle'

        correction_table_file_path = f'{this_dir}/aia_correction_table.pickle'

        CHECK_POINT_EXIST = os.path.isfile(pointing_table_file_path)

        CHECK_CORRECT_EXIST = os.path.isfile(correction_table_file_path)

        start = convert_datetime.pythondatetime_to_astropytime(pd.Timestamp('2010-05-01T00:00:00', tz = 'utc'))

        end = convert_datetime.pythondatetime_to_astropytime(pd.Timestamp('2020-05-01T23:59:59', tz = 'utc')) 


        if not CHECK_POINT_EXIST:
            print('no AIA pointing table')
            # sys.exit()
                # print('RUNNING')

            # flare_candidate_date_time = helio_reg_exp_module.date_time_from_flare_candidate_working_dir(this_dir)
            
            # start = convert_datetime.pythondatetime_to_astropytime(flare_candidate_date_time) - 3*u.h

            # end = convert_datetime.pythondatetime_to_astropytime(flare_candidate_date_time) + 3*u.h

            pointing_table = get_pointing_table(start,end)

            pickle.dump(pointing_table, open(pointing_table_file_path, 'wb'))

        if not CHECK_CORRECT_EXIST:

            print('no AIA correction table')
            # sys.exit()
                # print('RUNNING')
        # The same applies for the correction table.
            correction_table = get_correction_table()

            pickle.dump(correction_table, open(correction_table_file_path, 'wb'))



        return(pointing_table_file_path, correction_table_file_path)

    

    # ###################################################3
    # ######### CLEAN AIA DATA WITH POINTER AND CONTAMINATION####

    # print(config)
    # image_directory = config['working_dir']

    image_directory = dataconfig.DATA_DIR_IMG_DATA

    # make_aia_calibration_files

    pointing_table_path, correction_table_path = make_aia_calibration_files(image_directory)

    load_data_start = datetime.now(cst)

    pointing_table = pickle.load(open(pointing_table_path, 'rb'))
    correction_table = pickle.load(open(correction_table_path, 'rb'))

    load_data_end = datetime.now(cst)

    print(f'Load AIA point and corr in clean_img_02: {load_data_end - load_data_start}')

    # sometimes satellite metadata in header not good for poining correction
    try:

        load_data_start = datetime.now(cst)

        map_updated_pointing = update_pointing(level_1_maps, pointing_table=pointing_table)

        load_data_end = datetime.now(cst)

        print(f'Updated pointing AIA in clean_img_02: {load_data_end - load_data_start}')

    except ValueError as e:
        # print(e)
        if str(e) == "'nan' did not parse as unit: At col 0, nan is not a valid unit. Did you mean aN, nA, nN or na? If this is meant to be a custom unit, define it with 'u.def_unit'. To have it recognized inside a file reader or other code, enable it with 'u.add_enabled_units'. For details, see https://docs.astropy.org/en/latest/units/combining_and_defining.html":

            copy_dict = config.copy()

            copy_dict.update({
                'QUALITY': 9, # satx, saty, and sat rot are nan
                'img_data_max': np.nan,
                'img_data_min': np.nan,
                'instrument': 'AIA',
                'telescope': 'SDO',
                'data_level': 'lev0', 
                'img_flux': np.nan,
                'exp_time': raw_aia_header['EXPTIME'],
                'wavelength': raw_aia_header['WAVELNTH']})

            return(copy_dict, raw_aia_data, raw_aia_header)

    # sometimes registering returns an error stating that the image is not a map
    try:

        load_data_start = datetime.now(cst)

        # register the image
        map_registered = register(map_updated_pointing)

        load_data_end = datetime.now(cst)

        print(f'Register AIA in clean_img_02: {load_data_end - load_data_start}')

        # return(map_registered)
    except ValueError as e:

        if str(e) == "Input must be a full disk image.":

            copy_dict = config.copy()

            copy_dict.update({
                'QUALITY': 8, # input is not a full disk image
                'img_data_max': np.nan,
                'img_data_min': np.nan,
                'instrument': 'AIA',
                'telescope': 'SDO',
                'data_level': 'lev0', 
                'img_flux': np.nan,
                'exp_time': raw_aia_header['EXPTIME'],
                'wavelength': raw_aia_header['WAVELNTH']})

            return(copy_dict, raw_aia_data, raw_aia_header)
        

    load_data_start = datetime.now(cst)

    map_degradation = correct_degradation(map_registered, correction_table=correction_table)

    load_data_end = datetime.now(cst)

    print(f'Correct deg AIA in clean_img_02: {load_data_end - load_data_start}')
    
    load_data_start = datetime.now(cst)

    map_normalized = normalize_exposure(map_degradation)
    
    load_data_end = datetime.now(cst)

    print(f'exposure correct AIA in clean_img_02: {load_data_end - load_data_start}')

    new_header = map_normalized.fits_header

    # make a copy of the config and reorganize to follow 
    # the way that SXI config goes
    # need to fix this at making availdb 
    # by normalizing the information inserted into AIA and SXI db's

    copy_dict = config.copy()


    #check that exposure time is not zero (AKA QUALITY 65536)

    if map_degradation.fits_header['EXPTIME'] == 0:

        copy_dict.update({
                    'QUALITY': map_degradation.fits_header['QUALITY'],
                    'img_data_max': 0,
                    'img_data_min': 0,
                    'instrument': 'AIA',
                    'telescope': 'SDO',
                    'data_level': 'lev0', 
                    'img_flux': 0,
                    'exp_time': map_degradation.fits_header['EXPTIME'],
                    'wavelength': map_registered.fits_header['WAVELNTH']})

        return(copy_dict, raw_aia_data, raw_aia_header)

        

    #####################################################################
    ########### CONVERT TO FLUX ####################################

    if new_header['EFF_AREA'] and new_header['DN_GAIN'] != 'nan':

        convert_to_flux_cst = (new_header['EFF_AREA']*new_header['DN_GAIN'])**-1

        aia_flux_data = map_normalized.data * convert_to_flux_cst

        aia_data_max, aia_data_min = aia_flux_data.max(), aia_flux_data.min()

    # #create clean file path. Note: we are not saving the clean file to save disk space. 
    # # but we need to keep this to maintain uniformity with SXI cleaning columns.

        # clean_fits_full_path = re.sub('.fits', f'_clean.fits', file_path)

        CHECK_FOR_NEG_DOMINANCE = np.sum(aia_flux_data>0) - np.sum(aia_flux_data<0)
        
        if CHECK_FOR_NEG_DOMINANCE > 0:

            copy_dict.update({
                                'QUALITY': 0,
                                'img_data_max': aia_data_max,
                                'img_data_min': aia_data_min,
                                'instrument': 'AIA',
                                'telescope': 'SDO',
                                'data_level': 'lev2', 
                                'img_flux': np.sum(aia_flux_data),
                                'exp_time': map_degradation.fits_header['EXPTIME'],
                                'wavelength': map_registered.fits_header['WAVELNTH']})

        if CHECK_FOR_NEG_DOMINANCE < 0:
            copy_dict.update({'QUALITY': 4,
                'img_data_max': aia_data_max,
                'img_data_min': aia_data_min,
                'instrument': 'AIA',
                'telescope': 'SDO',
                'data_level': 'lev0', 
                'img_flux': np.sum(aia_flux_data),
                'exp_time': map_degradation.fits_header['EXPTIME'],
                'wavelength': map_registered.fits_header['WAVELNTH']})


        # print(f'aia cleaned ok for {file_path}')

        return(copy_dict, aia_flux_data, new_header)

    else:

        # cannot convert to gain and effarea

        copy_dict.update({'QUALITY': 7,
                'img_data_max': np.nan,
                'img_data_min': np.nan,
                'instrument': 'AIA',
                'telescope': 'SDO',
                'data_level':'lev0',
                'exp_time': map_degradation.fits_header['EXPTIME'],
                'wavelength': map_registered.fits_header['WAVELNTH']})

        return(copy_dict, raw_aia_data, raw_aia_header)