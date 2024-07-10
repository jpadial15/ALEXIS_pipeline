import pytz
from datetime import datetime
# Get the CST timezone
cst = pytz.timezone('US/Central')

load_data_start = datetime.now(cst)
from astropy.io import fits
from reproject import reproject_interp

import pandas as pd
import os
import numpy as np
# from matplotlib.path import Path

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

from scipy import interpolate

import json

load_data_end = datetime.now(cst)

print(f'loaded libraries sxi module: {load_data_end - load_data_start}')


def open_fits_aia(data):
    load_data_start = datetime.now(cst)
    with fits.open(data) as hdul:
        
        hdul.verify('fix')
        
        raw_data = hdul[1].data
        
        raw_header = hdul[1].header

        
    load_data_end = datetime.now(cst)

    print(f'open aia in sxi_module: {load_data_end - load_data_start}')
        
    return(raw_data,raw_header)

def open_fits_sxi(data):
    load_data_start = datetime.now(cst)
    with fits.open(data) as hdul:
        
        hdul.verify('fix')
        
        raw_data = hdul[0].data
        
        raw_header = hdul[0].header
        
    load_data_end = datetime.now(cst)

    print(f'open SXI in sxi_module: {load_data_end - load_data_start}')   
     
    return(raw_data,raw_header)

def organize_df(dataframe, column):

    df = dataframe.sort_values(by = column).reset_index(drop = True)

    return(df)


def join_file_to_directory(this_file, that_directory):

    full_path = lambda x: os.path.join(that_directory, this_file)

    return(full_path)

def resize_sxi_data(sxi_data):

    increase_y_sxi = np.vstack([np.zeros((100,sxi_data.shape[1])), sxi_data, np.zeros((100,sxi_data.shape[1]))])

    sxi_reset_axis = np.hstack([np.zeros((increase_y_sxi.shape[0],100)), increase_y_sxi, np.zeros((increase_y_sxi.shape[0],100))])

    return(sxi_reset_axis)

def find_solar_borders(resized_sxi_data):

    # outline edges
    edges = canny(resized_sxi_data, sigma=3, low_threshold=30, high_threshold=50)

    # edges
    hough_radii = np.arange(189, 225, 1)

    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 2 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    return(cx, cy, radii)

def crop_sxi_data(resized_sxi_data, resized_x, resized_y):

    cropped_im = resized_sxi_data[np.int(resized_y - 256):np.int(resized_y + 256), np.int(resized_x - 256): np.int(resized_x + 256)]

    return(cropped_im)

def convert_ang_radius_pix_to_km(pix_angular_radius, header):

    d_sun = 1.503*10**8 #km

    R_INSTRUMENT_KM = pix_angular_radius*header['CDELT1'] * (1/3600) * (np.pi/180) * d_sun

    return(R_INSTRUMENT_KM)


def convert_ang_radius_pix_to_arcsec(pix_angular_radius, header):

    d_sun = 1.503*10**8 #km

    R_INSTRUMENT_ARCSEC = pix_angular_radius*header['CDELT1'] #* (1/3600) * (np.pi/180) * d_sun

    return(R_INSTRUMENT_ARCSEC)



def fix_sxi_header(input_data, input_header):

    new_sxi_header = input_header.copy()

    new_sxi_header['CTYPE1'] = 'HPLN-TAN'

    new_sxi_header['CTYPE2'] = 'HPLT-TAN'

    new_sxi_header['CUNIT1'] = 'arcsec'

    new_sxi_header['CUNIT2'] = 'arcsec'

    new_sxi_header['DSUN_REF'] = 149597870691

    new_sxi_header['RSUN_REF'] = 696000000

    new_sxi_header['wavelength'] =  new_sxi_header['WAVELNTH']

    del new_sxi_header['WAVELNTH']


    # Returns two circles:
    ## (1) radius ~ solar radius
    ## (2) radius that seems to be the extent of the corona
    pixel_x, pixel_y, radii = find_solar_borders(input_data)

    # R_sxi_km = convert_ang_radius_pix_to_km(radii[0], sxi_header)

    R_sxi_arcsec = convert_ang_radius_pix_to_arcsec(radii[0], new_sxi_header)

    new_sxi_header['RSUN'] = radii[0]

    new_sxi_header['RSUN_OBS'] = R_sxi_arcsec


    #check that center pixel is center of 512X512 array
    # if (pixel_x[0], pixel_y[0]) != (256,256):
    #     print(f'Error: (pixel_x[0], pixel_y[1]) == {(pixel_x[0], pixel_y[1])}')
    
    # else:
    new_sxi_header['CRPIX1'] = pixel_x[0]
    new_sxi_header['CRPIX2'] = pixel_y[0]

    # # center of image relative to sun
    # x_cen = ((sxi_header['CRPIX1'] - 256)**2)**(1/2)
    # y_cen = ((sxi_header['CRPIX2'] - 256)**2)**(1/2)

    new_sxi_header['XCEN'] = 0
    new_sxi_header['YCEN'] = 0
    
    new_sxi_header['NAXIS1'] = 512

    new_sxi_header['NAXIS2'] = 512

    new_sxi_header['observer'] = 'Helioprojective'

    new_sxi_header['DATE-OBS'] = new_sxi_header['DATE_OBS']

    ###### CHECK FOR CCD READOUT CONTAMINATION

    x = input_data[:,490:]


    if (x.std()) > 700:
        #bad CCD Quali

        new_sxi_header['BAD_CCD_READOUT'] = 1
    else:
        new_sxi_header['BAD_CCD_READOUT'] = 0



    ######


    return(new_sxi_header)

def reproject_images(data_from, header_from, header_to):
    array, footprint = reproject_interp((data_from, header_from), (header_to))
    return(array)

def clean_this_reprojection(reprojected_data):

    take_nans_out = np.nan_to_num(reprojected_data, copy=True, nan=0.0, posinf=None, neginf=None)

    take_nans_out[take_nans_out < 0] = 0

    return(take_nans_out)

def min_max_norm(data):

    data_min = data.min()

    data_max = data.max()

    norm = (data-data_min)/ (data_max - data_min)

    return(norm)  

def convert_aia_counts_into_flux(input_data, input_header):

    flux = input_data * (1/input_header['DN_GAIN'])* (1/input_header['EXPTIME']) * (1/input_header['EFF_AREA'])

    #units returned are (photon / (cm^2 * seconds))

    return(flux)

def fit_the_data(list_of_timestamps, list_of_xray_flux):

    """ 
    Takes in list of timestamps and list of raw X-Ray values

    Returns list of spline fit of the raw xray fit.

    """

    linear_interpolation = interpolate.splrep(list_of_timestamps, list_of_xray_flux, s = 0, k = 1)

    return(linear_interpolation)


def create_resampling_rate_df(start_datetime, end_datetime, resample_freq = '1min'):

    """
    Takes in a start_datetime and end_datetime with a default resample_freq = '1min'

    Creates a pd.date_range from start_time to end_time at resample_freq for tz = utc

    returns df of resampled_date_time and resampled_time_stamp 
    """

    new_time_datetime = pd.date_range(start = start_datetime, end = end_datetime, freq = resample_freq, tz = 'UTC')

    # convert new datetime range into timestamps
            
    resample_list_of_dict = []
    
    for datetime_obj_to_convert in new_time_datetime:
        
        unix_timestamp = (datetime_obj_to_convert - pd.Timestamp("1970-01-01", tz = 'UTC')) // pd.Timedelta('1s')

        resample_list_of_dict.append({'resampled_date_time': datetime_obj_to_convert, 
                                    'resampled_time_stamp': unix_timestamp})

    
    resampling_df = pd.DataFrame(resample_list_of_dict).sort_values(by = 'resampled_date_time').reset_index(drop = True)

    return(resampling_df)


def resample_fitted_data(timestamp_list, fitted_data):

    """
    Takes in a list of resampled timestamps from the "create_resampling_rate" function

    paired with a list of the spline fitted data and returns
    """


    new_value = interpolate.splev(timestamp_list, fitted_data, der = 0)

    return(new_value)

def zerocross_sxi_flux(resampled_df, resampled_fitted_data, this_filter):

    y_diff = np.diff(resampled_fitted_data)

    zero_crossings_list_dict = []
               
    for i in range(1, len(y_diff)):
        if (y_diff[i-1] >= 0) and (y_diff[i]) < 0:
                zero_crossings_list_dict.append({'zerocross_date_time': resampled_df.resampled_date_time.iloc[i],
                                        'zerocross_time_stamp': resampled_df.resampled_time_stamp.iloc[i],
                                        'resampled_value': resampled_fitted_data[i],
                                        'filter': this_filter})
    
    zerocross_df = pd.DataFrame(zero_crossings_list_dict)

    # print(zerocross_df)
    # zerocross_df = pd.DataFrame(zero_crossings_list_dict).sort_values(by = 'zerocross_date_time').reset_index(drop = True)

    return(zerocross_df)

def test_if_sun_borders_ok(center_x, center_y, input_header):

    x_min_lim, x_max_lim = center_x - 20 , center_x + 20
    y_min_lim, y_max_lim = center_y - 20 , center_y + 20

    x_pix, y_pix = input_header['CRPIX1'], input_header['CRPIX2']

    if (x_pix >= x_min_lim) & (x_pix <= x_max_lim) & (y_pix >= y_min_lim) & (y_pix <= y_max_lim):
        return(True)
    else:
        return(False)

def json_serialize(serial_this):

    return(json.dumps(serial_this))

def json_deserialize(deserial_this):

    return(json.loads(deserial_this))


def make_square_region(reg_center, pixel_distance):

    list = []

    x_min, x_max = reg_center[0] - pixel_distance, reg_center[0] + pixel_distance

    y_min, y_max = reg_center[1] - pixel_distance, reg_center[1] + pixel_distance

    list = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]

    return(list)

def make_square_region_v2(reg_center, pixel_distance):

    list = []

    x_min, x_max = reg_center[0] - pixel_distance, reg_center[0] + pixel_distance

    y_min, y_max = reg_center[1] - pixel_distance, reg_center[1] + pixel_distance

    list = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min] ]

    return(list)

def create_region_mask(region_limit_coord_pair_list):

    tupVerts=region_limit_coord_pair_list
# tupVerts=list(points_in_circle_np(50, 150,225))
    tupVerts.append(tupVerts[0])


    x, y = np.meshgrid(np.arange(512), np.arange(512)) # make a canvas with coordinates

    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T 

    p = Path(tupVerts) # make a polygon

    grid = p.contains_points(points)

    mask = grid.reshape(512,512)

    return(mask)

def create_region_mask_sxi(region_limit_coord_pair_list):

    tupVerts=region_limit_coord_pair_list
# tupVerts=list(points_in_circle_np(50, 150,225))
    tupVerts.append(tupVerts[0])


    x, y = np.meshgrid(np.arange(512), np.arange(512)) # make a canvas with coordinates

    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T 

    p = Path(tupVerts) # make a polygon

    grid = p.contains_points(points)

    mask = grid.reshape(512,512)

    return(mask)

def create_region_mask_shapeless(data, region_limit_coord_pair_list):

    tupVerts=region_limit_coord_pair_list
# tupVerts=list(points_in_circle_np(50, 150,225))
    tupVerts.append(tupVerts[0])

    shape = data.shape[0]


    x, y = np.meshgrid(np.arange(shape), np.arange(shape)) # make a canvas with coordinates

    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T 

    p = Path(tupVerts) # make a polygon

    grid = p.contains_points(points)

    mask = grid.reshape(shape,shape)

    return(mask)


def specific_sxi_wl_norm(data, min_val, max_val):

    data_min = min_val
    data_max = max_val

    norm = (data - data_min)/(data_max - data_min)

    return(norm)

# if __name__ == "__main__":
#     main()