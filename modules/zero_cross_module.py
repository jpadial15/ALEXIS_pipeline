import sys  
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

from datetime import datetime

import pandas as pd

import numpy as np

from datetime import datetime, timedelta, timezone

import time

from scipy import interpolate

import sqlalchemy as sa

from modules import dataconfig


def convert_datetime_to_timestamp(datetime_to_convert):

    """ 
    Takes in a date_time.

    Returns time_stamp 

    """

    unix_timestamp = (datetime_to_convert - pd.Timestamp("1970-01-01", tz = 'UTC')) / pd.Timedelta('1s')


    return(unix_timestamp)

def convert_timestamp_to_datetime(timestamp_to_convert):

    """ 
    Takes in a time_stamp.

    Returns date_time. 

    """
    
    date_time = pd.to_datetime(timestamp_to_convert, unit='s', utc = True)
    
    return(date_time)


def fit_the_data(list_of_timestamps, list_of_xray_flux):

    """ 
    Takes in list of timestamps and list of raw X-Ray values

    Returns list of spline fit of the raw xray fit.

    """

    linear_interpolation = interpolate.splrep(list_of_timestamps, list_of_xray_flux, s = 0, k = 1)

    return(linear_interpolation)


def create_resampling_rate_df(start_datetime, end_datetime, resample_freq = '3min'):

    """
    Takes in a start_datetime and end_datetime with a default resample_freq = '3min'

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


def zerocross_technique(resampled_df, resampled_fitted_data, this_instrument, zerocross_delimiter = 1e-8):

    """


    """

    y_diff = np.diff(resampled_fitted_data)

    zero_crossings_list_dict = []
               
    for i in range(1, len(y_diff)):
        if (y_diff[i-1] > 0) and (y_diff[i]) <=0:
            if y_diff[i-1] > zerocross_delimiter:# and new_flux_data['new_value'][i] >= 1.0e-6 and new_flux_data['new_value'][i] < 9.9e-6:
                zero_crossings_list_dict.append({'zerocross_date_time': resampled_df.resampled_date_time.iloc[i],
                                        'zerocross_time_stamp': resampled_df.resampled_time_stamp.iloc[i],
                                        'resampled_value': resampled_fitted_data[i],
                                        'instrument': this_instrument})
    
    zerocross_df = pd.DataFrame(zero_crossings_list_dict)

    # print(zerocross_df)
    # zerocross_df = pd.DataFrame(zero_crossings_list_dict).sort_values(by = 'zerocross_date_time').reset_index(drop = True)

    return(zerocross_df)


def xray_sql_db(start_date_time, end_date_time, this_instrument):

    start_unix_timestamp = convert_datetime_to_timestamp(start_date_time)

    end_unix_timestamp = convert_datetime_to_timestamp(end_date_time)

    # MAIN_DIR = '/data/padialjr/jorge-helio'

    engine = sa.create_engine(f'sqlite:///{dataconfig.DATA_DIR_PRODUCTS}/timestamp_xrayDB.db')

    metadata = sa.MetaData()

    conn = engine.connect()

    goes_data = sa.Table('goes_data', metadata, autoload = True, autoload_with = engine)

    flux_query = sa.select([goes_data]).where(sa.and_(
                                    goes_data.c.time_stamp >= start_unix_timestamp, 
                                    goes_data.c.time_stamp < end_unix_timestamp
                                            )
                                )

    flux_sql_df = pd.read_sql(flux_query,conn)

    if len(flux_sql_df) == 0:

        empty_df = pd.DataFrame(columns = ['time_stamp', 'instrument','wavelength', 'value'])

        return(empty_df)

    else:

        masked_sql_df = flux_sql_df[(flux_sql_df.wavelength == 'B') & (flux_sql_df.instrument == this_instrument)].sort_values(by = 'time_stamp').reset_index(drop = True)

        return(masked_sql_df)


def find_zero_crossings(start_date_time, end_date_time, this_instrument):

    raw_data_df = xray_sql_db(start_date_time, end_date_time, this_instrument)

    # print('this is debugging message',(len(raw_data_df)))

    if len(raw_data_df) != 0:

        is_there_data_df = pd.DataFrame([{'start_date': start_date_time, 'end_date': end_date_time, 'instrument': this_instrument, 'data': True, 'len_data': len(raw_data_df)}])
        
        try:

            spline_fit = fit_the_data(raw_data_df.time_stamp.to_list(), raw_data_df.value.to_list())

            resample_start_time_stamp, resample_end_time_stamp = raw_data_df.time_stamp.iloc[0], raw_data_df.time_stamp.iloc[-1]

            resample_start_date_time, resample_end_date_time = convert_timestamp_to_datetime(resample_start_time_stamp), convert_timestamp_to_datetime(resample_end_time_stamp)

            resampled_dataframe = create_resampling_rate_df(resample_start_date_time, resample_end_date_time, resample_freq = '3min')

            resampled_fitted_data = resample_fitted_data(resampled_dataframe.resampled_time_stamp.to_list(), spline_fit)

            zerocross_df = zerocross_technique(resampled_dataframe, resampled_fitted_data, this_instrument)

            error_df = pd.DataFrame([{'start_date': start_date_time, 'end_date': end_date_time, 'instrument': this_instrument, 'error': False}])

            # return(zerocross_df, is_there_data_df, error_df)
        
        except Exception as e:

            print(e)

            zerocross_df = pd.DataFrame(columns = ['zerocross_date_time', 'zerocross_time_stamp', 'resampled_value','instrument'])    

            error_df = pd.DataFrame([{'start_date': start_date_time, 'end_date': end_date_time, 'instrument': this_instrument, 'error': e}])

    if len(raw_data_df) == 0:

        zerocross_df = pd.DataFrame(columns = ['zerocross_date_time', 'zerocross_time_stamp', 'resampled_value','instrument'])

        is_there_data_df = pd.DataFrame([{'start_date': start_date_time, 'end_date': end_date_time, 'instrument': this_instrument, 'data': False, 'len_data': len(raw_data_df)}])

        error_df = pd.DataFrame([{'start_date': start_date_time, 'end_date': end_date_time, 'instrument': this_instrument, 'error': False}])

        # return(empty_zerocross_df, is_there_data_df, error_df)


        
        
    return(zerocross_df, is_there_data_df, error_df)




if __name__ == "__main__":
    find_zero_crossings(start_date_time, end_date_time, this_instrument)
