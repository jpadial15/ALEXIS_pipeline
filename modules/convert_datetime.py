
from datetime import datetime, timedelta
from astropy.time import Time
import pandas as pd

def astropytime_to_pythondatetime(time_object):

    date_time_str = time_object.strftime('%Y-%m-%dT%H:%M:%S.%f')

    datetime_obj = pd.Timestamp(date_time_str,tz = 'utc')
    
    return(datetime_obj)

def str_to_pythondatetime(time_str):

    datetime_obj = datetime.strptime(time_str,'%Y-%m-%dT%H:%M:%S.%f' )

    return(datetime_obj)

def pythondatetime_to_astropytime(datetime_obj):

    astropy_time_obj = Time(datetime_obj, scale = 'utc')

    return(astropy_time_obj)

def str_to_astropytime(time_str):

    astropy_time_obj = Time(time_str, scale = 'utc')

    return(astropy_time_obj)

def pythondatetime_to_timestamp(datetime_obj):
    """
    Return POSIX timestamp corresponding to the datetime instance. The return value is a float similar to that returned by time.time().
    
    """
    timestamp = datetime.timestamp(datetime_obj)

    return(timestamp)

def pythondatetime_to_str(datetime_obj):

    datetime_str = datetime.strftime(datetime_obj, format = '%Y-%m-%dT%H:%M:%S.%f')

    return(datetime_str)


def convert_timestamp_to_datetime(timestamp_to_convert):

    """ 
    Takes in a time_stamp.

    Returns date_time. 

    """
    
    date_time = pd.to_datetime(timestamp_to_convert, unit='s', utc = True)
    
    return(date_time)


def convert_datetime_to_timestamp(datetime_to_convert):

    """ 
    Takes in a date_time.

    Returns time_stamp 

    """

    unix_timestamp = (datetime_to_convert - pd.Timestamp("1970-01-01", tz = 'UTC')) / pd.Timedelta('1s')


    return(unix_timestamp)


if __name__ == "__main__":
    main()