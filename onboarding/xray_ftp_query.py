import sys  
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

from datetime import timedelta, date
import pickle
from ruffus import *
import os
import xarray as xr
import pandas as pd
import numpy as np
import re

import dataconfig

WORKING_DIR = dataconfig.DATA_DIR_GOES_FLUX

DATA_PRODUCT_DIR = dataconfig.DATA_DIR_PRODUCTS

tw = lambda x: os.path.join(WORKING_DIR, x)



###### REVIEW AND OPTIMIZE Jan17,2020 ###########


# some dictionaries we'll use sometimes
instruments = ['goes13', 'goes14', 'goes15']

inst_dict = {'goes13': 'g13', 'goes14': 'g14', 'goes15': 'g15'}

ftp_prename = 'ftp://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs'


#### set the date limits of the data you want to download

# start_download_from = input('Insert date and time you would like your XRay Flux Data to START from. For example: YYYY-MM-DD HH:MM:SS ' )

# end_download_at = input('Insert date and time you would like your XRay Flux Data to END at. For example: YYYY-MM-DD HH:MM:SS ' )

start_download_from = '2010-05-01T00:00:00'
end_download_at = '2010-05-03T00:00:00'

START_DATE_TIME = pd.Timestamp(start_download_from, tz = 'utc')

END_DATE_TIME = pd.Timestamp(end_download_at, tz = 'utc')


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)


all_dates_list = []

# start_dt = date(2010, 5, 1)
# end_dt = date(2020,3,5)
# for dt in daterange(start_dt, end_dt):
#     # all_dates_list.append(dt.strftime("%Y%m%d"))
#     all_dates_list.append(dt)


for dt in daterange(START_DATE_TIME, END_DATE_TIME):
    # all_dates_list.append(dt.strftime("%Y%m%d"))
    all_dates_list.append(dt)


ftp_name_dict = []

for full_instrument_name in instruments:

    for this_date in all_dates_list:

        specific_date_str = this_date.strftime("%Y%m%d")

        year, month, day = this_date.strftime('%Y'), this_date.strftime('%m'), this_date.strftime('%d')

        specific_file_name = f'sci_gxrs-l2-irrad_{inst_dict[full_instrument_name]}_d{specific_date_str}_v0-0-0.nc'

        ftp_name = f'{ftp_prename}/{full_instrument_name}/gxrs-l2-irrad_science/{year}/{month}/{specific_file_name}'

        outfile_str = tw(f'{year}-{month}-{day}_{full_instrument_name}_irrad.nc')

        wget_log_file = f'{outfile_str}.wget_log'

        ftp_name_dict.append({'download_file': ftp_name, 'out_name': outfile_str, 'wget_log_file': wget_log_file })

# print(ftp_name_dict)


# WORKING_DIR = '/data/padialjr/jorge-helio/goes_data/'
# tw = lambda x: os.path.join(WORKING_DIR, x)

def dl_params():
    for dictionary_element in ftp_name_dict:

        infile = None

        ftp_name = dictionary_element['download_file']

        out_name = dictionary_element['out_name']

        outfile = tw("{}.query".format(out_name))

        wget_log_file = dictionary_element['wget_log_file']

        yield(infile, outfile, ftp_name, out_name, wget_log_file)

@mkdir(WORKING_DIR)
@files(dl_params)
def make_request(infile, outfile, ftp_name, out_name, wget_log_file):

	"""
    make ruffus checkpoint of each individual daily xray data query.

	"""

	pickle.dump({'ftp_name': ftp_name, 'outname': out_name, 'wget_log_file': wget_log_file}, open(outfile, 'wb'))

@transform(make_request, suffix('.query'), ".downloaded")
def download_data(infile, outfile):

    """ download data """

    queries = pickle.load(open(infile, 'rb'))

    ftp_query_name = queries['ftp_name']

    outfile_name = queries['outname']

    wget_log_file = queries['wget_log_file']

    download_str = f'wget -e robots=off --recursive --no-parent -A --directory-prefix {WORKING_DIR} --no-directories --verbose False {ftp_query_name} -O {outfile_name} -o {wget_log_file}' 

    os.system(download_str)


        ############### END EDIT ###################


    pickle.dump({'infile': infile}, open(outfile, 'wb'))





# # from datetime import datetime as dt

raw_data_cdf4 = [ (f'{WORKING_DIR}/*goes13_irrad.nc'), (f'{WORKING_DIR}/*goes14_irrad.nc'), (f'{WORKING_DIR}/*goes15_irrad.nc')]


@transform(raw_data_cdf4, suffix(".nc"), "_clean_data.timestamp.df.pickle")
def dataframe_cleanup(infile, outfile):
    

    # print('start {} -> {}'.format(infile,outfile))

    try:

        data_w_meta = xr.open_dataset(infile)

        #the following converts raw 'time' index into a pandas timestamp column

        input_data = data_w_meta.to_dataframe().reset_index().sort_values(by = 'time')

        # input_data = pickle.load(open(infile, 'rb'))

        # test_df = pd.DataFrame(columns = ['date_time', 'time_stamp', 'instrument', 'wavelength', 'value'])


        ###### EDIT ####
            # include in helio reg exp module
        instrument = re.findall(r'(?<=)goes\d+', infile)[0]


        ##### END EDIT #####

        # instrument = infile[48:54]

        # instrument = infile[11:17]

        wavelength_array = ['A', 'B']

        for wavelength in wavelength_array:

            if wavelength == 'A':

                int_df_a = input_data[(input_data['a_flags'] == 0.0) &  (input_data['a_flux'] > 0)]

                date_time_w_timezone = int_df_a.time.dt.tz_localize('UTC')

                unix_timestamp = (date_time_w_timezone - pd.Timestamp("1970-01-01", tz = 'UTC')) / pd.Timedelta('1s')

                data_dict = {'time_stamp':  unix_timestamp,
                            # 'date_time': date_time_w_timezone,
                            'instrument': instrument, 
                            'wavelength': wavelength, 
                            'value': int_df_a['a_flux']
                            }
                            
                test_df_a = pd.DataFrame(data_dict)

            if wavelength == 'B':

                int_df_b = input_data[(input_data['b_flags'] == 0.0) & (input_data['b_flux'] > 0)]

                date_time_w_timezone = int_df_b.time.dt.tz_localize('UTC')

                unix_timestamp = (date_time_w_timezone - pd.Timestamp("1970-01-01", tz = 'UTC')) / pd.Timedelta('1s')

                data_dict = {'time_stamp': unix_timestamp,
                            # 'date_time': date_time_w_timezone,
                            'instrument': instrument, 
                            'wavelength': wavelength, 
                            'value': int_df_b['b_flux']
                            }

                test_df_b = pd.DataFrame(data_dict)

        concatenated_df = pd.concat([test_df_a, test_df_b])

        sorted_df = concatenated_df.sort_values('time_stamp').reset_index(drop = True)

        pickle.dump(sorted_df, open(outfile, 'wb'))

    except Exception as e: 

        no_data_df = pd.DataFrame(columns = ['time_stamp', 'instrument', 'wavelength', 'value'])
        
        # no_data_df = pd.DataFrame(columns = ['date_time','time_stamp', 'instrument', 'wavelength', 'value'])

        pickle.dump(no_data_df, open(outfile, 'wb'))




### CREATE SQLlite DB ###

import sqlalchemy as sa
import pandas as pd
# import os 
# import fastparquet
# #os.chdir('/data/padialjr/jorge-helio/data/goes_data/')

# data = pd.read_parquet('/data/padialjr/jorge-helio/goes_data/2010_2021_xray_data.df.timestamp.merged.parquet', engine='auto' )


engine = sa.create_engine(f'sqlite:////{DATA_PRODUCT_DIR}/timestamp_xrayDB.db', echo = True)

# Get the absolute path of the directory above the current working directory
# current_dir = os.getcwd()
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# # Construct the path to the database
# DATA_PRODUCT_DIR = os.path.join(parent_dir, 'timestamp_xrayDB.db')

metadata = sa.MetaData()

goes_data = sa.Table('goes_data', metadata, 
        sa.Column('time_stamp', sa.types.Float(), index = True),  
        sa.Column('instrument', sa.types.String(), index = True), 
        sa.Column('wavelength', sa.types.String(), index = True), 
        sa.Column('value', sa.types.Float(), index = True)
                        )


@mkdir(dataconfig.DATA_DIR_PRODUCTS)
@transform(dataframe_cleanup, suffix('_clean_data.timestamp.df.pickle'), '_sqlite.inserted')
def insert_flux_to_sqlite(infile, outfile):

    chunk = pickle.load(open(infile, 'rb'))

    data_dict = chunk.to_dict('records')

    metadata.create_all(engine)

    conn = engine.connect()
    
    conn.execute(goes_data.insert(), data_dict)

    who_inserted = pd.DataFrame({'timestamps': chunk['time_stamp'], 'inst':chunk['instrument']})

    pickle.dump(who_inserted, open(outfile, 'wb'))



if __name__ == "__main__":
    pipeline_run([download_data], multiprocess = 2, verbose = 4) 

    pipeline_run([insert_flux_to_sqlite], multiprocess= 14, verbose = 4)


# # pipeline_run([dataframe_cleanup], multiprocess = 14)

# # @merge(dataframe_cleanup, '/data/padialjr/jorge-helio/goes_data/2010_2021_xray_data.df.timestamp.merged.parquet')
# # def merge_data_per_day(infiles,outfile): 

# #     merged_df_list = []

# #     for infile in infiles:

# #         input_data = pickle.load(open(infile, 'rb'))

# #         merged_df_list.append(input_data)
        
# #     merged_df = pd.concat(merged_df_list)

# #     sorted_merged_df = merged_df.sort_values('time_stamp').reset_index(drop = True)

# #     # merged_df = merged_df.reset_index(drop = True)

# #     sorted_merged_df.to_parquet(outfile, allow_truncated_timestamps=True, coerce_timestamps = 'ms', engine = 'auto')

# #     # merged_df.to_parquet(outfile, allow_truncated_timestamps=True, engine = 'auto')
        
# # if __name__ == "__main__":
# #     pipeline_run([download_data], multiprocess = 14) 

# #     pipeline_run([download_data, merge_data_per_day], multiprocess= 14, verbose = 4)

# ### CREATE SQLlite DB ###

# import sqlalchemy as sa
# import pandas as pd
# import os 
# import fastparquet
# #os.chdir('/data/padialjr/jorge-helio/data/goes_data/')

# data = pd.read_parquet('/data/padialjr/jorge-helio/goes_data/2010_2021_xray_data.df.timestamp.merged.parquet', engine='auto' )

# engine = sa.create_engine('sqlite:////data/padialjr/jorge-helio/data_products/timestamp_xrayDB.db', echo = True)

# metadata = sa.MetaData()

# goes_data = sa.Table('goes_data', metadata, 
# 		sa.Column('time_stamp', sa.types.Float(), index = True),  
# 		sa.Column('instrument', sa.types.String(), index = True), 
# 		sa.Column('wavelength', sa.types.String(), index = True), 
# 		sa.Column('value', sa.types.Float(), index = True)
# 						)

# metadata.create_all(engine)

# i = 0 
# while i<= len(data['time_stamp']):
# 	j = i + 10000

# 	chunk = data[i:j]

# 	data_dict = chunk.to_dict('records')

# 	i = i + 10000

# 	conn = engine.connect()
	
# 	result = conn.execute(goes_data.insert(), data_dict)


# import sqlalchemy as sql
# from datetime import datetime, timedelta
# import time
# from scipy import interpolate

# @transform(raw_data_cdf4, suffix(".nc"), "_only_data.df.pickle")
# def load_cdf4_to_pickle(infile, outfile):

#     try:

#         data_w_meta = xr.open_dataset(infile)

#         #the following converts raw 'time' index into a pandas timestamp column

#         data_wo_meta = data_w_meta.to_dataframe().reset_index().sort_values(by = 'time')

#         # data_wo_meta.reset_index(inplace=True)

#         # data_wo_meta['time'] = pd.to_datetime(data_wo_meta['time'], format = '%Y-%m-%d %H:%M:%S')

#         # print('has data')

#         pickle.dump(data_wo_meta, open(outfile,'wb'))
    
#     except Exception as e: 
        
#         # print(e)

#         # print('exception')

#         no_data_dict = [{'time': np.nan, 'a_counts': np.nan, 'b_counts': np.nan, 'a_flux': np.nan, 'b_flux': np.nan, 'a_flags': np.nan, 'b_flags': np.nan, 'a_swpc_flags': np.nan, 'b_swpc_flags': np.nan}]

#         no_data_df = pd.DataFrame(no_data_dict)

#         pickle.dump(no_data_df, open(outfile,'wb'))


# pipeline_run([load_cdf4_to_pickle], multiprocess = 14)

# WORKING_DIR = '/data/padialjr/jorge-helio/goes_data'
# #### make date arrays

# start_dates = pd.date_range(start='5/1/2010', end='3/1/2020', freq = 'MS',closed = None)
# end_dates = pd.date_range(start = '5/31/2010', end = '3/31/2020', freq = 'M', closed = None)

# # print(start_dates)
# # print(end_dates)

# ### define start and end datetime objects for sql
# date_dict = []
# for i in range(0,len(start_dates)):
# 	date_dict.append({
# 		'start_date': start_dates[i],

# 		'end_date': end_dates[i]
# 		})

# date_df = pd.DataFrame(date_dict)
# # start_time = datetime(start_year,start_month,start_day,0,0,0,0)
# # end_time = datetime(end_year,end_month,end_day,0,0,0,0)

# instrument_list = ['goes13', 'goes14', 'goes15']

# for instrument in instrument_list:

#     for i in range(0, len(date_df.start_date)):

#         try:

#             print('-------------------------------')

#             # define new time range

#             start_time = datetime(date_df['start_date'][i].year,date_df['start_date'][i].month,date_df['start_date'][i].day,0,0,0,0)
#             end_time = datetime(date_df['end_date'][i].year,date_df['end_date'][i].month,date_df['end_date'][i].day,23,59,59,0)

#             # query for flux data only for wavelength B

#             engine = sql.create_engine('sqlite:////data/padialjr/jorge-helio/data_products/:xrayDB.db')

#             metadata = sql.MetaData()

#             conn = engine.connect()

#             goes_data = sql.Table('goes_data', metadata, autoload = True, autoload_with = engine)

#             # instrument = 'goes13'

#             flux_query = sql.select([goes_data]).where(sql.and_(
#                                     goes_data.c.timestamp >= start_time, 
#                                     goes_data.c.timestamp < end_time 
#                                     # goes_data.c.wavelength == 'B', 
#                                     # goes_data.c.instrument == instrument
#                                             )
#                                 )

#             flux_sql_result_all = pd.read_sql(flux_query,conn)

#             flux_data = flux_sql_result_all[(flux_sql_result_all.instrument == instrument) & (flux_sql_result_all.wavelength == 'B')]

#             timestamp_object = []
            
#             for datetime_object in flux_data['timestamp']:
#                 timestamp_object.append(datetime.timestamp(datetime_object))
            
#             # interpolate with scipy.interpolate with no smooting (s = 0) and linear polynomial (k = 1)

#             print('starting interpolation') 

#             linear_interpolation = interpolate.splrep(timestamp_object, flux_data['value'], s = 0, k = 1)
            
#             # make new datetimes range where we are going to interpolate our data ; determine the frequency of our timesteps
            
#             new_time_datetime = pd.date_range(start = start_time, end = end_time, freq = '3min')
            
#             # convert datetime range into timestamps
            
#             new_time_timestamp = []
            
#             for datetime_object in new_time_datetime:
#                 new_time_timestamp.append(datetime.timestamp(datetime_object))
            
#             # interpolate the new value for flux at new timesteps
#             new_value = interpolate.splev(new_time_timestamp, linear_interpolation, der = 0)
            
            
#             # create interpolated flux value df
            
#             new_flux_dict = {'datetime': new_time_datetime, 'timestamp': new_time_timestamp, 'instrument': instrument, 'new_value': new_value }
            
#             new_flux_data = pd.DataFrame(new_flux_dict)		

#             pickle.dump(new_flux_data, open('{}/{}_{}_{}_newflux.pickle'.format(WORKING_DIR, start_time.year, end_time.month, instrument), 'wb'))

#             print('ended interpolation and new flux data pickle dump')

#             # zero crossing technique

#             # define derivative array of size (n-1)

#             print('started calculating zero crossings')
            
#             y_diff = np.diff(new_flux_data['new_value'])
            
#             # find zero crossing and filter for positive slope value
        
                                        
#             zero_crossings = []
            
                                        
#             for i in range(1, len(y_diff)):
#                 if (y_diff[i-1] > 0) and (y_diff[i]) <=0:
#                     if y_diff[i-1] > 1e-8:# and new_flux_data['new_value'][i] >= 1.0e-6 and new_flux_data['new_value'][i] < 9.9e-6:
#                         zero_crossings.append(i)
                                        
#             # find datetime associated to interger index where a zero crossing occured
#             z_crossing_datetime = []
                                        
#             for i in zero_crossings:
#                 z_crossing_datetime.append(new_flux_data['datetime'][i])
            
#             z_crossing_timestamp = []
            
#             for i in zero_crossings:
#                 z_crossing_timestamp.append(new_flux_data['timestamp'][i])
                
                                        
#             # find flux associated to that zero crossing
                
#             zero_crossing_inter_flux = []
                                        
#             for i in zero_crossings:
#                 zero_crossing_inter_flux.append(new_flux_data['new_value'][i])
                
#             zero_cross_dict = { 'datetime': z_crossing_datetime, 'timestamp': z_crossing_timestamp, 'instrument': instrument, 'value': zero_crossing_inter_flux}
                
#             zero_cross_df = pd.DataFrame(zero_cross_dict)

#             pickle.dump(zero_cross_df, open('{}/{}_{}_{}_zero_cross_df.pickle'.format(WORKING_DIR,start_time.year, end_time.month, instrument), 'wb'))


#             print('end calculating and pickle dump for zero crossings df')
        
#             print('yes data and done making all DF for {} - {} - {}'.format(start_time.year, start_time.month, instrument))



#         # except Exception as e: print(e)
#         except:
#             # print('-------------------------------')
#             print('no data for {} - {} - {}'.format(start_time.year, start_time.month, instrument))

# print(ftp_name_list)   


    


# print(DL_DATASETS)
# wget -e robots=off --recursive --no-parent -A --directory-prefix goes_data --no-directories --verbose False ftp://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/goes13/gxrs-l2-irrad_science/2013/06/sci_gxrs-l2-irrad_g13_d20130601_v0-0-0.nc -O goes_data/this_file.nc






# wget -e robots=off --recursive --no-parent -A sci*xrs*l2*irrad*.nc --directory-prefix goes_data --no-directories --verbose False ftp://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/goes13

# wget -e robots=off --recursive --no-parent -A sci*xrs*l2*irrad*.nc --directory-prefix goes_data --no-directories --verbose False ftp://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/goes14 
# wget -e robots=off --recursive --no-parent -A sci*xrs*l2*irrad*.nc --directory-prefix goes_data --no-directories --verbose False ftp://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/goes15 

# wget -e robots=off --recursive --no-parent -A sci_xrs*l2*.nc --directory-prefix goes_data --no-directories --verbose False http://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/
# wget -e robots=off --recursive --no-parent -A sci_xrs*l2*.nc --directory-prefix goes_data --no-directories --verbose False http://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes17/l2/data/xrsf-l2-avg1m_science/

# wget -e robots=off --recursive --no-parent -A sci*xrs*l2*flsum*.nc --directory-prefix goes_data --no-directories --verbose False ftp://satdat.ngdc.noaa.gov/sem/goes/data/science/xrs/goes13 