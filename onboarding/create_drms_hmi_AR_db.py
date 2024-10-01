import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import drms
from ruffus import *
import os
import re
import pickle
import dataconfig
import warnings

warnings.filterwarnings("ignore")

from modules import convert_datetime
from modules import query_the_data
from modules import sxi_module
from modules import helio_reg_exp_module
from time import sleep
from random import randint
from astropy.time import Time
import sqlalchemy as sa
import pandas as pd

import random

import json





def make_hmi_jsoc_availability_query_string(time_ex_1, time_ex_2):

    time_str_1 = f'{time_ex_1:%Y.%m.%d}'

    time_str_2 = f'{time_ex_2:%Y.%m.%d}'

    query_str = f'hmi.sharp_720s[][{time_str_1} - {time_str_2}]'

    return(query_str)


def make_drms_ar_availability_filename(jsoc_query_str):

    sub2 = re.sub('hmi.sharp_720s\[', '', jsoc_query_str)

    sub3 = re.sub(' ','', sub2)

    sub4 = re.sub(']', '', sub3)

    sub5 = re.sub('\[', '', sub4)

    return(sub5)




def convert_TAI_to_UTC(TAI_STR):
    
    sub = re.sub('\.', '-',TAI_STR)

    sub2 = re.sub('_', 'T', sub)

    sub3 = re.sub('TTAI', '', sub2)

    t = Time(sub3, scale='tai')

    utc = t.utc

    pandas_timestamp = convert_datetime.astropytime_to_pythondatetime(utc)


    return(pandas_timestamp)



def create_HGS_bbox(AR_DF_ROW):

    x_min, x_max, y_min, y_max = AR_DF_ROW.LON_MIN, AR_DF_ROW.LON_MAX, AR_DF_ROW.LAT_MIN, AR_DF_ROW.LAT_MAX

    list = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]]

    return(list)


# def tai_to_utc(tai_str):




def prep_hmi_AR_drms_df(hmi_drms_df):

    copy_df = hmi_drms_df.copy()

    # convert TAI to date_time
    copy_df['obs_date_time'] = [convert_TAI_to_UTC(tai_str) for tai_str in copy_df['T_OBS']]

    copy_df['first_seen_date_time'] = [convert_TAI_to_UTC(tai_str) for tai_str in copy_df['T_FRST']]

    copy_df['last_seen_date_time'] = [convert_TAI_to_UTC(tai_str) for tai_str in copy_df['T_LAST']]


    copy_df['hgs_bbox'] = [create_HGS_bbox(row) for row in copy_df.itertuples()]

    # make timestamp columns

    copy_df['obs_time_stamp'] = [convert_datetime.convert_datetime_to_timestamp(this_date_time) for this_date_time in copy_df['obs_date_time']]

    copy_df['first_seen_time_stamp'] = [convert_datetime.convert_datetime_to_timestamp(this_date_time) for this_date_time in copy_df['first_seen_date_time']]

    copy_df['last_seen_time_stamp'] = [convert_datetime.convert_datetime_to_timestamp(this_date_time) for this_date_time in copy_df['last_seen_date_time']]

    # drop old columns that aren't needed anymore.


    output_df  = copy_df.drop(['T_OBS', 'T_FRST', 'T_LAST', 'LAT_MAX', 'LAT_MIN', 'LON_MAX', 'LON_MIN'], axis = 1)

    # PARSE OVER LIST OF NOAA_ARS. convert from str to list. 

    output_df['NOAA_ARS'] = [re.split(r',',this_str_list) for this_str_list in copy_df['NOAA_ARS']]


    return(output_df)

###################################################
jsoc_email = dataconfig.NOTIFY_EMAIL_ADDR
###################################################

client = drms.Client(email=jsoc_email)

these_key = 'T_OBS, HARPNUM,  AREA_ACR, QUALITY,NOAA_NUM, NOAA_AR, NOAA_ARS,AREA, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,T_FRST,T_LAST'



start_download_from = '2010-05-01T00:00:00'
end_download_at = '2020-03-06T00:00:00'


START_DATE_TIME = pd.Timestamp(start_download_from, tz = 'utc')

END_DATE_TIME = pd.Timestamp(end_download_at, tz = 'utc')

frequency = '1D'

datelist = pd.date_range(start = START_DATE_TIME , end = END_DATE_TIME, freq = frequency ).tolist()

this_query_list = [make_hmi_jsoc_availability_query_string(previous,current) for previous,current in zip(datelist[:], datelist[1:])]

#randomize the query list

random_query_list = random.sample(this_query_list,len(this_query_list))




WORKING_DIR = dataconfig.DATA_DIR_HMI_DRMS
DATA_PRODUCT_DIR = dataconfig.DATA_DIR_PRODUCTS

tw = lambda x: os.path.join(WORKING_DIR, x)

def dl_params():
    for jsoc_query in random_query_list:

        infile = None

        out_name = make_drms_ar_availability_filename(jsoc_query)

        outfile = tw(f'{out_name}_drms_hmi_AR.df.pickle')

        yield(infile, outfile, jsoc_query)

@mkdir(WORKING_DIR)
@files(dl_params)
def make_request(infile, outfile, jsoc_query):

    sleep(randint(4,27))

    hmi_ar_query_df = client.query(jsoc_query, key=these_key)

    hmi_ar_query_copy = hmi_ar_query_df.copy()

    if len(hmi_ar_query_copy) != 0:

        output_df = prep_hmi_AR_drms_df(hmi_ar_query_df)

        pickle.dump(output_df, open(outfile, 'wb'))
    
    else:

        print(f'no data for {jsoc_query}')

        # cache searches that are empty

        cache_dir = f'{WORKING_DIR}/no_data_dir/'

        os.makedirs(cache_dir, exist_ok=True) 

        file_reg_exp = re.findall(r'\d{4}.\d{2}.\d{2}-\d{4}.\d{2}.\d{2}_drms_hmi_AR.df.pickle', outfile )[0]

        cache_missing_file_name = os.path.join(cache_dir, file_reg_exp)

        with open(cache_missing_file_name, 'a'): os.utime(cache_missing_file_name, None)

        pickle.dump(hmi_ar_query_copy, open(outfile, 'wb'))


engine = sa.create_engine(f'sqlite:////{DATA_PRODUCT_DIR}/hmi_drms_AR.db', echo = False)

metadata = sa.MetaData()

hmi_drms_AR = sa.Table(
        'hmi_drms_AR', metadata,
        sa.Column('obs_time_stamp', sa.types.Float(), index = True), 
        sa.Column('first_seen_time_stamp', sa.types.Float(), index = True),  
        sa.Column('last_seen_time_stamp', sa.types.Float(), index = True),
        sa.Column('hgs_bbox', sa.types.String()),
        sa.Column('HARPNUM', sa.types.Integer(), index = True),
        sa.Column('QUALITY', sa.types.Integer()), 
        sa.Column('NOAA_ARS', sa.types.String()), 
        sa.Column('AREA', sa.types.Float()), 
        sa.Column('NOAA_AR', sa.types.Integer()),
        sa.Column('NOAA_NUM', sa.types.Integer()) 
        )


        
########################################################################
# @check_if_uptodate(make_request)
@jobs_limit(1)
@transform(make_request, suffix('df.pickle'), 'sqlite.inserted.pickle')
def insert_drms_AR_into_sqlite(infile, outfile):

    this_df = pickle.load(open(infile, 'rb'))

    chunk = this_df.copy()

    if len(chunk) != 0:

        # convert lists to json serialization

        chunk['hgs_bbox'] = [sxi_module.json_serialize(hgs_bbox) for hgs_bbox in chunk['hgs_bbox']]

        chunk['NOAA_ARS'] = [sxi_module.json_serialize(NOAA_ARS) for NOAA_ARS in chunk['NOAA_ARS']]

        metadata.create_all(engine)

        conn = engine.connect()

        data_dict = chunk.to_dict('records')

        conn.execute(hmi_drms_AR.insert(), data_dict)

        who_inserted = pd.DataFrame({'timestamps': chunk['obs_time_stamp'], 'HARPNUM':chunk['HARPNUM'],'hgs_bbox': this_df['hgs_bbox'] })

        pickle.dump(who_inserted, open(outfile, 'wb'))
    else:
        pickle.dump(pd.DataFrame(), open(outfile, 'wb'))

# def main():
        # pipeline_run([make_request], multiprocess = 10)

pipeline_run([insert_drms_AR_into_sqlite], multiprocess = 4)
    # pipeline_run([insert_aia_availability_into_sqlite], multiprocess = 1) 	


    # pipeline_run([insert_zerocross_to_sqlite], multiprocess = 1, verbose = 4)
    
# if __name__ == "__main__":
#   main()


