import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import drms
from ruffus import *
import re
import pickle
import warnings
# from create_sxi_availability_db import WORKING_DIR
warnings.filterwarnings("ignore")
from modules import convert_datetime
from modules import query_the_data
import dataconfig
from modules import sxi_module
from modules import helio_reg_exp_module
from time import sleep
from random import randint
import sqlalchemy as sa
import pandas as pd
from random import shuffle



def make_aia_jsoc_availability_query_string(time_ex_1, time_ex_2):

    time_str_1 = f'{time_ex_1:%Y-%m-%dT%H:%M:%S}'

    time_str_2 = f'{time_ex_2:%Y-%m-%dT%H:%M:%S}'

    query_str = f'aia.lev1_euv_12s[{time_str_1}Z-{time_str_2}Z]{{image}}'

    return(query_str)


def make_aia_availability_filename(jsoc_query_str):

    sub2 = re.sub(':', '_', jsoc_query_str)

    sub3 = re.sub('.lev1_euv_12s\[','_', sub2)

    sub4 = re.sub(']{image}', '', sub3)

    return(sub4)


def make_aia_fits_filename(date_time_instance,WL):

    this_file_time = f'{date_time_instance:%Y-%m-%dT%H_%M_%S_%f}'[:-4] # shave off the last 4 digits of the microseconds %f of the date_time string formatter

    sub = re.sub(':', '_', this_file_time)

    sub2 = re.sub('Z','z', sub)

    sub3 = re.sub('T','t', sub2)

    sub4 = re.sub('-', '_', sub3)

    this_filename = f'aia_lev1_{WL}a_{sub4}z_image_lev1.fits'

    return(this_filename)

###################################################
jsoc_email = dataconfig.NOTIFY_EMAIL_ADDR
###################################################

client = drms.Client(email=jsoc_email)
keys = [
    "EXPTIME",
    "QUALITY",
    "T_OBS",
    "WAVELNTH",
    "R_SUN", 
    "RSUN_OBS", 
    "RSUN_REF",
    "DSUN_REF", 
    "DSUN_OBS"
]



start_download_from = '2010-05-01T00:00:00'
end_download_at = '2020-03-06T00:00:00'


START_DATE_TIME = pd.Timestamp(start_download_from, tz = 'utc')

END_DATE_TIME = pd.Timestamp(end_download_at, tz = 'utc')

frequency = '4H'

datelist = pd.date_range(start = START_DATE_TIME , end = END_DATE_TIME, freq = frequency ).tolist()

this_query_list = [make_aia_jsoc_availability_query_string(previous,current) for previous,current in zip(datelist[:], datelist[1:])]

shuffle(this_query_list)

WORKING_DIR = dataconfig.DATA_DIR_AIA_AVAIL

DATA_PRODUCT_DIR = dataconfig.DATA_DIR_PRODUCTS

tw = lambda x: os.path.join(WORKING_DIR, x)

def dl_params():
    for jsoc_query in this_query_list:

        infile = None

        out_name = make_aia_availability_filename(jsoc_query)

        outfile = tw(f'{out_name}_availability.df.pickle')

        yield(infile, outfile, jsoc_query)

@mkdir(WORKING_DIR)
@files(dl_params)
def make_request(infile, outfile, jsoc_query):

    sleep_time = randint(1,11)

    # print(f'sleep time {sleep_time}')


    sleep(sleep_time)

    records, filenames = client.query(jsoc_query, key=keys, seg="image")

    if len(records) != 0:

        records['date_time'] = [pd.Timestamp(T_OBS, tz = 'utc') for T_OBS in records.T_OBS]

        records['url'] = [f"http://jsoc.stanford.edu{filename}" for filename in filenames.image]

        records['time_stamp'] = [convert_datetime.convert_datetime_to_timestamp(date_time) for date_time in records['date_time']]

        records['filename'] = [make_aia_fits_filename(date_time_instance, WL) for date_time_instance, WL in zip(records['date_time'], records['WAVELNTH'])]

        pickle.dump(records, open(outfile, 'wb'))
    
    else:

        print(f'no data for {jsoc_query}')

        pickle.dump(records, open(outfile, 'wb'))


engine = sa.create_engine(f'sqlite:////{DATA_PRODUCT_DIR}/aia_availability.db', echo = False)

metadata = sa.MetaData()

aia_availability = sa.Table(
        'aia_availability', metadata,
        sa.Column('time_stamp', sa.types.Float(), index = True), 
        sa.Column('EXPTIME', sa.types.Float(), index = True),  
        sa.Column('QUALITY', sa.types.Float(), index = True),
        sa.Column('WAVELNTH', sa.types.Integer(), index = True),
        sa.Column('url', sa.types.String(), index = True),
        sa.Column('filename', sa.types.String(), index = True)
                )


        
########################################################################
@jobs_limit(1)
@transform(make_request, suffix('df.pickle'), 'sqlite.inserted.pickle')
def insert_aia_availability_into_sqlite(infile, outfile):

    this_df = pickle.load(open(infile, 'rb'))

    if len(this_df) != 0:

        chunk = this_df.drop(columns = ['date_time', "T_OBS", "R_SUN", "RSUN_OBS", "RSUN_REF","DSUN_REF", "DSUN_OBS"])

        metadata.create_all(engine)

        conn = engine.connect()

        data_dict = chunk.to_dict('records')

        conn.execute(aia_availability.insert(), data_dict)

        who_inserted = pd.DataFrame({'timestamps': chunk['time_stamp'], 'file_name':chunk['filename'],'url': chunk['url'] })

        pickle.dump(who_inserted, open(outfile, 'wb'))
    else:
        pickle.dump(pd.DataFrame(), open(outfile, 'wb'))




if __name__ == "__main__":
    pipeline_run([insert_aia_availability_into_sqlite], multiprocess = 5, verbose=3)
    # pipeline_run([insert_aia_availability_into_sqlite], multiprocess = 1) 	


    # pipeline_run([insert_zerocross_to_sqlite], multiprocess = 1, verbose = 4)