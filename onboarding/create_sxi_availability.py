import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

from datetime import timedelta, date
import pandas as pd
import pickle
from ruffus import *
import os
import re
from bs4 import BeautifulSoup
from requests import get
import sqlalchemy as sa

import dataconfig
from modules import convert_datetime
from time import time, sleep
from random import randint
import random

import warnings as warn


WORKING_DIR = dataconfig.DATA_DIR_GOES_SXI
DATA_PRODUCT_DIR = dataconfig.DATA_DIR_PRODUCTS

tw = lambda x: os.path.join(WORKING_DIR, x)


# some dictionaries we'll use sometimes

instruments = ['goes13', 'goes14', 'goes15']

inst_dict = {'goes13': 'g13', 'goes14': 'g14', 'goes15': 'g15'}


#Set timelimits you want to download from

start_download_from = '2010-05-01T00:00:00'
end_download_at = '2020-03-06T00:00:00'

# start_download_from = '2010-05-01T00:00:00'
# end_download_at = '2010-05-02T00:00:00'


# start_download_from = '2014-12-01T00:00:00'
# end_download_at = '2015-01-01T23:59:59'

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





# Where from noaa FTP portal are you downloading from?

# ftp_prename = 'ftp://satdat.ngdc.noaa.gov/sxi/archive/fits'
http_prename = 'https://www.ncei.noaa.gov/data/goes-solar-xray-imager/access/fits'

# NOTE: we are interested in the ftp file that resembles:
#  https://www.ncei.noaa.gov/data/goes-solar-xray-imager/access/fits/goesXX/YYYY/MM/DD/*BA*.FTS

ftp_name_dict = []

for full_instrument_name in instruments:

    for this_date in all_dates_list:

        year, month, day = this_date.strftime('%Y'), this_date.strftime('%m'), this_date.strftime('%d')

        http_name = f'{http_prename}/{full_instrument_name}/{year}/{month}/{day}'

        outfile_str = tw(f'{year}-{month}-{day}_{full_instrument_name}')

        # ftp_name_dict.append({'download_file': ftp_name, 'availability_file': http_name , 'out_name': outfile_str })
        ftp_name_dict.append({ 'availability_file': http_name , 'out_name': outfile_str })


# shuffle ftp_name_dict 

random.shuffle(ftp_name_dict)

def dl_params():
    for dictionary_element in ftp_name_dict:

        infile = None

        # ftp_name = dictionary_element['download_file']

        http_name = dictionary_element['availability_file']

        out_name = dictionary_element['out_name']

        outfile = tw(f"{out_name}.start.pickle")

        yield(infile, outfile, http_name, out_name )

@mkdir(WORKING_DIR)
@files(dl_params)
def make_request(infile, outfile, http_name, out_name):

	"""
    make ruffus checkpoint of each individual daily xray data query.

	"""

	pickle.dump({'availability_file': http_name , 'out_name': outfile_str }, open(outfile, 'wb'))


@transform(make_request, suffix('.start.pickle'), ".available.pickle")
def check_data_availability(infile, outfile):

    """ Ask for a webresponse and parse throught the available links """

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

    queries = pickle.load(open(infile, 'rb'))

    http_query_name = queries['availability_file']

    response = get(http_query_name, headers = headers)
    # print(f'{response.status_code} for {http_query_name}')

    # Pause the loop
    sleep(randint(4,37))

        # ADD LONG SLEEP AFTER 429 AND ALSO 
        # RE-TRY LOOP FOR 429 ==> MAYBE

        # Throw a warning for non-200 status codes
    if response.status_code == 429:

        print('{} w/ Status code: {}'.format(http_query_name, response.status_code))
        print(BeautifulSoup(response.content, 'html.parser'))
        # break

    if response.status_code == 200:
        html_soup = BeautifulSoup(response.content, 'html.parser')

        sxi_links = []

        for link in html_soup.find_all('a'):

            file_name = str(link.get('href'))

            if file_name.startswith('SXI'):

                level= re.findall(r'(?<=\d{9}_)[A-Za-z0-9]{2}', file_name)[0]

                inst = re.findall(r'(?<=\d{9}_[A-Za-z0-9]{2}_)\d{2}', file_name)[0]

                date_time_str = re.findall(r'\d{8}_\d{9}', file_name )[0]

                date_time_obj = pd.to_datetime(date_time_str, format = '%Y%m%d_%H%M%S%f', utc = True)

                time_stamp = convert_datetime.convert_datetime_to_timestamp(date_time_obj)

                year, month, day = date_time_obj.strftime('%Y'), date_time_obj.strftime('%m'), date_time_obj.strftime('%d')

                full_instrument_name = f'goes{inst}'

                # ftp_prename = 'ftp://satdat.ngdc.noaa.gov/sxi/archive/fits'

                http_query_name = f'{http_prename}/{full_instrument_name}/{year}/{month}/{day}/{file_name}'

                download_str = f'wget -e robots=off --recursive --no-parent globs = True --directory-prefix {WORKING_DIR} --no-directories --verbose False {http_query_name}'

                sxi_links.append({'url': http_query_name, 'download_string': download_str, 'time_stamp': time_stamp, 'data_level': level, 'file_name': file_name, 'instrument': full_instrument_name})

        available_data_df = pd.DataFrame(sxi_links)

        # print(available_data_df)

        pickle.dump(available_data_df, open(outfile, 'wb'))

        # print('------------------------------')
    
    if response.status_code == 404:
        no_data = pd.DataFrame()

        pickle.dump(no_data, open(outfile, 'wb'))



##################################################################
# Make an sqlite db for zerocrossing from xray flux

engine = sa.create_engine(f'sqlite:////{DATA_PRODUCT_DIR}/sxi_availability.db', echo = False)

metadata = sa.MetaData()

sxi_availability = sa.Table('sxi_availability', metadata,
        sa.Column('download_string', sa.types.String()),
        sa.Column('time_stamp', sa.types.Float(), index = True), 
        sa.Column('url', sa.types.String()),
        sa.Column('data_level', sa.types.String()),  
        sa.Column('file_name', sa.types.String()),
        sa.Column('instrument', sa.types.String())
                        )

# metadata.create_all(engine)

# conn = engine.connect()



########################################################################
@jobs_limit(1)
@transform(check_data_availability, suffix('.available.pickle'), '.sqlite.inserted.pickle')
def insert_available_sxi_to_sqlite(infile, outfile):

    chunk = pickle.load(open(infile, 'rb'))

    if len(chunk) != 0:

        # insert_this_df = chunk.drop(columns = ['date_time'])

        metadata.create_all(engine)

        conn = engine.connect()

        data_dict = chunk.to_dict('records')

        conn.execute(sxi_availability.insert(), data_dict)

        who_inserted = pd.DataFrame({'timestamps': chunk['time_stamp'], 'inst':chunk['instrument'],'download_string': chunk['download_string'] })

        pickle.dump(who_inserted, open(outfile, 'wb'))
    else:
        pickle.dump(pd.DataFrame(), open(outfile, 'wb'))

if __name__ == "__main__":
    # pipeline_run([make_request], multiprocess = 20, verbose = 4)
    pipeline_run([insert_available_sxi_to_sqlite], multiprocess= 4)
    # pipeline_run([insert_available_sxi_to_sqlite], multiprocess= 1)