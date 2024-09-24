
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



WORKING_DIR = dataconfig.DATA_DIR_GOES_SXI
DATA_PRODUCT_DIR = dataconfig.DATA_DIR_PRODUCTS

tw = lambda x: os.path.join(WORKING_DIR, x)

from datetime import timedelta, date
# some dictionaries we'll use sometimes

instruments = ['goes13', 'goes14', 'goes15']

inst_dict = {'goes13': 'g13', 'goes14': 'g14', 'goes15': 'g15'}


#Set timelimits you want to download from

start_download_from = '2010-05-01T00:00:00'
end_download_at = '2010-05-02T00:00:00'


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

ftp_prename = 'ftp://satdat.ngdc.noaa.gov/sxi/archive/fits'
http_prename = 'https://satdat.ngdc.noaa.gov/sxi/archive/fits'

# NOTE: we are interested in the ftp file that resembles:
#  ftp://satdat.ngdc.noaa.gov/sxi/archive/fits/goesXX/YYYY/MM/DD/*BA*.FTS

ftp_name_dict = []

for full_instrument_name in instruments:

    for this_date in all_dates_list:

        year, month, day = this_date.strftime('%Y'), this_date.strftime('%m'), this_date.strftime('%d')

        # specific_file_name = 'sci_gxrs-l2-irrad_{}_d{}_v0-0-0.nc'.format(inst_dict[full_instrument_name], specific_date_str)

        ftp_name = f'{ftp_prename}/{full_instrument_name}/{year}/{month}/{day}'

        http_name = f'{http_prename}/{full_instrument_name}/{year}/{month}/{day}'

        outfile_str = tw(f'{year}-{month}-{day}_{full_instrument_name}')

        ftp_name_dict.append({'download_file': ftp_name, 'availability_file': http_name , 'out_name': outfile_str })


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

from bs4 import BeautifulSoup
from requests import get


for this_dict in ftp_name_dict:

    print('start')

    http_query_name = this_dict['availability_file']

    http_query_name = 'https://www.ncei.noaa.gov/data/goes-solar-xray-imager/access/fits/goes13/2011/09/06/'

# outfile_name = queries['outname']

# print(outfile)

# download_str = f'wget -e robots=off --recursive --no-parent globs = True --directory-prefix {WORKING_DIR} --no-directories --verbose False {ftp_query_name}'

# os.system(download_str)

    response = get(http_query_name, headers = headers)


    print(response.status_code)