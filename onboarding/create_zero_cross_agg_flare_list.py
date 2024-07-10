
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from ruffus import *

import pickle
import os
import sys  

sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

from modules import zero_cross_module #import find_zero_crossings 
import dataconfig

# WORKING_DIR = '/data/padialjr/jorge-helio/goes_data'
#### make date arrays

start_dates = pd.date_range(start='5/1/2010T00:00:00', end='3/1/2020T00:00:00', freq = 'MS',closed = None, tz = 'UTC')
end_dates = pd.date_range(start = '5/31/2010T23:59:59', end = '3/31/2020T23:59:59', freq = 'M', closed = None, tz = 'UTC')

# print(start_dates)
# print(end_dates)

### define start and end datetime objects for sql

instrument_list = ['goes13', 'goes14', 'goes15']
date_dict = []

for inst in instrument_list:
    for i in range(0,len(start_dates)):
        date_dict.append({
            'start_date': start_dates[i],

            'end_date': end_dates[i], 
            
            'instrument': inst
            })


WORKING_DIR = dataconfig.DATA_DIR_ZERO_CROSS

tw = lambda x: os.path.join(WORKING_DIR, x)

def dl_params():
    for dictionary_element in date_dict:

        infile = None

        start_time = dictionary_element['start_date']

        end_time = dictionary_element['end_date']

        this_instrument = dictionary_element['instrument']

        out_name = '{}_{}_{}.zerocross.start'.format(start_time.date(),end_time.date(), this_instrument)

        outfile = tw("{}".format(out_name))

        yield(infile, outfile, start_time, end_time, this_instrument)

@mkdir(WORKING_DIR)
@files(dl_params)
def make_request(infile, outfile, start_time, end_time, instrument):

	"""
    start ruffus checkpoint of each individual monthly xray zerocrossing finder

	"""

	pickle.dump({'start_date': start_time, 'end_date': end_time, 'instrument': instrument}, open(outfile, 'wb'))

# pipeline_run([make_request], multiprocess = 10, verbose = 1)

@transform(make_request, suffix('.zerocross.start'), ".zero_cross_df.pickle")
def find_xray_flare_candidates(infile, outfile):

    zerocross_finder_params = pickle.load(open(infile, 'rb'))

    start = zerocross_finder_params['start_date']

    end = zerocross_finder_params['end_date']

    this_instrument = zerocross_finder_params['instrument']

    zerocross_df, is_there_data_df, error_df = zero_cross_module.find_zero_crossings(start, end, this_instrument)

    pickle.dump(error_df, open(tw("{}_{}_{}_error_df.pickle".format(start.date(), end.date(), this_instrument)), 'wb'))

    pickle.dump(is_there_data_df, open(tw("{}_{}_{}_data_available.pickle".format(start.date(), end.date(), this_instrument)), 'wb'))

    pickle.dump(zerocross_df, open(outfile, 'wb'))


if __name__ == "__main__":
    pipeline_run([find_xray_flare_candidates], multiprocess = 10, verbose = 4)