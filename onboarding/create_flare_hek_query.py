import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

import pickle
import numpy as np
import pandas as pd


from sunpy.net import hek
from sunpy.time import parse_time

from ruffus import *
import re
from datetime import timedelta

import dataconfig

from modules import convert_datetime

WORKING_DIR = dataconfig.DATA_DIR_HEK_FLARES
tw = lambda x: os.path.join(WORKING_DIR, x)

# create initial time pairs for running through the hek AR query by month
import datetime
from datetime import timedelta


time_array = []


x = pd.date_range(start='4/01/2010', end='11/30/2020', freq = 'M')

# x starts with the end date of april such that for every starting point we need to add an extra day such that we begin in the same month as the end month. 
for previous, current in zip(x, x[1:]):
    j = ([previous + timedelta(days = 1), current ])
    my_dict = {'name': '{}_{}_flare_hek'.format(previous.year, previous.month),
    			'start': j[0], 
    			'end':j[1]}
    time_array.append(my_dict)

				

def create_initial_files():
	for file in time_array:
		infile = None
		name = file['name']
		start = file['start']
		end = file['end']
		outfile = tw('{}.start.pickle'.format(name))
		yield(infile,outfile,start,end)
		
@mkdir(WORKING_DIR)
@files(create_initial_files)
def create_request(infile, outfile, start, end):

	# print(outfile)

	pickle.dump({'start': start, 'end': end, 'event_type': 'FL'}, open(outfile, 'wb'))
	

# pipeline_run([create_request])

@transform(create_request, suffix('.start.pickle'), "_flares_hek_result.pickle")
def download_data(infile, outfile):

	start_file = pickle.load(open(infile, 'rb'))

	client = hek.HEKClient()

	event_type = start_file['event_type']

	tstart = start_file['start']

	tend = start_file['end']

	AR_results = client.search(hek.attrs.Time(tstart, tend),
	                           hek.attrs.EventType(event_type))

	pickle.dump(AR_results, open(outfile, 'wb'))

@transform(download_data, suffix('_flares_hek_result.pickle'), '_flare_hek_db.pickle')
def parse_HEK_FL_result(infile, outfile):

	    # want to condense the results of the query into a more manageable
	    # dictionary
	    # keep event data, start time, peak time, end time, GOES-class,
	    # location, active region source (as per GOES list standard)
	    # make this into a list of dictionaries

	goes_results = pickle.load(open(infile,'rb'))

	goes_event_list = []

	for r in goes_results:
		try:
			hgc_poly_string = r['hgc_bbox']
			sep_coord_from_string_hgc = re.split('[(-)]+', hgc_poly_string)
			sep_coord_from_string_hgc = re.split('[(-)]+', hgc_poly_string)
			only_coord_string_list_hgc = sep_coord_from_string_hgc[1].split(',')
			this_coord_array_hgc = []
			for coord_pair_string_hgc in only_coord_string_list_hgc:
				numerical_coord_pairs_hgc = [np.float(coord_pair_string_hgc.split(' ')[0]), np.float(coord_pair_string_hgc.split(' ')[1])]
				this_coord_array_hgc.append(numerical_coord_pairs_hgc)
		except:
			this_coord_array_hgc = [0]

		try:
			hpc_poly_string = r['hpc_bbox']
			sep_coord_from_string_hpc = re.split('[(-)]+', hpc_poly_string)
			sep_coord_from_string_hpc = re.split('[(-)]+', hpc_poly_string)
			only_coord_string_list_hpc = sep_coord_from_string_hpc[1].split(',')
			this_coord_array_hpc = []
			for coord_pair_string_hpc in only_coord_string_list_hpc:
				numerical_coord_pairs_hpc = [np.float(coord_pair_string_hpc.split(' ')[0]), np.float(coord_pair_string_hpc.split(' ')[1])]
				this_coord_array_hpc.append(numerical_coord_pairs_hpc)
		except:
			this_coord_array_hpc = [0]


		try:
			hgs_poly_string = r['hgs_bbox']
			sep_coord_from_string_hgs = re.split('[(-)]+', hgs_poly_string)
			sep_coord_from_string_hgs = re.split('[(-)]+', hgs_poly_string)
			only_coord_string_list_hgs = sep_coord_from_string_hgs[1].split(',')
			this_coord_array_hgs = []
			for coord_pair_string_hgs in only_coord_string_list_hgs:
				numerical_coord_pairs_hgs = [np.float(coord_pair_string_hgs.split(' ')[0]), np.float(coord_pair_string_hgs.split(' ')[1])]
				this_coord_array_hgs.append(numerical_coord_pairs_hgs)
		except:
			this_coord_array_hgs = [0]
			
		if r['noposition'] == 'true':
			flare_loc_bool = 1
		else:
			flare_loc_bool = 0

		goes_class = str(r['fl_goescls'])

		try:
			# will fail if not a number
			goes_number = np.float(goes_class[1:])
		except:
			goes_number = np.nan

		goes_event = {
			'event_date': parse_time(r['event_starttime']).strftime(
				'%Y-%m-%d'),
			'start_time': convert_datetime.astropytime_to_pythondatetime(parse_time(r['event_starttime'])),
			'peak_time': convert_datetime.astropytime_to_pythondatetime(parse_time(r['event_peaktime'])),
			'end_time': convert_datetime.astropytime_to_pythondatetime(parse_time(r['event_endtime'])),
			'goes_class': goes_class,
			'goes_letter': goes_class[:1],
			'goes_number': goes_number,
			'AR_num': r['ar_noaanum'],
			'hgs_x': r['hgs_x'],
			'hgs_y': r['hgs_y'],
			'hgs_bbox_poly': this_coord_array_hgs,
			'hgc_x': r['hgc_x'],
			'hgc_y': r['hgc_y'],
			'hgc_bbox_poly': this_coord_array_hgc,
			'hpc_x': r['hpc_x'],
			'hpc_y': r['hpc_y'],
			'hpc_bbox_poly': this_coord_array_hpc,
			'event_type': r['event_type'],
			'telescope_used': r['obs_observatory'],
			'id_institute': r['frm_institute'],
			'id_team': r['frm_identifier'],
			'search_instrument': r['search_instrument'],
			'search_channel': r['search_channelid'],
			'noposition': flare_loc_bool 
			}
		goes_event_list.append(goes_event)

	df = pd.DataFrame(goes_event_list)

	pickle.dump(df, open(outfile, 'wb'))

	# print('Ended {}'.format(tstart))


@merge(parse_HEK_FL_result, os.path.join(dataconfig.DATA_DIR_PRODUCTS, 'hek_flare_db.pickle'))
def merge_hek_query_per_month(infiles, outfile):
	merged_df_list = []

	for infile in infiles:
		input_data = pickle.load(open(infile, 'rb'))
		merged_df_list.append(input_data)
        
	merged_df = pd.concat(merged_df_list)

	merged_df = merged_df.sort_values('start_time')

	merged_df = merged_df.reset_index(drop = True)

	pickle.dump(merged_df, open(outfile, 'wb'))

pipeline_run([merge_hek_query_per_month], multiprocess = 10, verbose = 2)