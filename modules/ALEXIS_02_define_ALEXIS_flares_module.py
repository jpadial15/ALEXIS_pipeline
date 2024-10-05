import sys 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

import pickle
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
import dataconfig
from modules import convert_datetime


def apply_peakfinder(data, timeseries):

    peaks, _ = find_peaks(data, height = 0.1)

    peak_date_times = [timeseries[this_peak] for this_peak in peaks]
    peak_values = [data[this_peak] for this_peak in peaks]

    return(peak_date_times, peak_values)



def find_alexis_peaks(test_file, fwd_time_val = 180):

    input_df = test_file

    A_cvx, x_cvx, timeseries = input_df.cluster_matrix.T, input_df.xray_data, [convert_datetime.convert_timestamp_to_datetime(this_timestamp) for this_timestamp in input_df.resampled_time_stamp]

    # time_worth_of_data = (timeseries.to_list()[-1] - timeseries.to_list()[0])

    predicted_vector = input_df.linear_combo_fit

    pass_this_dict = input_df

    linear_combo_peak_list = apply_peakfinder(predicted_vector, timeseries)

    xray_peak_list = apply_peakfinder(x_cvx, timeseries)

    pass_this_dict['XRS_peaks_date_time'], pass_this_dict['XRS_peaks_value'] = xray_peak_list[0], xray_peak_list[1]

    pass_this_dict['linear_combo_peaks_date_time'], pass_this_dict['linear_combo_peaks_value']  = linear_combo_peak_list[0], linear_combo_peak_list[1]

    alexis_peaks = []

    not_enough_data_list = []

    # date_time_w_3_votes = []

    for tracking_number, cluster_label in enumerate(input_df.gridsearch_clusters):

        data = A_cvx[:,tracking_number]

        # create peaklist of this column data 

        vector_peak_date_time, vector_peak_value = apply_peakfinder(data, timeseries)

    #     ax2.scatter(vector_peak_date_time, vector_peak_value, s = 500, color = cluster_color_dict[cluster_label])

        date_time_w_3_votes = []

        copy_dict = pass_this_dict.copy() # use to track all the info


        if (len(vector_peak_date_time)* len(xray_peak_list[0]) * len(linear_combo_peak_list[0])) > 0: # there are some peaks in this column from the surviving matrix

            xray_peak_time, linear_combo_peak_time = pass_this_dict['XRS_peaks_date_time'], pass_this_dict['linear_combo_peaks_date_time']

            start_time, end_time = xray_peak_time[0], xray_peak_time[-1]

            xray_peak_timestamps = [convert_datetime.convert_datetime_to_timestamp(this_xray_datetime) for this_xray_datetime in xray_peak_time]

            start_time, end_time = convert_datetime.convert_datetime_to_timestamp(timeseries[0]), convert_datetime.convert_datetime_to_timestamp(timeseries[-1])

            normed_xray_timestamps = [(unnormed_xray_time - start_time)  for unnormed_xray_time in xray_peak_timestamps] 

            linear_combo_peak_timestamps = [convert_datetime.convert_datetime_to_timestamp(this_linear_combo_datetime) for this_linear_combo_datetime in linear_combo_peak_time]

            normed_linear_combo_timestamps = [(unnormed_linear_combo_time - start_time)  for unnormed_linear_combo_time in linear_combo_peak_timestamps] 

            ordered_df = pd.DataFrame([{'time':this_time, 'from': 'L'} for this_time in normed_linear_combo_timestamps] + [{'time':this_time, 'from': 'X'} for this_time in normed_xray_timestamps] + [{'time':convert_datetime.convert_datetime_to_timestamp(this_time) - start_time, 'from': 'V'} for this_time in vector_peak_date_time] ).sort_values(by = 'time')
            # ordered_df

            i = 0
            while i < len(ordered_df.time):

                first_entry_time = ordered_df.iloc[i].time

                ##########################################
                        # we define time window here # 
                        # default = 180 seconds #
                ##########################################

                fwd_time = first_entry_time + fwd_time_val # seconds

                mask = ordered_df[(ordered_df.time >= first_entry_time) & (ordered_df.time <= fwd_time)]

                i = i + len(mask.time)

                # print(first_entry_time)

                if len(mask['from'].unique()) == 3:

                    alexis_peak_date_time = convert_datetime.convert_timestamp_to_datetime(mask.time.mean()+start_time)


                    date_time_w_3_votes.append(alexis_peak_date_time)
                    
#                         else:
        else:# date_time_w_3_votes.append(pd.Timestamp('01-01-1980T00:00:00', tz = 'utc')) # if the vector has no peaks, then return in alexis peaktime Jan 1st 1980. One will also get empty lists for vector peak times
            pass

        copy_dict['final_cluster_label'] = cluster_label

        copy_dict['alexis_peaktime'] =  date_time_w_3_votes
        copy_dict['num_alexis_peaks'] = len(date_time_w_3_votes)
        copy_dict['vector_peaks_date_time'] = vector_peak_date_time
        copy_dict['vector_peaks_value'] =  vector_peak_value
        copy_dict['data_length'] = len(A_cvx[:,0])
    #     copy_dict['data_time_delta'] = time_worth_of_data

        if len(date_time_w_3_votes) != 0:

            copy_dict['ALEXIS_found'] = True
        else:
            copy_dict['ALEXIS_found'] = False

        alexis_peaks.append(copy_dict)

    return(pd.DataFrame(alexis_peaks))