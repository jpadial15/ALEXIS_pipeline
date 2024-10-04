import pandas as pd
import numpy as np


def return_good_data_quantities_per_wl_and_inst(clean_df):

    dict_list = []

    for label, group in clean_df.groupby(['wavelength', 'instrument', 'telescope']):

        # print(label)

        if label[1] == 'AIA':

            this_dict = {'wavelength': label[0], 'instrument': label[1], 'telescope': label[2]}

            # print(group.QUALITY.value_counts())

            good_qual_number = len(group[group.QUALITY == 0])

            this_dict['good_qual'] = good_qual_number

            this_dict['available_data'] = len(group)

            this_dict['working_dir'] = clean_df.iloc[0].working_dir

            this_dict['good_percent'] = good_qual_number/len(group)

            sorted_by_date_time = group[group.QUALITY == 0].sort_values(by = 'date_time')

            if len(sorted_by_date_time) > 1:

                time_list_in_seconds = [this_time.total_seconds() for this_time in np.diff(sorted_by_date_time.date_time)]
            
                this_dict['seconds_between_points_max'] = np.max(time_list_in_seconds)

                this_dict['seconds_between_points_min'] = np.min(time_list_in_seconds)

                this_dict['seconds_between_points_std'] = np.std(time_list_in_seconds)

                this_dict['seconds_between_points_var'] = np.var(time_list_in_seconds)

                dict_list.append(this_dict)

            else:

                pass

        # SXI has 2 "data_level" metadata possibilities
        # clean data that has a clear readout has a flag == 'lev2'
        # bad data with CCD readout contamination has flag == 'BA'
        # this flag is given after cleaning the SXI data

        if label[1] == 'SXI':

            # mask for lev2 data 

            lev2_mask = group[(group.data_level == 'lev2')]

            this_dict = {'wavelength': label[0], 'instrument': label[1], 'telescope': label[2]}

            if len(lev2_mask) != 0:

                # print(group.QUALITY.value_counts())

                good_qual_data = lev2_mask[lev2_mask.QUALITY == 0]

                good_qual_number = len(good_qual_data)

                this_dict['good_qual'] = good_qual_number

                this_dict['available_data'] = len(lev2_mask)

                this_dict['working_dir'] = clean_df.iloc[0].working_dir

                this_dict['good_percent'] = good_qual_number/len(lev2_mask)

                if len(good_qual_data) > 1:

                    sorted_by_date_time = good_qual_data.sort_values(by ='date_time')

                    time_list_in_seconds = [this_time_sxi.total_seconds() for this_time_sxi in np.diff(sorted_by_date_time.date_time)]
                
                    this_dict['seconds_between_points_max'] = np.max(time_list_in_seconds)

                    this_dict['seconds_between_points_min'] = np.min(time_list_in_seconds)

                    this_dict['seconds_between_points_std'] = np.std(time_list_in_seconds)

                    this_dict['seconds_between_points_var'] = np.var(time_list_in_seconds)

                if len(good_qual_data) == 0:

                    this_dict['seconds_between_points_max'] = 1000000

                    this_dict['seconds_between_points_min'] = 1000000

                    this_dict['seconds_between_points_std'] = 1000000

                    this_dict['seconds_between_points_var'] = 1000000

                if len(good_qual_data) == 1:

                    this_dict['seconds_between_points_max'] = 1000000

                    this_dict['seconds_between_points_min'] = 1000000

                    this_dict['seconds_between_points_std'] = 1000000

                    this_dict['seconds_between_points_var'] = 1000000



            if len(lev2_mask) == 0:

                good_qual_number = len(lev2_mask[lev2_mask.QUALITY == 0])

                this_dict['good_qual'] = 0

                this_dict['available_data'] = 0

                this_dict['working_dir'] = clean_df.iloc[0].working_dir

                this_dict['good_percent'] = 0

                this_dict['seconds_between_points_max'] = 1000000 # make up some ridiculous number

                this_dict['seconds_between_points_min'] = 1000000 # make up some ridiculous number

                this_dict['seconds_between_points_std'] = 1000000 # make up some ridiculous number

                this_dict['seconds_between_points_var'] = 1000000 # make up some ridiculous number



            dict_list.append(this_dict)



    ouput_df = pd.DataFrame(dict_list)

    return(ouput_df)


def return_better_than_90_percent_of_data(df):

    return(df[df.good_percent >= 0.9])



def return_enough_length_data(df):

    outputlist = []

    for _, row in df.iterrows():

        if row['instrument'] == 'AIA':

            data_lim = 125

            if row['good_qual'] >= data_lim:

                outputlist.append(row.to_dict())
        if row['instrument'] == 'SXI':

            data_lim = 10

            if row['good_qual'] >= data_lim:

                outputlist.append(row.to_dict())
    
    return(pd.DataFrame(outputlist))