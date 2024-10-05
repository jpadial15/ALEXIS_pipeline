import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')


import numpy as np
import pandas as pd

from modules import sxi_module
from modules import convert_datetime


def check_known_flare_wn_alexis_integration_area(AR_hpc_bbox, candidate_flare_coords):

        xvals = np.array(AR_hpc_bbox)[:,0]
        yvals = np.array(AR_hpc_bbox)[:,1]

        #create array of coordinates 
        x_check = np.array([xvals.max(), xvals.min(), candidate_flare_coords[0]])
        y_check = np.array([yvals.max(), yvals.min(), candidate_flare_coords[1]])

        #sort the values
        #candidate coord should lie in the middle with index 1
        sorted_xcheck = (np.sort(x_check))
        sorted_ycheck = (np.sort(y_check))

        # find the location of the candidate coords. 
        # They should be in position 1 of the array
        
        position_of_x = (np.where( sorted_xcheck ==  candidate_flare_coords[0]))[0]
        position_of_y = (np.where( sorted_ycheck ==  candidate_flare_coords[1]))[0]

        ### if coords are within the bounding box and we define r = position_of_x * position_of_y == 1
        
        r = position_of_x*position_of_y == 1
        
        return(r)


def find_if_known_flares_wn_alexis_integration_area(cluster_group, these_known_flares):

    alexis_x_hpc, alexis_y_hpc = cluster_group.x_hpc.iloc[0], cluster_group.y_hpc.iloc[0]

    location_in_array = np.where(cluster_group.cluster_labels.iloc[0] == cluster_group.final_cluster_label.iloc[0])[0][0]

    zoom_in_arcsec_length = cluster_group.zoom_in_type.iloc[0][location_in_array]

    square_region = sxi_module.make_square_region([alexis_x_hpc, alexis_y_hpc], zoom_in_arcsec_length)

    # known_x_hpc, known_y_hpc = these_known_flares.hpc_x, these_known_flares.hpc_y

    wn_int_list = [check_known_flare_wn_alexis_integration_area(square_region, [known_x_hpc, known_y_hpc ])[0] for known_x_hpc, known_y_hpc in zip(these_known_flares.hpc_x, these_known_flares.hpc_y)]

    these_known_flares['within_alexis_integration'] = wn_int_list

    these_known_flares['alexis_cluster_label'] = [cluster_group.final_cluster_label.iloc[0] for _ in wn_int_list]

    these_known_flares['alexis_x_hpc'] = [alexis_x_hpc for _ in wn_int_list]

    these_known_flares['alexis_y_hpc'] = [alexis_y_hpc for _ in wn_int_list]

    these_known_flares['alexis_hpc_bbox'] = [square_region for _ in wn_int_list]

    return(these_known_flares)


def find_if_known_flares_wn_same_alexis_peak_time(this_alexis_peaktime, known_flares_wn_int_df, cluster_group):

    copy_group = cluster_group.copy()

    these_known_date_time = [convert_datetime.astropytime_to_pythondatetime(this_known_flare_astropy_peaktime) for this_known_flare_astropy_peaktime in known_flares_wn_int_df.peak_time]

    these_time_df = pd.DataFrame({'peak_time': these_known_date_time, 'id_team': known_flares_wn_int_df.id_team.to_list(), 'x_hpc':known_flares_wn_int_df.hpc_x.to_list(),'y_hpc': known_flares_wn_int_df.hpc_y.to_list() , 'wn_int_reg': known_flares_wn_int_df.within_alexis_integration.to_list(),'goes_class': known_flares_wn_int_df.goes_class.to_list() })

    with_alexis_time_df = these_time_df.append({'peak_time': this_alexis_peaktime, 'id_team': 'ALEXIS', 'x_hpc': copy_group['x_hpc'].iloc[0], 'y_hpc': copy_group['y_hpc'].iloc[0], 'wn_int_reg': True, 'goes_class': 'TBD'}, ignore_index = True)
    
    start, end = this_alexis_peaktime - pd.Timedelta('180s'), this_alexis_peaktime + pd.Timedelta('180s')

    mask_known_flares = known_flares_wn_int_df[(known_flares_wn_int_df.peak_time >= start) & (known_flares_wn_int_df.peak_time <= end)]

    mask_df = with_alexis_time_df[(with_alexis_time_df.peak_time >= start) & (with_alexis_time_df.peak_time <= end)]

    team_points_dict = {'SolarSoft': 1, 'SWPC': 2, 'ALEXIS': 4}

    id_list =  [team_points_dict[this_team] for this_team in mask_df.id_team]

    found_id = np.sum(id_list)

    # copy_group['found_teams_in_time_bool'] = [[True for _ in mask_df.id_team]]

    copy_group['found_teams_in_time'] = [found_id for _ in copy_group.ALEXIS_found] # +- 3 min

    copy_group['all_teams_order_wn_time'] = [[this_team for this_team in mask_df.id_team]]

    copy_group['all_team_datetimes'] = [[this_team_datetime for this_team_datetime in mask_df.peak_time]]

    copy_group['all_team_wn_alexis_bbox'] = [[this_bool for this_bool in mask_df.wn_int_reg]]

    copy_group['ALEXIS_hpc_bbox'] = [known_flares_wn_int_df.alexis_hpc_bbox.iloc[0]]

    # print(mask_df)

    # print('-----------')

    copy_group['all_team_hpc_coords'] = [[[this_x, this_y] for this_x, this_y in zip(mask_df.x_hpc.to_list(), mask_df.y_hpc.to_list())]]

    copy_group['all_team_goes_class'] = [[this_class for this_class in mask_df.goes_class]]

    # take care of the meta
    copy_group['all_teams_order_meta'] = [[this_team for this_team in known_flares_wn_int_df.id_team]]

    copy_group['all_team_datetimes_meta'] = [[this_datetime for this_datetime in known_flares_wn_int_df.peak_time]]

    copy_group['all_team_wn_alexis_bbox_meta'] = [[wn_alexis_int for wn_alexis_int in known_flares_wn_int_df.within_alexis_integration]]

    copy_group['ALEXIS_hpc_bbox_meta'] = [known_flares_wn_int_df.alexis_hpc_bbox.iloc[0]]

    copy_group['all_team_hpc_coords_meta'] =[[[this_x, this_y] for this_x, this_y in zip(known_flares_wn_int_df.hpc_x.to_list(), known_flares_wn_int_df.hpc_y.to_list())]]

    copy_group['all_team_goes_class_meta'] =[[this_class for this_class in known_flares_wn_int_df.goes_class]]  
    

    # copy_group['wn_alexis_integration'] = []

    
        


    return(copy_group)



def filter_the_multiple_entries_from_hek_flares(flare_df):

    data_frame_list = []

    for id_team, id_group in flare_df.groupby('id_team'):

        # drop duplicate peak times

        df = id_group.drop_duplicates(subset= 'peak_time').sort_values(by = 'peak_time')
        
        # sometimes peak times are not the same and they differ by minutes 
        # or they differ by miliseconds etc. 
        # lets search for entries that are within 3 minutes of each other

        date_time_list = df['peak_time']

        # date_time_list
        time_df = pd.DataFrame(date_time_list).reset_index(drop = True)



        i = 0 
        while i < len(time_df):

            first_time = time_df.peak_time.iloc[i]

            next_time = first_time + pd.Timedelta('3min')

            this_list = time_df[ (time_df.peak_time >= first_time ) & (time_df.peak_time <= next_time)]

            if len(this_list) > 1:

                this_time = this_list.peak_time.iloc[-1]

            if len(this_list) == 1:

                this_time = this_list.peak_time.iloc[0]

            ########

            this_data_frame_to_keep = df[df.peak_time == this_time]

            data_frame_list.append(this_data_frame_to_keep)

            i = i + len(this_list)

    output_df = pd.concat(data_frame_list)

    return(output_df)
    


    # print(this_list)

#################### old code ######################


    # ###############################################################

    # def check_known_flare_wn_alexis_integration_area(AR_hpc_bbox, candidate_flare_coords):

    #     xvals = np.array(AR_hpc_bbox)[:,0]
    #     yvals = np.array(AR_hpc_bbox)[:,1]

    #     #create array of coordinates 
    #     x_check = np.array([xvals.max(), xvals.min(), candidate_flare_coords[0]])
    #     y_check = np.array([yvals.max(), yvals.min(), candidate_flare_coords[1]])

    #     #sort the values
    #     #candidate coord should lie in the middle with index 1
    #     sorted_xcheck = (np.sort(x_check))
    #     sorted_ycheck = (np.sort(y_check))

    #     # find the location of the candidate coords. 
    #     # They should be in position 1 of the array
        
    #     position_of_x = (np.where( sorted_xcheck ==  candidate_flare_coords[0]))[0]
    #     position_of_y = (np.where( sorted_ycheck ==  candidate_flare_coords[1]))[0]

    #     ### if coords are within the bounding box and we define r = position_of_x * position_of_y == 1
        
    #     r = position_of_x*position_of_y == 1
        
    #     return(r)

    # def find_if_known_flares_wn_alexis_integration_area(cluster_group, these_known_flares):

    #     alexis_x_hpc, alexis_y_hpc = cluster_group.x_hpc.iloc[0], cluster_group.y_hpc.iloc[0]

    #     location_in_array = np.where(cluster_group.cluster_labels.iloc[0] == cluster_group.final_cluster_label.iloc[0])[0][0]

    #     zoom_in_arcsec_length = cluster_group.zoom_in_type.iloc[0][location_in_array]

    #     square_region = sxi_module.make_square_region([alexis_x_hpc, alexis_y_hpc], zoom_in_arcsec_length)

    #     # known_x_hpc, known_y_hpc = these_known_flares.hpc_x, these_known_flares.hpc_y

    #     wn_int_list = [check_known_flare_wn_alexis_integration_area(square_region, [known_x_hpc, known_y_hpc ])[0] for known_x_hpc, known_y_hpc in zip(these_known_flares.hpc_x, these_known_flares.hpc_y)]

    #     these_known_flares['within_alexis_integration'] = wn_int_list

    #     these_known_flares['alexis_cluster_label'] = [cluster_group.final_cluster_label.iloc[0] for _ in wn_int_list]

    #     these_known_flares['alexis_x_hpc'] = [alexis_x_hpc for _ in wn_int_list]

    #     these_known_flares['alexis_y_hpc'] = [alexis_y_hpc for _ in wn_int_list]

    #     these_known_flares['alexis_hpc_bbox'] = [square_region for _ in wn_int_list]

    #     return(these_known_flares)


    # def find_if_known_flares_wn_same_alexis_peak_time(this_alexis_peaktime, known_flares_wn_int_df, cluster_group):

    #     copy_group = cluster_group.copy()

    #     these_known_date_time = [convert_datetime.astropytime_to_pythondatetime(this_known_flare_astropy_peaktime) for this_known_flare_astropy_peaktime in known_flares_wn_int_df.peak_time]

    #     these_time_df = pd.DataFrame({'peak_time': these_known_date_time, 'id_team': known_flares_wn_int_df.id_team.to_list(), 'x_hpc':known_flares_wn_int_df.hpc_x.to_list(),'y_hpc': known_flares_wn_int_df.hpc_y.to_list() , 'wn_int_reg': known_flares_wn_int_df.within_alexis_integration.to_list(),'goes_class': known_flares_wn_int_df.goes_class.to_list() })

    #     with_alexis_time_df = these_time_df.append({'peak_time': this_alexis_peaktime, 'id_team': 'ALEXIS', 'x_hpc': copy_group['x_hpc'].iloc[0], 'y_hpc': copy_group['y_hpc'].iloc[0], 'wn_int_reg': True, 'goes_class': 'TBD'}, ignore_index = True)
        
    #     start, end = this_alexis_peaktime - pd.Timedelta('180s'), this_alexis_peaktime + pd.Timedelta('180s')

    #     mask_known_flares = known_flares_wn_int_df[(known_flares_wn_int_df.peak_time >= start) & (known_flares_wn_int_df.peak_time <= end)]

    #     mask_df = with_alexis_time_df[(with_alexis_time_df.peak_time >= start) & (with_alexis_time_df.peak_time <= end)]

    #     team_points_dict = {'SolarSoft': 1, 'SWPC': 2, 'ALEXIS': 4}

    #     id_list =  [team_points_dict[this_team] for this_team in mask_df.id_team]

    #     found_id = np.sum(id_list)

    #     copy_group['found_teams_in_time'] = [found_id for _ in copy_group.ALEXIS_found] # +- 3 min

    #     copy_group['all_teams_order_wn_time'] = [[this_team for this_team in mask_df.id_team]]

    #     copy_group['all_team_datetimes'] = [[this_team_datetime for this_team_datetime in mask_df.peak_time]]

    #     copy_group['all_team_wn_alexis_bbox'] = [[this_bool for this_bool in mask_df.wn_int_reg]]

    #     copy_group['ALEXIS_hpc_bbox'] = [known_flares_wn_int_df.alexis_hpc_bbox.iloc[0]]

    #     # print(mask_df)

    #     # print('-----------')

    #     copy_group['all_team_hpc_coords'] = [[[this_x, this_y] for this_x, this_y in zip(mask_df.x_hpc.to_list(), mask_df.y_hpc.to_list())]]

    #     copy_group['all_team_goes_class'] = [[this_class for this_class in mask_df.goes_class]]

    #     # take care of the meta
    #     copy_group['all_teams_order_meta'] = [[this_team for this_team in known_flares_wn_int_df.id_team]]

    #     copy_group['all_team_datetimes_meta'] = [[this_datetime for this_datetime in known_flares_wn_int_df.peak_time]]

    #     copy_group['all_team_wn_alexis_bbox_meta'] = [[wn_alexis_int for wn_alexis_int in known_flares_wn_int_df.within_alexis_integration]]

    #     copy_group['ALEXIS_hpc_bbox_meta'] = [known_flares_wn_int_df.alexis_hpc_bbox.iloc[0]]

    #     copy_group['all_team_hpc_coords_meta'] =[[[this_x, this_y] for this_x, this_y in zip(known_flares_wn_int_df.hpc_x.to_list(), known_flares_wn_int_df.hpc_y.to_list())]]

    #     copy_group['all_team_goes_class_meta'] =[[this_class for this_class in known_flares_wn_int_df.goes_class]]  
        

    #     # copy_group['wn_alexis_integration'] = []

        
            


    #     return(copy_group)