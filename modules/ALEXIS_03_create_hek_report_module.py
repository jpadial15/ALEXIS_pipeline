import sys 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import pandas as pd
import numpy as np
from glob import glob
import pickle

import dataconfig
from modules import convert_datetime
from modules import helio_reg_exp_module




def create_span_of_harps(infile):

    wd = helio_reg_exp_module.work_dir_from_flare_candidate_input_string(infile)
    origi_span_harps = f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/{wd}/span_of_available_harps.pickle'

    origi_span_harps_df= pickle.load(open(origi_span_harps, 'rb'))

    harp_file = f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/{wd}/drms_harp_availability.pickle'

    harp_df= pickle.load(open(harp_file, 'rb'))


    output_list = []
    for _, this_dict in origi_span_harps_df.iterrows():

        harp = this_dict['HARPNUM']

        mask = harp_df[harp_df.HARPNUM == harp]

        this_dict['HARPS_NOAA_AR_list'] = mask.NOAA_AR.unique()

        flat_list = [item for sublist in mask.NOAA_ARS.to_list() for item in sublist]

        unique_list = list(set(flat_list))

        this_dict['HARPS_NOAA_ARS_list'] = unique_list
    #     this_dict['HARPS_NOAA_AR_list'] = mask.NOAA_AR.unique()

        output_list.append(this_dict)

    span_of_harps_df = pd.DataFrame(output_list)   
    
    return(span_of_harps_df)




def check_known_flare_wn_alexis_integration_area(AR_hpc_bbox, candidate_flare_coords):

        xvals = np.array(AR_hpc_bbox)[:,0]
        yvals = np.array(AR_hpc_bbox)[:,1]

        # print(xvals,yvals)
        # print(candidate_flare_coords)

        # #create array of coordinates 
        x_check = np.array([xvals.max(), xvals.min(), candidate_flare_coords[0]])
        y_check = np.array([yvals.max(), yvals.min(), candidate_flare_coords[1]])

        # print(x_check, y_check)

        # #sort the values
        # #candidate coord should lie in the middle with index 1
        sorted_xcheck = (np.sort(x_check))
        sorted_ycheck = (np.sort(y_check))
        
        # print(sorted_xcheck, sorted_ycheck)
        # # find the location of the candidate coords. 
        # # They should be in position 1 of the array
        
        position_of_x = (np.where( sorted_xcheck ==  candidate_flare_coords[0]))[0]
        position_of_y = (np.where( sorted_ycheck ==  candidate_flare_coords[1]))[0]

        # print(position_of_x, position_of_y)

        # ### if coords are within the bounding box and we define r = position_of_x * position_of_y == 1
        
        r = position_of_x*position_of_y == 1

        # print(r)
        
        return(r[0])
    
    
    
def coords_wn_harp_meta(in_df, canonical_data_row):
    
    mask_for_truth_in_one_harp = in_df[in_df.found_harp_in_hgs == True]

    
    if len(mask_for_truth_in_one_harp) == 1:
        
        canonical_data_row['HARPNUM_by_coords'] = [mask_for_truth_in_one_harp.iloc[0].HARP]
        canonical_data_row['found_HARPNUM_by_coords'] = True
        
        
    if len(mask_for_truth_in_one_harp) == 0:
        
        canonical_data_row['HARPNUM_by_coords'] = []
        canonical_data_row['found_HARPNUM_by_coords'] = False
        
    if len(mask_for_truth_in_one_harp) > 1:
        
        canonical_data_row['HARPNUM_by_coords'] = mask_for_truth_in_one_harp.HARP.to_list()
        canonical_data_row['found_HARPNUM_by_coords'] = True
        
    return(canonical_data_row)






# check_known_flare_wn_alexis_integration_area(span,coordinates) for span in zip(AR_hpc_bbox_row)

# for _, canonical_data_row in master_hek_flares[:1].iterrows():

def find_canonical_harp_by_coords_per_flare_entry(canonical_data_row, span_of_harps_df):

    coordinates_hpc = [canonical_data_row['hpc_x'],canonical_data_row['hpc_y']]

    coordinates_hgs = [canonical_data_row['hgs_x'],canonical_data_row['hgs_y']]

    canonical_wn_hpc_span = [check_known_flare_wn_alexis_integration_area(span, coordinates_hpc) for span in span_of_harps_df.span_hpc_bbox]

    canonical_wn_hgs_span = [check_known_flare_wn_alexis_integration_area(span, coordinates_hgs) for span in span_of_harps_df.span_hgs_bbox]

    canonical_to_harp_df = pd.DataFrame({'HARP': span_of_harps_df.HARPNUM.to_list(), 
                                         'found_harp_in_hgs': canonical_wn_hgs_span, 
                                         'found_harp_in_hpc': canonical_wn_hpc_span, 
                                         'id_team':canonical_data_row['id_team'],
                                    'HARPS_NOAA_AR_list': span_of_harps_df.HARPS_NOAA_AR_list,
                                    'HARPS_NOAA_ARS_list': span_of_harps_df.HARPS_NOAA_ARS_list})

    

    canonical_row_dict = canonical_data_row.to_dict()
    
    output_dict_list = []
    for _, row in canonical_to_harp_df.iterrows():

        copy_row = row.copy()

        copy_dict = copy_row.to_dict()

        combined_dict = {**copy_dict, **canonical_row_dict}

        output_dict_list.append(combined_dict)

    output_df = pd.DataFrame(output_dict_list)
    
    #######
    
    known_flare_harps_found_in_coords = coords_wn_harp_meta(output_df, canonical_data_row)
    
#     print(known_flare_harps_found_in_coords.AR_num)
    
    

    return(known_flare_harps_found_in_coords)




# check_known_flare_wn_alexis_integration_area(span,coordinates) for span in zip(AR_hpc_bbox_row)

# for _, canonical_data_row in master_hek_flares[:1].iterrows():

def find_canonical_harp_by_hekAR_per_flare_entry(canonical_data_row, span_of_harps_df):
    
    id_team = canonical_data_row.id_team
    
    if id_team == 'SolarSoft':

        canonical_hek_ar_num = [canonical_data_row.AR_num]
        
        if canonical_hek_ar_num[0] == 0:
            
            canonical_data_row['HARPNUM_by_hek_AR'] = []
            
            canonical_data_row['found_HARPNUM_by_hek_AR'] = False
            
        if canonical_hek_ar_num[0] !=0:
        
            add_pre_fix_to_SS_list = [np.float(f'1{element}') for element in canonical_hek_ar_num]

            harp_ar_num_list = span_of_harps_df.HARPS_NOAA_AR_list.to_list()

            A = ([item for sublist in harp_ar_num_list for item in sublist])
    
            B = add_pre_fix_to_SS_list*len(A)
    
            compareAR =([a_i - b_i for a_i, b_i in zip(A,B)])
        
            ar_match = (0 in compareAR)
            
#             print(compareAR)

#             print([element == 0 for element in compareAR].index(True))

            if ar_match== True:
            
                indices = [compareAR.index(0)]

                associated_by_AR_num_df = pd.DataFrame([span_of_harps_df.iloc[this_one] for this_one in indices])

                canonical_data_row['HARPNUM_by_hek_AR'] = associated_by_AR_num_df.HARPNUM.to_list()

                canonical_data_row['found_HARPNUM_by_hek_AR'] = True

            if ar_match == False:
                
#                 print('False')
                
                all_harps_list = []
                
                for _, row in span_of_harps_df.iterrows():
                    
                    all_ar_for_this_harp = row['HARPS_NOAA_ARS_list']
                    
                    diff = add_pre_fix_to_SS_list*len(all_ar_for_this_harp)# - [np.float(string_ar) for string_ar in all_ar_for_this_harp]
                    harp_ar = [np.float(string_ar) for string_ar in all_ar_for_this_harp if string_ar != 'MISSING']
                    
#                     print(diff,harp_ar)
                    compare =([a_i - b_i for a_i, b_i in zip(diff,harp_ar)])
    
#                     print(diff,harp_ar)
    
                    if 0 in compare:
            
                        

                        all_harps_list.append(row['HARPNUM'])

                # take care of what happens if AR->HARP maps fails
                # if it fails, lets try to get the spatiotemporal HARP
                if len(all_harps_list) == 0:
                    
                    canonical_data_row['found_HARPNUM_by_hek_AR'] = False
            
                    canonical_data_row['HARPNUM_by_hek_AR'] = all_harps_list
                    
                if len(all_harps_list) != 0:
        
                    canonical_data_row['found_HARPNUM_by_hek_AR'] = True
            
                    canonical_data_row['HARPNUM_by_hek_AR'] = all_harps_list
                

            
    if id_team == 'SWPC':
        
        canonical_hek_ar_num = [canonical_data_row.AR_num]
        
#         print(canonical_hek_ar_num)
        
        if canonical_hek_ar_num[0] == 0:
            
            canonical_data_row['HARPNUM_by_hek_AR'] = []
            
            canonical_data_row['found_HARPNUM_by_hek_AR'] = False
            
        if canonical_hek_ar_num[0] !=0:
        
            harp_ar_num_list = span_of_harps_df.HARPS_NOAA_AR_list.to_list()

            A = ([item for sublist in harp_ar_num_list for item in sublist])
    
            B = canonical_hek_ar_num*len(A)
    
            compareAR =([a_i - b_i for a_i, b_i in zip(A,B)])
        
            ar_match = (0 in compareAR)

            if ar_match== True:
                
                indices = [compareAR.index(0)]

                associated_by_AR_num_df = pd.DataFrame([span_of_harps_df.iloc[this_one] for this_one in indices])

                canonical_data_row['HARPNUM_by_hek_AR'] = associated_by_AR_num_df.HARPNUM.to_list()

                canonical_data_row['found_HARPNUM_by_hek_AR'] = True

            if ar_match == False:
                
#                 print('False')
                
                all_harps_list = []
                
                for _, row in span_of_harps_df.iterrows():
                    
                    all_ar_for_this_harp = row['HARPS_NOAA_ARS_list']
                    
                    diff = canonical_hek_ar_num*len(all_ar_for_this_harp)# - [np.float(string_ar) for string_ar in all_ar_for_this_harp]
                    harp_ar = [np.float(string_ar) for string_ar in all_ar_for_this_harp if string_ar != 'MISSING']
                    
#                     print(diff,harp_ar)
                    compare =([a_i - b_i for a_i, b_i in zip(diff,harp_ar)])
    
                    if 0 in compare:
            
                        all_harps_list.append(row['HARPNUM'])
        
                # take care of what happens if AR->HARP maps fails
                # if it fails, lets try to get the spatiotemporal HARP
                if len(all_harps_list) == 0:
                    
                    canonical_data_row['found_HARPNUM_by_hek_AR'] = False
            
                    canonical_data_row['HARPNUM_by_hek_AR'] = all_harps_list
                    
                if len(all_harps_list) != 0:
        
                    canonical_data_row['found_HARPNUM_by_hek_AR'] = True
            
                    canonical_data_row['HARPNUM_by_hek_AR'] = all_harps_list
    
    return(canonical_data_row)




def standardize_alexis_for_comparison(alexis_df):
    
#     event_date = helio_reg_exp_module.date_time_from_flare_candidate_working_dir()
    
    output_dict = {'event_date': [specific_time.strftime('%Y-%m-%d') for specific_time in alexis_df.final_ALEXIS_peaktime],
                    'peak_time':alexis_df.final_ALEXIS_peaktime,
                    'goes_class': alexis_df.final_ALEXIS_goes_class,
                    'AR_num': alexis_df.HARPS_NOAA_AR_list,
                    'hpc_x': alexis_df.final_cluster_x_hpc, 
                    'hpc_y': alexis_df.final_cluster_y_hpc, 
                    'hpc_bbox_poly': alexis_df.final_cluster_hpc_integration_bbox, 
                    'id_team': 'ALEXIS', 
                    'HARPNUM': alexis_df.HARPNUM_list, 
                    'AR_num': alexis_df.HARPS_NOAA_AR_list,
                    'search_instrument': [f'{img_tel}-{img_wl}-{xrs_tel}-{xrs_wl}' for img_tel,img_wl,xrs_tel,xrs_wl in zip(alexis_df.img_telescope,alexis_df.img_wavelength,alexis_df.xrs_telescope,alexis_df.xrs_wavelength)],
                   'final_cluster_label': alexis_df.final_cluster_label
                        }
    
    return(pd.DataFrame(output_dict))
    

    
def standardize_others_for_comparison(other_row):
    
#     event_date = helio_reg_exp_module.date_time_from_flare_candidate_working_dir()

    if other_row['found_HARPNUM_by_hek_AR'] == True:
        
        for_other_not_complete = other_row[['event_date', 'peak_time','goes_class','AR_num',
                         'hpc_x', 'hpc_y', 'hpc_bbox_poly', 'id_team', 
                          'AR_num','search_instrument']]
        
        for_other_not_complete['HARPNUM'] = other_row['HARPNUM_by_hek_AR']

    
    if other_row['found_HARPNUM_by_hek_AR'] == False:
        
        for_other_not_complete = other_row[['event_date', 'peak_time','goes_class','AR_num',
                         'hpc_x', 'hpc_y', 'hpc_bbox_poly', 'id_team', 
                          'AR_num','search_instrument']]
        
        for_other_not_complete['HARPNUM'] = other_row['HARPNUM_by_coords']
        
#         for_hek_others = for_other_not_complete.rename(columns = {'HARPNUM_by_coords': 'HARPNUM'}, inplace = True)
        
    return(for_other_not_complete.to_dict())
        

