from glob import glob
import re
import pickle
import numpy as np
import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import pandas as pd
import ast



import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from scipy.signal import find_peaks
import cvxpy as cp

from sklearn.cluster import DBSCAN
import scipy.stats
from scipy.stats import mstats
from sklearn.metrics import mean_squared_error


from modules import query_the_data
from modules import convert_datetime
from modules import helio_reg_exp_module

import dataconfig

def create_best_fit_data_frame_for_a_model(given_data, model_type):

    ordered_data_by_rms = given_data[(given_data.model_type == model_type) & (given_data.xrs_wavelength == 'B')].sort_values(by = 'model_RMS')

    best_fit = ordered_data_by_rms.iloc[0]

    best_xrs_wavelength, best_xrs_instrument, best_img_wavelength, best_img_instrument, best_zoom_bboxes = best_fit.xrs_wavelength, best_fit.xrs_instrument, best_fit.img_wavelength, best_fit.img_instrument, best_fit.permutation_type

    best_data_mask = ordered_data_by_rms[(ordered_data_by_rms.xrs_wavelength == best_xrs_wavelength) & 
                                        (ordered_data_by_rms.xrs_instrument == best_xrs_instrument) &
                                        (ordered_data_by_rms.img_wavelength == best_img_wavelength) &
                                        (ordered_data_by_rms.img_instrument == best_img_instrument) &
                                        (ordered_data_by_rms.permutation_type == best_zoom_bboxes) ]

    ordered_by_date_time = best_data_mask.sort_values(by = 'resampled_date_time')

    # print(best_fit.model_RMS) 


    return(ordered_by_date_time)




def use_singular_model_to_filter(example_convex_file):

    best_example_df = create_best_fit_data_frame_for_a_model(example_convex_file, 2 )

    # Lets filter for coeff that have an RMS of > 0.1

    coef_list = []

    for _, cluster_df in best_example_df.groupby(['final_cluster_label']):

        output_dict = cluster_df.to_dict('records')[0]

        coef_list.append(output_dict)


    choose_by_coeff_df = pd.DataFrame(coef_list)

    choose_by_coeff_df['normed_coeff'] = [(this_coef/max(choose_by_coeff_df.coeff)) for this_coef in choose_by_coeff_df['coeff']]

    filter_for_greater_than_10_percent = choose_by_coeff_df[choose_by_coeff_df.normed_coeff > 0.1]

    return(filter_for_greater_than_10_percent)




def prep_the_resultant_matrices(input_data_frame):

    # convert zoom in area bbox size from string to list

    zoom_bbox_list = ast.literal_eval(input_data_frame.permutation_type.iloc[0])
    # find bbox size to use
    array_of_this_permutation = np.array(zoom_bbox_list)

    integration_area_arcsec_vector = (1/ ( array_of_this_permutation)**2 )

    # make array of this zoom combo integrated flux offset corrected

    list_of_img_data = []

    data_frame_list = []

    coeff_list = []

    for tracking_number, groupby in enumerate(input_data_frame.groupby(['final_cluster_label'])):

        final_cluster_label, data_for_this_candidate_w_all_zoom_lvl = groupby[0], groupby[1].sort_values(by = 'resampled_date_time')

        this_zoom_in_data = data_for_this_candidate_w_all_zoom_lvl[data_for_this_candidate_w_all_zoom_lvl['bbox_linear_size_arcsec'] == array_of_this_permutation[tracking_number]]

        data_frame_list.append(this_zoom_in_data)

        coeff_list.append(data_for_this_candidate_w_all_zoom_lvl.coeff.iloc[0])

        integrated_flux_timeseries = (np.array(this_zoom_in_data.img_resampled_value_cluster) * integration_area_arcsec_vector[tracking_number])

        offset_correct_integrated_flux = integrated_flux_timeseries - np.min(integrated_flux_timeseries)

        list_of_img_data.append(offset_correct_integrated_flux)

    # make into matrix

    offset_corrected_img_matrix = np.vstack(list_of_img_data).T

    # normalize img matrix 

    A_cvx = offset_corrected_img_matrix / np.max(offset_corrected_img_matrix)

    # prep xray data
    # note: the last zoom in data for the last cluster will have the exact xrs data

    raw_xray_data = np.array(this_zoom_in_data.xrs_resampled_data)

    # offset correct and normalize

    offset_corrected_xray = raw_xray_data - np.min(raw_xray_data)

    max_offset_corrected_xray = np.max(offset_corrected_xray)

    x_cvx = (offset_corrected_xray)/max_offset_corrected_xray


    # Create resulting vector for model 1

    resultant_list_model_1 = []

    # coeff = 
    num_of_clusters = np.linspace(0,min(A_cvx.shape)-1, min(A_cvx.shape), dtype = int)

    for cluster in num_of_clusters:

        this_normed_cluster = offset_corrected_img_matrix.T[cluster] / np.max(offset_corrected_img_matrix)

        # coeff_list = 

        resultant_list_model_1.append(this_normed_cluster*coeff_list[cluster])


    resulting_data_model_1 = np.sum(resultant_list_model_1,axis=0).tolist()

    offset_corrected_resultant_model1 = (resulting_data_model_1 - np.min(resulting_data_model_1))

    normed_resultant_model1 = offset_corrected_resultant_model1/np.max(offset_corrected_resultant_model1)


    ### pass the time for plotting

    time_list = this_zoom_in_data.resampled_date_time.to_list()

    #### resultant matrix 

    resultant_matrix = np.vstack(resultant_list_model_1).T # 40 x 3

    return(normed_resultant_model1, x_cvx, time_list, resultant_matrix)





def redo_the_convex_fit_after_filtering(input_data_frame):

    # convert zoom in area bbox size from string to list

    zoom_bbox_list = ast.literal_eval(input_data_frame.permutation_type.iloc[0])
    # find bbox size to use
    array_of_this_permutation = np.array(zoom_bbox_list)

    integration_area_arcsec_vector = (1/ ( array_of_this_permutation)**2 )

    # make array of this zoom combo integrated flux offset corrected

    list_of_img_data = []

    data_frame_list = []

    # coeff_list = []

    for tracking_number, groupby in (input_data_frame.groupby(['final_cluster_label'])):

        data_for_this_candidate_w_all_zoom_lvl =  groupby.sort_values(by = 'resampled_date_time')

        this_zoom_in_data = data_for_this_candidate_w_all_zoom_lvl[data_for_this_candidate_w_all_zoom_lvl['bbox_linear_size_arcsec'] == array_of_this_permutation[tracking_number]]

        data_frame_list.append(this_zoom_in_data)

        # coeff_list.append(data_for_this_candidate_w_all_zoom_lvl.coeff.iloc[0])

        integrated_flux_timeseries = (np.array(this_zoom_in_data.img_resampled_value_cluster) * integration_area_arcsec_vector[tracking_number])

        offset_correct_integrated_flux = integrated_flux_timeseries - np.min(integrated_flux_timeseries)

        list_of_img_data.append(offset_correct_integrated_flux)

    # make into matrix

    offset_corrected_img_matrix = np.vstack(list_of_img_data).T

    # normalize img matrix 

    A_cvx = offset_corrected_img_matrix / np.max(offset_corrected_img_matrix)

    # prep xray data
    # note: the last zoom in data for the last cluster will have the exact xrs data

    raw_xray_data = np.array(this_zoom_in_data.xrs_resampled_data)

    # offset correct and normalize

    offset_corrected_xray = raw_xray_data - np.min(raw_xray_data)

    max_offset_corrected_xray = np.max(offset_corrected_xray)

    x_cvx = (offset_corrected_xray)/max_offset_corrected_xray

    # fit the filtered regions using the composite model

    ### USE CVXYPY for Composite model (model1) and Singular model (model2)

    # model 1: cxpy constraints = [c>=0, c <= 1]

    c_model1 = cp.Variable(A_cvx.shape[1])

    obj_model1 = cp.Minimize(cp.sum_squares(A_cvx@c_model1 - x_cvx))

    constraints_model1 = [c_model1>=0, c_model1 <= 1]

    prob_model1 = cp.Problem(obj_model1, constraints_model1)

    result_model1 = prob_model1.solve()

    #find out how many clusters we have and do linear regression on every cluster
    num_of_clusters = np.linspace(0,min(A_cvx.shape)-1, min(A_cvx.shape), dtype = int)


    # Create resulting vector for model 1

    resultant_list_model_1 = []

    for cluster in num_of_clusters:

        this_normed_cluster = offset_corrected_img_matrix.T[cluster] / np.max(offset_corrected_img_matrix)

        resultant_list_model_1.append(this_normed_cluster*c_model1.value[cluster])


    resulting_data_model_1 = np.sum(resultant_list_model_1,axis=0).tolist()

    offset_corrected_resultant_model1 = (resulting_data_model_1 - np.min(resulting_data_model_1))

    normed_resultant_model1 = offset_corrected_resultant_model1/np.max(offset_corrected_resultant_model1)


    ### pass the time for plotting

    time_list = this_zoom_in_data.resampled_date_time.to_list()

    #### resultant matrix 

    resultant_matrix = np.vstack(resultant_list_model_1).T # 40 x 3

    return(normed_resultant_model1, x_cvx, time_list, resultant_matrix)







def use_composite_model_to_filter_regions(full_df):

    best_convex_fit = create_best_fit_data_frame_for_a_model(full_df, 1)

    # keep track of original RMS to compare after the filtering

    start_rms = best_convex_fit.model_RMS.iloc[0]

    # make xrs/img timeseries, original matrix, timelist, 
    
    normed_corrected_resultant, normed_xrs_vector, time_list, resultant_matrix = prep_the_resultant_matrices(best_convex_fit)

    avg_power_xray = np.sum((normed_xrs_vector))/ len(normed_xrs_vector)

    xray_E_tot = np.sum((normed_xrs_vector))

    datapoints = len(normed_xrs_vector)

    # make list to keep track of dataframes where we store the E_tot and Avg_pow of each candidate region

    integration_df_list = []

    for i in range(resultant_matrix.shape[1]):

        single_cluster_for_hpc = best_convex_fit[best_convex_fit.final_cluster_label == i].sort_values(by = 'resampled_date_time')

        this_avg_power = np.sum((resultant_matrix[:,i]))/ len(resultant_matrix[:,i])

        this_tot_energy = np.sum((resultant_matrix[:,i]))

        # keep track of single region power and tot energy

        single_cluster_for_hpc['img_avg_power'] = [this_avg_power for element in single_cluster_for_hpc.working_dir]

        single_cluster_for_hpc['img_total_energy'] = [this_tot_energy for element in single_cluster_for_hpc.working_dir]

        # keep track of XRS tot energy and avg pow

        single_cluster_for_hpc['xrs_total_energy'] = [xray_E_tot for element in single_cluster_for_hpc.working_dir]

        single_cluster_for_hpc['xrs_avg_power'] = [avg_power_xray for element in single_cluster_for_hpc.working_dir]

        integration_df_list.append(single_cluster_for_hpc)

    # concat best img/xrs pair to create normalized filters.

    composite_data_frame_for_filtering = pd.concat(integration_df_list)

    # normalize the tot energy

    xrs_tot_energy_array = composite_data_frame_for_filtering.xrs_total_energy.unique() 

    img_tot_energy_array = composite_data_frame_for_filtering.img_total_energy.unique()

    max_tot_energy = np.max(np.concatenate((xrs_tot_energy_array, img_tot_energy_array)))

    composite_data_frame_for_filtering['normed_img_tot_energy'] = [(this_img_energy/max_tot_energy) for this_img_energy in composite_data_frame_for_filtering.img_total_energy] 

    # normalize the avg pow

    xrs_avg_power_array = composite_data_frame_for_filtering.xrs_avg_power.unique() 

    img_avg_power_array = composite_data_frame_for_filtering.img_avg_power.unique()

    max_avg_power = np.max(np.concatenate((xrs_avg_power_array, img_avg_power_array)))

    composite_data_frame_for_filtering['normed_img_avg_power'] = [(this_img_avg_power/max_avg_power) for this_img_avg_power in composite_data_frame_for_filtering.img_avg_power]

    # normalize the coeff

    max_coeff = np.max(composite_data_frame_for_filtering.coeff)

    composite_data_frame_for_filtering['normed_coeff'] = [(this_coef/max_coeff) for this_coef in composite_data_frame_for_filtering.coeff] 

    # do the filtering

    df = composite_data_frame_for_filtering.copy()

    # all must be greater than 10%

    filtered_mask_df = df[(df.normed_img_tot_energy > 0.1) & (df.normed_coeff > 0.1) & (df.normed_img_avg_power > 0.1)]

    filtered_normed_corrected_resultant, filtered_normed_xrs_vector, filtered_time_list, filtered_resultant_matrix = redo_the_convex_fit_after_filtering(filtered_mask_df)

    MSE_model1 = mean_squared_error(filtered_normed_xrs_vector, filtered_normed_corrected_resultant) # mean squared error
    # RMS_model1 = round(np.sqrt(MSE_model1), 4 ) # Root mean squared round to 2 decimals
    RMS_model1 = np.sqrt(MSE_model1)

    check_RMS = start_rms - RMS_model1

    filtered_mask_df['filtered_RMS'] = [RMS_model1 for element in filtered_mask_df.working_dir]

    return(filtered_mask_df)










###################################################


def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


def return_matrices_for_this_zoom_in_combo(do_this_permutation, data_frame):

    array_of_this_permutation = np.array(do_this_permutation)

    integration_area_arcsec_vector = (1/ ( array_of_this_permutation)**2 )

    # make array of this zoom combo integrated flux offset corrected

    list_of_img_data = []

    data_frame_list = []

    cluster_label_list = []

    for tracking_number, groupby in enumerate(data_frame.sort_values(by = ['final_cluster_label']).groupby(['final_cluster_label'])):

        final_cluster_label, data_for_this_candidate_w_all_zoom_lvl = groupby[0], groupby[1].sort_values(by = 'resampled_date_time')

        cluster_label_list.append(final_cluster_label)

        this_zoom_in_data = data_for_this_candidate_w_all_zoom_lvl[data_for_this_candidate_w_all_zoom_lvl['bbox_linear_size_arcsec'] == array_of_this_permutation[tracking_number]].sort_values(by = 'resampled_date_time')

        # keep track of the data
        data_frame_list.append(this_zoom_in_data)

        integrated_flux_timeseries = (np.array(this_zoom_in_data.img_resampled_value_cluster) * integration_area_arcsec_vector[tracking_number])

        offset_correct_integrated_flux = integrated_flux_timeseries - np.min(integrated_flux_timeseries)

        list_of_img_data.append(offset_correct_integrated_flux)

    # make into matrix

    offset_corrected_img_matrix = np.vstack(list_of_img_data).T

    # normalize img matrix 

    A_cvx = offset_corrected_img_matrix / np.max(offset_corrected_img_matrix)

    # prep xray data
    # note: the last zoom in data for the last cluster will have the exact xrs data

    raw_xray_data = np.array(this_zoom_in_data.xrs_resampled_data)

    # offset correct and normalize

    offset_corrected_xray = raw_xray_data - np.min(raw_xray_data)

    max_offset_corrected_xray = np.max(offset_corrected_xray)

    x_cvx = (offset_corrected_xray)/max_offset_corrected_xray

    timeseries = this_zoom_in_data.resampled_date_time


    return(A_cvx, x_cvx,timeseries, cluster_label_list )


def norm_coeff_and_drop_1_percenters(this_fit):

    normed_coeff = this_fit.coeff

    drop_these_array = []

    for index, element in enumerate(normed_coeff):

        if element <= 0.01:

            drop_these_array.append(index)

    return(drop_these_array)



def make_zero_coeff_grid_search_df(this_fit,this_xrs_w_this_img_df, final_grid_search_coeff):

    A_cvx, x_cvx, timeseries, this_cluster_label_list = return_matrices_for_this_zoom_in_combo(this_fit.zoom_in_type, this_xrs_w_this_img_df)


    zero_coeff_result_list = []

    for this_coeff in final_grid_search_coeff:

        copy_this_fit = this_fit.copy()

        resultant_list2 = []

        for i in range(A_cvx.shape[1]):

            vector = A_cvx[:,i]

            # print(this_coeff, i)

            coeff = this_coeff[i]

            resultant_list2.append((vector*coeff))


        # resultant_vector2 = np.sum(np.vstack(resultant_list2).T, axis = 1)



        x_test = np.sum(np.vstack(resultant_list2).T, axis = 1)

        # x_test = np.dot(A_cvx,np.array(grid_search_this))

        #### calculate linearity measurments


        # pearsons
        corr, pval = mstats.pearsonr(x_cvx, x_test)

        # print(corr,(this_coeff))

        copy_this_fit['pearson'] = corr

        copy_this_fit['pval'] = pval

        copy_this_fit['zoom_in_type'] = copy_this_fit.zoom_in_type

        copy_this_fit['coeff'] = this_coeff

        # spearmanr

        # RMSE

        copy_this_fit['RMSE'] = np.sqrt(mean_squared_error(x_cvx, x_test))

        # MSE
        copy_this_fit['MSE'] = (mean_squared_error(x_cvx, x_test))

                    ############### Data made, fit this lambda 

        
        E_tot_img = np.sum(x_test)

        E_tot_xray = np.sum(x_cvx)

        variables_E_tot = cp.Variable(1)
        variables_full_time = cp.Variable(1)

        obj_model_E_tot = cp.Minimize(cp.sum_squares(E_tot_img*variables_E_tot - E_tot_xray ))

        obj_model_full_time = cp.Minimize(cp.sum_squares(x_test*variables_full_time - x_cvx ))

        prob_model_E_tot = cp.Problem(obj_model_E_tot)
        prob_model_full_time = cp.Problem(obj_model_full_time)

        result_model_E_tot = prob_model_E_tot.solve()
        result_model_full_time = prob_model_full_time.solve()

        ###################

        ###################

        try:
            copy_this_fit['alpha_E_tot'] = variables_E_tot.value[0]
        except:
            copy_this_fit['alpha_E_tot'] = 10000

        try:
            copy_this_fit['alpha_full_time'] = variables_full_time.value[0]
        except:
            copy_this_fit['alpha_full_time'] = 10000

        #####################
        # label_dict['lamda'] = each_row.lamda
        # labelled_dict['coeff'] =  [each_row.coeff]
        copy_this_fit['cluster_order'] =  this_cluster_label_list
        # labelled_dict['RMSE'] = RMSE_TEST

        # alpha_dict_list.append(pd.DataFrame(labelled_dict))


        zero_coeff_result_list.append((copy_this_fit))

    output_df = pd.DataFrame(zero_coeff_result_list)

    return(output_df)



def return_matrices_for_this_zoom_in_combo3(data_frame):

    # make array of this zoom combo integrated flux offset corrected

    list_of_img_data = []

    for tracking_number, groupby in enumerate(data_frame.sort_values(by = ['final_cluster_label']).groupby(['final_cluster_label'])):

        final_cluster_label, data_for_this_candidate_w_all_zoom_lvl = groupby[0], groupby[1].sort_values(by = 'resampled_date_time')

        half_linear_size_arcsec = data_for_this_candidate_w_all_zoom_lvl.bbox_linear_size_arcsec.iloc[0]

        integration_area_arcsec_vector = (1/ ( half_linear_size_arcsec)**2 )

        integrated_flux_timeseries = (np.array(data_for_this_candidate_w_all_zoom_lvl.img_resampled_value_cluster) * integration_area_arcsec_vector)

        offset_correct_integrated_flux = integrated_flux_timeseries - np.min(integrated_flux_timeseries)

        list_of_img_data.append(offset_correct_integrated_flux)

    # make into matrix

    offset_corrected_img_matrix = np.vstack(list_of_img_data).T

    # normalize img matrix 

    A_cvx = offset_corrected_img_matrix / np.max(offset_corrected_img_matrix)

    # prep xray data
    # note: the last zoom in data for the last cluster will have the exact xrs data

    raw_xray_data = np.array(data_for_this_candidate_w_all_zoom_lvl.xrs_resampled_data)

    # offset correct and normalize

    offset_corrected_xray = raw_xray_data - np.min(raw_xray_data)

    max_offset_corrected_xray = np.max(offset_corrected_xray)

    x_cvx = (offset_corrected_xray)/max_offset_corrected_xray

    timeseries = data_for_this_candidate_w_all_zoom_lvl.resampled_date_time

    resultant_vector = np.sum(np.vstack(A_cvx), axis = 1)


    return(A_cvx, x_cvx, timeseries ) 



def apply_peakfinder(data, timeseries):

        peaks, _ = find_peaks(data, height = 0.20)

        peak_date_times = [timeseries[this_peak] for this_peak in peaks]
        peak_values = [data[this_peak] for this_peak in peaks]

        return(peak_date_times, peak_values)

def peak_to_flare(x_ray_peaks,resultant_peak_list, vector_peaks, i):

    first_date_time = x_ray_peaks[0][0]

    first_time_stamp = convert_datetime.convert_datetime_to_timestamp(first_date_time)

    # define the model
    # dbscan_model = DBSCAN(eps =15000, min_samples=2)

    xray_resultant_filter_timestamp = x_ray_peaks[0]+ vector_peaks[0] + resultant_peak_list[0]


    timestamps = np.array([convert_datetime.convert_datetime_to_timestamp(these_time) for these_time in xray_resultant_filter_timestamp]).reshape(-1,1)

    timestamps_normed = timestamps - first_time_stamp

    fit_these_timestamps = np.sort(timestamps_normed[:,0]).reshape(-1,1)

    # print(fit_these_timestamps)

    dbscan_model = DBSCAN(eps = 120, min_samples=3)

    # train the model
    dbscan_model.fit(fit_these_timestamps)

    dbscan_result = dbscan_model.fit_predict(fit_these_timestamps)

    # find unique labels in dbscan_result

    all_labels = np.unique(dbscan_result)

    # print(all_labels)


    # drop outliers if part of all_labels. outliers are returned by
    # the DBSCAN  defined as == float(-1)

    good_labels = all_labels[all_labels != -1]

    this_time_might_be_a_flare = []

    for label in good_labels:

        # copy_dict = config.copy()

        which_pixels = np.where(dbscan_result == label)

        timestamp2 = np.mean((fit_these_timestamps[which_pixels] + first_time_stamp))

        # np.mean(timestamp2)

        # plot_vline = [convert_datetime.convert_timestamp_to_datetime(this_timestamp) for this_timestamp in timestamp2]

        plot_vline = convert_datetime.convert_timestamp_to_datetime(timestamp2)

        this_time_might_be_a_flare.append(plot_vline)

        

        # for time in plot_vline:

        number_colors = {0: 'blue', 1: 'orange', 2: 'green', 3: 'purple', 4: 'cyan'}

        # ax.axvline(plot_vline, linewidth = 10, alpha = 0.5, color = number_colors[i])

    ########

    return(this_time_might_be_a_flare)


#####################################################################

def return_matrices_for_this_zoom_in_combo_low_pass_filter(do_this_permutation, data_frame):

    array_of_this_permutation = np.array(do_this_permutation)

    integration_area_arcsec_vector = (1/ ( array_of_this_permutation)**2 )

    # make array of this zoom combo integrated flux offset corrected

    list_of_img_data = []

    data_frame_list = []

    cluster_label_list = []

    for tracking_number, groupby in enumerate(data_frame.sort_values(by = ['final_cluster_label']).groupby(['final_cluster_label'])):

        final_cluster_label, data_for_this_candidate_w_all_zoom_lvl = groupby[0], groupby[1].sort_values(by = 'resampled_date_time')

        cluster_label_list.append(final_cluster_label)

        this_zoom_in_data = data_for_this_candidate_w_all_zoom_lvl[data_for_this_candidate_w_all_zoom_lvl['bbox_linear_size_arcsec'] == array_of_this_permutation[tracking_number]].sort_values(by = 'resampled_date_time')

        # keep track of the data
        data_frame_list.append(this_zoom_in_data)

        integrated_flux_timeseries = (np.array(this_zoom_in_data.img_resampled_value_cluster) * integration_area_arcsec_vector[tracking_number])

        offset_correct_integrated_flux = integrated_flux_timeseries - np.min(integrated_flux_timeseries)

        list_of_img_data.append(offset_correct_integrated_flux)

    # make into matrix

    offset_corrected_img_matrix = np.vstack(list_of_img_data).T

    # normalize img matrix 

    A_cvx = offset_corrected_img_matrix / np.max(offset_corrected_img_matrix)

    # prep xray data
    # note: the last zoom in data for the last cluster will have the exact xrs data

    raw_xray_data = np.array(this_zoom_in_data.xrs_resampled_data)

    # offset correct and normalize

    offset_corrected_xray = raw_xray_data - np.min(raw_xray_data)

    max_offset_corrected_xray = np.max(offset_corrected_xray)

    x_cvx = (offset_corrected_xray)/max_offset_corrected_xray

    timeseries = this_zoom_in_data.resampled_date_time

    ####################################################
    ######### APPLY LOW PASS FILTER ###################

    
    filtered_array = []
    for j in range(A_cvx.shape[1]):
        # filter each column for the img signal
        apply_filter = scipy.signal.filtfilt(np.ones(3)/3, [1], A_cvx[:,j])
        #offset correct the filtered img signal
        offset_correct = apply_filter - apply_filter.min()
        #append to list
        filtered_array.append(offset_correct)
    
    #make img matrix
    filtered_img_matrix = np.array(filtered_array).T

    # create normalized img matrix 

    filt_norm_A_cvx = filtered_img_matrix / filtered_img_matrix.max() 

    # filter the xray signal
    x_cvx_filt = scipy.signal.filtfilt(np.ones(3)/3, [1], x_cvx)

    # normalize the xray
    filt_normed_xray = x_cvx_filt/x_cvx_filt.max()


    return(filt_norm_A_cvx, filt_normed_xray,timeseries, cluster_label_list )