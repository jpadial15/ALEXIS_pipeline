import sys 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pickle
import dataconfig
import query_the_data
import helio_reg_exp_module
import convert_datetime
import os
import sxi_module


def load_pickle(load_this_file):
    
    df = pickle.load(open(load_this_file, 'rb'))
    
    return(df)



def load_hek_report_csv(csv_file_path = f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_hek_report.csv'):
    ''' Load CSV file into DataFrame
    
    default file path is set to f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_hek_report.csv'
    
    '''
    csv_loaded_df = pd.read_csv(csv_file_path)

    return(csv_loaded_df)


def load_hek_report_pickle(pickle_file_path = f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_hek_report.pickle'):
    ''' Load pickle file into DataFrame
    
    default file path is set to f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_hek_report.pickle'
    
    Must be using aiadl_2 conda environment to load pickle file.

    '''
    pickle_load_df = load_pickle(pickle_file_path)

    return(pickle_load_df)


def load_hek_report_parquet(parquet_file_path = f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_hek_report.parquet'):
    ''' Load CSV file into DataFrame
    
    default file path is set to f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_hek_report.csv'
    
    '''
    # Load DataFrame from Parquet file
    parquet_loaded_df = pd.read_parquet(parquet_file_path)
    parquet_loaded_df

    return(parquet_loaded_df)



# lets create the functions to load the alexis full convex fits for files ALEXIS_1000_all_fits


def load_all_fits_csv(csv_file_path = f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_all_fits.csv'):
    ''' Load CSV file into DataFrame
    
    default file path is set to f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_all_fits.csv'
    
    '''
    csv_loaded_df = pd.read_csv(csv_file_path)

    return(csv_loaded_df)


def load_hek_report_pickle(pickle_file_path = f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_all_fits.pickle'):
    ''' Load pickle file into DataFrame
    
    default file path is set to f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_all_fits.pickle'
    
    Must be using aiadl_2 conda environment to load pickle file.

    '''
    pickle_load_df = load_pickle(pickle_file_path)

    return(pickle_load_df)


def load_hek_report_parquet(parquet_file_path = f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_all_fits.parquet'):
    ''' Load parquet file into DataFrame
    
    default file path is set to f'{dataconfig.DATA_DIR_PRODUCTS}/ALEXIS_1000_all_fits.parquet'
    
    '''
    # Load DataFrame from Parquet file
    input_fits_parquet = pd.read_parquet(parquet_file_path)
    

    # Convert lists back to tuples for each column

    input_fits_parquet['hpc_coord_tuple'] = input_fits_parquet['hpc_coord_tuple'].apply(lambda x: tuple(tuple(item) for item in x))
    input_fits_parquet['zoom_in_type'] = [tuple(inner_list) for inner_list in input_fits_parquet['zoom_in_type']]
    input_fits_parquet['gridsearch_clusters'] = [tuple(inner_list) for inner_list in input_fits_parquet['gridsearch_clusters']]
    input_fits_parquet['pix_coord_tuple'] = input_fits_parquet['pix_coord_tuple'].apply(lambda x: tuple(tuple(item) for item in x))
    input_fits_parquet['cluster_matrix'] = [np.array(sxi_module.json_deserialize(this_serialized_list_to_array)) for this_serialized_list_to_array in  input_fits_parquet['cluster_matrix']]

    return(input_fits_parquet)



