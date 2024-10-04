import sys  
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

import pandas as pd
import sqlalchemy as sa
from datetime import datetime, timedelta
import pickle
from modules import convert_datetime
import dataconfig
import json

import numpy as np

import sxi_module




MAIN_DIR = dataconfig.MAIN_DIR_NEW
DATA_PROD_DIR = dataconfig.DATA_DIR_PRODUCTS


def xray_sql_db_timea_to_timeb(time_a , time_b , wavelength = None):


    start_timestamp = convert_datetime.convert_datetime_to_timestamp(time_a)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(time_b)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:///{MAIN_DIR}/data_products/timestamp_xrayDB.db')

    metadata = sa.MetaData()

    conn = engine.connect()

    goes_data = sa.Table('goes_data', metadata, autoload = True, autoload_with = engine)

    if wavelength == None:
        flux_query = sa.select([goes_data]).where(sa.and_(
                                    goes_data.c.time_stamp >= start_timestamp, 
                                    goes_data.c.time_stamp < end_timestamp
                                            )
                                )

        flux_sql_df = pd.read_sql(flux_query,conn)

        flux_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['time_stamp'])

        return(flux_sql_df)

    else:

        flux_query = sa.select([goes_data]).where(sa.and_(
                                    goes_data.c.time_stamp >= start_timestamp, 
                                    goes_data.c.time_stamp < end_timestamp, 
                                    goes_data.c.wavelength == wavelength
                                            )
                                )

        flux_sql_df = pd.read_sql(flux_query,conn)

        flux_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['time_stamp'])

        return(flux_sql_df)


def lin_reg_df(time_a , time_b , wavelength = None):


    start_timestamp = convert_datetime.convert_datetime_to_timestamp(time_a)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(time_b)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:///{MAIN_DIR}/data_products/lin_reg.db')

    metadata = sa.MetaData()

    conn = engine.connect()

    lin_reg = sa.Table('lin_reg', metadata, autoload = True, autoload_with = engine)

    if wavelength == None:
        flux_query = sa.select([lin_reg]).where(sa.and_(
                                    lin_reg.c.resampled_time_stamp >= start_timestamp, 
                                    lin_reg.c.resampled_time_stamp < end_timestamp
                                            )
                                )

        flux_sql_df = pd.read_sql(flux_query,conn)

        flux_sql_df['resampled_date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['resampled_time_stamp'])

        return(flux_sql_df)

    else:

        # flux_query = sa.select([lin_reg]).where(sa.and_(
        #                             lin_reg.c.time_stamp >= start_timestamp, 
        #                             lin_reg.c.time_stamp < end_timestamp, 
        #                             goes_data.c.wavelength == wavelength
        #                                     )
                                # )

        flux_sql_df = pd.read_sql(flux_query,conn)

        flux_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['time_stamp'])

        return(flux_sql_df)



def xray_sql_db(input_datetime, query_time = 30, wavelength = None):


    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:///{MAIN_DIR}/data_products/timestamp_xrayDB.db')

    metadata = sa.MetaData()

    conn = engine.connect()

    goes_data = sa.Table('goes_data', metadata, autoload = True, autoload_with = engine)

    if wavelength == None:
        flux_query = sa.select([goes_data]).where(sa.and_(
                                    goes_data.c.time_stamp >= start_timestamp, 
                                    goes_data.c.time_stamp < end_timestamp
                                            )
                                )

        flux_sql_df = pd.read_sql(flux_query,conn)

        flux_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['time_stamp'])

        return(flux_sql_df)

    else:

        flux_query = sa.select([goes_data]).where(sa.and_(
                                    goes_data.c.time_stamp >= start_timestamp, 
                                    goes_data.c.time_stamp < end_timestamp, 
                                    goes_data.c.wavelength == wavelength
                                            )
                                )

        flux_sql_df = pd.read_sql(flux_query,conn)

        flux_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['time_stamp'])

        return(flux_sql_df)


def xray_sql_db_for_flare_def_class(input_datetime, img_instrument):

    query_time_dict = {'AIA': 20, 'SXI': 40}

    query_time = query_time_dict[img_instrument]

    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:///{MAIN_DIR}/data_products/timestamp_xrayDB.db')

    metadata = sa.MetaData()

    conn = engine.connect()

    goes_data = sa.Table('goes_data', metadata, autoload = True, autoload_with = engine)

    
    flux_query = sa.select([goes_data]).where(sa.and_(
                                goes_data.c.time_stamp >= start_timestamp, 
                                goes_data.c.time_stamp < end_timestamp
                                        )
                            )

    flux_sql_df = pd.read_sql(flux_query,conn)

    flux_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(flux_sql_df['time_stamp'])

    return(flux_sql_df)


def query_downloaded_image_availability(input_datetime):

    output_data_list_to_concat = []

    query_time_dict = {'AIA': 20, 'SXI': 40}

    for instrument in ['AIA', 'SXI']:

        start_time = input_datetime - timedelta(minutes = query_time_dict[instrument])

        end_time = input_datetime + timedelta(minutes =  query_time_dict[instrument])

        start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

        end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

        # query Xray DB

        engine = sa.create_engine(f'sqlite:///{dataconfig.DATA_DIR_PRODUCTS}/image_data_availability.db')

        metadata = sa.MetaData()

        conn = engine.connect()

        image_data = sa.Table('image_data_availability', metadata, autoload = True, autoload_with = engine)
    
        downloaded_data = sa.select([image_data]).where(sa.and_(
                                    image_data.c.time_stamp >= start_timestamp, 
                                    image_data.c.time_stamp < end_timestamp, 
                                    image_data.c.instrument == instrument
                                            )
                                )

        downloaded_data = pd.read_sql(downloaded_data,conn)

        downloaded_data['date_time'] = convert_datetime.convert_timestamp_to_datetime(downloaded_data['time_stamp'])

        # downloaded_data['hpc_coord_pairs'] = [json.loads(this_hpc_string) for this_hpc_string in downloaded_data['hpc_coord_pairs']]

        # downloaded_data['pix_coord_pairs'] = [json.loads(this_hpc_string) for this_hpc_string in downloaded_data['pix_coord_pairs']]

        output_data_list_to_concat.append(downloaded_data)

    output_dataframe = pd.concat(output_data_list_to_concat)

    return(output_dataframe)

def sxi_availability_sql_db(input_datetime, query_time = 30):


    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/sxi_availability.db', echo = False)

    metadata = sa.MetaData()

    conn = engine.connect()

    sxi_avail = sa.Table('sxi_availability', metadata, autoload = True, autoload_with = engine)

    sxi_avail_query = sa.select([sxi_avail]).where(sa.and_(
                            sxi_avail.c.time_stamp >= start_timestamp, 
                            sxi_avail.c.time_stamp < end_timestamp
                                    )
                        )

    sxi_sql_df = pd.read_sql(sxi_avail_query,conn)

    sxi_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(sxi_sql_df['time_stamp'])

    return(sxi_sql_df)


def aia_availability(input_datetime, query_time = 20):

    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/aia_availability.db', echo = False)

    metadata = sa.MetaData()

    conn = engine.connect()

    aia_availability = sa.Table('aia_availability', metadata, autoload = True, autoload_with = engine)

    aia_avail_query = sa.select([aia_availability]).where(sa.and_(
                            aia_availability.c.time_stamp >= start_timestamp, 
                            aia_availability.c.time_stamp < end_timestamp
                                    )
                        )

    aia_avail_df = pd.read_sql(aia_avail_query,conn)

    aia_avail_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(aia_avail_df['time_stamp'])

    return(aia_avail_df)

# def sxi_clean_fits_sql_db(input_datetime, query_time = 30):


#     start_time = input_datetime - timedelta(minutes = query_time)

#     end_time = input_datetime + timedelta(minutes = query_time)

#     start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

#     end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

#     # query Xray DB

#     engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/sxi_clean_fits.db', echo = False)

#     metadata = sa.MetaData()

#     conn = engine.connect()

#     clean_fits_w_convex_flux = sa.Table('sxi_clean_fits', metadata, autoload = True, autoload_with = engine)

#     sxi_clean_fits_query = sa.select([clean_fits_w_convex_flux]).where(sa.and_(
#                             clean_fits_w_convex_flux.c.time_stamp >= start_timestamp, 
#                             clean_fits_w_convex_flux.c.time_stamp < end_timestamp
#                                     )
#                         )

#     sxi_sql_df = pd.read_sql(sxi_clean_fits_query,conn)

#     sxi_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(sxi_sql_df['time_stamp'])

#     sxi_sql_df['clean_file_name'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['clean_file_name']), axis=1)

#     # sxi_sql_df['radius'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['radius']), axis=1)

#     # # sxi_sql_df['radius'] = sxi_sql_df.apply(lambda x: np.int(x), axis=1)

#     # sxi_sql_df['radius'].astype(int)

#     sxi_sql_df['hull_sxi_pix_list'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hull_sxi_pix_list']), axis=1)

#     sxi_sql_df['hull_world_hpc_arcsec'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hull_world_hpc_arcsec']), axis=1)

#     sxi_sql_df['hull_center_pix_sxi'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hull_center_pix_sxi']), axis=1)

#     sxi_sql_df['hull_cent_world_sxi_arcsec'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hull_cent_world_sxi_arcsec']), axis=1)

#     # sxi_sql_df['json_list'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['json_list']), axis=1)

#     ordered_df = sxi_sql_df.sort_values(by = 'date_time').reset_index(drop = True)

#     return(ordered_df)


def sxi_region_flux(input_datetime, query_time = 30):


    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)


    engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/sxi_region_flux.db', echo = False)

    metadata = sa.MetaData()

    conn = engine.connect()

    region_flux = sa.Table('sxi_region_flux', metadata, autoload = True, autoload_with = engine)

    sxi_region_flux_query = sa.select([region_flux]).where(sa.and_(
                                region_flux.c.time_stamp >= start_timestamp, 
                                region_flux.c.time_stamp < end_timestamp
                                        )
                            )

    sxi_sql_df = pd.read_sql(sxi_region_flux_query,conn)

    sxi_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(sxi_sql_df['time_stamp'])

    sxi_sql_df['clean_file_name'] = sxi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['clean_file_name']), axis=1)

    ordered_df = sxi_sql_df.sort_values(by = 'date_time').reset_index(drop = True)

    return(sxi_sql_df)


def hinode_availability_sql_db(input_datetime, query_time = 30):


    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

    engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/hinode_availability.db', echo = False)

    metadata = sa.MetaData()

    conn = engine.connect()

    hinode_avail = sa.Table('hinode_availability', metadata, autoload = True, autoload_with = engine)

    sxi_avail_query = sa.select([hinode_avail]).where(sa.and_(
                            hinode_avail.c.time_stamp >= start_timestamp, 
                            hinode_avail.c.time_stamp < end_timestamp
                                    )
                        )

    hinode_sql_df = pd.read_sql(sxi_avail_query,conn)

    hinode_sql_df['date_time'] = convert_datetime.convert_timestamp_to_datetime(hinode_sql_df['time_stamp'])

    return(hinode_sql_df)

# def hmi_ar_hek_sql_db(input_datetime, query_time = 500):


#     start_time = input_datetime - timedelta(minutes = query_time)

#     end_time = input_datetime + timedelta(minutes = query_time)

#     start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

#     end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

#     engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/hmi_hek_ar.db', echo = False)

#     metadata = sa.MetaData()

#     conn = engine.connect()

#     hmi_ar_avail = sa.Table('hmi_hek_ar', metadata, autoload = True, autoload_with = engine)

#     hmi_ar_query = sa.select([hmi_ar_avail]).where(sa.and_(
#                             hmi_ar_avail.c.event_start_time_stamp >= start_timestamp, 
#                             hmi_ar_avail.c.event_end_time_stamp < end_timestamp
#                                     )
#                         )

#     hmi_ar_sql_df = pd.read_sql(hmi_ar_query,conn)

#     hmi_ar_sql_df['event_start_date_time'] = convert_datetime.convert_timestamp_to_datetime(hmi_ar_sql_df['event_start_time_stamp'])

#     hmi_ar_sql_df['event_end_date_time'] = convert_datetime.convert_timestamp_to_datetime(hmi_ar_sql_df['event_end_time_stamp'])

#     hmi_ar_sql_df['AR_noaanum'] = hmi_ar_sql_df.apply(lambda x: sxi_module.json_deserialize(x['AR_noaanum']), axis=1)

#     hmi_ar_sql_df['sharp_noaa_ars'] = hmi_ar_sql_df.apply(lambda x: sxi_module.json_deserialize(x['sharp_noaa_ars']), axis=1)

#     hmi_ar_sql_df['hgs_bbox_poly'] = hmi_ar_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hgs_bbox_poly']), axis=1)

#     hmi_ar_sql_df['hpc_bbox_poly'] = hmi_ar_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hpc_bbox_poly']), axis=1)

#     return(hmi_ar_sql_df)
    






def hek_flares(input_datetime, query_time = 30, MAIN_DIR = '/data/padialjr/jorge-helio', ID_TEAM = None, flare_lower_lim = 'C0.0'):

    """
    ID_TEAM can be any from ['SolarSoft', 'SWPC', 'Feature Finding Team']
    """

    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    astropy_start_time, astropy_end_time = convert_datetime.pythondatetime_to_astropytime(start_time), convert_datetime.pythondatetime_to_astropytime(end_time)

    hek_flares = pickle.load(open('{}/data_products/hek_flare_db.pickle'.format(MAIN_DIR), 'rb'))

    if ID_TEAM == None:

        masked_df = hek_flares[(hek_flares['peak_time'] > astropy_start_time) \
                    & (hek_flares['peak_time'] < astropy_end_time)  \
                    & (hek_flares['goes_class'] >= flare_lower_lim) \
                    #& (hek_flares['id_team'] == 'SWPC') \
                                        ].sort_values(by = 'peak_time').reset_index(drop= True)

        return(masked_df)
    
    else:
        masked_df = hek_flares[(hek_flares['peak_time'] > astropy_start_time) \
            & (hek_flares['peak_time'] < astropy_end_time)  \
            & (hek_flares['id_team'] == ID_TEAM) \
            & (hek_flares['goes_class'] >= flare_lower_lim) \
                                ].sort_values(by = 'peak_time').reset_index(drop= True)

        return(masked_df)

    
def interpolated_xrays(input_datetime, query_time = 30, MAIN_DIR = '/data/padialjr/jorge-helio', instrument = None):

    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    interpolated_xrays = pickle.load(open('{}/data_products/2010_2020_new_flux.df.merged.pickle'.format((MAIN_DIR)), 'rb'))

    if instrument == None:
        masked_df = interpolated_xrays[(interpolated_xrays['datetime'] > start_time) \
                    & (interpolated_xrays['datetime'] < end_time)  \
                    # & (interpolated_xrays['instrument'] == 'goes13')\
                                        ].reset_index(drop= True)

        return(masked_df)

    else:

        masked_df = interpolated_xrays[(interpolated_xrays['datetime'] > start_time) \
            & (interpolated_xrays['datetime'] < end_time)  \
            & (interpolated_xrays['instrument'] == instrument)\
                                ].reset_index(drop= True)

        return(masked_df)


        
def zero_crossings(input_datetime, query_time = 30, MAIN_DIR = '/data/padialjr/jorge-helio', flare_class_min = 'C'):

    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    flare_class_dict_query = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}

    zero_cross_df = pickle.load(open('{}/data_products/2010_2020_zero_cross.df.merged.pickle'.format(MAIN_DIR), 'rb'))

    masked_df = zero_cross_df[(zero_cross_df['datetime'] > start_time) \
                & (zero_cross_df['datetime'] < end_time) \
                #& (zero_cross_df['instrument'] == 'goes15') \
                & (zero_cross_df['value'] > flare_class_dict_query[flare_class_min])
                                ].reset_index(drop= True)

    return(masked_df)


def noaa_ar_db(input_datetime, query_time = 720, MAIN_DIR = '/data/padialjr/jorge-helio', ID_TEAM = 'HMI SHARP'):

    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    noaa_ar_df = pickle.load(open('{}/data_products/noaa_AR_db.pickle'.format(MAIN_DIR), 'rb'))

    masked_df = noaa_ar_df[(noaa_ar_df['event_start'] > start_time) \
                & (noaa_ar_df['event_end'] < end_time) \
                & (noaa_ar_df['AR_type_name'] == ID_TEAM)].reset_index(drop= True)

    return(masked_df)


def drms_AR_hmi(input_datetime, query_time = 30):


    start_time = input_datetime - timedelta(minutes = query_time)

    end_time = input_datetime + timedelta(minutes = query_time)

    start_timestamp = convert_datetime.convert_datetime_to_timestamp(start_time)

    end_timestamp = convert_datetime.convert_datetime_to_timestamp(end_time)

    # query Xray DB

    engine = sa.create_engine(f'sqlite:////{DATA_PROD_DIR}/hmi_drms_AR.db', echo = False)

    metadata = sa.MetaData()

    conn = engine.connect()

    hmi_drms_AR_table = sa.Table('hmi_drms_AR', metadata, autoload = True, autoload_with = engine)

    hmi_drms_select = sa.select([hmi_drms_AR_table]).where(sa.and_(
                            hmi_drms_AR_table.c.obs_time_stamp >= start_timestamp, 
                            hmi_drms_AR_table.c.obs_time_stamp < end_timestamp
                                    )
                        )

    hmi_sql_df = pd.read_sql(hmi_drms_select,conn)

    hmi_sql_df['obs_date_time'] = convert_datetime.convert_timestamp_to_datetime(hmi_sql_df['obs_time_stamp'])

    hmi_sql_df['first_seen_date_time'] = convert_datetime.convert_timestamp_to_datetime(hmi_sql_df['first_seen_time_stamp'])

    hmi_sql_df['last_seen_date_time'] = convert_datetime.convert_timestamp_to_datetime(hmi_sql_df['first_seen_time_stamp'])

    #deseralize NOAA_ARS and hgs_bbox

    hmi_sql_df['hgs_bbox'] = hmi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['hgs_bbox']), axis=1)

    hmi_sql_df['NOAA_ARS'] = hmi_sql_df.apply(lambda x: sxi_module.json_deserialize(x['NOAA_ARS']), axis=1)


    ordered_df = hmi_sql_df.sort_values(by = 'obs_date_time').reset_index(drop = True)

    return(ordered_df)