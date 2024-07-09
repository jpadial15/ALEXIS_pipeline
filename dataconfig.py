import os


HOME_DIR = '/home/padialjr/ALEXIS_pipeline'
MAIN_DIR_NEW = '/home/padialjr/ALEXIS_pipeline'
MAIN_DATA_DIR = '/mnt/e/image_data'

#############################################
JSOC_DL_DIR = "jsoc.download"

NOTIFY_EMAIL_ADDR = 'jorge.r.padial.doble@vanderbilt.edu'

############################################

# paths that live in server
DATA_DIR_PRODUCTS = f'{MAIN_DIR_NEW}/data_products'

DATA_DIR_HMI_HEK_AR = f'{MAIN_DIR_NEW}/hmi_hek_ar'

DATA_DIR_FLARE_CANDIDATES = f'{MAIN_DIR_NEW}/flare_candidates'

DATA_DIR_HMI_DRMS = f'{MAIN_DIR_NEW}/hmi_drms_ar'

DATA_DIR_PLOTS = f'{MAIN_DIR_NEW}/PLOTS'

DATA_DIR_IMG_DATA = f'{MAIN_DATA_DIR}'

# paths that live on harddrive

DATA_DIR_HEK_FLARES = f'{MAIN_DATA_DIR}/hek_flares_data'

DATA_DIR_GOES_FLUX = f'{MAIN_DATA_DIR}/xrs_data_flux'

DATA_DIR_GOES_SXI = f'{MAIN_DIR_NEW}/goes_data_sxi'

DATA_DIR_IMG_RUFFUS_OUTPUT = f'{DATA_DIR_IMG_DATA}/ruffus_output_files'

DATA_DIR_AIA = f'{MAIN_DATA_DIR}/aia_data'

RUFFUS_SQLITE_CHECKPOINT_DIR = f'{HOME_DIR}/ruffus_sqlite_checkpoints'

