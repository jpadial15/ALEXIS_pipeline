import re
import numpy as np
import pandas as pd




def parse_sxi_filename(input_str):

    """
    Takes in the full path of original sxi FTS file

    Returns dict of {'date_time': date_time_obj,'data_level': level,'instrument': f'goes{inst}','filename': input_str}


    """

    level= re.findall(r'(?<=\d{9}_)[A-Za-z0-9]{2}', input_str)[0]

    inst = re.findall(r'(?<=\d{9}_[A-Za-z0-9]{2}_)\d{2}', input_str)[0]

    date_time_str = re.findall(r'\d{8}_\d{9}', input_str )[0]

    date_time_obj = pd.to_datetime(date_time_str, format = '%Y%m%d_%H%M%S%f', utc = True)

    output_dict = {'date_time': date_time_obj, 
                        'data_level': level, 
                        'instrument': f'goes{inst}', 
                        'filename': input_str}

    return(output_dict)


def parse_hinode_filename(input_str):

    """
    Takes in the full path of original sxi FTS file

    Returns dict of {'date_time': date_time_obj,'data_level': level,'instrument': f'goes{inst}','filename': input_str}


    """

    # level= re.findall(r'(?<=\d{9}_)[A-Za-z0-9]{2}', input_str)[0]

    # inst = re.findall(r'(?<=\d{9}_[A-Za-z0-9]{2}_)\d{2}', input_str)[0]

    date_time_str = re.findall(r'\d{8}_\d{6}.\d{1}', input_str )[0]

    date_time_obj = pd.to_datetime(date_time_str, format = '%Y%m%d_%H%M%S.%f', utc = True)

    output_dict = {'date_time': date_time_obj, 'file_name': input_str}

    return(output_dict)

def parse_aia_filename(input_str):

    """
    Takes in the full path of original aia fits file

    Returns dict of {'date_time': date_time_obj, 'WL': WL, 'filename': input_str}
    """

    WL = re.findall(r'(?<=aia_lev\d{1}_).+?(?=a)', input_str)[0]

    date_time_str = re.findall(r'\d{4}_\d{2}_\d{2}t\d{2}_\d{2}_\d{2}_\d{2}', input_str )[0]

    date_time_obj = pd.to_datetime(date_time_str, format = '%Y_%m_%dt%H_%M_%S_%f', utc = True)

    output_dict = {'date_time': date_time_obj, 'WL': WL, 'filename': input_str}

    return(output_dict)

def parse_sxi_fits_from_start(input_str):

    filename = re.search(r"^.*(?=(\.download.pickle))", input_str)[0]

    return(filename)

def parse_sxi_wo_FTS(input_str):

    filename = re.search(r"^.*(?=(\.FTS.download.pickle))", input_str)[0]

    return(filename)


def date_time_from_flare_candidate_working_dir(working_dir):

    date_time_str = re.findall(r'\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}',working_dir  )[0][:-3]

    clean_fits_full_path = re.sub('_', ':', date_time_str)

    return(pd.Timestamp(clean_fits_full_path, tz = 'utc'))

def flare_class_from_flare_candidate_working_dir(working_dir):

    merged_flare_class = re.findall(r'[A-Z]\d{1,}.\d{1,}',working_dir  )[0]


    return(merged_flare_class)

def work_dir_from_flare_candidate_input_string(working_dir):

    work_dir = re.findall(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working',working_dir  )[0]


    return(work_dir)