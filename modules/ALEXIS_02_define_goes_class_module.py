import sys 
import os 
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

from glob import glob
import re
import pickle
import numpy as np
import os
import pandas as pd
# from ruffus.task import fill_queue_with_job_parameters
from sunpy.time import TimeRange
import time
from datetime import datetime, timedelta
import sqlalchemy as sql
import re
import warnings


warnings.filterwarnings("ignore")
from astropy.io import fits
from skimage.feature import peak_local_max
from ruffus import *
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from scipy.stats import linregress

from skimage.transform import rotate

import matplotlib.pyplot as plt
from astropy.wcs import WCS
import sunpy.map
from astropy import units as u
from scipy.signal import find_peaks

import sys


from astropy.coordinates import SkyCoord

import scipy.stats
from scipy.stats import mstats
from modules import query_the_data


import convert_datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import clean_img_data_02

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import sxi_module
from time import sleep
from random import randint

from modules import helio_reg_exp_module

import itertools
from scipy.optimize import nnls

import cvxpy as cp

from sunpy.coordinates import frames

import coordinate_conversion_module

import ast
import itertools

from scipy.spatial import KDTree

import itertools









GOES_CONVERSION_DICT = {
    "X": u.Quantity(1e-4, "W/m^2"),
    "M": u.Quantity(1e-5, "W/m^2"),
    "C": u.Quantity(1e-6, "W/m^2"),
    "B": u.Quantity(1e-7, "W/m^2"),
    "A": u.Quantity(1e-8, "W/m^2"),
}



def find_closest_peak(plus_minus_2_min_data):

    peaks, _ = find_peaks(plus_minus_2_min_data.value, distance = 15)

    # timeseries

    peak_date_times = [plus_minus_2_min_data.date_time.iloc[this_peak] for this_peak in peaks]
    peak_values = [plus_minus_2_min_data.value.iloc[this_peak] for this_peak in peaks]

    # return()

    return(peak_date_times, peak_values)



def mag_to_class(goesflux):

    decade = np.floor(np.log10(goesflux.to("W/m**2").value))
    # invert the conversion dictionary
    conversion_dict = {v: k for k, v in GOES_CONVERSION_DICT.items()}
    if decade < -8:
        str_class = "A"
        decade = -8
    elif decade > -4:
        str_class = "X"
        decade = -4
    else:
        str_class = conversion_dict.get(u.Quantity(10**decade, "W/m**2"))
    goes_subclass = 10**-decade * goesflux.to("W/m**2").value
    return f"{str_class}{goes_subclass:.3g}"



def find_xrs_data(work_group):

    query_time = helio_reg_exp_module.date_time_from_flare_candidate_working_dir(work_group.working_dir.iloc[0])
    
    img_instrument = work_group.iloc[0].img_instrument
    
    xrs_data = query_the_data.xray_sql_db_for_flare_def_class(query_time,img_instrument )

    masked_xrs = xrs_data[(xrs_data.instrument == work_group.iloc[0].xrs_telescope)]

    xrs_b = masked_xrs[masked_xrs.wavelength == 'B'].sort_values(by = 'date_time')
    
    return(xrs_b)


IN_HOUSE_GOES_CONVERSION_DICT = {
     -4:"X",
     -5:"M",
     -6:"C",
     -7:"B",
     -8:"A",
}

def in_house_mag_to_class(this_mag):

    formatted_result = f"{this_mag:.3e}"

    flare_number = re.findall(r'\d{1}.\d{2}', formatted_result)[0]

    flare_exponent = np.float(re.findall(r'-\d{2}', formatted_result)[0])

    if flare_exponent < -8.0:
        flare_str = 'A'
    elif flare_exponent > -4.0:
        flare_str = 'X'
    else:
        flare_str = IN_HOUSE_GOES_CONVERSION_DICT[flare_exponent]

    # make output flare class

    output_flare_class = f'{flare_str}{flare_number}'

    # print('done', output_flare_class, formatted_result,flare_exponent)

    return(output_flare_class)

