import os 
import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../modules/')

from glob import glob
import re
import pickle
import numpy as np
import os
import pandas as pd
import sys


from ruffus import *

import dataconfig


from modules import ALEXIS_02_define_goes_class_module
from modules import ALEXIS_02_plotting_module_movie
from modules import clean_img_data_02
from modules import helio_reg_exp_module



# start_flare_list = glob(f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/*/ALEXIS_flares_w_harp_goes_class_and_known_flare_meta.pickle')

start_flare_list = glob(f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/*/defined_flares_w_harp_meta_and_goes_flare_class.pickle')

# start_flare_list = [f'{dataconfig.DATA_DIR_FLARE_CANDIDATES}/flarecandidate_C1.1_at_2011-11-29T01_16_00_16.working/ALEXIS_flares_w_harp_goes_class_and_known_flare_meta.pickle']

print(start_flare_list)


@subdivide(start_flare_list, formatter(),
            # Output parameter: Glob matches any number of output file names
            "{path[0]}/{basename[0]}.*.v3_initialize_summary_jpegs_for_movie.pickle",
            # Extra parameter:  Append to this for output file names
            "{path[0]}/{basename[0]}")
def initialize_summary_jpeg_for_movie(infile, outfiles, output_file_name_root):

    # start_time = time.time()

    # print(infile)

    these_flares = pd.DataFrame(pickle.load(open(infile, 'rb')))

    working_dir = these_flares.iloc[0].working_dir

    good_quality_data_df = pickle.load(open(f'{working_dir}/initialize_with_these_files_df.pickle', 'rb'))

    mask_properties = these_flares.iloc[0]

    this_wl, this_inst, this_tel = mask_properties.img_wavelength, mask_properties.img_instrument, mask_properties.img_telescope

    these_images = good_quality_data_df[(good_quality_data_df.wavelength == this_wl) & (good_quality_data_df.telescope == this_tel) & (good_quality_data_df.instrument == this_inst) & (good_quality_data_df.QUALITY == 0)]

    these_images['catalog_file'] = [infile for _ in these_images.file_path]

    # load and save xrs data

    xrs_b_data = ALEXIS_02_define_goes_class_module.find_xrs_data(these_flares)

    xrs_real_data_file_path = os.path.join(working_dir, 'real_xrs_data.pickle')

    pickle.dump(xrs_b_data, open(xrs_real_data_file_path, 'wb'))

    these_images['real_xrs_data_path'] = [xrs_real_data_file_path for _ in these_images.file_path]

    for output_number, output_row in enumerate(these_images.sort_values(by = 'date_time').to_dict('records')):

        # print(output_row['date_time'])

        output_file_name = f'{output_file_name_root}.movie_making_{output_number}.v3_initialize_summary_jpegs_for_movie.pickle'

        pickle.dump(output_row, open(output_file_name, 'wb'))

@transform(initialize_summary_jpeg_for_movie, suffix('.v3_initialize_summary_jpegs_for_movie.pickle'), '.img_for_movie_made.pickle')
def make_movie_plots(infile, outfile):

    input_dict = pickle.load(open(infile, 'rb'))

    image_df = pd.DataFrame([input_dict])

    xrs_df = pickle.load(open(image_df.iloc[0].real_xrs_data_path, 'rb'))

    flares_df = pickle.load(open(image_df.iloc[0].catalog_file, 'rb'))

    if input_dict['instrument'] == 'AIA':

        img_dict, data, header = clean_img_data_02.clean_aia_data(image_df.to_dict('records')[0])

    if input_dict['instrument'] == 'SXI':

        img_dict, data, header = clean_img_data_02.clean_sxi_data(image_df.to_dict('records')[0])

    figure_name = ALEXIS_02_plotting_module_movie.plot_results_for_movie(image_df, flares_df, xrs_df, data, header)

    image_df['figure_name'] = [figure_name for _ in image_df.file_path]

    pickle.dump(image_df, open(outfile, 'wb'))


# pipeline_run([make_movie_plots], multiprocess= 15)

@collate(make_movie_plots, formatter(r'flarecandidate_[A-Z]\d{1,}.\d{1,}_at_\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_\d{2}.working\/defined_flares_w_harp_meta_and_goes_flare_class.movie_making_\d{1,3}.img_for_movie_made.pickle'), output = '{subpath[0][0]}/collate_jpeg_df.pickle')
def join_movie_files(infiles, outfile):

    output_df = pd.concat([pickle.load(open(infile, 'rb')) for infile in infiles])

    pickle.dump(output_df, open(outfile, 'wb'))

    # print(len(infiles), outfile)

# pipeline_run([join_movie_files], multithread = 30, verbose = 1)

@transform(join_movie_files, suffix('collate_jpeg_df.pickle'), 'made_movie.pickle')
def ffmpeg_movie_maker(infile, outfile):

    input_df = pickle.load(open(infile, 'rb'))

    working_dir = input_df.iloc[0].working_dir

    wd = helio_reg_exp_module.work_dir_from_flare_candidate_input_string(working_dir)

    ffmpeg_string = f"ffmpeg -framerate 10 -pattern_type glob -i '{working_dir}/movie_file_*.jpeg' -c:v libx264 -pix_fmt yuv420p {working_dir}/{wd}_summary_vid.mp4"

    os.system(ffmpeg_string)    # $ ffmpeg -framerate 1 -pattern_type glob -i '*.jpg' -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4

    input_df['movie_path'] = [f'{working_dir}/{wd}_summary_vid.mp4' for _ in input_df.figure_name]

    pickle.dump(input_df, open(outfile, 'wb'))

pipeline_run([ffmpeg_movie_maker], multiprocess= 8)

