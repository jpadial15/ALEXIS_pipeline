
import re

import numpy as np
import os
import pandas as pd

from astropy.wcs import WCS
import sunpy.map
from astropy import units as u
import sys

from astropy.coordinates import SkyCoord

from sunpy.coordinates import frames
import math


# our coord conversion without WCS module

def pix_to_hpc(coord_pair, raw_header):

    coord_array = np.array(coord_pair)

    x_hpc = (coord_array[:,0] - raw_header['CRPIX1']) * raw_header['CDELT1']

    y_hpc = (coord_array[:,1] - raw_header['CRPIX2']) * raw_header['CDELT1']

    return([[x_hpc, y_hpc] for x_hpc, y_hpc in zip(x_hpc, y_hpc)])

def hpc_to_pix(coord_pair, raw_header):

    coord_array = np.array(coord_pair)

    x_pix = (coord_array[:,0] * (raw_header['CDELT1'])**-1) + raw_header['CRPIX1']

    y_pix = (coord_array[:,1] * (raw_header['CDELT1'])**-1) + raw_header['CRPIX2']

    return([[x_pix, y_pix] for x_pix, y_pix in zip(x_pix, y_pix)])

# our coord conversion without WCS module end

def is_coods_outside_of_solar_borders( x_pix, y_pix, data_map):

    z_squared = (x_pix - data_map.fits_header['CRPIX1'])**2 + (y_pix - data_map.fits_header['CRPIX2'])**2


    try:
        check_if_wn = z_squared - data_map.fits_header['R_SUN']**2
    except:
        check_if_wn = z_squared - data_map.fits_header['RSUN']**2

    # print(check_if_wn)

    if check_if_wn < 0:

        return False

    else:
        return True


def find_closest_point_to_circle(outer_perimiter, x_pix, y_pix):


        outer_perimiter['x_dist'] = [((x_pix - x_circ)**2) for x_circ in outer_perimiter.x_circle]

        outer_perimiter['y_dist'] = [((y_pix - y_circ)**2) for y_circ in outer_perimiter.y_circle]

        outer_perimiter['distance'] = [np.sqrt(x_dist + y_dist) for x_dist, y_dist in zip(outer_perimiter.x_dist, outer_perimiter.y_dist)]

        closest_point_row = outer_perimiter.sort_values(by = 'distance', ascending= True).iloc[0]

        return(closest_point_row)


def pixel_to_hpc(these_pixels, data_map):

    x_pix, y_pix = these_pixels[0], these_pixels[1]

    sky = WCS(data_map.fits_header).pixel_to_world(x_pix, y_pix)


    sky_x_arc, sky_y_arc = sky.Tx, sky.Ty

    hpc_sky_cooord = SkyCoord(sky_x_arc, sky_y_arc, frame = frames.Helioprojective, observer = data_map.observer_coordinate)


    hpc_x, hpc_y = sky_x_arc.value, sky_y_arc.value

    return(hpc_x, hpc_y)




    # return(hpc_sky_cooord)

def pixel_to_hgs(these_pixels, data_map):

    x_pix, y_pix = these_pixels[0], these_pixels[1]

    sky = WCS(data_map.fits_header).pixel_to_world(x_pix, y_pix)

    sky_x_arc, sky_y_arc = sky.Tx, sky.Ty

    hpc_sky_cooord = SkyCoord(sky_x_arc, sky_y_arc, frame = frames.Helioprojective, observer = data_map.observer_coordinate)

    hgs_sky_coor = hpc_sky_cooord.transform_to(frame = frames.HeliographicStonyhurst)

    hgs_x, hgs_y = hgs_sky_coor.lon.value, hgs_sky_coor.lat.value

    return(hgs_x, hgs_y)

def hpc_to_hgs(hpc_coords, data_map):

    hpc_x, hpc_y = hpc_coords[0]*u.arcsec, hpc_coords[1]*u.arcsec

    hpc_sky_cooord = SkyCoord(hpc_x, hpc_y, frame = frames.Helioprojective, observer = data_map.observer_coordinate)

    hgs_sky_coor = hpc_sky_cooord.transform_to(frame = frames.HeliographicStonyhurst)

    hgs_x, hgs_y = hgs_sky_coor.lon.value, hgs_sky_coor.lat.value

    return(hgs_x, hgs_y)

def algorithm_to_project_pixel_onto_surface(x_pix, y_pix, data_map):

    raw_header = data_map.fits_header

    try:
        r_sun = raw_header['R_SUN']
    except:
        r_sun = raw_header['RSUN']

    # make outerperimiter
    rad_list = np.linspace(0,2, 10000)*np.pi

    x2 = [raw_header['CRPIX1'] + r_sun * np.cos(rad) for rad in rad_list]
    y2 = [raw_header['CRPIX2'] + r_sun * np.sin(rad) for rad in rad_list]

    outer_perimiter = pd.DataFrame({'x_circle': x2, 'y_circle': y2})

    is_outside_solar_border = is_coods_outside_of_solar_borders(x_pix, y_pix, data_map)

    print(is_outside_solar_border)

    hpc_coords = pixel_to_hpc([x_pix, y_pix], data_map)

    print(hpc_coords)

    hgs_coords = hpc_to_hgs(hpc_coords, data_map)

    original_point_not_good = math.isnan(hgs_coords[0]) # boolean

        # ax.scatter(first_row.x_pix, first_row.y_pix, c = 'magenta', s = 100, marker = 'x')

    # print(original_point_not_good)
        
    if original_point_not_good == True:

        closest_point = (find_closest_point_to_circle(outer_perimiter, x_pix, y_pix))

        closest_point_pixels = [closest_point.x_circle, closest_point.y_circle]

        # ax.scatter(first_row.x_pix, first_row.y_pix, s = 100, c = 'magenta', marker = 'x')

        new_hgs = (pixel_to_hgs(closest_point_pixels, data_map))

        
        closest_point_not_good = math.isnan(new_hgs[0]) # boolean


        new_vector = np.array(closest_point_pixels)

        old_vector = np.array([x_pix, y_pix])

        test_resultant_vector = (old_vector - new_vector)


        # make sure that the resultant vector isn't very small.
        # this happens when the clustered pixels lie outside the
        # solar border but are still very close to that perimiter. 

        bool_array = [i <= 9e-2 for i in test_resultant_vector]

        are_both_coords_small = bool_array[0]*bool_array[1]

        if are_both_coords_small == True:
            resultant_vector = (old_vector - new_vector)*100
        else: 
            resultant_vector = (old_vector- new_vector)/100

        print(test_resultant_vector)
            
        print('resultant',resultant_vector)

        # print(closest_point_not_good)


        # if the closest point on the circle doesnt work, lets take 
        if closest_point_not_good == True:

            # make a list of possible locations by taking the difference of the new point to the old point

            number_of_attempts = np.arange(10000) + 1

            iterate_through_these = (closest_point_pixels - ((resultant_vector/2)*N) for N in number_of_attempts)
            n = 0 
            for vector in iterate_through_these:

                print(vector)

                are_new_hgs_no_good2 = math.isnan(pixel_to_hgs([vector[0], vector[1]], data_map)[0])

                # print(n)

                if are_new_hgs_no_good2 == False:

                    print(n)

                    these_pixels_work = vector

                    break

                if n >= 9999:

                    print(f'iteration through possible projects of candidate coords to solar surface exceded.')
                    sys.exit()
                # print(vector)
                n = n+1

                # print

        else:

            these_pixels_work = closest_point_pixels  

    else:

        these_pixels_work = x_pix, y_pix


    return(these_pixels_work)