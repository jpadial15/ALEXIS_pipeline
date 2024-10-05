

import numpy as np

from astropy.wcs import WCS

from astropy import units as u



from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames







def return_harp_lifetime_span_hgs_bbox(this_example):

    x_max_list = []
    y_max_list = []

    x_min_list = []
    y_min_list = []

    for hgs_bbox in this_example.hgs_bbox:

        x_max_list.append(np.max(np.array(hgs_bbox)[:,0]))
        x_min_list.append(np.min(np.array(hgs_bbox)[:,0]))
        y_max_list.append(np.max(np.array(hgs_bbox)[:,1]))
        y_min_list.append(np.min(np.array(hgs_bbox)[:,1]))


    lower_left = [np.min(x_min_list), np.min(y_min_list)]
    upper_left = [np.min(x_min_list), np.max(y_max_list)]
    upper_right = [np.max(x_max_list), np.max(y_max_list)]
    lower_right = [np.max(x_max_list), np.min(y_min_list)]


    span_of_harp_over_time = [lower_left, lower_right, upper_right, upper_left, lower_left]

    return(span_of_harp_over_time)


def return_pixel_ar_drms_bbox(element, data_map):

    hgs_bbox = element*u.deg

    hgs_sky_coord = SkyCoord(hgs_bbox, frame = frames.HeliographicStonyhurst, observer = data_map.observer_coordinate)

    sky = WCS(data_map.fits_header).world_to_pixel(hgs_sky_coord)

    # print(sky)
    list_of_pix_coords = [[x_coord,y_coord] for x_coord,y_coord in zip(sky[0],sky[1])]

    return(list_of_pix_coords)




def check_flare_coord_wn_AR(AR_hpc_bbox, candidate_flare_coords):

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