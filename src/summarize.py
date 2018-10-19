import numpy as np


def summarize(img, boundary_locations):
    # get shape
    y = img.shape[0]
    x = img.shape[1]
    loc_diffs = np.diff(boundary_locations)
    # get boundary count
    count = boundary_locations.shape[0]
    # get max and min boundary distance
    min = np.min(loc_diffs)
    max = np.max(loc_diffs)
    return np.array([x,y,count,max,min])


