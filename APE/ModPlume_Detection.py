#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:07:00 2022.

@author: Manu Goudar
"""

from scipy.interpolate import LinearNDInterpolator as lint
import numpy as np
from skimage.filters import gaussian, threshold_local, sobel
from skimage.segmentation import watershed

# from skimage.morphology import medial_axis
from skimage.measure import label
from .ModuleDataContainers import DataContainer


def watershed_segmentation(init_img, mask):
    """
    Create a marker, elevation map and then segment the image.

    Parameters
    ----------
    init_img : ndarray [n,n]
        Input Image.
    mask : Bool [m,n]
        Masked values for the image.

    Returns
    -------
    Dict of 4 images (ndarrays)
        elevation_map : ndarray Float (m,n)
            Elevation map highlighted by Sobel operator.
        marker_img : ndarray Int (m,n)
            Computed Image Markers.
        segmented_img : ndarray Int (m,n)
            Segmented image from watershed algorithm.
        labeled_img : ndarray Int (m,n)
            Labelled image after segmentation.
    """
    seg_vars = DataContainer()
    # STEP 1:  Highlight the edges by using sobel
    highlight_edges = sobel(init_img)
    seg_vars.__setattr__("elevation_map", highlight_edges)
    # Step 2: Create markers image
    # Center 5x5 pixels
    sh = init_img.shape
    i11 = sh[0] // 2 - 2
    i12 = sh[0] // 2 + 3
    i21 = sh[1] // 2 - 2
    i22 = sh[1] // 2 + 3
    # Center 15x15 pixels
    j11 = i11 - 5
    j12 = i12 + 5
    j21 = i21 - 5
    j22 = i22 + 5
    # initialize marker image
    markers = np.zeros_like(init_img, np.int_)
    # LOW THRESHOLD
    # compute a local threshold based on local mean
    thresh_local = threshold_local(init_img, block_size=7, method="mean")
    cco_1 = init_img <= thresh_local
    markers[cco_1] = 1
    markers[:2, :] = 1
    markers[-2:, :] = 1
    markers[:, :2] = 1
    markers[:, -2:] = 1
    markers[init_img <= np.median(init_img)] = 1
    # UPPER THRESHOLD
    mk1 = np.zeros_like(markers, np.bool_)
    mk1[markers != 1] = True
    co_int_th = label(mk1)
    ids = np.unique(co_int_th[i11:i12, i21:i22])
    mk1[:, :] = 0
    for _id in ids:
        if _id != 0:
            mk1[j11:j12, j21:j22][(co_int_th == _id)[j11:j12, j21:j22]] = 1
            # markers[j11:j12, j21:j22][((init_img >= np.mean(init_img[mk1])) & (markers != 1))[j11:j12, j21:j22]] = 2
    markers[((init_img >= np.mean(init_img[mk1])) & (markers != 1))] = 2
    seg_vars.__setattr__("marker_img", markers)
    # STEP 3: Segment image and label
    segment = watershed(highlight_edges, markers)
    seg_vars.__setattr__("segmented_img", segment)
    # STEP 4: Label the image
    segment[mask] = 1
    labeled = label(segment, background=1, connectivity=1)
    seg_vars.__setattr__("labeled_img", labeled)
    return seg_vars


def get_distance(lat_pts, lon_pts, transform):
    """
    Compute distance for given lat-lon points.

    Parameters
    ----------
    lat_pts : ndarray
        Latitude.
    lon_pts : ndarray
        Longitude.
    transform : Class
        Transfrom from lat-lon to xy.

    Returns
    -------
    Float
        Maximum distance.

    """
    x, y = transform.latlon2xykm(lat_pts, lon_pts)
    dist = np.sqrt(x * x + y * y)
    return dist.max()


def get_segmented_plume(label_img, blocksize, latc, lonc, transform):
    """
    Filter and extract the label of segmented plume.

    Checks number of labels in the region
    Check the size of that plume
    Parameters
    ----------
    label_img : ndarray
        Labeled image.
    blocksize : Int
        Block size where the labels need to be searched.
    latc : ndarray
        Latitude.
    lonc : ndarray
        Longitude.
    transform : Class
        Transfrom from lat-lon to xy.

    Returns
    -------
    Label : Integer
        Label representing the plume.
    flag : bool
        If plume is present of absent.

    """
    """
    This checks if plume is present and returns label with flag

    This returns a flag and the label of the plume
    """
    filt_dist = 25  # km in distance
    sh = label_img.shape
    i1 = sh[0] // 2 - blocksize
    i2 = sh[1] // 2 + blocksize + 1
    # get labels and remove background if present
    idd_labels = list(np.unique(label_img[i1:i2, i1:i2]))
    if 0 in idd_labels:
        idd_labels.remove(0)
    # get the size (total number of points) and
    # maximum distance from source for each above label.
    # If length is less than 40km or max no of points is less than 5 drop it
    _len = len(idd_labels)  # number of identified labels
    # compute stats
    if _len > 0:
        # 3 - [label, number of points, maximum distance]
        _nos = np.zeros((_len, 3))
        ii = 0
        for _lb in idd_labels:
            _nos[ii, 0] = _lb
            cc = label_img == _lb
            _nos[ii, 1] = np.sum(cc)
            _tmp = get_distance(latc[cc], lonc[cc], transform)
            _nos[ii, 2] = _tmp
            ii += 1
    else:  # no labels in the ROI
        print("      No enhancement found")
        return 0, False
    # If maximum number of points is small or the
    # max distance is less than 40 km then dump it
    # Select the one that has maximum numbers
    if _len > 1:
        _nmax = np.argmax(_nos[:, 1])
        # if (_nos[_nmax, 1].max() < 5) or (_nos[_nmax, 2] < filt_dist):
        if _nos[_nmax, 2] < filt_dist:
            print("      Short plumes")
            return 0, False
        else:
            return np.int_(_nos[_nmax, 0]), True
    elif _len == 1:
        # if (_nos[0, 1].max() < 5) or (_nos[0, 2] < filt_dist):
        if _nos[0, 2] < filt_dist:
            print("      Short plumes")
            return 0, False
        else:
            return np.int_(_nos[0, 0]), True


def segment_image_plume(_lat, _lon, co_orig, mask, transform, blocksize=1):
    """
    Segments the input image using watershed algorithm.

    Segmentation and labelling is done by this algorithm

    Parameters
    ----------
    data : Container
        Contains the extracted satellite data for a plume.
    transform : Transfrom class
        To transform from lat-lon to xy coordinates.
    blocksize : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    flag : Bool
        Plume is detected or not.
    seg_vars : Dict Container
        Contains the intermediate data and the detected plume.

    """
    # Interpolate nan values for plume detection
    idx = np.isnan(co_orig)
    intp = lint((_lat[~idx].ravel(), _lon[~idx].ravel()), co_orig[~idx].ravel(), fill_value=0)
    co_int = intp(_lat, _lon)

    # smoothen the image by gaussian
    co_gauss = gaussian(co_int, sigma=0.5, preserve_range=True)

    # Watershed segmentation
    seg_vars = watershed_segmentation(co_gauss.copy(), mask)

    # Check if plume is present
    label_id, flag = get_segmented_plume(seg_vars.labeled_img, blocksize, _lat, _lon, transform)

    seg_vars.__setattr__("label_id", label_id)
    seg_vars.__setattr__("f_plumedetect", flag)
    if flag:
        # get mask
        plumemask = seg_vars.labeled_img == label_id
        seg_vars.__setattr__("plumemask", plumemask)
    return seg_vars
