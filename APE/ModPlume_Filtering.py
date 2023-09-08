#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter if there are firesources near identified plume.

Created on Tue Jun  7 12:40:25 2022.
@author: Manu Goudar
"""
import numpy as np
from scipy.spatial import cKDTree
from .ModuleRefineGrids import RefineGridsUniform
from .ModDataPrepare_VIIRSData import extract_roi


def pointswithincertaindistance(lat_nodes, lon_nodes, plumemask, _firedata, distanceindeg=0.05):
    """Find indices that are within 0.05deg from the plume.

    check if points are in detected plume and get their cluster labels
    Do instead of searching each point in a given grid, find the minimum distance
    between points and say if it is inside or outside.

    Parameters
    ----------
    lat_nodes : Array
        latitude nodes
    lon_nodes : Array
        Longitude nodes
    plumemask : Array(Bool)
        Plume mask
    _firedata : Dataframe
        all VIIRS pixels inside the extent
    distanceindeg : Float
        distance in degrees from the plume

    Return
    --------
    _lls : Int
        Indices corresponding to the _firedata

    """
    ff = RefineGridsUniform(4)
    xn, yn, xc, yc = ff.resize_coordinates(lat_nodes, lon_nodes)
    plm = (ff.resize_values(plumemask)).astype(np.bool_)
    cc = np.c_[xc[plm], yc[plm]]
    tree = cKDTree(cc)

    # Find all points that are 0.05 deg distance from the plume pixels
    dis, loc = tree.query(_firedata[["latitude", "longitude"]].values)
    _lls = _firedata.labels[(dis < distanceindeg)]
    return _lls


def nonclusteredpixels(_lls, _tmp_nos):
    """Find all non-clustered VIIRS pixels

    Parameters
    ----------
    _lls : Array(Int)
        Indices
    _tmp_nos : Int
        Indicator for the filter

    Return
    --------
    _tmo_nos : Int
        Indicator for the filter
        Indicates that if other fires exist next to plume or not
    """
    if (_lls == -1).sum() > 10:
        _tmp_nos += 1
        print("          There are more than 10 fire counts")
    return _tmp_nos


def checkforotherclusters(_lls, fireclusters, orbit, _tmp_nos):
    """Check for other clusters

    Parameters
    ----------
    _lls : Array(Int)
        Indices
    fireclusters : Pandas Dataframe
        Data frame containing clustered fires
    orbit : Int
        Orbit of the fire in consideration
    _tmp_nos : Int
        Indicator for the filter

    Return
    --------
    _tmo_nos : Int
        Indicator for the filter
        Indicates that if other fires exist next to plume or not

    Examples
    --------
    """
    # First compute all fire clusters for a given orbit
    _lbs = _lls.unique()
    if len(_lbs) > 0:
        flg1 = fireclusters.labels.isin(_lbs)
        flg2 = fireclusters.orbits.values == orbit
        _clstdata = (fireclusters[(flg1 & flg2)]).copy(deep=True)
        if len(_clstdata) > 0:
            _tmp_nos += len(_clstdata)
            print("          Other clusters are present. Number: ", len(_clstdata))
    return _tmp_nos


def filter_good_plumes(_data, plumemask, fireclusters, viirsdata):
    """filter_good_plumes _summary_

    Parameters
    ----------
    _data : Data Class
        Container for satellite data
    plumemask : Matrix(bool)
        Plumemask
    cluster_final : DataFrame
        Pandas dataframe containing clustered fire.
    viirsdata : DataFrame
        Contain all fire data

    Returns
    -------
    True : Bool
        Flag if the plume is good or not
    cluster_final : Dataframe
        Contains clustered fire
    """
    # source
    src = _data.source
    # define extent and get all VIIRS fire counts in that box
    _extent = np.array([src[1] - .5, src[1] + .5, src[0] - .5, src[0] + .5])
    fire_roi = extract_roi(viirsdata, _extent)
    # Remove all viirs pixels corresponding to the fire
    # corresponding to the actual fire source. And reset index
    _firedata = fire_roi[fire_roi.labels != fireclusters.labels[_data.fire_id]]
    _firedata.reset_index(drop=True, inplace=True)
    # No of clusters indicator in data
    _tmp_nos = 0

    # get all VIIRS pixels within certain distance from plume
    _lls = pointswithincertaindistance(_data.lat_nodes.copy(),
                                       _data.lon_nodes.copy(),
                                       plumemask, _firedata)
    # Two filters
    # Find all individual fire counts or non clustered VIIRS pixels
    _tmp_nos = nonclusteredpixels(_lls, _tmp_nos)
    # Check if there are other identified clusters
    _tmp_nos = checkforotherclusters(_lls, fireclusters, _data.orbit, _tmp_nos)
    # return values
    if _tmp_nos < 1:
        return True, fireclusters
    else:
        return False, fireclusters
