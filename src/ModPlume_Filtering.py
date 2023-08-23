#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter if there are firesources near identified plume.

Created on Tue Jun  7 12:40:25 2022.
@author: Manu Goudar
"""
import numpy as np
from scipy.spatial import cKDTree
from ModuleRefineGrids import RefineGridsUniform
from ModDataPrepare_VIIRSData import extract_roi


def filter_good_plumes(kk, _data, plumemask, cluster_final, viirsdata):
    """filter_good_plumes _summary_

    Parameters
    ----------
    kk : int
        index of the fire.
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
    _extent = np.array([src[1] - 0.5, src[1] + 0.5, src[0] - 0.5, src[0] + 0.5])
    fire_roi = extract_roi(viirsdata, _extent)
    # Get all points inside an extent but remove all fire counts
    # corresponding to the actual fire source. And reset index
    _firedata = fire_roi[fire_roi.labels != cluster_final.labels[kk]]
    _firedata.reset_index(drop=True, inplace=True)

    # No of clusters indicator in data
    _tmp_nos = 0
    # check if points are in detected plume and get their cluster labels
    # Do instead of searching each point in a given grid, find the
    #  minimum distance between points and say if it is inside or outside.
    # step 1 : Create a tree
    ff = RefineGridsUniform(4)
    xn, yn, xc, yc = ff.resize_coordinates(_data.lat_nodes.copy(), _data.lon_nodes.copy())
    plm = (ff.resize_values(plumemask)).astype(np.bool_)
    cc = np.c_[xc[plm], yc[plm]]
    tree = cKDTree(cc)

    # Find all points that are 0.05 deg distance from the plume pixels
    dis, loc = tree.query(_firedata[["latitude", "longitude"]].values)
    _lls = _firedata.labels[(dis < 0.05)]

    # Find all individual fire counts
    _lbs = _lls.unique()
    if -1 in _lbs:
        i_nos = (_lls == -1).sum()
        # indivudual fire counts are greater than 10
        if i_nos > 10:
            _tmp_nos += 1
            print("          There are more than 10 fire counts")

    # Check if there are other identified clusters nearby
    # First compute all fire clusters for a given orbit
    # Step 2 find the distance of the plume of the cluster
    if len(_lbs) > 0:
        flg1 = cluster_final.labels.isin(_lbs)
        flg2 = cluster_final.orbits.values == _data.orbit
        _clstdata = (cluster_final[(flg1 & flg2)]).copy(deep=True)
        #
        if len(_clstdata) > 0:
            _tmp_nos += len(_clstdata)
            print("          Other clusters are present. Number: ", len(_clstdata))
            for i, _df in _clstdata.iterrows():
                cluster_final.loc[i, "readflag"] = False
    # return values
    if _tmp_nos < 1:
        return True, cluster_final
    else:
        return False, cluster_final
