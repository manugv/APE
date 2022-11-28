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


# TODO rewrite this
def filter_good_plumes(kk, _data, plumemask, cluster_final, viirsdata):
    """
    kk is the index in cluster_final.

    cluster_final is the Orbits and labels
    fire_srcs are sources of fire
    """
    # source
    src = np.array([cluster_final.latitude[kk], cluster_final.longitude[kk]])
    _extent = np.array([src[1] - 0.5, src[1] + 0.5, src[0] - 0.5, src[0] + 0.5])
    fire_roi = extract_roi(viirsdata, _extent)

    _tmp_nos = 0

    # check if points are in detected plume and get their cluster labels
    ff = RefineGridsUniform(4)
    xn, yn, xc, yc = ff.resize_coordinates(_data.lat_nodes.copy(), _data.lon_nodes.copy())
    plm = (ff.resize_values(plumemask)).astype(np.bool_)
    cc = np.c_[xc[plm], yc[plm]]
    tree = cKDTree(cc)
    _firedata = fire_roi[fire_roi.labels != cluster_final.labels[kk]]
    _firedata.reset_index(drop=True, inplace=True)
    dis, loc = tree.query(_firedata[["latitude", "longitude"]].values)
    _lls = _firedata.labels[(dis < 0.05)]
    # _frp = np.sum(_firedata.frp.values[(dis < 0.05)])
    _lbs = _lls.unique()
    if -1 in _lbs:
        i_nos = (_lls == -1).sum()
    else:
        i_nos = 0
    # indivudual fire counts
    if (i_nos > 10) or (len(_lls) > 20):
        _tmp_nos += 1

    # get all lat-lon data from identified nearby labels
    if len(_lbs) > 0:
        _clstdata = (
            cluster_final[
                ((cluster_final.labels.isin(_lbs)) & (cluster_final.orbits.values == _data.orbit))
            ]
        ).copy(deep=True)
        _clstdata["dis"], loc1 = tree.query(_clstdata[["latitude", "longitude"]].values)

        # switch readflag label to false for all labels near to the plume
        for i, _df in _clstdata.iterrows():
            info = True
            pt = np.array([_df.latitude, _df.longitude])
            _dist = np.sqrt(np.sum((src[:] - pt[:]) ** 2))
            if _dist < 0.05:
                # very near to source then assimilate
                fire_roi.labels[fire_roi.labels == _df.labels] = cluster_final.labels[kk]
                cluster_final.loc[i, "readflag"] = False
                info = False
            # switch readflag label to false for all labels near to the plume
            if info & (_df.dis < 0.1):
                _tmp_nos += 1
                cluster_final.loc[i, "readflag"] = False
    # return values
    if _tmp_nos < 1:
        return fire_roi, True, cluster_final
    else:
        print("cluster are many", _tmp_nos)
        return fire_roi, False, cluster_final
