#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to read VIIRS data and cluster the data.

Created on Fri Jun  3 11:44:14 2022
@author: Manu Goudar
"""

from pandas import read_csv
from pathlib import Path
from numpy import column_stack
from sklearn.cluster import DBSCAN

# from sklearn.metrics import silhouette_score


def extract_roi(df, extent):
    """
    Extract data in the region of interest.

    Parameters
    ----------
    df : Pandas dataframe
        Data frame containing VIIRS data.
    extent : Array of size 4
        The extent withing which we need to extract the data.

    Returns
    -------
    Pandas Dataframe
        VIIRS data withing given region of interest.

    """
    return df[
        (df.longitude > extent[0])
        & (df.longitude < extent[1])
        & (df.latitude > extent[2])
        & (df.latitude < extent[3])
    ]


def read_firedata(init_params, st_date):
    """
    Read and extract VIIRS data for a given Region of interest.

    Parameters
    ----------
    init_params : A dict of parameters containing following
        extent : Array of size 4
            Region of interest.
        transform : Class to transform coordinate system
            Transforms from lat-lon to xy coordinates.
        viirsdatadir: str
            Points to the VIIRS data directory
    st_date : datetime type
        Date on which the data needs to be read.

    Returns
    -------
    Bool
        If file exists or not
    Pandas Dataframe
        VIIRS database.

    """
    _file = init_params.viirsdir + st_date.strftime("%Y-%m-%d") + ".csv"
    print(_file)
    # Check if the file is present in the given directory
    if (Path(_file)).is_file():
        # read VIIRS data
        _tmp0 = read_csv(_file)
        # Filter for day time data
        _tmp0 = _tmp0[_tmp0.daynight == "D"]
        # Extract the ROI and reset dataframe
        result = extract_roi(_tmp0, init_params.roi)
        res = result.reset_index(drop=True)
        # Transform data to x,y in kms
        x, y = init_params.transform.latlon2xykm(res.latitude, res.longitude)
        _id = res.shape[1]
        res.insert(_id, "xkms", x)
        res.insert(_id, "ykms", y)
        return True, res
    else:
        return False, []


def get_clusters(xy, eps, minsamples):
    """
    Clusters given data based on eps and minsamples.

    Parameters
    ----------
    xy : ndarray of shape (n_samples, 2)
        Array containing x-y coordinates of firecounts.
    eps : Float
        The maximum distance between two firecounts
    minsamples : int
        The number of samples (or total weight) in a neighborhood for a point
    to be considered as a core point

    Returns
    -------
    labels : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
    n_clusters_ : Int
        Estimated number of clusters.
    n_noise_ : Int
        Estimated number of noise points.

    """
    # Cluster using density based clustering
    db = DBSCAN(eps=eps, min_samples=minsamples).fit(xy)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    # print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
    return labels, n_clusters_, n_noise_


def get_clusterdata(viirsdata):
    """
    Cluster VIIRS data based on DBSCAN. Add a label to the input data.

    Parameters
    ----------
    viirsdata : pandas Dataframe
        Raw VIIRS data.

    Returns
    -------
    viirsdata : pandas Dataframe
        VIIRS data with cluster labels column.
    firesources : pandas Dataframe
        Fire source is computed based on computed clusters.

    """
    # Compute DBSCAN lvl1
    xy = column_stack((viirsdata.xkms.values, viirsdata.ykms.values))
    c_labels, no_clusters, c_noise = get_clusters(xy, 4, 10)
    # viirsdata is level 0 data
    viirsdata["labels"] = c_labels
    # extract clustered data as a single point
    res = viirsdata[viirsdata.labels != -1]
    mn1 = res[["latitude", "longitude", "labels"]].groupby(["labels"]).mean()
    mn2 = res[["frp", "labels"]].groupby(["labels"]).sum()
    firesources = mn1.join(mn2)
    firesources.reset_index(inplace=True)
    firesources["firecounts"] = (res.labels.value_counts(sort=False)).values
    return viirsdata, firesources


def get_firedata(day, params):
    """
    Read and cluster fire data based on day.

    Parameters
    ----------
    day : datetime
        Date on which the data needs to read.
    params : Data Container
        Data container with initialized parameters.

    Returns
    -------
    flagdata : bool
        Flag about if fire data is present or not
    flagcluster : bool
        Flag on if cluster exists or not
    viirsdata : Dataframe
        Dataframe containing all the fire data from VIIRS.
    clusterdata : Dataframe
        Dataframe containing clustered data.

    """
    flagdata, viirsdata = read_firedata(params, day)
    if flagdata:
        viirsdata, firesources = get_clusterdata(viirsdata)
        if len(firesources) > 0:
            return flagdata, True, viirsdata, firesources
        else:
            return flagdata, False, viirsdata, []
    else:
        print("Data not there")
        return flagdata, False, [], []
