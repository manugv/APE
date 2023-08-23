#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:24:23 2022.

@author: Manu Goudar
"""

from netCDF4 import Dataset  # type: ignore
import numpy as np
import pathlib
from pandas import DataFrame
from ModDataPrepare_Destriping import calc_stripe_mask
from ModuleDataContainers import DataContainer


# data that needs to be extracted [DO NOT CHANGE THE VARIABLE NAMES]
prefix = "PRODUCT/SUPPORT_DATA/"
name_list = {
    "time": "PRODUCT/time",
    "deltatime": "PRODUCT/delta_time",
    "lat": "PRODUCT/latitude",
    "lon": "PRODUCT/longitude",
    "lat_corners": prefix + "GEOLOCATIONS/latitude_bounds",
    "lon_corners": prefix + "GEOLOCATIONS/longitude_bounds",
    "surface_class": prefix + "INPUT_DATA/surface_classification",
    "qa_value": "PRODUCT/qa_value",
    "aerosol_thick": prefix + "DETAILED_RESULTS/height_scattering_layer",
    "avg_kernel": prefix + "DETAILED_RESULTS/column_averaging_kernel",
    "co_column": "PRODUCT/carbonmonoxide_total_column",
}

name_list_qa = {
    "z_cloud": prefix + "DETAILED_RESULTS/height_scattering_layer",
    "aerosol_thickness": prefix + "DETAILED_RESULTS/scattering_optical_thickness_SWIR",
    "solar_za": prefix + "GEOLOCATIONS/solar_zenith_angle",
}


def lat_lon_gap_correction(ar):
    """
    Correction to make adjacent points same.

    Taken from sron_toolbox.
    This is done by converting two adjacent points to a
    single point i.e., mid-point between two points.
    In flight direction, vertices (corners) of adjacent latitude and
    longitude pixels do not coincide. There is a small gap between them.
    so for internal points (4 pixel neighbours):
        array[i,:,3] != array[i+1,:,0] & array[i,:,2] != array[i+1,:,1]
    In Swath direction points are same so
    for internal points (4 pixel neighbours),
        array[:,i,2] = array[:,i+1,3] & array[:,i,1] = array[:,i+1,0]

    Parameters
    ----------
    ar : ndarray (m,n,4)
        Describes corner points of latitude or longitude.

    Returns
    -------
    ar_corr : ndarray of shape (m,n,4)
        Corrected latitude/longitude values.

    ar_nodes : ndarray of shape (m+1,n+1)
        Nodes of the pixels.

    """
    nos = ar.shape
    ar_corr = ar.copy()
    ar_nodes = np.zeros((nos[0] + 1, nos[1] + 1))
    # Border pixels in Flight direction
    # Bottom border
    ar_corr[1:, 0, 0] = ar_corr[:-1, 0, 3] = (ar[:-1, 0, 3] + ar[1:, 0, 0]) * 0.5
    # Top border
    ar_corr[:-1, -1, 2] = ar_corr[1:, -1, 1] = (ar[:-1, -1, 2] + ar[1:, -1, 1]) * 0.5
    # Corners with 4 neighbours
    ar_corr[:-1, :-1, 2] = (ar[:-1, :-1, 2] + ar[1:, :-1, 1]) * 0.5
    ar_corr[1:, :-1, 1] = ar_corr[:-1, :-1, 2]
    ar_corr[:-1, 1:, 3] = ar_corr[:-1, :-1, 2]
    ar_corr[1:, 1:, 0] = ar_corr[1:, :-1, 1]
    # transform corners to nodes
    ar_nodes[:-1, :-1] = ar_corr[:, :, 0]
    ar_nodes[-1, :-1] = ar_corr[-1, :, 3]
    ar_nodes[:-1, -1] = ar_corr[:, -1, 1]
    ar_nodes[-1, -1] = ar_corr[-1, -1, 2]
    return ar_corr, ar_nodes


def get_filenames(datadir):
    """
    Get all satellite files in a given directory.

    Parameters
    ----------
    datadir : str
        Directory containing satellite data.

    Returns
    -------
    Dataframe
        Dataframe containing filename, orbit number and version.

    """
    flist = []
    for p in pathlib.Path(datadir).rglob("*.nc"):
        if p.is_file():
            flist.append(str(p))
    fl_ = DataFrame({"filename": flist})
    ff_1 = fl_.filename.str.split("_", expand=True)
    fl_["orbit"] = np.int_(ff_1.iloc[:, -4].values)
    fl_["mod_date"] = ff_1.iloc[:, -1].values
    fl_["version"] = np.int_(ff_1.iloc[:, -2].values)
    fl_.sort_values(by=["orbit", "mod_date"], inplace=True)
    ff_ = fl_.drop_duplicates(subset=["orbit"], keep="last")
    ff_.reset_index(inplace=True)
    return ff_[["orbit", "filename", "version"]]


def recompute_qavalues(z_cloud, solar_za, aerosol_thickness):
    """
    Compute the qa values for orbits < 2818.

    Parameters
    ----------
    z_cloud : Array
        height_scattering_layer.
    solar_za : Array
        solar_zenith_angle.
    aerosol_thickness : Array
        scattering_optical_thickness_SWIR.

    Returns
    -------
    Array
        qa_value

    """
    _sh = z_cloud.shape
    qa = np.zeros(_sh)
    qa[
        (aerosol_thickness.data < 0.5)
        & (z_cloud.data < 500)
        & ~z_cloud.mask
        & ~aerosol_thickness.mask
    ] = 1
    qa[
        (aerosol_thickness.data >= 0.5)
        & (z_cloud.data < 5000)
        & ~z_cloud.mask
        & ~aerosol_thickness.mask
    ] = 0.7
    qa[
        ((aerosol_thickness.data >= 0.5) & (z_cloud.data >= 5000))
        | ((aerosol_thickness.data <= 0.5) & (z_cloud.data >= 500))
        & ~z_cloud.mask
        & ~aerosol_thickness.mask
    ] = 0.4
    qa[(solar_za > 80) | z_cloud.mask | aerosol_thickness.mask] = 0
    return np.ma.array(qa)


def get_qa_values(orbit, version, _val, f):
    """
    If needed, recompute qa values for old orbits and versions.

    Parameters
    ----------
    orbit : Int
        Orbit number.
    version : Int
        TROPOMI Processing version.
    _val : str
        field name in satellite file.
    f : File identifier
        File pointer to read data.

    Returns
    -------
    qa_val : array
        QA values.

    """
    _tmp = {}
    if (orbit <= 2818) & (version < 10202):
        for _kqa, _valqa in name_list_qa.items():
            _tmp[_kqa] = f[_valqa][0]
        qa_val = recompute_qavalues(_tmp["z_cloud"], _tmp["solar_za"], _tmp["aerosol_thickness"])
        del _tmp
    else:
        qa_val = f[_val][0]
    return qa_val


def get_co_column(orbit, version, _val, f, qa_value):
    """
    Read CO column data based on version and orbit number (destriping).

    Parameters
    ----------
    orbit : Int
        Orbit number.
    version : Int
        TROPOMI Processing version.
    _val : str
        field name in satellite file.
    f : File identifier
        File pointer to read data.
    qa_value: Array of bools
        Quality values.

    Returns
    -------
    codata : array
        CO column data.

    """
    # Destrip the data
    if (orbit < 19258) & (version < 20200):
        codata = f[_val][0]
        qa_idx = (qa_value <= 0) | (qa_value > 1) | codata.mask
        co1 = codata.data
        co = co1.copy()
        co[qa_idx] = np.nan
        ds_fft = calc_stripe_mask(co, "fft")
        co_fft = np.ma.MaskedArray(co1 - ds_fft, mask=qa_idx)
    else:
        codata = f[_val][0]
        co_fft = f[_val + "_corrected"][0]
    return codata, co_fft


def readsatellitedata(params, orbit):
    """
    Read TROPOMI Orbit data.

    Parameters
    ----------
    params : A dict of parameters containing following
        satellitedatadir : str
            Points to the satellite data directory
    orbit : Int
        Orbit number.

    Returns
    -------
    data : dictionary
        Contains data of the satellite for a given Orbit number.

    """
    # get file name and version based on orbit
    orbitinfo = params.sat_files[params.sat_files.orbit == orbit]
    version = orbitinfo.version.values[0]
    filename = orbitinfo.filename.values[0]
    # Read data
    data = DataContainer()
    data.__setattr__("filename", filename)
    data.__setattr__("orbit", orbit)
    f = Dataset(filename, "r")
    for _k, _val in name_list.items():
        # Recompute qa values
        if _k == "qa_value":
            data.__setattr__(_k, get_qa_values(orbit, version, _val, f))
        elif _k == "co_column":
            d1, d2 = get_co_column(orbit, version, _val, f, data.qa_value)
            data.__setattr__(_k, d1)
            data.__setattr__(_k + "_corr", d2)
        else:
            data.__setattr__(_k, f[_val][0])
    f.close()

    latcnr, latn = lat_lon_gap_correction(data.lat_corners[:, :, :])
    loncnr, lonn = lat_lon_gap_correction(data.lon_corners[:, :, :])
    data.__setattr__("lat_corners_e", latcnr)
    data.__setattr__("lon_corners_e", loncnr)
    data.__setattr__("lat_nodes", latn)
    data.__setattr__("lon_nodes", lonn)
    return data
