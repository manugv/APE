#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Identify orbits for the day based on location.

Created on Fri Jun  3 14:13:13 2022
@author: Manu Goudar
"""

import numpy as np
from pandas import DataFrame, read_csv
from datetime import date
from importlib.resources import files


def tropomi_mjd2orbit(mjd, fit_par=None):
    """
    Compute TROPOMI orbit number corresponding to a given time.

    where time is days from 2000-01-01.

    Parameters
    ----------
    mjd : Int/Float
        mjd 2000 value.
    fit_par : array, optional
        The default is None. Coefficients of a polynomial

    Returns
    -------
    Int
        Orbit number.
    """
    if fit_par is None:
        # 10.07.2018  Paul's formula (orbit range).
        fit_par = [227.0 / 16.0, -227.0 * 6495.385 / 16.0]

    func = np.poly1d(fit_par)
    return func(mjd)


def geoloc_in_orbit(lat0, lon0, orbit_nr):
    """
    Geolocation_in_orbit.

    The function tests if the geolocation lat0,lon0 is included in
    the measurements of the orbit with orbit_number.

    Output: True/False

    """
    table_orbits = files("APE.data").joinpath('orbits_loc_table.csv')
    df = read_csv(table_orbits)

    # find the index of the nearest latitude to lat0
    idx = np.abs(df.lat.values - lat0).argmin()

    # calculate the absolute lon range for the geolocation
    lon_range = np.array([df.min1[idx], df.max1[idx], df.min2[idx], df.max2[idx]]) + lon0

    # calculate the lon of the orbit at the equator
    orb_lon = 26.0 - 360 * (orbit_nr - 1) * 16.0 / 227.0

    # correct orb_lon to be within the following ranges
    if lat0 < 0:
        lon_a = lon0 - 270.0
        lon_b = lon0 + 90.0
    else:
        lon_a = lon0 - 90.0
        lon_b = lon0 + 270.0

    # in this case we need to add 360
    if orb_lon < lon_a:
        n = np.ceil(np.abs((lon_a - orb_lon) / 360.0))
        orb_lon = orb_lon + n * 360.0

    # in this case we need to substract 360
    if orb_lon > lon_b:
        n = np.ceil(np.abs((orb_lon - lon_b) / 360.0))
        orb_lon = orb_lon - n * 360.0

    result = ((orb_lon >= lon_range[0]) & (orb_lon <= lon_range[1])) | (
        (orb_lon >= lon_range[2]) & (orb_lon <= lon_range[3])
    )
    return result


def geolocation_orbitlist(lat0, lon0, start_orbit, stop_orbit):
    """
    Get an orbit list for a given location.

    Parameters
    ----------
    lat0 : float
        Latitude location.
    lon0 : float
        Latitude location.
    start_orbit : Integer
        Start orbit for the day.
    stop_orbit : Integer
        Last orbit for the day.

    Returns
    -------
    ndarray
        Orbit list.

    """
    cycle = [i for i in np.arange(227) + 1 if geoloc_in_orbit(lat0, lon0, i)]

    # construct the full orbit list
    result = np.array([], dtype=int)
    _start = np.int_(start_orbit / 227)
    _end = np.int_(stop_orbit / 227) + 1
    for i in range(_start, _end):
        result = np.append(result, np.array(cycle) + 227 * i)

    # reduce the orbit list to the specified range
    return result[(result >= start_orbit) & (result <= stop_orbit)]


def checkiforbitexists(satfiles, orbit):
    """Check if a orbit/orbits exist

    Parameters
    ----------
    satfiles : Pandas Dataframe
        Data frame containing Orbit values
    orbit : Int or Array(Int)
        Orbit numbers
    """
    allorb = np.array([], dtype=np.int_)
    if isinstance(orbit, np.ndarray):
        for orb in orbit:
            if orb in satfiles.orbit.values:
                allorb = np.append(allorb, orb)
    elif isinstance(orbit, np.int_):
        if orbit in satfiles.orbit.values:
            allorb = np.append(allorb, orbit)
    else:
        print("Orbit is not an integer", orbit)
    return allorb


def get_orbits_on_locations(st_date, sources, sat_files):
    """
    Get all orbits for the identified sources based on time.

    Parameters
    ----------
    st_date : date type
        Date on which the data needs to be read.
    sources : Numpy array (m,2)
        Array with 'm' sources of latitude and longitude.

    Returns
    -------
    _flg : Bool
        True is orbits are present for fire clusters.
    _grp : Pandas dataframe
        Contains the orbits where the fire source location is present.

    """
    # compute number of days from modified Julian date 2000.
    st_mjd = (st_date - date(2000, 1, 1)).days

    # get start and end orbit
    st_orb = tropomi_mjd2orbit(st_mjd)
    end_orb = tropomi_mjd2orbit(st_mjd + 1)

    l_orbs = np.array([], dtype=np.int_)
    lbl = np.array([], dtype=np.int_)

    # find the dimensions of the source arrray
    # single source
    sh = 1   # if sources.ndim == 1
    if sources.ndim > 1:   # multiple sources
        sh = sources.shape[0]

    for i in range(sh):
        # get all orbits is a certain range for a specified location
        if sources.ndim == 1:
            _orbits = geolocation_orbitlist(sources[0], sources[1], st_orb, end_orb)
        else:
            _orbits = geolocation_orbitlist(sources[i, 0], sources[i, 1], st_orb, end_orb)
        # check if these orbits exist
        _orbs = checkiforbitexists(sat_files, _orbits)

        if np.size(_orbs) != 0:
            # saves the orbits
            l_orbs = np.concatenate((l_orbs, _orbs))
            lbl = np.concatenate((lbl, np.ones(len(_orbs)) * i))

    # create a data frame and return
    _grp1 = DataFrame()
    _grp1["labels"] = lbl
    _grp1["orbits"] = l_orbs
    if len(_grp1) > 0:
        return True, _grp1
    else:
        return False, _grp1
