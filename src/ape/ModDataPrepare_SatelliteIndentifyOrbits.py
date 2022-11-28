#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Identify orbits for the day based on location.

Created on Fri Jun  3 14:13:13 2022
@author: Manu Goudar
"""

import numpy as np
from pandas import DataFrame, read_csv, merge
from datetime import date

flname = "orbits_loc_table.csv"
df = read_csv(flname)


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
    global df
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


def get_orbits_on_locations(st_date, firesources):
    """
    Get all orbits for the identified sources based on time.

    Parameters
    ----------
    st_date : date type
        Date on which the data needs to be read.
    firesources : pandas Dataframe
        Fire source is computed based on computed clusters.

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

    for i in range(len(firesources)):
        # get all orbits is a certain range for a specified location
        _orbits = geolocation_orbitlist(
            firesources.latitude.values[i], firesources.longitude.values[i], st_orb, end_orb
        )
        # saves the orbits
        l_orbs = np.concatenate((l_orbs, _orbits))
        lbl = np.concatenate((lbl, np.ones(len(_orbits)) * i))
    _grp1 = DataFrame()
    _grp1["labels"] = lbl
    _grp1["orbits"] = l_orbs
    _grp = merge(firesources, _grp1, on="labels")

    _grp.sort_values(
        by=["orbits", "frp"],
        ascending=[True, False],
        inplace=True,
        ignore_index=True,
    )
    if len(_grp) > 0:
        _grp["readflag"] = True
        _flg = True
        return _flg, _grp
    else:
        _flg = False
        return _flg, _grp
