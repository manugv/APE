#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:48:46 2022.

Filter and extract the satellite data

@author: Manu Goudar
"""
import numpy as np
from scipy.spatial import cKDTree
from .ModuleDataContainers import DataContainer
from datetime import timedelta, datetime   # , timezone


# Values to extract for each orbit DO NOT CHANGE THE SEQUENCE
val_ext = [
    "lat",
    "lon",
    "qa_value",
    "co_column",
    "co_column_corr",
    "lat_corners",
    "lon_corners",
    "lat_nodes",
    "lon_nodes",
    "aerosol_thick",
    "avg_kernel",
    "deltatime",
]
val_flag = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2]


def get_orbit_time(time, time_scanline):
    """
    Gets the time of measurement of plume for TROPOMI satellite.

    Args:
        time (int): Represents the reference time of the measurements in seconds
        dayinsec (int): Offset from reference start time of measurement in ‘milliseconds’

    Returns:
        measurement time (datetime): Date and time of measurement at the source.
    """
    return (
        datetime(2010, 1, 1, 0, 0, 0)  # , tzinfo=timezone.utc)
        + timedelta(seconds=time * 1.0)
        + timedelta(seconds=time_scanline / 1000.0)
    )


def get_nearest_srcpixel(lat1, lon1, src):
    """
    Find the nearest pixel where the source is located.

    Algorithm uses kdtree to find the nearest pixel.

    Parameters
    ----------
    lat1 : Array float
        Pixel centers latitude.
    lon1 : Array of float
        Pixel centers longitude.
    src : array of size 2
        Source location.

    Returns
    -------
    x : Array of size 2 Integer.
        Index of the grid.
    flag : Bool
        False : if distance between source and identified point > 1deg
        False : if identified location is at the edge.

    """
    flag = True
    # Find the nearest point to the given source (KDTree)
    cc = np.c_[lat1.ravel(), lon1.ravel()]
    tree = cKDTree(cc)
    dis, loc = tree.query(src)
    # Find the index based on the location
    sh = lat1.shape
    x = [loc // sh[1], loc % sh[1]]
    if any([dis > 1, np.abs(sh[0] - x[0]) < 3, np.abs(sh[1] - x[1]) < 3, x[0] < 3, x[1] < 3]):
        flag = False
    return x, flag


def point_in_polygon(yi, xi, py, px):
    """
    Find if a point is inside a polygon or not.

    This is based on convex hull where points of a polygon are given
    counter clockwise starting from left-bottom point.

    Parameters
    ----------
    yi : Float
        Value of y (lat) location of the point.
    xi : Float
        Value of x (lon) location of the point.
    py : Array of size n
        y (lat) coordinates of the polygon of n sides.
    px : Array of size n
        x (lon) coordinates of the polygon of n sides.

    Returns
    -------
    Bool
        True if the particle is inside quadrilateral.
    """
    # get vertices of the grid
    # find if points are inside or on the line. Convex hull idea
    r0 = (yi - py[0]) * (px[1] - px[0]) - (xi - px[0]) * (py[1] - py[0])
    r1 = (yi - py[1]) * (px[2] - px[1]) - (xi - px[1]) * (py[2] - py[1])
    r2 = (yi - py[2]) * (px[3] - px[2]) - (xi - px[2]) * (py[3] - py[2])
    r3 = (yi - py[3]) * (px[0] - px[3]) - (xi - px[3]) * (py[0] - py[3])
    return (r0 >= 0) & (r1 >= 0) & (r2 >= 0) & (r3 >= 0)


def get_source_pixel(data, src):
    """
    Find the pixel where source is located.

    Parameters
    ----------
    data : Data container
        Contains satellite data for an orbit.
    src : Array of size (2) Float
        Location of the fire source.

    Returns
    -------
    flag : Bool
        If the point is in the oribt or not.
    x : Array of size 2. Float
        Pixel containing the fire source.

    """
    # First get the nearest pixel
    px, flag = get_nearest_srcpixel(data.lat.data, data.lon.data, src)

    # If inside the orbit then get the exact location
    if flag:
        _flag = False
        num = [0, 1, 2, -1, -2]
        for i in num:
            for j in num:
                x = px.copy()
                x[0] += j
                x[1] += i
                _lat = data.lat_corners_e[x[0], x[1]]
                _lon = data.lon_corners_e[x[0], x[1]]
                _flag = point_in_polygon(src[0], src[1], _lat, _lon)
                # this means the box containig source point is found
                # and thus exit the function
                if _flag:
                    return flag, _flag, x
        # IF the location has not been found in sorrounding pixels
        return flag, _flag, px
    else:
        return flag, False, px


def filter_grid_size(x_pixel):
    """
    Filter based on grid size.

    Check if the point lines in acceptable grid size, that is, the source
    point has to lie between 45 and 171 pixel in swatch direction.

    Parameters
    ----------
    x_pixel : Integer
        Pixel location in swatch direction.

    Returns
    -------
    _flag : Bool
        If it is withing given grid locations.

    """
    if 45 <= x_pixel < 171:
        return True
    else:
        return False


def filter_good_data(co_mask, qa_value):
    """
    Filter based on quality of CO data.

    qa values and CO mask is used to filter. Quality inside center 7x7 pixel

    Parameters
    ----------
    co_mask : Bool (m,m)
        CO column masked values.
    qa_value : Bool (m,m)
        Quality values given in Tropomi.

    Returns
    -------
    bool
        True if good data.

    """
    # mask further bad values
    co_mask[qa_value < 0.5] = True
    # Flip the mask, so good values
    _co = ~co_mask
    nx = np.int_(_co.shape[0] / 2)
    _inner = _co[nx - 3: nx + 4, nx - 3: nx + 4]
    # QA and other quality check
    if (_inner.sum() / _inner.size > 0.85) & (_co.sum() / _co.size > 0.8):
        print("    Good Data:-  ", _inner.sum() / _inner.size, _co.sum() / _co.size)
        return True, co_mask
    else:
        return False, []


def extract_data(x, data, dta, nos=20):
    """
    Check for nan and extract data.

    Parameters
    ----------
    x : Integer [2]
        Pixel location containing source.
    data : data Container.
        Container consisting of Orbit data.
    dta : data Container.
        Container for extracting a granule from orbit data.
    nos : Integer, optional
        Pixels to be extracted around center. The default is 20.

    Returns
    -------
    flag : Bool
        False, if all CO values are NAN values.
    dta : data Container
        Data of extracted satellite values.

    """
    # Create max-min pixels to extract a block
    i1 = x[0] - nos
    i2 = x[0] + nos + 2
    j1 = x[1] - nos
    j2 = x[1] + nos + 2
    # Check if all CO values are nans
    co = data.co_column[i1: i2 - 1, j1: j2 - 1].filled(fill_value=np.nan)
    flag = np.logical_not(np.isnan(co).sum() == co.size)
    dta.__setattr__("f_not_all_nans", flag)
    # Get data
    if dta.f_not_all_nans:
        dta.__setattr__("source_pixel_id", x)
        dta.__setattr__("orbit_ref_time", data.time)
        for i, fl in zip(val_ext, val_flag):
            if fl == 0:
                _tmp = data.__getattribute__(i)[i1: i2 - 1, j1: j2 - 1]
                dta.__setattr__(i, _tmp)
            elif fl == 2:
                _tmp1 = data.__getattribute__(i)[i1:i2].data
                dta.__setattr__(i, _tmp1)
                dta.__setattr__(
                    "measurement_time", get_orbit_time(dta.orbit_ref_time, dta.deltatime.mean())
                )
            else:
                _tmp2 = data.__getattribute__(i)[i1:i2, j1:j2]
                dta.__setattr__(i, _tmp2)
        # print(dta.co_column_corr.mask, dta.qa_value)
        fflag, _com = filter_good_data(dta.co_column_corr.mask, dta.qa_value)
        dta.__setattr__("f_qavalue_filter", fflag)
        if dta.f_qavalue_filter:
            dta.__setattr__("co_qa_mask", _com)
            dta.__setattr__("f_good_satellite_data", True)
            dta.__setattr__("orbit", data.orbit)
            dta.__setattr__("orbit_filename", data.filename)
        else:
            print("  Quality of data fails")
            dta.__setattr__("f_good_satellite_data", False)
    else:
        print("  Data is just NANs")
        dta.__setattr__("f_good_satellite_data", False)
    return dta


def extract_and_filter_satellitedata(data, src):
    """Filter and extract data.

    Parameters
    ----------
    data : Data container
        Satellite data container.
    src : Array Float [2]
        Source location of fire or industry.

    Returns
    -------
    flag : bool
        Good data or not.
    dat1 : Data container [dict]
        Data containing extracted data.
    """
    extracted_firedata = DataContainer()
    # Get pixel containing the fire source
    inorb_flag, loc_flag, pixel_loc = get_source_pixel(data, src)
    extracted_firedata.__setattr__("source", src)
    extracted_firedata.__setattr__("f_source_in_orbit", loc_flag)
    # extracted_firedata.__setattr__("f_source_pixel_loc", )
    # Filter based on grid size
    if extracted_firedata.f_source_in_orbit:
        extracted_firedata.__setattr__("f_filter_gridsize", filter_grid_size(pixel_loc[1]))
        # Extract the data
        if extracted_firedata.f_filter_gridsize:
            extracted_firedata = extract_data(pixel_loc, data, extracted_firedata)
            return extracted_firedata
        else:
            print("  Grid sizes are large")
            extracted_firedata.__setattr__("f_good_satellite_data", False)
            return extracted_firedata
    else:
        print("  Source not in the orbit")
        extracted_firedata.__setattr__("f_good_satellite_data", False)
        return extracted_firedata
