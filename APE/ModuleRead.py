#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  16 11:59 2023.

@author: Manu Goudar
"""

import pandas as pd
import h5py
from .ModDataPrepare_SatelliteDataFiltering import get_orbit_time


viirs_data_keys = ["latitude", "longitude"]
satellite_data_keys = ["lat", "lon", "co_column_corr",
                       "lat_nodes", "lon_nodes"]


class DataCont:
    pass


def get_group(r_grp, grpname):
    if grpname in r_grp.keys():
        return r_grp[grpname]
    # else create the group
    else:
        return r_grp.create_group(grpname)


def read_viirsdata(viirs_grp):
    data = {}
    for ky in viirs_data_keys:
        data[ky] = viirs_grp[ky][:]
    return pd.DataFrame.from_dict(data)


def read_satellitedata(grp):
    data = DataCont()
    data.__setattr__("source", grp.attrs['source'])
    measuretime = get_orbit_time(grp.attrs['orbit_ref_time'],
                                 (grp['deltatime'][:]).mean())
    data.__setattr__("measurement_time", measuretime)
    data.__setattr__("fire_name", grp.attrs["fire_name"])
    for ky in satellite_data_keys:
        data.__setattr__(ky, grp[ky][:])
    return data


def read_plumedata(grp):
    data = DataCont()
    data.__setattr__("plumemask", grp["plumemask"][:])
    return data


def read_data(filename, firekey):
    # open file
    fl = h5py.File(filename, "r")
    fire_grp = get_group(fl, firekey)
    # VIIRS data
    viirs_grp = get_group(fire_grp, "VIIRS")
    viirscont = read_viirsdata(viirs_grp)
    # Satellite container
    satellite_grp = get_group(fire_grp, "Satellite")
    satdata = read_satellitedata(satellite_grp)
    # plume container
    plume_grp = get_group(fire_grp, "PlumeDetection")
    plumecont = read_plumedata(plume_grp)
    return satdata, viirscont, plumecont


def get_detectedplumekeys(file):
    f = h5py.File(file, 'r')
    keys_list = []
    for key in f.keys():
        if (f[key]["PlumeDetection"].attrs['f_firearoundplume']):
            keys_list.append(key)
    f.close
    return keys_list
