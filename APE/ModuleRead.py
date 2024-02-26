#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  16 11:59 2023.

@author: Manu Goudar
"""

import pandas as pd
import h5py
from datetime import datetime
from .ModuleDataContainers import DataContainer


class ReadData:
    def __init__(self, filename, _days):
        self.filename = filename + ".h5"
        self.keys = self.get_keysfordays(_days)
        self.satellitegrpname = "Satellite"
        self.viirsgrpname = "VIIRS"
        self.plumedetectgrpname = "PlumeDetection"
        self.massflux = "Massflux"
        self.divergence = "Divergence"

    def get_group(self, r_grp, grpname):
        if grpname in r_grp.keys():
            return r_grp[grpname]
        # else create the group
        else:
            return r_grp.create_group(grpname)

    def get_keysfordays(self, _days):
        _daylist = [_day.strftime("%Y%m%d") for _day in _days]
        fl = h5py.File(self.filename, "r")
        allkys = list(fl.keys())
        fl.close()
        _keys = []
        for ky in allkys:
            if ky.split("_")[0] in _daylist:
                _keys.append(ky)
        return _keys

    def satellite(self, _grpname):
        fl = h5py.File(self.filename, "r")
        srcgrp = self.get_group(fl, _grpname)
        _grp = self.get_group(srcgrp, self.satellitegrpname)
        data = DataContainer()
        for _ky in _grp.keys():
            if _ky == "measurement_time":
                da = str(_grp[_ky].asstr()[...])
                data.__setattr__(_ky, datetime.strptime(da, "%Y/%m/%d_%H:%M:%S"))
            elif "flag" in _ky:
                data.__setattr__(_ky, _grp[_ky][...])
            elif _ky in ["orbit", "orbit_ref_time"]:
                data.__setattr__(_ky, _grp[_ky][...])
            elif _ky == "orbit_filename":
                data.__setattr__(_ky, str(_grp[_ky].asstr()[...]))
            else:
                data.__setattr__(_ky, _grp[_ky][:])
        fl.close()
        return data

    def plume(self, _grpname):
        fl = h5py.File(self.filename, "r")
        srcgrp = self.get_group(fl, _grpname)
        _grp = self.get_group(srcgrp, self.plumedetectgrpname)
        data = DataContainer()
        for _ky in _grp.keys():
            if "flag" in _ky:
                data.__setattr__(_ky, _grp[_ky][...])
            else:
                data.__setattr__(_ky, _grp[_ky][:])
        fl.close()
        return data


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
