#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  16 11:59 2023.

@author: Manu Goudar
"""

import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from .ModuleDataContainers import DataContainer

class ReadData:
    def __init__(self, filename, _days, onlyplumes=False):
        self.filename = filename + ".h5"
        self.satellitegrpname = "Satellite"
        self.viirsgrpname = "VIIRS"
        self.plumegrpname = "PlumeDetection"
        self.massflux = "Massflux"
        self.divergence = "Divergence"
        self.allkeys = self.get_keysfordays(_days)
        if onlyplumes:
            self.keys = self.detectedplumeskeys()
        else:
            self.keys = self.allkeys
        
    def get_group(self, r_grp, grpname):
        if grpname in r_grp.keys():
            return r_grp[grpname]
        else:
            print("Group does not exist")
            return []

    def get_keysfordays(self, _days):
        if isinstance(_days, list):
            _daylist = [_day.strftime("%Y%m%d") for _day in _days]
        else:
            _daylist = [_days.strftime("%Y%m%d")]
        fl = h5py.File(self.filename, "r")
        allkys = list(fl.keys())
        fl.close()
        _keys = []
        for ky in allkys:
            if ky.split("_")[0] in _daylist:
                _keys.append(ky)
        return _keys

    def detectedplumeskeys(self):
        new_keys = []
        fl = h5py.File(self.filename, "r")
        for _ky in self.allkeys:
            kygrp = self.get_group(fl, _ky)
            srcgrp = self.get_group(kygrp, self.plumegrpname)
            # check if flag exists
            if "flag_goodplume" in srcgrp.keys():
                # if exists then store the key
                if srcgrp["flag_goodplume"][()]:
                    new_keys.append(_ky)
            else:
                print("flag_detected variable is missing")
        return new_keys

    def satellite(self, _grpname):
        fl = h5py.File(self.filename, "r")
        srcgrp = self.get_group(fl, _grpname)
        _grp = self.get_group(srcgrp, self.satellitegrpname)
        data = DataContainer()
        for _ky in _grp.keys():
            # if time then convert it to datetime variable
            if _ky == "measurement_time":
                da = str(_grp[_ky].asstr()[()])
                data.__setattr__(_ky, datetime.strptime(da, "%Y/%m/%d_%H:%M:%S"))
            # String nees to be converted in h5py
            elif _ky == "orbit_filename":
                data.__setattr__(_ky, str(_grp[_ky].asstr()[()]))
            else:
                data.__setattr__(_ky, _grp[_ky][()])
        fl.close()
        return data

    def plume(self, _grpname):
        fl = h5py.File(self.filename, "r")
        srcgrp = self.get_group(fl, _grpname)
        _grp = self.get_group(srcgrp, self.plumegrpname)
        data = DataContainer()
        for _ky in _grp.keys():
            data.__setattr__(_ky, _grp[_ky][()])
        fl.close()
        return data

    def viirs(self, _grpname):
        fl = h5py.File(self.filename, "r")
        srcgrp = self.get_group(fl, _grpname)
        _grp = self.get_group(srcgrp, self.viirsgrpname)
        data = DataContainer()
        for _ky in _grp.keys():
            data.__setattr__(_ky, _grp[_ky][()])
        fl.close()
        return data

    def getgroupdata(self, _grpname):
        """Satellite, plume and VIIRS
        Parameters
        ----------
        _grpname : String
            Key in the daya
        """
        satellite = self.satellite(_grpname)
        plume = self.plume(_grpname)
        viirs = self.viirs(_grpname)
        return satellite, viirs, plume
        
    def datafordownloadml(self):
        points = []
        fires_id = []
        fires_time = []
        f = h5py.File(self.filename, "r")
        for key in self.keys:
            da = str(f[key][self.satellitegrpname]["measurement_time"].asstr()[()])
            cluster_time = datetime.strptime(da, "%Y/%m/%d_%H:%M:%S")

            _source = f[key][self.satellitegrpname]["source"]
            file_points = np.column_stack((_source[0], _source[1]))

            points.append(file_points)
            fires_id.append(key)
            fires_time.append(cluster_time)
        if len(points) > 1:
            points1 = np.concatenate(points, axis=0)
        else:
            points1 = points
        f.close
        return points1, np.array(fires_id), np.array(fires_time)
