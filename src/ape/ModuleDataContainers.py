#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:07:00 2022.

@author: Manu Goudar
"""

import numpy as np
from datetime import timedelta
import pandas as pd


def get_time_in_sec(_key):
    hr = 0
    if "Hour" in _key.keys():
        hr = _key["Hour"]
    mn = 0
    if "Minute" in _key.keys():
        mn = _key["Minute"]
    sec = 0
    if "Second" in _key.keys():
        sec = _key["Second"]
    return hr * 3600 + mn * 60 + sec


class DataContainer:
    """Data container for different data."""

    pass


class ROI:
    def __init__(self):
        self.lat = [-90, 90]
        self.lon = [0, 360]
        self.z = [0, 10000]

    def update_roi(self, lat, lon, z=[0, 10000]):
        self.lat = lat
        self.lon = lon
        self.z = z


class Flowinfo:
    def __init__(self, _key):
        self.inputdir = _key["era5_dir"]
        self.file_flow = _key["FlowField"]
        self.file_pres = _key["SurfacePresGeop"]
        self.modellevels = _key["ModelLevels"]
        self.origin = np.zeros((2))

    def update_origin(self, lat, lon):
        self.origin[0] = lat
        self.origin[1] = lon


class SimulationTime:
    def __init__(self, _key):
        self.sim_time_sec = get_time_in_sec(_key["TotalTime"])
        self.sim_time_hrs = np.int_(self.sim_time_sec / 3600)
        self.dt = get_time_in_sec(_key["TimeStep"])
        self.savedt = get_time_in_sec(_key["SaveDataTimeStep"])

    def set_start_time(self, plume_time):
        return plume_time - timedelta(seconds=self.sim_time_sec)


class Dispersion:
    def __init__(self, _key):
        self.simtype = _key["Type"]
        self.model = _key["Model"]
        # self.turbulence = _key['Turbulence']
        self.method = _key["Method"]


class MultipleParticleRelease:
    def __init__(self, _key, endsec=0):
        self.flag = _key["Flag"]
        if self.flag is True:
            if endsec == 0:
                self.endtime_sec = get_time_in_sec(_key["EndTime"])
            else:
                self.endtime_sec = endsec - 60
            self.deltatime = get_time_in_sec(_key["TimeStep"])
            tt = 0
            self.no_releases = 0
            while tt < self.endtime_sec:
                self.no_releases += 1
                tt += self.deltatime
        else:
            self.deltatime = 0
            self.no_releases = 1


class ParticleSplitting:
    def __init__(self, _key):
        self.split = _key["Split"]
        if self.split is True:
            self.splittime = get_time_in_sec(_key["SplitTime"])


class Traj2ConcInfo:
    def __init__(self, _key):
        self.flag = _key["Convert"]
        if self.flag is True:
            self.starttime = get_time_in_sec(_key["StartSaveTime"])
            self.avgtime = get_time_in_sec(_key["AveragingTime"])
            self.deltatime = get_time_in_sec(_key["TimeStep"])
            self.roi_resolution = np.asarray(_key["GridResolution"])
