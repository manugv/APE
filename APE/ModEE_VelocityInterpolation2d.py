#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:07:49 2022.

@author: Manu Goudar

usage:
tmpvel = VelocityInterpolation("/nfs/METEO/ERA5")
tmpvel.computefunction(measurementtime)
tmpvel.interpolatevel(lat, lon)
tmpvel.v_vel
"""

import xarray as xr
import numpy as np
# from datetime import datetime, timedelta


class VelocityInterpolation:
    """
    Interpolations is set up such that the longitude goes from -180 to 180
    """

    def __init__(self, dirprefix, height=100):
        if height == 10:
            self.u_name = "u10"
            self.v_name = "v10"
        elif height == 100:
            self.u_name = "u100"
            self.v_name = "v100"
        self.dirprefix = dirprefix

    def computefunction(self, measuretime):
        self.measuretime = measuretime
        # filename
        filename = (self.dirprefix + self.measuretime.strftime("%Y") + "/sl_"
                    + self.measuretime.strftime("%Y%m%d") + ".grib")
        # read grib data
        grbs = xr.open_dataset(filename, engine="cfgrib", backend_kwargs={'indexpath': ''})
        # get velocity and interpolate at time t
        self.u_vel = ((grbs[self.u_name]).interp(time=self.measure_time)).data
        self.v_vel = ((grbs[self.v_name]).interp(time=self.measure_time)).data

    def convert_longitude(self, _ln):
        """
        Data is from -180 to 180
        """
        if _ln < 0:
            _ln = _ln + 360
        return _ln

    def interpolate_vel(self, lat, lon):
        """
        Interpolation for scalar, 1d array and 2d array
        Data should be given as:
        latitude: -90 to 90
        longitude: -180 to 180: is converted to 0-360
        """
        if np.isscalar(lat):
            _lon = self.convert_longitude(self, lon)
            return self.u_vel.interp(latitude=lat, longitude=_lon), self.v_vel.interp(latitude=lat, longitude=_lon)
        # one dimensional array
        elif lat.ndim == 1:
            # Get longitude individually
            _lon = np.zeros_like(lat)
            for i in range(lat.shape[0]):
                _lon[i] = self.convert_longitude(lon[i])
            return self.u_vel.interp(latitude=lat, longitude=_lon), self.v_vel.interp(latitude=lat, longitude=_lon)
        # two dimensional array
        elif lat.ndim == 2:
            # Get longitude individually
            _lon = np.zeros_like(lat)
            sh = lat.shape
            for i in range(sh[0]):
                for j in range(sh[1]):
                    _lon[i, j] = self.longitude_in_radians(lon[i, j])
            u = (self.u_vel.interp(latitude=lat.ravel(), longitude=_lon.ravel())).reshape(sh)
            v = (self.v_vel.interp(latitude=lat.ravel(), longitude=_lon.ravel())).reshape(sh)
            return u, v
        else:
            print("something went wrong")
