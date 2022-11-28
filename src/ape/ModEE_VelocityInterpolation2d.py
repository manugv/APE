#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  12 13:07:49 2022.

@author: Manu Goudar
"""

import pygrib
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline


class VelocityInterpolation:
    """
    Interpolations is set up such that the longitude goes from -180 to 180
    """

    def __init__(self, t_utc, z="100"):
        self.t_utc = t_utc
        self.deg_to_rad = np.pi / 180
        # compute ratios
        mjd0 = (self.t_utc.minute * 60 + self.t_utc.second) / 3600
        mjd1 = 1 - mjd0
        path = "/METEO/ERA5/%s/sl_%s%s%s.grib" % (
            str(self.t_utc.year),
            self.t_utc.year,
            str(self.t_utc.month).zfill(2),
            str(self.t_utc.day).zfill(2),
        )

        grbs = pygrib.open(path)
        if z == "10":
            grbs_u10 = grbs.select(
                name="10 metre U wind component", hour={t_utc.hour, t_utc.hour + 1}
            )
            grbs_v10 = grbs.select(
                name="10 metre V wind component", hour={t_utc.hour, t_utc.hour + 1}
            )
        elif z == "100":
            grbs_u10 = grbs.select(
                name="100 metre U wind component", hour={t_utc.hour, t_utc.hour + 1}
            )
            grbs_v10 = grbs.select(
                name="100 metre V wind component", hour={t_utc.hour, t_utc.hour + 1}
            )

        lattmp, lontmp = grbs_u10[0].latlons()
        self.lat_deg = np.flip(lattmp[:, 0])
        self.lat_rad = (self.lat_deg + 90) * self.deg_to_rad
        self.lon_deg = lontmp[0, :]
        self.lon_rad = self.lon_deg * self.deg_to_rad

        U10_1 = grbs_u10[0].values
        U10_2 = grbs_u10[1].values
        V10_1 = grbs_v10[0].values
        V10_2 = grbs_v10[1].values
        # interpolate the values to a given time
        testU = U10_1 * mjd1 + U10_2 * mjd0
        testV = V10_1 * mjd1 + V10_2 * mjd0
        self.u_vel = np.flip(testU, axis=0)
        self.v_vel = np.flip(testV, axis=0)
        self.fu = RectSphereBivariateSpline(self.lat_rad[1:-1], self.lon_rad, self.u_vel[1:-1, :])
        self.fv = RectSphereBivariateSpline(self.lat_rad[1:-1], self.lon_rad, self.v_vel[1:-1, :])

    def latitude_in_radians(self, _lt):
        """
        Data is from -90 to 90
        """
        return (_lt + 90) * self.deg_to_rad

    def longitude_in_radians(self, _ln):
        """
        Data is from -180 to 180
        """
        if _ln < 0:
            _ln = _ln + 360
        return _ln * self.deg_to_rad

    def interpolate_vel(self, lat, lon):
        """
        Interpolation for scalar, 1d array and 2d array
        Data should be given as:
        latitude: -90 to 90
        longitude: -180 to 180
        """
        if np.isscalar(lat):
            _lt_rd = self.latitude_in_radians(lat)
            _ln_rd = self.longitude_in_radians(lon)
            return self.fu.ev(_lt_rd, _ln_rd), self.fv.ev(_lt_rd, _ln_rd)
        # one dimensional array
        elif lat.ndim == 1:
            # Get longitude individually
            _ln_rd = np.zeros_like(lat)
            for i in range(lat.shape[0]):
                _ln_rd[i] = self.longitude_in_radians(lon[i])
            _lt_rd = self.latitude_in_radians(lat)
            return self.fu.ev(_lt_rd, _ln_rd), self.fv.ev(_lt_rd, _ln_rd)
        # two dimensional array
        elif lat.ndim == 2:
            # Get longitude individually
            _ln_rd = np.zeros_like(lat)
            sh = lat.shape
            for i in range(sh[0]):
                for j in range(sh[1]):
                    _ln_rd[i, j] = self.longitude_in_radians(lon[i, j])
            _lt_rd = self.latitude_in_radians(lat)
            u = self.fu.ev(_lt_rd.ravel(), _ln_rd.ravel()).reshape(sh)
            v = self.fv.ev(_lt_rd.ravel(), _ln_rd.ravel()).reshape(sh)
            return u, v
        else:
            print("something went wrong")
