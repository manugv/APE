#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:44:57 2022.

@author: Manu Goudar
"""

from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import cdsapi


class InjectionHeight:
    """
    Compute injection height based on on bi-linear interpolation.

    """

    def __init__(self, datadir, day):
        """Short summary.

        Parameters
        ----------
        filename : string
            Description of parameter `filename`.
        day : integer
            Description of parameter `day`.

        Returns
        -------
        type
            Description of returned object.

        """
        ma = date(1900, 1, 1)
        filename = self._getfilename(datadir, day)
        ff = Dataset(filename, "r")  # data/2020_09_gfas_data.nc
        lattmp = ff["latitude"][:].data
        lontmp = ff["longitude"][:].data

        time = ff["time"][:]
        alldays = np.array([(ma + timedelta(hours=tm * 1.0)) for tm in time])
        _tp = np.where(alldays == day)[0]
        if len(_tp) > 0:
            idx = _tp[0]
            injh = ff["injh"][idx].data
            ff.close()
        else:
            ff.close()
            print("injection height doesn't exist for this day")
            # TODO Change this and create a warning or error
            injh = np.zeros((lattmp.size, lontmp.size))
        self.lat_deg = np.flip(lattmp)
        self.lon_deg = lontmp
        self.injh = np.flip(injh, axis=0)
        self.f_ht = RegularGridInterpolator((self.lat_deg, self.lon_deg), self.injh, method="nearest")

    def _getfilename(self, datadir, day):
        _date = day.strftime("%Y_%m_%d")
        _file = datadir + _date + ".nc"
        if Path(_file).exists():
            return _file
        _file1 = datadir + day.strftime("%Y_%m") + ".nc"
        if Path(_file1).exists():
            return _file1

    def correct_longitude(self, longitude):
        """Convert longitude to 0-360 range.

        Parameters
        ----------
        longitude : float
            Longitude of -180 to 180 range.

        Returns
        -------
        float
            longitude in 0-360 range.
        """
        if longitude < 0:
            return longitude + 360
        else:
            return longitude

    def interpolate(self, lat, lon):
        """
        Interpolation for scalar, 1d array and 2d array
        Data should be given as:
        latitude: -90 to 90
        longitude: -180 to 180
        """
        if np.isscalar(lat):
            _lt = lat
            _ln = self.correct_longitude(lon)
            return self.f_ht((_lt, _ln))
        # one dimensional array
        elif lat.ndim == 1:
            # Get longitude individually
            _lt = lat
            _ln = np.zeros_like(lat)
            for i in range(lat.shape[0]):
                _ln[i] = self.correct_longitude(lon[i])
            return self.f_ht((_lt, _ln))
        # two dimensional array
        elif lat.ndim == 2:
            # Get longitude individually
            _ln = np.zeros_like(lat)
            sh = lat.shape
            for i in range(sh[0]):
                for j in range(sh[1]):
                    _ln[i, j] = self.correct_longitude(lon[i, j])
            _lt = lat
            u = self.f_ht((_lt.ravel(), _ln.ravel())).reshape(sh)
            return u
        else:
            print("something went wrong")


def download_injectionheight(outputdir, _day, url0, key0):
    """Download injection height.

    Parameters
    ----------
    outputdir : Str
        Output injection height data
    _day : Datetime
        Date from datetime
    url0 : str
        ADS API url https://ads.atmosphere.copernicus.eu/api-how-to
    key0 : str
        ADS key https://ads.atmosphere.copernicus.eu/api-how-to

    """
    _date = _day.strftime("%Y_%m_%d")
    _file = outputdir + _date + ".nc"
    _file1 = outputdir + _day.strftime("%Y_%m") + ".nc"
    fl = Path(_file)
    fl1 = Path(_file1)

    if fl.exists() or fl1.exists():
        print(" Injection height data exists")
    else:
        print("   Downloading Injection height data")
        c = cdsapi.Client(url=url0, key=key0)
        c.retrieve(
            "cams-global-fire-emissions-gfas",
            {
                "date": _date,
                "format": "netcdf",
                "variable": [
                    "altitude_of_plume_bottom",
                    "altitude_of_plume_top",
                    "injection_height",
                    "mean_altitude_of_maximum_injection",
                ],
            },
            _file,
        )
        print("                    Downloaded")
