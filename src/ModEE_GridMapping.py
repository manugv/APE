#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:05:21 2020

@author: manu
"""
from shapely.geometry import Polygon
import numpy as np


class GridMapping:
    def __init__(self, co_lat, co_lon, t2c_lat, t2c_lon, t2c_z):
        """
        Define grid transformation from trajectory2concentration grid
        to CO data grid
        Has a container per grid block containing all ids and vol fraction
        of traj2conc grid
        """
        self.map = []
        self.vertical, self.ids = self.__get_vertical_ids(t2c_z)
        self.__create_surface_map(co_lat, co_lon, t2c_lat, t2c_lon)
        # Assumes that the tropomi has 1000mts vertical resolution
        self.tropomi_shape = tuple(np.asarray(co_lat.shape) - 1) + (self.vertical,)

    def __get_vertical_ids(self, z):
        nos = np.int_(np.ceil(z[-1] / 1000))
        if nos == 0:
            print("error")
        else:
            ids = np.zeros((nos), dtype=np.int_)

        if nos == 1:
            ids[0] = z.size
        else:
            # get number of points in vertical direction for tropomi
            for k in range(nos):
                # Find index for 1000mts multiples
                kk = np.where(z == 1000 * (k + 1))[0]
                if kk.size > 0:
                    ids[k] = kk[0] + 1
                else:
                    ids[k] = z.size
        return nos, ids

    def __create_polygon(self, x1, x2, y1, y2):
        return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def __get_all_polygons(self, bounds, lat, lon):
        """
        Get all t2c indices that encapsulate grid polygon
        Input:
            bounds is bounding region (minx, miny, maxx, maxy)
            lat, lon : trajectory to concentration lat and lon
        Output:
            return min and max indices
        """
        # check the bounds
        lat_len = lat.size
        lon_len = lon.size
        x1 = np.searchsorted(lat, bounds[0], side="right")
        # If point is between 0 and 1 then 0 should be index
        if x1 > 0:
            x1 -= 1
        x2 = np.searchsorted(lat, bounds[2], side="right")
        # If point is above max then -1 has to be
        # subtracted to get last index
        if x2 == lat_len:
            x2 -= 1
        y1 = np.searchsorted(lon, bounds[1], side="right")
        # If point is between 0 and 1 then 0 should be index
        if y1 > 0:
            y1 -= 1
        y2 = np.searchsorted(lon, bounds[3], side="right")
        # If point is above max then -1 has to be
        # subtracted to get last index
        if y2 == lon_len:
            y2 -= 1
        polys = []
        for i in range(x1, x2):
            for j in range(y1, y2):
                _poly = self.__create_polygon(lat[i], lat[i + 1], lon[j], lon[j + 1])
                polys.append([_poly, i, j])
        return polys

    def __create_surface_map(self, co_lat, co_lon, t2c_lat, t2c_lon):
        # Create a polygon from t2c grid full bounds
        # This is done to check if grid polygon intercescts or not
        _t2c_roi = self.__create_polygon(t2c_lat[0], t2c_lat[-1], t2c_lon[0], t2c_lon[-1])
        # shape of the co data latitude
        _sh = co_lat.shape
        for i in range(_sh[0] - 1):
            for j in range(_sh[1] - 1):
                # Create a polygon
                _co = self.__create_polygon(
                    co_lat[i, j], co_lat[i + 1, j + 1], co_lon[i, j], co_lon[i + 1, j + 1]
                )
                # Check if conc ROI intersects co data polygon
                if _co.intersects(_t2c_roi):
                    # If true get polygons
                    _polys = self.__get_all_polygons(_co.bounds, t2c_lat, t2c_lon)
                    # map polygons
                    area_fraction = []
                    ids = []
                    for poly in _polys:
                        area = _co.intersection(poly[0]).area / poly[0].area
                        if area > 0:
                            area_fraction.append(area)
                            ids.append([poly[1], poly[2]])
                    # dump all values into a container
                    self.map.append(
                        {"lat_id": i, "lon_id": j, "id_conc": ids, "area_frac": area_fraction}
                    )

    def conc2tropomi(self, conc):
        tropomi_conc = np.zeros(self.tropomi_shape)
        k1 = 0
        for k in range(self.vertical):
            # flatten the conc array
            f_conc = np.sum(conc[:, :, k1 : self.ids[k]], axis=2)
            k1 = self.ids[k]
            # Map the values
            for grid in self.map:
                i = grid["lat_id"]
                j = grid["lon_id"]
                for ij, ar_f in zip(grid["id_conc"], grid["area_frac"]):
                    tropomi_conc[i, j, k] += f_conc[ij[0], ij[1]] * ar_f
        return tropomi_conc
