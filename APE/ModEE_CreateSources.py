#!/usr/bin/env python
# coding: utf-8
"""
Created on July 3 13:00 2020
Parameter Containers for different variables
@author: Manu Goudar
"""
import numpy as np
from pandas import read_csv, DataFrame, concat


# TODO : to be removed
class SourcesInit:
    def __init__(self, _keymain, _grp_src=[]):
        self.nos = 0
        self.locs = 0
        self.mass = 0
        self.ids = 0
        self.topo_ht = 0

        if _keymain["TopologyZ"] == "Relative":
            self.topo_flag = True
        else:
            self.topo_flag = False

        # Sources from HDF File
        if _keymain["Specification"] == "H5":
            # read data and dump it in terms of Dataframe
            self.nos, self.locs, self.mass, self.ids = self.__get_sources_from_hdf(
                _grp_src, _keymain["Height"]
            )

        # Sources from File
        if _keymain["Specification"] == "File":
            self.nos, self.locs, self.mass, self.ids = self.__get_sources_file(_keymain["File"])

        # On the fly
        if _keymain["Specification"] == "Input":
            pass

        # For Manual data
        if _keymain["Specification"] == "Manual":
            _key = _keymain["Manual"]
            self.no_seeds = _key["NumberOfSeeds"]
            for i in range(self.no_seeds):
                _no = _key["Seed" + str(i + 1)]["Nos"]
                self.nos += _no[0] * _no[1] * _no[2]
            self.locs = np.zeros((self.nos, 3))
            self.mass = np.zeros((self.nos))
            self.ids = np.zeros((self.nos, 2), dtype=np.int32)
            no_i = 0
            for i in range(self.no_seeds):
                no_i = self.__get_sources_manual(_key["Seed" + str(i + 1)], no_i, i)

    def __get_sources_file(self, filekey):
        flname = filekey["Filename"]
        input_param = filekey["Input"]
        cc = read_csv(flname)
        lat = cc.latitude.values
        lon = cc.longitude.values
        if "z" in input_param:
            z = cc.z.values
            dim_z = 1  # Dimensions in z direction
        else:
            pts = filekey["Dir_Z"]["Points"]
            dim_z = filekey["Dir_Z"]["Nos"]
            z = np.linspace(pts[0], pts[1], dim_z)
            # Mass definitions
        if "mass" in input_param:
            tmp_mass = cc.mass.values
        else:
            if filekey["Mass"]["Type"] == "Same":
                tmp_mass = filekey["Mass"]["Value"]
            else:
                tmp_mass = 1
        # Define sources
        nos_lat = lat.size
        nos = nos_lat * dim_z
        locs = np.zeros((nos, 3))
        ids = np.zeros((nos, 2))
        mass = np.zeros((nos))
        src_ids = np.arange(nos_lat)
        for i in range(dim_z):
            k1 = i * nos_lat
            k2 = k1 + nos_lat
            locs[k1:k2, 0] = lat.copy()
            locs[k1:k2, 1] = lon.copy()
            locs[k1:k2, 2] = z[i]
            ids[k1:k2, 0] = src_ids  # Source ids
            ids[k1:k2, 1] = i  # vertical ids
            mass[k1:k2] = tmp_mass
        return nos, locs, mass, ids

    # TODO : For sphere and line and square/rectangle
    def __get_sources_manual(self, __key, i1, src_no):
        if __key["Type"] == "Box":
            pt1 = np.asarray(__key["Point1"])
            pt2 = np.asarray(__key["Point2"])
            _no = np.asarray(__key["Nos"])
            xx = np.linspace(pt1[0], pt2[0], _no[0])
            yy = np.linspace(pt1[1], pt2[1], _no[1])
            zz = np.linspace(pt1[2], pt2[2], _no[2])
            ll = i1
            kk = zz.size
            vids = np.arange(kk, dtype=np.int32)
            for i in range(xx.size):
                for j in range(yy.size):
                    self.locs[ll: ll + kk, 0] = xx[i]
                    self.locs[ll: ll + kk, 1] = yy[j]
                    self.locs[ll: ll + kk, 2] = zz[:]
                    self.ids[ll: ll + kk, 1] = vids
                    self.ids[ll: ll + kk, 0] = src_no
                    ll += kk
            nums = _no[0] * _no[1] * _no[2]
            i2 = i1 + nums
            self.mass[i1:i2] = self.__get_mass(__key["Mass"], nums)
            return i2

    # TODO different values
    def __get_mass(self, _key, nums):
        mass = np.zeros((nums))
        if _key["Type"] == "Same":
            mass[:] = np.int_(_key["Value"])
        return mass

    def get_actual_loc(self):
        _loc = self.locs.copy()
        _loc[:, 2] += self.topo_ht
        return _loc

    def get_data(self):
        """
        Takes time in seconds as input and output is source, mass and ids
        """
        if self.topo_flag:
            loc = self.get_actual_loc()
        else:
            loc = self.locs.copy()
        return loc, self.mass.copy(), self.ids.copy()

    def _create_srcs_vert(self, dd, k):
        _dist = 500
        dim_z = 21
        dx = 50
        if (dd.z - _dist) > 0:
            z = np.linspace(dd.z - _dist, dd.z + _dist, dim_z)
        else:
            dim_z1 = np.int_((dd.z - dx) / dx)
            dim_z = np.int_(dim_z1 + (dim_z - 1) / 2 + 1)
            _dist1 = dx * dim_z1
            z = np.linspace(dd.z - _dist1, dd.z + _dist, dim_z)
        locs = np.zeros((dim_z, 3))
        ids = np.zeros((dim_z, 2))
        mass = np.ones((dim_z))
        locs[:, 0] = dd.lat
        locs[:, 1] = dd.lon
        locs[:, 2] = z
        ids[:, 0] = k
        ids[:, 1] = np.arange(dim_z)
        return locs, ids, mass

    def _create_srcs_from_zero(self, dd, k):
        """Short summary.

        Parameters
        ----------
        dd : pandas series
            Containf lat lon and z `dd`.
        k : type
            the source id `k`.

        Returns
        -------
        type
            Description of returned object.

        """
        dim_z = np.int_(dd.z / 50)
        z = np.linspace(50, dd.z, dim_z)
        locs = np.zeros((dim_z, 3))
        ids = np.zeros((dim_z, 2))
        mass = np.ones((dim_z))
        locs[:, 0] = dd.lat
        locs[:, 1] = dd.lon
        locs[:, 2] = z
        ids[:, 0] = k
        ids[:, 1] = np.arange(dim_z)
        return locs, ids, mass

    def __get_sources_from_hdf(self, _grp, _key):
        df = DataFrame()
        df["lat"] = _grp["latitude"][:]
        df["lon"] = _grp["longitude"][:]
        df["z"] = _grp["injection_height"][:]
        if _key == "Same":
            df1 = df[df["z"] > 0]
            df1.reset_index(drop=True, inplace=True)
        else:
            df1 = df[df["z"] > 0]
            df1.reset_index(drop=True, inplace=True)

        locs = np.array([]).reshape(0, 3)
        ids = np.array([]).reshape(0, 2)
        mass = np.array([]).reshape(0)
        for i in range(len(df1)):
            if _key == "Same":
                _lcs, _ids, _mass = self._create_srcs_vert(df1.loc[i], i)
            else:
                _lcs, _ids, _mass = self._create_srcs_from_zero(df1.loc[i], i)

            locs = np.concatenate((locs, _lcs))
            ids = np.concatenate((ids, _ids))
            mass = np.concatenate((mass, _mass))
        nos = locs.shape[0]
        return nos, locs, mass, ids

    # def initialize_fire_src(self, lat, lon, z, topology):
    #     return InitializeFireSource(lat, lon, z, self.topo_flag, topology)


class InitializeSource:
    def __init__(self, srcs, topology, heightfromsurface):
        self.topo_flag = heightfromsurface
        df0 = self.get_locations_fromarray(srcs[0], srcs[1], srcs[2], shift=-500)
        df1 = self.get_locations_fromarray(srcs[0], srcs[1], srcs[2], shift=0)
        df2 = self.get_locations_fromarray(srcs[0], srcs[1], srcs[2], shift=500)
        df = concat((df0, df1, df2))
        self.nos = len(df)
        self.locs = df[["lat", "lon", "z"]].values
        self.mass = np.ones(self.nos)
        self.ids = df[["src_ids", "vert_id"]].values
        if self.topo_flag:
            _loc = (self.locs[:, :2]).copy()
            self.topo_ht = topology.get_topology(_loc)
            self.loc_topo = self.locs.copy()
            self.loc_topo[:, 2] += self.topo_ht

    def get_data(self):
        """
        Takes time in seconds as input and output is source, mass and ids
        """
        if self.topo_flag:
            loc = self.loc_topo.copy()
        else:
            loc = self.locs.copy()
        return loc, self.mass.copy(), self.ids.copy()

    def get_locations_fromarray(self, lat, lon, z, shift=0):
        if shift == 0:
            df = DataFrame()
            df["lat"] = lat
            df["lon"] = lon
            df["z"] = z
            df1 = df[df["z"] > 0]
            df1.reset_index(drop=True, inplace=True)
            df1.insert(0, "src_ids", np.arange(len(df1)))
            df1.insert(1, "vert_id", 1)
        else:
            df = DataFrame()
            df["lat"] = lat
            df["lon"] = lon
            df["z"] = z
            df1 = (df[df["z"] > 0]).copy(deep=True)
            df1.reset_index(drop=True, inplace=True)
            arr = df1["z"] + shift
            arr[arr < 50] = 50
            df1["z"] = arr
            df1.insert(0, "src_ids", np.arange(len(df1)))
            if shift == -500:
                df1.insert(1, "vert_id", 0)
            else:
                df1.insert(1, "vert_id", 2)
        return df1
