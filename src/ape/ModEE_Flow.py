#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:56:58 2020

@author: manu
"""
from ModEE_VerticalData import get_vertical_coord_vel
from datetime import datetime, timedelta
from netCDF4 import Dataset
import numpy as np
import glob

# import pygrib


class FlowData:
    """
    Get flow data
    """

    def __init__(self, flow, disp, simstarttime, simtimesec):
        self.lat = 0
        self.lon = 0
        self.u = 0
        self.v = 0
        if disp.simtype == "3d":
            self.levels = 0
            self.z = 0
            self.w = 0
            self.dimens = 0
        # Length of time for simulation
        self.time_len = 0
        # TODO
        self.time = 0
        self.data1 = 0
        # Start and End times in ERA5 database
        st_time, end_time = self.__get_start_end_time(disp.model, simstarttime, simtimesec)
        self.st_tim = st_time
        self.end_time = end_time
        if disp.simtype == "2d":
            # TODO: Implement start and end time in 2d
            self.getvelocities_2d(flow.file_flowfield)
        elif disp.simtype == "3d":
            self.getvelocities_3d(flow, st_time, end_time)

    def __gettime_inera5format(self, _vardatetime):
        """
        Takes a variable and converts it to
        number of hours since 1900-01-01,00:00:00
        """
        a = datetime(1900, 1, 1, 0, 0, 0)
        return (_vardatetime - a).total_seconds() / 3600.0

    def __get_start_end_time(self, dispmodel, sim_starttime, sim_timesec):
        """
        Compute start and end time in same format as ERA5 database
        Inputs:
               Dispersion model and simulation time
        """
        if dispmodel == "Forward":
            st_time = np.int_(self.__gettime_inera5format(sim_starttime))
            tmp = sim_starttime + timedelta(seconds=sim_timesec)
            _last = self.__gettime_inera5format(tmp)
            _last1 = np.ceil(_last)
            end_time = np.int_(_last1)
            # For interpolation in emissions
            if _last == _last1:
                self.emission_idx = [end_time - st_time]
                self.emission_fact = [0]
            else:
                _t = end_time - st_time
                fx = _last1 - _last
                self.emission_idx = [_t - 1, _t]
                self.emission_fact = [fx, 1 - fx]
        elif dispmodel == "Backward":
            tmp = sim_starttime - timedelta(seconds=sim_timesec)
            st_time = np.int_(self.__gettime_inera5format(tmp))
            end_time = np.int_(np.ceil(self.__gettime_inera5format(sim_starttime)))
        self.time_len = end_time - st_time + 1
        return st_time, end_time

    def __rearrange_data(self, var):
        """
        Rearrange variables data
        Latitude and levels are in decending order so convert
        them to ascending order and change variables accordingly
        """
        a_lv = 1  # self.dimens.index('level')
        a_lt = 2  # self.dimens.index('latitude')
        a_ln = 3  # self.dimens.index('longitude')
        a_t = 0  # self.dimens.index('time')
        v1 = np.flip(var, axis=a_lt)
        v1 = np.flip(v1, axis=a_lv)
        sh = var.shape
        v = np.zeros((sh[a_t], sh[a_lt], sh[a_ln], sh[a_lv] + 1))
        for ti in range(sh[a_t]):
            for lv in range(sh[a_lv]):
                v[ti, :, :, lv + 1] = v1[ti, lv, :, :]
        return v

    def __gettime_blocks(self, tt, flnames):
        """
        This function searches in different files for the
        given time and outputs list of [[starttime, endtime, filename]]
        """
        # Start time variable
        st_time = tt[0]
        block = []
        # find start file
        for fl in flnames:
            dd = Dataset(fl, "r")
            st = np.argwhere(dd["time"][:].data == st_time)
            if len(st) > 0:
                if tt[-1] <= dd["time"][:][-1]:
                    en = np.argwhere(dd["time"][:].data == tt[-1])[0]
                    block.append([st[0][0], en[0], fl])
                    dd.close()
                    return block, 1
                else:
                    block.append([st[0][0], (dd["time"][:]).size - 1, fl])
                    st_time = dd["time"][:][-1] + 1
            dd.close()
        return block, 0

    def __read_alllevel_data(self, block, _vars, time_len, _invarients, lvl):
        data = {}
        # Read invarients
        d1 = Dataset(block[0][-1], "r")
        _tmp_lvl = d1["level"][:]
        _l1 = np.where(_tmp_lvl == lvl[0])[0][0]
        _l2 = np.where(_tmp_lvl == lvl[1])[0][0] + 1
        for i in _invarients:
            if i == "level":
                data[i] = d1[i][_l1:_l2].data
            else:
                data[i] = d1[i][:].data
        # Dimension shape and names
        sh = d1["u"][0, 0].shape  # find array size in level, lat, lon
        self.dimens = d1["u"].dimensions  # Define Dimensions names
        d1.close()
        # lvl shape
        ll = _l2 - _l1
        # Define array size with time dimensions
        sh = tuple([time_len, ll] + list(sh))
        # print(sh, "vel time")
        # Read variables
        for i in _vars:
            tmp = np.zeros(sh)
            k = 0
            for blk in block:
                d1 = Dataset(blk[-1], "r")
                k1 = k + blk[1] - blk[0] + 1
                # print("vel", i, k, k1, blk[0], blk[1])
                tmp[k:k1] = d1[i][blk[0] : blk[1] + 1, _l1:_l2].data
                k = k1
                d1.close()
            data[i] = tmp
        return data

    def __read_singlelevel_data(self, block, _vars, time_len):
        data = {}
        # Define array size with time dimensions
        d1 = Dataset(block[0][-1], "r")
        sh = d1["z"][0].shape

        d1.close()
        sh = tuple([time_len] + list(sh))
        # print("sh", sh)
        # Read variables
        for i in _vars:
            tmp = np.zeros(sh)
            k = 0
            for blk in block:
                d1 = Dataset(blk[-1], "r")
                k1 = k + blk[1] - blk[0] + 1
                # print("single", i, k, k1, blk[1] - blk[0])
                tmp[k:k1] = d1[i][blk[0] : blk[1] + 1].data
                k = k1
                d1.close()
            if i == "z":
                data["geop"] = tmp
            elif i == "lnsp":
                data["pres"] = np.exp(tmp)
        return data

    # def get_grib_parameter(self, ff, _name):
    #     u = ff.select(name=_name)
    #     velu = np.zeros((24,137,41,41))
    #     for _uu in u:
    #         ii = _uu.level - 1
    #         velu[_uu.hour, ii, :, :] = _uu.values
    #     return velu

    # def get_grib_parameter_lvl1(self, ff, _name):
    #     u = ff.select(name=_name)
    #     velu = np.zeros((24,41,41))
    #     for _uu in u:
    #         velu[_uu.hour, :, :] = _uu.values
    #     return velu

    # def get_data_grib(self):
    #     ff = pygrib.open('/work/SRON/RF11/velo_temp_vars.grb')
    #     # data = []
    #     uvel = self.get_grib_parameter(ff,'U component of wind')
    #     vvel = self.get_grib_parameter(ff, 'V component of wind')
    #     temp = self.get_grib_parameter(ff, 'Temperature')
    #     q = self.get_grib_parameter(ff, 'Specific humidity')
    #     wvel = self.get_grib_parameter(ff,'Vertical velocity')
    #     _lat, _lon = ff[1].latlons()
    #     flat = _lat[:,0]
    #     flon = _lon[0,:]
    #     ff.close()

    # ff = pygrib.open('/work/SRON/RF11/pres_geop.grb')
    # geop = self.get_grib_parameter_lvl1(ff, 'Geopotential')
    # pres = self.get_grib_parameter_lvl1(ff,'Logarithm of surface pressure')
    # level = np.arange(1,138)
    # ff.close()
    # dataf = {}
    # dataf['u'] = uvel[12:20,80:,:,:]
    # dataf['v'] = vvel[12:20,80:,:,:]
    # dataf['w'] = wvel[12:20,80:,:,:]
    # dataf['t'] = temp[12:20,80:,:,:]
    # dataf['q'] = q[12:20,80:,:,:]
    # dataf['pres'] = np.exp(pres)[12:20,:,:]
    # dataf['geop'] = geop[12:20,:,:]
    # dataf['level'] = level[80:]
    # dataf['latitude'] = flat
    # dataf['longitude'] = flon
    # dataf['time'] = np.arange(7)
    # return dataf

    def getvelocities_3d(self, param, start_t, end_t):
        """
        Grep all files in a folder
        """
        # TODO read grib files
        # __tmp = True
        # if __tmp:
        allfiles = glob.glob(param.inputdir + "*.nc")
        # for velocity
        vel_qt_fls = [s for s in allfiles if param.file_flow in s]
        vel_qt_fls.sort()
        _key_invarients = "latitude longitude level time".split(" ")
        _key_vars1 = "u v w q t".split(" ")
        # for geopotential and surface pressure
        zlnsp_fls = [s for s in allfiles if param.file_pres in s]
        zlnsp_fls.sort()
        _key_vars2 = "z lnsp".split(" ")
        tt = np.arange(start_t, end_t + 1)
        vel_blk, info1 = self.__gettime_blocks(tt, vel_qt_fls)
        zlnsp_blk, info2 = self.__gettime_blocks(tt, zlnsp_fls)
        # print("velocity", vel_blk)
        # print("z levl", zlnsp_blk)
        data = self.__read_alllevel_data(
            vel_blk, _key_vars1, tt.size, _key_invarients, param.modellevels
        )
        data.update(self.__read_singlelevel_data(zlnsp_blk, _key_vars2, tt.size))
        # Functions to read grib files [Commented out now]
        # data = self.get_data_grib()
        # self.data1 = data
        # convert velocity in pa/s to m/s
        zh, w = get_vertical_coord_vel(data)
        # Flip variables to ascending order in height and latitude
        self.u = self.__rearrange_data(data["u"])
        self.v = self.__rearrange_data(data["v"])
        # In actual data negative velocity means upward motion so convert it
        # such that positive velocity corresponds to vertical motion
        self.w = -self.__rearrange_data(w)
        # level zero has non zero values
        self.z = self.__rearrange_data(zh)
        sh = self.z.shape[0]
        geo_ar = data["geop"][0, :, :] / 9.80665
        for ii in range(sh):
            self.z[ii, :, :, 0] = np.flip(geo_ar, axis=0)
        self.lat = (np.flip(data["latitude"])).astype(np.float64)
        if data["longitude"].max() > 180:
            self.lon = (data["longitude"] - 360).astype(np.float64)
        else:
            self.lon = (data["longitude"]).astype(np.float64)
        self.levels = data["level"]
        # self.time = tt
        del data
        # TODO: Return success variables
        return

    # TODO: Check if the source height is within limits
    def check_velo_source_ht(self, flowheight, sourceht):
        pass

    def getvelocities_2d(self, filename):
        self.filename_2d = filename

    #     # read velocity data at 10mts and 100mts (lower resolution)
    #     df1 = Dataset(filename, 'r')
    #     # Flip if the latitude values are in descending order
    #     self.lat = np.flip(df1['latitude'][:].data)
    #     self.lon = df1['longitude'][:].data
    #     # Here axis=0 is time (24hrs data) and axis=1 is latitude
    #     if flowheight == '10':
    #         self.u = np.flip(df1['u10'][:, :, :], axis=1)
    #         self.v = np.flip(df1['v10'][:, :, :], axis=1)
    #     elif flowheight == '100':
    #         self.u = np.flip(df1['u100'][:, :, :], axis=1)
    #         self.v = np.flip(df1['v100'][:, :, :], axis=1)
    #     df1.close()
    #     # TODO: Return success variables
    #     return
