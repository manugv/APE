#!/usr/bin/env python
# coding: utf-8
"""
Created on July 3 12:50 2020
Convert trajectory to concentrations in 3d
@author: Manu Goudar
"""

import numpy as np
from .ModEE_GridMapping import GridMapping


class Traj2Conc:
    """
    This class defines the 3d grids. Concentration info is written
    into these 3d grids. These grids follow surface topology (z).
    Resolution is in degree for lat & lon and has same value.
    For z direction, it is given in mts.
    """

    def __init__(self, params, roi, simtime, topology):
        self.param = params
        self.roi = roi
        self.dzi = 1.0 / self.param.roi_resolution

        # Time definitions
        # Number of time blocks
        self.blocks = 0
        # initial time
        tt = self.param.starttime - self.param.avgtime
        while tt <= simtime:
            self.blocks += 1
            tt += self.param.deltatime
        # Define start and end time arrays
        st_time = np.zeros((self.blocks))
        for i in range(self.blocks):
            st_time[i] = self.param.starttime + i * self.param.deltatime - self.param.avgtime
        self.start_t = st_time.copy()
        self.end_t = st_time + self.param.avgtime

        # Define grid
        # bounds of region of interest
        self.bound = np.asarray([self.roi.lat, self.roi.lon, self.roi.z])
        # Create one dimensional grid
        self.lat, self.lon, self.z, self.z1 = self.__create_grid()
        # Create surface grid (lat, lon, ht)
        lt, ln = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.surface = np.zeros(np.shape(lt) + (3,))
        self.surface[:, :, 0] = lt
        self.surface[:, :, 1] = ln
        self.topology = topology
        # update z based on surface_topology
        self.surface[:, :, 2] = self.topology(self.surface[:, :, :2])

        # Containers to store concentrations
        # dimensions [lat, lon, ht, time]
        shp = tuple(np.asarray(self.z1.shape) - 1)
        self.final_conc = np.zeros(shp + (self.blocks,))
        self.concentrations = np.zeros(shp)
        self.sample = np.zeros(shp)
        self.ids = 0
        # create box around a particle
        bx = self.param.roi_resolution / 2.0
        self.box_x = np.array([-bx[0], bx[0]])
        self.box_y = np.array([-bx[1], bx[1]])
        self.box_z = np.array([-bx[2], bx[2]])
        # volume of each box
        dx = self.param.roi_resolution
        self.unit_vol = dx[0] * dx[1] * dx[2]

        # Define grid mapping
        # Create mapping from concentration to tropomi grid
        # self.mapping = GridMapping(co_lat, co_lon, self.lat, self.lon, self.z)
        # Mapping data
        self.mapped_data = []

    def __create_grid(self):
        lat = np.arange(self.roi.lat[0], self.roi.lat[1], self.param.roi_resolution[0])
        lon = np.arange(self.roi.lon[0], self.roi.lon[1], self.param.roi_resolution[1])
        z = np.arange(self.roi.z[0], self.roi.z[1], self.param.roi_resolution[2])
        z1 = np.zeros((lat.size, lon.size, z.size))
        for i in range(z.size):
            z1[:, :, i] = z[i]
        return lat, lon, z, z1

    def init_id(self, time):
        """
        Modify start of concentration block based on simulation time
        """
        self.ids = -999
        for i in range(self.end_t.size):
            if time < self.end_t[i]:
                self.ids = i
                break
        if (self.ids >= self.blocks) or (self.ids == -999):
            return False
        else:
            return True

    def __check_bounds(self, p0, p1, p2):
        # Create lambda statement to check all these statements
        index = (
            (p0 >= self.bound[0, 0])
            & (p0 <= self.bound[0, 1])
            & (p1 >= self.bound[1, 0])
            & (p1 <= self.bound[1, 1])
            & (p2 >= self.bound[2, 0])
            & (p2 <= self.bound[2, 1])
        )
        return index

    def __get_grid_id(self, p):
        return (np.multiply((p - self.bound[:, 0]), self.dzi)).astype(int)

    def __get_adjecent_boxes(self, part, grid, ids):
        if part < grid:
            xids = np.array([-1, 0])
            # If ids is the first point
            if ids == 0:
                xids = np.array([0])
            else:
                xids = xids + ids
        else:
            xids = np.array([0, 1]) + ids
        return xids

    def update_conc(self, pos, mass):
        # Subtract topology height by getting height at lat, lon
        p1 = pos.copy()
        p1[2, :] -= self.topology(p1.T[:, :2])
        # Check if the point is inside the bounds
        info = self.__check_bounds(p1[0, :], p1[1, :], p1[2, :])
        # if yes,
        for p in range(len(mass)):
            # get index of the position
            # If inside the bounds then get box enclosing the particle
            # and compute the intersecting volume
            if info[p]:
                idx = (np.multiply((p1[:, p] - self.bound[:, 0]), self.dzi)).astype(int)
                # Get x and y limits of the particle box
                px = p1[0, p] + self.box_x
                py = p1[1, p] + self.box_y
                pz = p1[2, p] + self.box_z
                # Find eight adjecent boxes w.r.t first point in lat, lon and z
                _xids = self.__get_adjecent_boxes(px[0], self.lat[idx[0]], idx[0])
                _yids = self.__get_adjecent_boxes(py[0], self.lon[idx[1]], idx[1])
                _zids = self.__get_adjecent_boxes(pz[0], self.z[idx[2]], idx[2])
                # Get intersecting points to compute volume for adjecent boxes
                rx = np.minimum(self.lat[_xids + 1], px[1]) - np.maximum(self.lat[_xids], px[0])
                ry = np.minimum(self.lon[_yids + 1], py[1]) - np.maximum(self.lon[_yids], py[0])
                rz = np.minimum(self.z[_zids + 1], pz[1]) - np.maximum(self.z[_zids], pz[0])
                # compute volume for adjecent boxes
                for i in range(len(_xids)):
                    for j in range(len(_yids)):
                        for k in range(len(_zids)):
                            vol = rx[i] * ry[j] * rz[k]
                            conc = mass[p] * vol / self.unit_vol
                            self.concentrations[_xids[i], _yids[j], _zids[k]] += conc
                            self.sample[_xids[i], _yids[j], _zids[k]] += 1

    def compute_final_concentrations(self):
        # Average over a sample
        self.sample[self.sample == 0] = 1
        self.concentrations[:, :, :] /= self.sample
        self.final_conc[:, :, :, self.ids] += self.concentrations
        self.ids += 1
        self.concentrations[:, :, :] = 0
        self.sample[:, :, :] = 0
        # this says to stop computation even if the simulatin time continues
        if self.ids >= self.blocks:
            return False
        else:
            return True

    def to_tropomi(self, avg_kernel):
        """
        Convert concentration in regular grid to tropomi grid
        """
        # Flatten the regular grid in vertical direction
        # Tropomi resolution in vertical direction is 1000mts
        for i in range(self.blocks):
            a = self.mapping.conc2tropomi(self.final_conc[:, :, :, i])
            kk = self.mapping.vertical
            a1 = np.multiply(avg_kernel[:, :, :kk], a)
            self.mapped_data.append(a1)
        print("Mapped concentration data to tropomi")
