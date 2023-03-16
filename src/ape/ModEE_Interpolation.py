#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 11 15:48:10 2021
Particles interpolation
@author: Manu Goudar
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI


def get_interpolation_func_2d(flow, time):
    """
    Based on time update interpolation functions
    They are initialized at time 0
    Time in hours and is an integer
    """
    f_u = RGI((flow.lat, flow.lon), flow.u[time, :, :].data)
    f_v = RGI((flow.lat, flow.lon), flow.v[time, :, :].data)
    return f_u, f_v


# =============================================================================
#
# =============================================================================
class Regular_2d_Interpolate:
    """
    This algorithm computes interpolation based on 2d regular grid (xy)
    """

    def __init__(self, lat, lon):
        self.low_latlim, self.up_latlim, self.dlat_i = self.__get_vars(lat)
        self.low_lonlim, self.up_lonlim, self.dlon_i = self.__get_vars(lon)
        self.low_lim = np.array([self.low_latlim, self.low_lonlim])
        self.dxi = np.array([self.dlat_i, self.dlon_i])

    def __get_vars(self, xx):
        low_lim = xx[0]
        up_lim = xx[-1]
        dxi = 1.0 / (xx[1] - xx[0])
        return low_lim, up_lim, dxi

    def __get_loc(self, pt):
        return (np.multiply((pt - self.low_lim), self.dxi) + 1).astype(int)

    def __get_differences(self, pc, c):
        x1 = (pc - c[0]) / (c[1] - c[0])
        return np.array([1 - x1, x1])

    def get_velocity(self, flow, time, pt):
        # get index of 4 vertices that encapsulate the point
        [kx, ky] = self.__get_loc(pt)

        # compute difference ratio between given pt and coordinates of vortices
        xd = self.__get_differences(pt[0], flow.lat[kx - 1 : kx + 1])
        yd = self.__get_differences(pt[1], flow.lon[ky - 1 : ky + 1])

        # get velocities at those vertices
        u1 = flow.u[time, kx - 1 : kx + 1, ky - 1 : ky + 1]
        v1 = flow.v[time, kx - 1 : kx + 1, ky - 1 : ky + 1]

        # Get velocity fact*(x1*(vel*y1))
        vel_u = np.matmul(xd, np.matmul(u1, yd))
        vel_v = np.matmul(xd, np.matmul(v1, yd))
        return np.array([vel_u, vel_v])


# =============================================================================
#
# =============================================================================


class Interpolate_3d:
    """
    This algorithm computes interpolation based on 2d regular grid (xy) and
    an irregular grid on third direction (z).
    """

    def __init__(self, lat, lon, zlen):
        low_latlim, up_latlim, dlat_i = self.__get_vars(lat)
        low_lonlim, up_lonlim, dlon_i = self.__get_vars(lon)
        self.low_lim = np.array([low_latlim, low_lonlim])
        self.up_lim = np.array([up_latlim, up_lonlim])
        self.dxi = np.array([dlat_i, dlon_i])
        self.zlen = zlen
        self.u1 = np.zeros((2, 2))
        self.v1 = np.zeros((2, 2))
        self.w1 = np.zeros((2, 2))
        self.velo = np.zeros((3))
        self.xd = np.zeros((2))
        self.yd = np.zeros((2))

    def __get_vars(self, xx):
        low_lim = xx[0]
        up_lim = xx[-1]
        dxi = 1.0 / (xx[1] - xx[0])
        return low_lim, up_lim, dxi

    def __get_z_loc(self, z, ptz):
        # get z in the box
        for kk in range(1, self.zlen):
            if ptz < z[kk]:
                zidx = kk - 1
                zd = (ptz - z[zidx]) / (z[zidx + 1] - z[zidx])
                break
        return zidx, zd

    def __get_differences(self, pc, c):
        x1 = (pc - c[0]) / (c[1] - c[0])
        return 1 - x1, x1

    def get_velocity(self, flow, time, pt):
        # Check if the given point is inside the domain
        if any(
            [
                pt[0] < self.low_lim[0],
                pt[0] > self.up_lim[0],
                pt[1] < self.low_lim[1],
                pt[1] > self.up_lim[1],
            ]
        ):
            self.velo[:] = 0
            lmt = False
            # print("Out of bounds")
        else:
            # Get 2 indices of regular grid xy (get lower index)
            [kx, ky] = (np.multiply((pt[:2] - self.low_lim), self.dxi)).astype(int)
            # compute difference ratio between given pt and
            # coordinates of vortices
            self.xd[:] = self.__get_differences(pt[0], flow.lat[kx : kx + 2])
            self.yd[:] = self.__get_differences(pt[1], flow.lon[ky : ky + 2])
            # Interpolate in z direction
            # 8 points in cube become 4 points
            # z has 4 points as it is non-uniform
            info1 = 0
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                # Get z for i and j
                z = flow.z[kx + i, ky + j, :, time]
                # check if point lies inside the z domain
                info = (pt[2] <= z[-1]) & (pt[2] >= z[0])
                if info:
                    # get z location and difference ratio
                    kk, zd = self.__get_z_loc(z, pt[2])
                    zd1 = 1 - zd
                    self.u1[i, j] = (
                        zd * flow.u[kx + i, ky + j, kk + 1, time]
                        + zd1 * flow.u[kx + i, ky + j, kk, time]
                    )
                    self.v1[i, j] = (
                        zd * flow.v[kx + i, ky + j, kk + 1, time]
                        + zd1 * flow.v[kx + i, ky + j, kk, time]
                    )
                    self.w1[i, j] = (
                        zd * flow.w[kx + i, ky + j, kk + 1, time]
                        + zd1 * flow.w[kx + i, ky + j, kk, time]
                    )
                else:
                    info1 += -1
                    self.u1[i, j] = 0
                    self.v1[i, j] = 0
                    self.w1[i, j] = 0
            if info1 == -4:
                self.velo[:] = 0
                lmt = False
                # print("Out of bounds")
            else:
                # Get velocity in 2d regular grid space (xd*(vel*yd))
                self.velo[0] = np.matmul(self.xd, np.matmul(self.u1, self.yd))
                self.velo[1] = np.matmul(self.xd, np.matmul(self.v1, self.yd))
                self.velo[2] = np.matmul(self.xd, np.matmul(self.w1, self.yd))
                lmt = True
        return self.velo, lmt


# class Interpolate_3d:
#     """
#     This algorithm computes interpolation based on 2d regular grid (xy) and
#     an irregular grid on third direction (z).
#     """

#     def __init__(self, lat, lon, zlen):
#         low_latlim, up_latlim, dlat_i = self.__get_vars(lat)
#         low_lonlim, up_lonlim, dlon_i = self.__get_vars(lon)
#         self.low_lim = np.array([low_latlim, low_lonlim])
#         self.up_lim = np.array([up_latlim, up_lonlim])
#         self.dxi = np.array([dlat_i, dlon_i])
#         self.zlen = zlen
#         self.u1 = np.zeros((2, 2))
#         self.v1 = np.zeros((2, 2))
#         self.w1 = np.zeros((2, 2))
#         self.velo = np.zeros((3))
#         self.xd = np.zeros((2))
#         self.yd = np.zeros((2))

#     def __get_vars(self, xx):
#         low_lim = xx[0]
#         up_lim = xx[-1]
#         dxi = 1.0/(xx[1] - xx[0])
#         return low_lim, up_lim, dxi

#     def __get_z_loc(self, z, ptz):
#         # get z in the box
#         for kk in range(1, self.zlen):
#             if ptz < z[kk]:
#                 zidx = kk - 1
#                 zd = (ptz-z[zidx])/(z[zidx+1]-z[zidx])
#                 break
#         return zidx, zd

#     def __get_differences(self, pc, c):
#         x1 = (pc-c[0])/(c[1]-c[0])
#         return 1-x1, x1

#     def get_velocity(self, flow, time, pt):
#         # define limits
#         low_lat = flow.lat[0]
#         up_lat = flow.lat[-1]
#         dlat_i = flow.lat[1] - flow.lat[0]
#         low_low = flow.lon[0]
#         up_low = flow.lon[-1]
#         dlon_i = flow.lon[1] - flow.lon[0]

#         # for p in range(particles.no_particles):
#         # Check if the given point is inside the domain
#         if any([pt[0] < flow.lat[0], pt[0] > flow.lat[-1],
#                 pt[1] < flow.lon[0], pt[1] > flow.lon[-1]]):
#             self.velo[:] = 0
#             lmt = False
#             # print("Out of bounds")
#         else:


#             # Get 2 indices of regular grid xy (get lower index)
#             kx = np.int((pt[0]-flow.lat[0])*dxi)
#             ky = np.int((pt[1]-flow.lon[0])*dyi)

#             # compute difference ratio between given pt and
#             # coordinates of vortices
#             self.xd[:] = self.__get_differences(pt[0], flow.lat[kx:kx+2])
#             self.yd[:] = self.__get_differences(pt[1], flow.lon[ky:ky+2])
#             # Interpolate in z direction
#             # 8 points in cube become 4 points
#             # z has 4 points as it is non-uniform
#             info1 = 0
#             for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
#                 # Get z for i and j
#                 z = flow.z[kx+i, ky+j, :, time]
#                 # check if point lies inside the z domain
#                 info = (pt[2] <= z[-1]) & (pt[2] >= z[0])
#                 if info:
#                     # get z location and difference ratio
#                     kk, zd = self.__get_z_loc(z, pt[2])
#                     zd1 = 1-zd
#                     self.u1[i, j] = (zd*flow.u[kx+i, ky+j, kk+1, time]
#                                      + zd1*flow.u[kx+i, ky+j, kk, time])
#                     self.v1[i, j] = (zd*flow.v[kx+i, ky+j, kk+1, time]
#                                      + zd1*flow.v[kx+i, ky+j, kk, time])
#                     self.w1[i, j] = (zd*flow.w[kx+i, ky+j, kk+1, time]
#                                      + zd1*flow.w[kx+i, ky+j, kk, time])
#                 else:
#                     info1 += -1
#                     self.u1[i, j] = 0
#                     self.v1[i, j] = 0
#                     self.w1[i, j] = 0
#             if info1 == -4:
#                 self.velo[:] = 0
#                 lmt = False
#                 # print("Out of bounds")
#             else:
#                 # Get velocity in 2d regular grid space (xd*(vel*yd))
#                 self.velo[0] = np.matmul(self.xd, np.matmul(self.u1, self.yd))
#                 self.velo[1] = np.matmul(self.xd, np.matmul(self.v1, self.yd))
#                 self.velo[2] = np.matmul(self.xd, np.matmul(self.w1, self.yd))
#                 lmt = True
#         return self.velo, lmt


class SurfaceTopology:
    def __init__(self, lat, lon, z):
        self.lat = lat
        self.lon = lon
        self.z = z
        self.fh = RGI((self.lat, self.lon), self.z)

    def get_topology(self, loc):
        return self.fh(loc)


class SurfaceTopology1:
    def __init__(self, lat, lon, z):
        self.lat = lat
        self.lon = lon
        self.z = z
        self.fh = RGI((self.lat, self.lon), self.z, bounds_error=False)

    def get_topology(self, loc):
        return self.fh(loc)
