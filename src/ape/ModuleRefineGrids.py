#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine structured grids.

Refinement assumes grids are co-rectangular.

Created on Wed Jun  8 15:14:13 2022.
@author: Manu Goudar
"""

import numpy as np


class RefineGridsUniform:
    def __init__(self, no_times):
        self.no_times = no_times

    def split_data_x(self, x, nx, ny):
        """
        Add data in x direction
        """
        xnew = np.zeros((nx, ny))
        # splits single box into multiple based on no_times
        dx = (x[1:, :] - x[:-1, :])/self.no_times
        xnew[::self.no_times, :] = x
        for i in range(1, self.no_times):
            xnew[i::self.no_times, :] = x[:-1, :] + dx*i
        return xnew

    def split_data_y(self, y, nx, ny):
        """
        Add data in x direction
        """
        ynew = np.zeros((nx, ny))
        # splits single box into multiple based on no_times
        dy = (y[:, 1:] - y[:, :-1])/self.no_times
        ynew[:, ::self.no_times] = y
        for i in range(1, self.no_times):
            ynew[:, i::self.no_times] = y[:, :-1] + dy*i
        return ynew

    @staticmethod
    def get_center_values(x):
        x1 = (x[1:, :] + x[:-1, :])*0.5
        return (x1[:, 1:] + x1[:, :-1])*0.5

    def resize_values(self, z):
        orig = z.shape
        nx = orig[0]*self.no_times
        ny = orig[1]*self.no_times
        # put values in x direction
        z_new = np.zeros((nx, orig[1]), dtype=z.dtype)
        z_new[::self.no_times, :] = z
        for i in range(1, self.no_times):
            z_new[i::self.no_times, :] = z
        # put values in y direction
        z_new1 = np.zeros((nx, ny), dtype=z.dtype)
        z_new1[:, ::self.no_times] = z_new
        for i in range(1, self.no_times):
            z_new1[:, i::self.no_times] = z_new
        return z_new1

    # REFINE DATA
    def resize_coordinates(self, x, y):
        """
        Refine the grids by splitting each grid into 4 grids.
        CO data remais same as it is described per sq.mts
        x and y are data on nodes
        """
        sh = x.shape
        orig_nx = sh[0] - 1
        orig_ny = sh[1] - 1
        nx = orig_nx*self.no_times+1
        ny = orig_ny*self.no_times+1

        # define new arrays for new data
        _extx = self.split_data_x(x, nx, orig_ny+1)
        x_nodes = self.split_data_y(_extx, nx, ny)
        x_centers = self.get_center_values(x_nodes)

        _exty = self.split_data_x(y, nx, orig_ny+1)
        y_nodes = self.split_data_y(_exty, nx, ny)
        y_centers = self.get_center_values(y_nodes)
        return x_nodes, y_nodes, x_centers, y_centers


class RefineGrids:
    def __init__(self, no_times_x, no_times_y):
        self.no_times_x = no_times_x
        self.no_times_y = no_times_y

    def split_data_x(self, x, nx, ny):
        """
        Add data in x direction
        """
        xnew = np.zeros((nx, ny))
        # splits single box into multiple based on no_times
        dx = (x[1:, :] - x[:-1, :])/self.no_times_x
        xnew[::self.no_times_x, :] = x
        for i in range(1, self.no_times_x):
            xnew[i::self.no_times_x, :] = x[:-1, :] + dx*i
        return xnew

    def split_data_y(self, y, nx, ny):
        """
        Add data in y direction
        """
        ynew = np.zeros((nx, ny))
        # splits single box into multiple based on no_times
        dy = (y[:, 1:] - y[:, :-1])/self.no_times_y
        ynew[:, ::self.no_times_y] = y
        for i in range(1, self.no_times_y):
            ynew[:, i::self.no_times_y] = y[:, :-1] + dy*i
        return ynew

    @staticmethod
    def get_center_values(x):
        x1 = (x[1:, :] + x[:-1, :])*0.5
        return (x1[:, 1:] + x1[:, :-1])*0.5

    def resize_values(self, z):
        orig = z.shape
        nx = orig[0]*self.no_times_x
        ny = orig[1]*self.no_times_y
        # put values in x direction
        z_new = np.zeros((nx, orig[1]), dtype=z.dtype)
        z_new[::self.no_times_x, :] = z
        for i in range(1, self.no_times_x):
            z_new[i::self.no_times_x, :] = z
        # put values in y direction
        z_new1 = np.zeros((nx, ny), dtype=z.dtype)
        z_new1[:, ::self.no_times_y] = z_new
        for i in range(1, self.no_times_y):
            z_new1[:, i::self.no_times_y] = z_new
        return z_new1

    # REFINE DATA
    def resize_coordinates(self, x, y):
        """
        Refine the grids by splitting each grid into 4 grids.
        CO data remais same as it is described per sq.mts
        x and y are data on nodes
        """
        sh = x.shape
        orig_nx = sh[0] - 1
        orig_ny = sh[1] - 1
        nx = orig_nx*self.no_times_x+1
        ny = orig_ny*self.no_times_y+1

        # define new arrays for new data
        _extx = self.split_data_x(x, nx, orig_ny+1)
        x_nodes = self.split_data_y(_extx, nx, ny)
        x_centers = self.get_center_values(x_nodes)

        _exty = self.split_data_x(y, nx, orig_ny+1)
        y_nodes = self.split_data_y(_exty, nx, ny)
        y_centers = self.get_center_values(y_nodes)
        return x_nodes, y_nodes, x_centers, y_centers


# refine = Refine_grids(2,4)
# xn, yn, xc, yc = refine.resize_coordinates(dat1['lat_nodes'].copy(), dat1['lon_nodes'].copy())
# zval = refine.resize_values(plumemask)
# co_cal = refine.resize_values(co_orig)

# class Refine_grids:
#     def __init__(self, no_times_x, no_times_y):
#         self.no_times_x = no_times_x
#         self.no_times_y = no_times_y

#     def split_data_x(self, x, nx, ny):
#         """
#         Add data in x direction
#         """
#         xnew = np.zeros((nx,ny))
#         # splits single box into multiple based on no_times
#         dx = (x[1:, :] - x[:-1, :])/self.no_times_x
#         xnew[::self.no_times_x,:] = x
#         for i in range(1, self.no_times_x):
#             xnew[i::self.no_times_x,:] = x[:-1,:] + dx*i
#         return xnew

#     def split_data_y(self, y, nx, ny):
#         """
#         Add data in y direction
#         """
#         ynew = np.zeros((nx,ny))
#         # splits single box into multiple based on no_times
#         dy = (y[:, 1:] - y[:, :-1])/self.no_times_y
#         ynew[:,::self.no_times_y] = y
#         for i in range(1, self.no_times_y):
#             ynew[:, i::self.no_times_y] = y[:,:-1] + dy*i
#         return ynew

#     def get_center_values(self, x):
#         x1 = (x[1:,:] + x[:-1,:])*0.5
#         return (x1[:,1:] + x1[:,:-1])*0.5


#     def resize_values(self, z):
#         orig = z.shape
#         nx = orig[0]*self.no_times_x
#         ny = orig[1]*self.no_times_y
#         # put values in x direction
#         z_new = np.zeros((nx, orig[1]))
#         z_new[::self.no_times_x,:] = z
#         for i in range(1, self.no_times_x):
#             z_new[i::self.no_times_x,:] = z
#         # put values in y direction
#         z_new1 = np.zeros((nx, ny))
#         z_new1[:,::self.no_times_y] = z_new
#         for i in range(1, self.no_times_y):
#             z_new1[:, i::self.no_times_y] = z_new
#         return z_new1

#     # REFINE DATA
#     def resize_coordinates(self, x, y):
#         """
#         Refine the grids by splitting each grid into 4 grids.
#         CO data remais same as it is described per sq.mts
#         x and y are data on nodes
#         """
#         sh = x.shape
#         orig_nx = sh[0] - 1
#         orig_ny = sh[1] - 1
#         nx = (orig_nx)*self.no_times_x+1
#         ny = (orig_ny)*self.no_times_y+1

#         # define new arrays for new data
#         _extx = self.split_data_x(x, nx, orig_ny+1)
#         x_nodes = self.split_data_y(_extx, nx, ny)
#         x_centers = self.get_center_values(x_nodes)

#         _exty = self.split_data_x(y, nx, orig_ny+1)
#         y_nodes = self.split_data_y(_exty, nx, ny)
#         y_centers = self.get_center_values(y_nodes)
#         return x_nodes, y_nodes, x_centers, y_centers
