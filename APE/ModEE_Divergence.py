#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept 27 16:11:59 2023.

@author: Arthur B, Manu Goudar
"""

import numpy as np
from scipy.interpolate import griddata
from .ModuleDataContainers import DataContainer
from .ModEE_VelocityInterpolation2d import VelocityInterpolation


def createcartesiangrid(resolution, x=[-80000, 80000], y=[-80000, 80000]):
    """Create grid with default 80kmx80km.

    Parameters
    ----------
    resolution : Float
        In meters
    x : Array(2), Optional
        min and max x values in meters
    y : Array(2), Optional
        min and max y values in meters

    """
    xnew = np.arange(x[0], x[1]+1, resolution)
    ynew = np.arange(y[0], y[1]+1, resolution)
    return np.meshgrid(xnew, ynew)


def creategrid_divergence(res, transform):
    """Create grid at input res.


    Parameters
    ----------
    res : Float
        Grid should be at this resolution(input in mts)
    transform : Class TransformCoords
        Transfrom coords

    """
    data = DataContainer()
    # create new grid
    data.__setattr__("dx", res)
    data.__setattr__("dy", res)
    xnew, ynew = createcartesiangrid(res)
    data.__setattr__("x", xnew)
    data.__setattr__("y", ynew)
    lat, lon = transform.xymts2latlon(xnew, ynew)
    data.__setattr__("lat", lat)
    data.__setattr__("lon", lon)
    return data


def regrid_and_interpolate(grid, satx, saty, conc, wind, intmethod="linear"):
    # Compare with linear/cubic
    new_co = griddata((satx.flatten(), saty.flatten()), conc.flatten(), (grid.lat, grid.lon), method=intmethod)
    data = DataContainer()
    data.__setattr__("conc", new_co)
    # wind
    u, v = wind.interpolate(grid.lat, grid.lon)
    data.__setattr__("u", u)
    data.__setattr__("v", v)
    return data


def computedivergence(data, grid):
    Fx = data.conc*data.u
    Fy = data.conc*data.v
    dfx = np.zeros(np.shape(data.conc))
    dfy = np.zeros(np.shape(data.conc))
    # inner indices
    dfx[2:-2, 2:-2] = (-Fx[2:-2, 4:] + 8*Fx[2:-2, 3:-1]
                       - 8*Fx[2:-2, 1:-3] + Fx[2:-2, 0:-4])/(12*grid.dx)
    dfy[2:-2, 2:-2] = (-Fy[4:, 2:-2] + 8*Fy[3:-1, 2:-2]
                       - 8*Fy[1:-3, 2:-2] + Fy[0:-4, 2:-2])/(12*grid.dy)
    # add data to data class
    data.__setattr__("div", dfx+dfy)
    data.__setattr__("dfdx", dfx)
    data.__setattr__("dfdy", dfy)
    return data


def divergence(params, satellitedata, newgrid):
    """Divergence method

    Parameters
    ----------
    params : Class
        Class of parameters
    satellitedata : Data Class
        Data containing satellite data
    newgrid : Grid class
        Class containing grid data
    """
    # Wind data
    emis = params.estimateemission  # get the class containing emission params
    dirprefix = emis.flow.inputdir+params.source_name+"_"
    wind = VelocityInterpolation(dirprefix, emis.plumeheight)
    wind.computefunction(satellitedata.measurement_time)
    # regrid interpolate
    data = regrid_and_interpolate(newgrid, satellitedata.lat, satellitedata.lon, satellitedata.co_column_corr, wind)
    # add the pume height variable
    data.__setattr__("plumeheight", emis.plumeheight)
    # compute massflux
    data = computedivergence(data, newgrid)
    return data
