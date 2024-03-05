#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept  8 16:11:59 2023.

@author: Manu Goudar
"""

from .ModEE_TransactionalLines import create_tlines_remove_background
from .ModEE_VelocityInterpolation2d import VelocityInterpolation
from .ModuleDataContainers import DataContainer
from numpy import nanmean, nansum, sqrt


def _velocityacrosstransect(coords, dir_vect, flow):
    # compute velocity at the constant injection height
    u, v = flow.interpolate(coords[:, 0], coords[:, 1])
    # compute velocity across the line
    vel_mag = u * dir_vect[0] + v * dir_vect[1]
    # Check if mean magnitude of velocity is > 2m/s
    flag = nanmean(sqrt(u**2 + v**2)) < 2
    return flag, vel_mag, u, v


def _emissionoftransect(_ln, ds_km, flow, molarmass):
    emis = DataContainer()
    # To compute emissions create a factor
    fact_emis = _ln.final_co * (ds_km*1000) * molarmass
    # compute velocity at the constant height
    flag, vel_mag, u, v = _velocityacrosstransect(_ln.final_coords_deg, _ln.dir_vect, flow)
    setattr(emis, "u", u)
    setattr(emis, "v", v)
    setattr(emis, "velmag", vel_mag)
    setattr(emis, "flag_velocitylessthan2", flag)
    if not emis.flag_velocitylessthan2:
        setattr(emis, "lineemission", fact_emis * emis.velmag)
        setattr(emis, "emission", nansum(emis.lineemission))
    return emis


def _estimateemissionconst(massflux, flow, var_name, molarmass):
    # the transaction lines
    _ed = min(20, len(massflux.tlines))
    emission = []
    for _ln in massflux.tlines[:_ed]:
        # If the difference between two sides is not high then continue
        if _ln.flag_backgroundremovalsuccess:
            # compute emissions over a line and dump into a container
            emis = _emissionoftransect(_ln, massflux.line_spacing_km, flow, molarmass)
            setattr(_ln, "emission_"+var_name, emis)
            if emis.flag_velocitylessthan2:
                continue
            emission.append(emis.emission)
    return emission


def crosssectionalflux_constant(params, satellitedata, plumedata, transform):
    # Create transaction lines and remove background
    massflux = create_tlines_remove_background(satellitedata, plumedata, transform)
    # Check if the plume was good after background subtraction
    if not massflux.flag_goodplume:
        return massflux, 0

    # constant plume height
    if params.estimateemission.plumeheighttype == "Constant":
        # get the velocity
        dirprefix = params.estimateemission.flow.inputdir + params.source_name + "_"
        wind = VelocityInterpolation(dirprefix, params.estimateemission.plumeheight)
        wind.computefunction(satellitedata.measurement_time)
        estimatedemission = _estimateemissionconst(
            massflux, wind, params.estimateemission.emisname, params.estimateemission.molarmass
        )
    else:
        estimatedemission = 0 
    return massflux, estimatedemission


