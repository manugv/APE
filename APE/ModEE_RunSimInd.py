#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:11:59 2022.

@author: Manu Goudar
"""

# from .ModEE_Simulation import Simulation3d
from .ModEE_EmissionEstimate import create_tlines_remove_background
from .ModEE_VelocityInterpolation2d import VelocityInterpolation
import numpy as np


def emission_estimates_industry(massflux, flow):
    # the transaction lines
    _ed = min(20, len(massflux.tlines))
    flag2 = 0
    flag3 = 0
    for _ln in massflux.tlines[:_ed]:
        # If the difference between two sides is not high then continue
        if _ln.f_background_good:
            # To compute emissions create a factor
            fact_emis = _ln.final_co * 28.01 * 0.001 * _ln.ds
            # velocity
            coords = _ln.final_coords_deg.copy()
            # get velocity
            u, v = flow.interpolate_vel(coords[:, 0], coords[:, 1])
            vel_mag = u * _ln.dir_vect[0] + v * _ln.dir_vect[1]
            _ln.__setattr__("vel_mag", vel_mag.copy())
            flag = np.nanmean(vel_mag) > 2.0
            flag2 += 1
            flag3 += flag
            _ln.__setattr__("flag_vel", flag)
            # emissions
            _ln.__setattr__("emission_line", fact_emis * _ln.vel_mag)
            _ln.__setattr__("emission", np.nansum(_ln.emission_line))
    if flag3 / flag2 < 0.5:
        print("         Velocity < 2m/s")
        massflux.__setattr__("velocity_flag", False)
    else:
        massflux.__setattr__("velocity_flag", True)


def emissionestimation(day, globalparams, satdata, plumecontainer):
    # Create transaction lines and remove background
    massflux = create_tlines_remove_background(satdata, plumecontainer, satdata.transform)
    # Initialize velocity calss
    flow = VelocityInterpolation(globalparams.param_flowinfo.inputdir)
    flow.computefunction(satdata.measurement_time)
    # IF the plume was good after background subtraction
    # then compute the lagrangian simulations and extract height

    if massflux.f_good_plume_bs:
        # compute emission at 100m
        emission_estimates_industry(massflux, flow)
    else:
        print("          Background subtraction fails")
    return massflux
