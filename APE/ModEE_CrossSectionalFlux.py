#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sept  8 16:11:59 2023.

@author: Manu Goudar
"""

from .ModEE_Simulation import Simulation3d
from .ModEE_MassFlux import create_tlines_remove_background
from .ModEE_EmissionEstimate import (
    get_constant_plume_height,
    get_varying_plume_height,
    emission_estimates_varying_ht,
    emission_estimates_const_ht,
)
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
            # compute emissions over a line
            emis = _emissionoftransect(_ln, massflux.line_spacing_km, flow, molarmass)
            setattr(_ln, "emission_"+var_name, emis)
            if emis.flag_velocitylessthan2:
                continue
            emission.append(emis.emission)
    return emission


def emission_varyingheight(massflux, origin_src, transform, sources, paramee, measurement_time, unique_id):
    # IF the plume was good after background subtraction
    # then compute the lagrangian simulations and extract height
    if massflux.f_good_plume_bs:
        sim3d = Simulation3d(origin_src, transform, paramee, sources, measurement_time)
        sim3d.run()
        simname = paramee.particledir + unique_id
        sim3d.save(simname)

        # compute varying plume height and its emissions
        particle_data = sim3d.get_particle_data()
        get_varying_plume_height(massflux, particle_data)
        if massflux.f_particle_plume_alignment:
            emission_estimates_varying_ht(massflux, sim3d.flow)
        else:
            print("         Plume alignment fails")
    else:
        print("          Background subtraction fails")
        massflux.f_particle_plume_alignment = False
    return massflux


def crosssectionalflux(params, satellitedata, plumedata, transform, sources=None):
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
    # varying plume height
    elif params.estimateemission.plumeheighttype == "Varying":
        if sources is None:
            print("sources input needs to be given")
            exit()
        else:
            massflux = emission_varyingheight(
                massflux,
                satellitedata.source,
                transform,
                sources,
                params.estimateemission,
                satellitedata.measurement_time,
                satellitedata.uniqueid,
            )
            estimatedemission = 0  # TODO
    return massflux, estimatedemission


#     # Create transaction lines and remove background
#     massflux = create_tlines_remove_background(fire_satdata, plumecontainer, transform)

#     # IF the plume was good after background subtraction
#     # then compute the lagrangian simulations and extract height
#     if massflux.f_good_plume_bs:
#         sim3d = Simulation3d(fire_satdata.source, transform, globalparams,
#                              fire_viirs, fire_satdata.measurement_time)
#         sim3d.run()
#         simname = (
#             globalparams.output_particlefile_prefix
#             + day.strftime("%Y_%m_%d")
#             + "_"
#             + fire_satdata.fire_name
#         )

#         sim3d.save(simname)

#         # compute constant injection height and emissions
#         get_constant_plume_height(fire_viirs.injection_height, massflux.tlines, sim3d.topology)
#         emission_estimates_const_ht(massflux, sim3d.flow)
#         # compute varying plume height and its emissions
#         particle_data = sim3d.get_particle_data()
#         get_varying_plume_height(massflux, particle_data)
#         if massflux.f_particle_plume_alignment:
#             emission_estimates_varying_ht(massflux, sim3d.flow)
#         else:
#             print("         Plume alignment fails")
#     else:
#         print("          Background subtraction fails")
#         massflux.f_particle_plume_alignment = False
#     return massflux
