#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:11:59 2022.

@author: Manu Goudar
"""

from .ModEE_Simulation import Simulation3d
from .ModEE_EmissionEstimate import (
    create_tlines_remove_background,
    get_constant_plume_height,
    get_varying_plume_height,
    emission_estimates_varying_ht,
    emission_estimates_const_ht,
)
from .ModuleTransform import TransformCoords


def compute_emissions(day, globalparams, fire_satdata, fire_viirs, plumecontainer):
    # Create a transform
    transform = TransformCoords(fire_satdata.source)

    # Create transaction lines and remove background

    massflux = create_tlines_remove_background(fire_satdata, plumecontainer, transform)

    # IF the plume was good after background subtraction
    # then compute the lagrangian simulations and extract height
    if massflux.f_good_plume_bs:
        sim3d = Simulation3d(fire_satdata.source, transform, globalparams,
                             fire_viirs, fire_satdata.measurement_time)
        sim3d.run()
        simname = (
            globalparams.output_particlefile_prefix
            + day.strftime("%Y_%m_%d")
            + "_"
            + fire_satdata.fire_name
        )

        sim3d.save(simname)

        # compute constant injection height and emissions
        get_constant_plume_height(fire_viirs.injection_height, massflux.tlines, sim3d.topology)
        emission_estimates_const_ht(massflux, sim3d.flow)
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
