#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:11:59 2022.

@author: Manu Goudar
"""

import numpy as np
import pandas as pd
from .ModuleDataContainers import DataContainer
from .ModEE_Simulation import Simulation3d
from .ModEE_TransactionalLines import create_tlines_remove_background

try:
    from .functions import get_velocity
except ImportError:
    print("Cython function is not compiled well")


def particles_height_at_tlines(coords, ff, weight=True):
    """
    Finds all particles around given coordinates.

    Parameters
    ----------
    coords : Float array type
        Description of parameter `coords of a transaction line`.
    ff : Pandas dataframe
        Description of parameter `ff`.
    weight : bool
        Description of parameter should use weights or not. Default is False.

    Returns
    -------
    type
        height of the transaction line.

    """

    # for each line find the center and remove > 1 particles
    ll = coords[coords.shape[0] // 2]
    f2 = ff[np.sqrt((ff.lat.values - ll[0]) ** 2 + (ff.lon.values - ll[1]) ** 2) < 1]
    data_line = pd.DataFrame()
    for ll in coords:
        _tmpdata = f2[np.sqrt((f2.lat.values - ll[0]) ** 2 + (f2.lon.values - ll[1]) ** 2) < 0.0025]
        data_line = pd.concat((data_line, _tmpdata), axis=0, ignore_index=True)
    if weight:
        return (
            data_line.height.values,
            (data_line.height * data_line.wts).sum() / data_line.wts.sum(),
        )
    else:
        if data_line.empty:
            return 0, np.nan
        else:
            return data_line.height.values, np.mean(data_line.height.values)


def get_plumeheightfromparticles(massflux, particles, emisname):
    """_summary_

    Args:
        firecontainer (_type_): _description_
        plumecontainer (_type_): _description_
        transform (_type_): _description_
        particles (_type_): _description_
        flow (_type_): _description_
    """
    # get height of the transaction lines
    lines_aligned = 0
    _ed = min(20, len(massflux.tlines))
    for _ln in massflux.tlines[:_ed]:
        # If the difference between two sides is not high then continue
        if _ln.flag_backgroundremovalsuccess:
            emis = DataContainer()
            # Compute height of to compute velocity field
            coords = _ln.final_coords_deg.copy()
            all_hts, ht_tline = particles_height_at_tlines(coords, particles[particles.vert_id == 1], False)
            setattr(emis, "height", ht_tline)
            if np.isnan(emis.height):
                setattr(emis, "flag_lineparticleplume_aligned", False)
            else:
                setattr(emis, "flag_lineparticleplume_aligned", True)
                lines_aligned += 1
            setattr(_ln, "emission_"+emisname, emis)
    if lines_aligned > 4:
        massflux.__setattr__("flag_particleplume_aligned", True)
    else:
        massflux.__setattr__("flag_particleplume_aligned", False)


def get_velocitymag(coords, height_tline, dir_vect, flow):
    """_summary_

    Args:
        coords (_type_): _description_
        height_tline (_type_): _description_
        dir_vect (_type_): _description_
        flow (_type_): _description_

    Returns:
        _type_: _description_
    """
    _nx = coords.shape[0]
    _crds = np.zeros((_nx, 3))
    _crds[:, :2] = coords
    _crds[:, 2] = height_tline
    idx = flow.emission_idx[0]
    vl = np.zeros_like(_crds)
    vl, limits = get_velocity(_crds, flow.lat, flow.lon, flow.z[idx, :, :, :],
                              flow.u[idx, :, :, :], flow.v[idx, :, :, :], flow.w[idx, :, :, :])
    if len(flow.emission_idx) > 1:
        vl1 = np.zeros_like(_crds)
        idx = flow.emission_idx[1]
        vl1, limits = get_velocity(_crds, flow.lat, flow.lon, flow.z[idx, :, :, :],
                                   flow.u[idx, :, :, :], flow.v[idx, :, :, :], flow.w[idx, :, :, :])
        vl = vl * flow.emission_fact[0] + vl1 * flow.emission_fact[1]
    vel_mag = np.zeros((_nx))
    vel_mag[:] = vl[:, 0] * dir_vect[0] + vl[:, 1] * dir_vect[1]

    # Check if mean magnitude of velocity is > 2m/s
    flag = np.nanmean(np.sqrt(vl[:, 0] ** 2 + vl[:, 1] ** 2)) < 2
    return flag, vel_mag, vl[:,0], vl[:,1]


def line_velocityemission(_ln, flow, fact_emis, emisname):
    """Get emission and velocity for a line.


    Parameters
    ----------
    _ln : Class
        Class containing line data
    flow : Class
        Containing flow details
    fact_emis : Float
        constant factor to multiply emission [line integration + molarmass]
    emisname : String
        Name of the emission class
    """

    emis = getattr(_ln, emisname)
    flag, vel_mag, u, v = get_velocitymag(_ln.final_coords_deg.copy(), emis.height, _ln.dir_vect, flow)
    line_emis = fact_emis * vel_mag
    # data if the velocity is > 2m/s
    setattr(emis, "vel_mag" , vel_mag.copy())
    setattr(emis, "flag_velocitylessthan2", flag)
    if not emis.flag_velocitylessthan2:
        setattr(emis, "emission_line", line_emis)
        setattr(emis, "emission", np.nansum(line_emis))
    return flag



def emission_estimates_varying_ht(massflux, flow, emisname, molarmass):
    # get height of the transaction lines
    _ed = min(20, len(massflux.tlines))
    flag2 = 0
    flag3 = 0
    factor_const = molarmass * massflux.line_spacing_km * 1000  # convert to mts
    for _ln in massflux.tlines[:_ed]:
        emis = getattr(_ln, "emission_"+emisname)
        # If the difference between two sides is not high then continue
        if _ln.flag_backgroundremovalsuccess and emis.flag_lineparticleplume_aligned:
            # To compute emissions: create a factoer
            fact_emis = _ln.final_co * factor_const
            # compute velocity at plume height from Lagrangian simulation
            flag1 = line_velocityemission(_ln, flow, fact_emis, "emission_"+emisname)
            flag2 += 1
            flag3 += flag1
    if (flag2 - flag3) / flag2 < 0.5:
        massflux.__setattr__("flag_velocitylessthan2", True)
        print("         Velocity < 2m/s")
    else:
        massflux.__setattr__("flag_velocitylessthan2", False)


# varying plume height
def crosssectionalflux_varying(_key, params, satellitedata, plumedata, sources):
    # update the transform
    params.transform.update_origin(satellitedata.source)
    
    # Create transaction lines and remove background    
    massflux = create_tlines_remove_background(satellitedata, plumedata, params.transform)
    # then compute the lagrangian simulations and extract height
    if massflux.flag_goodplume:
        sim3d = Simulation3d(satellitedata.source, params.transform, params.estimateemission,
                             sources, satellitedata.measurement_time)
        print("            Simulating particles")
        sim3d.run()
        print("                   ...Done")
        simname = (params.estimateemission.particledir + _key)
        sim3d.save(simname)
        
        # compute varying plume height and its emissions
        particle_data = sim3d.get_particle_data()
        get_plumeheightfromparticles(massflux, particle_data, params.estimateemission.emisname)
        if massflux.flag_particleplume_aligned:
            emission_estimates_varying_ht(massflux, sim3d.flow,
                                          params.estimateemission.emisname,
                                          params.estimateemission.molarmass)
        else:
            print("         Plume alignment fails")
    return massflux
