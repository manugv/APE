#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:11:59 2022.

@author: Manu Goudar
"""

from .ModEE_MassFlux import (compute_medial_line,  get_plumepoints_slope,
                             get_tlines, particles_height_at_tlines)
import functions as func
from .ModuleDataContainers import DataContainer
import numpy as np


def get_velocity_mag(coords, height_tline, dir_vect, flow):
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
    vl, limits = func.get_velocity(
        _crds,
        flow.lat,
        flow.lon,
        flow.z[idx, :, :, :],
        flow.u[idx, :, :, :],
        flow.v[idx, :, :, :],
        flow.w[idx, :, :, :],
    )
    if len(flow.emission_idx) > 1:
        vl1 = np.zeros_like(_crds)
        idx = flow.emission_idx[1]
        vl1, limits = func.get_velocity(
            _crds,
            flow.lat,
            flow.lon,
            flow.z[idx, :, :, :],
            flow.u[idx, :, :, :],
            flow.v[idx, :, :, :],
            flow.w[idx, :, :, :],
        )
        vl = vl * flow.emission_fact[0] + vl1 * flow.emission_fact[1]
    vel_mag = np.zeros((_nx))
    vel_mag[:] = vl[:, 0] * dir_vect[0] + vl[:, 1] * dir_vect[1]

    # Check if mean magnitude of velocity is > 2m/s
    flag = np.nanmean(np.sqrt(vl[:, 0] ** 2 + vl[:, 1] ** 2)) > 2
    return flag, vel_mag


def vel_and_emis(_ln, plume_ht, flow, fact_emis, suffix=""):
    """_summary_

    Args:
        _ln (_type_): _description_
        plume_ht (_type_): _description_
        flow (_type_): _description_
        fact_emis (_type_): _description_
        _idx (_type_): _description_
        suffix (str, optional): _description_. Defaults to "".
    """
    flag, vel_mag = get_velocity_mag(_ln.final_coords_deg.copy(), plume_ht, _ln.dir_vect, flow)
    line_emis = fact_emis * vel_mag
    # data if the velocity is > 2m/s
    _ln.__setattr__("vel_mag_" + suffix, vel_mag.copy())
    _ln.__setattr__("emission_line_" + suffix, line_emis)
    _ln.__setattr__("emission_" + suffix, np.nansum(line_emis))
    return flag


def create_tlines_remove_background(satellitedata, plumecontainer, transform):
    """_summary_

    Args:
        firecontainer (_type_): _description_
        plumecontainer (_type_): _description_
        transform (_type_): _description_
        particles (_type_): _description_
        flow (_type_): _description_
    """

    massflux = DataContainer()

    # Plume and transaction lines
    _pline = compute_medial_line(
        satellitedata.lat_nodes,
        satellitedata.lon_nodes,
        plumecontainer.plumemask,
        satellitedata.source,
        transform,
    )
    massflux.__setattr__("fitted_plumeline", _pline)

    # Create line
    plume_ds = 2.5

    # extract points along plume line at plume_ds
    dataline = get_plumepoints_slope(
        massflux.fitted_plumeline, transform, ds=plume_ds, plume_len_km=50
    )
    # this is done as dataline is a dict and massflux is a container
    for key, value in dataline.items():
        massflux.__setattr__(key, value)

    # Transform lat-lon of satellite
    xx, yy = transform.latlon2xykm(satellitedata.lat, satellitedata.lon)
    satellitedata.__setattr__("xkm", xx)
    satellitedata.__setattr__("ykm", yy)

    # Define transaction lines
    tlines = get_tlines(
        massflux,
        satellitedata.co_column_corr,
        satellitedata.xkm,
        satellitedata.ykm,
        transform,
        80,
    )
    massflux.__setattr__("tlines", tlines)

    # Filter based on background
    _ed = min(20, len(tlines))
    lines_total = 0
    for _ln in tlines[:_ed]:
        lines_total += _ln.f_background_good
    if lines_total > 5:
        massflux.__setattr__("f_good_plume_bs", True)
    else:
        massflux.__setattr__("f_good_plume_bs", False)

    return massflux


def get_constant_plume_height(injectionht, tlines, topology):
    """get_constant_plume_height _summary_

    Args:
        viirsdata (_type_): _description_
        tlines (_type_): _description_
        topology (_type_): _description_
    """
    injht = (injectionht[injectionht > 0]).mean()
    # Set a constant plume height variable for each line
    _ed = min(20, len(tlines))
    for _ln in tlines[:_ed]:
        if _ln.f_background_good:
            ht = topology.get_topology(_ln.final_coords_deg.copy())
            _ln.__setattr__("const_plume_ht", ht.mean() + injht)


def get_varying_plume_height(massflux, particles):
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
        if _ln.f_background_good:
            # Compute height of to compute velocity field
            coords = _ln.final_coords_deg.copy()
            all_hts, ht_tline = particles_height_at_tlines(
                coords, particles[particles.vert_id == 1], False
            )
            _ln.__setattr__("varying_plume_ht", ht_tline)

            all_hts, ht_tline = particles_height_at_tlines(
                coords, particles[particles.vert_id == 0], False
            )
            _ln.__setattr__("varying_plume_ht_m500", ht_tline)

            all_hts, ht_tline = particles_height_at_tlines(
                coords, particles[particles.vert_id == 2], False
            )
            _ln.__setattr__("varying_plume_ht_p500", ht_tline)
            if np.isnan(_ln.varying_plume_ht):
                _ln.__setattr__("f_lineparticle_plume_alignment", False)
            else:
                _ln.__setattr__("f_lineparticle_plume_alignment", True)
                lines_aligned += 1
    if lines_aligned > 4:
        massflux.__setattr__("f_particle_plume_alignment", True)
    else:
        massflux.__setattr__("f_particle_plume_alignment", False)


def emission_estimates_varying_ht(massflux, flow):
    # get height of the transaction lines
    _ed = min(20, len(massflux.tlines))
    flag2 = 0
    flag3 = 0
    for _ln in massflux.tlines[:_ed]:
        # If the difference between two sides is not high then continue
        if _ln.f_background_good and _ln.f_lineparticle_plume_alignment:
            # To compute emissions: create a factoer
            fact_emis = _ln.final_co * 28.01 * 0.001 * _ln.ds
            # compute velocity at plume height from Lagrangian simulation
            flag1 = vel_and_emis(_ln, _ln.varying_plume_ht, flow, fact_emis, "lag")
            flag2 += 1
            flag3 += flag1
            _ln.__setattr__("f_veldiff_varinght", flag1)
            fl = vel_and_emis(_ln, _ln.varying_plume_ht_m500, flow, fact_emis, "lag_m500")
            fl = vel_and_emis(_ln, _ln.varying_plume_ht_p500, flow, fact_emis, "lag_p500")
    if flag3 / flag2 < 0.5:
        print("         Velocity < 2m/s")


def emission_estimates_const_ht(massflux, flow):
    # the transaction lines
    _ed = min(20, len(massflux.tlines))
    for _ln in massflux.tlines[:_ed]:
        # If the difference between two sides is not high then continue
        if _ln.f_background_good:
            # To compute emissions create a factor
            fact_emis = _ln.final_co * 28.01 * 0.001 * _ln.ds
            # compute velocity at constant injection height
            flag1 = vel_and_emis(_ln, _ln.const_plume_ht, flow, fact_emis, "inj")
            _ln.__setattr__("f_veldiff_constht", flag1)
            fl = vel_and_emis(_ln, _ln.const_plume_ht - 500, flow, fact_emis, "inj_m500")
            fl = vel_and_emis(_ln, _ln.const_plume_ht + 500, flow, fact_emis, "inj_p500")
