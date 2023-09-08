#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:17:16 2020
Get vertical velocity and z-coordinates
@author: Manu Goudar
"""

import pandas as pd
import numpy as np
from importlib.resources import files


"""
k is used to denote levels in atmosphere
Note that k-1/2 is above k and k+1/2 is below k.
Vertical distance is from atmosphere top to ground.
lt : is to denote latitude
ln : longitude
t : for time
levels are defined from 1-137 and if all levels
are present then k goes from 0-136.

"""

R_d = 461.5
R_v = 287.058


def get_Rm_Tv(q, Temp):
    """
    Factor to compute R_m and T_v
    This computes (1+(R_v/R_d)q)
    Input:
        q : Specific humidity
        Temp : Temperature same shape as q
    Output:
        Rm : Same shape as q
        Tv : Virtual temperature same shape as q
    """
    const = R_v / R_d - 1
    fact = np.zeros(q.shape)
    Tv = np.zeros(q.shape)
    fact = 1 + const * q
    Rm = fact * R_d
    Tv = np.multiply(fact, Temp)
    del fact
    return Rm, Tv


def get_pres(c, ps):
    if c.b.size == 1:
        return c.a_Pa + c.b * ps
    else:
        return c.a_Pa.values + c.b.values * ps


# Compute pressure
def get_full_pressure(lvl, ps, ab):
    """
    Input:
        lvl - the full level at which pressure needs to computed
    output:
        pf - pressure at full level
    """
    # Initialize arrays
    sh = ps.shape
    lv = len(lvl)
    pf = np.zeros((sh[0], lv, sh[1], sh[2]))
    phf = np.zeros((sh[0], lv + 1, sh[1], sh[2]))

    # Compute the pressure
    for l1 in range(lv + 1):
        if l1 == 0:
            phf[:, l1, :, :] = get_pres(ab.iloc[lvl[0] - 1], ps)
        if l1 > 0:
            phf[:, l1, :, :] = get_pres(ab.iloc[lvl[l1 - 1]], ps)

    pf = 0.5 * (phf[:, :-1, :, :] + phf[:, 1:, :, :])
    return pf, phf


def get_ifg(lat):
    return 9.780327 * (
        1 + 0.0053024 * np.power(np.sin(lat), 2) - 0.0000058 * np.power(np.sin(2.0 * lat), 2)
    )


def get_geo_height(lvl, phf, Tv, geop_s):
    """
    phi_k = phi_k+1/2 + alpha*Rd*Tv
    RHS Term 1: phi_k+1/2 = phi_s+sum[Rd*Tv*ln(p{j+1/2}/p_{j_1/2})]_j=k+1,N
    RHS Term-2: alpha*Rd*Tv
    """
    lvl_r = lvl[::-1]
    maxlvl = 137
    # Ratio of pressure used to compute RHS TERMs 1 and 2
    pres_ratio = phf[:, 1:, :, :] / phf[:, :-1, :, :]
    phi = np.zeros(Tv.shape)
    ss = np.zeros(geop_s.shape)
    # RHS1
    for ll in lvl_r:
        k = np.argwhere(ll == lvl)[0][0]
        if ll == maxlvl:
            phi[:, k, :, :] = geop_s
        else:
            ss = ss + R_d * Tv[:, k + 1, :, :] * np.log(pres_ratio[:, k + 1, :, :])
            phi[:, k, :, :] = geop_s + ss
    # RHS Term-2: alpha*Rd*Tv
    alpha = np.zeros(geop_s.shape)
    for k in range(len(lvl)):
        ll = lvl[k]
        # RHS TERM 2 computation
        # RHS Term-2: alpha*Rd*Tv
        if ll == 1:
            alpha[:, :, :] = np.log(2)
        else:
            deltapk = phf[:, k + 1, :, :] - phf[:, k, :, :]
            alpha[:, :, :] = 1 - ((phf[:, k, :, :] / deltapk) * np.log(pres_ratio[:, k, :, :]))
        # Add RHS 2
        phi[:, k, :, :] = phi[:, k, :, :] + R_d * np.multiply(alpha[:, :, :], Tv[:, k, :, :])
    return phi / 9.80665


def get_vertical_coord_vel(data):
    table_level137s = files("APE.data").joinpath("table_lvl137.csv")
    params_lvl = pd.read_csv(table_level137s, index_col=0)

    # get factor that is used to compute R_m and T_v
    Rm, Tv = get_Rm_Tv(data["q"], data["t"])

    # v = (Rm.T.w)/(p.g)
    vel_w = np.multiply((np.multiply(Rm, data["t"])), data["w"])  # Rm.T.w

    # compute pressure
    pf, phf = get_full_pressure(data["level"], data["pres"], params_lvl)

    # v = (Rm.T.w)/(p.g)
    vel_w = np.divide(vel_w, pf)

    # Compute gravity g = IFG + FAC
    ifg = get_ifg(data["latitude"])
    # FAC = phi_k/g_0
    # phi_k = phi_k+1/2 + alpha*Rd*Tv
    # alpha*Rd*Tv
    zh = get_geo_height(data["level"], phf, Tv, data["geop"])
    fac = -3.086 * 1e-6 * zh
    # Total gravity
    gravity = np.zeros(zh.shape)
    for l1 in range(len(data["latitude"])):
        gravity[:, :, l1, :] = ifg[l1] + fac[:, :, l1, :]
    vel_w = np.divide(vel_w, gravity)
    data["zh"] = zh
    data["velw"] = vel_w
    return zh, vel_w
