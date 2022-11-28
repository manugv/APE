#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:07:49 2022.

@author: Manu Goudar
"""

import numpy as np


class TransformCoords:

    def __init__(self, origin):
        self.rd = np.pi/180.0
        phi0 = origin[0]*self.rd
        self.ld0 = origin[1]*self.rd
        self.s_p0 = np.sin(phi0)
        self.c_p0 = np.cos(phi0)
        self.fact = 6371.0
        self.factmts = self.fact*1000

    def latlon2xykm(self, lat, lon):
        ld = lon*self.rd
        phi = lat*self.rd
        s_p = np.sin(phi)
        c_p = np.cos(phi)
        ll = ld - self.ld0
        c_l = np.cos(ll)
        s_l = np.sin(ll)
        c_pl = c_p*c_l
        w = np.sqrt(2.0/(np.maximum(1.0 + self.s_p0*s_p
                                    + self.c_p0*c_pl, 1.0e-10)))
        x = c_p*s_l*w
        y = (self.c_p0*s_p - self.s_p0*c_pl)*w
        return [x*self.fact, y*self.fact]

    def latlon2xymts(self, lat, lon):
        ld = lon*self.rd
        phi = lat*self.rd
        s_p = np.sin(phi)
        c_p = np.cos(phi)
        ll = ld - self.ld0
        c_l = np.cos(ll)
        s_l = np.sin(ll)
        c_pl = c_p*c_l
        w = np.sqrt(2.0/(np.maximum(1.0 + self.s_p0*s_p
                                    + self.c_p0*c_pl, 1.0e-10)))
        x = c_p*s_l*w
        y = (self.c_p0*s_p - self.s_p0*c_pl)*w
        return [x*self.factmts, y*self.factmts]

    def xykm2latlon(self, x1, y1):
        """docstring for la_xy2latlon"""
        x, y = x1/self.fact, y1/self.fact
        p = np.maximum(np.sqrt(x**2+y**2), 1.0e-10)
        c = 2.0*np.arcsin(p/2.0)
        s_c = np.sin(c)
        c_c = np.cos(c)
        phi = np.arcsin(c_c*self.s_p0 + y*s_c*self.c_p0/p)
        ld = self.ld0 + np.arctan2(x*s_c, (p*self.c_p0*c_c - y*self.s_p0*s_c))
        lat = phi/self.rd
        lon = ld/self.rd
        if isinstance(lat, np.ndarray):
            lat[lat > 90.0] -= 180.0
            lat[lat < -90.0] += 180.0
            lon[lon > 180.0] -= 360.0
            lon[lon < -180.0] += 360.0
        else:
            if abs(lat) > 90.0:
                if lat > 0:
                    lat = lat - 180.0
                else:
                    lat = lat + 180.0
            if abs(lon) > 180.0:
                if lon > 0:
                    lon = lon - 360.0
                else:
                    lon = lon + 360.0
        return [lat, lon]

    def xymts2latlon(self, x1, y1):
        """docstring for la_xy2latlon"""
        x, y = x1/self.factmts, y1/self.factmts
        p = np.maximum(np.sqrt(x**2+y**2), 1.0e-10)
        c = 2.0*np.arcsin(p/2.0)
        s_c = np.sin(c)
        c_c = np.cos(c)
        phi = np.arcsin(c_c*self.s_p0 + y*s_c*self.c_p0/p)
        ld = self.ld0 + np.arctan2(x*s_c, (p*self.c_p0*c_c - y*self.s_p0*s_c))
        lat = phi/self.rd
        lon = ld/self.rd
        if isinstance(lat, np.ndarray):
            lat[lat > 90.0] -= 180.0
            lat[lat < -90.0] += 180.0
            lon[lon > 180.0] -= 360.0
            lon[lon < -180.0] += 360.0
        else:
            if abs(lat) > 90.0:
                if lat > 0:
                    lat = lat - 180.0
                else:
                    lat = lat + 180.0
            if abs(lon) > 180.0:
                if lon > 0:
                    lon = lon - 360.0
                else:
                    lon = lon + 360.0
        return [lat, lon]
