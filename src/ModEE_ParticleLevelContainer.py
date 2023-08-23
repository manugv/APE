#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 8 2020.

@author: Manu Goudar
"""

# -*- coding: utf-8 -*-
import numpy as np


# =============================================================================
# Particles class for 3d
# =============================================================================
class LevelData:
    def __init__(
        self, lvl, loc_source, mass, dt, save_dt, starttime, initial_particle_id, ids, flag_lims
    ):
        """
        Initialize number of levels and their length in
        seconds based on total simulation time
        """
        self.lvl_id = lvl[0]
        self.lvl_time = lvl[1]
        # Start time of the level
        self.start_time = starttime

        # Time saving dt
        self.save_dt = save_dt

        # Number of sources in old level or input
        no_sources = np.size(loc_source[:, 0])

        # number of iterations on this level
        self.itrs = np.int_(self.lvl_time / dt) + 1

        # total number of points where data is saved in this level
        _t1 = self.lvl_time / save_dt
        _flag = _t1 == np.int_(_t1)
        if _flag:
            self.save_pts = np.int_(self.lvl_time / save_dt) + 1
        else:
            self.save_pts = np.int_(self.lvl_time / save_dt) + 2

        # number of particles in this level
        if self.lvl_id == 0:
            self.no_particles = no_sources
        else:
            ss = flag_lims.sum()
            self.no_particles = ss * 2

        # position array to store data (particles, iterations, dimensions)
        self.pos = np.zeros((self.no_particles, self.save_pts, 3))

        # Time at which the pos data is stored
        self.time = np.zeros((self.save_pts))
        if _flag:
            self.time[:] = np.arange(0, self.lvl_time + 1, self.save_dt)
        else:
            self.time[:-1] = np.arange(0, self.lvl_time + 1, self.save_dt)
            self.time[-1] = self.lvl_time
        # Mass of the particless
        self.mass = np.zeros((self.no_particles))

        # Number of ids
        self.ids = np.zeros((self.no_particles, 3), dtype=np.int32)

        # Flag to define if particles are out of limits
        self.limits = np.zeros((self.no_particles), dtype=np.bool_)

        # Initialize sources and masses
        if self.lvl_id == 0:
            for i in range(self.no_particles):
                self.pos[i, 0, :] = loc_source[i]
                self.mass[i] = mass[i]
                self.ids[i, 0] = initial_particle_id + i + 1
                self.ids[i, 1:] = ids[i, :]
                self.limits[i] = True
        else:
            for i in range(0, self.no_particles, 2):
                ii = np.int_(i / 2)
                if flag_lims[ii]:
                    self.pos[i, 0, :] = loc_source[ii]
                    self.pos[i + 1, 0, :] = loc_source[ii] + np.random.rand() / 100
                    self.mass[i] = mass[ii] / 2
                    self.mass[i + 1] = mass[ii] / 2
                    self.ids[i, 0] = ids[ii, 0]  # Put old id value
                    self.ids[i, 1:] = ids[ii, 1:]  # Put old id value
                    # Initialize new id for new particle
                    self.ids[i + 1, 0] = initial_particle_id + ii + 1
                    self.ids[i + 1, 1:] = ids[ii, 1:]
                    self.limits[i : i + 2] = True


# =============================================================================
# Particles class for 2d
# =============================================================================
class Tracers2d:
    def __init__(self, dims, nos, loc, i):
        self.pos = np.zeros((dims, nos))
        #        self.vel = np.zeros((dims, nos))
        self.pos[:, 0] = loc
        self.id = i
