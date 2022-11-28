#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:40:40 2021
Sepup class for 3d
@author: Manu Goudar
"""

from ModEE_Flow import FlowData
from ModEE_Trajectory2Concentration import Traj2Conc
from scipy.interpolate import RegularGridInterpolator as RGI
import numpy as np
from copy import copy


class InitializeSim:
    def __init__(self, src, transform, param, firedata, firetime):

        # Simulation type
        self.simtype = copy(param.param_disp.simtype)
        # Define simulation parameters
        self.sim_dt = copy(param.param_simtime.dt)
        self.sim_time_sec = copy(param.param_simtime.sim_time_sec)
        self.save_dt = copy(param.param_simtime.savedt)
        # create simulation time
        self.start_time = param.param_simtime.set_start_time(firetime)

        # Create Transform
        self.origin = src
        self.transform = transform

        # Read velocity data
        self.flow = FlowData(
            param.param_flowinfo, param.param_disp, self.start_time, self.sim_time_sec
        )

        # Initialize the class to get surface topology
        self.topology = SurfaceTopology(self.flow.lat, self.flow.lon, self.flow.z[0, :, :, 0])

        # create fire sources for a fire
        # Define actual height of sources
        if self.simtype == "3d":
            self.param_source = param.param_source.initialize_fire_src(
                firedata.latitude, firedata.longitude, firedata.injection_height, self.topology
            )

        # Particles splitting
        self.particlesplit = param.param_partsplit.split
        if self.particlesplit:
            self.particle_splittime = param.param_partsplit.splittime

        # Particle releases
        self.no_part_releases = param.param_partrelease.no_releases
        self.dt_particle_release = param.param_partrelease.deltatime

        # Initialize Trajectories2concentrations if true
        self.param_traj2concflag = copy(param.param_traj2conc.flag)
        if param.param_traj2conc.flag:
            # intialize class
            self.conc = Traj2Conc(
                param.param_traj2conc,
                param.param_sim.roi,
                param.param_simtime.sim_time_sec,
                self.topology.get_topology,
            )

        # Running simulation time in seconds
        self.run_time_sec = 0
        if param.param_disp.model == "Forward":
            # this is time in minutes that needs to be tracked
            self.start_interp_sec = self.start_time.minute * 60 + self.start_time.second
            # Define interpolation subroutine
            self.get_interp_id = self.get_interp_id_forward
            # Define postion update subroutine
            self.update_position = self.update_position_forward
        elif param.param_disp.model == "Backward":
            # time should go in reverse. -1 as array starts at 0
            # TODO: Edit start to end time for backward simulations
            tmp = self.start_time.minute
            if tmp > 0:
                self.start_interp_sec = (60 - tmp) * 60
            else:
                self.start_interp_sec = 0
            # Define interpolation subroutine
            self.get_interp_id = self.get_interp_id_backward
            # Define postion update subroutine
            self.update_position = self.update_position_backward

        # Keep track of number of particles in simulation
        self.particles_nos = 0

    def update_position_forward(self, dt, xmts, vel):
        """
        Update position for forward dispersion
        """
        xmts += dt * vel

    def update_position_backward(self, dt, xmts, vel):
        """
        Update position for backward dispersion
        """
        xmts -= dt * vel

    def get_interp_id_forward(self, time_sec):
        """
        Get id for forward interpolation
        """
        # compute seconds for interpolation
        info = True
        fact = np.zeros((2), dtype=np.float64)
        tot_sec = self.start_interp_sec + time_sec
        idx = np.int_(tot_sec / 3600)
        if tot_sec % 3600 == 0:
            info = False
        fact[0] = tot_sec / 3600 - idx
        fact[1] = 1 - fact[0]
        return idx, fact, info

    def get_interp_id_backward(self, time_sec):
        """
        Get id for backward interpolation
        """
        # Update interpolation time in hrs
        tot_sec = self.start_interp_sec + time_sec
        idx = -np.int_(tot_sec / 3600) - 1
        return idx

    def check_simulation_time_bounds(self):
        """
        TODO
        """
        pass

    def check_source_loc_bounds(self):
        """
        TODO
        """
        # check lon, lat, based on input grid
        # Check height based on available velocity values
        pass


#    change_sim_hrs(self, hrs):
#        self.sim_hrs = hrs
#    def change_dt(self, dt):
#        self.dt = dt
#
#    def change_start_time(self, st_time):
#        self.start_time = st_time
#
#    def change_simulation_time(self, st_time):
#        self.start_time = st_time


class SurfaceTopology:
    def __init__(self, lat, lon, z):
        self.lat = lat
        self.lon = lon
        self.z = z
        self.fh = RGI((self.lat, self.lon), self.z, bounds_error=False)

    def get_topology(self, loc):
        return self.fh(loc)
