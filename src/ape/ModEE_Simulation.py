#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:48:10 2020
Simulation class for 3d
@author: Manu Goudar
"""

import numpy as np
import pandas as pd
from ModEE_Setup import InitializeSim

# from ModuleInterpolation import Interpolate_3d
from ModEE_ParticleLevelContainer import LevelData
import ModEE_SaveSimulation as mss
import functions as func


class Simulation3d(InitializeSim):
    def __init__(self, src, transform, param, firedata, firetime):
        super().__init__(src, transform, param, firedata, firetime)
        # initialize levels (splitting of particles) and particles
        self.all_levels = []
        self.particles = []
        # Keep track of number of particles in simulation
        self.particles_nos = 0
        self.particle_data = 0

    def __get_levels(self, init_time):
        tt = []
        lvl = []
        tot_time = init_time
        k = 0
        # If particle split flag is on
        if self.particlesplit is True:
            while tot_time < self.sim_time_sec:
                lvl.append(k)
                t1 = self.particle_splittime * np.power(2, k)
                tot_time += t1
                if tot_time > self.sim_time_sec:
                    excess = tot_time - self.sim_time_sec
                    t1 = t1 - excess
                tt.append(t1)
                k += 1
        else:  # Particle split flag is off
            lvl.append(0)
            tt.append(self.sim_time_sec - tot_time)
        return lvl, tt

    def compute_trajectories(self, particles, info_conc=None):
        # Define variables (x, mts, vel) for one time instance
        xmts = np.zeros((particles.no_particles, 3))
        _v1 = np.zeros((particles.no_particles, 3))
        vel = np.zeros((particles.no_particles, 3))
        x = np.zeros((particles.no_particles, 3))
        # vel_purt = np.zeros((3, particles.no_particles), order='F')
        # convert first 2 coordinates to mts
        xmts[:, 0], xmts[:, 1] = self.transform.latlon2xymts(
            particles.pos[:, 0, 0], particles.pos[:, 0, 1]
        )
        xmts[:, 2] = particles.pos[:, 0, 2]
        # Push position values into x
        # for p in range(particles.no_particles):
        x[:, :] = particles.pos[:, 0, :]

        # Set local simulation time to zero
        itr_time = 0
        # Variable helps to save data at pre-defined intervals
        k = 1
        for i in range(particles.itrs - 1):
            # Update time for interpolation based on old itr time
            # Get the index of the time for interpolation
            # print('time', particles.start_time + itr_time)
            idx, time_fact, _ifo = self.get_interp_id(particles.start_time + itr_time)

            # Temporal and spatial interpolation
            vel[:, :], particles.limits[:] = func.get_velocity(
                x,
                self.flow.lat,
                self.flow.lon,
                self.flow.z[idx, :, :, :],
                self.flow.u[idx, :, :, :],
                self.flow.v[idx, :, :, :],
                self.flow.w[idx, :, :, :],
            )
            if _ifo:
                _v1[:, :], particles.limits[:] = func.get_velocity(
                    x,
                    self.flow.lat,
                    self.flow.lon,
                    self.flow.z[idx + 1, :, :, :],
                    self.flow.u[idx + 1, :, :, :],
                    self.flow.v[idx + 1, :, :, :],
                    self.flow.w[idx + 1, :, :, :],
                )
                vel = _v1 * time_fact[0] + vel * time_fact[1]

            # update position at time t+1 from velocity at t
            self.update_position(self.sim_dt, xmts[:, :], vel[:, :])

            # Convert position in mts at t+1 to lat_lon
            x[:, 0], x[:, 1] = self.transform.xymts2latlon(xmts[:, 0], xmts[:, 1])
            x[:, 2] = xmts[:, 2]
            # Update iteration time for this itr
            itr_time += self.sim_dt
            # Store data for all particles
            if itr_time == particles.time[k]:
                particles.pos[:, k, :] = x[:, :]
                k += 1

    def run(self):
        """
        source is a point in (lat, lon)
        data is a container for data
        start is starting hour in int
        end is end hour in int
        dt in seconds and should be <= 60seconds
        nos is number of points per hour
        """
        # This loop is for particle release at pre-defined time
        for rls in range(self.no_part_releases):
            # Compute the start time of the release
            self.run_time_sec = rls * self.dt_particle_release

            # Add actual height to relative source height in first step
            source_loc, mass, ids = self.param_source.get_data()
            flag_lims = np.array([True])
            _release = []

            # compute number of levels and time for each level
            levels, lvl_times = self.__get_levels(self.run_time_sec)
            self.all_levels.append([levels, lvl_times, self.run_time_sec])
            # check if concentrations needs to be computed decide
            if self.param_traj2concflag is True:
                info_conc = self.conc.init_id(self.run_time_sec)
            # print("Release and time in secs:", rls, "  and  ", self.run_time_sec, "sec")
            # Compute particle trajectory over each level
            for lvl in zip(levels, lvl_times):
                # Break out of the loop if all particles are out of domain
                if sum(flag_lims) <= 0:
                    break
                # initialize a level
                p = LevelData(
                    lvl,
                    source_loc,
                    mass,
                    self.sim_dt,
                    self.save_dt,
                    self.run_time_sec,
                    self.particles_nos,
                    ids,
                    flag_lims,
                )

                # Update number of particles
                self.particles_nos = p.ids[-1, 0]

                # Compute Trajectories
                # if self.param_traj2conc.flag is True:
                #     self.__compute_trajectories(p, info_conc)
                # else:
                self.compute_trajectories(p)

                # Update simulation time and time of interpolation
                self.run_time_sec += lvl[1]
                # print("  Level:", lvl[0], " completed")
                # append level
                _release.append(p)
                # get sources, mass and ids for next level
                source_loc = p.pos[:, -1, :]
                mass = p.mass
                ids = p.ids
                flag_lims = p.limits
            # Append levels to release
            self.particles.append(_release)

        print("Completed trajectory simulation")
        # Map conc to tropomi pixels
        if self.param_traj2concflag:
            self.conc.to_tropomi(self.codata.avg_kernel)

    def get_particle_data(self):
        # Get data
        ff = pd.DataFrame()
        for i in range(len(self.particles)):
            rel = self.particles[i]
            f1 = pd.DataFrame()
            for j in range(len(rel)):
                lvl = rel[j]
                f1["lat"] = lvl.pos[:, -1, 0].copy()
                f1["lon"] = lvl.pos[:, -1, 1].copy()
                f1["height"] = lvl.pos[:, -1, 2].copy()
                f1["vert_id"] = lvl.ids[:, 2]
                f2 = f1.dropna()
                ff = pd.concat((ff, f2)) if not ff.empty else f2
        return ff.reset_index(drop=True)

    def save(self, simname, transform=False, scale=None):
        if scale is None:
            scale = [1, 1, 1]
        if transform:
            tf = self.transform.latlon2xymts
        else:
            tf = None

        if self.param_traj2concflag is True:
            mss.save_concentrationdata(
                self.particles,
                self.conc,
                scale=scale,
                transform=tf,
                topology=self.topology.get_topology,
                simulationname=simname,
            )
        else:
            mss.save_concentrationdata(
                self.particles,
                simulationname=simname,
                scale=scale,
                transform=tf,
                topology=self.topology.get_topology,
            )
