#!/usr/bin/env python
# coding: utf-8
"""
Created on July 3 13:00 2020
Parameter Containers for different variables
@author: Manu Goudar
"""

from numpy import array
from yaml import load, FullLoader
from datetime import date, datetime, timedelta
from ModuleDataContainers import Flowinfo, SimulationTime, Dispersion
from ModuleDataContainers import ParticleSplitting, MultipleParticleRelease
from ModuleDataContainers import Traj2ConcInfo
from ModEE_CreateSources import SourcesInit

try:
    from ModuleTransform import TransformCoords
except ImportError:
    print("Module loading failed")


class InputParameters:
    def __init__(self, filename):
        fl = open(filename)
        _f = load(fl, FullLoader)

        # Fire parameters
        if _f["Source"] == "Fire":
            _fire = _f["Fire"]
            self.viirsdir = _fire["viirs_dir"]
            self.gfasdir = _fire["gfas_dir"]
            self.gfasfile = self.gfasdir + _fire["gfas_file"]
            self.roi = array(_fire["roi"])
            self.roi_name = _fire["roi_name"]

            # date specification
            self.days = self.get_days(_fire)

            self.transform = TransformCoords([self.roi[2], self.roi[0]])
            self.o_filename = self.roi_name + "_"

        # Industrial sourcess
        if _f["Source"] == "Industrial":
            _ind = _f["Industrial"]
            self.ind_source = _ind["Source"]
            self.ind_source_name = _ind["Source_name"]

            # date specification
            self.days = self.get_days(_ind)

            self.transform = TransformCoords([self.ind_source[0], self.ind_source[1]])
            self.o_filename = (
                self.ind_source_name + "_"
            )

        # Satellite input and output directories
        _dirs = _f["Directories"]
        self.satellite_dir = _dirs["satellite_dir"]
        self.output_dir = _dirs["output_dir"]
        self.output_file_prefix = self.output_dir + self.o_filename
        self.output_particles_dir = _dirs["output_particles_dir"]
        self.output_particlefile_prefix = self.output_particles_dir + self.o_filename

        # Flow
        self.param_flowinfo = Flowinfo(_f["Flow"])

        # Simulation time
        self.param_simtime = SimulationTime(_f["SimulationTime"])

        # Dispersion parameters
        self.param_disp = Dispersion(_f["Dispersion"])

        # Particle Release
        self.param_partrelease = MultipleParticleRelease(
            _f["MultipleParticleRelease"], self.param_simtime.sim_time_sec
        )

        # Particle Splitting
        self.param_partsplit = ParticleSplitting(_f["ParticleSplitting"])

        # Trajectory data to concentration
        self.param_traj2conc = Traj2ConcInfo(_f["TrajectoriesToConcentrations"])

        # sources
        self.param_source = SourcesInit(_f["Sources"])
        fl.close()

    def get_days(self, _date):
        self.startdate = datetime.strptime(_date["startdate"], "%Y-%m-%d").date()
        self.enddate = datetime.strptime(_date["enddate"], "%Y-%m-%d").date()
        if self.startdate > self.enddate:
            print("Start date is smaller then end date")
            exit()
        days = []
        stdate = self.startdate
        while stdate <= self.enddate:
            days.append(stdate)
            stdate += timedelta(days=1)
        return days
