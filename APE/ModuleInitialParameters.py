#!/usr/bin/env python
# coding: utf-8
"""
Created on July 3 13:00 2020
Parameter Containers for different variables
@author: Manu Goudar
"""

try:
    from numpy import array
    from yaml import safe_load
    from datetime import datetime, timedelta, date
    from .ModuleDataContainers import Flowinfo, SimulationTime, Dispersion
    from .ModuleDataContainers import ParticleSplitting, MultipleParticleRelease
    from .ModuleDataContainers import Traj2ConcInfo, DataContainer
    from .ModuleTransform import TransformCoords
    from .ModDataPrepare_SatelliteRead import get_filenames
except ImportError:
    print("Module loading failed while initializing parameters")


class InputParameters:
    """Class containing input parameters.

    Defines directories, types of sources,
    and all that is in the input yaml file.

    """

    def __init__(self, filename):
        # open file
        try:
            fl = open(filename)
        except FileNotFoundError:
            print("{filename} does not exist")
            return None

        # load yaml using load
        _f = safe_load(fl)

        # get days to run simulation
        self.days = self._get_days(_f)
        # Satellite and output directories
        self.satellite_dir = _f["SatelliteDir"]
        self.output_dir = _f["OutputDir"]
        self.sourcetype = _f["Source"]

        # Fire parameters
        if _f["Source"] == "Fire":
            self._init_fireparams(_f["Fire"])
            self.output_file_prefix = self.output_dir + self.file_prefix + "_"

        # Industrial sources
        if _f["Source"] == "Industrial":
            self._init_industrialparams(_f["Industrial"])
            self.output_file = self.output_dir + self.file_prefix

        # Get all satellite files in the folder and orbits
        self.sat_files = get_filenames(self.satellite_dir)

        # Plume detection should be done or not
        self.detectplume = _f["PlumeDetection"]["Flag"]

        # Emission estimation
        self.estimateemission = self._init_emissionestimation(_f["EmissionEstimation"])
        fl.close()

    def _init_emissionestimation(self, _em):
        """Initialize emission estimation variables.

        Parameters
        ----------
        _em : Dict
            Dict of emission parameters.

        Return
        --------
        emis: class of parameters.

        """
        emis = DataContainer()
        emis.flag = _em["Flag"]
        # If the emission flag is false
        if not emis.flag:
            return emis
        emis.molarmass = _em["MolarmassGas"]
        emis.method = _em["Method"]
        if emis.method == "CFM":
            _em1 = _em["CFM"]
            # plume height
            emis.plumeheighttype = _em1["Plumeheight"]["Type"]
            emis.plumeheight = _em1["Plumeheight"]["Height"]
            emis.emisname = _em1["Plumeheight"]["Name"]
            emis.plumeheightfromsurface = _em1["Plumeheight"]["HeightFromSurface"]
            # flow
            emis.flow = Flowinfo(_em1["Flow"])
            # if injection height is given then file to it needed
            if emis.plumeheight == "injht":
                _tmp = _em1["Plumeheight"]["InjectionHeight"]
                emis.injht_dir = _tmp["Dir"]
                if "adsapi" in _tmp.keys():
                    emis.injht_adsapiurl = _tmp["adsapi"]["url"]
                    emis.injht_adsapikey = _tmp["adsapi"]["key"]                    
            # if the plume height is varying
            if emis.plumeheighttype == "Varying":
                _sim = _em1["Simulation"]
                emis.particledir = _sim["OutputParticlesDir"]
                # Simulation time
                emis.param_simtime = SimulationTime(_sim["Time"])
                # Dispersion parameters
                emis.param_disp = Dispersion(_sim["Dispersion"])
                # Particle Release
                emis.param_partrelease = MultipleParticleRelease(
                    _sim["MultipleParticleRelease"], emis.param_simtime.sim_time_sec
                )
                # Particle Splitting
                emis.param_partsplit = ParticleSplitting(_sim["ParticleSplitting"])
                # Trajectory data to concentration
                emis.param_traj2conc = Traj2ConcInfo(_sim["TrajectoriesToConcentrations"])
        elif emis.method == "Divergence":
            _em1 = _em["Divergence"]
            emis.name = _em1["Name"]
            emis.plumeheight = _em1["Plumeheight"]
            emis.flow = Flowinfo(_em1["Flow"])
            emis.plumeheightfromsurface = _em1["HeightFromSurface"]
        return emis

    def _get_days(self, _date):
        """List all days

        Get all days from start date to end date

        Parameters
        ----------
        _date : dict
            Contains two keys start and end date

        Return
        --------
        days : date
           Contains all days in date format
        """
        # define start date
        if isinstance(_date["StartDate"], (date, datetime)):
            self.startdate = _date["StartDate"]
        else:
            self.startdate = datetime.strptime(_date["StartDate"], "%Y-%m-%d").date()
        # define end date
        if isinstance(_date["EndDate"], (date, datetime)):
            self.enddate = _date["EndDate"]
        else:
            self.enddate = datetime.strptime(_date["EndDate"], "%Y-%m-%d").date()
        if self.startdate > self.enddate:
            print("Input start date is larger than the end date")
            exit()
        days = []
        stdate = self.startdate
        while stdate <= self.enddate:
            days.append(stdate)
            stdate += timedelta(days=1)
        return days

    def _init_fireparams(self, _fire):
        """Create fire parameters

        Fire parameters from the input dict

        Parameters
        ----------
        _fire : Dict
            Dict containing data

        """
        self.viirsdir = _fire["viirs_dir"]
        self.roi = array(_fire["roi"])
        self.roi_name = _fire["roi_name"]
        self.transform = TransformCoords([self.roi[2], self.roi[0]])
        self.file_prefix = self.roi_name
        self.source_loc = ""
        self.source_name = ""

    def _init_industrialparams(self, _ind):
        """Create Industrial parameters.

        Industrial parameters from he input dict

        Parameters
        ----------
        _ind : Dict
            Dict containing data

        """
        self.source_loc = array(_ind["Source"])
        self.source_name = _ind["Source_name"]
        self.transform = TransformCoords([self.source_loc[0], self.source_loc[1]])
        self.file_prefix = self.source_name
