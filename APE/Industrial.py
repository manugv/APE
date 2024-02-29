#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:00:00 2023.

@author: Manu Goudar
"""

# import warnings
# warnings.simplefilter('error', RuntimeWarning)

try:
    from numpy import concatenate, nanmean
    from .ModuleDataContainers import DataContainer
    from .ModDataPrepare_SatelliteIndentifyOrbits import get_orbits_on_locations
    from .ModDataPrepare_SatelliteRead import readsatellitedata
    from .ModDataPrepare_SatelliteDataFiltering import extractfilter_satellitedata
    from .ModPlume_Detection import segment_image_plume
    from .ModuleWrite import WriteData
    from .ModuleRead import ReadData
    from .ModEE_CrossSectionalFlux import crosssectionalflux
    from .ModEE_Divergence import creategrid_divergence, divergence
except ImportError:
    print("Module loading failed")
    exit()


def datapreparationperday(day, params):
    """Prepare data: get orbits, extract per day.

    Parameters
    ----------
    day : Date
        data of the day
    params : Data container
        Parameters containing global parameters
    writedata : Class to write
        Write the data

    """
    # dump extracted data per day
    data = DataContainer()
    # setattr(data, "extracteddata", [])
    data.__setattr__("extracteddata", [])
    # loop over all orbits for the day
    data.__setattr__("goodcases", 0)

    # DATA PREPARATION Step 1: get all orbits at the location
    f_orbit, orbits = get_orbits_on_locations(day, params.source_loc, params.sat_files)
    data.__setattr__("flag_orbits", f_orbit)
    # If there is no data and something failed then return none
    if not data.flag_orbits:
        print("  No orbit present")
        return data

    # add the orbits to read
    data.__setattr__("orbits", orbits)
    # loop over the orbits
    for orb in orbits.orbits:
        print(f" orbit:- {orb}")
        # DATA PREPARATION Step 2 : Satellite data
        # read orbit data if the old orbit is not same as new orbit data
        orbitinfo = params.sat_files[params.sat_files.orbit == orb]
        orbitdata = readsatellitedata(orbitinfo.filename.iloc[0], orbitinfo.orbit.iloc[0], orbitinfo.version.iloc[0])

        # Extract satellite data based on fire source
        # reset grid size filter to 21, 194
        satellitecontainer = extractfilter_satellitedata(orbitdata, params.source_loc)
        print("      Good satellite data filter: ", satellitecontainer.flag_goodsatellitedata)
        if satellitecontainer.flag_goodsatellitedata:
            data.goodcases += 1
            data.extracteddata.append(satellitecontainer)
    return data


def preprocessdata(params, writedata=None):
    """Preprocess data.

    From location, satellite orbit for given inputs is determined
    and satellite data is extracted based on its quality.

    Parameters
    ----------
    params : InputParameters Class
        Parameters containing the inputs
    writedata : Writedata class
        Class to write data

    """
    # Initialize a file to write data
    if writedata is None:
        writedata = WriteData(params.output_file)

    # Run APE Algorithm for a day
    for _day in params.days:
        print(_day)
        # data preparation
        dataforday = datapreparationperday(_day, params)
        print("Data preparation stage successful for date :", _day)

        # write data from data preparation if data exists
        if dataforday.goodcases == 0:
            continue
        for satellitedata in dataforday.extracteddata:
            # define the group name by unique identifier
            writedata.satellite(satellitedata.uniqueid, satellitedata)


def computedivergence(params, writedata=None):
    """Compute divergence based on input parameters.

    Reads the extracted satellite data and the downloaded velocity
    to compute the divergence of the scene. The divergence is returned
    but not the emission.

    Parameters
    ----------
    params : InputParameters Class
        Parameters containing the inputs
    writedata : Writedata class
        Class to write data

    """
    # Initialize a file to read data
    _readdata = ReadData(params.output_file, params.days)
    # Initialize a file to write data
    if writedata is None:
        writedata = WriteData(params.output_file)
    # Run APE Algorithm for a day
    temporaldivergencedata = None
    # create a divergence grid
    datagrid = creategrid_divergence(1000, params.transform)
    # Read the data and compute divergence
    for _key in _readdata.keys:
        data = _readdata.satellite(_key)
        divergencedata = divergence(params, data, datagrid)
        # Write individual divergences
        writedata.divergence(_key, params.estimateemission.name, divergencedata, datagrid)
        if temporaldivergencedata is None:
            temporaldivergencedata = divergencedata.div[None, :, :]
        else:
            temporaldivergencedata = concatenate((temporaldivergencedata, [divergencedata.div]))
    # estimate emission by thresholding
    # TODO: Add function to compute massflux
    meandivergence = nanmean(temporaldivergencedata, axis=0)
    return meandivergence, temporaldivergencedata


def detectplume(params, writedata=None):
    """Plume detection based on input parameters.

    Reads good satellite data and extractes plume based plume detection algorithm.

    Parameters
    ----------
    params : InputParameters Class
        Parameters containing the inputs
    writedata : Writedata class
        Class to write data

    """
    # Initialize a file to read data
    _readdata = ReadData(params.output_file, params.days)
    # Initialize a file to write data
    if writedata is None:
        writedata = WriteData(params.output_file)
    for _key in _readdata.keys:
        print(_key)
        data = _readdata.satellite(_key)
        plumecontainer = segment_image_plume(data.lat, data.lon, data.co_column_corr, data.co_qa_mask, params.transform)
        writedata.plume(_key, plumecontainer)


def estimatecfmemission(params, writedata):
    """Compute emission based cross-sectional flux.

    Reads the extracted satellite data and the downloaded velocity
    to compute the divergence of the scene. The divergence is returned
    but not the emission.

    Parameters
    ----------
    params : InputParameters Class
        Parameters containing the inputs
    writedata : Writedata class
        Class to write data

    """
    # Initialize a file to read data
    _readdata = ReadData(params.output_file, params.days, onlyplumes=True)
    # Initialize a file to write data
    if writedata is None:
        writedata = WriteData(params.output_file)

    for _key in _readdata.keys:
        print("Computing emissions for:", _key)
        data = _readdata.satellite(_key)
        plumecont = _readdata.plume(_key)
        massflux, estd_emission = crosssectionalflux(params, data, plumecont, params.transform)
        print("       computed")
        writedata.write_cfm(_key, massflux)
