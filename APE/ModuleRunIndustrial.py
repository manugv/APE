#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:11:59 2022.

@author: Manu Goudar
"""

# import warnings
# warnings.simplefilter('error', RuntimeWarning)


try:
    from .ModuleInitialParameters import InputParameters
    from .ModuleDataContainers import DataContainer
    from .ModDataPrepare_SatelliteIndentifyOrbits import get_orbits_on_locations
    from .ModDataPrepare_SatelliteRead import readsatellitedata, get_filenames
    from .ModDataPrepare_SatelliteDataFiltering import extract_and_filter_satellitedata
    from .ModuleTransform import TransformCoords
    from .ModPlume_Detection import segment_image_plume
    from .ModuleWrite import WriteData
    from .ModEE_RunSimInd import emissionestimation
except ImportError:
    print("Module loading failed")


def satellitedataoveranorbit(orbitdata, orbittoread, params):
    """Get Satellite data

    Check if the exist satellite data orbit is same as given orbit
    if it's not then read new orbit into it

    Parameters
    ----------
    orbitdata : Class
        Satellite data for an orbit
    orbittoread : Int
        Orbit at which satellite data is needed
    params : Class
        InitialParameters class

    Examples
    --------
    """
    if orbitdata.orbit != orbittoread:
        orbitdata = readsatellitedata(params, orbittoread)
    return orbitdata


def pointsrc_plumedetection(satcont):
    """Plume detection for a fire source

    Wrapper for plume detection algorithm

    Parameters
    ----------
    satcont : Class
        Satellite data class
    firesrcs : Pandas DataFrame
        Dataframe containing clustered VIIRS data
    viirsdata : Pandas DataFrame
        VIIRS data

    Examples
    --------

    """
    # PLUME DETECTION
    # Plume detection : segment image
    satcont.__setattr__("transform", TransformCoords(satcont.source))
    plumecontainer = segment_image_plume(satcont.lat, satcont.lon, satcont.co_column_corr,
                                         satcont.co_qa_mask, satcont.transform)
    # Print if plume is segmented or not
    print(f"        Image segmented : {plumecontainer.f_plumedetect}")

    return plumecontainer


def saveplumedetectiondata(params, day, satdata, plumedata, writedata):
    """Save parameters after plume detection

    Write data to a file

    Parameters
    ----------
    params : InitialParameters Class
        Class containing variables
    day : Date
        Date of the day
    satdata : Class
        Class containing satellite data
    viirsdata : Class
        Class containing VIIRS data
    plumedata : Class
        Class containing plume data
    writedata : Class
        Class containing methods to write data

    Examples
    --------

    """
    writedata.updatefilename(params.output_file_prefix + day.strftime("%Y"))
    writedata.grpname = day.strftime("%m_%d") + "_" + str(satdata.orbit)
    writedata.write_industry(satdata, plumedata)


def datapreparationandplumedetection(day, params, writedata):
    # DATA PREPARATION Step 1 : fire sources
    # read fire data, cluster and corresponding orbits

    f_orbit, orbits = get_orbits_on_locations(day, params.ind_source, params.sat_files)
    # If there is no data and something failed then return none
    if not f_orbit:
        return None, []

    # Create a contanier to read orbit data: Multiple fire sources can exist in same orbit
    # so this container helps to stop re-reading data
    orbitdata = DataContainer()
    orbitdata.__setattr__("orbit", 0)
    detectedplumeorbit = []
    detectedplumetime = []
    # loop over all orbits for the day
    for f_id in range(len(orbits)):
        # read orbit data if the old orbit is not same as new orbit data
        orbitdata = satellitedataoveranorbit(orbitdata, orbits.orbits[f_id], params)

        # DATA PREPARATION Step 2 : Satellite data
        # Extract satellite data based on fire source
        # source of fire
        src = params.ind_source
        print(f" orbit:- {orbits.orbits[f_id]}")
        satellitecontainer = extract_and_filter_satellitedata(orbitdata, src)
        satellitecontainer.__setattr__("id", f_id)
        print("      Good satellite data filter: ", satellitecontainer.f_good_satellite_data)
        print("      Data preparation stage successful and satellite data is available")
        if not satellitecontainer.f_good_satellite_data:  # Data preparation has failed
            continue

        # PLUME DETECTION
        plumecontainer = pointsrc_plumedetection(satellitecontainer)
        if not plumecontainer.f_plumedetect:  # Plume detection has failed
            continue

        detectedplumeorbit.append(satellitecontainer.orbit)
        detectedplumetime.append(satellitecontainer.measurement_time)
        # WRITE DATA
        saveplumedetectiondata(params, day, satellitecontainer, plumecontainer, writedata)
        # plume detected or not
    if len(detectedplumeorbit) > 0:
        return True, [detectedplumeorbit, detectedplumetime]
    else:
        return False, []


def downloadmeteofields(day, dataneededtodownload):
    pass


def emissionestimationfires(day, params, writedata):
    """Estimate emission for fires

    Read injection height, plume keys and change flow parameters
    And then estimate emissions

    Parameters
    ----------
    day : Date
        Day on which the emission need to be estimated
    params : InitialParameters class
        Class containing initial paramaters
    writedata : class
        Class containing methods to write data

    """
    # fix some inputs
    params.param_flowinfo.__setattr__("prefix_inputdir", params.param_flowinfo.inputdir)
    params.param_flowinfo.__setattr__("prefix_flow", params.param_flowinfo.file_flow)
    params.param_flowinfo.__setattr__("prefix_pres", params.param_flowinfo.file_pres)

    # compute emissions
    # for orb in orbits:
    #     # Read all data
    #     satellitedata, firedata, plumedata = read_data(params.outputfile, _key)
    #     flag_injheight = fireape_injectionheight(firedata, inj_ht, writedata)
    #     if ~flag_injheight:  # No injection height
    #         continue
    #     # the directory where the flow data
    #     changetheflowparams(day.strftime("%Y/%m/%d"), _key, params)
    #     # final emission estimation
    #     fire_massfluxcontainer = compute_emissions(day, params, satellitedata, firedata, plumedata)
    #     # WRITE MASSFLUX DATA
    #     writedata.append_massflux(fire_massfluxcontainer)


def run_fire(filename):
    # Read input file
    params = InputParameters(filename)

    # Get all satellite files in the folder and orbits
    params.__setattr__("sat_files", get_filenames(params.satellite_dir))
    print("Satellite file read done")

    # Initialize a file to write data
    writedata = WriteData(params.output_file_prefix)
    # Run APE Algorithm for a day
    for day in params.days:
        print(day)
        flag, datafordownload = datapreparationandplumedetection(day, params, writedata)
        if flag:
            downloadmeteofields(day, datafordownload)
            emissionestimation(day, params, writedata)
