#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:00:00 2023.

@author: Manu Goudar
"""

# import warnings
# warnings.simplefilter('error', RuntimeWarning)


try:
    from .ModuleInitialParameters import InputParameters
    from .ModuleDataContainers import DataContainer
    from .ModDataPrepare_SatelliteIndentifyOrbits import get_orbits_on_locations
    from .ModDataPrepare_SatelliteRead import readsatellitedata
    from .ModDataPrepare_SatelliteDataFiltering import extractfilter_satellitedata
    from .ModPlume_Detection import segment_image_plume
    from .ModuleWrite import WriteData
    from .ModuleRead import ReadData
    from .ModEE_ComputeEmissions import crosssectionalflux
    from .ModEE_Divergence import creategrid_divergence, divergence
    from .ModCheckDownloadVelocity import checkanddownloadvelocity
    import numpy as np
except ImportError:
    print("Module loading failed")
    exit()

def saveplumedetectiondata(params, day, satdata, plumedata, writedata):
    """Save parameters after plume detection.

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
    """
    writedata.updatefilename(params.output_file_prefix + day.strftime("%Y"))
    writedata.grpname = day.strftime("%m_%d") + "_" + str(satdata.orbit)
    writedata.write_industry(satdata, plumedata)


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
            temporaldivergencedata = np.concatenate((temporaldivergencedata, [divergencedata.div]))
    # estimate emission by thresholding
    # TODO: Add function to compute massflux
    meandivergence = np.nanmean(temporaldivergencedata, axis=0)
    return meandivergence, temporaldivergencedata


def detectplume(params, writedata=None):
    # Initialize a file to read data
    _readdata = ReadData(params.output_file, params.days)
    # data to read
    datatoread = ["Satellite"]
    # Initialize a file to write data
    if writedata is None:
        writedata = WriteData(params.output_file)
    for _key in _readdata.keys:
        data = _readdata.read(_key, datatoread)
        plumecontainer = segment_image_plume(data.lat, data.lon, data.co_column_corr,
                                             data.co_qa_mask, params.transform)
        writedata.write_plumedata(_key, plumecontainer)


def estimatecfmemission(params):
    # Initialize a file to read data
    _readdata = ReadData(params.output_file, params.days)
    # data to read
    datatoread = ["Satellite", "Plume"]
    # Initialize a file to write data
    writedata = WriteData(params.output_file)
    for _key in _readdata.keys:
        data, plumecont = _readdata.read(_key, datatoread)
        massflux = crosssectionalflux(params, data, plumecont, params.transform)
        writedata.write_cfm(massflux)



def run(filename):
    """Industrial source for an given input.

    Parameters
    ----------
    filename : string
        Filename

    Examples
    --------
    filename = "input.yaml"
    run_industrial(filename)

    """
    # Read input file
    params = InputParameters(filename)
    # Initialize a file to write data
    writedata = WriteData(params.output_file)

    # prepare data prepare
    preprocessdata(params)

    # if emission estimation is Divergence method
    # this doesn't need plume detection
    if (params.estimateemission.flag) & (params.estimateemission.method == "Divergence"):
        # download velocity fields
        checkanddownloadvelocity(params)
        # compute divergence
        return computedivergence(params)

    # if plume detection is true
    # if not params.detectplume:
    #     return None

    # emission estimates
    if (params.estimateemission.flag) & (params.estimateemission.method == "CFM"):
        # Initialize a file to read data
        _readdata = ReadData(params.output_file, params.days)
        # data to read
        datatoread = ["Satellite"]
        for _key in _readdata.keys:
            data = _readdata.read(_key, datatoread)
            plumecontainer = segment_image_plume(data.lat, data.lon, data.co_column_corr,
                                                 data.co_qa_mask, params.transform)
            writedata.write_plumedata(plumecontainer)
            # check if plume was not detected
            if not plumecontainer.flag_plumedetected:
                continue
            # download velocity fields
            checkanddownloadvelocity(_key)
            # plume was detected
            massflux = crosssectionalflux(params, data, plumecontainer, params.transform)
            # writedata.write_cfm(massflux)
            # TODO: Add variable to define individual emission
            try:
                cfm.append([massflux.emission, data.measurement_time, data.orbit])
            except NameError:
                cfm = [massflux.emission, data.measurement_time, data.orbit]
        return cfm
    else:
        # only detect plumes
        detectplume(params, writedata)
        return None
