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
    from .ModDataPrepare_SatelliteRead import readsatellitedata
    from .ModDataPrepare_SatelliteDataFiltering import extract_and_filter_satellitedata
    from .ModuleTransform import TransformCoords
    from .ModPlume_Detection import segment_image_plume
    from .ModuleWrite import WriteData
except ImportError:
    print("Module loading failed")


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


def datapreparation(day, params):
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
    f_orbit, orbits = get_orbits_on_locations(day, params.ind_source, params.sat_files)
    data.__setattr__("flag_orbits", f_orbit)
    # If there is no data and something failed then return none
    if not data.flag_orbit:
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
        orbitdata = readsatellitedata(orbitinfo.filename, orbitinfo.orbit, orbitinfo.version)

        # Extract satellite data based on fire source
        satellitecontainer = extract_and_filter_satellitedata(orbitdata, params.ind_source)
        print("      Good satellite data filter: ", satellitecontainer.f_good_satellite_data)
        if satellitecontainer.f_good_satellite_data:
            data.goodcases += 1
            data.extracteddata.append(satellitecontainer)
    return data


# def plumedetection():
#     # PLUME DETECTION
#     plumecontainer = pointsrc_plumedetection(satellitecontainer)
#     if not plumecontainer.f_plumedetect:  # Plume detection has failed
#         continue

#     detectedplumeorbit.append(satellitecontainer.orbit)
#     detectedplumetime.append(satellitecontainer.measurement_time)
#     # WRITE DATA
#     saveplumedetectiondata(params, day, satellitecontainer, plumecontainer, writedata)
#     # plume detected or not
#     if len(detectedplumeorbit) > 0:
#         return True, [detectedplumeorbit, detectedplumetime]
#     else:
#         return False, []


# def downloadmeteofields(day, dataneededtodownload): 
#     pass


# def emissionestimationfires(day, params, writedata):
#     """Estimate emission for fires

#     Read injection height, plume keys and change flow parameters
#     And then estimate emissions

#     Parameters
#     ----------
#     day : Date
#         Day on which the emission need to be estimated
#     params : InitialParameters class
#         Class containing initial paramaters
#     writedata : class
#         Class containing methods to write data

#     """
#     # fix some inputs
#     params.param_flowinfo.__setattr__("prefix_inputdir", params.param_flowinfo.inputdir)
#     params.param_flowinfo.__setattr__("prefix_flow", params.param_flowinfo.file_flow)
#     params.param_flowinfo.__setattr__("prefix_pres", params.param_flowinfo.file_pres)

#     # compute emissions
#     # for orb in orbits:
#     #     # Read all data
#     #     satellitedata, firedata, plumedata = read_data(params.outputfile, _key)
#     #     flag_injheight = fireape_injectionheight(firedata, inj_ht, writedata)
#     #     if ~flag_injheight:  # No injection height
#     #         continue
#     #     # the directory where the flow data
#     #     changetheflowparams(day.strftime("%Y/%m/%d"), _key, params)
#     #     # final emission estimation
#     #     fire_massfluxcontainer = compute_emissions(day, params, satellitedata, firedata, plumedata)
#     #     # WRITE MASSFLUX DATA
#     #     writedata.append_massflux(fire_massfluxcontainer)


def run_industrial(filename):
    """Industrial source for a day.

    2

    Parameters
    ----------
    filename : 3
        4

    Examples
    --------
    5

    """
    # Read input file
    params = InputParameters(filename)
    # Initialize a file to write data
    writedata = WriteData(params.output_file_prefix)
    # Transform will always be same for industrial source
    transform = TransformCoords(params.ind_source)

    # Run APE Algorithm for a day
    for _day in params.days:
        print(_day)
        # data preparation
        dataforday = datapreparation(_day, params)
        print("      Data preparation stage successful")

        # write data from data preparation
        writedata.write(dataforday)

        # continue if there are no cases identified by data preparation
        if dataforday.goodcases == 0:
            continue

        # Perform plume detection for all cases
        for i in dataforday.goodcases:
            data = dataforday.extracteddata[i]
            plumecontainer = segment_image_plume(data.lat, data.lon,
                                                 data.co_column_corr,
                                                 data.co_qa_mask, transform)
            writedata.write_plumedata(plumecontainer)

        # Emission estimation
        # emissionestimation(ay, params, writedata)
