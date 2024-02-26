#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:11:59 2022.

@author: Manu Goudar
"""

# import warnings
# warnings.simplefilter('error', RuntimeWarning)


try:
    import numpy as np
    from pandas import merge, read_csv
    from .ModuleInitialParameters import InputParameters
    from .ModuleDataContainers import DataContainer
    from .ModDataPrepare_VIIRSData import get_firedata
    from .ModDataPrepare_SatelliteIndentifyOrbits import get_orbits_on_locations
    from .ModDataPrepare_SatelliteRead import readsatellitedata, get_filenames
    from .ModDataPrepare_SatelliteDataFiltering import extract_and_filter_satellitedata
    from .ModuleTransform import TransformCoords
    from .ModuleInjectionHeight import InjectionHeight
    from .ModPlume_Detection import segment_image_plume
    from .ModPlume_Filtering import filter_good_plumes
    from .ModuleWrite import WriteData
    from .ModEE_RunSimFire import compute_emissions
    from .ModuleRead import read_data, get_detectedplumekeys
except ImportError:
    print("Module loading failed")


def getsatelliteorbitsatfiresrcs(day, params, firesrcs):
    """Get fire sources and orbits corresponding to those fire sources.

    Read fire data. Cluster fires and then get all orbits for each custer location

    Parameters
    ----------
    day : Date (datetime)
        Day on which the fire clusters need to be read
    params : Class (InitialParameters)
        Class containing all initialparameters.

    Return
    --------
    Flag : Bool
        If the function was successful.
    viirsdata : Pandas Dataframe
        All VIIRS data read from the csv file
    firesrcs : Pandas Dataframe
        Clustered fire sources and the orbits
    """
    # If fire cluster/s is/are present then continue with finding orbit at locations
    # get orbits corresponding the detected fire clusters
    lat_lon = firesrcs[["latitude", "longitude", "labels"]].values
    o_flag, orb_lbls = get_orbits_on_locations(day, lat_lon, params.sat_files)
    # if no orbits found then return false
    if ~o_flag:
        return False, firesrcs
    # if orbits found then
    firesrc_orb = merge(firesrcs, orb_lbls, on="labels")
    firesrc_orb.sort_values(by=["orbits", "frp"], ascending=[True, False],
                            inplace=True, ignore_index=True)
    return True, firesrcs


def satellitedataoveranorbit(orbitdata, orbittoread, params):
    """Get Satellite data.

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

    """
    if orbitdata.orbit != orbittoread:
        orbitdata = readsatellitedata(params, orbittoread)
    return orbitdata


def fireape_plumedetection(satcont, firesrcs, viirsdata):
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

    # if plume is not detected then break the loop
    if ~plumecontainer.f_plumedetect:
        return False, [], []
    # Check for other fires nearby and filter the plume
    # filter for other fires around
    f_plumefilter, firesrcs = filter_good_plumes(satcont, plumecontainer.plumemask,
                                                 firesrcs, viirsdata)
    # Set a variable that there is nofirearoundplume
    plumecontainer.__setattr__("f_nofirearoundplume", f_plumefilter)
    if f_plumefilter:
        viirscontainer = viirsdata.loc[viirsdata.labels == firesrcs.labels[satcont.fire_id]]
        return f_plumefilter, plumecontainer, viirscontainer
    else:
        return f_plumefilter, plumecontainer, []


def fireape_injectionheight(viirscontainer, inj_ht, writedata):
    """Injection height.

    Check for injection height and write data

    Parameters
    ----------
    viirsdata : Pandas DataFrame
        VIIRS data
    inj_ht : Injection height class
        Class containing methods to get Injection height
    writedata : Class
        Class containing methods to write data

    Return
    --------
    flag_injht: Bool

    """
    # Later activate this flag to only save good plumes that are filtered
    flag_injht = True
    # get injection ht
    injheight = inj_ht.interpolate(viirscontainer.latitude.values, viirscontainer.longitude.values)
    viirscontainer.insert(2, "injection_height", injheight)
    # if injection height doesn't exist then continue
    if np.sum(injheight > 0) < 1:
        print("          Injection doesn't exists")
        # append injection height
        flag_injht = False
    writedata.append_injection_ht(flag_injht, injheight)
    return flag_injht


def saveplumedetectiondata(params, day, satdata, viirsdata, plumedata, writedata):
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

    Return
    --------
    None
    """
    writedata.updatefilename(params.output_file + day.strftime("%Y_%m"))
    writedata.firegrpname = ("D" + str(day.day).zfill(2) + "_" + satdata.fire_name)
    writedata.write(satdata, viirsdata, plumedata)


def datapreparationandplumedetection(day, params, writedata):
    """Macro function using APE Data processing and Plume detection stage for fires.

    Wrappers on APE functions for fires from VIIRS dataset

    Parameters
    ----------
    day : Date
        Day on which the data needs to processed
    params : InitialParameters class
        Parameters of the simulations
    writedata : Class to write data
        Class to write data

    Return
    --------
    flag : Bool
        If a plume was detected and there is data for the day
    """
    # DATA PREPARATION Step 1 : fire sources
    # read fire data, cluster and corresponding orbits
    # Get clustered VIIRS active fire data
    dataexists_flag, cluster_flag, viirsdata, firesrcs1 = get_firedata(day, params)
    # If fire cluster/s is/are not present then stop the function for day
    if ~cluster_flag:
        return False, []
    # get satellite orbits for clustered fire data
    f_orbit, viirsdata, firesrcs = getsatelliteorbitsatfiresrcs(day, params, firesrcs1)
    if ~f_orbit:   # If there is no orbit data or something failed
        return False, []
    # Create a contanier to read orbit data: Multiple fire sources can exist in same orbit
    # so this container helps to stop re-reading data
    orbitdata = DataContainer()
    orbitdata.__setattr__("orbit", 0)
    # identify all plume detected labels and return it
    detectedplumefiresources = []
    detectedplumefiretimes = []
    # loop over all fire sources
    for f_id in range(len(firesrcs)):
        # read orbit data if the old orbit is not same as new orbit data
        orbitdata = satellitedataoveranorbit(orbitdata, firesrcs.orbits[f_id], params)
        # DATA PREPARATION Step 2 : Satellite data
        # Extract satellite data based on fire source
        # source of fire
        src = [firesrcs.latitude[f_id], firesrcs.longitude[f_id]]
        print(f"label:- {firesrcs.labels[f_id]}  fire index:- {f_id}  src:- {src}  orbit:- {firesrcs.orbits[f_id]}")
        satellitecontainer = extract_and_filter_satellitedata(orbitdata, src)
        # Give this fire id as plume is detected
        satellitecontainer.__setattr__("fire_name", "Fire_" + str(f_id).zfill(3))
        satellitecontainer.__setattr__("fire_id", f_id)
        print("      Good satellite data filter: ", satellitecontainer.f_good_satellite_data)
        print("      Data preparation stage successful and satellite data is available")
        if ~satellitecontainer.f_good_satellite_data:  # Data preparation has failed
            continue
        # PLUME DETECTION
        plumeflag, plumecontainer, viirscontainer = fireape_plumedetection(satellitecontainer, firesrcs, viirsdata)
        if ~plumeflag:  # Plume detection has failed
            continue
        detectedplumefiresources.append(satellitecontainer.source)
        detectedplumefiretimes.append(satellitecontainer.measurement_time)
        # WRITE DATA
        saveplumedetectiondata(params, day, satellitecontainer, viirscontainer, plumecontainer, writedata)
    # plumes were detected or not
    if len(detectedplumefiresources) > 0:
        return True, [detectedplumefiresources, detectedplumefiretimes]
    else:
        return False, []


# TODO: Insert JORD's code to cluster and download
# use async method to download
def cluster_downloaddata(day):
    # injection height
    # ERA5 data
    pass


# TODO: Create a new method here. The location and time should be searched
# and the files have to be identified based on that
def getclusterid(filedir, key):
    filename = filedir + "/" + (filedir[-10:]).replace("/", "-") + "_cluster_table.csv"
    df = read_csv(filename)
    return ((df[[key in fire for fire in df.fires]]).filename.values[0]).split("_")[-1]


def changetheflowparams(datedir, _key, params):
    clusterid = getclusterid(params.param_flowinfo.prefix_inputdir + datedir, _key)
    # change the file directory
    clusterid = clusterid + "_"
    params.param_flowinfo.inputdir = params.param_flowinfo.prefix_inputdir + datedir
    params.param_flowinfo.file_flow = clusterid + params.param_flowinfo.prefix_flow
    params.param_flowinfo.inputdir = clusterid + params.param_flowinfo.prefix_pres


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
    # get injection height
    inj_ht = InjectionHeight(params.gfasfile, day)
    # read keys for all plumes
    allplumedetectedkeys = get_detectedplumekeys(writedata.filename)
    # compute emissions
    for _key in allplumedetectedkeys:
        # Read all data
        satellitedata, firedata, plumedata = read_data(params.outputfile, _key)
        flag_injheight = fireape_injectionheight(firedata, inj_ht, writedata)
        if ~flag_injheight:  # No injection height
            continue
        # the directory where the flow data
        changetheflowparams(day.strftime("%Y/%m/%d"), _key, params)
        # final emission estimation
        fire_massfluxcontainer = compute_emissions(day, params, satellitedata, firedata, plumedata)
        # WRITE MASSFLUX DATA
        writedata.append_massflux(fire_massfluxcontainer)


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
        flag, dataneededtodownload = datapreparationandplumedetection(day, params, writedata)
        if flag:
            cluster_downloaddata(day, dataneededtodownload)
            emissionestimationfires(day, params, writedata)
