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
    from .ModuleDataContainers import DataContainer
    from .ModDataPrepare_VIIRSData import get_firedata
    from .ModDataPrepare_SatelliteIndentifyOrbits import get_orbits_on_locations
    from .ModDataPrepare_SatelliteRead import readsatellitedata
    from .ModDataPrepare_SatelliteDataFiltering import extractfilter_satellitedata
    from .ModuleInjectionHeight import InjectionHeight
    from .ModPlume_Detection import segment_image_plume
    from .ModPlume_Filtering import filter_good_plumes
    from .ModuleWrite import WriteData
    from .ModuleRead import ReadData
    from .ModEE_ParticlesBasedCrosssectionalFlux import crosssectionalflux_varying
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
    if not o_flag:
        return False, firesrcs
    # if orbits found then
    firesrc_orb = merge(firesrcs, orb_lbls, on="labels")
    firesrc_orb.sort_values(by=["orbits", "frp"], ascending=[True, False],
                            inplace=True, ignore_index=True)
    return True, firesrc_orb


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
        orbitinfo = params.sat_files[params.sat_files.orbit == orbittoread]
        orbitdata = readsatellitedata(orbitinfo.filename.iloc[0], orbitinfo.orbit.iloc[0], orbitinfo.version.iloc[0])
    return orbitdata


def fireape_plumedetection(satcont, firesrcs, viirsdata, transform):
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
    transform.update_origin(satcont.source)
    plumecontainer = segment_image_plume(satcont.lat, satcont.lon, satcont.co_column_corr,
                                         satcont.co_qa_mask, transform)
    # Print if plume is segmented or not
    print(f"        Image segmented : {plumecontainer.flag_plumedetected}")

    # initialize default plume as bad plume and set it good later
    setattr(plumecontainer, "flag_goodplume", False)

    # if plume is not detected then break the loop
    if not plumecontainer.flag_plumedetected:
        return plumecontainer, []
    # Check for other fires nearby and filter the plume
    # filter for other fires around
    f_plumefilter, firesrcs = filter_good_plumes(satcont, plumecontainer.plumemask,
                                                 firesrcs, viirsdata)
    # Set a variable that there is nofirearoundplume
    setattr(plumecontainer, "flag_nofirearoundplume", f_plumefilter)
    if plumecontainer.flag_nofirearoundplume:
        plumecontainer.flag_goodplume = True
        viirscontainer = viirsdata.loc[viirsdata.labels == firesrcs.labels[satcont.fire_id]]
        return plumecontainer, viirscontainer
    else:
        return plumecontainer, []



def datapreparation_plumedetection(_day, params, writedata=None):
    """Macro function using APE Data processing for fires.

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
    # Initialize a file to write data
    if writedata is None:
        writedata = WriteData(params.output_file_prefix + _day.strftime("%Y_%m"))
    
    # Define the outfile name in write data class
    writedata.updatefilename(params.output_file_prefix + _day.strftime("%Y_%m"))

    # DATA PREPARATION Step 1 : fire sources
    # read fire data, cluster and corresponding orbits
    # Get clustered VIIRS active fire data
    flag_firedataexists, flag_dataclustered, viirsdata, firesrcs1 = get_firedata(_day, params)
    # writedata.preprocess_viirs(_day.strftime("%Y%m%d"), viirsdata, {"flag_firedataexists": flag_firedataexists,
    #                                        "flag_dataclustered": flag_dataclustered})

    # If fire cluster/s is/are not present then stop the function for day
    if not flag_dataclustered:   # if False then data  is not clustered so return
        return False, []
    # get satellite orbits for clustered fire data
    flag_orbits, firesrcs = getsatelliteorbitsatfiresrcs(_day, params, firesrcs1)
    
    if not flag_orbits:   # If there is no orbit data or something failed
        return False, []
    # Create a contanier to read orbit data: Multiple fire sources can exist in same orbit
    # so this container helps to stop re-reading data
    orbitdata = DataContainer()
    setattr(orbitdata, "orbit", 0)

    # number of plumes detected
    nogoodplumes= 0
    # loop over all fire sources for the day
    for f_id in range(len(firesrcs)):
        # read orbit data
        orbitdata = satellitedataoveranorbit(orbitdata, firesrcs.orbits[f_id], params)

        # DATA PREPARATION Step 2 : Satellite data
        # Extract satellite data based on fire source
        # source of fire
        src = [firesrcs.latitude[f_id], firesrcs.longitude[f_id]]
        print(f"label:- {firesrcs.labels[f_id]}  fire index:- {f_id}  src:- {src}  orbit:- {firesrcs.orbits[f_id]}")
        satellitecontainer = extractfilter_satellitedata(orbitdata, src)
        print("      Good satellite data filter: ", satellitecontainer.flag_goodsatellitedata)
        print("      Data preparation stage successful and satellite data is available")
        if not satellitecontainer.flag_goodsatellitedata:  # Data preparation has failed
            continue

        # Give this fire id as plume is detected
        # setattr(satellitecontainer, "fire_name", "Fire_" + str(f_id).zfill(3))
        setattr(satellitecontainer, "fire_id", f_id)
        
        # PLUME DETECTION
        plumecontainer, viirscontainer = fireape_plumedetection(satellitecontainer, firesrcs, viirsdata, params.transform)
        if not plumecontainer.flag_goodplume:  # Plume detection has failed
            continue
        # Write plume data
        # save data satellite and VIIRS
        writedata.satellite(satellitecontainer.uniqueid, satellitecontainer)
        writedata.viirs(satellitecontainer.uniqueid, viirscontainer)
        writedata.plume(satellitecontainer.uniqueid, plumecontainer)
        nogoodplumes += 1
    return nogoodplumes


# TODO: Create a new method here. The location and time should be searched
# and the files have to be identified based on that
def getclusterid(filedir, key):
    filename = filedir + "/" + "cluster_table.csv"
    df = read_csv(filename)
    return ((df[[key in fire for fire in df.fires]]).filename_suffix.values[0]).split("_")[-1]


def changetheflowparams(datedir, _key, params):
    clusterid = getclusterid(params.flow.inputdir +  datedir, _key)
    # change the file directory
    params.flow.flowdir = params.flow.inputdir +  datedir
    params.flow.file_flow = clusterid + params.flow.file_flow_suffix
    params.flow.file_pres = clusterid + params.flow.file_pres_suffix

    
def fireape_injectionheight(viirscontainer, inj_ht):
    """Injection height.

    Check for injection height

    Parameters
    ----------
    viirsdata : Pandas DataFrame
        VIIRS data
    inj_ht : Injection height class
        Class containing methods to get Injection height
    Return
    --------
    flag_injht: Bool

    """
    # Later activate this flag to only save good plumes that are filtered
    flag_injht = True
    # get injection ht
    injheight = inj_ht.interpolate(viirscontainer.latitude, viirscontainer.longitude)
    setattr(viirscontainer, "injection_height", injheight)
    # if injection height doesn't exist then continue
    if np.sum(injheight > 0) < 1:
        print("          Injection doesn't exists")
        # append injection height
        flag_injht = False
    return flag_injht


def emissionestimationfires(day, params, writedata=None):
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
    if writedata is None:
        writedata = WriteData(params.output_file_prefix + day.strftime("%Y_%m"))
    # Define the outfile name in write data class
    writedata.updatefilename(params.output_file_prefix + day.strftime("%Y_%m"))
    
    # initialize injection height class
    inj_ht = InjectionHeight(params.estimateemission.injht_dir, day)
    # initialize a read class
    flname = params.output_file_prefix + day.strftime("%Y_%m")
    _read = ReadData(flname, day, True)

    # compute emissions for each key
    for _key in _read.keys:
        print("Reading data ", _key)
        # Read all data
        satellitedata, firedata, plumedata = _read.getgroupdata(_key)
        flag_injheight = fireape_injectionheight(firedata, inj_ht)

        # write injection height
        writedata.injection_ht(_key, flag_injheight, firedata.injection_height)

        if not flag_injheight:  # No injection height
            continue
        
        # the directory where the flow data
        changetheflowparams(day.strftime("%Y/%m/%d"), _key, params.estimateemission)
        # final emission estimation
        print("   Estimate emissions ") 
        massflux = crosssectionalflux_varying(_key, params, satellitedata, plumedata, firedata,)
        print("             Done")
        # WRITE MASSFLUX DATA
        writedata.write_cfm(_key, massflux)
