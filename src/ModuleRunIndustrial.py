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
    from ModuleInitialParameters import InputParameters
    from ModDataPrepare_VIIRSData import get_firedata
    from ModDataPrepare_SatelliteIndentifyOrbits import get_orbits_on_locations
    from ModDataPrepare_SatelliteRead import readsatellitedata, get_filenames
    from ModDataPrepare_SatelliteDataFiltering import extract_and_filter_satellitedata
    from ModuleTransform import TransformCoords
    from ModuleInjectionHeight import InjectionHeight
    from ModPlume_Detection import segment_image_plume
    from ModPlume_Filtering import filter_good_plumes
    from ModuleWrite import WriteData
    from ModEE_RunSim import compute_emissions
except ImportError:
    print("Module loading failed")


def check_get_data(day, params):
    # Get clustered VIIRS active fire data
    d_flag, c_flag, viirsdata, firesrcs = get_firedata(day, params)
    # If fire cluster/s is/are present
    if c_flag:
        # get orbits corresponding the detected fire clusters
        o_flag, firesrcs = get_orbits_on_locations(day, firesrcs)
    else:
        o_flag = False
    return o_flag, viirsdata, firesrcs


def run_fire(filename):
    # Read input file
    params = InputParameters(filename)

    # TODO : Check if files exists and then get them
    # Satellite files
    # Define the files names and orbits for satellite data based on month and year
    # Assumes the data is already present
    params.__setattr__("sat_files", get_filenames(params.satellite_dir))
    print("Satellite file read done")

    # Initialize a file to write data
    writedata = WriteData(params.output_file)

    # Algorithm
    for day in params.days:
        print(day)
        # inj_ht = InjectionHeight(params.gfasfile, day)
        # DATA PREPARATION
        # Cluster and get VIIRS active fire data and orbits of clustered data
        f_orbit, viirsdata, firesrcs = check_get_data(day, params)

        # If orbits were detected, loop over all fire clusters for that day
        read_orbit = 0  # Stops re-reading of orbit from harddisk
        f_id = 0  # iteration
        nos_fires = len(firesrcs)
        # Change this f_id buisness ##################
        while f_orbit and (f_id < nos_fires):
            # If the fire source has to be read or not. This is useful as
            # while processing, some fireclusters are integrated with each other.
            if firesrcs.readflag[f_id]:
                src = [firesrcs.latitude[f_id], firesrcs.longitude[f_id]]  # source of fire
                print(f"fire index:- {f_id}     src:- {src}")
                # Flag to say the data is read or not
                if read_orbit != firesrcs.orbits[f_id]:
                    orbit_satdata = readsatellitedata(params, firesrcs.orbits[f_id])
                # Extract satellite data based on fire source
                fire_satellitecontainer = extract_and_filter_satellitedata(orbit_satdata, src)
                print(
                    "      Good satellite data filter: ",
                    fire_satellitecontainer.f_good_satellite_data,
                )

                # PLUME DETECTION
                if fire_satellitecontainer.f_good_satellite_data:  # If orbit data is good
                    # Plume detection : segment image
                    transform = TransformCoords(src)
                    fire_plumecontainer = segment_image_plume(fire_satellitecontainer, transform)
                    # Print if plume is segmented or not
                    print(f"image segmented : {fire_plumecontainer.f_plumedetect}")
                    # Check for other fires nearby and filter the plume
                    if fire_plumecontainer.f_plumedetect:
                        # Give this fire an id as plume is detected
                        fire_satellitecontainer.__setattr__(
                            "fire_name", "Fire_" + str(f_id).zfill(3)
                        )
                        fire_satellitecontainer.__setattr__("fire_id", f_id)

                        # filter for other fires around
                        firedata_roi, f_plumefilter, firesrcs = filter_good_plumes(
                            f_id,
                            fire_satellitecontainer,
                            fire_plumecontainer.plumemask,
                            firesrcs,
                            viirsdata,
                        )
                        fire_plumecontainer.__setattr__("f_firearoundplume", f_plumefilter)
                        fire_viirscontainer = firedata_roi.loc[
                            viirsdata.labels == firesrcs.labels[f_id]
                        ]
                        # firecontainer.__setattr__("viirs_data", src_fire)
                        # Write data
                        writedata.firegrpname = (
                            "D" + str(day.day).zfill(2) + "_" + fire_satellitecontainer.fire_name
                        )
                        writedata.write(
                            fire_satellitecontainer, fire_viirscontainer, fire_plumecontainer
                        )

                        # # Later activate this flag to only save good plumes that are filtered
                        # if fire_plumecontainer.f_firearoundplume:
                        #     # get injection ht
                        #     fire_viirscontainer.insert(
                        #         2,
                        #         "injection_height",
                        #         inj_ht.interpolate(
                        #             fire_viirscontainer.latitude.values,
                        #             fire_viirscontainer.longitude.values,
                        #         ),
                        #     )
                        #     if np.sum(fire_viirscontainer["injection_height"] > 0) > 1:
                        #         f_InjectionHeightExists = True
                        #     else:
                        #         f_InjectionHeightExists = False
                        #     # Append injection height
                        #     writedata.append_injection_ht(
                        #         f_InjectionHeightExists,
                        #         fire_viirscontainer["injection_height"].values,
                        #     )
                        #     # compute emissions
                        #     if f_InjectionHeightExists:
                        #         # compute emissions
                        #         fire_massfluxcontainer = compute_emissions(
                        #             day,
                        #             params,
                        #             fire_satellitecontainer,
                        #             fire_viirscontainer,
                        #             fire_plumecontainer,
                        #         )
                        #         writedata.append_massflux(fire_massfluxcontainer)
            f_id += 1


def run_industrial(filename):
    # Read input file
    params = InputParameters(filename)

    # TODO : Check if files exists and then get them
    # Satellite files
    # Define the files names and orbits for satellite data based on month and year
    # Assumes the data is already present
    params.__setattr__("sat_files", get_filenames(params.satellite_dir))
    print("Satellite file read done")

    # Initialize a file to write data
    writedata = WriteData(params.output_file)

    # Algorithm
    for day in params.days:
        print(day)
        # DATA PREPARATION
        # Cluster and get VIIRS active fire data and orbits of clustered data
        f_orbit, viirsdata, firesrcs = check_get_data(day, params)

        # If orbits were detected, loop over all fire clusters for that day
        read_orbit = 0  # Stops re-reading of orbit from harddisk
        f_id = 0  # iteration
        nos_fires = len(firesrcs)
        # Change this f_id buisness ##################
        while f_orbit and (f_id < nos_fires):
            # If the fire source has to be read or not. This is useful as
            # while processing, some fireclusters are integrated with each other.
            if firesrcs.readflag[f_id]:
                src = [firesrcs.latitude[f_id], firesrcs.longitude[f_id]]  # source of fire
                print(f"fire index:- {f_id}     src:- {src}")
                # Flag to say the data is read or not
                if read_orbit != firesrcs.orbits[f_id]:
                    orbit_satdata = readsatellitedata(params, firesrcs.orbits[f_id])
                # Extract satellite data based on fire source
                fire_satellitecontainer = extract_and_filter_satellitedata(orbit_satdata, src)
                print(
                    "      Good satellite data filter: ",
                    fire_satellitecontainer.f_good_satellite_data,
                )

                # PLUME DETECTION
                if fire_satellitecontainer.f_good_satellite_data:  # If orbit data is good
                    # Plume detection : segment image
                    transform = TransformCoords(src)
                    fire_plumecontainer = segment_image_plume(fire_satellitecontainer, transform)
                    # Print if plume is segmented or not
                    print(f"image segmented : {fire_plumecontainer.f_plumedetect}")
                    # Check for other fires nearby and filter the plume
                    if fire_plumecontainer.f_plumedetect:
                        # Give this fire an id as plume is detected
                        fire_satellitecontainer.__setattr__(
                            "fire_name", "Fire_" + str(f_id).zfill(3)
                        )
                        fire_satellitecontainer.__setattr__("fire_id", f_id)

                        # filter for other fires around
                        firedata_roi, f_plumefilter, firesrcs = filter_good_plumes(
                            f_id,
                            fire_satellitecontainer,
                            fire_plumecontainer.plumemask,
                            firesrcs,
                            viirsdata,
                        )
                        fire_plumecontainer.__setattr__("f_firearoundplume", f_plumefilter)
                        fire_viirscontainer = firedata_roi.loc[
                            viirsdata.labels == firesrcs.labels[f_id]
                        ]
                        # firecontainer.__setattr__("viirs_data", src_fire)
                        # Write data
                        writedata.firegrpname = (
                            "D" + str(day.day).zfill(2) + "_" + fire_satellitecontainer.fire_name
                        )
                        writedata.write(
                            fire_satellitecontainer, fire_viirscontainer, fire_plumecontainer
                        )

                        # # Later activate this flag to only save good plumes that are filtered
                        # if fire_plumecontainer.f_firearoundplume:
                        #     # get injection ht
                        #     fire_viirscontainer.insert(
                        #         2,
                        #         "injection_height",
                        #         inj_ht.interpolate(
                        #             fire_viirscontainer.latitude.values,
                        #             fire_viirscontainer.longitude.values,
                        #         ),
                        #     )
                        #     if np.sum(fire_viirscontainer["injection_height"] > 0) > 1:
                        #         f_InjectionHeightExists = True
                        #     else:
                        #         f_InjectionHeightExists = False
                        #     # Append injection height
                        #     writedata.append_injection_ht(
                        #         f_InjectionHeightExists,
                        #         fire_viirscontainer["injection_height"].values,
                        #     )
                        #     # compute emissions
                        #     if f_InjectionHeightExists:
                        #         # compute emissions
                        #         fire_massfluxcontainer = compute_emissions(
                        #             day,
                        #             params,
                        #             fire_satellitecontainer,
                        #             fire_viirscontainer,
                        #             fire_plumecontainer,
                        #         )
                        #         writedata.append_massflux(fire_massfluxcontainer)
            f_id += 1
