#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:00:00 2024.

@author: Arthur B, Manu Goudar
"""
from .ModuleRead import ReadData
from pathlib import Path
import cdsapi
from datetime import datetime, timedelta
from numpy import sign 

def timefilename_download(_kydatetime, filename):
    """Define date, time, filename to download.

    List of [[year, month, day, time, filename]] is defined to download ERA5 data.

    Parameters
    ----------
    _kydatetime : String
        String containing date_time; YYYYMMDD_HHMM
    filename : String
        Fuill path of the filename

    """
    
    # convert the string to datetime format
    measuretime = datetime.strptime(_kydatetime, "%Y%m%d_%H%M")
    # if the data is at the end of the day then down data for next day
    measuretime1 = measuretime + timedelta(hours=1)
    if measuretime.hour == 23:
        d1 = [measuretime.strftime("%Y"), measuretime.strftime("%m"), measuretime.strftime("%d"),
              measuretime.strftime("%H"), filename+"_1.nc"]
        d2 = [measuretime1.strftime("%Y"), measuretime1.strftime("%m"), measuretime1.strftime("%d"),
              measuretime1.strftime("%H"), filename+"_2.nc"]
        downloadtime = [d1, d2]
    else:
        d1 = [measuretime.strftime("%Y"), measuretime.strftime("%m"),  measuretime.strftime("%d"),
              [measuretime.strftime("%H"), measuretime1.strftime("%H")], filename+".nc"]
        downloadtime = [d1]
    return downloadtime


def download_era_pressure(area, filetime_fields, cdsapiurl, cdsapikey):
    """Download ERA5 pressure data.

    Parameters
    ----------
    area : List
        Retrieve bounds in degrees latitude / longitude [North, West, South, East]
    filetime_fields : List
        List containing a list of [year, month, day, time, filename]
    cdsapiurl: String
        String containing url from cdsapi : https://cds.climate.copernicus.eu/api-how-to
    cdsapikey: String
        String containing string from cdsapi : https://cds.climate.copernicus.eu/api-how-to
    """
    print("downloading_in_progress")
        
    c = cdsapi.Client(url=cdsapiurl, key=cdsapikey)
    # loop over different days if needed
    for dd in filetime_fields:
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['geopotential', 'u_component_of_wind', 'v_component_of_wind'],
                'pressure_level': ['900', '925', '950', '975', '1000'],
                'year': dd[0],
                'month': dd[1],
                'day': dd[2],
                'time': dd[3],
                'area': area},
            dd[4])


def roundoff(num, _upper=False, delta=0.25):
    """Round off numbers to a certain precision.

    Purely done to get data at certain lat-lon grids

    Parameters
    ----------
    num : Float
        Number to be rounded off
    _upper : Bool
        Says should it be upper limit or lower limit
    delta : Float, default=0.25 degprecision
        the precision to be rounded off to.

    """
    if _upper:
        ii = num//delta
        return (sign(ii) + ii)*delta
    else:
        return (num//delta)*delta
   
    
    

def get_area(latlon):
    """Compute the bounds to retrieve data.

    Parameters
    ----------
    _ky : String
        String containing lat and lon

    """
    plume_location_lat = float(latlon[1:6])
    if latlon[0] == "S":
        plume_location_lat = -plume_location_lat

    plume_location_lon = float(latlon[7:])
    if latlon[6] == "W":
        plume_location_lon = -plume_location_lon

    # [North, West, South, East] and convert decimals of lat-lon to it to 0.25 deg similar to resolutions of data
    maxlat = plume_location_lat + 2
    maxlat = ((maxlat//0.25) + 1)*0.25

    return [roundoff(plume_location_lat + 2, True), roundoff(plume_location_lon - 2),
            roundoff(plume_location_lat - 2), roundoff(plume_location_lon + 2, True)]


def checkanddownloadvelocity(_key, plumeheight, flow, source_name):
    """Check and download velocity data

    Parameters
    ----------
    _key : String
        Data Key
    plumeheight : String
        params.estimateemission.plumeheight
    flow : Class
         params.estimateemission.flow, class containing flow details
    source_name : String
        params.ind_source_name

    """
    # check if the velocity fields exist or not and download

    if (plumeheight == 10) or (plumeheight == 100):
        filename = flow.inputdir + _key[:4] + "/" + "sl_"+_key[:8]+".grib"
        if not (Path(filename)).is_file():
            print("Plume height data is not present")
        else:
            print("        Velocity data exists")
    else:
        filename = flow.inputdir + source_name + "_" + _key[:13]
        if not (Path(filename  + ".nc")).is_file():
            print("      Downloading data")
            # get time and filename
            timefile_fields = timefilename_download(_key[:13], filename)
            # get area
            area = get_area(_key.split('_')[2])
            # download data
            download_era_pressure(area, timefile_fields, flow.cdsapiurl, flow.cdsapikey)
            print("             .....Done")
        else:
            print("        Velocity data exists")


def checkanddownloadvelocity_alldata(params, onlyplumes=False):
    """Check for velocity data and download it if required. 

    Downloads the velocity data based on the input.

    Parameters
    ----------
    params : InputParameters class
        Contains input paramters
    onlyplumes : Bool
        Download velocity fields for only detected plumes.

    """
    # get data for which velocity is needed
    _readdata = ReadData(params.output_file, params.days, onlyplumes)
    
    # check if the velocity fields exist or not and download
    for _ky in _readdata.keys:
        print("   Checking velcity data for ", _ky)
        checkanddownloadvelocity(_ky, params.estimateemission.plumeheight,
                                 params.estimateemission.flow, params.source_name)
        
