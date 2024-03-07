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
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


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


def download_pressureleveldata(area, filetime_fields, cdsapiurl, cdsapikey):
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
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ["geopotential", "u_component_of_wind", "v_component_of_wind"],
                "pressure_level": ["900", "925", "950", "975", "1000"],
                "year": dd[0],
                "month": dd[1],
                "day": dd[2],
                "time": dd[3],
                "area": area},
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
        return (np.sign(ii) + ii)*delta
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
            area = get_area(_key.split("_")[2])
            # download data
            download_pressureleveldata(area, timefile_fields, flow.cdsapiurl, flow.cdsapikey)
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
        

################################################################################################
# Fire
################################################################################################

def clusterpoints(points):
    """Cluster lat-lon.

    Parameters
    ----------
    points : Array(m,2)
        Array of lat-lon

    """
    # Apply DBSCAN clustering
    epsilon = 30  # distance threshold for clustering in deg
    min_samples = 1  # minimum number of points to form a cluster
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = dbscan.labels_
    # Find unique clusters
    unique_labels = np.unique(labels)
    return unique_labels, labels


def _getareamodellvl(cl1_latlon, extent_increase=2):
    min_lat, min_lon = np.min(cl1_latlon, axis=0)
    max_lat, max_lon = np.max(cl1_latlon, axis=0)
    return [min_lat - extent_increase, min_lon - extent_increase,
            max_lat + extent_increase, max_lon + extent_increase]


def _get_minmaxtime(cl1_time):
    cl_time_min = cl1_time.min() - timedelta(hours=7)
    cl_time_min = cl_time_min.replace(minute=0, second=0, microsecond=0)

    cl_time_max = cl1_time.max() + timedelta(hours=1)
    cl_time_max = cl_time_max.replace(minute=0, second=0, microsecond=0)
    
    return cl_time_min, cl_time_max


def create_listoffiles(fires_id, fires_time, points, unique_labels, labels):
    """Cluster data to download

    Parameters
    ----------
    fires_id : List[Str]
        Ids of lat-lon
    fires_time : List[Datatime]
        Time of lat-lons
    points : Array (m,2)
        Lat-lon points
    unique_labels : Array
        Unique labels
    labels : Array(m)
        Labels from clustering

    Return
    --------
    Cluster Dataframe

    """
    #store plumes per cluster and add required information to csv 
    area_l  = []
    cl_id_l = []
    cl_time_min_l = []
    cl_time_max_l = []
    filename_prefix = []
    filename_suffix = []
    _date = fires_time[0].strftime("%Y%m%d")
    # get unique labels
    for _key in unique_labels:
        cl_id_l.append(fires_id[labels == _key])
        # lat lon
        _area = _getareamodellvl(points[labels == _key])
        area_l.append(_area)
        # time data
        cl1_time = fires_time[labels == _key]
        timemin, timemax = _get_minmaxtime(cl1_time)
        cl_time_min_l.append(timemin)
        cl_time_max_l.append(timemax)
        filename_prefix.append(_date)
        filename_suffix.append("_cl"+ str(_key)+"_")
        
    cluster = pd.DataFrame()
    cluster["fires"]=cl_id_l
    cluster["area"]=area_l
    cluster["min_time"]=cl_time_min_l 
    cluster["max_time"]=cl_time_max_l
    cluster["filename_prefix"]=filename_prefix
    cluster["filename_suffix"]=filename_suffix    
    return cluster


def _checkdownload_modellvldata(_area, filetime_fields, flowsuffix, pressuffix,  cdsapiurl, cdsapikey): 
    """Download ERA5 pressure data.

    Parameters
    ----------
    _area : List
        Retrieve bounds in degrees latitude / longitude [North, West, South, East]
    filetime_fields : List
        List containing a list of [year, month, day, time, filename]
    cdsapiurl: String
        String containing url from cdsapi : https://cds.climate.copernicus.eu/api-how-to
    cdsapikey: String
        String containing string from cdsapi : https://cds.climate.copernicus.eu/api-how-to
    """
    breakpoint()
    c = cdsapi.Client(url=cdsapiurl, key=cdsapikey)
    for dd in filetime_fields:
        _file = Path(dd[2]+flowsuffix)
        if not _file.exists():
            print("   Downloading model level velocity....")
            c.retrieve("reanalysis-era5-complete",
                       {"date": dd[0],
                        "levelist": "80/to/137/by/1",
                        "levtype": "ml",
                        "param": "130/131/132/133/135",
                        "stream": "oper",
                        "time": dd[1],
                        "type": "an",
                        "area": _area,
                        "grid": "0.25/0.25",
                        "format": "netcdf"},
                       dd[2]+flowsuffix)
        else:
            print("   Velocity data exists")

        _file1 = Path(dd[2]+pressuffix)
        if not _file1.exists():
            print("   Downloading model level surface pressure....")
            c.retrieve("reanalysis-era5-complete",
                       {"date": dd[0],
                        "levelist": "1",
                        "levtype": "ml",
                        "param": "129/152",
                        "stream": "oper",
                        "time": dd[1],
                        "type": "an",
                        "area": _area,
                        "grid": "0.25/0.25",
                        "format": "netcdf"},
                       dd[2]+pressuffix)
        else:
            print("   Surface pressure data exists")


def _check_create_dir(mydir):
    _path = Path(mydir)
    if not _path.exists():
        _path.mkdir(parents=True)
        

def download_modelleveldata(inputfilename, flow, _day, cdsapiurl, cdsapikey):
    """Check for velocity data and download it if required. 

    Downloads the velocity data based on the input.

    Parameters
    ----------
    inputfilenaame : string
        Contains file name of the fire data (output of APE)
    outputdir : String
        Output dir for model level data params.estimateemission.flow.inputdir
    day : Datetime
        Date from date time
    cdsapiurl: String
        String containing url from cdsapi : https://cds.climate.copernicus.eu/api-how-to
    cdsapikey: String
        String containing string from cdsapi : https://cds.climate.copernicus.eu/api-how-to
    """
    # load data for a day based on input datafile
    _read = ReadData(inputfilename, _day, True)
    _points, fires_id, fires_time = _read.datafordownloadml()

    if not len(fires_id) > 0:
        print("No plumes detected for this day")
        return
    
    # create cluster of firesources for a day
    unique_labels, labels = clusterpoints(_points)
    cluster = create_listoffiles(fires_id, fires_time, _points, unique_labels, labels)

    # create directories if they do not exist
    new_outdir = flow.inputdir + _day.strftime("%Y/%m/%d") + "/"
    _check_create_dir(new_outdir)
    # write cluster data
    cluster.to_csv(new_outdir +"cluster_table.csv")
    
    # download and check data
    for i in range(len(cluster)):
        dd = cluster.loc[i]
        # if measure time is a day before change data 
        if dd.min_time.day != dd.max_time.day:
            _date = dd.min_time.strftime("%Y-%m-%d")
            _time = dd.min_time.strftime("%H")+"/to/23/by/1"
            d1 = [_date, _time, new_outdir + dd.filename_prefix+"_0"+dd.filename_suffix]

            _date = dd.max_time.strftime("%Y-%m-%d")
            _time = "00/to/"+ dd.max_time.strftime("%H")+"/by/1"
            d2 = [_date, _time, new_outdir +dd.filename_prefix+"_1"+dd.filename_suffix]
            filetime_fields = [d1, d2]
        else:
            _date = dd.max_time.strftime("%Y-%m-%d")
            _time = dd.min_time.strftime("%H") + "/to/" + dd.max_time.strftime("%H")+"/by/1"
            filetime_fields = [[_date, _time, new_outdir + dd.filename_prefix + dd.filename_suffix]]
        # check if the velocity fields exist or not and download
        print("Checking/downloading model level data for following fires: ", dd.fires)
        _checkdownload_modellvldata(dd.area, filetime_fields, flow.file_flow_suffix,
                                    flow.file_pres_suffix, cdsapiurl, cdsapikey)
        print("                            Done!")
