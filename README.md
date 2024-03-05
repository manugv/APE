# ape #

APE is Automated Plume detection and Emission estimation algorithm.

### What is this repository for? ###
  - To compute Carbon Monoxide (CO) emissions for fires/industrial sources.
  - It integrates four freely available data sources, VIIRS active fire dataset, TROPOMI CO dataset, injection height from GFAS and ERA5 meteorological data.
    - TROPOMI CO dataset: ftp://ftp.sron.nl/open-access-data-2/TROPOMI/tropomi/co/
    - VIIRS active fire dataset: https://firms.modaps.eosdis.nasa.gov/active_fire/
    - GFAS: https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-fire-emissions-gfas
    - ERA5: https://apps.ecmwf.int/data-catalogues/era5 
  - Can also be used to identify fires by integrating VIIRS active fire dataset and TROPOMI CO dataset

### Documentation ###

* Detail documentation will be made available with time in docs.

##### Versions
  - X.YY
      - X is based on an algorithm and does not change with minor versions.
          - X <= 1 is based on the paper: Plume detection and estimate emissions for biomass burning plumes from TROPOMI Carbon monoxide observations using APE (2022) 
      - XX refers to
          - bug fixes,
          - removal of deprecated functions/methods,
          - change in output files (like adding units),
          - codes to automatically download the data, and
          - converting code to cython (if needed).


### Installation ###

  * At present, copy the code (src/ape) and example files to a folder.
    * Use cython to compile functions.pyx using setup.py
	  - python setup.py build_ext --inplace
    * Examples can be found in examples folder.
    * Runs on python 3.12
  * requirements.txt covers all required packages to run the code.

##### Contact 
  - Manu Goudar: m.goudar@sron.nl
  - Tobias Borsdorff: t.borsdorff@sron.nl
