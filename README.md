# Automated Plume detection and Emission estimation algorithm (APE) #

Goudar, M., Anema, J. C. S., Kumar, R., Borsdorff, T., and Landgraf, J.: Plume detection and emission estimate for biomass burning plumes from TROPOMI carbon monoxide observations using APE v1.1, Geosci. Model Dev., 16, 4835–4852, https://doi.org/10.5194/gmd-16-4835-2023, 2023.

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


### Installation ###

  * At present, copy the code (src/ape) and example files to a folder.
    * Use cython to compile functions.pyx using setup.py
	 - python setup.py build_ext --inplace
    	 - Move functions.**.so to APE/functions.so 	
    * Examples can be found in examples folder.
    * Runs on python 3.12
  * requirements.txt covers all required packages to run the code.

##### Contact 
  - Manu Goudar: manu.gvm@gmail.com
