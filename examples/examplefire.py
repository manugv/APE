import sys
sys.path.insert(1, "/data/manugv/github/APE")

from APE.ModuleInitialParameters import InputParameters
import APE.Fire as iape
from APE.ModCheckDownloadVelocity import download_modelleveldata
from APE.ModuleInjectionHeight import download_injectionheight, InjectionHeight

filename = "/data/manugv/github/APE/examples/fire_example.yaml"

params = InputParameters(filename)

_day = params.days[0]

# plume detection
nogoodplumes= iape.datapreparation_plumedetection(params.days[0], params)

# check/download velocity data
inputfilename = params.output_file_prefix + _day.strftime("%Y_%m")
download_modelleveldata(inputfilename, params.estimateemission.flow, _day,
                        params.estimateemission.flow.cdsapiurl, params.estimateemission.flow.cdsapikey)

# check/download injection height data
emis = params.estimateemission
download_injectionheight(emis.injht_dir, _day, emis.injht_adsapiurl, emis.injht_adsapikey)

#emission estimation
iape.emissionestimationfires(_day, params)

