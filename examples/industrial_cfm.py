import sys
sys.path.insert(0, "../../APE")


from APE.ModuleInitialParameters import InputParameters
from APE.ModCheckDownloadVelocity import checkanddownloadvelocity_alldata
import APE.Industrial as ind

filename = "steelplant_cfm.yaml"

# Read input file
params = InputParameters(filename)
# Initialize a file to write data
writedata = ind.WriteData(params.output_file)

# prepare data prepare
ind.preprocessdata(params, writedata)

# plume detection
ind.detectplume(params, writedata)

# Get/check velocity fields for good data
# download velocity fields
checkanddownloadvelocity_alldata(params, onlyplumes=True)

# emission estimation for all plumes
ind.estimatecfmemission(params, writedata)
