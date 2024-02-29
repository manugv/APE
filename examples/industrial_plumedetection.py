import sys
sys.path.insert(0, "/data/manugv/github/APE")

import APE.Industrial as ind
from APE.ModuleInitialParameters import InputParameters


filename = "steelplant_plumedetection.yaml"

# Read input file
params = InputParameters(filename)
# Initialize a file to write data
writedata = ind.WriteData(params.output_file)

# prepare data prepare
ind.preprocessdata(params, writedata)

# plume detection
ind.detectplume(params, writedata)
