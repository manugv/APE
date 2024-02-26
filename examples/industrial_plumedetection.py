import sys
sys.path.insert(0, "../APE")
import APE.ModuleRunIndustrial as ind

filename = "example.yaml"
# Read input file
params = ind.InputParameters(filename)
# Initialize a file to write data
writedata = ind.WriteData(params.output_file)

# prepare data prepare
ind.preparedata(params)

# plume detection
ind.detectplume(params, writedata)
