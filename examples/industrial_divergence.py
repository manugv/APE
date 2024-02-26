import sys
sys.path.insert(0, "/data/manugv/github/APE")
import APE.Industrial as ind
from APE.ModCheckDownloadVelocity import checkanddownloadvelocity_alldata

filename = "steelplant_divergence.yaml"
# Read input file
params = ind.InputParameters(filename)
# Initialize a file to write data
writedata = ind.WriteData(params.output_file)

# PREPROCESSING
# prepare data prepare
# ind.preprocessdata(params, writedata)

# Get/check velocity fields for good data
# download velocity fields
# checkanddownloadvelocity_alldata(params)

# Estimate divergence
meandiv, temp_div_list = ind.computedivergence(params)
