import sys
sys.path.insert(0, "../APE")
import APE.ModuleRunIndustrial as ind

filename = "example.yaml"
# Read input file
params = ind.InputParameters(filename)
# Initialize a file to write data
writedata = ind.WriteData(params.output_file)

# prepare data prepare
ind.preparedata(params, writedata)

# Initialize a class method to read data
_readdata = ind.ReadData(params.output_file, params.days)
# read satellite data
datatoread = ["Satellite"]

# Detect plume for all good data
for _key in _readdata.keys:
    data = _readdata.read(_key, datatoread)
    plumecontainer = ind.segment_image_plume(data.lat, data.lon, data.co_column_corr,
                                             data.co_qa_mask, params.transform)
    # write data
    writedata.write_plumedata(plumecontainer)

    # check if plume was detected or not
    # if not detected then go to next key
    if not plumecontainer.flag_plumedetected:
        continue

# download velocity for all keys
for _key in _readdata.keys:
    # download velocity fields
    ind.checkanddownloadvelocity(_key, params.estimateemission.plumeheight,
                                 params.estimateemission.flow.inputdir, params.ind_source_name)

# emission estimation for all plumes
datatoread = ["Satellite", "PlumeDetection"]
for _key in _readdata.keys:
    data = _readdata.read(_key, datatoread)
    # plume was detected
    massflux = ind.crosssectionalflux(params, data, data.plumecontainer, params.transform)
    # writedata.write_cfm(massflux)
    # TODO: Add variable to define individual emission
    try:
        cfm.append([massflux.emission, data.measurement_time, data.orbit])
    except NameError:
        cfm = [massflux.emission, data.measurement_time, data.orbit]
