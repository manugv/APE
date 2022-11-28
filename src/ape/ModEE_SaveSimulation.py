#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save Simulation function
@author: Manu Goudar
"""

from turtle import shape
import h5py
import numpy as np
import ModEE_Xdmf as mx


# XDMF FORMAT
# WRITE XDMF FILE
def __create_topo_geo(domain, topo_name, geo_name, mesh, dims, root_loc):
    mx.createtopology(domain, topo_name, mesh, dims)
    # Create geometry
    geo = mx.creategeometry(domain, geo_name, "X_Y_Z")
    mx.createdataitem(
        geo,
        Name="X",
        Dimensions=dims,
        NumberType="Float",
        Precision="8",
        Format="HDF",
        datafile=root_loc + "X",
    )
    mx.createdataitem(
        geo,
        Name="Y",
        Dimensions=dims,
        NumberType="Float",
        Precision="8",
        Format="HDF",
        datafile=root_loc + "Y",
    )
    mx.createdataitem(
        geo,
        Name="Z",
        Dimensions=dims,
        NumberType="Float",
        Precision="8",
        Format="HDF",
        datafile=root_loc + "Z",
    )


def __create_timecollection(
    domain,
    grid_name,
    blocks,
    time,
    dims,
    root_loc,
    topology_ref,
    geometry_ref,
    attr_name,
):

    timegrid = mx.creategrid(
        domain, Name=grid_name, GridType="Collection", CollectionType="Temporal"
    )
    # time loop of grids
    for i in range(blocks):
        timestep = mx.creategrid(
            timegrid, Name="t_" + str(i), GridType="Uniform", CollectionType=None
        )
        mx.createtimevalue(timestep, str(time[i]))
        mx.createreference(timestep, "Topology", topology_ref)
        mx.createreference(timestep, "Geometry", geometry_ref)
        attr = mx.createattribute(timestep, Name=attr_name, att_type="Scalar", center="Cell")
        mx.createdataitem(
            attr,
            Name="Conc_" + str(i),
            Dimensions=dims,
            NumberType="Float",
            Precision="8",
            Format="HDF",
            datafile=root_loc + str(i),
        )


def __create_particleseries(domain, name, particleinfo, fileloc):

    timegrid = mx.creategrid(domain, Name=name, GridType="Collection", CollectionType="Temporal")
    # time loop of grids
    for i in range(len(particleinfo)):
        dataloc = fileloc + "Step" + str(i) + "/"
        sz = particleinfo[i][0]
        time = particleinfo[i][1]
        timestep = mx.creategrid(
            timegrid, Name="Step" + str(i), GridType="Uniform", CollectionType=None
        )

        mx.createtimevalue(timestep, str(time))
        # Create topology
        mx.createtopology(timestep, name=None, topologytype="Polyvertex", NodesPerElement=str(sz))
        # Create Geometry
        geo = mx.creategeometry(timestep, name=None, geotype="XYZ")
        dims = str(sz) + " 3"
        mx.createdataitem(
            geo,
            Name=None,
            Dimensions=dims,
            NumberType="Float",
            Precision="8",
            Format="HDF",
            datafile=dataloc + "position",
        )

        attr = mx.createattribute(timestep, Name="GlobalIds", att_type="Scalar", center="Node")
        mx.createdataitem(
            attr,
            Name=None,
            Dimensions=str(sz),
            NumberType="Int",
            Precision="4",
            Format="HDF",
            datafile=dataloc + "GlobalIds",
        )
        attr = mx.createattribute(timestep, Name="SourceIds", att_type="Scalar", center="Node")
        mx.createdataitem(
            attr,
            Name=None,
            Dimensions=str(sz),
            NumberType="Int",
            Precision="4",
            Format="HDF",
            datafile=dataloc + "SourceIds",
        )
        attr = mx.createattribute(timestep, Name="VerticalIds", att_type="Scalar", center="Node")
        mx.createdataitem(
            attr,
            Name=None,
            Dimensions=str(sz),
            NumberType="Int",
            Precision="4",
            Format="HDF",
            datafile=dataloc + "VerticalIds",
        )
        attr = mx.createattribute(timestep, Name="Mass", att_type="Scalar", center="Node")
        mx.createdataitem(
            attr,
            Name=None,
            Dimensions=str(sz),
            NumberType="Float",
            Precision="8",
            Format="HDF",
            datafile=dataloc + "mass",
        )


# HDF5 write
def _arrayflip(data):
    sh = data.shape
    newdata = np.zeros(np.flip(sh))
    for i in range(sh[0]):
        for j in range(sh[1]):
            newdata[:, j, i] = data[i, j, :]
    return newdata


def __writeT2C_hdf(grp, conc, scale, transform):
    # Concentration data from trajectories
    X, Y, Z = np.meshgrid(conc.lat, conc.lon, conc.z, indexing="ij")
    if transform is not None:
        for k in range(X[0, 0, :].size):
            for j in range(X[0, :, 0].size):
                for i in range(X[:, 0, 0].size):
                    X[i, j, k], Y[i, j, k] = transform(X[i, j, k], Y[i, j, k])

    # Add surface position to the z coordinate (make z absolute)
    for i in range(Z[0, 0, :].size):
        Z[:, :, i] += conc.surface[:, :, 2]

    # Write geometry
    grp.create_dataset("X", data=_arrayflip(X / scale[0]))
    grp.create_dataset("Y", data=_arrayflip(Y / scale[1]))
    grp.create_dataset("Z", data=_arrayflip(Z / scale[2]))
    lvl = grp.create_group("Concs")
    # Write concentrations
    for i in range(conc.blocks):
        d_name = "t_" + str(i)
        dset = lvl.create_dataset(d_name, data=_arrayflip(conc.final_conc[:, :, :, i]))
        dset.attrs["TimeValue"] = np.array([conc.end_t[i] * 1.0])


def __writeTropomi_orig(grp1, codata, scale, transform, topology):
    X1 = codata.lat.copy()
    Y1 = codata.lon.copy()
    Z1 = np.zeros(X1.shape)
    if transform is not None:
        for j in range(X1[0, :].size):
            for i in range(X1[:, 0].size):
                X1[i, j], Y1[i, j] = transform(codata.lat[i, j], codata.lon[i, j])
    # Get Surface topology of COData
    topology.fh.bounds_error = False
    for j in range(X1[0, :].size):
        for i in range(X1[:, 0].size):
            Z1[i, j] = topology((codata.lat[i, j], codata.lon[i, j]))
    topology.fh.bounds_error = True
    # Write geometry
    grp1.create_dataset("lat", data=codata.lat)
    grp1.create_dataset("lon", data=codata.lon)
    # Transpose these quantities to arrange in ZYX format for XDMF
    grp1.create_dataset("X", data=X1.T / scale[0])
    grp1.create_dataset("Y", data=Y1.T / scale[1])
    grp1.create_dataset("Z", data=Z1.T / scale[2])

    # Write concentrations
    grp1.create_dataset("Satellite", data=codata.co.T)


def __writeTropomi_computed(grp1, conc):
    lvl2 = grp1.create_group("Computed")
    for i in range(conc.blocks):
        d_name = "t_" + str(i)
        conc_flat = np.sum(conc.mapped_data[i], axis=2)
        dset = lvl2.create_dataset(d_name, data=conc_flat.T)
        dset.attrs["TimeValue"] = np.array([conc.end_t[i] * 1.0])


def __get_time_simulation(release):
    c = None
    for j in range(len(release)):
        level = release[j]
        if c is None:
            c = level.time + level.start_time
        else:
            c = np.concatenate((c, level.time + level.start_time))
    return np.unique(c)


def __write_data(pos, mass, ids, g, rls_id):
    if "position" in g.keys():
        nos = pos.shape[0]
        g["position"].resize(g["position"].shape[0] + nos, axis=0)
        g["position"][-nos:, :] = pos
        g["mass"].resize(g["mass"].shape[0] + nos, axis=0)
        g["mass"][-nos:] = mass
        g["GlobalIds"].resize(g["GlobalIds"].shape[0] + nos, axis=0)
        g["GlobalIds"][-nos:] = ids[:, 0]
        g["SourceIds"].resize(g["SourceIds"].shape[0] + nos, axis=0)
        g["SourceIds"][-nos:] = ids[:, 1]
        g["VerticalIds"].resize(g["VerticalIds"].shape[0] + nos, axis=0)
        g["VerticalIds"][-nos:] = ids[:, 2]
        g["ReleaseIds"].resize(g["ReleaseIds"].shape[0] + nos, axis=0)
        g["ReleaseIds"][-nos:] = rls_id
    else:
        g.create_dataset("position", data=pos, maxshape=(None, 3))
        g.create_dataset("mass", data=mass, maxshape=(None,))
        g.create_dataset("GlobalIds", data=ids[:, 0], maxshape=(None,))
        g.create_dataset("SourceIds", data=ids[:, 1], maxshape=(None,))
        g.create_dataset("VerticalIds", data=ids[:, 2], maxshape=(None,))
        g.create_dataset("ReleaseIds", data=rls_id, maxshape=(None,))


def __writeParticle_hdf(root, data, scale, transform):
    """ """
    # Find total time array
    time = __get_time_simulation(data[0])
    # create all groups
    for t in range(time.size):
        grp_name = "Step" + str(t)
        grp = root.create_group(grp_name)
        grp.attrs["TimeValue"] = np.array([time[t] * 1.0])

    # write data is groups
    for i in range(len(data)):
        rel = data[i]
        for j in range(len(rel)):
            lvl = rel[j]
            for t in range(lvl.time.size):
                tt = lvl.time[t] + lvl.start_time
                # Get group name
                g = root["Step" + str(np.argwhere(time == tt)[0][0])]
                # Convert lat, lon to mts
                pos = lvl.pos[:, t, :].copy()
                if transform is not None:
                    pos[:, 0], pos[:, 1] = transform(pos[:, 0], pos[:, 1])
                # Scale all positions everything
                pos[:, 0] /= scale[0]
                pos[:, 1] /= scale[1]
                pos[:, 2] /= scale[2]
                rls_id = np.ones(pos.shape[0], dtype=np.int_) * i
                # Write all data
                __write_data(pos, lvl.mass, lvl.ids, g, rls_id)
    # Get the no of particles in each step for XDMF
    part_size = []
    for t in range(time.size):
        g = root["Step" + str(t)]
        part_size.append([g["position"].shape[0], time[t]])
    return part_size


def _write_data(pos, mass, ids, time, g):
    g.create_dataset("position", data=pos)
    g.create_dataset("mass", data=mass)
    g.create_dataset("time", data=time)
    g.create_dataset("globalids", data=ids[:, 0])
    g.create_dataset("sourceids", data=ids[:, 1])
    g.create_dataset("verticalids", data=ids[:, 2])


def _writeParticle_hdf(root, data, scale, transform):
    """ """
    # write data first releases
    for i in range(len(data)):
        rel = data[i]
        rls_name = "Release" + str(i).zfill(3)
        rls_grp = root.create_group(rls_name)
        # write levels in releases
        for j in range(len(rel)):
            lvl = rel[j]
            lvl_name = "Level" + str(j).zfill(2)
            lvl_grp = rls_grp.create_group(lvl_name)
            # Write data
            time = lvl.time + lvl.start_time
            pos = lvl.pos.copy()
            if transform is not None:
                for k in range(pos, shape[0]):
                    pos[k, :, 0], pos[k, :, 1] = transform(pos[k, :, 0], pos[k, :, 1])
            # Scale all positions everything
            pos[:, :, 0] /= scale[0]
            pos[:, :, 1] /= scale[1]
            pos[:, :, 2] /= scale[2]
            # Write all data
            _write_data(pos, lvl.mass, lvl.ids, time, lvl_grp)


def save_concentrationdata(
    particledata,
    simulationname="simulation",
    codata=None,
    conc=None,
    scale=None,
    transform=None,
    topology=None,
):
    """
    Write concentration data from trajectories and conc to tropomi
    data along with actual tropomi data in 2d
    """
    simfilename = simulationname + ".h5"

    # Create root for HDF5 file
    f = h5py.File(simfilename, "a")
    # create root and domain for XDMF file
    # root = mx.createheader()
    # domain = mx.createdomain(root)

    # Write Particle data
    # HDF5
    # part = f.create_group("Particles")
    _writeParticle_hdf(f, particledata, scale, transform)
    f.close()
    # XDMF
    # _root_particles = simfilename + ":/Particles/"
    # __create_particleseries(domain, "Particles", particleinfo, _root_particles)

    # # Write Tropomi data Original
    # # HDF5
    # grp_trop = "Tropomi"
    # grp1 = f.create_group(grp_trop)
    # __writeTropomi_orig(grp1, codata, scale, transform, topology)
    # # XDMF
    # _root_tropomi = simfilename + ":/Tropomi/"
    # # define dimensions in string format
    # dims = (codata.co.shape)[::-1]
    # tropomi_topodims = str(dims[0] + 1) + " " + str(dims[1] + 1)
    # tropomi_concdims = str(dims[0]) + " " + str(dims[1])
    # __create_topo_geo(
    #     domain, "Topology1", "Geo1", "2DSMesh", tropomi_topodims, _root_tropomi
    # )
    # # Create Uniform, Spatial grid
    # orig_grid = mx.creategrid(
    #     domain, Name="TropomiMeasured", GridType="Uniform", CollectionType="Spatial"
    # )
    # mx.createreference(orig_grid, "Topology", "Topology[1]")
    # mx.createreference(orig_grid, "Geometry", "Geometry[1]")
    # attr = mx.createattribute(
    #     orig_grid, Name="MeasuredTropomiConc", att_type="Scalar", center="Cell"
    # )
    # flname = _root_tropomi + "Satellite"
    # mx.createdataitem(
    #     attr,
    #     Name="Conc",
    #     Dimensions=tropomi_concdims,
    #     NumberType="Float",
    #     Precision="8",
    #     Format="HDF",
    #     datafile=flname,
    # )

    # if conc is not None:
    #     # Write Tropomi computed
    #     # HDF 5
    #     __writeTropomi_computed(grp1, conc)
    #     # XDMF
    #     # Create time collection
    #     __create_timecollection(
    #         domain,
    #         "TropomiComputed",
    #         conc.blocks,
    #         conc.end_t,
    #         tropomi_concdims,
    #         _root_tropomi + "Computed/t_",
    #         "Topology[1]",
    #         "Geometry[1]",
    #         "ComputedTropomiConc",
    #     )

    #     # Write Tajectories to conc data
    #     grp_conc = "Traj2Conc"
    #     grp = f.create_group(grp_conc)
    #     __writeT2C_hdf(grp, conc, scale, transform)
    #     # XDMF
    #     _root_traj2conc = simfilename + ":/Traj2Conc/"
    #     # define dimensions in string format and flip array sizes
    #     dims = (conc.concentrations.shape)[::-1]
    #     t2c_topodims = str(dims[0] + 1) + " " + str(dims[1] + 1) + " " + str(dims[2] + 1)
    #     t2c_concdims = str(dims[0]) + " " + str(dims[1]) + " " + str(dims[2])
    #     # create topology and geometry for Trajectories to concentrations
    #     __create_topo_geo(domain, "Topology2", "Geo2", "3DSMesh", t2c_topodims, _root_traj2conc)

    #     # Create time collection for Trajectories to concentrations
    #     __create_timecollection(
    #         domain,
    #         "T2C_Time",
    #         conc.blocks,
    #         conc.end_t,
    #         t2c_concdims,
    #         _root_traj2conc + "Concs/t_",
    #         "Topology[2]",
    #         "Geometry[2]",
    #         "ConcFromTraj",
    #     )
    # # Write header file
    # mx.writexml(root, simulationname)
    # f.close()
