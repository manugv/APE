#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:07:00 2022.

@author: Manu Goudar
"""

import h5py


class WriteData:
    def __init__(self, filename):
        self.filename = filename + ".h5"
        self.grpname = ""
        self.preprocessgrp = "Preprocess"
        self.preprocessviirsgrp = "RawVIIRS"
        self.satellitegrpname = "Satellite"
        self.viirsgrpname = "VIIRS"
        self.plumedetectgrpname = "PlumeDetection"
        self.cfmgrpname = "CFM"
        self.grpdivergence = "Divergence"
        self.tlinesgrp = "Transactionlines"

    def updatefilename(self, newname):
        self.filename = newname + ".h5"

    def get_group(self, r_grp, grpname):
        if grpname in r_grp.keys():
            return r_grp[grpname]
        # else create the group
        else:
            return r_grp.create_group(grpname)

    def preprocess_viirs(self, _key, data, _flags):
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, self.preprocessgrp)
        _rawgrp = self.get_group(_grp, self.preprocessviirsgrp)
        for _ky in list(data.columns):
            _rawgrp.create_dataset(_ky, data=data[_ky].values)
        for _ky, val in _flags.items:
            _rawgrp.create_dataset(_ky, data=val)
        fl.close()

        
    def preprocess_satelliteorbit(self, _day, data, _flags):
        # only for fires
        pass
        
    def satellite(self, uniqueid, data):
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, uniqueid)
        _satgrp = self.get_group(_grp, self.satellitegrpname)
        for _ky, _val in data.__dict__.items():
            if _ky not in ["uniqueid"]:
                if _ky == "measurement_time":
                    _satgrp.create_dataset(_ky, data=_val.strftime("%Y/%m/%d_%H:%M:%S"))
                elif "co_column" in _ky:
                    # mask data needs to be written
                    _satgrp.create_dataset(_ky, data=_val.data)
                    _satgrp.create_dataset(_ky+"_mask", data=_val.mask)
                else:
                    _satgrp.create_dataset(_ky, data=_val)
        fl.close()

    def divergence(self, uniqueid, divsimname, data, grid):
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, uniqueid)
        _divgrp = self.get_group(_grp, self.grpdivergence)
        _simgrp = self.get_group(_divgrp, divsimname)
        for _ky, _val in data.__dict__.items():
            _simgrp.create_dataset(_ky, data=_val)
        for _ky in ["lat", "lon"]:
            _simgrp.create_dataset(_ky, data=grid.__dict__[_ky])
        fl.close()

    def plume(self, uniqueid, data):
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, uniqueid)
        _plmgrp = self.get_group(_grp, self.plumedetectgrpname)
        for _ky, _val in data.__dict__.items():
            _plmgrp.create_dataset(_ky, data=_val)
        fl.close()

    def write_cfm(self, uniqueid, data):
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, uniqueid)
        _cfmgrp = self.get_group(_grp, self.cfmgrpname)
        for _ky, _val in data.__dict__.items():
            if "tlines" in _ky:
                _tlgrp = self.get_group(_cfmgrp, self.tlinesgrp)
                # write data for each transaction line in a group
                i = 0
                for _line in _val:
                    _lgrp = self.get_group(_tlgrp, "line_"+str(i).zfill(2))
                    self._writetlines(_lgrp, _line)
                    i+=1
            else:
                _cfmgrp.create_dataset(_ky, data=_val)

    def _writetlines(self, grp, data):
        for _ky, _val in data.__dict__.items():
            if "emission" in _ky:
                _emgrp = self.get_group(grp, _ky)
                for _ky1, _val1 in _val.__dict__.items():
                    _emgrp.create_dataset(_ky1, data=_val1)
            else:
                grp.create_dataset(_ky, data=_val)

    
    def viirs(self, uniqueid, data):
        # open file
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, uniqueid)
        viirs_grp = self.get_group(_grp, self.viirsgrpname)
        for _ky in list(data.columns):
            viirs_grp.create_dataset(_ky, data=data[_ky].values)
        fl.close()

    def injection_ht(self, uniqueid, flag, injection_height):
        # open file
        fl = h5py.File(self.filename, "a")
        _grp = self.get_group(fl, uniqueid)
        viirs_grp = self.get_group(_grp, self.viirsgrpname)
        viirs_grp.create_dataset("flag_injectionheightexists", flag)
        if flag:
            viirs_grp.create_dataset("injectionheight", data=injection_height)
        fl.close()

    # def write(self, sat_data, fire_data, plume_data):
    #     # file name is the name of the file based on month
    #     # day is the day on which fire was detected
    #     # fire number of the day
    #     # data contains aeverything
    #     satellite_data_attrs = [
    #         "orbit",
    #         "orbit_filename",
    #         "fire_name",
    #         "fire_id",
    #         "orbit_ref_time",
    #         "source",
    #         "f_source_in_orbit",
    #         "source_pixel_id",
    #         "f_filter_gridsize",
    #         "f_not_all_nans",
    #         "f_qavalue_filter",
    #         "f_good_satellite_data",
    #     ]
    #     satellite_data_keys = [
    #         "lat",
    #         "lon",
    #         "qa_value",
    #         "co_column",
    #         "co_column_corr",
    #         "lat_corners",
    #         "lon_corners",
    #         "lat_nodes",
    #         "lon_nodes",
    #         "aerosol_thick",
    #         "avg_kernel",
    #         "co_qa_mask",
    #         "deltatime",
    #     ]
    #     viirs_data_keys = [
    #         "latitude",
    #         "longitude",
    #         "scan",
    #         "track",
    #         "acq_date",
    #         "acq_time",
    #         "satellite",
    #         "instrument",
    #         "confidence",
    #         "version",
    #         "frp",
    #         "daynight",
    #     ]
    #     plume_detection_attrs = ["f_plumedetect", "f_nofirearoundplume"]
    #     plume_detection_keys = ["plumemask", "labeled_img", "segmented_img"]
    #     # Debug true
    #     # ['elevation_map', 'marker_img', 'label_id']
    #     # open file
    #     fl = h5py.File(self.filename, "a")

    #     # if group exists then read the name of group
    #     fire_grp = self.get_group(fl, self.grpname)

    #     # satellite data
    #     st_grp = fire_grp.create_group(self.satellitegrpname)
    #     for _ky in satellite_data_attrs:
    #         st_grp.attrs[_ky] = sat_data.__getattribute__(_ky)

    #     for _ky in satellite_data_keys:
    #         st_grp.create_dataset(_ky, data=sat_data.__getattribute__(_ky))

    #     # viirs_data
    #     viirs_grp = fire_grp.create_group(self.viirsgrpname)
    #     for _ky in viirs_data_keys:
    #         viirs_grp.create_dataset(_ky, data=fire_data[_ky].values)

    #     # Plume detection
    #     plume_grp = fire_grp.create_group(self.plumedetectgrpname)
    #     for _ky in plume_detection_attrs:
    #         plume_grp.attrs[_ky] = plume_data.__getattribute__(_ky)
    #     for _ky in plume_detection_keys:
    #         plume_grp.create_dataset(_ky, data=plume_data.__getattribute__(_ky))
    #     fl.close()

    # def append_injection_ht(self, flag, injection_height):
    #     """append_injection_ht _summary_

    #     Args:
    #         flag (_type_): _description_
    #         injection_height (_type_): _description_
    #     """
    #     # open file
    #     fl = h5py.File(self.filename, "a")
    #     fire_grp = self.get_group(fl, self.firegrpname)
    #     viirs_grp = self.get_group(fire_grp, self.viirsgrpname)
    #     viirs_grp.attrs["f_InjectionHeightExists"] = flag
    #     if flag:
    #         viirs_grp.create_dataset("injectionheight", data=injection_height)
    #     fl.close()

    # def append_massflux(self, massflux_data):
    #     """append_massflux _summary_

    #     Args:
    #         massflux (_type_): _description_
    #     """
    #     massflux_attrs = ["f_good_plume_bs", "f_particle_plume_alignment", "spacing"]
    #     massflux_keys = [
    #         "fitted_plumeline",
    #         "plumeline",
    #         "plumeslope",
    #         "dist_from_src",
    #         "direction_vector",
    #     ]
    #     massflux_lineinit_attrs = [
    #         "dist_from_src",
    #         "ds",
    #         "f_background_good",
    #     ]
    #     massflux_lineinit_keys = [
    #         "pre_origin",
    #         "coeffs",
    #         "nos",
    #         "dir_vect",
    #         "pre_coords_deg",
    #         "pre_coords_xy",
    #         "origin",
    #         "pre_co",
    #         "pre_co_int",
    #         "co",
    #         "line_dist",
    #         "coords_deg",
    #         "coords_xy",
    #     ]
    #     background_is_good_keys = [
    #         "gaussfit_x",
    #         "gaussfit_co",
    #         "gaussfit_params",
    #         "back_removed_co",
    #         "final_co",
    #         "final_line_dist",
    #         "final_coords_deg",
    #         "final_coords_xy",
    #     ]
    #     const_plume_ht_attrs = [
    #         "f_veldiff_constht",
    #         "emission_inj",
    #         "emission_inj_m500",
    #         "emission_inj_p500",
    #     ]
    #     const_plume_ht_keys = [
    #         "const_plume_ht",
    #         "vel_mag_inj",
    #         "emission_line_inj",
    #         "vel_mag_inj_m500",
    #         "emission_line_inj_m500",
    #         "vel_mag_inj_p500",
    #         "emission_line_inj_p500",
    #     ]
    #     varying_plume_ht_keys = [
    #         "f_lineparticle_plume_alignment",
    #         "varying_plume_ht",
    #         "varying_plume_ht_m500",
    #         "varying_plume_ht_p500",
    #     ]
    #     varying_emis_attrs = [
    #         "f_veldiff_varinght",
    #         "emission_lag",
    #         "emission_lag_m500",
    #         "emission_lag_p500",
    #     ]

    #     varying_emis_keys = [
    #         "vel_mag_lag",
    #         "emission_line_lag",
    #         "vel_mag_lag_m500",
    #         "emission_line_lag_m500",
    #         "vel_mag_lag_p500",
    #         "emission_line_lag_p500",
    #     ]
    #     fl = h5py.File(self.filename, "a")
    #     fire_grp = self.get_group(fl, self.grpname)
    #     mass_grp = self.get_group(fire_grp, self.massflux)

    #     # General massflux keys
    #     for _ky in massflux_attrs:
    #         mass_grp.attrs[_ky] = massflux_data.__getattribute__(_ky)

    #     for _ky in massflux_keys:
    #         mass_grp.create_dataset(_ky, data=massflux_data.__getattribute__(_ky))

    #     # Transaction lines
    #     for i in range(len(massflux_data.tlines)):
    #         linename = "line_" + str(i).zfill(2)
    #         line_grp = self.get_group(mass_grp, linename)
    #         _ln = massflux_data.tlines[i]
    #         for _ky in massflux_lineinit_attrs:
    #             line_grp.attrs[_ky] = _ln.__getattribute__(_ky)

    #         for _ky in massflux_lineinit_keys:
    #             line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))

    #         # line background is good
    #         if _ln.f_background_good:
    #             for _ky in background_is_good_keys:
    #                 line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))

    #             # Good backgound subtraction so plume is good
    #             # compute constant plume height emission
    #             # computes the varying plume height and has data on it
    #             if massflux_data.f_good_plume_bs:
    #                 # constant plume height
    #                 for _ky in const_plume_ht_attrs:
    #                     line_grp.attrs[_ky] = _ln.__getattribute__(_ky)
    #                 for _ky in const_plume_ht_keys:
    #                     line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))

    #                 # Changing plume height
    #                 _ky = varying_plume_ht_keys[0]
    #                 line_grp.attrs[_ky] = _ln.__getattribute__(_ky)
    #                 for _ky in varying_plume_ht_keys[1:]:
    #                     line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))
    #                 # If the particles and plume aligns
    #                 # then compute velocity and emissions
    #                 if (
    #                     _ln.f_lineparticle_plume_alignment
    #                     & massflux_data.f_particle_plume_alignment
    #                 ):
    #                     for _ky in varying_emis_attrs:
    #                         line_grp.attrs[_ky] = _ln.__getattribute__(_ky)
    #                     for _ky in varying_emis_keys:
    #                         line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))
    #     fl.close()

    # def write_satellite(self, sat_data, _grp):
    #     # file name is the name of the file based on month
    #     # day is the day on which fire was detected
    #     # fire number of the day
    #     # data contains aeverything
    #     satellite_data_attrs = [
    #         "orbit",
    #         "orbit_filename",
    #         "id",
    #         "orbit_ref_time",
    #         "source",
    #         "f_source_in_orbit",
    #         "source_pixel_id",
    #         "f_filter_gridsize",
    #         "f_not_all_nans",
    #         "f_qavalue_filter",
    #         "f_good_satellite_data",
    #     ]
    #     satellite_data_keys = [
    #         "lat",
    #         "lon",
    #         "qa_value",
    #         "co_column",
    #         "co_column_corr",
    #         "lat_corners",
    #         "lon_corners",
    #         "lat_nodes",
    #         "lon_nodes",
    #         "aerosol_thick",
    #         "avg_kernel",
    #         "co_qa_mask",
    #         "deltatime",
    #     ]
    #     # satellite data
    #     st_grp = _grp.create_group(self.satellitegrpname)
    #     for _ky in satellite_data_attrs:
    #         st_grp.attrs[_ky] = sat_data.__getattribute__(_ky)

    #     for _ky in satellite_data_keys:
    #         st_grp.create_dataset(_ky, data=sat_data.__getattribute__(_ky))

    # def write_plumedata(self, plume_data, _grp):
    #     # file name is the name of the file based on month
    #     # day is the day on which fire was detected
    #     plume_detection_attrs = ["f_plumedetect"]
    #     plume_detection_keys = ["plumemask", "labeled_img", "segmented_img"]
    #     # Debug true
    #     # ['elevation_map', 'marker_img', 'label_id']
    #     # Plume detection
    #     plume_grp = _grp.create_group(self.plumedetectgrpname)
    #     for _ky in plume_detection_attrs:
    #         plume_grp.attrs[_ky] = plume_data.__getattribute__(_ky)
    #     for _ky in plume_detection_keys:
    #         plume_grp.create_dataset(_ky, data=plume_data.__getattribute__(_ky))

    # def write_industry(self, sat_data, plume_data):
    #     # open file
    #     fl = h5py.File(self.filename, "a")

    #     # if group exists then read the name of group
    #     _grp = self.get_group(fl, self.grpname)
    #     # satellite data
    #     self.write_satellite(sat_data, _grp)
    #     if plume_data.f_plumedetect:
    #         self.write_plumedata(plume_data, _grp)
    #     # close file
    #     fl.close()

    # def _massfluxgeneral(self, massflux_data, mass_grp):
    #     massflux_attrs = ["f_good_plume_bs",
    #                       "f_particle_plume_alignment",
    #                       "spacing"]
    #     massflux_keys = ["fitted_plumeline",
    #                      "plumeline",
    #                      "plumeslope",
    #                      "dist_from_src",
    #                      "direction_vector",
    #                      ]
    #     # General massflux keys
    #     for _ky in massflux_attrs:
    #         mass_grp.attrs[_ky] = massflux_data.__getattribute__(_ky)

    #     for _ky in massflux_keys:
    #         mass_grp.create_dataset(_ky, data=massflux_data.__getattribute__(_ky))

    # def append_massflux1(self, massflux_data):
    #     """append_massflux _summary_

    #     Args:
    #         massflux (_type_): _description_
    #     """
    #     massflux_lineinit_attrs = [
    #         "dist_from_src",
    #         "ds",
    #         "f_background_good",
    #     ]
    #     massflux_lineinit_keys = [
    #         "pre_origin",
    #         "coeffs",
    #         "nos",
    #         "dir_vect",
    #         "pre_coords_deg",
    #         "pre_coords_xy",
    #         "origin",
    #         "pre_co",
    #         "pre_co_int",
    #         "co",
    #         "line_dist",
    #         "coords_deg",
    #         "coords_xy",
    #     ]
    #     background_is_good_keys = [
    #         "gaussfit_x",
    #         "gaussfit_co",
    #         "gaussfit_params",
    #         "back_removed_co",
    #         "final_co",
    #         "final_line_dist",
    #         "final_coords_deg",
    #         "final_coords_xy",
    #     ]
    #     const_plume_ht_attrs = [
    #         "f_veldiff_constht",
    #         "emission_inj",
    #         "emission_inj_m500",
    #         "emission_inj_p500",
    #     ]
    #     const_plume_ht_keys = [
    #         "const_plume_ht",
    #         "vel_mag_inj",
    #         "emission_line_inj",
    #         "vel_mag_inj_m500",
    #         "emission_line_inj_m500",
    #         "vel_mag_inj_p500",
    #         "emission_line_inj_p500",
    #     ]
    #     varying_plume_ht_keys = [
    #         "f_lineparticle_plume_alignment",
    #         "varying_plume_ht",
    #         "varying_plume_ht_m500",
    #         "varying_plume_ht_p500",
    #     ]
    #     varying_emis_attrs = [
    #         "f_veldiff_varinght",
    #         "emission_lag",
    #         "emission_lag_m500",
    #         "emission_lag_p500",
    #     ]

    #     varying_emis_keys = [
    #         "vel_mag_lag",
    #         "emission_line_lag",
    #         "vel_mag_lag_m500",
    #         "emission_line_lag_m500",
    #         "vel_mag_lag_p500",
    #         "emission_line_lag_p500",
    #     ]
    #     fl = h5py.File(self.filename, "a")
    #     fire_grp = self.get_group(fl, self.grpname)
    #     mass_grp = self.get_group(fire_grp, self.massflux)

    #     # General massflux keys
    #     self._massfluxgeneral(self, massflux_data, mass_grp)
    #     # for _ky in massflux_attrs:
    #     #     mass_grp.attrs[_ky] = massflux_data.__getattribute__(_ky)
    #     # for _ky in massflux_keys:
    #     #     mass_grp.create_dataset(_ky, data=massflux_data.__getattribute__(_ky))

    #     # Transaction lines
    #     for i in range(len(massflux_data.tlines)):
    #         linename = "line_" + str(i).zfill(2)
    #         line_grp = self.get_group(mass_grp, linename)
    #         _ln = massflux_data.tlines[i]
    #         for _ky in massflux_lineinit_attrs:
    #             line_grp.attrs[_ky] = _ln.__getattribute__(_ky)

    #         for _ky in massflux_lineinit_keys:
    #             line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))

    #         # line background is good
    #         if _ln.f_background_good:
    #             for _ky in background_is_good_keys:
    #                 line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))

    #             # Good backgound subtraction so plume is good
    #             # compute constant plume height emission
    #             # computes the varying plume height and has data on it
    #             if massflux_data.f_good_plume_bs:
    #                 # constant plume height
    #                 for _ky in const_plume_ht_attrs:
    #                     line_grp.attrs[_ky] = _ln.__getattribute__(_ky)
    #                 for _ky in const_plume_ht_keys:
    #                     line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))

    #                 # Changing plume height
    #                 _ky = varying_plume_ht_keys[0]
    #                 line_grp.attrs[_ky] = _ln.__getattribute__(_ky)
    #                 for _ky in varying_plume_ht_keys[1:]:
    #                     line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))
    #                 # If the particles and plume aligns
    #                 # then compute velocity and emissions
    #                 if (
    #                     _ln.f_lineparticle_plume_alignment
    #                     & massflux_data.f_particle_plume_alignment
    #                 ):
    #                     for _ky in varying_emis_attrs:
    #                         line_grp.attrs[_ky] = _ln.__getattribute__(_ky)
    #                     for _ky in varying_emis_keys:
    #                         line_grp.create_dataset(_ky, data=_ln.__getattribute__(_ky))
    #     fl.close()


