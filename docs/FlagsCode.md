- VIIRS Data for a day
- d_flag: If file exists or not
    - c_flag: If fires clusters are present or not
        - Orbits for the fire clusters
        - o_flag: If orbits for given fire clusters is present or not
            - Extract Satellite data
            - loc_flag (source_in_orbit): If the source is present in the orbit and is within 1deg
                - inorb_flag (source_pixel_loc): If the pixel exact location can be obatined
                    - filter_gridsize: If the location of source pixel is withing certain grid size
                        - Extract Data
                        - not_all_nans: True if all values in extracted granule are not nans
                            - flag_filterdata: filter based on qa values and nans in the data
                                - good_satellite_data: True: if the data is good [Final flag for data preparation]

                                    - Plume detection
                                    - plumedetectflag: if the plume is present or not
                                        - filteredplumeflag: If no other sources exist around
                                        - Injection height
                                            - inj_ht exists : Than lagrangian simulation
                                            



                
