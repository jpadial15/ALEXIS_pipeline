
**If you only want the event list, please register and download the event list from the data_products directory**

#

### 1. Creating the SQLite databases: 
   - Make or download the XRS availability database.
   - Remember to make the data_products directory and verify the permissions of all directories and data products. Permissions need to be changed for ruffus database. 
      ```
      python xray_ftp_query.py
      ```
  - Make HEK Known flares database.
      ```
      python create_flare_hek_query.py
      ```
  - Make or download the differential analysis aggregated results with SolarSoft and SWPC.
      ```
      python create_zero_cross_agg_flare_list.py
      ```
  - Make or download the AIA availability database.
      ```
      python create_aia_12s_availability.py
      ```
  - Make or download the HARP DRMS availability database.
      ```
      python create_drms_hmi_AR_db.py
      ```
  - Make or download the SXI availability database.
      ```
      python create_sxi_availability.py
      ```


