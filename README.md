How to use this code
=====================

This code requires the dynacomp dataset to be used.


Data required:

* For sensor space analysis, only raw MEG data is required 
* For source space analysis, it is required:
    * preprocessed MEG data (SSS + downsampling at 400Hz, '_trans_sss_nofilt_ds5_raw.fif' files)
    * inverse operators for source reconstruction ('trans_sss_nofilt_ds5_meg-oct-6-inv.fif' files)
    * ICA files ('-nofilt-all-ica.fif' files)


Information about data location, groups, subjects and conditions must be defined in the scripts:
    * meg_info.py, for source space analysis
    * meg_info_sensor_space.py, for sensor space analysis


## How to run multifractal analysis on source space ##

* Define a list of parameters for source reconstruction in source_reconstruction_params.py
* Define a list of parameters for multifractal analysis in mf_config.py
* Run source reconstruction: source_reconstruction/source_rec_on_parcel.py
* Run multifractal analysis: mf_analysis_source_space/mfa_on_parcel_parallel.py


## How to run multifractal analysis on sensor space ##

* Define a list of parameters for multifractal analysis in mf_config_sensor_space.py
* Run multifractal analysis: mf_analysis_sensor_space/mf_analysis_on_sensors.py


## How to access results ##

* Outputs for source space (source reconstruction and MF analysis) are stored in data_out/source_rec_on_parcel

* Outputs for sensor space are stored in data_out/mf_sensors


The following scripts contain functions to easily load the desired data:

* source_mf_results.py, to access MF analysis results on source space
* ~~sensor_mf_results.py, to access MF analysis results on source space ()~~ (to be implemented)


## Scripts to analyze/visualize results ##

* analyze_mf_source/vis_log_cumulants.py:  hypotesis testing for H (c1) and M (-c2) and plot on cortical surface 



