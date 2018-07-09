"""
This scripts reads the .h5 files in the folder /SSS containing the results of
multifractal analysis, puts all the results in a pandas DataFrame and save
a file containing everything


IMPORTANT:
    The following changes will soon be done:
        snr_list -> source_reconstruction_params (list)
        analysis_settings -> mf_analysis_params  (list)
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import matplotlib.pyplot as plt
import h5py
import numpy as np
import time
import pandas as pd


#===============================================================================
# Load info
#===============================================================================
import meg_info
info   = meg_info.get_info()

#===============================================================================
# GLOBAL PARAMETERS
#===============================================================================
# conditions
conditions = ['rest0', 'rest5', 'pretest', 'posttest']

# Output file version, '' or '_v2'
FILE_VERSION = '_v2'

# SNR list and folder name
snr_list = ['1', '3']
snr_1_dir_name = 'MF_parcel_python_400Hz_SNR_1'
snr_3_dir_name = 'MF_parcel_python_400Hz_SNR_3'

# groups and subjects
groups      = ['AV', 'V', 'AVr']
subjects    = {}
subjects['AV'] = ['nc_110174','da_110453','mb_110421','bl_110396','fp_110067','kr_080082', \
        'ks_110142','ld_110370','mp_110340','na_110353','pc_110210','pe_110338']
subjects['V'] = ['jh_100405','gc_100388','jm_100109','vr_100551','fb_110137','aa_100234', \
   'cl_100240','jh_110224','mn_080208','in_110286','tl_110313','cm_110222']
subjects['AVr'] = ['jm_100042','cd_100449','ap_110299','ma_130185','mj_130216','rg_110386', \
  'ga_130053','jd_110235','sa_130042','bd_120417','ak_130184','mr_080072']

# number of labels
n_labels = 138

# Settings for MF analysis
analysis_settings = ['DWT', 'LWT', 'P1', 'P2']

# output folder, where to save the .h5 containing the big arrays
save_dir = 'features'

# date, e.g. 20180516
timestr = time.strftime("%Y%m%d")

#===============================================================================
# MAIN SCRIPT
#===============================================================================
def main():
    # * filename = mf_filenames[snr][group][subject][condition]
    mf_filenames = get_mf_filenames()

    # Create list containing (group, subject) pairs.
    group_subject_list = []
    for group in groups:
        for subject in subjects[group]:
            group_subject_list.append( (group, subject) )

    #---------------------------------------------------------------------------
    # Build array containing log-cumulants
    #---------------------------------------------------------------------------
    n_snr        = len(snr_list)
    n_conditions = len(conditions)
    n_subjects   = len(group_subject_list) # over all groups
    n_analysis_settings = len(analysis_settings) # dwt, lwt, p = 1, p = 2
    n_labels     = 138
    n_cumulants  = 3
    max_j_in_cumulants = 14 # maximum scale over all computed comulants for all data


    #---------------------------------------------------------------------------
    # Initialize data dictionary (for DataFrame)
    #---------------------------------------------------------------------------
    fields = []
    # group
    fields += ['group']

    # subject
    fields += ['subject']

    # condition(session)
    fields += ['session']

    # label (on parcelation)
    fields += ['label']

    # mf_analysis_params
    fields += ['mf_param']

    # source_reconstruction_params
    fields += ['source_rec_param']

    # Log-cumulants
    fields += ['c'+str(c) for c in range(1,n_cumulants+1)]
    # Cumulants
    for c in range(1,n_cumulants+1):
        fields += ['C'+str(c)+'_'+str(i+1) for i in range(max_j_in_cumulants)]
    # nj
    fields += ['n'+str(i+1) for i in range(max_j_in_cumulants)]


    data_dict = {}
    for f in fields:
        data_dict[f] = []

    #---------------------------------------------------------------------------
    # Put data in data_dict
    #---------------------------------------------------------------------------
    print("\n-- reading files and organizing data frame...")
    for snr_index, snr in enumerate(snr_list):   # mapped as index
        for cond_index, condition in enumerate(conditions): # mapped as string
            for gs_index, group_subject in enumerate(group_subject_list): # mapped as strings
                group = group_subject[0]
                subject = group_subject[1]
                filename = mf_filenames[snr][group][subject][condition]
                with h5py.File(filename, 'r') as f:
                    for set_index, setting in enumerate(analysis_settings): # mapped as index

                        cp = f[setting+'/cp'][:] # array (n_labels, n_cumulants)
                        cj = f[setting+'/Cj'][:] # array  (n_labels, n_cumulants, number_of_scales_j)
                        nj = f[setting+'/nj'][:] # array (n_labels, number_of_scales_j)

                        for label in range(n_labels):
                            # Store group, subject and condition
                            data_dict['group'].append(group)
                            data_dict['subject'].append(subject)
                            data_dict['session'].append(condition)

                            # Store source_rec_param and mf_param (indexes)
                            data_dict['source_rec_param'].append(snr_index)
                            data_dict['mf_param'].append(set_index)

                            # Store label
                            data_dict['label'].append(label)

                            # Store log-cumulants
                            for c in range(1,n_cumulants+1):
                                field = 'c'+str(c)
                                data_dict[field].append(cp[label, c-1])

                            # Store cumulants
                            for c in range(1,n_cumulants+1):
                                for j in range(max_j_in_cumulants):
                                    field = 'C'+str(c)+'_'+str(j+1)
                                    if j < cj.shape[2]:
                                        data_dict[field].append(cj[label, c-1, j])
                                    else:
                                        data_dict[field].append(0.0)
                            # Store nj
                            for j in range(max_j_in_cumulants):
                                field = 'n'+str(j+1)
                                if j < nj.shape[1]:
                                    data_dict[field].append(nj[label, j])
                                else:
                                    data_dict[field].append(0)

    print("...done! \n")

    #---------------------------------------------------------------------------
    # Save file
    #---------------------------------------------------------------------------
    if not os.path.exists(info['dataset_dir_mf_output']):
        os.mkdir(info['dataset_dir_mf_output'])

    out_filename = os.path.join(info['dataset_dir_mf_output'], 'mf_results_' + timestr + '.gzip')
    df = pd.DataFrame(data = data_dict)
    df.to_csv(out_filename, encoding='utf-8', compression = 'gzip')

    print("\n Saved file: ", out_filename)

#
#     #---------------------------------------------------------------------------
#     # Testing...
#     #---------------------------------------------------------------------------
#     reproduce_fig_4(cumulants, log_cumulants, number_coeffs_j, C_idx = 0)
#     reproduce_fig_4(cumulants, log_cumulants, number_coeffs_j, C_idx = 1)

#===============================================================================
# I/O PARAMETERS
#===============================================================================
def get_mf_filenames():
    """
    Returns a dictionary mf_filename such that:
        mf_filenames[snr][group][subject][condition]
    contains the name of a MF output file
    """

    # data folder
    base_dir = '/volatile/omar/Documents/old_projects/omar_darwiche_domingues'
    data_dir = {}
    for group in groups:
        data_dir[group] = os.path.join(base_dir, 'data', 'MEG', 'dataset1', 'SSS', group)

    # output folders containing MF results / example: output_dir['AV']['nc_110174']
    output_dir = {}
    for snr in snr_list:
        output_dir[snr] = {}
        for group in groups:
            output_dir[snr][group] = {}
            for subject in subjects[group]:
                assert snr == '1' or snr == '3', "No files for SNR = " + snr

                if snr == '1':
                    output_dir[snr][group][subject] = \
                        os.path.join(data_dir[group], subject, snr_1_dir_name)
                elif snr == '3':
                    output_dir[snr][group][subject] = \
                        os.path.join(data_dir[group], subject, snr_3_dir_name)


    # files containing the results of MF analysis
    output_files = {}
    for snr in snr_list:
        output_files[snr] = {}
        for group in groups:
            output_files[snr][group] = {}
            for subject in subjects[group]:
                output_files[snr][group][subject] = {}
                for condition in conditions:
                    output_files[snr][group][subject][condition] = \
                        os.path.join(output_dir[snr][group][subject], condition + FILE_VERSION +'.h5')

    return output_files



if __name__ == '__main__':
    main()
