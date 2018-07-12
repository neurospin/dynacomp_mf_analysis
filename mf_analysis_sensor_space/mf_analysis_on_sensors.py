import os, sys
from os.path import dirname, abspath
# add project dir to path
project_dir = dirname(dirname(abspath(__file__))) 
sys.path.insert(0, project_dir)

import numpy as np 
import mfanalysis as mf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os.path as op
import mne
import os
import h5py

import preprocessing 

import meg_info_sensor_space as meginfo
import mf_config_sensor_space
info =  meginfo.get_info()
# MF parameters
mf_params = mf_config_sensor_space.get_mf_params()

"""
This file contains information about multifractal analysis parameters.
"""

import sys, os
from os.path import dirname, abspath
import numpy as np

#-------------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------------

# Channel type
MEG_TYPE = 'mag' # 'mag' or 'grad'

# Subjects and conditions
groups   = ['V', 'AVr'] #, 'V', 'AVr']
subjects = {}
# subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']
conditions  = info['sessions']


# conditions <-> files, 
#      example: runs['posttest'] = ['posttest.mat', 'posttest_bis.mat']
#               runs['rest0']    = ['rest0.mat']
runs = {}
for cond in conditions:
    runs[cond] = [cond + '.mat']
    if 'test' in cond:
        runs[cond].append( cond + '_bis.mat'  )


# Output folder
#   - usage: output_dirs[group][subject]
output_dirs = info['paths_to_subjects_output']


    
#-------------------------------------------------------------------------------
# MF analysis
#-------------------------------------------------------------------------------
def single_mf_analysis(args):
    """
    Apply MF analysis on (group, subject, condition, params) using the parameters in the
    mf_params dictionary
    """
    # try:
    if True:
        group, subject, condition, params_index_list, max_j = args

        for params_index in params_index_list: # iterate through mf parameters
            params = mf_params[params_index]
            mfa = mf.MFA(**params)
            mfa.verbose = 1

            ###############################################
            # preprocess data before the analysis
            data_pointers = [None for i in range(len(runs[condition]))]
            for run_idx, run in enumerate(runs[condition]):

                bis = False
                if 'bis' in run:
                    bis = True

                # Get raw data
                raw = preprocessing.get_raw(group, subject, condition, bis)


                # Preprocess raw
                raw, ica = preprocessing.preprocess_raw(raw)


                # Pick MEG magnetometers or gradiometers
                picks = mne.pick_types(raw.info, meg=MEG_TYPE, eeg=False, stim=False, eog=False,
                                       exclude='bads')

                picks_ch_names = [raw.ch_names[i] for i in picks]

                data = raw.get_data(picks)

                n_channels = len(picks)    

                data_pointers[run_idx] = (data, picks, picks_ch_names, n_channels)


            ###############################################

            cumulants = [] # list of cumulants objects, length = n_channels
            # output arrays
            all_log_cumulants = np.zeros((n_channels, mfa.n_cumul))
            all_cumulants     = np.zeros((n_channels, mfa.n_cumul, max_j))
            all_hmin          = np.zeros(n_channels)


            for run_idx, run in enumerate(runs[condition]): # e.g. ['posttest.mat', 'posttest_bis.mat']

                # get preprocessed data
                data, picks, picks_ch_names, n_channels = data_pointers[run_idx]
                print("!!!!!!!!!!!!! n_channels = ",n_channels)


                bis = False
                if 'bis' in run:
                    bis = True

                print("-------------------------------------------------")
                print("Performing mf analysis for (%s, %s, %s)"%(group, subject, condition))
                print("bis = ", bis)
                print("MF params: %d/%d"%(params_index, len(params_index_list)-1))
                print("-------------------------------------------------")

                for ii in range(n_channels):
                   # Run analysis
                    signal = data[ii, :]
                    mfa.analyze(signal)

                    if run_idx == 0:  # first run
                        # remove mrq from cumulants to save memory (very important!)
                        mfa.cumulants.mrq = None  
                        cumulants.append(mfa.cumulants)
                        all_hmin[ii] = mfa.hmin 
                    else:            # other runs
                        cumulants[ii].sum(mfa.cumulants)
                        all_hmin[ii] = min(mfa.hmin, all_hmin[ii])

                    if run_idx == len(runs[condition]) - 1: #last run
                        all_log_cumulants[ii, :] = cumulants[ii].log_cumulants

                        max_idx = min(max_j, cumulants[ii].values.shape[1])
                        all_cumulants[ii, :, :max_idx] = cumulants[ii].values[:, :max_idx]


            # store results
            subject_output_dir = os.path.join(output_dirs[group][subject])
            if not os.path.exists(subject_output_dir):
                os.makedirs(subject_output_dir)

            output_filename = os.path.join(subject_output_dir, condition + "_channel_%s_params_%d"%(MEG_TYPE, params_index) +'.h5')

            with h5py.File(output_filename, "w") as f:
                params_string = np.string_(str(params))
                f.create_dataset('params', data = params_string )
                f.create_dataset('log_cumulants', data = all_log_cumulants )
                f.create_dataset('cumulants', data = all_cumulants )
                f.create_dataset('hmin', data = all_hmin )
                f.create_dataset('channels_picks', data = picks)
                channels_name_list = [n.encode("ascii", "ignore") for n in picks_ch_names]
                f.create_dataset('picks_ch_names', (len(channels_name_list),1),'S10', channels_name_list)

            print("-------------------------------------------------")
            print("*** saved file ", output_filename)
            print("-------------------------------------------------")

    # except Exception as e:
    #     print('!!! Error in mf_analysis_on_sensors: '+ str(e))

    # return raw, output_filename


if __name__ == '__main__':
    
    # # debug ------
    # group = groups[0]
    # subject = subjects[group][0]
    # condition = 'pretest'
    # params_index_list = [0]
    # max_j = 15

    # args = (group, subject, condition, params_index_list, max_j)
    # single_mf_analysis(args)
    # # ------


    params_index_list = [0, 1, 2, 3]
    # Select params
    max_j = 15

    arg_instances = []
    for gg in groups:
        for ss in subjects[gg]:
            for cond in conditions:
                    arg_instances.append( (gg, ss, cond, params_index_list, max_j) )


    # remove already computed instances
    new_arg_instances = []
    for args in arg_instances:
        group, subject, condition, params_index_list, max_j = args
        subject_output_dir = os.path.join(output_dirs[group][subject])

        output_filename = os.path.join(subject_output_dir, condition + "_channel_%s_params_%d"%(MEG_TYPE, params_index_list[-1]) +'.h5')
        if op.isfile(output_filename):
            continue
        else:
            new_arg_instances.append(args)
    arg_instances = new_arg_instances


    Parallel(n_jobs=1, verbose=1, backend="threading")(map(delayed(single_mf_analysis), arg_instances))