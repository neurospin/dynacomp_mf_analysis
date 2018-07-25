"""
This scripts reads the .mat files in the folder /SSS and performs multifractal
analysis on the data.


This file should be executed independently for each source reconstruction parameter,
by chosing the appropriate raw_dir_name

# NOTE:
    Running time: - about 3h30min for one group/one value of SNR
                  - Intel Xeon E5440 @ 2.83Ghz x 8
                  - N_PROCESS = 4
"""

import sys, os
from os.path import dirname, abspath
from scipy.io import loadmat
from multiprocessing import Process
import h5py
import numpy as np
import json
import mfanalysis as mf

#===============================================================================
# GLOBAL PARAMETERS
#===============================================================================
# folder containing /data_out   (attention here if this file is put in another folder)
project_dir = dirname(dirname(abspath(__file__))) 
sys.path.insert(0, project_dir)

# time string
time_str = '20180724'

# Raw data folder name
raw_dir_name = 'raw_on_parc_400Hz_no_ica_'+ time_str +'_rec_param_0'

# Output folder name (the index of the source reconstruction 
#                     parameters will be appended to out_dir_name)
out_dir_name = 'mf_parcel_400Hz_no_ica_'+ time_str + '_rec_param_0'

# Number of processes to run in parallel
N_PROCESS = 4

#------------------------------------------------------------------------------
# Load info and params
#------------------------------------------------------------------------------
import source_reconstruction_params
import meg_info
import mf_config
info              = meg_info.get_info()
mf_params         = mf_config.get_mf_params()

#===============================================================================
# MAIN SCRIPT
#===============================================================================

def main():
    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------

    # groups and subjects
    groups   = ['AV', 'V', 'AVr']
    # groups   = ['V', 'AVr']
    subjects = {}
    subjects['AV'] = info['subjects']['AV']
    subjects['V'] = info['subjects']['V']
    subjects['AVr'] = info['subjects']['AVr']
    conditions  = info['sessions']
    

    # folder containing raw data
    raw_data_dir = {}
    for gg in groups:
        raw_data_dir[gg] = {}
        for ss in subjects[gg]:
            raw_data_dir[gg][ss] = os.path.join(info['paths_to_subjects_output'][gg][ss], 
                                                raw_dir_name)


    # output folders / example: output_dir['AV']['nc_110174']
    output_dir   = {}
    for gg in groups:
        output_dir[gg] = {}
        for ss in subjects[gg]:
            output_dir[gg][ss] = os.path.join(info['paths_to_subjects_output'][gg][ss], 
                                              out_dir_name)


    # conditions <-> files, 
    #      example: runs['posttest'] = ['posttest.mat', 'posttest_bis.mat']
    #               runs['rest0']    = ['rest0.mat']
    runs = {}
    for cond in conditions:
        runs[cond] = [cond + '.mat']
        if 'test' in cond:
            runs[cond].append( cond + '_bis.mat'  )


    # number of labels
    n_labels = 138

    #---------------------------------------------------------------------------
    # Start processes
    #---------------------------------------------------------------------------
    process_list  = []
    for group in groups:
        for subject in subjects[group]:
            for condition in conditions:
                new_process = Process(
                              target = run_mf_analysis,
                              args = (group, subject, condition,
                                      mf_params,
                                      runs,
                                      raw_data_dir,
                                      output_dir,
                                      n_labels)
                              )
                new_process.start()
                process_list.append(new_process)
                if len(process_list) == N_PROCESS:
                    for process in process_list:
                        process.join()
                    process_list = []


def run_mf_analysis(group, subject, condition,
                    mf_params,
                    runs,
                    raw_data_dir,
                    output_dir,
                    n_labels):

    for mf_param_idx, mf_params_instance in enumerate(mf_params):
        #---------------------------------------------------------------------------
        # Create MF object
        #---------------------------------------------------------------------------
        mfa = mf.MFA(**mf_params_instance)
        mfa.verbose = 1

        #---------------------------------------------------------------------------
        # Run
        #---------------------------------------------------------------------------
        print("*** Running: ", group, subject, condition)

        # cumulants and log-cumulants
        cumulants = []
        cp = np.zeros((n_labels, mf_params_instance['n_cumul']))

        # iterate through files of a given condition
        for file_idx, filename in enumerate(runs[condition]):
            print("-- analyzing file: ", filename)
            # load data
            filename_full = os.path.join(raw_data_dir[group][subject],filename)
            contents = loadmat(filename_full)
            radial   = contents['Radial']
            nrows, ncols = radial.shape

            assert nrows == n_labels # verification

            # iterate through labels
            for row in range(n_labels):
                data = radial[row,:]
                mfa.analyze(data)

                if file_idx == 0:
                    # remove mrq from cumulants to save memory (very important!)
                    mfa.cumulants.mrq = None

                    # store cumulants
                    cumulants.append(mfa.cumulants)

                else:
                    cumulants[row].sum(mfa.cumulants)


                if file_idx == len(runs[condition]) - 1:
                    # update cp
                    cp[row,:] = cumulants[row].log_cumulants

                    # Update cumulants C_m(j)
                    if row == 0:
                        # - initialize
                        cj = np.zeros( (n_labels,) + cumulants[row].values.shape )
                        nj = np.zeros( (n_labels, cumulants[row].values.shape[1]))

                    # - update
                    cj[row, :, :] = cumulants[row].values
                    nj[row, :]  = cumulants[row].get_nj()

        # save file for current condition
        if not os.path.exists(output_dir[group][subject]):
            os.makedirs(output_dir[group][subject])

        out_filename = os.path.join(output_dir[group][subject], condition + '_mf_param_%d'%mf_param_idx +'.h5')

        with h5py.File(out_filename, "w") as f:
            params_string = np.string_(str(mf_params_instance))
            f.create_dataset('params', data = params_string )
            f.create_dataset('nj', data = nj )
            f.create_dataset('cp', data = cp )
            f.create_dataset('Cj', data = cj )      
        print("-- saved file ", out_filename)


        # save params file
        mf_params_filename = os.path.join(info['paths_to_subjects_output'][group][subject],
                                          'mf_params_'+time_str+'.json')

        with open(mf_params_filename, 'w') as outfile:
                json.dump(mf_params, outfile, indent=4, sort_keys=True)

if __name__ == '__main__':
    main()
