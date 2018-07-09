"""
This scripts reads the .mat files in the folder /SSS and performs multifractal
analysis on the data.

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
import mfanalysis as mf

#===============================================================================
# GLOBAL PARAMETERS
#===============================================================================
# folder containing /data   (attention here if this file is put in another folder)
project_dir = dirname(dirname(abspath(__file__))) 

# Raw data folder name
raw_dir_name = 'Raw_on_Parc_mydata_400Hz_20180412'

# Output folder name
out_dir_name = 'MF_parcel_python_400Hz_SNR_1'

# Number of processes to run in parallel
N_PROCESS = 2


#===============================================================================
# MAIN SCRIPT
#===============================================================================

def main():
    #---------------------------------------------------------------------------
    # Parameters
    #---------------------------------------------------------------------------

    # groups and subjects
    groups      = ['AV','AVr', 'V']
    subjects    = {}
    subjects['AV'] = ['nc_110174','da_110453','mb_110421','bl_110396','fp_110067','kr_080082', \
            'ks_110142','ld_110370','mp_110340','na_110353','pc_110210','pe_110338']

    subjects['V'] = ['jh_100405','gc_100388','jm_100109','vr_100551','fb_110137','aa_100234', \
       'cl_100240','jh_110224','mn_080208','in_110286','tl_110313','cm_110222']

    subjects['AVr'] = ['jm_100042','cd_100449','ap_110299','ma_130185','mj_130216','rg_110386', \
      'ga_130053','jd_110235','sa_130042','bd_120417','ak_130184','mr_080072']


    # data folder
    data_dir = {}
    for group in groups:
        data_dir[group] = os.path.join(project_dir, 'data', 'MEG', 'dataset1', 'SSS', group) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # raw data folder / example: raw_data_dir['AV']['nc_110174']
    raw_data_dir = {}
    for group in groups:
        raw_data_dir[group] = {}
        for subject in subjects[group]:
            raw_data_dir[group][subject] = \
                os.path.join(data_dir[group], subject, raw_dir_name)



    # create output folders / example: output_dir['AV']['nc_110174']
    output_dir = {}
    for group in groups:
        output_dir[group] = {}
        for subject in subjects[group]:
            output_dir[group][subject] = \
                os.path.join(data_dir[group], subject, out_dir_name)

            if not os.path.exists(output_dir[group][subject]):
                os.makedirs(output_dir[group][subject])


    # conditions <-> files
    runs = {}
    runs['rest0']    = ['rest0.mat']
    runs['rest5']    = ['rest5.mat']
    runs['pretest']  = ['pretest.mat', 'pretest_bis.mat']
    runs['posttest'] = ['posttest.mat', 'posttest_bis.mat']

    # number of labels
    n_labels = 138

    #---------------------------------------------------------------------------
    # MF params
    #---------------------------------------------------------------------------

    # 'global' args
    mf_args = {}
    mf_args['wt_name'] = 'db3'
    mf_args['j1'] = 8
    mf_args['j2'] = 12
    mf_args['q'] = np.arange(-8, 9)
    mf_args['n_cumul'] = 3
    mf_args['gamint'] = 1.0
    mf_args['wtype'] = 0
    mf_args['verbose'] = 1

    #---------------------------------------------------------------------------
    # Start processes
    #---------------------------------------------------------------------------
    process_list  = []
    for group in groups:
        for subject in subjects[group]:
            for condition in runs:
                new_process = Process(
                              target = run_mf_analysis,
                              args = (group, subject, condition,
                                      mf_args,
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
                    mf_args,
                    runs,
                    raw_data_dir,
                    output_dir,
                    n_labels):

    #---------------------------------------------------------------------------
    # Create MF objects
    #---------------------------------------------------------------------------

    # Wavelet coefficients
    mfa_dwt = mf.MFA(**mf_args)
    mfa_dwt.p = None
    mfa_dwt.formalism = 'wcmf'

    # Wavelet leaders
    mfa_lwt = mf.MFA(**mf_args)
    mfa_lwt.p = np.inf

    # p-leaders, p = 1
    mfa_p1 = mf.MFA(**mf_args)
    mfa_p1.p = 1.0

    # p-leaders, p = 2
    mfa_p2 = mf.MFA(**mf_args)
    mfa_p2.p = 2.0

    #---------------------------------------------------------------------------
    # Run
    #---------------------------------------------------------------------------
    print("*** Running: ", group, subject, condition)

    # cumulants and log-cumulants
    cumulants_dwt = []
    cumulants_lwt = []
    cumulants_p1  = []
    cumulants_p2  = []

    cp_dwt = np.zeros((n_labels, mf_args['n_cumul']))
    cp_lwt = np.zeros((n_labels, mf_args['n_cumul']))
    cp_p1  = np.zeros((n_labels, mf_args['n_cumul']))
    cp_p2  = np.zeros((n_labels, mf_args['n_cumul']))

    # iterate through files of a given condition
    for file_idx, filename in enumerate(runs[condition]):

        print("-- analyzing file: ", filename)
        # load data
        filename_full = os.path.join(raw_data_dir[group][subject], filename)
        contents = loadmat(filename_full)
        radial   = contents['Radial']
        nrows, ncols = radial.shape

        assert nrows == n_labels # verification

        # iterate through labels
        for row in range(n_labels):
            data = radial[row,:]
            mfa_dwt.analyze(data)
            mfa_lwt.analyze(data)
            mfa_p1.analyze(data)
            mfa_p2.analyze(data)

            if file_idx == 0:
                # remove mrq from cumulants to save memory (very important!)
                mfa_dwt.cumulants.mrq = None
                mfa_lwt.cumulants.mrq = None
                mfa_p1.cumulants.mrq = None
                mfa_p2.cumulants.mrq = None
                # store cumulants
                cumulants_dwt.append(mfa_dwt.cumulants)
                cumulants_lwt.append(mfa_lwt.cumulants)
                cumulants_p1.append(mfa_p1.cumulants)
                cumulants_p2.append(mfa_p2.cumulants)
            else:
                cumulants_dwt[row].sum(mfa_dwt.cumulants)
                cumulants_lwt[row].sum(mfa_lwt.cumulants)
                cumulants_p1[row].sum(mfa_p1.cumulants)
                cumulants_p2[row].sum(mfa_p2.cumulants)

            if file_idx == len(runs[condition]) - 1:
                # update cp
                cp_dwt[row,:] = cumulants_dwt[row].log_cumulants
                cp_lwt[row,:] = cumulants_lwt[row].log_cumulants
                cp_p1[row,:]  = cumulants_p1[row].log_cumulants
                cp_p2[row,:]  = cumulants_p2[row].log_cumulants

                # Update cumulants C_m(j)
                if row == 0:
                    # - initialize
                    cj_dwt = np.zeros( (n_labels,) + cumulants_dwt[row].values.shape )
                    cj_lwt = np.zeros( (n_labels,) + cumulants_lwt[row].values.shape )
                    cj_p1 = np.zeros( (n_labels,) +  cumulants_p1[row].values.shape )
                    cj_p2 = np.zeros( (n_labels,) +  cumulants_p2[row].values.shape )

                    nj_dwt = np.zeros( (n_labels, cumulants_dwt[row].values.shape[1]))
                    nj_lwt = np.zeros( (n_labels, cumulants_lwt[row].values.shape[1]))
                    nj_p1 = np.zeros( (n_labels, cumulants_p1[row].values.shape[1]) )
                    nj_p2 = np.zeros( (n_labels, cumulants_p2[row].values.shape[1]) )

                # - update
                cj_dwt[row, :, :] = cumulants_dwt[row].values
                cj_lwt[row, :, :] = cumulants_lwt[row].values
                cj_p1[row, :, :]  = cumulants_p1[row].values
                cj_p2[row, :, :]  = cumulants_p2[row].values

                nj_dwt[row, :]  = cumulants_dwt[row].get_nj()
                nj_lwt[row, :]  = cumulants_lwt[row].get_nj()
                nj_p1[row,  :]  = cumulants_p1[row].get_nj()
                nj_p2[row,  :]  = cumulants_p2[row].get_nj()


    # save file for current condition
    out_filename = os.path.join(output_dir[group][subject], condition + '_v2.h5')

    with h5py.File(out_filename, "w") as f:
        params = f.create_dataset('params', data = np.array([0]) )
        params.attrs['j1']     = mf_args['j1']
        params.attrs['j2']     = mf_args['j2']
        params.attrs['wtype']  = mf_args['wtype']
        params.attrs['n_cumul']  = mf_args['n_cumul']
        params.attrs['gamint']  = mf_args['gamint']


        dwt_group = f.create_group('DWT')
        dwt_group.attrs['params'] = np.string_(str(mf_args))
        dwt_nj_dataset  = dwt_group.create_dataset('nj', data = nj_dwt)
        dwt_cp_dataset  = dwt_group.create_dataset('cp', data = cp_dwt )
        dwt_cj_dataset  = dwt_group.create_dataset('Cj', data = cj_dwt )
        dwt_group       = None

        lwt_group = f.create_group('LWT')
        lwt_group.attrs['params'] = np.string_(str(mf_args))
        lwt_nj_dataset = lwt_group.create_dataset('nj', data = nj_lwt)
        lwt_cp_dataset = lwt_group.create_dataset('cp', data = cp_lwt )
        lwt_cj_dataset  = lwt_group.create_dataset('Cj', data = cj_lwt )
        lwt_group       = None

        p1_group = f.create_group('P1')
        p1_group.attrs['params'] = np.string_(str(mf_args))
        p1_nj_dataset = p1_group.create_dataset('nj', data = nj_p1)
        p1_cp_dataset = p1_group.create_dataset('cp', data = cp_p1 )
        p1_cj_dataset  = p1_group.create_dataset('Cj', data = cj_p1 )
        p1_group       = None

        p2_group = f.create_group('P2')
        p2_group.attrs['params'] = np.string_(str(mf_args))
        p2_nj_dataset = p2_group.create_dataset('nj', data = nj_p2)
        p2_cp_dataset = p2_group.create_dataset('cp', data = cp_p2 )
        p2_cj_dataset  = p2_group.create_dataset('Cj', data = cj_p2 )
        p2_group       = None

    print("-- saved file ", out_filename)



# if __name__ == '__main__':
#     main()
