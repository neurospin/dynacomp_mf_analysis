"""
This script reads preprocessed MEG data in the sensor space (+ MRI data)
and performs source reconstruction.

The parameters are read from:
    meg_info.py
    source_reconstruction_params.py
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import numpy as np
import mne
import mne.io
from mne.io import Raw
from mne.minimum_norm import apply_inverse_epochs,read_inverse_operator
from mne.minimum_norm import apply_inverse_raw
from  scipy.io import loadmat, savemat
import time
import json
import h5py
from multiprocessing import Process

#------------------------------------------------------------------------------
# Maximum number of processes to run in parallel
#------------------------------------------------------------------------------
N_PROCESS = 1

#------------------------------------------------------------------------------
# Info for output files
#------------------------------------------------------------------------------
# date, e.g. 20180516
timestr = time.strftime("%Y%m%d")
raw_folder_name = 'raw_on_parc_400Hz'

#------------------------------------------------------------------------------
# Load info and params
#------------------------------------------------------------------------------
import source_reconstruction_params
import meg_info
info   = meg_info.get_info()
params = source_reconstruction_params.get_params()


#-------------------------------------------------------------------------------
# Select subjects
#-------------------------------------------------------------------------------
groups   = ['AV', 'V', 'AVr']
subjects = {}
subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']

#-------------------------------------------------------------------------------
# Other params and functions
#-------------------------------------------------------------------------------
ds_factor = 5
reject=dict(grad=4000e-13,mag=4e-12)
parcellation= 'aparc.split-small'

def get_my_selection(file_name):
    # Read events from a matlab file
    trl = loadmat(file_name).get('trl')

    # Adjust in function of the decimation
    trl[:,:2]-=1
    trl[:,:3]= np.round(trl/ds_factor)
    # return indexes
    start= trl[0,0] # +first_sample? not here !
    stop = trl[0,1] # +first_sample? not here !
    return start,stop

#-------------------------------------------------------------------------------
# Save params in subject folder
#-------------------------------------------------------------------------------
def save_params_file():
    for group in groups:
        for subject in subjects[group]:
            write_dir       = info['paths_to_subjects_output'][group][subject]
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)

            params_filename = os.path.join(write_dir,
                                           'params_'+timestr+'.json')
            with open(params_filename, 'w') as outfile:
                json.dump(params, outfile, indent=4, sort_keys=True)

save_params_file()

#-------------------------------------------------------------------------------
#  Function to run one case
#-------------------------------------------------------------------------------
def run_source_rec(param_idx, param, group, subject):
    lamb = param['lambda']
    method = param['method']
    print(" ")
    print("-------------------------------------------------------------------")
    print('** Running: snr = %f, method =  %s, group = %s, subject = %s' \
            % (param['snr'], method, group, subject))
    print("-------------------------------------------------------------------")

    write_dir = os.path.join(info['paths_to_subjects_output'][group][subject],
                                  raw_folder_name + '_'+ timestr + \
                                  '_rec_param_' + str(param_idx) )
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    surf_sub = subject[0:2] + subject[3:9] +'_landmark'
    labels= list(mne.read_labels_from_annot(subject=surf_sub,
                                            hemi='both',
                                            parc=parcellation,
                                            subjects_dir = os.path.join(info['subjects_dir'],
                                                          'subjects')
                                            ))

    Nbvert=np.array([len(labels[ilab].vertices) for ilab in np.arange(len(labels))]).reshape(-1,1)
    Nlabel=[labels[ilab].name for ilab in np.arange(len(labels))]

    for cond in info['sessions']:
        if 'rest' in cond:
            icagroup='rest'
        elif np.array(['test' in cond,'learn' in cond]).any():
            icagroup='task'

        fname_inv = os.path.join(info['paths_to_subjects'][group][subject],
                                'trans_sss_nofilt_ds5_meg-oct-6-inv.fif')
        fname_raw =os.path.join(info['paths_to_subjects'][group][subject],
                                 group + '_%s_trans_sss_nofilt_ds5_raw.fif' % cond)
        ica_f_name = os.path.join(info['paths_to_subjects'][group][subject],
                                group +'_'+icagroup+'-nofilt-all-ica.fif')
        inverse_operator = read_inverse_operator(fname_inv)
        src = inverse_operator['src']
        raw = Raw(fname_raw, preload=True).copy()
        ica=mne.preprocessing.read_ica(ica_f_name).copy()
        raw = ica.apply(raw)
        sfreq = raw.info['sfreq']
        if 'test' in fname_raw:
            part= ['','_bis']
        else:
            part= ['']
        for p in part:
            # load selection indexes
            start,stop= get_my_selection(os.path.join(info['paths_to_events'][group][subject],
                         cond + p +'.mat'))


            # split into epochs to reduce memory peak
            duration = 10.0
            events = mne.make_fixed_length_events(raw, duration=duration)
            tmax = duration - 1. / raw.info['sfreq']  # remove 1 point at the end that would be duplicated
            epochs = mne.Epochs(raw, events, tmin=0, tmax=tmax, baseline = None)

            # compute the source time courses and average by label
            mstcs_list = []
            for i_epoch, _ in enumerate(epochs):
                stcs = apply_inverse_epochs(epochs[i_epoch],
                                            inverse_operator,
                                            lamb,
                                            method,
                                            pick_ori='normal')
                mstcs=np.array(mne.extract_label_time_course(stcs, labels, src, mode='mean_flip'))
                mstcs = np.squeeze(mstcs, axis = 0)
                mstcs_list.append(mstcs)

            mstcs_on_label_Radial = np.concatenate(mstcs_list, axis=1)
            time = (np.arange(mstcs_on_label_Radial.shape[1]) / float(raw.info['sfreq'])).reshape(-1, 1)

            mstcs_on_label_Radial = mstcs_on_label_Radial[:, start:stop]
            time                  = time[start:stop]
            del stcs, mstcs_list

            # save file for current condition
            out_filename = os.path.join(write_dir, cond + p + '.mat')
            data = {'Radial':mstcs_on_label_Radial,'time':time,'Nbvert':Nbvert,'Nlabel':Nlabel}
            savemat(out_filename,data)


            print("-----------------------------------------------------------------------------------")
            print("*** saved file ", out_filename)
            print("-----------------------------------------------------------------------------------")

            # out_filename = os.path.join(write_dir, cond + p + '.h5')
            #
            # with h5py.File(out_filename, "w") as f:
            #     f.create_dataset('Radial', data = mstcs_on_label_Radial )
            #     f.create_dataset('time', data = time )
            #     f.create_dataset('Nbvert', data = Nbvert )
            #     f.create_dataset('Nlabel', data = Nlabel )
            #
            #
            # print("*** saved file: ", out_filename)



#-------------------------------------------------------------------------------
#  Run!
#-------------------------------------------------------------------------------
process_list = []
for param_idx, param in enumerate(params):
    for group in groups:
        for subject in subjects[group]:
            # run_source_rec(param_idx,param, group, subject)
            new_process = Process(
                         target = run_source_rec,
                         args = (param_idx,param, group, subject)
                         )
            new_process.start()
            process_list.append(new_process)
            if len(process_list) == N_PROCESS:
                for process in process_list:
                    process.join()
                process_list = []
