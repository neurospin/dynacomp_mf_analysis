"""
This script performs multifractal analysis on CamCAM data and generates
output files.
"""

import os, sys
from os.path import dirname, abspath
# add project dir to path
project_dir = dirname(dirname(abspath(__file__))) 
sys.path.insert(0, project_dir)

import numpy as np
import matplotlib.pyplot as plt
from   joblib import Parallel, delayed
import os.path as op
import mne
from scipy.io import loadmat

import h5py

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs


SEED = 123
#===============================================================================
# Global parameters
#===============================================================================
import meg_info_sensor_space
info =  meg_info_sensor_space.get_info()


def get_my_selection(file_name):
    # Read events from a matlab file
    trl = loadmat(file_name).get('trl')

    # Adjust in function of the decimation
    trl[:,:2]-=1
    trl[:,:3]= np.round(trl) #np.round(trl/ds_factor)
    # return indexes
    start= trl[0,0] # +first_sample? not here !
    stop = trl[0,1] # +first_sample? not here !
    return start,stop



def get_raw(group, subject, condition, bis = True):
    subject_path = info['paths_to_subjects'][group][subject]
    raw_filename = os.path.join( subject_path,     
                                 '%s_%s_raw_trans_sss.fif'%(group,condition))


    if not bis:
        events_file = os.path.join(info['paths_to_events'][group][subject],
                                   condition+'.mat')
    else:
        events_file = os.path.join(info['paths_to_events'][group][subject],
                           condition + '_bis'+'.mat')

    start,stop = get_my_selection(events_file)

    raw = mne.io.read_raw_fif(raw_filename)


    tmin = start/raw.info['sfreq']
    tmax = stop/raw.info['sfreq']


    raw = raw.crop(tmin = tmin, tmax = tmax)

    return raw



def get_ica(raw, method = 'fastica',
            n_components = 0.99,
            bad_seconds = None,
            decim = 10,
            n_max_ecg = 3,
            n_max_eog = 2,
            max_iter  = 250,
            reject = dict(mag=5e-12, grad=4000e-13),
            random_state = SEED,
            plot = False):
    """
    Fit ICA to raw.

    Args:
        raw
        n_components: see mne.preprocessing.ICA
        fmin        : cutoff frequency of the high-pass filter before ICA
        bad_seconds : the first 'bad_seconds' seconds of data is annotated as 'BAD'
                      and not used for ICA. If None, no annotation is done
        decim       : see mne.preprocessing.ICA.fit()
        n_max_ecg   : maximum number of ECG components to remove
        n_max_eog   : maximum number of EOG components to remove
        max_iter    : maximum number of iterations during ICA fit.
    """

    # For the sake of example we annotate first 10 seconds of the recording as
    # 'BAD'. This part of data is excluded from the ICA decomposition by default.
    # To turn this behavior off, pass ``reject_by_annotation=False`` to
    # :meth:`mne.preprocessing.ICA.fit`.
    if bad_seconds is not None:
        raw.annotations = mne.Annotations([0], [bad_seconds], 'BAD')



    ica = ICA(n_components=n_components, method=method, random_state=random_state,
              verbose='warning', max_iter=max_iter)


    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                               stim=False, exclude='bads')

    #--------------------------------------------------------------------------
    # Fit ICA
    #--------------------------------------------------------------------------
    ica.fit(raw, picks=picks_meg, decim=decim, reject=reject)

    #--------------------------------------------------------------------------
    # Advanced artifact detection
    #--------------------------------------------------------------------------

    # EOG
    # eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
    # eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
    eog_inds, scores = ica.find_bads_eog(raw)
    #
    if plot:
        eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
        eog_average = create_eog_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                        picks=picks_meg).average()
        ica.plot_scores(scores, exclude=eog_inds, show=False)
        ica.plot_sources(eog_average, exclude=eog_inds, show=False)
        ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
        if len(eog_inds) > 0:
            ica.plot_properties(eog_epochs, picks=eog_inds[0], psd_args={'fmax': 35.},
                                image_args={'sigma': 1.}, show=False)



    # ECG
    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks_meg)
    ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')  # find via correlation #, threshold = 0.125

    if plot:
        #
        ecg_average = create_ecg_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                        picks=picks_meg).average()
        ica.plot_scores(scores, exclude=ecg_inds, show=False)
        ica.plot_sources(ecg_average, exclude=ecg_inds, show=False)
        ica.plot_overlay(ecg_average, exclude=ecg_inds, show=False)
        if len(ecg_inds) > 0:
            ica.plot_properties(ecg_epochs, picks=ecg_inds[0], psd_args={'fmax': 35.},
                                image_args={'sigma': 1.}, show=False)

    if plot:
        plt.show()

    # Exluce bad components
    ica.exclude.extend(eog_inds[:n_max_eog])
    ica.exclude.extend(ecg_inds[:n_max_ecg])

    # uncomment this for reading and writing
    # ica.save('my-ica.fif')
    # ica = read_ica('my-ica.fif')
    return ica


def preprocess_raw(raw, plot = True):

    mne.channels.fix_mag_coil_types(raw.info)

    # subsample at 1000 Hz, fmax = 330 Hz
    raw.load_data()
    raw.filter(l_freq = 0.08, h_freq = 330, n_jobs=1, fir_design='firwin')
    raw.resample(sfreq = 1000.0, npad="auto")

    # Apply ICA
    ica = get_ica(raw,  plot = plot)

    if plot:
        plt.show()

    raw_copy = raw.copy()
    ica.apply(raw_copy)
    return raw_copy, ica



if __name__ == '__main__':

    for group in info['groups']:
        for subject in info['subjects'][group]:
            for condition in ['pretest']:#info['sessions']:
                raw = get_raw(group, subject, condition)
                raw_, ica = preprocess_raw(raw)
                break
            break
        break