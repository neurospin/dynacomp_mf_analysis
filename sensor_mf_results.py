
"""
Provides functions to load MF results on sensor space.
"""

import os
import numpy as np 
import h5py
import meg_info_sensor_space
info = meg_info_sensor_space.get_info()


def load_data(group, subject, condition, 
              mf_param_idx = 1,
              channel_type = 'mag'):
    """
    group = 'AV', 'V' or 'AVr'
    subject = one of the subjects in the given group
    condition = 'rest0', 'rest5', 'pretest', 'posttest'
    mf_param_idx = index of MF parameters
    channel_type = 'mag' or 'grad' 
    """
    data_dir = info['paths_to_subjects_output'][group][subject]


    filename = '%s_channel_%s_params_%d.h5'%(condition, channel_type, mf_param_idx)
    filename = os.path.join(data_dir, filename)

    with h5py.File(filename, 'r') as f:
        log_cumulants = f['log_cumulants'][:]
        cumulants = f['cumulants'][:]
        params = f['params'].value
        channels_picks = f['channels_picks'].value
        channels_names = [name.decode('ascii') for name in f['picks_ch_names'][:].squeeze()]

    ch_name2index  = dict( zip( channels_names, list(range(len(channels_picks))) ) )


    return log_cumulants, cumulants, params, channels_picks, channels_names, ch_name2index


def load_data_groups_subjects(condition,
                              groups = ['AV', 'V', 'AVr'],
                              subjects = info['subjects'],
                              mf_param_idx = 1,
                              max_j = 14,
                              channel_type = 'mag',
                              n_cumul = 3):
    """ 
    conition = 'rest0', 'rest5', 'pretest' or 'posttest'
    groups: list
    subjects: dictionary, e.g. subjets['AV'] = list of AV subjects
    """
    
    if channel_type == 'mag':
        n_channels = 102
    if channel_type == 'grad':
        n_channels = 204

    n_subjects = sum([ len(subjects[gg]) for gg in groups ])

    all_log_cumulants = np.zeros((n_subjects, n_channels, n_cumul))
    all_cumulants = np.zeros((n_subjects, n_channels, n_cumul, max_j))

    subjects_list = []
    idx = 0
    for gg in groups:
        for ss in subjects[gg]:
            subjects_list.append((gg, ss))

            log_cumulants, cumulants, params, channels_picks, channels_names, ch_name2index = \
                                                        load_data(gg, ss, condition, 
                                                                  mf_param_idx ,
                                                                  channel_type)

            all_log_cumulants[idx, :, :] = log_cumulants
            all_cumulants[idx, :, :, :max_j] = cumulants[:,:,:max_j]

            idx += 1


    return all_log_cumulants, all_cumulants, subjects_list, params, channels_picks, channels_names, ch_name2index

if __name__ == '__main__':
    group = info['groups'][0]
    subject = info['subjects'][group][0]
    condition = 'rest0'

    log_cumulants, cumulants, params, channels_picks, channels_names, ch_name2index = load_data(group, subject, condition)
        
    all_log_cumulants, all_cumulants, subjects_list, params, channels_picks, channels_names, ch_name2index =\
         load_data_groups_subjects('rest0')