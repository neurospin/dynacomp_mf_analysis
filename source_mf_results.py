"""
Provides a function to load MF results on source space.
"""

import os
import numpy as np 
import h5py
import meg_info
info = meg_info.get_info()


def load_data(group, subject, condition, 
              mf_param_idx = 1, 
              source_rec_param_idx = 0,
              time_str = '20180710'):
    """
    group = 'AV', 'V' or 'AVr'
    subject = one of the subjects in the given group
    condition = 'rest0', 'rest5', 'pretest', 'posttest'
    param_idx = index of MF parameters
    source_rec_param_idx = index of source reconstruction parameters
    """
    data_dir = info['paths_to_subjects_output'][group][subject]
    mf_dir_name = 'mf_parcel_400Hz_'+ time_str +'_rec_param_%d'%source_rec_param_idx

    filename = os.path.join(data_dir, 
                            mf_dir_name, 
                            condition + '_mf_param_%d'%mf_param_idx +'.h5')

    with h5py.File(filename, 'r') as f:
        log_cumulants = f['cp'][:]
        cumulants = f['Cj'][:]

    return log_cumulants, cumulants


def load_data_groups_subjects(condition,
                              groups = ['AV'],
                              subjects = info['subjects'],
                              mf_param_idx = 1, 
                              source_rec_param_idx = 0,
                              time_str = '20180710',
                              n_labels = 138,
                              n_cumul = 3,
                              max_j = 13):
    """ 
    conition = 'rest0', 'rest5', 'pretest' or 'posttest'
    groups: list
    subjects: dictionary, e.g. subjets['AV'] = list of AV subjects
    """
    
    n_subjects = sum([ len(subjects[gg]) for gg in groups ])

    all_log_cumulants = np.zeros((n_subjects, n_labels, n_cumul))
    all_cumulants = np.zeros((n_subjects, n_labels, n_cumul, max_j))

    subjects_list = []
    idx = 0
    for gg in groups:
        for ss in subjects[gg]:
            subjects_list.append((gg, ss))

            log_cumulants, cumulants = load_data(gg, ss, condition, 
                                                 mf_param_idx, 
                                                 source_rec_param_idx,
                                                 time_str)

            all_log_cumulants[idx, :, :] = log_cumulants
            all_cumulants[idx, :, :, :max_j] = cumulants[:,:,:max_j]

            idx += 1


    return all_log_cumulants, all_cumulants, subjects_list



if __name__ == '__main__':
    group = info['groups'][0]
    subject = info['subjects'][group][0]
    condition = 'rest0'

    log_cumulants, cumulants = load_data(group, subject, condition)
        
    all_log_cumulants, all_cumulants, subjects_list = load_data_groups_subjects('rest0')