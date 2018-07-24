
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



def load_logcumul_for_classification(conditions_0, conditions_1,
                                     log_cumulants,
                                     groups, subjects,
                                     mf_param_idx,
                                     channel_type = 'mag',
                                     n_cumul = 3,
                                     use_max_c2_j = True):
    """
    conditions_0 : conditions for classif. label 0 (e.g. ['rest0', 'rest5'])
    conditions_1 : conditions for classif. label 1 (e.g. ['pretest', 'posttest'])
    log_cumulants: log_cumulants to use, [0] for c1 only, [1] for c2 only, [0, 1] for c1 and c2

    returns:
    X: (n_samples, n_features)
    y: (n_samples,)
    subject_idx: (n_samples,) index of the subject from whom the sample was taken
    group_idx  : (n_samples,) index of the group of the subject from whom the sample was taken

    IMPORTANT:

    if use_max_c2_j = True, instead of c2, returns max{C2(j)} for 9<=j<=13 
    """
   
    ###
    # Data for classif. label 0
    ###
    X0_tuple = ()
    subjects_0 = []
    subjects_0_idx = []
    for cond in conditions_0:
        # all_log_cumulants: 
        all_log_cumulants, all_cumulants, subjects_list, params, channels_picks, channels_names, ch_name2index = \
                    load_data_groups_subjects(cond,
                                              groups = groups,
                                              subjects = subjects,
                                              mf_param_idx = mf_param_idx,
                                              channel_type = channel_type,
                                              n_cumul = n_cumul)

        subjects_0 = subjects_0 + subjects_list
        subjects_0_idx = subjects_0_idx + list(range((len(subjects_list))))

        X_cond = ()
        for c_idx in log_cumulants:
            data = all_log_cumulants[:, :, c_idx]  # shape (n_subjects, n_cortical_labels)

            if c_idx == 1 and use_max_c2_j:
                data = (all_cumulants[:, :, c_idx, 8:13]).max(axis=2)

            X_cond = X_cond + (data,)

        X_cond = np.hstack(X_cond)   # stacking features -> (n_subjects, n_cortical_labels*len(log_cumulants))

        X0_tuple = X0_tuple + (X_cond,)

    X0 = np.vstack(X0_tuple) # stacking samples (n_subjects*len(conditions_0), n_cortical_labels*len(log_cumulants))
    y0 = np.zeros(X0.shape[0])

    ###
    # Data for classif. label 1
    ###
    X1_tuple = ()
    subjects_1 = [] 
    subjects_1_idx = []
    for cond in conditions_1:
        # all_log_cumulants: 
        all_log_cumulants, all_cumulants, subjects_list, params, channels_picks, channels_names, ch_name2index = \
                    load_data_groups_subjects(cond,
                                              groups = groups,
                                              subjects = subjects,
                                              mf_param_idx = mf_param_idx,
                                              channel_type = channel_type,
                                              n_cumul = n_cumul)


        subjects_1 = subjects_1 + subjects_list
        subjects_1_idx = subjects_1_idx + list(range((len(subjects_list))))

        X_cond = ()
        for c_idx in log_cumulants:
            data = all_log_cumulants[:, :, c_idx]  # shape (n_subjects, n_cortical_labels)

            if c_idx == 1 and use_max_c2_j:
                data = (all_cumulants[:, :, c_idx, 8:13]).max(axis=2)

            X_cond = X_cond + (data,)    

        X_cond = np.hstack(X_cond)   # stacking features -> (n_subjects, n_cortical_labels*len(log_cumulants))

        X1_tuple = X1_tuple + (X_cond,)
    X1 = np.vstack(X1_tuple) # stacking samples (n_subjects*len(conditions_0), n_cortical_labels*len(log_cumulants))
    y1 = np.ones(X1.shape[0])


    ##
    # Stack to get X, y
    ##
    X = np.vstack((X0, X1)) # (n_samples, n_features)
    y = np.hstack((y0, y1)) # (n_samples,)
    subjects_idx = np.array(subjects_0_idx + subjects_1_idx)
    subjects_classif    = subjects_0 + subjects_1
    groups_idx   = np.array( [get_subject_group_idx(ss) for ss in subjects_classif]  )


    extra_info = (params, channels_picks, channels_names, ch_name2index)

    return X, y, subjects_idx, groups_idx, subjects_list, extra_info



def get_subject_group_idx(subject):
    """
    subject = (group_id, subject_id)
    returns groud_id_index

    'AV' -> 0
    'V'  -> 1
    'AVr'-> 2
    """
    group_id, subject_id = subject

    group_index = -1

    if group_id == 'AV':
        group_index = 0

    elif group_id == 'V':
        group_index = 1

    elif group_id == 'AVr':
        group_index = 2

    return group_index


if __name__ == '__main__':
    group = info['groups'][0]
    subject = info['subjects'][group][0]
    condition = 'rest0'

    log_cumulants, cumulants, params, channels_picks, channels_names, ch_name2index = load_data(group, subject, condition)
        
    all_log_cumulants, all_cumulants, subjects_list, params, channels_picks, channels_names, ch_name2index =\
         load_data_groups_subjects('rest0')