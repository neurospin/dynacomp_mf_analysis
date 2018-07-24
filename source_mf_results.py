"""
Provides functions to load MF results on source space.
"""

import os
import numpy as np 
import h5py
import meg_info
info = meg_info.get_info()


def load_data(group, subject, condition, 
              mf_param_idx = 1, 
              source_rec_param_idx = 0,
              time_str = '20180713'):
    """
    group = 'AV', 'V' or 'AVr'
    subject = one of the subjects in the given group
    condition = 'rest0', 'rest5', 'pretest', 'posttest'
    mf_param_idx = index of MF parameters
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
                              time_str = '20180713',
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



def load_logcumul_for_classification(conditions_0, conditions_1,
                                     log_cumulants,
                                     groups, subjects,
                                     mf_param_idx, source_rec_param_idx,
                                     time_str = '20180713',
                                     n_labels = 138,
                                     n_cumul = 3,
                                     max_j = 13,
                                     clip_c2 = True):
    """
    conditions_0 : conditions for classif. label 0 (e.g. ['rest0', 'rest5'])
    conditions_1 : conditions for classif. label 1 (e.g. ['pretest', 'posttest'])
    log_cumulants: log_cumulants to use, [0] for c1 only, [1] for c2 only, [0, 1] for c1 and c2;
                   put -1 instead of 1 to get average of C2(j) instead of c2

    returns:
    X: (n_samples, n_features)
    y: (n_samples,)
    subject_idx: (n_samples,) index of the subject from whom the sample was taken
    group_idx  : (n_samples,) index of the group of the subject from whom the sample was taken
    """
   
    ###
    # Data for classif. label 0
    ###
    X0_tuple = ()
    subjects_0 = []
    subjects_0_idx = []
    for cond in conditions_0:
        # all_log_cumulants: 
        all_log_cumulants, all_cumulants, subjects_list = \
                    load_data_groups_subjects(cond,
                                            groups = groups,
                                            subjects = subjects,
                                            mf_param_idx = mf_param_idx, 
                                            source_rec_param_idx = source_rec_param_idx,
                                            time_str = time_str,
                                            n_labels = n_labels,
                                            n_cumul = n_cumul,
                                            max_j = max_j)

        subjects_0 = subjects_0 + subjects_list
        subjects_0_idx = subjects_0_idx + list(range((len(subjects_list))))

        X_cond = ()
        for c_idx in log_cumulants:
            if c_idx >= 0:
                data = all_log_cumulants[:, :, c_idx]  # shape (n_subjects, n_cortical_labels)

                if clip_c2 and c_idx == 1:
                    data = data.clip(max = 0)

            elif c_idx == -1:
                data = (all_cumulants[:, :, 1, 7:12]).mean(axis=2)
            elif c_idx == -2:
                data = (all_cumulants[:, :, 1, 7:12]).max(axis=2)
            elif c_idx == -3:
                data = (all_cumulants[:, :, 1, 7:12]).max(axis=2) \
                       -(all_cumulants[:, :, 1, 7:12]).min(axis=2)

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
        all_log_cumulants, all_cumulants, subjects_list = \
                    load_data_groups_subjects(cond,
                                            groups = groups,
                                            subjects = subjects,
                                            mf_param_idx = mf_param_idx, 
                                            source_rec_param_idx = source_rec_param_idx,
                                            time_str = time_str,
                                            n_labels = n_labels,
                                            n_cumul = n_cumul,
                                            max_j = max_j)

        subjects_1 = subjects_1 + subjects_list
        subjects_1_idx = subjects_1_idx + list(range((len(subjects_list))))

        X_cond = ()
        for c_idx in log_cumulants:
            if c_idx >= 0:
                data = all_log_cumulants[:, :, c_idx]  # shape (n_subjects, n_cortical_labels)

                if clip_c2 and c_idx == 1:
                    data = data.clip(max = 0)
            elif c_idx == -1:
                data = (all_cumulants[:, :, 1, 7:12]).mean(axis=2)
            elif c_idx == -2:
                data = (all_cumulants[:, :, 1, 7:12]).max(axis=2)
            elif c_idx == -3:
                data = (all_cumulants[:, :, 1, 7:12]).max(axis=2) \
                       -(all_cumulants[:, :, 1, 7:12]).min(axis=2)


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


    return X, y, subjects_idx, groups_idx, subjects_list



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

    log_cumulants, cumulants = load_data(group, subject, condition)
        
    all_log_cumulants, all_cumulants, subjects_list = load_data_groups_subjects('rest0')