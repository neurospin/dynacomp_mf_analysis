"""
Define features parameters and provides functions to create files containing
features for classification
"""


import numpy as np
import source_mf_results as mfr
import meg_info
info   = meg_info.get_info()

#-----------------------------------------------------------------------
# MF features
#-----------------------------------------------------------------------

# Possible conditions for classification
cond0_list = [ ['rest5'],    ['rest0', 'rest5'],       ['rest0'],   ['rest0']  ]
cond1_list = [ ['posttest'], ['pretest', 'posttest'],  ['posttest'],['pretest']]
conditions_classif = list(zip(cond0_list, cond1_list))


def get_mf_features(save = False,
                    conditions_classif = conditions_classif,
                    mf_params_idx_list = [1, 2, 3],
                    source_rec_params_idx_list = [0, 1],
                    log_cumulants_list = [[0,1], [0], [1]]):

    # Select groups and subjects
    groups = ['AV', 'V', 'AVr']
    subjects = {}
    subjects['AV'] = info['subjects']['AV']
    subjects['V'] = info['subjects']['V']
    subjects['AVr'] = info['subjects']['AVr']


    # Different combinations of features
    feature_name  = []
    feature_index = []
    feature_params = []
    feature_conditions = []
    feature_conditions_string = []

    idx = 0
    for conditions_0, conditions_1 in conditions_classif:
        for mf_params_idx in mf_params_idx_list:
            for source_rec_params_idx in source_rec_params_idx_list:
                for log_cumulants in log_cumulants_list:
                    if log_cumulants == [0,1]:
                        features_info = 'c1_c2'
                    elif log_cumulants == [0]:
                        features_info = 'c1'
                    elif log_cumulants == [1]:
                        features_info = 'c2'


                    name = '%s_mf_%d_rec_%d'%(features_info, mf_params_idx, source_rec_params_idx)
                    conditions_string = '-'.join(conditions_0) + '_' + '-'.join(conditions_1)



                    feature_name.append(name)
                    feature_index.append(idx)
                    feature_params.append((log_cumulants, mf_params_idx, source_rec_params_idx))
                    feature_conditions.append((conditions_0, conditions_1))
                    feature_conditions_string.append(conditions_string)

                    idx += 1

                    if save:
                        # Load features
                        X, y, subjects_idx, groups_idx, subjects_list = \
                            mfr.load_logcumul_for_classification(conditions_0, conditions_1,
                                                                 log_cumulants,
                                                                 groups, subjects,
                                                                 mf_params_idx, source_rec_params_idx)

                        


    output  = {}
    output['feature_name']       = feature_name
    output['feature_index']      = feature_index
    output['feature_params']     = feature_params
    output['feature_conditions'] = feature_conditions
    output['feature_conditions_string'] = feature_conditions_string

    return output


if __name__ == '__main__':
    mf_features = get_mf_features(save = True)

