"""
Plots a summary of the classification results
"""
import numpy as np 
import h5py
import os
import classification_utils as clf_utils


SELECT = 'MF_FEATS'
#-----------------------------------------------------------------------
# Summary using MF features
#-----------------------------------------------------------------------
if SELECT == 'MF_FEATS':

    fignum = 'classification  with mf features'

    classifiers = ['linear_svm']
    features_names = ['c1_c2']

    mf_params_list = [1]
    source_rec_params_list = [0]

    # Possible conditions for classification
    cond0_list = [ ['rest5'],    ['rest0', 'rest5'],         ]
    cond1_list = [ ['posttest'], ['pretest', 'posttest']]

    conditions_classif = list(zip(cond0_list, cond1_list))

    # Plot learning curves in different cases:
    for conditions_0, conditions_1 in conditions_classif:
        conditions_folder = '-'.join(conditions_0) + '_' + '-'.join(conditions_1)

        for mf_params in mf_params_list:
            for rec_params in source_rec_params_list:
                for classif in classifiers:
                    for feat_name in features_names:

                        # File containing classification results
                        filename = '%s_%s_mf_%d_rec_%d.h5'%(classif, feat_name, mf_params, rec_params)
                        outdir   = os.path.join('learning_curves_raw_data', conditions_folder)
                        filename = os.path.join(outdir, filename)

                        # Load learning curves
                        with h5py.File(filename, "r") as f:
                            # train_sizes is divided by len(conditions_0) so that
                            # train_sizes = number of subjects in training set
                            train_sizes  = f['train_sizes'][:] / len(conditions_0)  
                            train_scores = f['train_scores'][:]
                            test_scores  = f['test_scores'][:]

                            
                            
                        print(filename)

                    # 
    # 
    #                         f.create_dataset('train_sizes',  data = train_sizes_abs )
    #                         f.create_dataset('train_scores', data = train_scores )
    #                         f.create_dataset('test_scores',  data = test_scores )

