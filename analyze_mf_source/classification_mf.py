"""
Perform binary classification on two conditions, e.g. rest vs. task
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import source_mf_results as mfr
import plots

import classification_utils as clf_utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_validate, learning_curve
from sklearn.linear_model import LogisticRegression
import h5py

RANDOM_STATE = 123
N_JOBS       = 6

PLOT_LEARNING    = True
PLOT_IMPORTANCES = True
SHOW_PLOTS       = False

SAVE             = False

#===============================================================================
# Global parameters 
#===============================================================================
# Load info
import meg_info
info   = meg_info.get_info()

# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = {}
subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']

# 
# mf_params_idx = 1
# source_rec_params_idx = 0

# Conditions for classification
conditions_0 = ['rest5']
conditions_1 = ['posttest']

# 
# log_cumulants = [0, 1]


#===============================================================================
# Classification parameters 
#===============================================================================
# Choose classifier
classifier_name_list = ['linear_svm']

# Define cross validation scheme
n_splits   = 30
test_size  = 0.2          # used to obtain feature importances
scoring    = ['accuracy']


# Select classifier
for classifier_name in classifier_name_list:
    # Select MF parameters and source reconstruction parameters
    for mf_params_idx in [1]:    # 0, 1, 2 or 3 
        for source_rec_params_idx in [0]:  # 0 or 1

            # Cumulants used for classification
            for log_cumulants in [[0]]:  # [0, 1], [0] or [1]
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("Running: ", mf_params_idx, source_rec_params_idx, log_cumulants)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


                # String to save file 
                conditions_folder = '-'.join(conditions_0) + '_' + '-'.join(conditions_1)
                features_info = None
                if log_cumulants == [0,1]:
                    features_info = 'c1_c2'
                elif log_cumulants == [0]:
                    features_info = 'c1'
                elif log_cumulants == [1]:
                    features_info = 'c2'

                #===============================================================================
                # Load classification data
                #===============================================================================
                X, y, subjects_idx, groups_idx, subjects_list = \
                            mfr.load_logcumul_for_classification(conditions_0, conditions_1,
                                                                 log_cumulants,
                                                                 groups, subjects,
                                                                 mf_params_idx, source_rec_params_idx)

                n_subjects = len(subjects_list)



                #===============================================================================
                # Run classification
                #===============================================================================

                # List with sizes of training set
                train_sizes = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

                # Get learning curve and feature importances
                train_sizes_abs, train_scores, test_scores, w, positive_only = \
                    clf_utils.run_classification(classifier_name, 
                                                X, y, subjects_idx,
                                                train_sizes,
                                                scoring,
                                                n_splits,
                                                RANDOM_STATE,
                                                N_JOBS, 
                                                ref_train_size = 1-test_size)


                if PLOT_LEARNING:
                    clf_utils.plot_learning_curve(train_sizes_abs, train_scores, test_scores, title = classifier_name)

                    if SAVE:
                        # Save learning curve image 
                        filename = '%s_%s_mf_%d_rec_%d.png'%(classifier_name, features_info, mf_params_idx, source_rec_params_idx)
                        outdir   = os.path.join('learning_curves', conditions_folder)
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        filename =  os.path.join(outdir, filename)
                        plt.savefig(filename)
                        del filename

                        # Save learning curve raw data
                        filename = '%s_%s_mf_%d_rec_%d.h5'%(classifier_name, features_info, mf_params_idx, source_rec_params_idx)
                        outdir   = os.path.join('learning_curves_raw_data', conditions_folder)
                        if not os.path.exists(outdir):
                            os.makedirs(outdir)
                        filename =  os.path.join(outdir, filename)   

                        with h5py.File(filename, "w") as f:
                            f.create_dataset('train_sizes',  data = train_sizes_abs )
                            f.create_dataset('train_scores', data = train_scores )
                            f.create_dataset('test_scores',  data = test_scores )


                    # Show curve
                    if SHOW_PLOTS:
                        plt.show()


                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                #===============================================================================
                # Plot feature importances
                #===============================================================================
                if PLOT_IMPORTANCES:
                    n_features = X.shape[1]
                    for ii, lc in enumerate(log_cumulants):
                        if SAVE:
                            png_filename = 'c%d_%s_%s_mf_%d_rec_%d.png'%(lc+1,classifier_name, features_info, mf_params_idx, source_rec_params_idx)
                            outdir = os.path.join('feature_importances', conditions_folder)
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)
                            png_filename = os.path.join(outdir, png_filename)
                        else: 
                            png_filename = None



                        feat_to_plot = w[ii*138:(ii+1)*138]
                        plots.plot_brain(feat_to_plot, 
                                         fmin = -np.abs(feat_to_plot).max(), 
                                         fmax =  np.abs(feat_to_plot).max(), 
                                         png_filename = png_filename,
                                         positive_only = positive_only)


                #del X
                del y
                # plt.close()