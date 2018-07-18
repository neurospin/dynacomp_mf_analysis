"""
Perform binary classification on two conditions, e.g. rest vs. task
"""

import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import source_mf_results as mfr
import plots

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression



#===============================================================================
# Global parameters 
#===============================================================================
# Load info
import meg_info
info   = meg_info.get_info()

# Select groups and subjects
groups = ['AV', 'V']
subjects = {}
subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']


# Select MF parameters and source reconstruction parameters
mf_params_idx = 1 
source_rec_params_idx = 0

# Conditions for classification
conditions_0 = ['rest5']
conditions_1 = ['posttest']

# Cumulants used for classification
log_cumulants = [0, 1]


#===============================================================================
# Load classification data
#===============================================================================
X, y, subjects_idx, groups_idx = mfr.load_logcumul_for_classification(conditions_0, conditions_1,
                                                                      log_cumulants,
                                                                      groups, subjects,
                                                                      mf_params_idx, source_rec_params_idx)



