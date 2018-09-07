import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))
# sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))


import source_mf_results as mfr
import plots_ as plots

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_validate, learning_curve
from sklearn.linear_model import LogisticRegression
import h5py

RANDOM_STATE = 123
N_JOBS       = 1

PLOT_LEARNING    = True
PLOT_IMPORTANCES = True
SHOW_PLOTS       = True

SAVE             = True


# def get_simulated_data():
#     from sklearn.datasets import load_digits
#     digits = load_digits()
#     X, y = digits.data, digits.target
#     subjects_idx = np.arange(len(y))

#     return X, y, subjects_idx, None, None



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


log_cumulants = [1]

POSITIVE = True

mf_params_idx = 1
source_rec_params_idx = 0


# String to save file 
conditions_folder = '-'.join(conditions_0) + '_' + '-'.join(conditions_1)
features_info = None
if log_cumulants == [0,1]:
    features_info = 'c1_c2'
elif log_cumulants == [0]:
    features_info = 'c1'
elif log_cumulants == [1]:
    features_info = 'c2'

elif log_cumulants == [0,-1]:
    features_info = 'c1_avgC2j'
elif log_cumulants == [-1]:
    features_info = 'avgC2j'

elif log_cumulants == [0,-2]:
    features_info = 'c1_maxC2j'
elif log_cumulants == [-2]:
    features_info = 'maxC2j'

elif log_cumulants == [0,-3]:
    features_info = 'c1_maxminC2j'
elif log_cumulants == [-3]:
    features_info = 'maxminC2j'

elif log_cumulants == [0,100]:
    features_info = 'c1_EOG'
elif log_cumulants == [-3, 100]:
    features_info = 'maxminC2j_EOG'
elif log_cumulants == [100]:
    features_info = 'EOG'

elif log_cumulants == [-3, 200]:
    features_info = 'maxminC2j_EOGmaxminC2j'
elif log_cumulants == [200]:
    features_info = 'EOGmaxminC2j'

elif log_cumulants == [1,101]:
    features_info = 'c2_EOGc2'
elif log_cumulants == [101]:
    features_info = 'EOGc2'

#===============================================================================
# Load classification data
#===============================================================================
X, y, subjects_idx, groups_idx, subjects_list = \
            mfr.load_logcumul_for_classification(conditions_0, conditions_1,
                                                 log_cumulants,
                                                 groups, subjects,
                                                 mf_params_idx, source_rec_params_idx)

n_subjects = len(subjects_list)


def my_mean(a,axis=0,outcoef=None):
    x= a.copy()
    #print (x)
    #remove NaN values
    mask= ~np.isnan(x)
    x[~mask]=0
    y= np.sum(x,axis)/np.sum(mask,axis)
    if outcoef:
        x[~mask]=np.nan
        mask2= np.abs(y-x) > outcoef*my_std(x,axis)
        x[mask2]=np.nan
        y = my_mean(x,axis)
    return y

def my_std(a,axis=0):
    x= a.copy()
    #remove NaN values
    mask= ~np.isnan(x)
    x[~mask]=0
    y= np.sqrt(np.sum((x-np.sum(x,axis)/np.sum(mask,axis))**2,axis)/np.sum(mask,axis))
    return y

X_cond0 = my_mean(X[y==0, :], outcoef = 2.)
X_cond1 = my_mean(X[y==1, :], outcoef = 2.)


#-------------------------------------------------------------------------------
# Plot and save
#-------------------------------------------------------------------------------
filename_1 = features_info + '_' + '-'.join(conditions_0) + '.png'
filename_2 = features_info + '_' + '-'.join(conditions_1) + '.png'


factor = 1
minval = 0.8
maxval = 1.2
if 'c2' in features_info:
    factor = -1
    minval = 0
    maxval = 0.03

plots.plot_brain(factor*X_cond0, fmin = minval, fmax = maxval, 
                    png_filename = filename_1, positive_only = POSITIVE)
plots.plot_brain(factor*X_cond1, fmin = minval, fmax = maxval, 
                    png_filename = filename_2, positive_only = POSITIVE)