import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import source_mf_results as mfr
import plots

import numpy as np

import matplotlib.pyplot as plt


# Load info
import meg_info
info   = meg_info.get_info()

# Select groups and subjects
groups = ['AV', 'V', 'AVr']
subjects = {}
subjects['AV'] = info['subjects']['AV']
subjects['V'] = info['subjects']['V']
subjects['AVr'] = info['subjects']['AVr']


log_cumulants = [0]
mf_params_idx = 1
source_rec_params_idx = 0

X, y, subjects_idx, groups_idx, subjects_list = \
            mfr.load_logcumul_for_classification(['rest5'], ['posttest'],
                                                 log_cumulants,
                                                 groups, subjects,
                                                 mf_params_idx, source_rec_params_idx)


# # Visualization --------------------------
c1_rest = X[:36, :138]#.mean(axis = 0)
c1_task = X[36:, :138]#.mean(axis = 0)
# c2_rest = -X[:36, 138:].mean(axis = 0)
# c2_task = -X[36:, 138:].mean(axis = 0)

# plots.plot_brain(c1_rest, 
#                  fmin = 0.8, 
#                  fmax = c1_rest.max(), 
#                  png_filename = 'debug/debug_c1_rest',
#                  positive_only = True)

# plots.plot_brain(c1_task, 
#                  fmin = 0.8, 
#                  fmax = c1_rest.max(), 
#                  png_filename = 'debug/debug_c1_task',
#                  positive_only = True)

# plots.plot_brain(c2_rest, 
#                  fmin = 0, 
#                  fmax = c2_rest.max(), 
#                  png_filename = 'debug/debug_c2_rest',
#                  positive_only = True)

# plots.plot_brain(c2_task, 
#                  fmin = 0, 
#                  fmax = c2_rest.max(), 
#                  png_filename = 'debug/debug_c2_task',
#                  positive_only = True)
# # ---------------------------------------

# Classification --------------------------

subs_train    = [2, 3] 

samples_train = subs_train + list(np.array(subs_train) + 36)
samples_test  = list(set(list(range(72))) - set(samples_train))

subjects_train = set(subjects_idx[samples_train])
subjects_test = set(subjects_idx[samples_test])


X_train = X[samples_train, :]
X_test  = X[samples_test, :]

y_train = y[samples_train]
y_test  = y[samples_test]


from sklearn.svm import SVC

svm = SVC(kernel='linear', C = 0.01)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

err_abs = (y_pred != y_test).sum()

err = err_abs/len(y_pred)

print("err = ", err)