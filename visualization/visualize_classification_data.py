import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20


import source_mf_results as mfr_source
import sensor_mf_results as mfr_sensor

import meg_info as meg_info_source
import meg_info_sensor_space as meg_info_sensor

info_source   = meg_info_source.get_info()
info_sensor   = meg_info_sensor.get_info()


#----------------------------------------------------------------------------------------------
# Options
#----------------------------------------------------------------------------------------------

space = 'sensor'
channel_type = 'mag'


mf_params_idx = 0
source_rec_params_idx = 0


clip_c2 = True # set c_2 to zero if c2 > 0

mfidx2formalism = {0:'WCMF', 1:'WLMF', 2:'p = 1', 3:'p = 2'}

conditions_0 = ['rest5']
conditions_1 = ['posttest']

log_cumulants = [-1]

#----------------------------------------------------------------------------------------------
# Load data
#----------------------------------------------------------------------------------------------
if space == 'source':
    # Select groups and subjects
    groups = ['AV', 'V', 'AVr']
    subjects = {}
    subjects['AV']  = info_source['subjects']['AV']
    subjects['V']   = info_source['subjects']['V']
    subjects['AVr'] = info_source['subjects']['AVr']


    X, y, subjects_idx, groups_idx, subjects_list = \
                                mfr_source.load_logcumul_for_classification(conditions_0, conditions_1,
                                                                     log_cumulants,
                                                                     groups, subjects,
                                                                     mf_params_idx, source_rec_params_idx,
                                                                     clip_c2 = clip_c2)

    xlabel = 'cortical region'
    ylabel = '$c_{%d}$'%(log_cumulants[0]+1)


if space == 'sensor':
    # Select groups and subjects
    groups = ['AV', 'V', 'AVr']
    subjects = {}
    subjects['AV']  = info_sensor['subjects']['AV']
    subjects['V']   = info_sensor['subjects']['V']
    subjects['AVr'] = info_sensor['subjects']['AVr']


    X, y, subjects_idx, groups_idx, subjects_list, extra_info = \
                                mfr_sensor.load_logcumul_for_classification(conditions_0, conditions_1,
                                                                     log_cumulants,
                                                                     groups, subjects,
                                                                     mf_params_idx,
                                                                     channel_type,
                                                                     clip_c2 = clip_c2)
    xlabel = 'sensor'
    ylabel = '$c_{%d}$'%(log_cumulants[0]+1)


#----------------------------------------------------------------------------------------------
# Visualize
#----------------------------------------------------------------------------------------------
def plot_comparison(data_0, data_1, 
                    title=str(mfidx2formalism[mf_params_idx]),
                    ylim = None, 
                    fignum = None, labels = ['-'.join(conditions_0), '-'.join(conditions_1)],
                    clip = (clip_c2 and log_cumulants[0]==1)):

    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)


    indices = np.arange(data_0.shape[0])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    data_0_mean = np.mean(data_0, axis=1)
    data_0_std = np.std(data_0, axis=1)
    data_1_mean = np.mean(data_1, axis=1)
    data_1_std = np.std(data_1, axis=1)
    plt.grid(True)


    if not clip:
        plt.fill_between(indices, data_0_mean - data_0_std,
                         data_0_mean + data_0_std, alpha=0.1,
                         color="r")
        plt.fill_between(indices, data_1_mean - data_1_std,
                         data_1_mean + data_1_std, alpha=0.1, color="g")

    else:
        plt.fill_between(indices, (data_0_mean - data_0_std).clip(max = 0),
                         (data_0_mean + data_0_std).clip(max = 0), alpha=0.1,
                         color="r")
        plt.fill_between(indices, (data_1_mean - data_1_std).clip(max = 0),
                         (data_1_mean + data_1_std).clip(max = 0), alpha=0.1, color="g")        

    plt.plot(indices, data_0_mean, 'o-', color="r",
             label=labels[0])
    plt.plot(indices, data_1_mean, 'o-', color="g",
             label=labels[1])

    plt.legend(loc="best")



X0 = X[y==0, :].T  # (cortical regions, subjects)
X1 = X[y==1, :].T  #

plot_comparison(X0, X1)





#----------------------------------------------------------------------------------------------
# Test classification
#----------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_validate, cross_val_score
from sklearn.svm import SVC

cv = GroupShuffleSplit(n_splits= 20, 
                       test_size = 0.8, 
                       random_state = 42)

# clf = RandomForestClassifier(n_estimators=300, random_state=42)
svm = SVC(kernel='linear')
# parameters for grid search
p_grid = {}
p_grid['C'] = np.power(10.0, np.linspace(-4, 4, 10))
# classifier
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=cv)

fit_params = {'groups':subjects_idx}


scoring = 'accuracy'
scores = cross_val_score(clf, X, y, cv=cv, scoring =scoring, groups=subjects_idx, fit_params = fit_params)


print('Cross-validation ' + scoring + ' score = %1.3f (+/- %1.5f)' % (np.mean(scores), np.std(scores)))






plt.show()