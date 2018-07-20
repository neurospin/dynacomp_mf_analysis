import sys, os
from os.path import dirname, abspath
# Add parent dir to path
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import source_mf_results as mfr
import plots

import numpy as np

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



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



## One way to load the data 
X, y, subjects_idx, groups_idx, subjects_list = \
            mfr.load_logcumul_for_classification(['rest5'], ['posttest'],
                                                 log_cumulants,
                                                 groups, subjects,
                                                 mf_params_idx, source_rec_params_idx)


## Another way -> Plots verified with Daria
all_log_cumulants_rest, _ ,subjects_list = \
                mfr.load_data_groups_subjects('rest5', groups, subjects,
                                              mf_param_idx = mf_params_idx, 
                                              source_rec_param_idx = source_rec_params_idx)

all_log_cumulants_task, _ ,subjects_list = \
                mfr.load_data_groups_subjects('posttest', groups, subjects,
                                              mf_param_idx = mf_params_idx, 
                                              source_rec_param_idx = source_rec_params_idx)

c1_rest = all_log_cumulants_rest[:, :, 0]
c1_task = all_log_cumulants_task[:, :, 0]
# c2_rest = all_log_cumulants_rest[:, :, 1].clip(max=0)
# c2_task = all_log_cumulants_task[:, :, 1].clip(max=0)

print( "Checking c1_rest, error = ", (np.abs(c1_rest-X[:36, :138]) > 0).sum() )
print( "Checking c1_task, error = ", (np.abs(c1_task-X[36:, :138]) > 0).sum() )

# print( "Checking c2_rest, error = ", (np.abs(c2_rest-X[:36, 138:]) > 0).sum() )
# print( "Checking c2_task, error = ", (np.abs(c2_task-X[36:, 138:]) > 0).sum() )

# # Visualization --------------------------
# c1_rest = X[:36, :138].mean(axis = 0)
# c1_task = X[36:, :138].mean(axis = 0)
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

np.random.seed(456)
#subs_train    = [0,35]#,3,4,5,6,7,8,9,10,11]  # subjects used in training set
N_TRAIN = 5 # number of subjects in training set
N_TEST  = 36 - N_TRAIN
N_RUNS  = 1 # number of simulations

err_list = []
for run in range(N_RUNS):

	subs_train = np.random.permutation(36)[:N_TRAIN].tolist()

	samples_train = subs_train + list(np.array(subs_train) + 36)
	samples_test  = list(set(list(range(72))) - set(samples_train))

	subjects_train = set(subjects_idx[samples_train])
	subjects_test = set(subjects_idx[samples_test])

	print('train: ', subjects_train)
	print('test: ',  subjects_test)

	X_train = X[samples_train, :]
	X_test  = X[samples_test, :]

	y_train = y[samples_train]
	y_test  = y[samples_test]


	clf = SVC(kernel='linear', C = 0.05, tol = 1e-8)
	# clf = RandomForestClassifier(n_estimators = 50)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	err_abs = (y_pred != y_test).sum()

	err = err_abs/len(y_pred)

	err_list.append(err)
	# print("err = ", err)


err_list = np.array(err_list)
acc_test = 1 - err_list

print("-----------------")
print("acc = %f +- %f"%(acc_test.mean(), acc_test.std()))


# Compare distributions
c1_train_rest = X_train[:N_TRAIN, :138]
c1_train_task = X_train[N_TRAIN:, :138]


c1_test_rest = X_test[:N_TEST, :138]
c1_test_task = X_test[N_TEST:, :138]

diff_train  =  c1_train_rest - c1_train_task
diff_test  =  c1_test_rest - c1_test_task


def compare_distributions(data_train, data_test, title = ''):
    """
    For each cortical region, plot mean and std for train and test data.

    data_train: shape (n_subjects_train, n_features)
    data_test:  shape (n_subjects_test, n_features)
    """
    plt.figure()
    plt.title(title)
    plt.plot(np.arange(138), data_train.mean(axis=0), 'bo-', label = 'train')
    plt.plot(np.arange(138), data_test.mean(axis=0), 'ro-', label = 'test')
    plt.legend()


    plt.fill_between(np.arange(138),data_train.mean(axis=0) - data_train.std(axis=0),
                                data_train.mean(axis=0) + data_train.std(axis=0), alpha=0.25,
                             color="b")
    plt.fill_between(np.arange(138),data_test.mean(axis=0) - data_test.std(axis=0),
                             data_test.mean(axis=0) + data_test.std(axis=0), alpha=0.25,
                             color="r")


compare_distributions(c1_train_rest, c1_test_rest, 'c1 rest')
compare_distributions(c1_train_task, c1_test_task, 'c1 task')
compare_distributions(diff_train, diff_test, 'c1 rest-task')


plt.show()

# w = clf.feature_importances_
# plots.plot_brain(w, 
#                  fmin = 0, 
#                  fmax = w.max(), 
#                  png_filename = 'debug/debug_feat_importances_random_forest_more_subjects',
#                  positive_only = True)

# w = clf.coef_.squeeze()
# plots.plot_brain(w, 
#                  fmin = -np.abs(w).max(), 
#                  fmax = np.abs(w).max(), 
#                  png_filename = 'debug/debug_feat_importances_svm',
#                  positive_only = False)
